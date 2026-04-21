"""Transformer model for Rolling Stock Stars AlphaZero training.

Token-based architecture: each game entity is a separate input token. Type-specific
linear projections -> L pre-LN transformer blocks -> entity-readout policy heads + value head.

Key differences from the MLP model (nn/template.py):
  - Input: (batch, num_tokens, token_dim) token features, not flat state vector
  - No state rotation: active player marked with is_active flag
  - Entity-readout policy: each entity token produces its own action logits
  - ACQ factored into three single-entity sub-phases (corp/company/price)
  - Per-phase action indices (max 53 for INVEST), not a global action vector
  - Value read from player tokens directly (no un-rotation needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.actions import MAX_LEGAL_ACTIONS_PY
from core.data import (
    AUCTION_CAP,
    PHASE_ACTION_SIZES,
    DecisionPhase,
)
from core.token_data import TokenDataSize, TokenWidth, get_token_widths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Decision phases / action sizes all live in ``core.data`` and are imported
# above. This module is strictly a consumer; editing policy head widths or
# adding token types happens over there.

NUM_PHASES = len(DecisionPhase)
K_MAX = int(MAX_LEGAL_ACTIONS_PY)

_GELU_APPROX = "tanh"


def _slice_proj(x: torch.Tensor, proj: nn.Linear, idx: int | slice) -> torch.Tensor:
    """Slice ``x`` to ``proj.in_features`` at ``idx`` and project.

    Output shape depends on ``idx``: int → ``(B, d)``, slice → ``(B, n, d)``.
    Int call sites add a trailing ``.unsqueeze(1)`` themselves — keeping the
    per-site fixup explicit avoids a Python-side isinstance branch that
    Dynamo would have to trace through.

    Module-level rather than a closure or staticmethod so Dynamo can inline
    it without guarding on a fresh function id per call or routing through
    class-attribute lookup.
    """
    return proj(x[:, idx, :proj.in_features])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """All dimensions parameterized. Defaults are 3-player with d_model=128."""

    # Core architecture
    num_players: int = 3  # 3-5 supported
    d_model: int = 128
    num_heads: int = 2
    num_layers: int = 10
    ff_mult: float = 3.0  # FFN inner dimension = ceil(ff_mult * d_model)

    # Raw feature width per token (zero-padded to same size across types).
    # Sourced from core.token_data so the model and the Cython extractor
    # can't drift out of sync.
    token_dim: int = int(TokenDataSize.TOKEN_DIM)

    def __post_init__(self) -> None:
        assert 3 <= self.num_players <= 5, f"num_players must be 3-5, got {self.num_players}"

    @property
    def num_tokens(self) -> int:
        """Input-buffer token count: 65 fixed entity/phase tokens + N players.

        The trunk sequence is 1 wider because ``_project_tokens`` concatenates
        a single learned pass anchor after projection; that row has no
        input features so it doesn't exist in the engine-side buffer.
        """
        return self.num_players + 65


def _validate_layout(num_players: int) -> None:
    """Assert the hardcoded token indices in ``RSSTransformerNet.__init__``
    line up with ``core.token_data.get_token_widths`` for the given player
    count.

    The two layouts are sources of truth for the same buffer: the Cython
    side writes each token's features, the Python side slices them into
    per-type projections. Drift between them is invisible at runtime (the
    trunk just sees permuted / mis-sized rows) so we check once at
    construction and crash loudly on mismatch.
    """
    expected = (
        [int(TokenWidth.TW_MARKET_SLOT_PRICES)]
        + [int(TokenWidth.TW_COMPANY)] * 36
        + [int(TokenWidth.TW_MARKET_AVAILABILITY)]
        + [int(TokenWidth.TW_COMPANY_LOCATION)] * 4
        + [int(TokenWidth.TW_COMPANY_ADJ_INCOME)]
        + [int(TokenWidth.TW_FI)]
        + [int(TokenWidth.TW_ACTIVE_PLAYER)]
        + [int(TokenWidth.TW_ACTIVE_CORP)]
        + [int(TokenWidth.TW_ACTIVE_COMPANY)]
        + [int(TokenWidth.TW_PHASE)]
        + [int(TokenWidth.TW_NUM_PLAYERS)]
        + [int(TokenWidth.TW_GAME_PROGRESS)]
        + [int(TokenWidth.TW_INVEST)]
        + [int(TokenWidth.TW_AUCTION)]
        + [int(TokenWidth.TW_DIVIDEND)]
        + [int(TokenWidth.TW_ISSUE)]
        + [int(TokenWidth.TW_PAR)]
        + [int(TokenWidth.TW_ACQ_OFFER)]
        + [int(TokenWidth.TW_ACQ_PRICE)]
        + [int(TokenWidth.TW_CORP)] * 8
        + [int(TokenWidth.TW_PLAYER)] * num_players
    )
    actual = get_token_widths(num_players).tolist()
    assert actual == expected, (
        f"token layout drift between nn/transformer.py and core/token_data.pyx "
        f"for {num_players}p: actual widths {actual} vs expected {expected}"
    )


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN -> MHSA -> residual, LN -> SwiGLU FFN -> residual.

    Attention is implemented via ``F.scaled_dot_product_attention`` over a
    manually-packed QKV projection rather than ``nn.MultiheadAttention``.
    ``nn.MultiheadAttention`` dispatches to ``aten._native_multi_head_attention``,
    which doesn't support fake-tensor tracing and therefore causes a
    graph break per layer under ``torch.compile``. SDPA traces cleanly
    and Inductor fuses it into a single Triton kernel per block.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model {d_model} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_norm = nn.RMSNorm(d_model)
        # Packed Q/K/V projection. Same parameter count as
        # nn.MultiheadAttention's in_proj_weight/in_proj_bias.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        B, N, D = h.shape
        qkv = self.qkv_proj(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        # (3, B, heads, N, head_dim) so unbind(0) yields three (B, heads, N, head_dim) tensors.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        # (B, heads, N, head_dim) -> (B, N, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        h = self.out_proj(attn_out)
        x = x + h
        h = self.ffn_norm(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x + h


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RSSTransformerNet(nn.Module):
    """Transformer with entity-readout policy heads and per-player value output."""

    # Declared for pyright: ``register_buffer`` adds dynamic attributes whose
    # type isn't otherwise visible to the checker.
    _k_range: torch.Tensor
    # Cached per-phase head dispatch tuple, indexed by phase_id 0..10.
    _dispatch: tuple[Callable[[torch.Tensor], torch.Tensor], ...]

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        np_ = cfg.num_players

        # --- Token index bookkeeping ---
        # Buffer layout (matches core/token_data.pyx::_fill_buffer):
        #   static: market_slot_prices, companies×36
        #   dynamic: market_availability, company_location×4 (REMOVED / AUCTION
        #     / REVEALED / CORP_ACQ), company_adj_income, FI, active_player,
        #     active_corp, active_company, phase, num_players, game_progress
        #   phase-specific: invest, auction, dividend, issue, par, acq_offer,
        #     acq_price_info
        #   corps×8, then players×N (trailing so padding for higher player
        #   counts is a no-op on the prefix).
        # The pass anchor is concatenated after projection; see
        # ``_project_tokens``. It lives beyond the player slice, so player
        # indices stay contiguous for the value head and the padding contract.
        self._market_slot_prices_idx = 0
        self._company_slice = slice(1, 37)
        self._market_idx = 37                          # market_availability
        self._company_location_slice = slice(38, 42)   # REMOVED/AUCTION/REVEALED/CORP_ACQ
        self._company_adj_income_idx = 42
        self._fi_idx = 43
        self._active_player_idx = 44
        self._active_corp_idx = 45
        self._active_company_idx = 46
        self._phase_idx = 47
        self._num_players_idx = 48
        self._game_progress_idx = 49
        self._invest_idx = 50
        self._auction_idx = 51
        self._dividend_idx = 52
        self._issue_idx = 53
        self._par_idx = 54
        self._acq_offer_idx = 55
        self._acq_price_info_idx = 56
        self._corp_slice = slice(57, 65)
        self._player_slice = slice(65, 65 + np_)
        # Single learned pass anchor, appended after the player slice. Shared
        # across every pass-using phase — the trunk picks up phase-specific
        # context through attention so one anchor can back all 7 passes.
        self._pass_idx = 65 + np_

        # Drift guard: hardcoded positions above must match the Cython-side
        # ``get_token_widths`` layout. Checking here fires loudly at model
        # construction rather than silently feeding mis-aligned features to
        # the trunk.
        _validate_layout(np_)

        # --- Type-specific input projections ---
        # Each projection takes only its ``TokenWidth.TW_<type>`` prefix of the
        # zero-padded token row. The engine-side buffer is rectangular at
        # ``TOKEN_DIM=92`` so ``get_token_data`` can fill it with a single
        # nogil memcpy pattern, but projection weights on the padding are
        # inert waste — slicing to the actual width here (both in sizing and
        # in ``_project_tokens``) drops those parameters. Widths are pulled
        # from ``TokenWidth`` so the model and the Cython extractor can't
        # drift out of sync. ``company_location_proj`` is shared across all
        # four location tokens (REMOVED / AUCTION / REVEALED / CORP_ACQ) —
        # they carry the same 36-bit "which companies are at this location"
        # bitmap semantics, so distinct weights would just be four copies of
        # the same mapping.
        self.player_proj = nn.Linear(int(TokenWidth.TW_PLAYER), d)
        self.corp_proj = nn.Linear(int(TokenWidth.TW_CORP), d)
        self.company_proj = nn.Linear(int(TokenWidth.TW_COMPANY), d)
        self.fi_proj = nn.Linear(int(TokenWidth.TW_FI), d)
        self.market_proj = nn.Linear(int(TokenWidth.TW_MARKET_AVAILABILITY), d)
        self.market_slot_prices_proj = nn.Linear(int(TokenWidth.TW_MARKET_SLOT_PRICES), d)
        self.company_location_proj = nn.Linear(int(TokenWidth.TW_COMPANY_LOCATION), d)
        self.company_adj_income_proj = nn.Linear(int(TokenWidth.TW_COMPANY_ADJ_INCOME), d)
        self.active_player_proj = nn.Linear(int(TokenWidth.TW_ACTIVE_PLAYER), d)
        self.active_corp_proj = nn.Linear(int(TokenWidth.TW_ACTIVE_CORP), d)
        self.active_company_proj = nn.Linear(int(TokenWidth.TW_ACTIVE_COMPANY), d)
        self.phase_proj = nn.Linear(int(TokenWidth.TW_PHASE), d)
        self.num_players_proj = nn.Linear(int(TokenWidth.TW_NUM_PLAYERS), d)
        self.game_progress_proj = nn.Linear(int(TokenWidth.TW_GAME_PROGRESS), d)
        self.invest_proj = nn.Linear(int(TokenWidth.TW_INVEST), d)
        self.auction_proj = nn.Linear(int(TokenWidth.TW_AUCTION), d)
        self.dividend_proj = nn.Linear(int(TokenWidth.TW_DIVIDEND), d)
        self.issue_proj = nn.Linear(int(TokenWidth.TW_ISSUE), d)
        self.par_proj = nn.Linear(int(TokenWidth.TW_PAR), d)
        self.acq_offer_proj = nn.Linear(int(TokenWidth.TW_ACQ_OFFER), d)
        self.acq_price_proj = nn.Linear(int(TokenWidth.TW_ACQ_PRICE), d)
        # Pass token: no input features. Its representation is a single learned
        # vector (BERT [CLS]-style) that rides the residual stream and picks up
        # game-state context through attention. All other token types are
        # discriminated via their own per-type projection (and its bias), so a
        # shared type-embedding table would be redundant there.
        self.pass_embed = nn.Parameter(torch.empty(d))

        # --- Transformer trunk ---
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.num_heads, math.ceil(cfg.ff_mult * cfg.d_model))
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.RMSNorm(d)

        # --- Entity-readout policy heads ---
        # Shared per-entity-type heads (weight-shared across all tokens of same type)
        self.pass_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1)
        )
        # Shared company-selection head used by INVEST (which company to
        # auction), ACQ_SELECT_COMPANY (which company to acquire), and CLOSING
        # (which company to close). One logit per company token; the phase
        # context that discriminates the three decisions arrives through the
        # trunk's phase-specific context tokens + attention.
        self.company_select_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1)
        )
        # Shared corp-selection head used by ACQ_SELECT_CORP (which corp does
        # the acquiring) and IPO (which corp floats the active company). Same
        # structure as company_select_head; phase context reaches each corp
        # token through attention on the phase-specific context tokens.
        self.corp_select_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1)
        )
        self.corp_trade_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 2),  # buy, sell
        )

        # Phase-specific context token heads. BID bids at face_value + offset
        # for offset ∈ [0, AUCTION_CAP), so the head produces AUCTION_CAP
        # logits — one per legal bid offset (both opening and subsequent).
        self.auction_raise_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, int(AUCTION_CAP)),
        )
        self.dividend_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 26),  # 26 dividend levels
        )
        self.issue_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),  # issue logit (pass from pass_head)
        )
        self.acq_offer_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),  # buy logit (pass from pass_head)
        )

        # ACQ is factored into three sequential single-entity selections:
        # pick the acquiring corp (shared corp_select_head above), pick the
        # target company (shared company_select_head above), pick the price.
        # SELECT_PRICE: 51 price offsets + FI_BUY = 52 logits, read off a
        # dedicated acq_price_info token that the engine populates with
        # (active_corp, active_company) context during PHASE_ACQ_SELECT_PRICE.
        self.price_acq_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 52),
        )

        # PAR reads 14 par-price logits from the par info token. No pass
        # anchor: PAR has no pass action — once a corp is selected the owner
        # must commit to a price.
        self.par_price_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 14),
        )

        # --- Value head (applied per player token) ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
            nn.Tanh(),
        )

        # Index range used by per-phase output masking. Registered as a
        # non-persistent buffer so it rides the module's device and shows
        # up in `.to(...)`, but is not saved to checkpoints.
        self.register_buffer(
            "_k_range", torch.arange(K_MAX, dtype=torch.long), persistent=False,
        )

        # Per-phase head dispatch, keyed by ``DecisionPhase`` member so
        # re-ordering the enum can't silently misroute logits. Materialized
        # to a tuple indexed by integer phase id after asserting that all
        # members are covered and their ids pack into ``[0, NUM_PHASES)`` —
        # the same drift-guard pattern ``_validate_layout`` uses for token
        # widths. Bound-method refs stay valid for the module's lifetime.
        dispatch_by_phase: dict[DecisionPhase, Callable[[torch.Tensor], torch.Tensor]] = {
            DecisionPhase.DPHASE_INVEST: self._policy_invest,
            DecisionPhase.DPHASE_BID: self._policy_bid,
            DecisionPhase.DPHASE_ACQ_SELECT_CORP: self._policy_acq_select_corp,
            DecisionPhase.DPHASE_ACQ_OFFER: self._policy_acq_offer,
            DecisionPhase.DPHASE_CLOSING: self._policy_closing,
            DecisionPhase.DPHASE_DIVIDENDS: self._policy_dividends,
            DecisionPhase.DPHASE_ISSUE: self._policy_issue,
            DecisionPhase.DPHASE_IPO: self._policy_ipo,
            DecisionPhase.DPHASE_PAR: self._policy_par,
            DecisionPhase.DPHASE_ACQ_SELECT_COMPANY: self._policy_acq_select_company,
            DecisionPhase.DPHASE_ACQ_SELECT_PRICE: self._policy_acq_select_price,
        }
        assert len(dispatch_by_phase) == NUM_PHASES, (
            f"dispatch covers {len(dispatch_by_phase)} phases but "
            f"NUM_PHASES={NUM_PHASES} — DecisionPhase gained/lost a member"
        )
        phase_ids = sorted(int(p) for p in dispatch_by_phase)
        assert phase_ids == list(range(NUM_PHASES)), (
            f"DecisionPhase ids must pack into [0, {NUM_PHASES}), got {phase_ids}"
        )
        self._dispatch = tuple(
            dispatch_by_phase[DecisionPhase(i)] for i in range(NUM_PHASES)
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Input projection
    # ------------------------------------------------------------------

    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw token features to d_model via type-specific projections.

        The pass anchor is concatenated here (not present in the input buffer),
        so the output sequence is 1 wider than the input.

        Each row is sliced to its projection's ``in_features`` width before
        projecting. The engine-side buffer is rectangularly zero-padded to
        ``TOKEN_DIM=92`` so ``get_token_data`` can fill it with a uniform
        nogil memcpy pattern, but each type only uses its ``TokenWidth.TW_*``
        prefix — the tail is zero padding that would otherwise be multiplied
        against inert projection weights.

        Args:
            x: (batch, num_players + 65, token_dim) zero-padded raw features.
        Returns:
            (batch, num_players + 66, d_model) embeddings: projected input
            tokens followed by the single pass anchor.
        """
        # Pass anchor: (d,) → (B, 1, d) without mixing SymInt with ``-1`` in
        # the expand target. Under AOT autograd, ``.expand(B, 1, -1)`` where
        # ``B`` is ``x.shape[0]`` sometimes concretizes the batch size into
        # a static guard and forces per-batch-size recompiles; feeding the
        # last dim explicitly lets the symbolic-shape tracker carry ``B``
        # cleanly through.
        d = self.cfg.d_model
        parts: list[torch.Tensor] = [
            _slice_proj(x, self.market_slot_prices_proj, self._market_slot_prices_idx).unsqueeze(1),
            _slice_proj(x, self.company_proj, self._company_slice),                      # (B, 36, d)
            _slice_proj(x, self.market_proj, self._market_idx).unsqueeze(1),
            _slice_proj(x, self.company_location_proj, self._company_location_slice),    # (B, 4, d)
            _slice_proj(x, self.company_adj_income_proj, self._company_adj_income_idx).unsqueeze(1),
            _slice_proj(x, self.fi_proj, self._fi_idx).unsqueeze(1),
            _slice_proj(x, self.active_player_proj, self._active_player_idx).unsqueeze(1),
            _slice_proj(x, self.active_corp_proj, self._active_corp_idx).unsqueeze(1),
            _slice_proj(x, self.active_company_proj, self._active_company_idx).unsqueeze(1),
            _slice_proj(x, self.phase_proj, self._phase_idx).unsqueeze(1),
            _slice_proj(x, self.num_players_proj, self._num_players_idx).unsqueeze(1),
            _slice_proj(x, self.game_progress_proj, self._game_progress_idx).unsqueeze(1),
            _slice_proj(x, self.invest_proj, self._invest_idx).unsqueeze(1),
            _slice_proj(x, self.auction_proj, self._auction_idx).unsqueeze(1),
            _slice_proj(x, self.dividend_proj, self._dividend_idx).unsqueeze(1),
            _slice_proj(x, self.issue_proj, self._issue_idx).unsqueeze(1),
            _slice_proj(x, self.par_proj, self._par_idx).unsqueeze(1),
            _slice_proj(x, self.acq_offer_proj, self._acq_offer_idx).unsqueeze(1),
            _slice_proj(x, self.acq_price_proj, self._acq_price_info_idx).unsqueeze(1),
            _slice_proj(x, self.corp_proj, self._corp_slice),                            # (B, 8, d)
            _slice_proj(x, self.player_proj, self._player_slice),                        # (B, N, d)
            self.pass_embed.view(1, 1, d).expand(x.shape[0], 1, d),  # Pass: learned anchor
        ]
        tokens = torch.cat(parts, dim=1)                      # (B, num_tokens, d)
        return tokens

    # ------------------------------------------------------------------
    # Phase-specific policy heads
    # ------------------------------------------------------------------

    def _policy_invest(self, t: torch.Tensor) -> torch.Tensor:
        """INVEST: pass(1) + company-select(36) + trade(8*2) = 53."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        # (n, 36, 1) → (n, 36) company-select logits, one per company token.
        auction = self.company_select_head(t[:, self._company_slice]).squeeze(-1)
        trade = self.corp_trade_head(t[:, self._corp_slice])                  # (n, 8, 2)
        # flatten(1) instead of reshape(n, -1): the latter is ambiguous when
        # n == 0 (empty mask in dispatch), since 0 elements / 0 rows is
        # undefined for the inferred dim.
        return torch.cat([pass_logit, auction, trade.flatten(1)], dim=-1)

    def _policy_bid(self, t: torch.Tensor) -> torch.Tensor:
        """BID: pass(1) + offsets(AUCTION_CAP) = 16. Pass = leave the auction."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        raises = self.auction_raise_head(t[:, self._auction_idx])             # (n, 15)
        return torch.cat([pass_logit, raises], dim=-1)

    def _policy_acq_select_corp(self, t: torch.Tensor) -> torch.Tensor:
        """ACQ_SELECT_CORP: pass(1) + per-corp select logit(8) = 9."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        select = self.corp_select_head(t[:, self._corp_slice]).squeeze(-1)    # (n, 8)
        return torch.cat([pass_logit, select], dim=-1)

    def _policy_acq_select_company(self, t: torch.Tensor) -> torch.Tensor:
        """ACQ_SELECT_COMPANY: 36 company-select logits. No pass."""
        return self.company_select_head(t[:, self._company_slice]).squeeze(-1)  # (n, 36)

    def _policy_acq_select_price(self, t: torch.Tensor) -> torch.Tensor:
        """ACQ_SELECT_PRICE: 52 price logits (51 offsets + FI_BUY). No pass."""
        return self.price_acq_head(t[:, self._acq_price_info_idx])            # (n, 52)

    def _policy_acq_offer(self, t: torch.Tensor) -> torch.Tensor:
        """ACQ_OFFER: pass(1) + buy(1) = 2."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        buy = self.acq_offer_head(t[:, self._acq_offer_idx])                  # (n, 1)
        return torch.cat([pass_logit, buy], dim=-1)

    def _policy_closing(self, t: torch.Tensor) -> torch.Tensor:
        """CLOSING: pass(1) + company_close(36) = 37."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        close = self.company_select_head(t[:, self._company_slice])           # (n, 36, 1)
        return torch.cat([pass_logit, close.squeeze(-1)], dim=-1)

    def _policy_dividends(self, t: torch.Tensor) -> torch.Tensor:
        """DIVIDENDS: 26 amounts."""
        return self.dividend_head(t[:, self._dividend_idx])                   # (n, 26)

    def _policy_issue(self, t: torch.Tensor) -> torch.Tensor:
        """ISSUE: pass(1) + issue(1) = 2."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        issue = self.issue_head(t[:, self._issue_idx])                        # (n, 1)
        return torch.cat([pass_logit, issue], dim=-1)

    def _policy_ipo(self, t: torch.Tensor) -> torch.Tensor:
        """IPO: pass(1) + per-corp select logit(8) = 9."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        select = self.corp_select_head(t[:, self._corp_slice]).squeeze(-1)    # (n, 8)
        return torch.cat([pass_logit, select], dim=-1)

    def _policy_par(self, t: torch.Tensor) -> torch.Tensor:
        """PAR: 14 par-price logits. No pass."""
        return self.par_price_head(t[:, self._par_idx])                       # (n, 14)

    # ------------------------------------------------------------------
    # Policy dispatch (handles mixed-phase batches)
    # ------------------------------------------------------------------

    def _policy_forward(
        self, tokens: torch.Tensor,
        action_ids: torch.Tensor, n_legals: torch.Tensor,
        phase_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Route each batch element to its phase-specific policy head and
        gather the per-row legal slice in one shot.

        The per-phase heads emit logits of shape ``(n_phase, phase_action_size)``
        — much narrower than the full ``MAX_ACTION_SIZE = 14977`` pad. Gathering
        against ``action_ids`` inside the model keeps the source tensor at
        phase-local width, so the evaluator / trainer / eval-server never have
        to allocate or scatter into the full ACQUISITION-wide scratch.

        Args:
            tokens: (batch, num_tokens, d_model) final token representations.
            action_ids: (batch, K_MAX) int tensor, phase-local legal action ids
                per row. Tail positions ([n_legals[i]:]) may hold arbitrary
                values — they are rewritten to 0 before ``gather`` and
                masked out before softmax.
            n_legals: (batch,) int tensor, legal action count per row.
            phase_indices: List of ``NUM_PHASES`` int64 tensors on
                ``tokens.device``. ``phase_indices[p]`` holds the row indices
                whose ``phase_id == p``. Dispatch uses ``index_select`` /
                ``index_copy_`` so the policy gather has no H←D sync — the
                per-iteration sync that boolean indexing would force (it has
                to read the mask's true-count to size the masked tensor)
                dominated eval latency on CPU-bound workloads. All callers
                build these on CPU from the pinned phase_ids buffer before the
                H→D copy and pass them in.

        Returns:
            (batch, K_MAX) float tensor of gathered logits, with positions
            beyond ``n_legals[i]`` filled with ``-1e9`` so they vanish after
            softmax.
        """
        B, K = action_ids.shape
        # Explicit fp32 rather than tokens.new_full: sentinels need no
        # precision, and a fixed dtype avoids a silent coupling to whatever
        # autocast leaves `tokens` as post-final_norm.
        out = torch.full((B, K), -1e9, dtype=torch.float32, device=tokens.device)

        # No `if not mask.any(): continue` early-exit: that path forces a
        # GPU→CPU sync (NUM_PHASES per forward) which dominates eval latency
        # on CPU-bound workloads like analyze_game. Empty masks/indices flow
        # cleanly through linear / cat / gather / masked_scatter as no-op
        # (0,*) ops, so we just dispatch all phases unconditionally and let
        # the GPU pipeline absorb the extra small launches.
        # `action_ids` tail positions [n_legals[i]:] may hold stale
        # garbage (shared-memory worker slots are reused across phases
        # and only [:n_legals[i]] is written per enumeration). We clamp
        # sub_ids to the phase head width before gather so the compiled
        # kernel never indexes out of range into narrow phase heads
        # (ACQ_OFFER / ISSUE have width 2). Clamp rather than
        # ``masked_fill(invalid, 0)``: Inductor will reorder a mask
        # whose effect is "overwritten later by -1e9" past the gather,
        # since the reasoning is that the gathered value at invalid
        # positions doesn't matter. That reordering is unsafe on CUDA
        # where gather bounds-checks and device-asserts regardless of
        # what happens to the result. An unconditional clamp has no
        # such dependence and survives fusion. Output is identical:
        # invalid-position values get masked to -1e9 after gather
        # either way.
        for phase_id in range(NUM_PHASES):
            idx = phase_indices[phase_id]
            phase_logits = self._dispatch[phase_id](tokens.index_select(0, idx))
            sub_ids = action_ids.index_select(0, idx).long()
            safe_ids = sub_ids.clamp(min=0, max=phase_logits.shape[1] - 1)
            gathered = phase_logits.gather(1, safe_ids)
            invalid = self._k_range[None, :] >= n_legals.index_select(0, idx)[:, None]
            # Cast to out.dtype: under autocast, tokens (post-RMSNorm) is
            # fp32 but phase_logits (from Linear) is bf16 — index_copy_
            # requires matching dtypes. torch.compile hides the mismatch
            # via fusion; eager (NNEvaluator) surfaces it.
            out.index_copy_(0, idx, gathered.masked_fill(invalid, -1e9).to(out.dtype))

        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor,
        action_ids: torch.Tensor, n_legals: torch.Tensor,
        phase_indices: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer.

        Args:
            x: (batch, num_tokens, token_dim) token features, zero-padded.
            action_ids: (batch, K_MAX) int tensor, phase-local legal action
                ids per row. Tail ([n_legals[i]:]) may be arbitrary — sanitized
                inside ``_policy_forward`` before gather.
            n_legals: (batch,) int tensor, legal action count per row.
            phase_indices: ``NUM_PHASES``-length list of int64 row indices per
                phase; see ``_policy_forward`` for details. Required — the
                policy gather uses ``index_select`` / ``index_copy_`` to avoid
                the per-batch H←D sync that boolean indexing would force.

        Returns:
            policy_logits: (batch, K_MAX) gathered logits over the per-row
                legal-action slice. Positions beyond ``n_legals[i]`` are
                filled with ``-1e9``.
            values: (batch, num_players) per-player expected outcomes in [-1, 1].
        """
        tokens = self._project_tokens(x)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        policy_logits = self._policy_forward(
            tokens, action_ids, n_legals, phase_indices,
        )
        values = self.value_head(tokens[:, self._player_slice]).squeeze(-1)  # (B, N)
        return policy_logits, values

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT/LLaMA-style trunc-normal init, zero-init residual outputs for identity start.

        kaiming_uniform_(nonlinearity="relu") is wrong for most Linears here
        (SDPA has no ReLU, SwiGLU/GELU heads aren't ReLU, value head feeds
        Tanh) and produces bounds ~10x wider than standard transformer init.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        # Pass token: small-random learned anchor (BERT [CLS] convention).
        nn.init.trunc_normal_(self.pass_embed, std=0.02)

        # Zero-init residual outputs so each block starts as identity
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            nn.init.zeros_(block.ffn_down.weight)
            nn.init.zeros_(block.out_proj.weight)
            nn.init.zeros_(block.out_proj.bias)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = TransformerConfig()
    model = RSSTransformerNet(cfg)
    total = count_parameters(model)

    print(f"Transformer model: {cfg.num_players}p")
    print(f"  d_model={cfg.d_model}, heads={cfg.num_heads}, "
          f"layers={cfg.num_layers}, d_ff={math.ceil(cfg.ff_mult * cfg.d_model)}")
    print(f"  tokens={cfg.num_tokens}, token_dim={cfg.token_dim}")
    print(f"  Trainable parameters: {total:,}")
    print()

    # --- Parameter breakdown ---
    proj_modules = [
        model.player_proj, model.corp_proj, model.company_proj,
        model.fi_proj, model.market_proj, model.market_slot_prices_proj,
        model.company_location_proj, model.company_adj_income_proj,
        model.active_player_proj, model.active_corp_proj, model.active_company_proj,
        model.phase_proj, model.num_players_proj, model.game_progress_proj,
        model.invest_proj, model.auction_proj, model.dividend_proj,
        model.issue_proj, model.par_proj, model.acq_offer_proj,
        model.acq_price_proj,
    ]
    proj_params = sum(sum(p.numel() for p in m.parameters()) for m in proj_modules)
    pass_params = model.pass_embed.numel()
    trunk_params = (
        sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.final_norm.parameters())
    )
    policy_modules = [
        model.pass_head, model.company_select_head, model.corp_select_head,
        model.corp_trade_head, model.auction_raise_head, model.dividend_head,
        model.issue_head, model.acq_offer_head,
        model.price_acq_head, model.par_price_head,
    ]
    policy_params = sum(sum(p.numel() for p in m.parameters()) for m in policy_modules)
    value_params = sum(p.numel() for p in model.value_head.parameters())

    print("Parameter breakdown:")
    for name, count in [
        ("Input projections", proj_params),
        ("Pass token", pass_params),
        ("Transformer trunk", trunk_params),
        ("Policy heads", policy_params),
        ("Value head", value_params),
    ]:
        print(f"  {name + ':':22s} {count:>10,}  ({count / total * 100:.1f}%)")

    phase_names = [
        "INVEST", "BID", "ACQ_SELECT_CORP", "ACQ_OFFER",
        "CLOSING", "DIVIDENDS", "ISSUE", "IPO", "PAR",
        "ACQ_SELECT_COMPANY", "ACQ_SELECT_PRICE",
    ]
    print()
    for name, size in zip(phase_names, PHASE_ACTION_SIZES):
        print(f"  {name:>12s}: {size:>3d} actions")

    # --- Smoke test ---
    print()
    batch_size = NUM_PHASES  # one sample per phase
    x = torch.randn(batch_size, cfg.num_tokens, cfg.token_dim)
    # Per-phase legal-action synthesis: use the first min(K_MAX, phase_size)
    # ids from each phase. Unused tail is zero-padded; n_legals masks it.
    action_ids = torch.zeros(batch_size, K_MAX, dtype=torch.long)
    n_legals = torch.zeros(batch_size, dtype=torch.long)
    for i in range(NUM_PHASES):
        n = min(K_MAX, PHASE_ACTION_SIZES[i])
        action_ids[i, :n] = torch.arange(n)
        n_legals[i] = n
    phase_indices = [torch.tensor([i], dtype=torch.int64) for i in range(NUM_PHASES)]

    policy_logits, values = model(x, action_ids, n_legals, phase_indices)

    print(f"policy_logits: {tuple(policy_logits.shape)}")
    print(f"values:        {tuple(values.shape)}")

    assert policy_logits.shape == (batch_size, K_MAX)
    assert values.shape == (batch_size, cfg.num_players)

    assert values.min() >= -1.0 and values.max() <= 1.0, "tanh output out of range"
    print("values in [-1, 1]: ok")

    for i in range(NUM_PHASES):
        n = int(n_legals[i])
        active = policy_logits[i, :n]
        inactive = policy_logits[i, n:]
        assert torch.isfinite(active).all(), f"{phase_names[i]}: non-finite logits"
        if inactive.numel() > 0:
            assert (inactive == -1e9).all(), f"{phase_names[i]}: leak beyond legal slice"
    print("per-row legal slice: ok")

    print("\nSmoke test passed.")
