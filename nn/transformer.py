"""Transformer model for Rolling Stock Stars AlphaZero training.

Token-based architecture: each game entity is a separate input token. Type-specific
linear projections -> L pre-LN transformer blocks -> entity-readout policy heads + value head.

Key differences from the MLP model (nn/template.py):
  - Input: (batch, num_tokens, token_dim) token features, not flat state vector
  - No state rotation: active player marked with is_active flag
  - Entity-readout policy: each entity token produces its own action logits
  - ACQ factored into three single-entity sub-phases (corp/company/price)
  - Unified policy output: every head writes into a static (B, UNIFIED_LOGIT_DIM)
    tensor and illegal slots are masked to -1e9 via a caller-supplied mask
  - Value read from player tokens directly (no un-rotation needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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
MAX_PHASE_ACTION_SIZE = max(PHASE_ACTION_SIZES)  # 53 (INVEST: pass + 36 + 16)

# Unified policy slot layout. Every per-row policy head emits its logits
# into a single (B, UNIFIED_LOGIT_DIM) tensor; callers pass a matching
# (B, UNIFIED_LOGIT_DIM) legal-mask so illegal slots can be zeroed before
# softmax. The static (NUM_PHASES, MAX_PHASE_ACTION_SIZE) action LUT maps
# each phase's phase-local action ids to unified slots — exposed via
# ``build_action_lut`` so callers can build masks and scatter training
# targets against the same layout. Offsets MUST match the concat order in
# ``RSSTransformerNet._build_unified_logits`` and the LUT construction.
_PASS_OFF = 0                                            # 1
_COMPANY_SELECT_OFF = _PASS_OFF + 1                      # 36
_CORP_SELECT_OFF = _COMPANY_SELECT_OFF + 36              # 8
_CORP_TRADE_OFF = _CORP_SELECT_OFF + 8                   # 16 (per-corp buy,sell)
_AUCTION_RAISE_OFF = _CORP_TRADE_OFF + 16                # AUCTION_CAP
_DIVIDEND_OFF = _AUCTION_RAISE_OFF + int(AUCTION_CAP)    # 26
_ISSUE_OFF = _DIVIDEND_OFF + 26                          # 1
_ACQ_OFFER_OFF = _ISSUE_OFF + 1                          # 1
_PRICE_OFF = _ACQ_OFFER_OFF + 1                          # 51
_PAR_OFF = _PRICE_OFF + 51                               # 14
UNIFIED_LOGIT_DIM = _PAR_OFF + 14

_GELU_APPROX = "tanh"


def build_action_lut() -> torch.Tensor:
    """Static (NUM_PHASES, MAX_PHASE_ACTION_SIZE) int64 LUT mapping each
    phase's phase-local action id to a slot in the unified logit tensor.

    Used externally by workers (to build (B, UNIFIED_LOGIT_DIM) legal masks
    from sparse (phase_id, action_ids[:n]) tuples) and by the trainer (to
    scatter sparse MCTS visit probabilities into dense (B, UNIFIED_LOGIT_DIM)
    policy targets). Tail entries (id >= PHASE_ACTION_SIZES[phase]) are 0 —
    a sentinel slot that workers must never mark as legal.

    Encoded layouts here MUST match the engine-side encoders in
    ``core/actions.pxd`` (``encode_invest_*`` etc). The smoke test below
    spot-checks the round-trip; the per-phase tests in ``tests/phases/``
    cover the action semantics end to end.
    """
    lut = torch.zeros(NUM_PHASES, MAX_PHASE_ACTION_SIZE, dtype=torch.long)

    # INVEST: 0=pass | 1..37=auction company 0..35 | 37..53=trade (corp i × {buy,sell})
    lut[DecisionPhase.DPHASE_INVEST, 0] = _PASS_OFF
    lut[DecisionPhase.DPHASE_INVEST, 1:37] = _COMPANY_SELECT_OFF + torch.arange(36)
    lut[DecisionPhase.DPHASE_INVEST, 37:53] = _CORP_TRADE_OFF + torch.arange(16)

    # BID: 0=pass | 1..16=raise offset 0..14
    lut[DecisionPhase.DPHASE_BID, 0] = _PASS_OFF
    lut[DecisionPhase.DPHASE_BID, 1:1 + int(AUCTION_CAP)] = (
        _AUCTION_RAISE_OFF + torch.arange(int(AUCTION_CAP))
    )

    # ACQ_SELECT_CORP: 0=pass | 1..9=corp 0..7
    lut[DecisionPhase.DPHASE_ACQ_SELECT_CORP, 0] = _PASS_OFF
    lut[DecisionPhase.DPHASE_ACQ_SELECT_CORP, 1:9] = _CORP_SELECT_OFF + torch.arange(8)

    # ACQ_OFFER: 0=pass | 1=accept-buy
    lut[DecisionPhase.DPHASE_ACQ_OFFER, 0] = _PASS_OFF
    lut[DecisionPhase.DPHASE_ACQ_OFFER, 1] = _ACQ_OFFER_OFF

    # CLOSING: 0=pass | 1..37=close company 0..35
    lut[DecisionPhase.DPHASE_CLOSING, 0] = _PASS_OFF
    lut[DecisionPhase.DPHASE_CLOSING, 1:37] = _COMPANY_SELECT_OFF + torch.arange(36)

    # DIVIDENDS: 0..26=level 0..25
    lut[DecisionPhase.DPHASE_DIVIDENDS, :26] = _DIVIDEND_OFF + torch.arange(26)

    # ISSUE: 0=pass | 1=issue
    lut[DecisionPhase.DPHASE_ISSUE, 0] = _PASS_OFF
    lut[DecisionPhase.DPHASE_ISSUE, 1] = _ISSUE_OFF

    # IPO: 0=pass | 1..9=corp 0..7 (shares same corp_select head as ACQ_SELECT_CORP)
    lut[DecisionPhase.DPHASE_IPO, 0] = _PASS_OFF
    lut[DecisionPhase.DPHASE_IPO, 1:9] = _CORP_SELECT_OFF + torch.arange(8)

    # PAR: 0..14=par index 0..13
    lut[DecisionPhase.DPHASE_PAR, :14] = _PAR_OFF + torch.arange(14)

    # ACQ_SELECT_COMPANY: 0..36=company 0..35 (no pass; shares company_select head)
    lut[DecisionPhase.DPHASE_ACQ_SELECT_COMPANY, :36] = _COMPANY_SELECT_OFF + torch.arange(36)

    # ACQ_SELECT_PRICE: 0..51=offset 0..50 (no pass; FI targets execute in SELECT_COMPANY)
    lut[DecisionPhase.DPHASE_ACQ_SELECT_PRICE, :51] = _PRICE_OFF + torch.arange(51)

    return lut


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
    d_model: int = 192
    num_heads: int = 3
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
        """Input-buffer token count: 55 fixed entity/phase tokens + N players.

        The trunk sequence is 1 wider because ``_project_tokens`` concatenates
        a single learned pass anchor after projection; that row has no
        input features so it doesn't exist in the engine-side buffer.
        """
        return self.num_players + 55


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
        [int(TokenWidth.TW_MARKET_INFO)]
        + [int(TokenWidth.TW_COMPANY)] * 36
        + [int(TokenWidth.TW_FI)]
        + [int(TokenWidth.TW_GLOBAL_INFO)]
        + [int(TokenWidth.TW_INVEST)]
        + [int(TokenWidth.TW_AUCTION)]
        + [int(TokenWidth.TW_DIVIDEND)]
        + [int(TokenWidth.TW_ISSUE)]
        + [int(TokenWidth.TW_PAR)]
        + [int(TokenWidth.TW_ACQ_SELECT_COMPANY)]
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

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        np_ = cfg.num_players

        # --- Token index bookkeeping ---
        # Buffer layout (matches core/token_data.pyx::_fill_buffer):
        #   info: market_info (slot prices + per-space availability),
        #     companies×36 (static data + CoO-adjusted income + is_selected +
        #     at_*/owner_* groups), FI, global_info (decision phase + CoO +
        #     end-card + cards-remaining + num_players)
        #   phase-specific: invest, auction, dividend, issue, par,
        #     acq_select_company, acq_offer, acq_price_info
        #   corps×8, then players×N (trailing so padding for higher player
        #   counts is a no-op on the prefix).
        # The pass anchor is concatenated after projection; see
        # ``_project_tokens``. It lives beyond the player slice, so player
        # indices stay contiguous for the value head and the padding contract.
        self._market_info_idx = 0
        self._company_slice = slice(1, 37)
        self._fi_idx = 37
        self._global_info_idx = 38
        self._invest_idx = 39
        self._auction_idx = 40
        self._dividend_idx = 41
        self._issue_idx = 42
        self._par_idx = 43
        self._acq_select_company_idx = 44
        self._acq_offer_idx = 45
        self._acq_price_info_idx = 46
        self._corp_slice = slice(47, 55)
        self._player_slice = slice(55, 55 + np_)
        # Single learned pass anchor, appended after the player slice. Shared
        # across every pass-using phase — the trunk picks up phase-specific
        # context through attention so one anchor can back all 7 passes.
        self._pass_idx = 55 + np_

        # Drift guard: hardcoded positions above must match the Cython-side
        # ``get_token_widths`` layout. Checking here fires loudly at model
        # construction rather than silently feeding mis-aligned features to
        # the trunk.
        _validate_layout(np_)

        # --- Type-specific input projections ---
        # Each projection takes only its ``TokenWidth.TW_<type>`` prefix of the
        # zero-padded token row. The engine-side buffer is rectangular at
        # ``TOKEN_DIM=93`` so ``get_token_data`` can fill it with a single
        # nogil memcpy pattern, but projection weights on the padding are
        # inert waste — slicing to the actual width here (both in sizing and
        # in ``_project_tokens``) drops those parameters. Widths are pulled
        # from ``TokenWidth`` so the model and the Cython extractor can't
        # drift out of sync.
        self.player_proj = nn.Linear(int(TokenWidth.TW_PLAYER), d)
        self.corp_proj = nn.Linear(int(TokenWidth.TW_CORP), d)
        self.company_proj = nn.Linear(int(TokenWidth.TW_COMPANY), d)
        self.fi_proj = nn.Linear(int(TokenWidth.TW_FI), d)
        self.market_info_proj = nn.Linear(int(TokenWidth.TW_MARKET_INFO), d)
        self.global_info_proj = nn.Linear(int(TokenWidth.TW_GLOBAL_INFO), d)
        self.invest_proj = nn.Linear(int(TokenWidth.TW_INVEST), d)
        self.auction_proj = nn.Linear(int(TokenWidth.TW_AUCTION), d)
        self.dividend_proj = nn.Linear(int(TokenWidth.TW_DIVIDEND), d)
        self.issue_proj = nn.Linear(int(TokenWidth.TW_ISSUE), d)
        self.par_proj = nn.Linear(int(TokenWidth.TW_PAR), d)
        self.acq_select_company_proj = nn.Linear(int(TokenWidth.TW_ACQ_SELECT_COMPANY), d)
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
        self.pass_head = self._make_policy_head(1)
        # Shared company-selection head used by INVEST (which company to
        # auction), ACQ_SELECT_COMPANY (which company to acquire), and CLOSING
        # (which company to close). One logit per company token; the phase
        # context that discriminates the three decisions arrives through the
        # trunk's phase-specific context tokens + attention.
        self.company_select_head = self._make_policy_head(1)
        # Shared corp-selection head used by ACQ_SELECT_CORP (which corp does
        # the acquiring) and IPO (which corp floats the active company). Same
        # structure as company_select_head; phase context reaches each corp
        # token through attention on the phase-specific context tokens.
        self.corp_select_head = self._make_policy_head(1)
        self.corp_trade_head = self._make_policy_head(2)

        # Phase-specific context token heads. BID bids at face_value + offset
        # for offset ∈ [0, AUCTION_CAP), so the head produces AUCTION_CAP
        # logits — one per legal bid offset (both opening and subsequent).
        self.auction_raise_head = self._make_policy_head(int(AUCTION_CAP))
        self.dividend_head = self._make_policy_head(26)
        self.issue_head = self._make_policy_head(1)
        self.acq_offer_head = self._make_policy_head(1)

        # ACQ is factored into three sequential single-entity selections:
        # pick the acquiring corp (shared corp_select_head above), pick the
        # target company (shared company_select_head above), pick the price.
        # SELECT_PRICE: 51 price offsets, read off a dedicated acq_price_info
        # token that the engine populates with (active_corp, active_company)
        # context during PHASE_ACQ_SELECT_PRICE. FI targets execute in
        # SELECT_COMPANY at the fixed FI price, so this head never fires for
        # them.
        self.price_acq_head = self._make_policy_head(51)

        # PAR reads 14 par-price logits from the par info token. No pass
        # anchor: PAR has no pass action — once a corp is selected the owner
        # must commit to a price.
        self.par_price_head = self._make_policy_head(14)

        # --- Value head (applied per player token) ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _make_policy_head(self, out_features: int) -> nn.Sequential:
        """Standard 2-layer policy head: Linear(d, d//2) -> GELU -> Linear(d//2, out)."""
        d = self.cfg.d_model
        return nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, out_features),
        )

    # ------------------------------------------------------------------
    # Input projection
    # ------------------------------------------------------------------

    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw token features to d_model via type-specific projections.

        The pass anchor is concatenated here (not present in the input buffer),
        so the output sequence is 1 wider than the input.

        Each row is sliced to its projection's ``in_features`` width before
        projecting. The engine-side buffer is rectangularly zero-padded to
        ``TOKEN_DIM=93`` so ``get_token_data`` can fill it with a uniform
        nogil memcpy pattern, but each type only uses its ``TokenWidth.TW_*``
        prefix — the tail is zero padding that would otherwise be multiplied
        against inert projection weights.

        Args:
            x: (batch, num_players + 55, token_dim) zero-padded raw features.
                Supported caller patterns are: fp16/fp32 under autocast on the
                eval path, or fp32 matching the projection weights on the
                non-autocast trainer / CPU paths. No explicit upcast happens
                here: an unconditional ``x.to(bf16)`` would mismatch fp32
                Linear weights in non-autocast paths (for example in-process
                NNEvaluator tests).
        Returns:
            (batch, num_players + 56, d_model) embeddings: projected input
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
            _slice_proj(x, self.market_info_proj, self._market_info_idx).unsqueeze(1),
            _slice_proj(x, self.company_proj, self._company_slice),                      # (B, 36, d)
            _slice_proj(x, self.fi_proj, self._fi_idx).unsqueeze(1),
            _slice_proj(x, self.global_info_proj, self._global_info_idx).unsqueeze(1),
            _slice_proj(x, self.invest_proj, self._invest_idx).unsqueeze(1),
            _slice_proj(x, self.auction_proj, self._auction_idx).unsqueeze(1),
            _slice_proj(x, self.dividend_proj, self._dividend_idx).unsqueeze(1),
            _slice_proj(x, self.issue_proj, self._issue_idx).unsqueeze(1),
            _slice_proj(x, self.par_proj, self._par_idx).unsqueeze(1),
            _slice_proj(x, self.acq_select_company_proj, self._acq_select_company_idx).unsqueeze(1),
            _slice_proj(x, self.acq_offer_proj, self._acq_offer_idx).unsqueeze(1),
            _slice_proj(x, self.acq_price_proj, self._acq_price_info_idx).unsqueeze(1),
            _slice_proj(x, self.corp_proj, self._corp_slice),                            # (B, 8, d)
            _slice_proj(x, self.player_proj, self._player_slice),                        # (B, N, d)
            self.pass_embed.view(1, 1, d).expand(x.shape[0], 1, d),  # Pass: learned anchor
        ]
        tokens = torch.cat(parts, dim=1)                      # (B, num_tokens, d)
        return tokens

    # ------------------------------------------------------------------
    # Unified policy: every head runs once on the full batch
    # ------------------------------------------------------------------

    def _build_unified_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run every per-row policy head once on the full batch and concat
        into a single ``(B, UNIFIED_LOGIT_DIM)`` tensor.

        Concat order MUST match the ``_*_OFF`` constants and ``build_action_lut``.
        Each head reads the token(s) it cares about — companies / corps for the
        entity-readout heads, the phase-context tokens (auction, dividend, etc.)
        for the others, and the shared pass anchor. Heads run unconditionally
        regardless of which phase a given row is in: the caller's legal mask
        zeroes out slots outside the current phase's action space. Total wasted
        FLOPs are ~2% of the trunk at d_model=128.
        """
        # All single-token heads keep their input dim of (B, d_model); their
        # output is (B, head_width). Multi-token heads stay (B, n_tokens, ...)
        # → flattened to (B, n_tokens * out_dim).
        pass_logit = self.pass_head(tokens[:, self._pass_idx])                  # (B, 1)
        # company_select_head: shared by INVEST / ACQ_SELECT_COMPANY / CLOSING.
        company_select = self.company_select_head(
            tokens[:, self._company_slice]
        ).squeeze(-1)                                                           # (B, 36)
        # corp_select_head: shared by ACQ_SELECT_CORP / IPO.
        corp_select = self.corp_select_head(
            tokens[:, self._corp_slice]
        ).squeeze(-1)                                                           # (B, 8)
        # corp_trade_head: only INVEST. Layout matches encode_invest_buy /
        # encode_invest_sell — slot 2i = corp i buy, slot 2i+1 = corp i sell.
        corp_trade = self.corp_trade_head(
            tokens[:, self._corp_slice]
        ).flatten(1)                                                            # (B, 16)
        auction_raise = self.auction_raise_head(tokens[:, self._auction_idx])   # (B, 15)
        dividend = self.dividend_head(tokens[:, self._dividend_idx])            # (B, 26)
        issue = self.issue_head(tokens[:, self._issue_idx])                     # (B, 1)
        acq_offer = self.acq_offer_head(tokens[:, self._acq_offer_idx])         # (B, 1)
        price_acq = self.price_acq_head(tokens[:, self._acq_price_info_idx])    # (B, 51)
        par_price = self.par_price_head(tokens[:, self._par_idx])               # (B, 14)
        return torch.cat(
            [
                pass_logit, company_select, corp_select, corp_trade,
                auction_raise, dividend, issue, acq_offer, price_acq, par_price,
            ],
            dim=-1,
        )                                                                        # (B, UNIFIED_LOGIT_DIM)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer.

        Args:
            x: ``(batch, cfg.num_tokens, cfg.token_dim)`` floating-point token
                features, zero-padded to ``cfg.token_dim``. There is no
                unconditional dtype conversion here: eval paths rely on
                autocast to run the projections/trunk in bf16/fp16, while the
                trainer and CPU tests pass fp32 directly.
            legal_mask: ``(batch, UNIFIED_LOGIT_DIM)`` bool tensor on the same
                device as ``x``. ``True`` marks a slot as legal for the current
                state's phase / entities; illegal slots are masked to ``-1e9``.
                Every real row must mark at least one legal slot. An all-false
                row is only valid for caller-reserved scratch rows whose output
                will be ignored (for example the eval server's trash row);
                if such a row is consumed by softmax downstream it becomes a
                near-uniform distribution over the unified slots.

        Returns:
            policy_logits: ``(batch, UNIFIED_LOGIT_DIM)`` fp32 logits with
                illegal slots set to ``-1e9``. Static-shape regardless of phase.
            values: ``(batch, num_players)`` per-player expected outcomes in
                ``[-1, 1]``.
        """
        if x.ndim != 3:
            raise AssertionError(f"x must be rank-3 (batch, num_tokens, token_dim); got {tuple(x.shape)}")
        expected_x_shape = (x.shape[0], self.cfg.num_tokens, self.cfg.token_dim)
        if tuple(x.shape) != expected_x_shape:
            raise AssertionError(f"x shape must be {expected_x_shape}; got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise AssertionError(f"x must be floating-point token features; got {x.dtype}")
        if legal_mask.dtype != torch.bool:
            raise AssertionError(
                f"legal_mask must be bool (uint8 would make ~legal_mask a bitwise "
                f"complement, not logical NOT); got {legal_mask.dtype}"
            )
        expected_mask_shape = (x.shape[0], UNIFIED_LOGIT_DIM)
        if tuple(legal_mask.shape) != expected_mask_shape:
            raise AssertionError(
                f"legal_mask shape must be {expected_mask_shape}; got {tuple(legal_mask.shape)}"
            )
        if legal_mask.device != x.device:
            raise AssertionError(
                f"legal_mask device must match x device; got {legal_mask.device} vs {x.device}"
            )
        tokens = self._project_tokens(x)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        # Cast to fp32 before the sentinel: under autocast ``unified`` is in
        # the autocast dtype (bf16/fp16) but downstream softmax / log_softmax
        # wants stable fp32 ``-1e9`` sentinels. Real rows are expected to have
        # at least one legal slot; all-false rows are reserved for caller-owned
        # scratch entries whose output will be dropped before softmax consumers
        # interpret them.
        unified = self._build_unified_logits(tokens).to(torch.float32)          # (B, U)
        policy_logits = unified.masked_fill(~legal_mask, -1e9)

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
        model.fi_proj, model.market_info_proj, model.global_info_proj,
        model.invest_proj, model.auction_proj, model.dividend_proj,
        model.issue_proj, model.par_proj, model.acq_select_company_proj,
        model.acq_offer_proj, model.acq_price_proj,
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

    # Synthesize a legal mask per row by running every phase's full
    # phase-local action list through the LUT. Row i gets phase i.
    lut = build_action_lut()
    legal_mask = torch.zeros(batch_size, UNIFIED_LOGIT_DIM, dtype=torch.bool)
    for i in range(NUM_PHASES):
        n = PHASE_ACTION_SIZES[i]
        legal_mask[i, lut[i, :n]] = True

    policy_logits, values = model(x, legal_mask)

    print(f"policy_logits: {tuple(policy_logits.shape)}")
    print(f"values:        {tuple(values.shape)}")

    assert policy_logits.shape == (batch_size, UNIFIED_LOGIT_DIM)
    assert values.shape == (batch_size, cfg.num_players)

    assert values.min() >= -1.0 and values.max() <= 1.0, "tanh output out of range"
    print("values in [-1, 1]: ok")

    for i in range(NUM_PHASES):
        legal = policy_logits[i][legal_mask[i]]
        illegal = policy_logits[i][~legal_mask[i]]
        assert torch.isfinite(legal).all(), f"{phase_names[i]}: non-finite legal logits"
        if illegal.numel() > 0:
            assert (illegal == -1e9).all(), f"{phase_names[i]}: leak into illegal slots"
    print("per-row legal mask: ok")

    print("\nSmoke test passed.")
