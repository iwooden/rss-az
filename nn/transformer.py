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
    GameConstants,
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
# softmax. Each DecisionPhase owns a contiguous, non-overlapping block —
# laid out in DecisionPhase order with block widths from PHASE_ACTION_SIZES.
# Head weights are unshared across phases (no cross-phase gradient
# interference on shared readouts), so the LUT reduces to
# ``lut[phase, i] = cumsum_offset[phase] + i``. ``build_action_lut`` is the
# sole public pointer into the layout — callers never import per-block
# offsets, which keeps this internal.
_PHASE_OFFSETS: list[int] = [0]
for _size in PHASE_ACTION_SIZES:
    _PHASE_OFFSETS.append(_PHASE_OFFSETS[-1] + int(_size))
UNIFIED_LOGIT_DIM = _PHASE_OFFSETS[-1]  # 255

# Pass actions that do not have a natural phase-info-token readout. These use
# learned anchors appended to the trunk sequence, one per phase below. BID,
# ACQ_OFFER, and ISSUE also have pass actions, but their full phase blocks are
# read directly from phase-specific information tokens because those tokens
# already score the non-pass alternatives.
PASS_PHASE_IDS: tuple[int, ...] = (
    int(DecisionPhase.DPHASE_INVEST),
    int(DecisionPhase.DPHASE_ACQ_SELECT_CORP),
    int(DecisionPhase.DPHASE_CLOSING),
    int(DecisionPhase.DPHASE_IPO),
)
NUM_PASS_PHASES = len(PASS_PHASE_IDS)  # 4

# Input-buffer token-type taxonomy. Index order must stay stable — the static
# ``_type_ids`` buffer built in ``RSSTransformerNet.__init__`` indexes into
# ``type_embeds`` using these ids, and multi-instance types (company / corp
# / player) share a single row. Pass tokens are absent here — they're
# already pure learned anchors (``pass_embeds``) so no type embedding is
# needed for them.
_TYPE_MARKET_INFO = 0
_TYPE_COMPANY = 1
_TYPE_FI = 2
_TYPE_GLOBAL_INFO = 3
_TYPE_INVEST = 4
_TYPE_AUCTION = 5
_TYPE_DIVIDEND = 6
_TYPE_ISSUE = 7
_TYPE_PAR = 8
_TYPE_ACQ_SELECT_COMPANY = 9
_TYPE_ACQ_OFFER = 10
_TYPE_ACQ_PRICE = 11
_TYPE_CORP = 12
_TYPE_PLAYER = 13
NUM_TOKEN_TYPES = 14

_GELU_APPROX = "tanh"
_NUM_COMPANIES = int(GameConstants.NUM_COMPANIES)
_NUM_CORPS = int(GameConstants.NUM_CORPS)
_MAX_MODEL_PLAYERS = 5
_CORP_REL_OFFSET = 44
_CORP_PRESIDENT_WIDTH = _MAX_MODEL_PLAYERS
_CORP_COMPANIES_WIDTH = _NUM_COMPANIES
_FI_COMPANIES_OFFSET = 2
_FI_COMPANIES_WIDTH = _NUM_COMPANIES
_PLAYER_REL_OFFSET = 12
_PLAYER_SHARES_WIDTH = _NUM_CORPS
_PLAYER_COMPANIES_WIDTH = _NUM_COMPANIES
_COMPANY_OWNER_OFFSET = 12
_COMPANY_OWNER_WIDTH = _NUM_CORPS + _MAX_MODEL_PLAYERS + 1
_GLOBAL_PHASE_WIDTH = NUM_PHASES


def _phase_action_size(phase: DecisionPhase) -> int:
    return int(PHASE_ACTION_SIZES[int(phase)])


def build_action_lut() -> torch.Tensor:
    """Static (NUM_PHASES, MAX_PHASE_ACTION_SIZE) int64 LUT mapping each
    phase's phase-local action id to a slot in the unified logit tensor.

    Used externally by workers (to build (B, UNIFIED_LOGIT_DIM) legal masks
    from sparse (phase_id, action_ids[:n]) tuples) and by the trainer (to
    scatter sparse MCTS visit probabilities into dense (B, UNIFIED_LOGIT_DIM)
    policy targets). Tail entries (id >= PHASE_ACTION_SIZES[phase]) are 0 —
    a sentinel slot that workers must never mark as legal.

    Layout is block-per-phase in DecisionPhase order: each phase owns
    ``PHASE_ACTION_SIZES[phase]`` contiguous slots starting at
    ``_PHASE_OFFSETS[phase]``. The phase-local action id *is* the
    intra-block offset, so the per-phase action encoding in
    ``core/actions.pxd`` is preserved 1:1 inside each block.
    """
    lut = torch.zeros(NUM_PHASES, MAX_PHASE_ACTION_SIZE, dtype=torch.long)
    for phase in range(NUM_PHASES):
        size = int(PHASE_ACTION_SIZES[phase])
        lut[phase, :size] = _PHASE_OFFSETS[phase] + torch.arange(size)
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
    """All dimensions parameterized. Defaults are 3-player with d_model=192."""

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

        The trunk sequence is ``NUM_PASS_PHASES`` wider because
        ``_project_tokens`` concatenates learned pass anchors for
        entity-readout pass phases after projection; those rows have no
        input features so they don't exist in the engine-side buffer.
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

    # Class-level annotation so pyright knows ``self._type_ids`` (registered
    # as a buffer in ``__init__``) is a Tensor. ``register_buffer`` otherwise
    # returns ``Tensor | Module | None`` per pytorch's stubs, which breaks
    # ``type_embeds[self._type_ids]`` indexing.
    _type_ids: torch.Tensor
    _company_ids: torch.Tensor
    _corp_ids: torch.Tensor
    _player_ids: torch.Tensor
    _active_ref_targets: torch.Tensor
    _phase_ref_targets: torch.Tensor

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        np_ = cfg.num_players

        # --- Token index bookkeeping ---
        # Buffer layout (matches core/token_data.pyx::_fill_buffer):
        #   info: market_info (slot prices + per-space availability),
        #     companies×36 (is_selected + static data + CoO-adjusted income +
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
        # One learned pass anchor per entity-readout pass phase, appended
        # after the player slice as a contiguous block. Direct-token phases
        # (BID, ACQ_OFFER, ISSUE) emit pass logits from their info-token heads.
        # Indices aligned with ``PASS_PHASE_IDS``.
        self._pass_base = 55 + np_
        self._pass_idxs: list[int] = [self._pass_base + i for i in range(NUM_PASS_PHASES)]

        # Drift guard: hardcoded positions above must match the Cython-side
        # ``get_token_widths`` layout. Checking here fires loudly at model
        # construction rather than silently feeding mis-aligned features to
        # the trunk.
        _validate_layout(np_)

        # --- Type-specific input projections ---
        # Most projections take only their ``TokenWidth.TW_<type>`` prefix of
        # the zero-padded token row. Company, corp, and player identity comes
        # from row order plus learned ID embeddings, not raw ID fields. Company
        # tokens skip their trailing ownership fields and receive an additive
        # owner reference from the matching corp / player / FI embedding. Corp
        # tokens skip their trailing president/company ownership fields and
        # receive additive player/company references from the same ID tables.
        # FI skips its trailing owned-company bitmap and receives additive
        # company references from ``company_id_embed``. Player tokens skip
        # their trailing owned-share/company fields and receive additive
        # corp/company references from the same ID tables.
        # The engine-side buffer is rectangular at ``TOKEN_DIM=85`` so
        # ``get_token_data`` can fill it with a single nogil memcpy pattern,
        # but projection weights on padding / replaced relation fields are inert
        # waste — slicing in both sizing and ``_project_tokens`` drops them.
        # Widths are pulled from ``TokenWidth`` so the model and the Cython
        # extractor can't drift out of sync.
        self._player_rel_offset = _PLAYER_REL_OFFSET
        self._player_shares_offset = _PLAYER_REL_OFFSET
        self._player_shares_width = _PLAYER_SHARES_WIDTH
        self._player_companies_offset = self._player_shares_offset + self._player_shares_width
        self._player_companies_width = _PLAYER_COMPANIES_WIDTH
        self.player_proj = nn.Linear(self._player_rel_offset, d)
        self._corp_rel_offset = _CORP_REL_OFFSET
        self._corp_president_offset = _CORP_REL_OFFSET
        self._corp_president_width = _CORP_PRESIDENT_WIDTH
        self._corp_companies_offset = self._corp_president_offset + self._corp_president_width
        self._corp_companies_width = _CORP_COMPANIES_WIDTH
        self.corp_proj = nn.Linear(self._corp_rel_offset, d)
        self._company_owner_offset = _COMPANY_OWNER_OFFSET
        self._company_owner_corp_offset = _COMPANY_OWNER_OFFSET
        self._company_owner_player_offset = _COMPANY_OWNER_OFFSET + _NUM_CORPS
        self._company_owner_fi_offset = self._company_owner_player_offset + _MAX_MODEL_PLAYERS
        self._company_owner_width = _COMPANY_OWNER_WIDTH
        self._company_owner_corp_width = _NUM_CORPS
        self._company_owner_player_width = _MAX_MODEL_PLAYERS
        self._company_owner_fi_width = 1
        self.company_proj = nn.Linear(
            self._company_owner_offset,
            d,
        )
        self._fi_companies_offset = _FI_COMPANIES_OFFSET
        self._fi_companies_width = _FI_COMPANIES_WIDTH
        self.fi_proj = nn.Linear(self._fi_companies_offset, d)
        self.market_info_proj = nn.Linear(int(TokenWidth.TW_MARKET_INFO), d)
        self._global_phase_width = _GLOBAL_PHASE_WIDTH
        self.global_info_proj = nn.Linear(
            int(TokenWidth.TW_GLOBAL_INFO) - self._global_phase_width,
            d,
        )
        self.invest_proj = nn.Linear(int(TokenWidth.TW_INVEST), d)
        self.auction_proj = nn.Linear(int(TokenWidth.TW_AUCTION), d)
        self.dividend_proj = nn.Linear(int(TokenWidth.TW_DIVIDEND), d)
        self.issue_proj = nn.Linear(int(TokenWidth.TW_ISSUE), d)
        self.par_proj = nn.Linear(int(TokenWidth.TW_PAR), d)
        self.acq_select_company_proj = nn.Linear(int(TokenWidth.TW_ACQ_SELECT_COMPANY), d)
        self.acq_offer_proj = nn.Linear(int(TokenWidth.TW_ACQ_OFFER), d)
        self.acq_price_proj = nn.Linear(int(TokenWidth.TW_ACQ_PRICE), d)
        # Pass tokens: no input features. Entity-readout pass phases get
        # learned (d_model,) vectors (BERT [CLS]-style anchors) that ride the
        # residual stream and pick up game-state context through attention.
        # Direct-token phases read pass from their phase-info token instead.
        self.pass_embeds = nn.Parameter(torch.empty(NUM_PASS_PHASES, d))
        # Learned entity identity embeddings are added from token row order.
        # The engine no longer writes entity ID one-hots into the token rows.
        self.company_id_embed = nn.Embedding(_NUM_COMPANIES, d)
        self.corp_id_embed = nn.Embedding(_NUM_CORPS, d)
        self.player_id_embed = nn.Embedding(_MAX_MODEL_PLAYERS, d)
        self.phase_embed = nn.Embedding(NUM_PHASES, d)
        # Per-type additive embedding for every non-pass token. Added
        # post-projection in ``_project_tokens`` so the trunk still sees a
        # type-distinct vector even when a token's feature slice is all-zero
        # (e.g. the DIVIDEND context token outside DIVIDENDS, or the owned-
        # company field of a player with no companies). Without this, zero
        # features + zero-initialized Linear biases collapse the token to the
        # zero vector on day one, and the only path to type discrimination is
        # an indirect gradient through the Linear's bias. Broadcast across
        # all instances of a type (36 companies, 8 corps, N players). Company,
        # corp, and player self-identity, plus relational references, come
        # from their dedicated learned additive paths.
        self.type_embeds = nn.Parameter(torch.empty(NUM_TOKEN_TYPES, d))
        # Static (num_players + 55,) type-id lookup. Built once here so
        # ``_project_tokens`` can do a single indexed gather against
        # ``type_embeds``. Registered as a buffer so ``.to(device)`` carries
        # it along. Must match the concat order inside ``_project_tokens``.
        type_ids = torch.empty(np_ + 55, dtype=torch.long)
        type_ids[self._market_info_idx] = _TYPE_MARKET_INFO
        type_ids[self._company_slice] = _TYPE_COMPANY
        type_ids[self._fi_idx] = _TYPE_FI
        type_ids[self._global_info_idx] = _TYPE_GLOBAL_INFO
        type_ids[self._invest_idx] = _TYPE_INVEST
        type_ids[self._auction_idx] = _TYPE_AUCTION
        type_ids[self._dividend_idx] = _TYPE_DIVIDEND
        type_ids[self._issue_idx] = _TYPE_ISSUE
        type_ids[self._par_idx] = _TYPE_PAR
        type_ids[self._acq_select_company_idx] = _TYPE_ACQ_SELECT_COMPANY
        type_ids[self._acq_offer_idx] = _TYPE_ACQ_OFFER
        type_ids[self._acq_price_info_idx] = _TYPE_ACQ_PRICE
        type_ids[self._corp_slice] = _TYPE_CORP
        type_ids[self._player_slice] = _TYPE_PLAYER
        self.register_buffer("_type_ids", type_ids, persistent=False)

        # Active-entity reference targets span the phase/query tokens plus the
        # learned pass anchors. MarketInfo, GlobalInfo, FI, and player/corp/
        # company entity tokens keep their own factual representations clean:
        # entity rows still carry ``is_selected`` in their projected feature
        # prefix, so the trunk can recover active identity through attention.
        projected_tokens = np_ + 55 + NUM_PASS_PHASES
        active_ref_targets = torch.zeros(projected_tokens, dtype=torch.float32)
        active_ref_targets[self._invest_idx:self._acq_price_info_idx + 1] = 1.0
        active_ref_targets[self._pass_idxs] = 1.0
        self.register_buffer("_active_ref_targets", active_ref_targets, persistent=False)

        phase_ref_targets = torch.ones(projected_tokens, dtype=torch.float32)
        phase_ref_targets[self._global_info_idx] = 0.0
        self.register_buffer("_phase_ref_targets", phase_ref_targets, persistent=False)

        self.register_buffer(
            "_company_ids", torch.arange(_NUM_COMPANIES, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "_corp_ids", torch.arange(_NUM_CORPS, dtype=torch.long), persistent=False,
        )
        self.register_buffer(
            "_player_ids", torch.arange(np_, dtype=torch.long), persistent=False,
        )

        # --- Transformer trunk ---
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.num_heads, math.ceil(cfg.ff_mult * cfg.d_model))
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.RMSNorm(d)

        # --- Entity-readout policy heads ---
        # Per-phase pass heads for entity-readout pass actions (aligned with
        # ``PASS_PHASE_IDS`` / ``pass_embeds``). Each reads its own anchor and
        # emits a single pass logit.
        self.pass_heads = nn.ModuleList(
            [self._make_policy_head(1) for _ in range(NUM_PASS_PHASES)]
        )
        # Company-selection heads: one per phase that picks a company. All
        # three are entity-readout heads — ``Linear(d, 1)`` applied per
        # company token (weight-shared across the 36 slots of the same phase,
        # distinct across phases). Phase context reaches company tokens via
        # attention on the phase-specific context tokens + global_info.
        self.invest_company_select_head = self._make_policy_head(1)
        self.closing_company_select_head = self._make_policy_head(1)
        self.acq_select_company_head = self._make_policy_head(1)
        # Corp-selection heads: one per phase that picks a corp. Same
        # entity-readout structure as the company heads above; distinct
        # weights for ACQ_SELECT_CORP vs. IPO avoid their gradients colliding.
        self.acq_select_corp_head = self._make_policy_head(1)
        self.ipo_corp_select_head = self._make_policy_head(1)
        self.corp_trade_head = self._make_policy_head(2)

        # Phase-specific context token heads. BID / ACQ_OFFER / ISSUE emit
        # full phase blocks from their phase-info tokens, including the pass
        # logit at phase-local action 0.
        self.bid_head = self._make_policy_head(
            _phase_action_size(DecisionPhase.DPHASE_BID)
        )
        self.dividend_head = self._make_policy_head(
            _phase_action_size(DecisionPhase.DPHASE_DIVIDENDS)
        )
        self.issue_head = self._make_policy_head(
            _phase_action_size(DecisionPhase.DPHASE_ISSUE)
        )
        self.acq_offer_head = self._make_policy_head(
            _phase_action_size(DecisionPhase.DPHASE_ACQ_OFFER)
        )

        # ACQ is factored into three sequential single-entity selections:
        # pick the acquiring corp (``acq_select_corp_head``), pick the target
        # company (``acq_select_company_head``), pick the price (below).
        # SELECT_PRICE: 51 price offsets, read off a dedicated acq_price_info
        # token that the engine populates with (active_corp, active_company)
        # context during PHASE_ACQ_SELECT_PRICE. FI targets execute in
        # SELECT_COMPANY at the fixed FI price, so this head never fires for
        # them.
        self.price_acq_head = self._make_policy_head(
            _phase_action_size(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)
        )

        # PAR reads 14 par-price logits from the par info token. No pass
        # anchor: PAR has no pass action — once a corp is selected the owner
        # must commit to a price.
        self.par_price_head = self._make_policy_head(
            _phase_action_size(DecisionPhase.DPHASE_PAR)
        )

        # --- Value head (applied per player token) ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

        self._validate_policy_layout()
        self._init_weights()

    def _make_policy_head(self, out_features: int) -> nn.Sequential:
        """Standard 2-layer policy head: Linear(d, d//2) -> GELU -> Linear(d//2, out)."""
        d = self.cfg.d_model
        return nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, out_features),
        )

    def _project_company_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project company tokens, adding row-order ID and owner references."""
        owner_start = self._company_owner_offset
        owner_stop = owner_start + self._company_owner_width
        company_raw = x[
            :,
            self._company_slice,
            :int(TokenWidth.TW_COMPANY),
        ]
        company_features = company_raw[:, :, :owner_start]
        company_tokens = self.company_proj(company_features)
        company_tokens = (
            company_tokens
            + self.company_id_embed(self._company_ids).to(company_tokens.dtype)
        )
        owner_onehot = company_raw[:, :, owner_start:owner_stop]
        owner_ref_table = torch.cat(
            [
                self.corp_id_embed.weight,
                self.player_id_embed.weight,
                self.type_embeds[_TYPE_FI].unsqueeze(0),
            ],
            dim=0,
        )
        owner_refs = owner_onehot.to(owner_ref_table.dtype) @ owner_ref_table
        return company_tokens + owner_refs.to(company_tokens.dtype)

    def _project_corp_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project corp tokens, adding row-order ID and relational references."""
        corp_raw = x[
            :,
            self._corp_slice,
            :int(TokenWidth.TW_CORP),
        ]
        corp_tokens = self.corp_proj(corp_raw[:, :, :self._corp_rel_offset])
        corp_tokens = (
            corp_tokens
            + self.corp_id_embed(self._corp_ids).to(corp_tokens.dtype)
        )

        president_start = self._corp_president_offset
        president_stop = president_start + self._corp_president_width
        president_onehot = corp_raw[:, :, president_start:president_stop]
        president_refs = (
            president_onehot.to(self.player_id_embed.weight.dtype)
            @ self.player_id_embed.weight
        )

        companies_start = self._corp_companies_offset
        companies_stop = companies_start + self._corp_companies_width
        owned_company_bitmap = corp_raw[:, :, companies_start:companies_stop]
        owned_company_refs = (
            owned_company_bitmap.to(self.company_id_embed.weight.dtype)
            @ self.company_id_embed.weight
        )
        owned_company_count = owned_company_bitmap.sum(dim=-1, keepdim=True).clamp_min(1.0)
        owned_company_refs = owned_company_refs / owned_company_count.sqrt().to(
            owned_company_refs.dtype
        )

        return (
            corp_tokens
            + president_refs.to(corp_tokens.dtype)
            + owned_company_refs.to(corp_tokens.dtype)
        )

    def _project_fi_token(self, x: torch.Tensor) -> torch.Tensor:
        """Project the FI token, adding owned-company references."""
        fi_raw = x[
            :,
            self._fi_idx,
            :int(TokenWidth.TW_FI),
        ]
        fi_token = self.fi_proj(fi_raw[:, :self._fi_companies_offset])
        companies_start = self._fi_companies_offset
        companies_stop = companies_start + self._fi_companies_width
        owned_company_bitmap = fi_raw[:, companies_start:companies_stop]
        owned_company_refs = (
            owned_company_bitmap.to(self.company_id_embed.weight.dtype)
            @ self.company_id_embed.weight
        )
        return fi_token + owned_company_refs.to(fi_token.dtype)

    def _project_global_info_token(self, x: torch.Tensor) -> torch.Tensor:
        """Project global info, excluding the phase one-hot prefix."""
        return self.global_info_proj(
            x[
                :,
                self._global_info_idx,
                self._global_phase_width:int(TokenWidth.TW_GLOBAL_INFO),
            ]
        )

    def _project_player_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project player tokens, adding row-order ID and relational references."""
        player_raw = x[
            :,
            self._player_slice,
            :int(TokenWidth.TW_PLAYER),
        ]
        player_tokens = self.player_proj(player_raw[:, :, :self._player_rel_offset])
        player_tokens = (
            player_tokens
            + self.player_id_embed(self._player_ids).to(player_tokens.dtype)
        )

        shares_start = self._player_shares_offset
        shares_stop = shares_start + self._player_shares_width
        owned_shares = player_raw[:, :, shares_start:shares_stop]
        share_refs = (
            owned_shares.to(self.corp_id_embed.weight.dtype)
            @ self.corp_id_embed.weight
        )

        companies_start = self._player_companies_offset
        companies_stop = companies_start + self._player_companies_width
        owned_company_bitmap = player_raw[:, :, companies_start:companies_stop]
        owned_company_refs = (
            owned_company_bitmap.to(self.company_id_embed.weight.dtype)
            @ self.company_id_embed.weight
        )

        return (
            player_tokens
            + share_refs.to(player_tokens.dtype)
            + owned_company_refs.to(player_tokens.dtype)
        )

    def _active_entity_refs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return active player/corp/company identity refs from is_selected slots."""
        active_player_ref = (
            x[:, self._player_slice, 0].to(self.player_id_embed.weight.dtype)
            @ self.player_id_embed(self._player_ids)
        )
        active_corp_ref = (
            x[:, self._corp_slice, 0].to(self.corp_id_embed.weight.dtype)
            @ self.corp_id_embed.weight
        )
        active_company_ref = (
            x[:, self._company_slice, 0].to(self.company_id_embed.weight.dtype)
            @ self.company_id_embed.weight
        )
        return active_player_ref, active_corp_ref, active_company_ref

    def _add_active_entity_refs(
        self,
        tokens: torch.Tensor,
        active_player_ref: torch.Tensor,
        active_corp_ref: torch.Tensor,
        active_company_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast active entity refs to phase/query tokens and pass anchors."""
        dtype = tokens.dtype
        active_ref = (
            active_player_ref.to(dtype)
            + active_corp_ref.to(dtype)
            + active_company_ref.to(dtype)
        )
        return tokens + (
            active_ref[:, None, :]
            * self._active_ref_targets.to(dtype=dtype).view(1, -1, 1)
        )

    def _phase_ref(self, x: torch.Tensor) -> torch.Tensor:
        """Return the current decision-phase ref from GlobalInfo's phase one-hot."""
        phase_onehot = x[
            :,
            self._global_info_idx,
            :self._global_phase_width,
        ]
        return phase_onehot.to(self.phase_embed.weight.dtype) @ self.phase_embed.weight

    def _add_phase_ref(self, tokens: torch.Tensor, phase_ref: torch.Tensor) -> torch.Tensor:
        """Broadcast the phase ref to every token except GlobalInfo."""
        dtype = tokens.dtype
        return tokens + (
            phase_ref.to(dtype)[:, None, :]
            * self._phase_ref_targets.to(dtype=dtype).view(1, -1, 1)
        )

    @staticmethod
    def _policy_head_width(head: nn.Module) -> int:
        if not isinstance(head, nn.Sequential) or len(head) == 0:
            raise AssertionError(
                f"policy head must be a non-empty Sequential, got {type(head)!r}"
            )
        last = head[-1]
        if not isinstance(last, nn.Linear):
            raise AssertionError(f"policy head must end in Linear, got {type(last)!r}")
        return int(last.out_features)

    def _validate_policy_layout(self) -> None:
        """Validate policy head widths against the shared action-size table.

        The unified output is manually concatenated in DecisionPhase order.
        This guard catches action-space edits that update ``core.data`` but
        forget to adjust the corresponding model head or block layout.
        """
        if len(PHASE_ACTION_SIZES) != NUM_PHASES:
            raise AssertionError(
                f"PHASE_ACTION_SIZES has {len(PHASE_ACTION_SIZES)} entries, "
                f"expected {NUM_PHASES}"
            )
        if int(AUCTION_CAP) != _phase_action_size(DecisionPhase.DPHASE_BID) - 1:
            raise AssertionError(
                f"AUCTION_CAP {int(AUCTION_CAP)} does not match BID raise width "
                f"{_phase_action_size(DecisionPhase.DPHASE_BID) - 1}"
            )

        company_start = self._company_slice.start
        company_stop = self._company_slice.stop
        corp_start = self._corp_slice.start
        corp_stop = self._corp_slice.stop
        if (
            company_start is None or company_stop is None
            or corp_start is None or corp_stop is None
        ):
            raise AssertionError(
                f"entity slices must be bounded; companies={self._company_slice}, "
                f"corps={self._corp_slice}"
            )
        num_companies = company_stop - company_start
        num_corps = corp_stop - corp_start
        if num_companies <= 0 or num_corps <= 0:
            raise AssertionError(
                f"invalid entity slices: companies={num_companies}, corps={num_corps}"
            )

        pass_widths = [
            self._policy_head_width(head) for head in self.pass_heads
        ]
        if len(pass_widths) != NUM_PASS_PHASES or any(w != 1 for w in pass_widths):
            raise AssertionError(
                f"pass heads must be {NUM_PASS_PHASES} single-logit heads; "
                f"got widths {pass_widths}"
            )

        block_widths = [0] * NUM_PHASES
        block_widths[int(DecisionPhase.DPHASE_INVEST)] = (
            pass_widths[0]
            + num_companies * self._policy_head_width(self.invest_company_select_head)
            + num_corps * self._policy_head_width(self.corp_trade_head)
        )
        block_widths[int(DecisionPhase.DPHASE_BID)] = self._policy_head_width(
            self.bid_head
        )
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_CORP)] = (
            pass_widths[1]
            + num_corps * self._policy_head_width(self.acq_select_corp_head)
        )
        block_widths[int(DecisionPhase.DPHASE_ACQ_OFFER)] = self._policy_head_width(
            self.acq_offer_head
        )
        block_widths[int(DecisionPhase.DPHASE_CLOSING)] = (
            pass_widths[2]
            + num_companies * self._policy_head_width(self.closing_company_select_head)
        )
        block_widths[int(DecisionPhase.DPHASE_DIVIDENDS)] = self._policy_head_width(
            self.dividend_head
        )
        block_widths[int(DecisionPhase.DPHASE_ISSUE)] = self._policy_head_width(
            self.issue_head
        )
        block_widths[int(DecisionPhase.DPHASE_IPO)] = (
            pass_widths[3]
            + num_corps * self._policy_head_width(self.ipo_corp_select_head)
        )
        block_widths[int(DecisionPhase.DPHASE_PAR)] = self._policy_head_width(
            self.par_price_head
        )
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_COMPANY)] = (
            num_companies * self._policy_head_width(self.acq_select_company_head)
        )
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)] = (
            self._policy_head_width(self.price_acq_head)
        )

        expected = [int(size) for size in PHASE_ACTION_SIZES]
        if block_widths != expected:
            raise AssertionError(
                f"policy block widths {block_widths} do not match "
                f"PHASE_ACTION_SIZES {expected}"
            )
        if sum(block_widths) != UNIFIED_LOGIT_DIM:
            raise AssertionError(
                f"policy block total {sum(block_widths)} != UNIFIED_LOGIT_DIM "
                f"{UNIFIED_LOGIT_DIM}"
            )

    # ------------------------------------------------------------------
    # Input projection
    # ------------------------------------------------------------------

    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw token features to d_model via type-specific projections.

        The pass anchors are concatenated here (not present in the input
        buffer), so the output sequence is ``NUM_PASS_PHASES`` wider than the
        input.

        Most rows are sliced to their projection's ``in_features`` width
        before projecting. Company, corp, and player identities are inferred
        from row order and added as learned additive identity embeddings.
        Company rows skip the ownership tail and receive the matching corp /
        player / FI additive owner reference directly. Corp rows skip the
        president / owned-company tail and receive additive player / company
        references directly. FI skips its owned-company tail and receives
        additive company references directly. Player rows skip their owned
        shares / owned-company tail and receive additive corp / company
        references directly. The engine-side buffer is rectangularly
        zero-padded to ``TOKEN_DIM=85`` so ``get_token_data`` can fill it with
        a uniform nogil memcpy pattern, but each type only uses its meaningful
        feature slice — padding and replaced relation fields would otherwise be
        multiplied against projection weights. After pass anchors are appended,
        active player/corp/company identity refs are broadcast only to phase/
        query tokens and learned pass anchors, leaving MarketInfo, GlobalInfo,
        FI, and entity rows unmodified by active refs. The decision-phase
        one-hot is also sliced out of GlobalInfo and broadcast as a learned
        phase ref to every token except GlobalInfo itself.

        Args:
            x: (batch, num_players + 55, token_dim) zero-padded raw features.
                Supported caller patterns are: fp16/fp32 under autocast on the
                eval path, or fp32 matching the projection weights on the
                non-autocast trainer / CPU paths. No explicit upcast happens
                here: an unconditional ``x.to(bf16)`` would mismatch fp32
                Linear weights in non-autocast paths (for example in-process
                NNEvaluator tests).
        Returns:
            (batch, num_players + 55 + NUM_PASS_PHASES, d_model) embeddings:
            projected input tokens followed by the per-phase pass anchors.
        """
        # Pass anchors: (NUM_PASS_PHASES, d) → (B, NUM_PASS_PHASES, d) without
        # mixing SymInt with ``-1`` in the expand target. Under AOT autograd,
        # ``.expand(B, NUM_PASS_PHASES, -1)`` where ``B`` is ``x.shape[0]``
        # sometimes concretizes the batch size into a static guard and forces
        # per-batch-size recompiles; feeding the last dim explicitly lets the
        # symbolic-shape tracker carry ``B`` cleanly through.
        d = self.cfg.d_model
        company_tokens = self._project_company_tokens(x)
        corp_tokens = self._project_corp_tokens(x)
        input_parts: list[torch.Tensor] = [
            _slice_proj(x, self.market_info_proj, self._market_info_idx).unsqueeze(1),
            company_tokens,                                                             # (B, 36, d)
            self._project_fi_token(x).unsqueeze(1),
            self._project_global_info_token(x).unsqueeze(1),
            _slice_proj(x, self.invest_proj, self._invest_idx).unsqueeze(1),
            _slice_proj(x, self.auction_proj, self._auction_idx).unsqueeze(1),
            _slice_proj(x, self.dividend_proj, self._dividend_idx).unsqueeze(1),
            _slice_proj(x, self.issue_proj, self._issue_idx).unsqueeze(1),
            _slice_proj(x, self.par_proj, self._par_idx).unsqueeze(1),
            _slice_proj(x, self.acq_select_company_proj, self._acq_select_company_idx).unsqueeze(1),
            _slice_proj(x, self.acq_offer_proj, self._acq_offer_idx).unsqueeze(1),
            _slice_proj(x, self.acq_price_proj, self._acq_price_info_idx).unsqueeze(1),
            corp_tokens,                                                                # (B, 8, d)
            self._project_player_tokens(x),                                              # (B, N, d)
        ]
        # Additive per-type embedding broadcast over the batch. A single
        # indexed gather against ``type_embeds`` gives every non-pass token a
        # type-distinct signal even when its feature slice is all-zero.
        input_tokens = torch.cat(input_parts, dim=1)                                     # (B, num_players+55, d)
        input_tokens = input_tokens + self.type_embeds[self._type_ids].to(input_tokens.dtype)
        pass_rows = (
            self.pass_embeds.to(dtype=input_tokens.dtype)
            .view(1, NUM_PASS_PHASES, d)
            .expand(x.shape[0], NUM_PASS_PHASES, d)
        )
        tokens = torch.cat([input_tokens, pass_rows], dim=1)                             # (B, num_tokens, d)
        tokens = self._add_phase_ref(tokens, self._phase_ref(x))
        active_player_ref, active_corp_ref, active_company_ref = self._active_entity_refs(x)
        tokens = self._add_active_entity_refs(
            tokens,
            active_player_ref,
            active_corp_ref,
            active_company_ref,
        )
        return tokens

    # ------------------------------------------------------------------
    # Unified policy: every head runs once on the full batch
    # ------------------------------------------------------------------

    def _build_unified_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run every per-phase policy head once on the full batch and concat
        into a single ``(B, UNIFIED_LOGIT_DIM)`` tensor.

        Blocks are emitted in DecisionPhase order (matching the offsets baked
        into ``build_action_lut``). Entity-readout phases with pass actions
        lead with a per-phase pass-anchor logit. Direct-token phases (BID,
        ACQ_OFFER, ISSUE) emit the whole phase block from their info token,
        including phase-local action 0. Heads run unconditionally regardless
        of which phase a given row is in: the caller's legal mask zeroes out
        slots outside the current phase's action space. Total wasted FLOPs
        are a small fraction of the trunk at d_model=128.
        """
        # Shared token reads. Multi-token entity heads stay (B, n_tokens, …)
        # and get squeezed / flattened at use site.
        company_tokens = tokens[:, self._company_slice]                          # (B, 36, d)
        corp_tokens = tokens[:, self._corp_slice]                                # (B, 8, d)

        # Pass anchors. ``self._pass_idxs[i]`` backs ``PASS_PHASE_IDS[i]``.
        pass_invest = self.pass_heads[0](tokens[:, self._pass_idxs[0]])          # (B, 1)
        pass_acq_select_corp = self.pass_heads[1](tokens[:, self._pass_idxs[1]]) # (B, 1)
        pass_closing = self.pass_heads[2](tokens[:, self._pass_idxs[2]])         # (B, 1)
        pass_ipo = self.pass_heads[3](tokens[:, self._pass_idxs[3]])             # (B, 1)

        # INVEST: pass + 36 company-select + 16 corp-trade (2i buy, 2i+1 sell).
        invest_company = self.invest_company_select_head(company_tokens).squeeze(-1)   # (B, 36)
        corp_trade = self.corp_trade_head(corp_tokens).flatten(1)                # (B, 16)
        # BID: pass + AUCTION_CAP raise offsets.
        bid = self.bid_head(tokens[:, self._auction_idx])                        # (B, 16)
        # ACQ_SELECT_CORP: pass + 8 corps.
        acq_select_corp = self.acq_select_corp_head(corp_tokens).squeeze(-1)     # (B, 8)
        # ACQ_OFFER: pass + 1 accept-buy.
        acq_offer = self.acq_offer_head(tokens[:, self._acq_offer_idx])          # (B, 2)
        # CLOSING: pass + 36 company-close.
        closing_company = self.closing_company_select_head(company_tokens).squeeze(-1)  # (B, 36)
        # DIVIDENDS: 26 levels (no pass).
        dividend = self.dividend_head(tokens[:, self._dividend_idx])             # (B, 26)
        # ISSUE: pass + 1 issue.
        issue = self.issue_head(tokens[:, self._issue_idx])                      # (B, 2)
        # IPO: pass + 8 corps.
        ipo_corp = self.ipo_corp_select_head(corp_tokens).squeeze(-1)            # (B, 8)
        # PAR: 14 par indices (no pass).
        par_price = self.par_price_head(tokens[:, self._par_idx])                # (B, 14)
        # ACQ_SELECT_COMPANY: 36 companies (no pass).
        acq_select_company = self.acq_select_company_head(company_tokens).squeeze(-1)  # (B, 36)
        # ACQ_SELECT_PRICE: 51 price offsets (no pass).
        price_acq = self.price_acq_head(tokens[:, self._acq_price_info_idx])     # (B, 51)

        return torch.cat(
            [
                pass_invest, invest_company, corp_trade,           # INVEST           (53)
                bid,                                               # BID              (16)
                pass_acq_select_corp, acq_select_corp,             # ACQ_SELECT_CORP  ( 9)
                acq_offer,                                         # ACQ_OFFER        ( 2)
                pass_closing, closing_company,                     # CLOSING          (37)
                dividend,                                          # DIVIDENDS        (26)
                issue,                                             # ISSUE            ( 2)
                pass_ipo, ipo_corp,                                # IPO              ( 9)
                par_price,                                         # PAR              (14)
                acq_select_company,                                # ACQ_SELECT_CO.   (36)
                price_acq,                                         # ACQ_SELECT_PRICE (51)
            ],
            dim=-1,
        )                                                          # (B, UNIFIED_LOGIT_DIM=255)

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

        # Pass anchors: small-random learned per-phase vectors (BERT [CLS] convention).
        nn.init.trunc_normal_(self.pass_embeds, std=0.02)
        nn.init.trunc_normal_(self.company_id_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.corp_id_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.player_id_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.phase_embed.weight, std=0.02)
        # Per-type additive embeddings: same small-random init.
        nn.init.trunc_normal_(self.type_embeds, std=0.02)

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
    company_id_params = model.company_id_embed.weight.numel()
    corp_id_params = model.corp_id_embed.weight.numel()
    player_id_params = model.player_id_embed.weight.numel()
    phase_params = model.phase_embed.weight.numel()
    pass_params = model.pass_embeds.numel()
    type_params = model.type_embeds.numel()
    trunk_params = (
        sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.final_norm.parameters())
    )
    policy_modules: list[nn.Module] = [
        model.pass_heads,
        model.invest_company_select_head, model.closing_company_select_head,
        model.acq_select_company_head,
        model.acq_select_corp_head, model.ipo_corp_select_head,
        model.corp_trade_head, model.bid_head, model.dividend_head,
        model.issue_head, model.acq_offer_head,
        model.price_acq_head, model.par_price_head,
    ]
    policy_params = sum(sum(p.numel() for p in m.parameters()) for m in policy_modules)
    value_params = sum(p.numel() for p in model.value_head.parameters())

    print("Parameter breakdown:")
    for name, count in [
        ("Input projections", proj_params),
        ("Company ID embeds", company_id_params),
        ("Corp ID embeds", corp_id_params),
        ("Player ID embeds", player_id_params),
        ("Phase embeds", phase_params),
        ("Pass anchors", pass_params),
        ("Type embeds", type_params),
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
