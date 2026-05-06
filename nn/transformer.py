"""Transformer model for Rolling Stock Stars AlphaZero training.

Token-based architecture: each game entity is a separate input token. Type-specific
linear projections feed a pre-LN transformer trunk, actor-conditioned policy
readouts, and a value head.

Key differences from the MLP model (nn/template.py):
  - Input: (batch, num_tokens, token_dim) token features, not flat state vector
  - No state rotation: active player marked with is_active flag
  - Actor-conditioned policy: actor queries score entity/action keys
  - ACQ factored into three single-entity sub-phases (corp/company/price)
  - Unified policy output: every readout writes into a static
    (B, UNIFIED_LOGIT_DIM) tensor, with illegal slots masked by caller input
  - Value read from player tokens directly (no un-rotation needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
)
from core.data import (
    ALL_PAR_PRICES,
    AUCTION_CAP,
    GameConstants,
    PHASE_ACTION_SIZES,
    DecisionPhase,
    PY_COMPANY_PRICE_DIVISOR,
)
from core.token_data import TokenDataSize, TokenWidth, get_num_tokens, get_token_widths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Decision phases / action sizes all live in ``core.data`` and are imported
# above. This module is strictly a consumer; editing policy readout widths or
# adding token types happens over there.

NUM_PHASES = len(DecisionPhase)
NUM_FIXED_TOKENS = 54
MAX_PHASE_ACTION_SIZE = max(PHASE_ACTION_SIZES)  # 53 (INVEST: pass + 36 + 16)

# Unified policy slot layout. Every per-phase readout emits its logits
# into a single (B, UNIFIED_LOGIT_DIM) tensor; callers pass a matching
# (B, UNIFIED_LOGIT_DIM) legal-mask so illegal slots can be suppressed before
# softmax. Each DecisionPhase owns a contiguous, non-overlapping block —
# laid out in DecisionPhase order with block widths from PHASE_ACTION_SIZES.
# Readout weights are unshared across phases (no cross-phase gradient
# interference on shared readouts), so the LUT reduces to
# ``lut[phase, i] = cumsum_offset[phase] + i``. ``build_action_lut`` is the
# sole public pointer into the layout — callers never import per-block
# offsets, which keeps this internal.
_PHASE_OFFSETS: list[int] = [0]
for _size in PHASE_ACTION_SIZES:
    _PHASE_OFFSETS.append(_PHASE_OFFSETS[-1] + int(_size))
UNIFIED_LOGIT_DIM = _PHASE_OFFSETS[-1]  # 255

# Decision phases whose pass action is emitted by a separate ``Linear(d, 1)``
# head rather than as action_id 0 of an action-key MLP. ACQ_OFFER ("reject")
# and ISSUE ("no issue") put the no-op at slot 0 of their action_embed and
# score it through the same MLP as the other action, so they are not in this
# list. Used for the pass-vs-action logit-scale TB diagnostic.
PHASES_WITH_PASS_HEAD: list[int] = [
    int(DecisionPhase.DPHASE_INVEST),
    int(DecisionPhase.DPHASE_BID),
    int(DecisionPhase.DPHASE_ACQ_SELECT_CORP),
    int(DecisionPhase.DPHASE_CLOSING),
    int(DecisionPhase.DPHASE_IPO),
]

# Input-buffer token-type taxonomy. Index order must stay stable — the static
# ``_type_ids`` buffer built in ``RSSTransformerNet.__init__`` indexes into
# ``type_embeds`` using these ids, and multi-instance types (company / corp
# / player) share a single row.
_TYPE_MARKET_INFO = 0
_TYPE_COMPANY = 1
_TYPE_FI = 2
_TYPE_GLOBAL_INFO = 3
_TYPE_INVEST = 4
_TYPE_AUCTION = 5
_TYPE_DIVIDEND = 6
_TYPE_ISSUE = 7
_TYPE_PAR = 8
_TYPE_ACQ_OFFER = 9
_TYPE_ACQ_PRICE = 10
_TYPE_CORP = 11
_TYPE_PLAYER = 12
NUM_TOKEN_TYPES = _TYPE_PLAYER + 1

_GELU_APPROX = "tanh"
_NUM_CORPS = int(GameConstants.NUM_CORPS)
_TOKEN_FEATURE_START = 1
_IS_SELECTED_OFFSET = 1
_GLOBAL_PHASE_OFFSET = 1
_GLOBAL_PHASE_STOP = _GLOBAL_PHASE_OFFSET + NUM_PHASES
# Offsets inside a company feature slice after the attention-mask slot is dropped.
_COMPANY_LOW_PRICE_FEATURE_OFFSET = 1
_COMPANY_FACE_VALUE_FEATURE_OFFSET = 2
# Raw token offsets where relation/reference tails begin. The engine still
# emits these fields for compatibility and diagnostics, but the model ignores
# them in token projection now that relation matrices feed attention directly.
# Aggregate "relational summary" scalars (owned-company counts, presidency
# count, total shares) sit immediately before each rel-tail start, so they
# stay in the projection while the multihots they summarize are dropped.
_COMPANY_REL_TAIL_START = 14
_FI_REL_TAIL_START = 4
_CORP_REL_TAIL_START = 54
# Player share amounts are scalar quantities, not just relation presence:
# keep OFF_SHARES (8 slots) in projected token features and drop only the
# owned-company relation tail.
_PLAYER_REL_TAIL_START = 26


def _phase_action_size(phase: DecisionPhase) -> int:
    return int(PHASE_ACTION_SIZES[int(phase)])


def _fourier_feature_width(input_width: int, num_bands: int) -> int:
    """Width after appending fixed sin/cos bands to raw scalar features."""
    return input_width * (1 + 2 * num_bands)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round positive integer ``value`` up to the next ``multiple`` boundary."""
    if value < 1:
        raise ValueError(f"value must be positive, got {value}")
    if multiple < 1:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


def _ffn_hidden_dim(cfg: TransformerConfig) -> int:
    """SwiGLU hidden width, rounded for tensor-core-friendly matmuls."""
    return _round_up_to_multiple(math.ceil(cfg.ff_mult * cfg.d_model), 64)


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
    """Drop the leading attention-mask slot, then project ``proj.in_features``.

    Output shape depends on ``idx``: int → ``(B, d)``, slice → ``(B, n, d)``.
    Int call sites add a trailing ``.unsqueeze(1)`` themselves — keeping the
    per-site fixup explicit avoids a Python-side isinstance branch that
    Dynamo would have to trace through.

    Module-level rather than a closure or staticmethod so Dynamo can inline
    it without guarding on a fresh function id per call or routing through
    class-attribute lookup.
    """
    start = _TOKEN_FEATURE_START
    stop = start + proj.in_features
    return proj(x[:, idx, start:stop])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """All dimensions parameterized. Defaults are 3-player with d_model=256."""

    # Core architecture
    # Model player-token capacity. In mixed training this is effective
    # max_players; each state still encodes its actual player count.
    num_players: int = 3  # 3-5 supported
    d_model: int = 256
    d_proj: int = 64
    num_heads: int = 4
    num_layers: int = 15
    ff_mult: float = 3.0  # FFN inner dimension is rounded up to a multiple of 64.
    phase_conditioning: bool = False
    price_slot_fourier_bands: int = 4
    # Blend weight between projected Fourier slot features and learned per-slot
    # embeddings: 0.0 = pure Fourier projection, 1.0 = pure embedding.
    price_slot_residual_scale: float = 1.0

    # Raw feature width per token (zero-padded to same size across types).
    # Sourced from core.token_data so the model and the Cython extractor
    # can't drift out of sync.
    token_dim: int = int(TokenDataSize.TOKEN_DIM)

    _num_tokens: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert 3 <= self.num_players <= 5, f"num_players must be 3-5, got {self.num_players}"
        assert self.d_model > 0, f"d_model must be positive, got {self.d_model}"
        assert self.d_proj > 0, f"d_proj must be positive, got {self.d_proj}"
        assert self.num_heads > 0, f"num_heads must be positive, got {self.num_heads}"
        assert self.num_layers > 0, f"num_layers must be positive, got {self.num_layers}"
        assert self.ff_mult > 0, f"ff_mult must be positive, got {self.ff_mult}"
        assert self.d_model % self.num_heads == 0, (
            f"d_model {self.d_model} must be divisible by num_heads {self.num_heads}"
        )
        assert isinstance(self.phase_conditioning, bool), (
            f"phase_conditioning must be bool, got {self.phase_conditioning!r}"
        )
        assert self.price_slot_fourier_bands >= 0, (
            "price_slot_fourier_bands must be >= 0, "
            f"got {self.price_slot_fourier_bands}"
        )
        assert 0.0 <= self.price_slot_residual_scale <= 1.0, (
            "price_slot_residual_scale must be in [0, 1], "
            f"got {self.price_slot_residual_scale}"
        )
        object.__setattr__(self, "_num_tokens", int(get_num_tokens(self.num_players)))

    @property
    def num_tokens(self) -> int:
        """Input-buffer token count: fixed entity/phase tokens + player capacity."""
        return self._num_tokens


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

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        phase_conditioning: bool = False,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model {d_model} must be divisible by num_heads {num_heads}"
        )
        self.phase_conditioning = phase_conditioning
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_norm = nn.RMSNorm(d_model)
        # Packed Q/K/V projection. The trunk follows the modern RMSNorm +
        # SwiGLU convention: projection matrices are biasless, while explicit
        # conditioning/bias mechanisms live in phase_mod and relation_bias_mult.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        # Direct per-phase adaptive RMSNorm + residual-gate parameters.
        # Zero-initialized in RSSTransformerNet._init_weights, so scale/shift
        # starts inert and residual gates start closed. This gives the
        # adaLN-Zero identity start without a phase-independent bias shortcut.
        self.phase_mod: nn.Embedding | None = (
            nn.Embedding(NUM_PHASES, 6 * d_model)
            if phase_conditioning
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        relation_bias: torch.Tensor,
        phase_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.attn_norm(x)
        B, N, D = h.shape
        phase_mod = self.phase_mod
        attn_resid_gate: torch.Tensor | None = None
        ffn_scale: torch.Tensor | None = None
        ffn_shift: torch.Tensor | None = None
        ffn_resid_gate: torch.Tensor | None = None
        if phase_mod is not None:
            if phase_ids is None:
                raise AssertionError(
                    "phase_ids are required when phase_conditioning is enabled"
                )
            mod = phase_mod(phase_ids).to(device=h.device, dtype=h.dtype)
            (
                attn_scale,
                attn_shift,
                attn_resid_gate,
                ffn_scale,
                ffn_shift,
                ffn_resid_gate,
            ) = mod.chunk(6, dim=-1)
            h = h * (1.0 + attn_scale[:, None, :]) + attn_shift[:, None, :]
        qkv = self.qkv_proj(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        # (3, B, heads, N, head_dim) so unbind(0) yields three
        # (B, heads, N, head_dim) tensors. The contiguous call gives Inductor's
        # SDPA autotune templates the packed layout they expect instead of the
        # batch/sequence-strided view produced directly by permute.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).contiguous().unbind(0)
        hidden = torch.finfo(relation_bias.dtype).min
        visibility_bias = torch.zeros_like(attn_mask, dtype=relation_bias.dtype)
        visibility_bias = visibility_bias.masked_fill(~attn_mask, hidden)
        sdpa_mask = relation_bias + visibility_bias
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        # (B, heads, N, head_dim) -> (B, N, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        h = self.out_proj(attn_out)
        if attn_resid_gate is not None:
            x = x + attn_resid_gate[:, None, :] * h
        else:
            x = x + h
        h = self.ffn_norm(x)
        if ffn_scale is not None and ffn_shift is not None:
            h = h * (1.0 + ffn_scale[:, None, :]) + ffn_shift[:, None, :]
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        if ffn_resid_gate is not None:
            return x + ffn_resid_gate[:, None, :] * h
        return x + h


@dataclass(frozen=True)
class _PolicyContext:
    raw_tokens: torch.Tensor
    tokens: torch.Tensor
    company_tokens: torch.Tensor
    corp_tokens: torch.Tensor
    active_player: torch.Tensor
    active_corp: torch.Tensor
    active_company: torch.Tensor


@dataclass(frozen=True)
class _SparseRelationContext:
    relation_ids: torch.Tensor
    flat_indices: torch.Tensor
    valid_edges: torch.Tensor


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RSSTransformerNet(nn.Module):
    """Transformer with actor-conditioned policy readouts and per-player values."""

    # Class-level annotation so pyright knows ``self._type_ids`` (registered
    # as a buffer in ``__init__``) is a Tensor. ``register_buffer`` otherwise
    # returns ``Tensor | Module | None`` per pytorch's stubs, which breaks
    # ``type_embeds(self._type_ids)`` lookups.
    _type_ids: torch.Tensor
    _corp_ids: torch.Tensor
    _bid_offset_features: torch.Tensor
    _bid_offset_dollar_norm: torch.Tensor
    _dividend_amount_features: torch.Tensor
    _par_price_features: torch.Tensor
    _acq_price_offset_features: torch.Tensor
    _issue_action_features: torch.Tensor
    _acq_offer_action_features: torch.Tensor
    _fourier_band_freqs: torch.Tensor

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
        #     acq_offer, acq_price_info
        #   corps×8, then players×N (trailing so padding for higher player
        #   counts is a no-op on the prefix).
        self._market_info_idx = 0
        self._company_slice = slice(1, 37)
        self._fi_idx = 37
        self._global_info_idx = 38
        self._invest_idx = 39
        self._auction_idx = 40
        self._dividend_idx = 41
        self._issue_idx = 42
        self._par_idx = 43
        self._acq_offer_idx = 44
        self._acq_price_info_idx = 45
        self._corp_slice = slice(46, NUM_FIXED_TOKENS)
        self._player_slice = slice(NUM_FIXED_TOKENS, NUM_FIXED_TOKENS + np_)

        # Drift guard: hardcoded positions above must match the Cython-side
        # ``get_token_widths`` layout. Checking here fires loudly at model
        # construction rather than silently feeding mis-aligned features to
        # the trunk.
        _validate_layout(np_)

        # --- Type-specific input projections ---
        # Projections drop slot 0 (the token attention mask) before feeding
        # data into Linear layers. Entity relation/reference tails are also
        # skipped here; those relations now enter the trunk as Graphormer-style
        # attention bias planes. Learned additive state is limited to type
        # embeddings and corp row-order identity embeddings.
        # The engine-side buffer is rectangular at ``TOKEN_DIM=92`` so
        # ``get_token_data`` can fill it with a single nogil memcpy pattern,
        # but each projection still sizes itself to that token type's meaningful
        # width so padding remains inert.
        # Widths are pulled from ``TokenWidth`` so the model and the Cython
        # extractor can't drift out of sync.
        self._token_feature_start = _TOKEN_FEATURE_START
        self._is_selected_offset = _IS_SELECTED_OFFSET
        self._company_rel_tail_start = _COMPANY_REL_TAIL_START
        self._fi_rel_tail_start = _FI_REL_TAIL_START
        self._corp_rel_tail_start = _CORP_REL_TAIL_START
        self._player_rel_tail_start = _PLAYER_REL_TAIL_START
        self._global_info_feature_start = (
            _GLOBAL_PHASE_STOP
            if cfg.phase_conditioning
            else self._token_feature_start
        )
        self.player_proj = nn.Linear(
            self._player_rel_tail_start - self._token_feature_start,
            d,
        )
        self.corp_proj = nn.Linear(
            self._corp_rel_tail_start - self._token_feature_start,
            d,
        )
        self.company_proj = nn.Linear(
            self._company_rel_tail_start - self._token_feature_start,
            d,
        )
        self.fi_proj = nn.Linear(
            self._fi_rel_tail_start - self._token_feature_start,
            d,
        )
        self.market_info_proj = nn.Linear(
            int(TokenWidth.TW_MARKET_INFO) - self._token_feature_start,
            d,
        )
        self.global_info_proj = nn.Linear(
            int(TokenWidth.TW_GLOBAL_INFO) - self._global_info_feature_start,
            d,
        )
        self.invest_proj = nn.Linear(
            int(TokenWidth.TW_INVEST) - self._token_feature_start,
            d,
        )
        self.auction_proj = nn.Linear(
            int(TokenWidth.TW_AUCTION) - self._token_feature_start,
            d,
        )
        self.dividend_proj = nn.Linear(
            int(TokenWidth.TW_DIVIDEND) - self._token_feature_start,
            d,
        )
        self.issue_proj = nn.Linear(
            int(TokenWidth.TW_ISSUE) - self._token_feature_start,
            d,
        )
        self.par_proj = nn.Linear(
            int(TokenWidth.TW_PAR) - self._token_feature_start,
            d,
        )
        self.acq_offer_proj = nn.Linear(
            int(TokenWidth.TW_ACQ_OFFER) - self._token_feature_start,
            d,
        )
        self.acq_price_proj = nn.Linear(
            int(TokenWidth.TW_ACQ_PRICE) - self._token_feature_start,
            d,
        )
        # Corp tokens keep a learned row-order identity embedding. Other entity
        # identity and relation fields are consumed as ordinary projected input.
        self.corp_id_embed = nn.Embedding(_NUM_CORPS, d)
        # Per-type additive embedding for every token. Added
        # post-projection in ``_project_tokens`` so the trunk still sees a
        # type-distinct vector even when a token's feature slice is all-zero
        # (e.g. the DIVIDEND context token outside DIVIDENDS, or the owned-
        # company field of a player with no companies). Without this, zero
        # features + zero-initialized Linear biases collapse the token to the
        # zero vector on day one, and the only path to type discrimination is
        # an indirect gradient through the Linear's bias. Broadcast across
        # all instances of a type (36 companies, 8 corps, N players).
        self.type_embeds = nn.Embedding(NUM_TOKEN_TYPES, d)
        # Static (num_players + NUM_FIXED_TOKENS,) type-id lookup. Built once here so
        # ``_project_tokens`` can do a single indexed gather against
        # ``type_embeds``. Registered as a buffer so ``.to(device)`` carries
        # it along. Must match the concat order inside ``_project_tokens``.
        type_ids = torch.empty(np_ + NUM_FIXED_TOKENS, dtype=torch.long)
        type_ids[self._market_info_idx] = _TYPE_MARKET_INFO
        type_ids[self._company_slice] = _TYPE_COMPANY
        type_ids[self._fi_idx] = _TYPE_FI
        type_ids[self._global_info_idx] = _TYPE_GLOBAL_INFO
        type_ids[self._invest_idx] = _TYPE_INVEST
        type_ids[self._auction_idx] = _TYPE_AUCTION
        type_ids[self._dividend_idx] = _TYPE_DIVIDEND
        type_ids[self._issue_idx] = _TYPE_ISSUE
        type_ids[self._par_idx] = _TYPE_PAR
        type_ids[self._acq_offer_idx] = _TYPE_ACQ_OFFER
        type_ids[self._acq_price_info_idx] = _TYPE_ACQ_PRICE
        type_ids[self._corp_slice] = _TYPE_CORP
        type_ids[self._player_slice] = _TYPE_PLAYER
        self.register_buffer("_type_ids", type_ids, persistent=False)

        self.register_buffer(
            "_corp_ids", torch.arange(_NUM_CORPS, dtype=torch.long), persistent=False,
        )
        bid_offset_features = (
            torch.arange(int(AUCTION_CAP), dtype=torch.float32).view(1, int(AUCTION_CAP), 1)
            / float(int(AUCTION_CAP) - 1)
        )
        self.register_buffer(
            "_bid_offset_features",
            bid_offset_features,
            persistent=False,
        )
        # Per-slot dollar offset in /COMPANY_PRICE_DIVISOR units; added to the
        # active company's normalized face_value at runtime to form the actual
        # candidate-bid price channel for the BID slot-key projection.
        bid_offset_dollar_norm = (
            torch.arange(int(AUCTION_CAP), dtype=torch.float32).view(1, int(AUCTION_CAP), 1)
            / float(PY_COMPANY_PRICE_DIVISOR)
        )
        self.register_buffer(
            "_bid_offset_dollar_norm",
            bid_offset_dollar_norm,
            persistent=False,
        )
        # Normalize by max actual amount (MAX_DIVIDEND - 1 = 25), not slot
        # count, so the feature is literally amount / max_amount and matches
        # the price-domain normalization used by BID / ACQ_PRICE / PAR.
        dividend_amount_features = (
            torch.arange(
                _phase_action_size(DecisionPhase.DPHASE_DIVIDENDS),
                dtype=torch.float32,
            ).view(1, _phase_action_size(DecisionPhase.DPHASE_DIVIDENDS), 1)
            / float(int(GameConstants.MAX_DIVIDEND) - 1)
        )
        self.register_buffer(
            "_dividend_amount_features",
            dividend_amount_features,
            persistent=False,
        )
        # Slot-identity inputs for the PAR structured projection: per-slot
        # ``[normalized_index, par_price / max_par_price]``. Both channels
        # are static (par prices are a fixed table), so we precompute the
        # buffer at init and reuse on every forward.
        num_par_prices = _phase_action_size(DecisionPhase.DPHASE_PAR)
        par_price_max = float(max(ALL_PAR_PRICES))
        par_index_norm = torch.arange(num_par_prices, dtype=torch.float32) / float(
            num_par_prices - 1
        )
        par_price_norm = torch.tensor(
            [float(p) / par_price_max for p in ALL_PAR_PRICES],
            dtype=torch.float32,
        )
        par_price_features = torch.stack([par_index_norm, par_price_norm], dim=-1)
        self.register_buffer(
            "_par_price_features",
            par_price_features,
            persistent=False,
        )
        # Two per-slot channels for ACQ_SELECT_PRICE:
        #   [0] slot_position_norm = offset / (K-1)        — 0-indexed [0,1],
        #       feeds the Fourier slot-key projection (matches BID/PAR/DIVIDENDS)
        #       AND pairs with the engine's ``max_offset = (high - low) / 50``
        #       (also 0-indexed, divisor 50 = K-1) so
        #       ``max_offset - slot_position_norm`` is a clean "options
        #       remaining" feature.
        #   [1] offset_price_delta = offset / 80           — added to the
        #       active company's normalized ``low_price`` to recover the
        #       candidate price in the same /COMPANY_PRICE_DIVISOR units.
        K = _phase_action_size(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)
        acq_price_offsets = torch.arange(K, dtype=torch.float32).view(1, K, 1)
        acq_price_offset_features = torch.cat(
            [
                acq_price_offsets / float(K - 1),
                acq_price_offsets / float(PY_COMPANY_PRICE_DIVISOR),
            ],
            dim=-1,
        )
        self.register_buffer(
            "_acq_price_offset_features",
            acq_price_offset_features,
            persistent=False,
        )
        issue_action_features = torch.arange(
            _phase_action_size(DecisionPhase.DPHASE_ISSUE),
            dtype=torch.float32,
        ).view(1, _phase_action_size(DecisionPhase.DPHASE_ISSUE), 1)
        self.register_buffer(
            "_issue_action_features",
            issue_action_features,
            persistent=False,
        )
        acq_offer_action_features = torch.arange(
            _phase_action_size(DecisionPhase.DPHASE_ACQ_OFFER),
            dtype=torch.float32,
        ).view(1, _phase_action_size(DecisionPhase.DPHASE_ACQ_OFFER), 1)
        self.register_buffer(
            "_acq_offer_action_features",
            acq_offer_action_features,
            persistent=False,
        )
        # Pre-multiplied Fourier band frequencies for price-slot identity:
        # ``2^k * 2π`` for k in [0, price_slot_fourier_bands). Empty when bands=0.
        fourier_band_freqs = torch.pow(
            torch.tensor(2.0),
            torch.arange(cfg.price_slot_fourier_bands, dtype=torch.float32),
        ) * (2.0 * math.pi)
        self.register_buffer(
            "_fourier_band_freqs",
            fourier_band_freqs,
            persistent=False,
        )

        # --- Transformer trunk ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d,
                cfg.num_heads,
                _ffn_hidden_dim(cfg),
                phase_conditioning=cfg.phase_conditioning,
            )
            for _ in range(cfg.num_layers)
        ])
        self.relation_bias_mult = nn.Parameter(torch.zeros(
            cfg.num_layers,
            cfg.num_heads,
            NUM_ATTENTION_RELATIONS,
        ))
        self.final_norm = nn.RMSNorm(d)

        # --- Actor-conditioned policy readouts ---
        # Invest is actor-conditioned: one active-player query is scored
        # against phase/action-specific company/corp key projections.
        # The output layout stays pass, 36 companies, then interleaved
        # buy/sell logits for the 8 corps.
        dp = cfg.d_proj
        slot_fourier_1 = _fourier_feature_width(1, cfg.price_slot_fourier_bands)
        slot_fourier_2 = _fourier_feature_width(2, cfg.price_slot_fourier_bands)
        self.invest_pass_head = nn.Linear(d, 1)
        self.invest_actor_proj = nn.Linear(d, dp, bias=False)
        self.invest_auction_company_proj = nn.Linear(d, dp, bias=False)
        self.invest_buy_corp_proj = nn.Linear(d, dp, bias=False)
        self.invest_sell_corp_proj = nn.Linear(d, dp, bias=False)

        self.closing_pass_head = nn.Linear(d, 1)
        self.closing_actor_proj = nn.Linear(d, dp, bias=False)
        self.closing_company_proj = nn.Linear(d, dp, bias=False)
        self.acq_select_company_actor_proj = nn.Linear(d, dp, bias=False)
        self.acq_select_company_company_proj = nn.Linear(d, dp, bias=False)
        self.acq_select_corp_pass_head = nn.Linear(d, 1)
        self.acq_select_corp_actor_proj = nn.Linear(d, dp, bias=False)
        self.acq_select_corp_corp_proj = nn.Linear(d, dp, bias=False)
        # IPO is actor-conditioned: active-player query, active-company/PAR
        # context, and one generated key per candidate corp.
        self.ipo_pass_head = nn.Linear(d, 1)
        self.ipo_actor_proj = nn.Linear(d, dp, bias=False)
        self.ipo_corp_proj = nn.Linear(d, dp, bias=False)
        self.ipo_context_proj = nn.Linear(2 * d, dp)
        self.ipo_key_mlp = nn.Sequential(
            nn.Linear(2 * dp, dp),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(dp, dp),
        )

        # BID is actor-conditioned with a direct active-player pass readout
        # plus generated bid-offset keys conditioned on auction mechanics and
        # the active company being auctioned. Slot identity is a Fourier
        # projection of normalized offset + candidate-bid price (face_value +
        # offset, /80), blended with a learned per-offset embedding.
        self.bid_pass_head = nn.Linear(d, 1)
        self.bid_actor_proj = nn.Linear(d, dp, bias=False)
        self.bid_info_proj = nn.Linear(2 * d, dp)
        self.bid_offset_proj = nn.Linear(slot_fourier_2, dp, bias=False)
        self.bid_offset_embed = nn.Embedding(int(AUCTION_CAP), dp)
        self.bid_key_mlp = self._make_action_key_mlp(action_feature_width=3)

        # Generated action-key heads for phase-info-token decisions.
        self.dividend_actor_proj = nn.Linear(d, dp, bias=False)
        self.dividend_info_proj = nn.Linear(d, dp)
        # Slot identity is a Fourier projection of normalized amount, blended
        # with a learned per-amount embedding.
        self.dividend_amount_proj = nn.Linear(slot_fourier_1, dp, bias=False)
        self.dividend_amount_embed = nn.Embedding(
            _phase_action_size(DecisionPhase.DPHASE_DIVIDENDS),
            dp,
        )
        # action_features now carry only the per-amount price-move impact;
        # normalized amount is consumed by ``dividend_amount_proj``.
        self.dividend_key_mlp = self._make_action_key_mlp(action_feature_width=1)
        self.issue_actor_proj = nn.Linear(d, dp, bias=False)
        self.issue_info_proj = nn.Linear(d, dp)
        self.issue_action_embed = nn.Embedding(
            _phase_action_size(DecisionPhase.DPHASE_ISSUE),
            dp,
        )
        self.issue_key_mlp = self._make_action_key_mlp(action_feature_width=2)
        self.acq_offer_actor_proj = nn.Linear(d, dp, bias=False)
        self.acq_offer_info_proj = nn.Linear(2 * d, dp)
        self.acq_offer_action_embed = nn.Embedding(
            _phase_action_size(DecisionPhase.DPHASE_ACQ_OFFER),
            dp,
        )
        self.acq_offer_key_mlp = self._make_action_key_mlp(action_feature_width=4)

        # ACQ is factored into three sequential single-entity selections:
        # pick the acquiring corp, pick the target company
        # (``_acq_select_company_logits``), pick the price (below).
        # SELECT_PRICE is actor-conditioned with generated price-offset keys.
        # Slot identity is a Fourier projection of [slot_position_norm,
        # candidate_price_norm], where slot_position_norm = offset / (K-1) and
        # candidate_price_norm = (low_price + offset) / COMPANY_PRICE_DIVISOR,
        # blended with a learned per-offset embedding. Per-slot action features
        # carry the remaining context.
        self.acq_price_actor_proj = nn.Linear(d, dp, bias=False)
        self.acq_price_info_proj = nn.Linear(2 * d, dp)
        self.acq_price_offset_proj = nn.Linear(slot_fourier_2, dp, bias=False)
        self.acq_price_offset_embed = nn.Embedding(
            _phase_action_size(DecisionPhase.DPHASE_ACQ_SELECT_PRICE),
            dp,
        )
        self.acq_price_key_mlp = self._make_action_key_mlp(action_feature_width=4)

        # PAR is actor-conditioned with generated par-price keys. The key
        # context includes the active corp/company and the PAR info token; the
        # per-action features come from the PAR token's price-specific tuples.
        # Slot identity is a Fourier projection of normalized index + actual
        # par price (max=37), blended with a learned per-par-index embedding.
        self.par_actor_proj = nn.Linear(d, dp, bias=False)
        self.par_info_proj = nn.Linear(3 * d, dp)
        self.par_price_proj = nn.Linear(slot_fourier_2, dp, bias=False)
        self.par_price_embed = nn.Embedding(
            _phase_action_size(DecisionPhase.DPHASE_PAR),
            dp,
        )
        self.par_key_mlp = self._make_action_key_mlp(action_feature_width=3)

        # --- Value head (applied per player token) ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

        self._validate_policy_layout()
        self._init_weights()

    def _make_action_key_mlp(self, action_feature_width: int) -> nn.Sequential:
        """Create an MLP that maps per-action features to policy key vectors."""
        dp = self.cfg.d_proj
        return nn.Sequential(
            nn.Linear(2 * dp + action_feature_width, dp),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(dp, dp),
        )

    @staticmethod
    def _match_dtype_device(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Move small buffers/raw slices to the current runtime dtype and device."""
        return tensor.to(device=ref.device, dtype=ref.dtype)

    def _slot_fourier_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the configured fixed Fourier expansion to slot scalars.

        Raw features stay first so the key projection can still learn monotone
        price/offset effects directly; sin/cos bands add slot-distinguishing
        high-frequency structure without making every slot fully independent.
        """
        if self.cfg.price_slot_fourier_bands == 0:
            return features
        bands = self._match_dtype_device(self._fourier_band_freqs, features)
        angles = features.unsqueeze(-1) * bands
        sincos = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        return torch.cat([features, sincos.flatten(start_dim=-3)], dim=-1)

    def _blend_price_slot_keys(
        self,
        keys: torch.Tensor,
        embedding: nn.Embedding,
    ) -> torch.Tensor:
        """Blend Fourier-projected slot keys with learned per-slot embeddings."""
        blend = float(self.cfg.price_slot_residual_scale)
        if blend == 0.0:
            return keys
        batch_size, num_actions, _ = keys.shape
        learned = self._match_dtype_device(embedding.weight, keys)
        learned = learned.unsqueeze(0).expand(batch_size, num_actions, -1)
        if blend == 1.0:
            return learned
        return (1.0 - blend) * keys + blend * learned

    def _project_company_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project company tokens from their raw feature fields."""
        return self.company_proj(
            x[
                :,
                self._company_slice,
                self._token_feature_start:self._company_rel_tail_start,
            ]
        )

    def _project_corp_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project corp tokens, adding learned row-order corp identity."""
        corp_tokens = self.corp_proj(
            x[
                :,
                self._corp_slice,
                self._token_feature_start:self._corp_rel_tail_start,
            ]
        )
        corp_ids = self._match_dtype_device(self.corp_id_embed(self._corp_ids), corp_tokens)
        return corp_tokens + corp_ids

    def _project_fi_token(self, x: torch.Tensor) -> torch.Tensor:
        """Project the FI token from its raw feature fields."""
        return self.fi_proj(
            x[
                :,
                self._fi_idx,
                self._token_feature_start:self._fi_rel_tail_start,
            ]
        )

    def _project_global_info_token(self, x: torch.Tensor) -> torch.Tensor:
        """Project global info.

        When direct phase conditioning is enabled, the decision-phase one-hot
        feeds the per-block modulation path and is omitted here. When it is
        disabled, keep the one-hot in the global token so phase remains visible
        to the trunk.
        """
        return self.global_info_proj(
            x[
                :,
                self._global_info_idx,
                self._global_info_feature_start:int(TokenWidth.TW_GLOBAL_INFO),
            ]
        )

    def _project_player_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project player tokens from their raw feature fields."""
        return self.player_proj(
            x[
                :,
                self._player_slice,
                self._token_feature_start:self._player_rel_tail_start,
            ]
        )

    def _attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Build SDPA key-visibility mask from input token rows.

        Shape is ``(B, 1, 1, N)`` so it broadcasts over heads and query
        positions against SDPA attention weights ``(B, H, N, N)``. The mask is
        tensor-only and has no data-dependent branches, keeping it compatible
        with ``torch.compile`` and CUDA graph capture. This suppresses padded
        rows as keys. Padded player query rows still execute, but their
        ``is_selected`` flags are zero and downstream consumers ignore their
        value slots in mixed-count training.
        """
        return (x[:, :, 0] > 0.5)[:, None, None, :]

    def _phase_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Return active decision-phase ids from the GlobalInfo one-hot."""
        phase_onehot = x[
            :,
            self._global_info_idx,
            _GLOBAL_PHASE_OFFSET:_GLOBAL_PHASE_STOP,
        ]
        return phase_onehot.argmax(dim=-1)

    def _relation_attention_bias(
        self,
        relation_flags: torch.Tensor,
        layer_idx: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Combine relation planes into an SDPA additive bias for one layer.

        ``relation_flags`` is ``(B, R, N, N)``. The learned multipliers for
        layer ``layer_idx`` are ``(H, R)``, producing ``(B, H, N, N)`` so the
        result lines up with SDPA attention weights.
        """
        relation_mult = self._match_dtype_device(
            self.relation_bias_mult[layer_idx],
            ref,
        )
        return torch.einsum("brij,hr->bhij", relation_flags, relation_mult)

    def _prepare_sparse_relation_context(
        self,
        relation_coords: torch.Tensor,
    ) -> _SparseRelationContext:
        """Precompute per-forward sparse relation indices shared by all layers."""
        batch_size = relation_coords.shape[0]
        num_heads = self.cfg.num_heads
        num_tokens = self.cfg.num_tokens

        coords = relation_coords.to(dtype=torch.long)
        relation_ids = coords[..., 0]
        query_tokens = coords[..., 1]
        key_tokens = coords[..., 2]
        valid_edges = relation_coords.any(dim=-1).to(dtype=torch.bool)

        batch_offsets = (
            torch.arange(batch_size, device=relation_coords.device, dtype=torch.long)
            .reshape(batch_size, 1, 1)
            * (num_heads * num_tokens * num_tokens)
        )
        head_offsets = (
            torch.arange(num_heads, device=relation_coords.device, dtype=torch.long)
            .reshape(1, num_heads, 1)
            * (num_tokens * num_tokens)
        )
        edge_offsets = (
            query_tokens[:, None, :] * num_tokens
            + key_tokens[:, None, :]
        )
        flat_indices = batch_offsets + head_offsets + edge_offsets

        return _SparseRelationContext(
            relation_ids=relation_ids,
            flat_indices=flat_indices,
            valid_edges=valid_edges,
        )

    def _sparse_relation_attention_bias(
        self,
        relation_ctx: _SparseRelationContext,
        layer_idx: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Build SDPA relation bias from sparse relation triplets.

        The result is still dense ``(B, H, N, N)`` because SDPA consumes a dense
        additive attention mask, but the expensive relation-type dimension is
        skipped: only emitted sparse edges gather learned per-head multipliers
        and scatter-add into the final bias tensor.
        """
        batch_size = relation_ctx.relation_ids.shape[0]
        num_heads = self.cfg.num_heads
        num_tokens = self.cfg.num_tokens
        relation_mult = self._match_dtype_device(
            self.relation_bias_mult[layer_idx],
            ref,
        )
        edge_values = F.embedding(
            relation_ctx.relation_ids,
            relation_mult.transpose(0, 1),
        ).transpose(1, 2)
        edge_values = edge_values.masked_fill(
            ~relation_ctx.valid_edges[:, None, :],
            0.0,
        )

        bias = ref.new_zeros(batch_size, num_heads, num_tokens, num_tokens)
        bias.reshape(-1).scatter_add_(
            0,
            relation_ctx.flat_indices.reshape(-1),
            edge_values.reshape(-1),
        )
        return bias

    def _validate_policy_layout(self) -> None:
        """Validate policy readout widths against the shared action-size table.

        The unified output is manually concatenated in DecisionPhase order.
        This guard catches action-space edits that update ``core.data`` but
        forget to adjust the corresponding model readout or block layout.
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

        if self.invest_pass_head.out_features != 1:
            raise AssertionError(
                f"invest pass head must be a single-logit head; "
                f"got width {self.invest_pass_head.out_features}"
            )
        block_widths = [0] * NUM_PHASES
        block_widths[int(DecisionPhase.DPHASE_INVEST)] = (
            self.invest_pass_head.out_features
            + num_companies
            + num_corps * 2
        )
        if self.bid_pass_head.out_features != 1:
            raise AssertionError(
                f"bid pass head must be a single-logit head; "
                f"got width {self.bid_pass_head.out_features}"
            )
        block_widths[int(DecisionPhase.DPHASE_BID)] = (
            self.bid_pass_head.out_features
            + int(AUCTION_CAP)
        )
        if self.acq_select_corp_pass_head.out_features != 1:
            raise AssertionError(
                f"acq select corp pass head must be a single-logit head; "
                f"got width {self.acq_select_corp_pass_head.out_features}"
            )
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_CORP)] = (
            self.acq_select_corp_pass_head.out_features
            + num_corps
        )
        block_widths[int(DecisionPhase.DPHASE_ACQ_OFFER)] = (
            self.acq_offer_action_embed.num_embeddings
        )
        if self.closing_pass_head.out_features != 1:
            raise AssertionError(
                f"closing pass head must be a single-logit head; "
                f"got width {self.closing_pass_head.out_features}"
            )
        block_widths[int(DecisionPhase.DPHASE_CLOSING)] = (
            self.closing_pass_head.out_features
            + num_companies
        )
        block_widths[int(DecisionPhase.DPHASE_DIVIDENDS)] = (
            int(self._dividend_amount_features.shape[1])
        )
        block_widths[int(DecisionPhase.DPHASE_ISSUE)] = (
            self.issue_action_embed.num_embeddings
        )
        if self.ipo_pass_head.out_features != 1:
            raise AssertionError(
                f"ipo pass head must be a single-logit head; "
                f"got width {self.ipo_pass_head.out_features}"
            )
        block_widths[int(DecisionPhase.DPHASE_IPO)] = (
            self.ipo_pass_head.out_features
            + num_corps
        )
        num_par_prices = int(self._par_price_features.shape[0])
        par_feature_width = int(TokenWidth.TW_PAR) - self._token_feature_start
        if par_feature_width != num_par_prices * 3:
            raise AssertionError(
                f"PAR token feature width {par_feature_width} must equal "
                f"3 fields * {num_par_prices} par prices"
            )
        block_widths[int(DecisionPhase.DPHASE_PAR)] = num_par_prices
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_COMPANY)] = num_companies
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)] = (
            int(self._acq_price_offset_features.shape[1])
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

        Token rows receive a learned token-type embed after projection, and
        corp rows also receive learned row-order corp ID embeds. Entity
        ownership/share/presidency reference tails are intentionally excluded
        from projection because the same relations are supplied as attention
        bias planes. Other entity IDs, active-entity refs, and phase refs are
        left as raw projected features rather than learned additive embeddings.

        Args:
            x: (batch, cfg.num_tokens, token_dim) zero-padded raw features.
                Supported caller patterns are: fp16/fp32 under autocast on the
                eval path, or fp32 matching the projection weights on the
                non-autocast trainer / CPU paths. No explicit upcast happens
                here: an unconditional ``x.to(bf16)`` would mismatch fp32
                Linear weights in non-autocast paths (for example in-process
                NNEvaluator tests).
        Returns:
            ``(batch, cfg.num_tokens, d_model)`` projected input embeddings.
        """
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
            _slice_proj(x, self.acq_offer_proj, self._acq_offer_idx).unsqueeze(1),
            _slice_proj(x, self.acq_price_proj, self._acq_price_info_idx).unsqueeze(1),
            corp_tokens,                                                                # (B, 8, d)
            self._project_player_tokens(x),                                              # (B, N, d)
        ]
        # Additive per-type embedding broadcast over the batch. A single
        # indexed gather against ``type_embeds`` gives every token a
        # type-distinct signal even when its feature slice is all-zero.
        input_tokens = torch.cat(input_parts, dim=1)                                     # (B, cfg.num_tokens, d)
        type_embeds = self._match_dtype_device(self.type_embeds(self._type_ids), input_tokens)
        return input_tokens + type_embeds                                               # (B, num_tokens, d)

    def _active_token(
        self,
        x: torch.Tensor,
        token_slice: slice,
        token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Select the active entity embedding from a token family."""
        selector = self._match_dtype_device(
            x[:, token_slice, self._is_selected_offset],
            token_embeddings,
        )
        return torch.bmm(selector.unsqueeze(1), token_embeddings).squeeze(1)

    def _policy_context(self, tokens: torch.Tensor, x: torch.Tensor) -> _PolicyContext:
        """Slice final entity tokens and compute active entity embeddings once."""
        company_tokens = tokens[:, self._company_slice]
        corp_tokens = tokens[:, self._corp_slice]
        return _PolicyContext(
            raw_tokens=x,
            tokens=tokens,
            company_tokens=company_tokens,
            corp_tokens=corp_tokens,
            active_player=self._active_token(
                x,
                self._player_slice,
                tokens[:, self._player_slice],
            ),
            active_corp=self._active_token(x, self._corp_slice, corp_tokens),
            active_company=self._active_token(x, self._company_slice, company_tokens),
        )

    def _actor_entity_logits(
        self,
        actor: torch.Tensor,
        entities: torch.Tensor,
        actor_proj: nn.Linear,
        entity_proj: nn.Linear,
    ) -> torch.Tensor:
        """Score each entity key against an actor query using scaled dot products."""
        query = actor_proj(actor)
        keys = entity_proj(entities)
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        return logits / math.sqrt(self.cfg.d_proj)

    def _actor_action_key_logits(
        self,
        actor: torch.Tensor,
        phase_info: torch.Tensor,
        action_features: torch.Tensor,
        actor_proj: nn.Linear,
        info_proj: nn.Linear,
        action_keys: torch.Tensor,
        key_mlp: nn.Module,
    ) -> torch.Tensor:
        """Score generated action keys against an actor query.

        ``action_keys`` is a ``(B, num_actions, d_proj)`` tensor of per-slot
        identity vectors. Categorical-key phases pass an
        ``nn.Embedding.weight`` broadcast to the batch dim; price-like phases
        pass a blend of Fourier-projected normalized slot features and learned
        slot embeddings. Per-batch keys are supported by passing a
        tensor that already has a batch dim varying with active-entity context.
        """
        batch_size, num_actions, _ = action_features.shape
        query = actor_proj(actor)
        info = (
            info_proj(phase_info)
            .unsqueeze(1)
            .expand(batch_size, num_actions, self.cfg.d_proj)
        )
        action_ids = self._match_dtype_device(action_keys, phase_info)
        key_input = torch.cat(
            [info, action_ids, self._match_dtype_device(action_features, phase_info)],
            dim=-1,
        )
        keys = key_mlp(key_input)
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        return logits / math.sqrt(self.cfg.d_proj)

    def _invest_logits(
        self,
        ctx: _PolicyContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the Invest block: pass, company auction, interleaved buy/sell."""
        actor = ctx.active_player
        pass_logit = self.invest_pass_head(actor)
        query = self.invest_actor_proj(actor)
        auction_company_keys = self.invest_auction_company_proj(ctx.company_tokens)
        buy_corp_keys = self.invest_buy_corp_proj(ctx.corp_tokens)
        sell_corp_keys = self.invest_sell_corp_proj(ctx.corp_tokens)
        keys = torch.cat(
            [auction_company_keys, buy_corp_keys, sell_corp_keys],
            dim=1,
        )
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        logits = logits / math.sqrt(self.cfg.d_proj)
        company_stop = ctx.company_tokens.shape[1]
        corp_stop = company_stop + ctx.corp_tokens.shape[1]
        auction_company = logits[:, :company_stop]
        buy_corp = logits[:, company_stop:corp_stop]
        sell_corp = logits[:, corp_stop:]
        corp_trade = torch.stack((buy_corp, sell_corp), dim=-1).flatten(1)
        return pass_logit, auction_company, corp_trade

    def _acq_select_corp_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build ACQ_SELECT_CORP logits: pass plus one logit per corp."""
        actor = ctx.active_player
        pass_logit = self.acq_select_corp_pass_head(actor)
        corp_logits = self._actor_entity_logits(
            actor,
            ctx.corp_tokens,
            self.acq_select_corp_actor_proj,
            self.acq_select_corp_corp_proj,
        )
        return torch.cat([pass_logit, corp_logits], dim=-1)

    def _acq_select_company_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build ACQ_SELECT_COMPANY logits: one active-corp logit per company."""
        return self._actor_entity_logits(
            ctx.active_corp,
            ctx.company_tokens,
            self.acq_select_company_actor_proj,
            self.acq_select_company_company_proj,
        )

    def _closing_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build CLOSING logits: pass plus one logit per company."""
        actor = ctx.active_player
        pass_logit = self.closing_pass_head(actor)
        company_logits = self._actor_entity_logits(
            actor,
            ctx.company_tokens,
            self.closing_actor_proj,
            self.closing_company_proj,
        )
        return torch.cat([pass_logit, company_logits], dim=-1)

    def _ipo_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build IPO logits: pass plus one actor-conditioned corp logit each."""
        actor = ctx.active_player
        query = self.ipo_actor_proj(actor)
        context = self.ipo_context_proj(
            torch.cat([ctx.active_company, ctx.tokens[:, self._par_idx]], dim=-1)
        )
        corp_base = self.ipo_corp_proj(ctx.corp_tokens)
        batch_size, num_corps, _ = corp_base.shape
        context = context.unsqueeze(1).expand(batch_size, num_corps, self.cfg.d_proj)
        keys = self.ipo_key_mlp(torch.cat([corp_base, context], dim=-1))
        corp_logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        corp_logits = corp_logits / math.sqrt(self.cfg.d_proj)
        pass_logit = self.ipo_pass_head(actor)
        return torch.cat([pass_logit, corp_logits], dim=-1)

    def _bid_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build the Bid block: leave-auction pass plus 15 bid offsets.

        Slot identity is ``bid_offset_proj(fourier([normalized_offset,
        face_value_norm + offset / COMPANY_PRICE_DIVISOR]))``, blended with a
        learned per-offset embedding. The second scalar is the candidate bid
        for slot k expressed in the same /80 units as the company's normalized
        face_value. Per-slot action features carry only the auction-context
        channels (relative_offset, min_bid_value, is_first_bid).
        """
        batch_size = ctx.tokens.shape[0]
        raw_auction = ctx.raw_tokens[
            :,
            self._auction_idx,
            self._token_feature_start:int(TokenWidth.TW_AUCTION),
        ]
        min_bid_idx = raw_auction[:, 0:1]
        min_bid_value = raw_auction[:, 1:2]
        is_first_bid = raw_auction[:, 2:3]
        bid_offsets = self._match_dtype_device(
            self._bid_offset_features,
            ctx.tokens,
        ).expand(batch_size, int(AUCTION_CAP), 1)
        relative_offsets = (
            bid_offsets
            - self._match_dtype_device(min_bid_idx, bid_offsets).unsqueeze(1)
        )
        action_features = torch.cat(
            [
                relative_offsets,
                self._match_dtype_device(min_bid_value, bid_offsets)
                .unsqueeze(1)
                .expand_as(bid_offsets),
                self._match_dtype_device(is_first_bid, bid_offsets)
                .unsqueeze(1)
                .expand_as(bid_offsets),
            ],
            dim=-1,
        )

        raw_company_features = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._company_slice,
                self._token_feature_start:int(TokenWidth.TW_COMPANY),
            ],
            ctx.tokens,
        )
        active_company_selector = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._company_slice,
                self._is_selected_offset,
            ],
            ctx.tokens,
        )
        active_company_raw = torch.bmm(
            active_company_selector.unsqueeze(1),
            raw_company_features,
        ).squeeze(1)
        face_value_norm = active_company_raw[
            :,
            _COMPANY_FACE_VALUE_FEATURE_OFFSET:_COMPANY_FACE_VALUE_FEATURE_OFFSET + 1,
        ]
        candidate_bid_norm = (
            face_value_norm.unsqueeze(1).expand_as(bid_offsets)
            + self._match_dtype_device(self._bid_offset_dollar_norm, ctx.tokens)
            .expand_as(bid_offsets)
        )
        bid_keys = self.bid_offset_proj(
            self._slot_fourier_features(
                torch.cat([bid_offsets, candidate_bid_norm], dim=-1)
            )
        )
        bid_keys = self._blend_price_slot_keys(bid_keys, self.bid_offset_embed)

        bid_context = torch.cat(
            [ctx.tokens[:, self._auction_idx], ctx.active_company],
            dim=-1,
        )
        bid_offsets_logits = self._actor_action_key_logits(
            ctx.active_player,
            bid_context,
            action_features,
            self.bid_actor_proj,
            self.bid_info_proj,
            bid_keys,
            self.bid_key_mlp,
        )
        pass_logit = self.bid_pass_head(ctx.active_player)
        return torch.cat([pass_logit, bid_offsets_logits], dim=-1)

    def _dividend_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build dividend amount logits from active-corp query and action keys.

        Slot identity comes from ``dividend_amount_proj`` evaluated on Fourier
        features of the normalized amount, blended with a learned per-amount
        embedding. The only per-slot action feature is the price-move impact
        written into the dividend phase token by the engine; normalized amount
        is already carried by the slot-identity projection.
        """
        batch_size = ctx.tokens.shape[0]
        dividend_impacts = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._dividend_idx,
                self._token_feature_start:int(TokenWidth.TW_DIVIDEND),
            ],
            ctx.tokens,
        ).unsqueeze(-1)
        amount_keys = self.dividend_amount_proj(
            self._slot_fourier_features(
                self._match_dtype_device(self._dividend_amount_features, ctx.tokens)
                .squeeze(0)
            )
        ).unsqueeze(0).expand(batch_size, -1, -1)
        amount_keys = self._blend_price_slot_keys(
            amount_keys,
            self.dividend_amount_embed,
        )
        return self._actor_action_key_logits(
            ctx.active_corp,
            ctx.tokens[:, self._dividend_idx],
            dividend_impacts,
            self.dividend_actor_proj,
            self.dividend_info_proj,
            amount_keys,
            self.dividend_key_mlp,
        )

    def _issue_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build ISSUE logits: pass/no-issue plus issue one share."""
        batch_size = ctx.tokens.shape[0]
        num_actions = self.issue_action_embed.num_embeddings
        action_is_issue = self._match_dtype_device(
            self._issue_action_features,
            ctx.tokens,
        ).expand(batch_size, num_actions, 1)
        issue_impact = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._issue_idx,
                self._token_feature_start:int(TokenWidth.TW_ISSUE),
            ],
            ctx.tokens,
        )
        issue_impact_taken = action_is_issue * issue_impact.unsqueeze(1)
        action_features = torch.cat([action_is_issue, issue_impact_taken], dim=-1)
        return self._actor_action_key_logits(
            ctx.active_corp,
            ctx.tokens[:, self._issue_idx],
            action_features,
            self.issue_actor_proj,
            self.issue_info_proj,
            self.issue_action_embed.weight.unsqueeze(0).expand(batch_size, -1, -1),
            self.issue_key_mlp,
        )

    def _acq_offer_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build ACQ_OFFER logits: pass/reject plus accept offer."""
        batch_size = ctx.tokens.shape[0]
        num_actions = self.acq_offer_action_embed.num_embeddings
        action_is_accept = self._match_dtype_device(
            self._acq_offer_action_features,
            ctx.tokens,
        ).expand(batch_size, num_actions, 1)
        raw_acq_offer = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._acq_offer_idx,
                self._token_feature_start:int(TokenWidth.TW_ACQ_OFFER),
            ],
            ctx.tokens,
        )
        offer_price_index = action_is_accept * raw_acq_offer[:, 0:1].unsqueeze(1)
        offer_price = action_is_accept * raw_acq_offer[:, 1:2].unsqueeze(1)
        fi_company = raw_acq_offer[:, 2:3].unsqueeze(1).expand_as(action_is_accept)
        action_features = torch.cat(
            [action_is_accept, offer_price_index, offer_price, fi_company],
            dim=-1,
        )
        acq_offer_context = torch.cat(
            [ctx.tokens[:, self._acq_offer_idx], ctx.active_company],
            dim=-1,
        )
        return self._actor_action_key_logits(
            ctx.active_corp,
            acq_offer_context,
            action_features,
            self.acq_offer_actor_proj,
            self.acq_offer_info_proj,
            self.acq_offer_action_embed.weight.unsqueeze(0).expand(batch_size, -1, -1),
            self.acq_offer_key_mlp,
        )

    def _par_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build par-price logits from active-player query and par-price keys.

        Slot identity is a Fourier projection of static
        ``[normalized_index, normalized_par_price]`` features, blended with a
        learned per-par-index embedding, broadcast across the batch. Per-slot
        action features carry only the PAR token's per-price effect tuple
        (``par_effects``); the price itself is in the slot key.
        """
        batch_size = ctx.tokens.shape[0]
        num_par_prices = int(self._par_price_features.shape[0])
        par_effects = (
            self._match_dtype_device(
                ctx.raw_tokens[
                    :,
                    self._par_idx,
                    self._token_feature_start:int(TokenWidth.TW_PAR),
                ],
                ctx.tokens,
            )
            .reshape(batch_size, num_par_prices, 3)
        )
        par_keys = self.par_price_proj(
            self._slot_fourier_features(
                self._match_dtype_device(self._par_price_features, ctx.tokens)
            )
        ).unsqueeze(0).expand(batch_size, -1, -1)
        par_keys = self._blend_price_slot_keys(par_keys, self.par_price_embed)
        par_context = torch.cat(
            [ctx.tokens[:, self._par_idx], ctx.active_corp, ctx.active_company],
            dim=-1,
        )
        return self._actor_action_key_logits(
            ctx.active_player,
            par_context,
            par_effects,
            self.par_actor_proj,
            self.par_info_proj,
            par_keys,
            self.par_key_mlp,
        )

    def _acq_price_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build ACQ_SELECT_PRICE logits from active-corp query and offset keys.

        Slot identity is ``acq_price_offset_proj(fourier([slot_position_norm,
        candidate_price_norm]))``, blended with a learned per-offset embedding,
        where ``slot_position_norm = offset / (K-1)`` is the 0-indexed slot
        index in [0, 1] (matching BID/PAR/DIVIDENDS) and
        ``candidate_price_norm = (low_price + offset) / COMPANY_PRICE_DIVISOR``.
        ``slot_position_norm`` doubles as the term subtracted from the
        engine's ``max_offset = (high - low) / 50`` to produce a clean
        "options remaining" action feature; both share divisor 50 = K-1.
        """
        batch_size = ctx.tokens.shape[0]
        num_offsets = int(self._acq_price_offset_features.shape[1])
        offset_features = self._match_dtype_device(
            self._acq_price_offset_features,
            ctx.tokens,
        ).expand(batch_size, num_offsets, -1)
        slot_position_norm = offset_features[:, :, 0:1]
        offset_price_delta = offset_features[:, :, 1:2]

        raw_acq_price = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._acq_price_info_idx,
                self._token_feature_start:int(TokenWidth.TW_ACQ_PRICE),
            ],
            ctx.tokens,
        )
        max_offset = raw_acq_price[:, 0:1].unsqueeze(1).expand_as(slot_position_norm)
        fi_flag = raw_acq_price[:, 1:2].unsqueeze(1).expand_as(slot_position_norm)
        total_synergies = (
            raw_acq_price[:, 2:3].unsqueeze(1).expand_as(slot_position_norm)
        )

        raw_company_features = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._company_slice,
                self._token_feature_start:int(TokenWidth.TW_COMPANY),
            ],
            ctx.tokens,
        )
        active_company_selector = self._match_dtype_device(
            ctx.raw_tokens[
                :,
                self._company_slice,
                self._is_selected_offset,
            ],
            ctx.tokens,
        )
        active_company_raw = torch.bmm(
            active_company_selector.unsqueeze(1),
            raw_company_features,
        ).squeeze(1)
        low_price = active_company_raw[
            :,
            _COMPANY_LOW_PRICE_FEATURE_OFFSET:_COMPANY_LOW_PRICE_FEATURE_OFFSET + 1,
        ]
        candidate_price = (
            low_price.unsqueeze(1).expand_as(slot_position_norm) + offset_price_delta
        )
        remaining_after_offset = max_offset - slot_position_norm
        action_features = torch.cat(
            [
                remaining_after_offset,
                max_offset,
                fi_flag,
                total_synergies,
            ],
            dim=-1,
        )
        acq_price_keys = self.acq_price_offset_proj(
            self._slot_fourier_features(
                torch.cat([slot_position_norm, candidate_price], dim=-1)
            )
        )
        acq_price_keys = self._blend_price_slot_keys(
            acq_price_keys,
            self.acq_price_offset_embed,
        )
        acq_price_context = torch.cat(
            [ctx.tokens[:, self._acq_price_info_idx], ctx.active_company],
            dim=-1,
        )
        return self._actor_action_key_logits(
            ctx.active_corp,
            acq_price_context,
            action_features,
            self.acq_price_actor_proj,
            self.acq_price_info_proj,
            acq_price_keys,
            self.acq_price_key_mlp,
        )

    # ------------------------------------------------------------------
    # Unified policy: every readout runs once on the full batch
    # ------------------------------------------------------------------

    def _build_unified_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Run every per-phase policy readout once on the full batch and concat
        into a single ``(B, UNIFIED_LOGIT_DIM)`` tensor.

        Blocks are emitted in DecisionPhase order (matching the offsets baked
        into ``build_action_lut``). Phases with a pass/no-op action keep it at
        phase-local slot 0. Readouts run unconditionally regardless of which
        phase a given row is in: the caller's legal mask suppresses slots
        outside the current phase's action space. The extra work is small
        relative to the trunk.
        """
        # INVEST: pass + 36 company-select + 16 corp-trade (2i buy, 2i+1 sell).
        pass_invest, invest_company, corp_trade = self._invest_logits(ctx)
        # BID: pass + AUCTION_CAP raise offsets.
        bid = self._bid_logits(ctx)                                              # (B, 16)
        # ACQ_SELECT_CORP: pass + 8 corps.
        acq_select_corp = self._acq_select_corp_logits(ctx)                     # (B, 9)
        # ACQ_OFFER: pass + 1 accept-buy.
        acq_offer = self._acq_offer_logits(ctx)                                  # (B, 2)
        # CLOSING: pass + 36 company-close.
        closing = self._closing_logits(ctx)                                      # (B, 37)
        # DIVIDENDS: 26 levels (no pass).
        dividend = self._dividend_logits(ctx)                                    # (B, 26)
        # ISSUE: pass + 1 issue.
        issue = self._issue_logits(ctx)                                          # (B, 2)
        # IPO: pass + 8 corps.
        ipo = self._ipo_logits(ctx)                                              # (B, 9)
        # PAR: 14 par indices (no pass).
        par_price = self._par_logits(ctx)                                        # (B, 14)
        # ACQ_SELECT_COMPANY: 36 companies (no pass).
        acq_select_company = self._acq_select_company_logits(ctx)                # (B, 36)
        # ACQ_SELECT_PRICE: 51 price offsets (no pass).
        price_acq = self._acq_price_logits(ctx)                                  # (B, 51)

        return torch.cat(
            [
                pass_invest, invest_company, corp_trade,           # INVEST           (53)
                bid,                                               # BID              (16)
                acq_select_corp,                                   # ACQ_SELECT_CORP  ( 9)
                acq_offer,                                         # ACQ_OFFER        ( 2)
                closing,                                           # CLOSING          (37)
                dividend,                                          # DIVIDENDS        (26)
                issue,                                             # ISSUE            ( 2)
                ipo,                                               # IPO              ( 9)
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
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        relations: torch.Tensor,
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
            relations: Either dense ``(batch, NUM_ATTENTION_RELATIONS,
                num_tokens, num_tokens)`` uint8/bool directed relation planes
                or sparse eval-server coordinates ``(batch,
                MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH)``
                uint8. Dense rows are attention queries and columns are
                attention keys; sparse rows are ``(relation_id, query, key)``
                triplets padded with ``(0, 0, 0)``.

        Returns:
            policy_logits: ``(batch, UNIFIED_LOGIT_DIM)`` fp32 logits with
                illegal slots set to ``-1e9``. Static-shape regardless of phase.
            values: ``(batch, cfg.num_players)`` per-player expected outcomes
                in ``[-1, 1]``. In mixed-count training this is padded to the
                model capacity; callers mask or slice slots beyond the state's
                actual player count.
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
        if relations.device != x.device:
            raise AssertionError(
                f"relations device must match x device; got {relations.device} vs {x.device}"
            )
        tokens = self._project_tokens(x)
        relation_flags: torch.Tensor | None = None
        sparse_relation_ctx: _SparseRelationContext | None = None
        expected_dense_rel_shape = (
            x.shape[0],
            NUM_ATTENTION_RELATIONS,
            self.cfg.num_tokens,
            self.cfg.num_tokens,
        )
        expected_sparse_rel_shape = (
            x.shape[0],
            MAX_ATTENTION_RELATION_EDGES,
            ATTENTION_RELATION_COORD_WIDTH,
        )
        if tuple(relations.shape) == expected_dense_rel_shape:
            if relations.dtype not in (torch.bool, torch.uint8):
                raise AssertionError(
                    f"dense relation planes must be bool or uint8; got {relations.dtype}"
                )
            relation_flags = relations.to(dtype=tokens.dtype)
        elif tuple(relations.shape) == expected_sparse_rel_shape:
            if relations.dtype != torch.uint8:
                raise AssertionError(
                    f"sparse relation coordinates must be uint8; got {relations.dtype}"
                )
            sparse_relation_ctx = self._prepare_sparse_relation_context(relations)
        else:
            raise AssertionError(
                f"relations shape must be {expected_dense_rel_shape} for dense "
                f"planes or {expected_sparse_rel_shape} for sparse coordinates; "
                f"got {tuple(relations.shape)}"
            )
        attn_mask = self._attention_mask(x)
        phase_ids = self._phase_ids(x) if self.cfg.phase_conditioning else None

        for layer_idx, block in enumerate(self.blocks):
            if relation_flags is not None:
                relation_bias = self._relation_attention_bias(
                    relation_flags,
                    layer_idx,
                    tokens,
                )
            else:
                assert sparse_relation_ctx is not None
                relation_bias = self._sparse_relation_attention_bias(
                    sparse_relation_ctx,
                    layer_idx,
                    tokens,
                )
            tokens = block(tokens, attn_mask, relation_bias, phase_ids)
        tokens = self.final_norm(tokens)

        # Cast to fp32 before the sentinel: under autocast ``unified`` is in
        # the autocast dtype (bf16/fp16) but downstream softmax / log_softmax
        # wants stable fp32 ``-1e9`` sentinels. Real rows are expected to have
        # at least one legal slot; all-false rows are reserved for caller-owned
        # scratch entries whose output will be dropped before softmax consumers
        # interpret them.
        policy_ctx = self._policy_context(tokens, x)
        unified = self._build_unified_logits(policy_ctx).to(torch.float32)      # (B, U)
        policy_logits = unified.masked_fill(~legal_mask, -1e9)

        values = self.value_head(tokens[:, self._player_slice]).squeeze(-1)  # (B, N)
        return policy_logits, values

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def pass_action_logit_abs(
        self,
        policy_logits: torch.Tensor,
        legal_mask: torch.Tensor,
        phase_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-phase mean ``|pass logit|`` and mean ``|action logit|`` over
        legal slots.

        Used as a TB diagnostic to detect a chronic scale mismatch between the
        ``Linear(d, 1)`` pass heads and the ``q·k/√dp`` scored logits in the
        same readout. Pass logits init at ~3.5× the std of scored logits at
        ``d_model=192, d_proj=64``; if the trained model does not converge to
        a comparable ratio, the architectural asymmetry is wasting capacity.

        Returns a ``(2 * len(PHASES_WITH_PASS_HEAD),)`` tensor packed as
        ``[pass_abs_p0, action_abs_p0, pass_abs_p1, action_abs_p1, ...]`` in
        ``PHASES_WITH_PASS_HEAD`` order. Phases with no rows in the batch (or
        all-illegal pass slots) return 0 for both stats — the caller is
        expected to filter via per-phase row counts.
        """
        abs_logits = policy_logits.detach().abs()
        stats: list[torch.Tensor] = []
        for phase in PHASES_WITH_PASS_HEAD:
            offset = _PHASE_OFFSETS[phase]
            size = int(PHASE_ACTION_SIZES[phase])
            pass_slot = offset
            action_start = offset + 1
            action_stop = offset + size

            rows = (phase_ids == phase)
            pass_legal = legal_mask[:, pass_slot] & rows
            action_legal = legal_mask[:, action_start:action_stop] & rows.unsqueeze(1)

            pass_count = pass_legal.sum().clamp_min(1).to(abs_logits.dtype)
            action_count = action_legal.sum().clamp_min(1).to(abs_logits.dtype)

            pass_sum = (abs_logits[:, pass_slot] * pass_legal.to(abs_logits.dtype)).sum()
            action_sum = (
                abs_logits[:, action_start:action_stop]
                * action_legal.to(abs_logits.dtype)
            ).sum()

            stats.append(pass_sum / pass_count)
            stats.append(action_sum / action_count)
        return torch.stack(stats)

    def phase_mod_diagnostics(self) -> dict[str, float]:
        """Per-layer phase_mod magnitude + phase-distinguishing component.

        ``abs_mean`` (mean of ``|phase_mod.weight|``) tracks whether per-block
        modulation is moving off zero at all. ``phase_var`` (mean over channels
        of ``var(phase_mod.weight, dim=0)``) isolates the phase-distinguishing
        signal — if rows learn similar values, ``abs_mean`` grows but
        ``phase_var`` stays small, indicating adaLN is learning a phase-
        independent constant rather than per-phase modulation.
        """
        if not self.cfg.phase_conditioning:
            return {}
        scalars: dict[str, float] = {}
        for layer_idx, block in enumerate(self.blocks):
            assert isinstance(block, TransformerBlock)
            assert block.phase_mod is not None
            weight = block.phase_mod.weight.detach()
            scalars[f"phase_mod/abs_mean/layer_{layer_idx}"] = weight.abs().mean().item()
            scalars[f"phase_mod/phase_var/layer_{layer_idx}"] = (
                weight.var(dim=0, unbiased=False).mean().item()
            )
        return scalars

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT/LLaMA-style trunc-normal init, zero-init phase gates for identity start.

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

        nn.init.trunc_normal_(self.corp_id_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.bid_offset_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.acq_price_offset_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.dividend_amount_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.issue_action_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.acq_offer_action_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.par_price_embed.weight, std=0.02)
        # Per-type additive embeddings: same small-random init.
        nn.init.trunc_normal_(self.type_embeds.weight, std=0.02)
        # Relation attention starts behavior-preserving; training can learn
        # positive or negative head/layer-specific biases from zero.
        nn.init.zeros_(self.relation_bias_mult)

        # Zero-init pass-head weights so the pass logit starts at exactly 0,
        # matching the mean of the action-key logits. With trunc_normal(0.02)
        # and the action path's /sqrt(dp) scaling, an unzeroed pass head has
        # ~3.6x the action-logit stdev and biases softmax toward pass at init.
        for pass_head in (
            self.invest_pass_head,
            self.bid_pass_head,
            self.acq_select_corp_pass_head,
            self.closing_pass_head,
            self.ipo_pass_head,
        ):
            nn.init.zeros_(pass_head.weight)

        # Zero-init phase modulation so each block starts as identity while
        # branch projections keep normal init and can feed gradients to gates.
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            if block.phase_mod is not None:
                nn.init.zeros_(block.phase_mod.weight)


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
    print(f"  d_model={cfg.d_model}, d_proj={cfg.d_proj}, heads={cfg.num_heads}, "
          f"layers={cfg.num_layers}, d_ff={_ffn_hidden_dim(cfg)}")
    print(f"  phase_conditioning={cfg.phase_conditioning}")
    print(f"  tokens={cfg.num_tokens}, token_dim={cfg.token_dim}")
    print(f"  Trainable parameters: {total:,}")
    print()

    # --- Parameter breakdown ---
    proj_modules = [
        model.player_proj, model.corp_proj, model.company_proj,
        model.fi_proj, model.market_info_proj, model.global_info_proj,
        model.invest_proj, model.auction_proj, model.dividend_proj,
        model.issue_proj, model.par_proj, model.acq_offer_proj,
        model.acq_price_proj,
    ]
    proj_params = sum(sum(p.numel() for p in m.parameters()) for m in proj_modules)
    corp_id_params = model.corp_id_embed.weight.numel()
    bid_offset_params = (
        sum(p.numel() for p in model.bid_offset_proj.parameters())
        + model.bid_offset_embed.weight.numel()
    )
    acq_price_offset_params = sum(
        p.numel() for p in model.acq_price_offset_proj.parameters()
    ) + model.acq_price_offset_embed.weight.numel()
    dividend_amount_params = (
        sum(p.numel() for p in model.dividend_amount_proj.parameters())
        + model.dividend_amount_embed.weight.numel()
    )
    issue_action_params = model.issue_action_embed.weight.numel()
    acq_offer_action_params = model.acq_offer_action_embed.weight.numel()
    par_price_params = (
        sum(p.numel() for p in model.par_price_proj.parameters())
        + model.par_price_embed.weight.numel()
    )
    type_params = model.type_embeds.weight.numel()
    phase_mod_params = 0
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        if block.phase_mod is not None:
            phase_mod_params += block.phase_mod.weight.numel()
    trunk_params = (
        sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.final_norm.parameters())
        - phase_mod_params
    )
    policy_modules: list[nn.Module] = [
        model.invest_pass_head,
        model.invest_actor_proj,
        model.invest_auction_company_proj,
        model.invest_buy_corp_proj,
        model.invest_sell_corp_proj,
        model.closing_pass_head,
        model.closing_actor_proj, model.closing_company_proj,
        model.acq_select_company_actor_proj, model.acq_select_company_company_proj,
        model.acq_select_corp_pass_head,
        model.acq_select_corp_actor_proj, model.acq_select_corp_corp_proj,
        model.ipo_pass_head, model.ipo_actor_proj, model.ipo_corp_proj,
        model.ipo_context_proj, model.ipo_key_mlp,
        model.bid_pass_head, model.bid_actor_proj, model.bid_info_proj,
        model.bid_key_mlp,
        model.dividend_actor_proj, model.dividend_info_proj, model.dividend_key_mlp,
        model.issue_actor_proj, model.issue_info_proj, model.issue_key_mlp,
        model.acq_offer_actor_proj, model.acq_offer_info_proj, model.acq_offer_key_mlp,
        model.acq_price_actor_proj, model.acq_price_info_proj, model.acq_price_key_mlp,
        model.par_actor_proj, model.par_info_proj, model.par_key_mlp,
    ]
    policy_params = sum(sum(p.numel() for p in m.parameters()) for m in policy_modules)
    value_params = sum(p.numel() for p in model.value_head.parameters())

    print("Parameter breakdown:")
    for name, count in [
        ("Input projections", proj_params),
        ("Corp ID embeds", corp_id_params),
        ("Bid slot keys", bid_offset_params),
        ("ACQ price slot keys", acq_price_offset_params),
        ("Dividend slot keys", dividend_amount_params),
        ("Issue action embeds", issue_action_params),
        ("ACQ offer action embeds", acq_offer_action_params),
        ("PAR slot keys", par_price_params),
        ("Type embeds", type_params),
        ("Phase mod embeds", phase_mod_params),
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
    relations = torch.zeros(
        batch_size,
        NUM_ATTENTION_RELATIONS,
        cfg.num_tokens,
        cfg.num_tokens,
        dtype=torch.uint8,
    )
    for i in range(NUM_PHASES):
        n = PHASE_ACTION_SIZES[i]
        legal_mask[i, lut[i, :n]] = True

    policy_logits, values = model(x, legal_mask, relations)

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
