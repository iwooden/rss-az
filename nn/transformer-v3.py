"""Transformer v3 model for Rolling Stock Stars AlphaZero training.

Token-based architecture: each game entity is a separate input token. Type-specific
linear projections feed one nonlinear relation-aggregation residual, a pre-LN
transformer trunk, actor-conditioned policy readouts, and a value head.

INVEST uses separate auction, trade, and pass MLPs with one GELU hidden
layer of width d_model. Auction/trade scorers share weights across candidates;
the trade scorer emits buy/sell together and also receives the actor's
normalized share count for that corporation. CLOSING uses player/company and
player-only pass MLPs with the same hidden-layer design. BID jointly scores
leaving and all bid levels from player/company/auction embeddings and raw
per-price bid and remaining-cash features. IPO uses corporation and pass MLPs;
PAR scores all prices jointly with the engine's raw capitalization previews.
DIVIDENDS jointly scores all amounts with payout, cash, actual movement, price, and net-worth features.
Acquisition uses corporation/company candidate MLPs and a joint price MLP.
ISSUE jointly scores pass/issue from player/corporation/issue embeddings.
ACQ_OFFER jointly scores reject/accept from player/corporation/company/offer embeddings.

V3 adds actor-relative INVEST trade history to corp tokens and persistent
per-player round-trip flags. Layout 3 is selected by this model's contract;
the game state and attention relation planes are shared with v2.

Key differences from the MLP model (nn/template.py):
  - Input: (batch, num_tokens, token_dim) token features, not flat state vector
  - No state rotation: active player marked with is_active flag
  - Actor-conditioned policy: phase-specific MLP readouts
  - ACQ factored into three single-entity sub-phases (corp/company/price)
  - Unified policy output: every readout writes into a static
    (B, UNIFIED_LOGIT_DIM) tensor, with illegal slots masked by caller input
  - Value read from player tokens directly (no un-rotation needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    ATTENTION_RELATION_SCALES,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
)
from core.data import (
    ALL_PAR_PRICES,
    AUCTION_CAP,
    GameConstants,
    PHASE_ACTION_SIZES,
    DecisionPhase,
    PY_CASH_DIVISOR,
    PY_COMPANY_PRICE_DIVISOR,
    PY_COMPANY_SYNERGY_DIVISOR,
    PY_SHARE_DIVISOR,
    PY_SHARE_PRICE_DIVISOR,
)
from core.token_data_v3 import TokenDataSize, TokenWidth, get_num_tokens, get_token_widths
from entities.company import COMPANIES
from nn.policy_layout import (
    NUM_PHASES,
    PHASE_OFFSETS,
    PHASES_WITH_PASS_SLOT,
    UNIFIED_LOGIT_DIM,
    build_action_lut,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Decision phases / action sizes all live in ``core.data`` and are imported
# above. This module is strictly a consumer; editing policy readout widths or
# adding token types happens over there.

# Worker/replay relations keep their existing contract. Static relations are
# built from engine data at model construction and never cross IPC.
_COMPANY_SYNERGY_RELATION = NUM_ATTENTION_RELATIONS
_COMPANY_SYNERGY_STRENGTH_RELATION = _COMPANY_SYNERGY_RELATION + 1
_NUM_MODEL_RELATIONS = _COMPANY_SYNERGY_STRENGTH_RELATION + 1

# Input-buffer token-type taxonomy. Index order must stay stable — the static
# ``_type_ids`` buffer built in ``RSSTransformerNet.__init__`` indexes into
# ``type_embeds`` using these ids, and multi-instance types (company / corp
# / player) share a single row.
class _TokenType(IntEnum):
    MARKET_INFO = 0
    COMPANY = 1
    FI = 2
    GLOBAL_INFO = 3
    INVEST = 4
    AUCTION = 5
    DIVIDEND = 6
    ISSUE = 7
    PAR = 8
    ACQ_OFFER = 9
    ACQ_PRICE = 10
    CORP = 11
    PLAYER = 12


INPUT_LAYOUT_VERSION = 3

_GELU_APPROX = "tanh"
_TOKEN_FEATURE_START = 1
_IS_SELECTED_OFFSET = 1
# Offsets inside a company feature slice after the attention-mask slot is dropped.
_COMPANY_LOW_PRICE_FEATURE_OFFSET = 1
_COMPANY_FACE_VALUE_FEATURE_OFFSET = 2
_COMPANY_ACQ_SYNERGY_OFFSET = 13  # Raw company-token slot.
# Raw token offsets where relation/reference tails begin. The engine still
# emits these fields for compatibility and diagnostics, but the model ignores
# them in token projection now that relation matrices feed attention directly.
# Aggregate "relational summary" scalars (owned-company counts, presidency
# count, total shares) sit immediately before each rel-tail start, so they
# stay in the projection while the multihots they summarize are dropped.
_COMPANY_REL_TAIL_START = 14
_FI_REL_TAIL_START = 4
_CORP_REL_TAIL_START = 57
# Player share amounts are scalar quantities, not just relation presence:
# keep OFF_SHARES (8 slots) in projected token features and drop only the
# owned-company relation tail.
_PLAYER_REL_TAIL_START = 26
# Raw player-token offsets; match core.token_data_v3::_fill_player_token.
_PLAYER_SHARES_START = 15
_PLAYER_CASH_OFFSET = 8
# Raw corporation-token offsets; match core.token_data_v3::_fill_corp_token.
_CORP_ISSUED_OFFSET = 6
_CORP_BANK_SHARES_OFFSET = 7
_CORP_SHARE_PRICE_OFFSET = 35
_CORP_CASH_OFFSET = 37
_CORP_ACQ_PROCEEDS_OFFSET = 38


def _phase_action_size(phase: DecisionPhase) -> int:
    return int(PHASE_ACTION_SIZES[int(phase)])


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
    num_heads: int = 4
    num_layers: int = 15
    ff_mult: float = 3.0  # FFN inner dimension is rounded up to a multiple of 64.

    # Raw feature width per token (zero-padded to same size across types).
    # Sourced from core.token_data_v3 so the model and the Cython extractor
    # can't drift out of sync.
    layout_version: int = field(default=INPUT_LAYOUT_VERSION, init=False)
    token_dim: int = int(TokenDataSize.TOKEN_DIM)

    _num_tokens: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert 3 <= self.num_players <= 5, f"num_players must be 3-5, got {self.num_players}"
        assert self.d_model > 0, f"d_model must be positive, got {self.d_model}"
        assert self.num_heads > 0, f"num_heads must be positive, got {self.num_heads}"
        assert self.num_layers > 0, f"num_layers must be positive, got {self.num_layers}"
        assert self.ff_mult > 0, f"ff_mult must be positive, got {self.ff_mult}"
        assert self.d_model % self.num_heads == 0, (
            f"d_model {self.d_model} must be divisible by num_heads {self.num_heads}"
        )
        object.__setattr__(self, "_num_tokens", int(get_num_tokens(self.num_players)))

    @property
    def num_tokens(self) -> int:
        """Input-buffer token count: fixed entity/phase tokens + player capacity."""
        return self._num_tokens


def _validate_layout(num_players: int) -> None:
    """Assert the hardcoded token indices in ``RSSTransformerNet.__init__``
    line up with ``core.token_data_v3.get_token_widths`` for the given player
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
        f"token layout drift between nn/transformer-v3.py and core/token_data_v3.pyx "
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
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model {d_model} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_norm = nn.RMSNorm(d_model)
        # Packed Q/K/V projection. The trunk follows the modern RMSNorm +
        # SwiGLU convention: projection matrices are biasless, while relation
        # attention bias is handled by relation_bias_mult.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        relation_bias: torch.Tensor,
    ) -> torch.Tensor:
        h = self.attn_norm(x)
        B, N, D = h.shape
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
        x = x + h
        h = self.ffn_norm(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
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
    query_tokens: torch.Tensor
    key_tokens: torch.Tensor
    flat_indices: torch.Tensor
    valid_edges: torch.Tensor
    edge_weights: torch.Tensor


class RelationInputMixing(nn.Module):
    """One simultaneous round of directed messages before the transformer.

    A shared nonlinear map encodes sources; each relation weights/sums its neighbors,
    divides by sqrt(neighbor count), and applies its own bias-free projection
    and independent signed gain. Empty neighborhoods contribute exactly zero.
    Source RMSNorm controls feature scale without erasing aggregate magnitude.
    Dense/sparse weights are already normalized. Sparse inputs aggregate
    directly, without materializing relation planes. Static company relations
    use the same source map, restricted to the company-token block.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.source_norm = nn.RMSNorm(d_model)
        self.source_mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.relation_projs = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=False)
            for _ in range(_NUM_MODEL_RELATIONS)
        ])
        # Nonzero gains let the message weights learn on the first step.
        self.relation_gains = nn.Parameter(torch.full((_NUM_MODEL_RELATIONS,), 0.1))

    def forward(
        self,
        tokens: torch.Tensor,
        visible: torch.Tensor,
        relation_flags: torch.Tensor | None,
        sparse_ctx: _SparseRelationContext | None,
        static_relations: torch.Tensor | None = None,
        company_slice: slice | None = None,
    ) -> torch.Tensor:
        sources: torch.Tensor = self.source_mlp(self.source_norm(tokens))
        gains = self.relation_gains.to(dtype=sources.dtype)
        messages = torch.zeros_like(tokens)

        edge_sources: torch.Tensor | None = None
        edge_visible: torch.Tensor | None = None
        if sparse_ctx is not None:
            edge_sources = sources.gather(
                1, sparse_ctx.key_tokens.unsqueeze(-1).expand(-1, -1, sources.shape[-1]),
            )
            edge_sources = edge_sources * sparse_ctx.edge_weights.unsqueeze(-1).to(
                dtype=sources.dtype,
            )
            edge_visible = sparse_ctx.valid_edges & visible.gather(1, sparse_ctx.key_tokens)

        for relation_id, proj in enumerate(self.relation_projs):
            if relation_id >= NUM_ATTENTION_RELATIONS:
                if static_relations is None:
                    continue
                assert company_slice is not None
                # Only companies participate. Use a compact, shared 36x36
                # matrix instead of expanding the sparse per-state edge list.
                adjacency = static_relations[relation_id - NUM_ATTENTION_RELATIONS].to(
                    dtype=sources.dtype,
                )[None] * visible[:, None, company_slice].to(dtype=sources.dtype)
                aggregate = torch.bmm(adjacency, sources[:, company_slice])
                count = (adjacency > 0).sum(-1, keepdim=True, dtype=torch.float32)
                scale = count.clamp_min(1).rsqrt().to(dtype=aggregate.dtype)
                contribution = gains[relation_id] * proj(aggregate * scale)
                messages[:, company_slice] = messages[:, company_slice] + contribution
                continue
            if relation_flags is not None:
                adjacency = relation_flags[:, relation_id].to(dtype=sources.dtype)
                adjacency = adjacency * visible[:, None, :].to(dtype=sources.dtype)
                aggregate = torch.bmm(adjacency, sources)
                # Count neighbors, not their weights: doubling share quantities
                # must double this contribution at a fixed neighborhood size.
                count = (adjacency > 0).sum(-1, keepdim=True, dtype=torch.float32)
            else:
                assert sparse_ctx is not None and edge_sources is not None
                assert edge_visible is not None
                selected = edge_visible & (sparse_ctx.relation_ids == relation_id)
                edge_messages = edge_sources.masked_fill(~selected.unsqueeze(-1), 0)
                query_indices = sparse_ctx.query_tokens.unsqueeze(-1)
                aggregate = torch.zeros_like(sources).scatter_add(
                    1, query_indices.expand_as(edge_messages), edge_messages,
                )
                count = sources.new_zeros(
                    sources.shape[0], sources.shape[1], 1, dtype=torch.float32,
                ).scatter_add(1, query_indices, selected.unsqueeze(-1).float())

            scale = count.clamp_min(1).rsqrt().to(dtype=aggregate.dtype)
            messages = messages + gains[relation_id] * proj(aggregate * scale)

        # Neither padded/hidden sources nor padded/hidden recipients communicate.
        return tokens + messages.masked_fill(~visible.unsqueeze(-1), 0)


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
    _relation_scales: torch.Tensor
    _static_company_relations: torch.Tensor
    _static_attention_relations: torch.Tensor
    _corp_ids: torch.Tensor
    _bid_offset_dollar_norm: torch.Tensor
    _dividend_amounts: torch.Tensor
    _par_prices: torch.Tensor
    _acq_price_offsets: torch.Tensor

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        np_ = cfg.num_players
        self._num_tokens = int(cfg.num_tokens)
        num_companies = int(GameConstants.NUM_COMPANIES)
        num_corps = int(GameConstants.NUM_CORPS)
        num_fixed_tokens = self._num_tokens - np_

        # --- Token index bookkeeping ---
        # Buffer layout (matches core/token_data_v3.pyx::_fill_buffer):
        #   info: market_info (slot prices + per-space availability),
        #     companies×36 (is_selected + static data + CoO-adjusted income +
        #     at_*/owner_* groups), FI, global_info (decision phase + CoO +
        #     end-card + cards-remaining + num_players)
        #   phase-specific: invest, auction, dividend, issue, par,
        #     acq_offer, acq_price_info
        #   corps×8, then players×N (trailing so padding for higher player
        #   counts is a no-op on the prefix).
        token_idx = 0
        self._market_info_idx = token_idx
        token_idx += 1
        self._company_slice = slice(token_idx, token_idx + num_companies)
        token_idx += num_companies
        self._fi_idx = token_idx
        token_idx += 1
        self._global_info_idx = token_idx
        token_idx += 1
        self._invest_idx = token_idx
        token_idx += 1
        self._auction_idx = token_idx
        token_idx += 1
        self._dividend_idx = token_idx
        token_idx += 1
        self._issue_idx = token_idx
        token_idx += 1
        self._par_idx = token_idx
        token_idx += 1
        self._acq_offer_idx = token_idx
        token_idx += 1
        self._acq_price_info_idx = token_idx
        token_idx += 1
        self._corp_slice = slice(token_idx, token_idx + num_corps)
        token_idx += num_corps
        if token_idx != num_fixed_tokens:
            raise AssertionError(
                f"token index layout produced {token_idx} fixed tokens, "
                f"expected {num_fixed_tokens}"
            )
        self._player_slice = slice(token_idx, token_idx + np_)

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
        # The engine-side buffer is rectangular at ``TOKEN_DIM`` so
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
            int(TokenWidth.TW_GLOBAL_INFO) - self._token_feature_start,
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
        self.corp_id_embed = nn.Embedding(num_corps, d)
        # Per-type additive embedding for every token. Added
        # post-projection in ``_project_tokens`` so the trunk still sees a
        # type-distinct vector even when a token's feature slice is all-zero
        # (e.g. the DIVIDEND context token outside DIVIDENDS, or the owned-
        # company field of a player with no companies). Without this, zero
        # features + zero-initialized Linear biases collapse the token to the
        # zero vector on day one, and the only path to type discrimination is
        # an indirect gradient through the Linear's bias. Broadcast across
        # all instances of a type (36 companies, 8 corps, N players).
        self.type_embeds = nn.Embedding(len(_TokenType), d)
        # Static ``(cfg.num_tokens,)`` type-id lookup. Built once here so
        # ``_project_tokens`` can do a single indexed gather against
        # ``type_embeds``. Registered as a buffer so ``.to(device)`` carries
        # it along. Must match the concat order inside ``_project_tokens``.
        type_ids = torch.empty(self._num_tokens, dtype=torch.long)
        type_ids[self._market_info_idx] = int(_TokenType.MARKET_INFO)
        type_ids[self._company_slice] = int(_TokenType.COMPANY)
        type_ids[self._fi_idx] = int(_TokenType.FI)
        type_ids[self._global_info_idx] = int(_TokenType.GLOBAL_INFO)
        type_ids[self._invest_idx] = int(_TokenType.INVEST)
        type_ids[self._auction_idx] = int(_TokenType.AUCTION)
        type_ids[self._dividend_idx] = int(_TokenType.DIVIDEND)
        type_ids[self._issue_idx] = int(_TokenType.ISSUE)
        type_ids[self._par_idx] = int(_TokenType.PAR)
        type_ids[self._acq_offer_idx] = int(_TokenType.ACQ_OFFER)
        type_ids[self._acq_price_info_idx] = int(_TokenType.ACQ_PRICE)
        type_ids[self._corp_slice] = int(_TokenType.CORP)
        type_ids[self._player_slice] = int(_TokenType.PLAYER)
        self.register_buffer("_type_ids", type_ids, persistent=False)

        self.register_buffer(
            "_corp_ids", torch.arange(num_corps, dtype=torch.long), persistent=False,
        )
        # Per-slot dollar offset in /COMPANY_PRICE_DIVISOR units; added to the
        # active company's normalized face_value at runtime to form the actual
        # candidate-bid price channel for the BID MLP.
        bid_offset_dollar_norm = (
            torch.arange(int(AUCTION_CAP), dtype=torch.float32).view(1, int(AUCTION_CAP), 1)
            / float(PY_COMPANY_PRICE_DIVISOR)
        )
        self.register_buffer(
            "_bid_offset_dollar_norm",
            bid_offset_dollar_norm,
            persistent=False,
        )
        # All monetary dividend features use /CASH_DIVISOR. Multiplying these
        # amounts by raw share counts gives payouts in the same units as cash.
        num_dividends = _phase_action_size(DecisionPhase.DPHASE_DIVIDENDS)
        self.register_buffer(
            "_dividend_amounts",
            torch.arange(num_dividends, dtype=torch.float32) / float(PY_CASH_DIVISOR),
            persistent=False,
        )
        # PAR prices share /CASH_DIVISOR units with the raw float previews.
        num_par_prices = _phase_action_size(DecisionPhase.DPHASE_PAR)
        self.register_buffer(
            "_par_prices",
            torch.tensor(ALL_PAR_PRICES, dtype=torch.float32) / float(PY_CASH_DIVISOR),
            persistent=False,
        )
        # Acquisition prices and buyer cash use /CASH_DIVISOR units.
        num_acq_prices = _phase_action_size(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)
        self.register_buffer(
            "_acq_price_offsets",
            torch.arange(num_acq_prices, dtype=torch.float32) / float(PY_CASH_DIVISOR),
            persistent=False,
        )

        # Explicit relation messages enrich projected tokens before attention.
        self.relation_input_mixing = RelationInputMixing(d)
        # The engine stores each synergy pair once for income accounting.
        # Attention/messages need both directions, independent of current owner.
        directed_synergies = torch.tensor([
            [company.get_synergy_with(other) for other in range(num_companies)]
            for company in COMPANIES
        ])
        synergy_bonus = torch.maximum(directed_synergies, directed_synergies.T).float()
        static_company = torch.stack([
            (synergy_bonus > 0).float(),
            synergy_bonus / float(PY_COMPANY_SYNERGY_DIVISOR),
        ])
        self.register_buffer("_static_company_relations", static_company, persistent=False)
        static_attention = torch.zeros(static_company.shape[0], self._num_tokens, self._num_tokens)
        static_attention[:, self._company_slice, self._company_slice] = static_company
        self.register_buffer("_static_attention_relations", static_attention, persistent=False)

        # --- Transformer trunk ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d,
                cfg.num_heads,
                _ffn_hidden_dim(cfg),
            )
            for _ in range(cfg.num_layers)
        ])
        self.register_buffer(
            "_relation_scales", torch.tensor(ATTENTION_RELATION_SCALES), persistent=False,
        )
        self.relation_bias_mult = nn.Parameter(torch.zeros(
            cfg.num_layers,
            cfg.num_heads,
            _NUM_MODEL_RELATIONS,
        ))
        self.final_norm = nn.RMSNorm(d)

        # --- INVEST candidate MLP readouts ---
        # Hidden width follows the trunk. All three scorers participate in
        # the same policy softmax.
        self.invest_auction_head = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.invest_trade_head = nn.Sequential(
            nn.Linear(3 * d + 1, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 2),
        )
        self.invest_pass_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )

        # --- CLOSING candidate MLP readouts (no phase context token) ---
        self.closing_company_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.closing_pass_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )

        # --- BID: one price-selection MLP, including leave-auction ---
        self.bid_head = nn.Sequential(
            nn.Linear(3 * d + 2 * int(AUCTION_CAP), d),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, _phase_action_size(DecisionPhase.DPHASE_BID)),
        )

        # --- IPO corporation selection and PAR price selection ---
        self.ipo_corp_head = nn.Sequential(
            nn.Linear(4 * d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.ipo_pass_head = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.par_head = nn.Sequential(
            nn.Linear(4 * d + 5 * num_par_prices, d),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, num_par_prices),
        )

        # --- DIVIDENDS: zero payout is a candidate, not a separate pass ---
        self.dividend_head = nn.Sequential(
            nn.Linear(3 * d + 9 * num_dividends, d),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, num_dividends),
        )

        # --- Acquisition: candidate selection followed by price selection ---
        self.acq_corp_head = nn.Sequential(
            nn.Linear(2 * d + 1, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.acq_pass_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.acq_company_head = nn.Sequential(
            nn.Linear(3 * d + 1, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.acq_price_head = nn.Sequential(
            nn.Linear(4 * d + 3 * num_acq_prices, d),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, num_acq_prices),
        )

        # --- ISSUE: jointly score pass and issuing one share ---
        self.issue_head = nn.Sequential(
            nn.Linear(3 * d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 2),
        )

        # --- ACQ_OFFER: jointly score rejecting and accepting the offer ---
        self.acq_offer_head = nn.Sequential(
            nn.Linear(4 * d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 2),
        )

        # --- Value head (applied per player token) ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

        self._validate_policy_layout()
        self._init_weights()

    @staticmethod
    def _match_dtype_device(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Move small buffers/raw slices to the current runtime dtype and device."""
        return tensor.to(device=ref.device, dtype=ref.dtype)

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
        """Project global info, including the decision-phase one-hot."""
        return self.global_info_proj(
            x[
                :,
                self._global_info_idx,
                self._token_feature_start:int(TokenWidth.TW_GLOBAL_INFO),
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

    def _normalize_dense_relations(
        self, relations: torch.Tensor, ref: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize raw uint8 values once for all attention layers/input messages."""
        # Match sparse normalization: round the product, not the reciprocal,
        # when entering bf16/fp16 (notably, 5 * bf16(1/7) != bf16(5/7)).
        weights = relations.to(dtype=self._relation_scales.dtype) * self._relation_scales.view(1, -1, 1, 1)
        return weights.to(dtype=ref.dtype)

    def _relation_attention_bias(
        self,
        relation_flags: torch.Tensor,
        layer_idx: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Combine relation planes into an SDPA additive bias for one layer.

        ``relation_flags`` contains normalized ``(B, R, N, N)`` weights. The
        learned multipliers for layer ``layer_idx`` are ``(H, R)``, producing
        ``(B, H, N, N)`` so the
        result lines up with SDPA attention weights.
        """
        relation_mult = self._match_dtype_device(
            self.relation_bias_mult[layer_idx],
            ref,
        )
        dynamic_bias = torch.einsum(
            "brij,hr->bhij", relation_flags, relation_mult[:, :NUM_ATTENTION_RELATIONS],
        )
        return dynamic_bias + self._static_relation_attention_bias(layer_idx, ref)

    def _static_relation_attention_bias(self, layer_idx: int, ref: torch.Tensor) -> torch.Tensor:
        """Static company relations broadcast over the batch, with no IPC data."""
        return torch.einsum(
            "rij,hr->hij",
            self._match_dtype_device(self._static_attention_relations, ref),
            self._match_dtype_device(self.relation_bias_mult[layer_idx, :, NUM_ATTENTION_RELATIONS:], ref),
        )

    def _prepare_sparse_relation_context(
        self,
        relation_coords: torch.Tensor,
    ) -> _SparseRelationContext:
        """Precompute per-forward sparse relation indices shared by all layers."""
        batch_size = relation_coords.shape[0]
        num_heads = self.cfg.num_heads
        num_tokens = self._num_tokens

        coords = relation_coords.to(dtype=torch.long)
        relation_ids = coords[..., 0]
        query_tokens = coords[..., 1]
        key_tokens = coords[..., 2]
        valid_edges = relation_coords[..., 3] > 0
        edge_weights = (
            relation_coords[..., 3].to(dtype=self._relation_scales.dtype)
            * self._relation_scales[relation_ids]
        )

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
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            flat_indices=flat_indices,
            valid_edges=valid_edges,
            edge_weights=edge_weights,
        )

    def _sparse_relation_attention_bias(
        self,
        relation_ctx: _SparseRelationContext,
        layer_idx: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """Build SDPA relation bias from sparse relation records.

        The result is still dense ``(B, H, N, N)`` because SDPA consumes a dense
        additive attention mask, but the expensive relation-type dimension is
        skipped: only emitted sparse edges gather learned per-head multipliers
        and scatter-add into the final bias tensor.
        """
        batch_size = relation_ctx.relation_ids.shape[0]
        num_heads = self.cfg.num_heads
        num_tokens = self._num_tokens
        relation_mult = self._match_dtype_device(
            self.relation_bias_mult[layer_idx],
            ref,
        )
        edge_values = F.embedding(
            relation_ctx.relation_ids,
            relation_mult.transpose(0, 1),
        ).transpose(1, 2)
        edge_values = edge_values * relation_ctx.edge_weights[:, None, :].to(
            dtype=edge_values.dtype,
        )

        bias = ref.new_zeros(batch_size, num_heads, num_tokens, num_tokens)
        bias.reshape(-1).scatter_add_(
            0,
            relation_ctx.flat_indices.reshape(-1),
            edge_values.reshape(-1),
        )
        return bias + self._static_relation_attention_bias(layer_idx, ref)

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

        block_widths = [0] * NUM_PHASES
        block_widths[int(DecisionPhase.DPHASE_INVEST)] = (
            1
            + num_companies
            + num_corps * 2
        )
        block_widths[int(DecisionPhase.DPHASE_BID)] = (
            1
            + int(AUCTION_CAP)
        )
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_CORP)] = (
            1
            + num_corps
        )
        if _phase_action_size(DecisionPhase.DPHASE_ACQ_OFFER) != 2:
            raise AssertionError(
                "ACQ_OFFER policy readout has two output logits; "
                f"PHASE_ACTION_SIZES reports {_phase_action_size(DecisionPhase.DPHASE_ACQ_OFFER)}"
            )
        block_widths[int(DecisionPhase.DPHASE_ACQ_OFFER)] = 2
        block_widths[int(DecisionPhase.DPHASE_CLOSING)] = (
            1
            + num_companies
        )
        block_widths[int(DecisionPhase.DPHASE_DIVIDENDS)] = (
            int(self._dividend_amounts.shape[0])
        )
        if _phase_action_size(DecisionPhase.DPHASE_ISSUE) != 2:
            raise AssertionError(
                "ISSUE policy readout has two output logits; "
                f"PHASE_ACTION_SIZES reports {_phase_action_size(DecisionPhase.DPHASE_ISSUE)}"
            )
        block_widths[int(DecisionPhase.DPHASE_ISSUE)] = 2
        block_widths[int(DecisionPhase.DPHASE_IPO)] = (
            1
            + num_corps
        )
        num_par_prices = int(self._par_prices.shape[0])
        par_feature_width = int(TokenWidth.TW_PAR) - self._token_feature_start
        if par_feature_width != num_par_prices * 3:
            raise AssertionError(
                f"PAR token feature width {par_feature_width} must equal "
                f"3 fields * {num_par_prices} par prices"
            )
        block_widths[int(DecisionPhase.DPHASE_PAR)] = num_par_prices
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_COMPANY)] = num_companies
        block_widths[int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)] = (
            int(self._acq_price_offsets.shape[0])
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
        from projection because the same relations supply input messages and
        attention biases. Other entity IDs, active-entity refs, and phase refs are
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

    def _actor_corp_shares(self, ctx: _PolicyContext) -> torch.Tensor:
        """Current actor's per-corp holdings, retaining /SHARE_DIVISOR units."""
        player_shares = self._match_dtype_device(
            ctx.raw_tokens[
                :, self._player_slice,
                _PLAYER_SHARES_START:_PLAYER_SHARES_START + ctx.corp_tokens.shape[1],
            ],
            ctx.tokens,
        )
        return self._active_token(ctx.raw_tokens, self._player_slice, player_shares)

    def _invest_logits(
        self,
        ctx: _PolicyContext,
    ) -> torch.Tensor:
        """Build the Invest block: pass, company auction, interleaved buy/sell."""
        invest = ctx.tokens[:, self._invest_idx]
        pass_logit = self.invest_pass_head(
            torch.cat([ctx.active_player, invest], dim=-1)
        )
        num_companies = ctx.company_tokens.shape[1]
        num_corps = ctx.corp_tokens.shape[1]
        auction_inputs = torch.cat(
            [
                ctx.active_player[:, None, :].expand(-1, num_companies, -1),
                ctx.company_tokens,
                invest[:, None, :].expand(-1, num_companies, -1),
            ],
            dim=-1,
        )
        auction_company = self.invest_auction_head(auction_inputs).squeeze(-1)

        actor_shares = self._actor_corp_shares(ctx)
        trade_inputs = torch.cat(
            [
                ctx.active_player[:, None, :].expand(-1, num_corps, -1),
                ctx.corp_tokens,
                invest[:, None, :].expand(-1, num_corps, -1),
                actor_shares.unsqueeze(-1),
            ],
            dim=-1,
        )
        # Last dimension is [buy, sell], matching sparse INVEST ids 37+2*c.
        corp_trade = self.invest_trade_head(trade_inputs).flatten(1)
        return torch.cat(
            [pass_logit, auction_company, corp_trade],
            dim=1,
        )

    def _acq_select_corp_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Pass or select a buying corp, with the actor's candidate holding."""
        num_corps = ctx.corp_tokens.shape[1]
        corp_inputs = torch.cat(
            [ctx.active_player[:, None, :].expand(-1, num_corps, -1),
             ctx.corp_tokens, self._actor_corp_shares(ctx).unsqueeze(-1)],
            dim=-1,
        )
        return torch.cat(
            [self.acq_pass_head(ctx.active_player),
             self.acq_corp_head(corp_inputs).squeeze(-1)],
            dim=-1,
        )

    def _acq_select_company_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Score each target company, including its raw marginal synergy gain."""
        num_companies = ctx.company_tokens.shape[1]
        synergy = self._match_dtype_device(
            ctx.raw_tokens[:, self._company_slice, _COMPANY_ACQ_SYNERGY_OFFSET],
            ctx.tokens,
        )
        company_inputs = torch.cat(
            [ctx.active_player[:, None, :].expand(-1, num_companies, -1),
             ctx.active_corp[:, None, :].expand(-1, num_companies, -1),
             ctx.company_tokens, synergy.unsqueeze(-1)],
            dim=-1,
        )
        return self.acq_company_head(company_inputs).squeeze(-1)

    def _closing_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build CLOSING logits: pass plus one logit per company."""
        pass_logit = self.closing_pass_head(ctx.active_player)
        company_inputs = torch.cat(
            [
                ctx.active_player[:, None, :].expand(
                    -1, ctx.company_tokens.shape[1], -1,
                ),
                ctx.company_tokens,
            ],
            dim=-1,
        )
        company_logits = self.closing_company_head(company_inputs).squeeze(-1)
        return torch.cat([pass_logit, company_logits], dim=-1)

    def _ipo_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Pass on the active company, or select one corporation to float it."""
        par = ctx.tokens[:, self._par_idx]
        pass_logit = self.ipo_pass_head(torch.cat(
            [ctx.active_player, ctx.active_company, par], dim=-1,
        ))
        num_corps = ctx.corp_tokens.shape[1]
        corp_inputs = torch.cat(
            [
                ctx.active_player[:, None, :].expand(-1, num_corps, -1),
                ctx.active_company[:, None, :].expand(-1, num_corps, -1),
                ctx.corp_tokens,
                par[:, None, :].expand(-1, num_corps, -1),
            ],
            dim=-1,
        )
        corp_logits = self.ipo_corp_head(corp_inputs).squeeze(-1)
        return torch.cat([pass_logit, corp_logits], dim=-1)

    def _bid_price_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per offset: [bid price, cash left if won], both in /price-divisor units.

        Select and subtract in fp32 even under autocast, before matching the
        trunk dtype. Player cash arrives in /CASH_DIVISOR units; company face
        value arrives in /COMPANY_PRICE_DIVISOR units. Negative balances are
        retained for unaffordable candidates, whose logits are legally masked.
        """
        company_rows = x[:, self._company_slice].float()
        face_offset = _TOKEN_FEATURE_START + _COMPANY_FACE_VALUE_FEATURE_OFFSET
        face_value = (
            company_rows[:, :, self._is_selected_offset]
            * company_rows[:, :, face_offset]
        ).sum(dim=1, keepdim=True)
        player_rows = x[:, self._player_slice].float()
        cash = (
            player_rows[:, :, self._is_selected_offset]
            * player_rows[:, :, _PLAYER_CASH_OFFSET]
        ).sum(dim=1, keepdim=True)
        cash = cash * (float(PY_CASH_DIVISOR) / float(PY_COMPANY_PRICE_DIVISOR))
        prices = face_value.unsqueeze(-1) + self._bid_offset_dollar_norm.float()
        remaining_cash = cash.unsqueeze(-1) - prices
        return torch.cat([prices, remaining_cash], dim=-1)

    def _bid_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Jointly score leave-auction then bids face_value + offsets 0..14."""
        price_features = self._match_dtype_device(
            self._bid_price_features(ctx.raw_tokens), ctx.tokens,
        ).flatten(1)
        return self.bid_head(torch.cat(
            [
                ctx.active_player, ctx.active_company,
                ctx.tokens[:, self._auction_idx], price_features,
            ],
            dim=-1,
        ))

    def _dividend_outcome_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per amount: amount, payout, cash left, actor/others/bank, move, price, worth.

        Monetary features use /CASH_DIVISOR. Share counts are decoded from
        /SHARE_DIVISOR before multiplication; arithmetic stays fp32 under
        autocast. Cash left is after payment, before any bankruptcy resolution.
        Movement and new dollar prices come from the engine's resolved
        destinations. Net-worth impact includes payout and repricing all
        shares held by the actor, including complete loss on bankruptcy.
        """
        corp_rows = x[:, self._corp_slice].float()
        selected_corp = corp_rows[:, :, self._is_selected_offset]
        issued = (selected_corp * corp_rows[:, :, _CORP_ISSUED_OFFSET]).sum(1, keepdim=True)
        issued = issued * float(PY_SHARE_DIVISOR)
        bank = (selected_corp * corp_rows[:, :, _CORP_BANK_SHARES_OFFSET]).sum(1, keepdim=True)
        bank = bank * float(PY_SHARE_DIVISOR)
        cash = (selected_corp * corp_rows[:, :, _CORP_CASH_OFFSET]).sum(1, keepdim=True)

        player_rows = x[:, self._player_slice].float()
        shares = player_rows[
            :, :, _PLAYER_SHARES_START:_PLAYER_SHARES_START + corp_rows.shape[1],
        ]
        holdings = (shares * selected_corp[:, None, :]).sum(-1) * float(PY_SHARE_DIVISOR)
        holdings = holdings * player_rows[:, :, 0]  # Exclude padded player rows.
        actor = (holdings * player_rows[:, :, self._is_selected_offset]).sum(1, keepdim=True)
        others = holdings.sum(1, keepdim=True) - actor

        amounts = self._dividend_amounts.float().unsqueeze(0).expand(x.shape[0], -1)
        total = amounts * issued
        previews = x[
            :, self._dividend_idx, _TOKEN_FEATURE_START:int(TokenWidth.TW_DIVIDEND),
        ].float()
        impacts, prices = previews.chunk(2, dim=-1)
        old_price = (
            selected_corp * corp_rows[:, :, _CORP_SHARE_PRICE_OFFSET]
        ).sum(1, keepdim=True) * (float(PY_SHARE_PRICE_DIVISOR) / float(PY_CASH_DIVISOR))
        net_worth = actor * (amounts + prices - old_price)
        return torch.stack(
            [amounts, total, cash - total, amounts * actor, amounts * others,
             amounts * bank, impacts, prices, net_worth],
            dim=-1,
        )

    def _dividend_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Jointly score dividend amounts 0..25 in action-id order."""
        outcomes = self._match_dtype_device(
            self._dividend_outcome_features(ctx.raw_tokens), ctx.tokens,
        ).flatten(1)
        return self.dividend_head(torch.cat(
            [ctx.active_player, ctx.active_corp,
             ctx.tokens[:, self._dividend_idx], outcomes],
            dim=-1,
        ))

    def _issue_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build ISSUE logits: pass/no-issue plus issue one share."""
        return self.issue_head(torch.cat(
            [ctx.active_player, ctx.active_corp, ctx.tokens[:, self._issue_idx]],
            dim=-1,
        ))

    def _acq_offer_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Build ACQ_OFFER logits: pass/reject plus accept offer."""
        return self.acq_offer_head(torch.cat(
            [
                ctx.active_player,
                ctx.active_corp,
                ctx.active_company,
                ctx.tokens[:, self._acq_offer_idx],
            ],
            dim=-1,
        ))

    def _par_outcome_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per price: price, payment, cash left, corp cash, issued shares.

        Monetary features use /CASH_DIVISOR; issued shares retain the engine's
        /FLOAT_SHARES_MAX normalization. Use the engine's preview directly,
        including its zero entries for tier-ineligible prices. The legal mask
        handles tier, market availability, and affordability. Subtraction stays
        in fp32 under autocast, as in the BID feature path.
        """
        num_prices = self._par_prices.shape[0]
        preview = x[
            :, self._par_idx, _TOKEN_FEATURE_START:int(TokenWidth.TW_PAR),
        ].float().reshape(x.shape[0], num_prices, 3)
        player_rows = x[:, self._player_slice].float()
        cash = (
            player_rows[:, :, self._is_selected_offset]
            * player_rows[:, :, _PLAYER_CASH_OFFSET]
        ).sum(dim=1, keepdim=True)
        prices = self._par_prices.float().unsqueeze(0).expand(x.shape[0], -1)
        return torch.stack(
            [prices, preview[:, :, 0], cash - preview[:, :, 0],
             preview[:, :, 1], preview[:, :, 2]],
            dim=-1,
        )

    def _par_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Jointly score all par prices in engine table order, with no pass."""
        outcomes = self._match_dtype_device(
            self._par_outcome_features(ctx.raw_tokens), ctx.tokens,
        ).flatten(1)
        return self.par_head(torch.cat(
            [ctx.active_player, ctx.active_corp, ctx.active_company,
             ctx.tokens[:, self._par_idx], outcomes],
            dim=-1,
        ))

    def _acq_price_features(self, x: torch.Tensor) -> torch.Tensor:
        """Per offset: price, buyer cash, seller balance, all /CASH_DIVISOR.

        Only non-FI companies reach this phase. Compute in fp32, translating
        company low price from /COMPANY_PRICE_DIVISOR before subtraction.
        Negative cash remains visible for unaffordable, legally masked slots.
        A corporation seller's balance includes existing acquisition proceeds
        plus this sale; proceeds are locked until the acquisition phase ends.
        Ownership tails select the seller directly, including non-actor players.
        """
        company_rows = x[:, self._company_slice].float()
        low_offset = _TOKEN_FEATURE_START + _COMPANY_LOW_PRICE_FEATURE_OFFSET
        low = (
            company_rows[:, :, self._is_selected_offset] * company_rows[:, :, low_offset]
        ).sum(1, keepdim=True)
        low = low * (float(PY_COMPANY_PRICE_DIVISOR) / float(PY_CASH_DIVISOR))
        corp_rows = x[:, self._corp_slice].float()
        cash = (
            corp_rows[:, :, self._is_selected_offset] * corp_rows[:, :, _CORP_CASH_OFFSET]
        ).sum(1, keepdim=True)
        selected_company = company_rows[:, :, self._is_selected_offset].unsqueeze(-1)
        owner_corp_start = self._company_rel_tail_start
        owner_player_start = owner_corp_start + corp_rows.shape[1]
        seller_corps = (
            selected_company * company_rows[:, :, owner_corp_start:owner_player_start]
        ).sum(1)
        player_rows = x[:, self._player_slice].float()
        seller_players = (
            selected_company * company_rows[
                :, :, owner_player_start:owner_player_start + player_rows.shape[1],
            ]
        ).sum(1)
        seller_balance = (
            seller_corps * (corp_rows[:, :, _CORP_CASH_OFFSET]
                            + corp_rows[:, :, _CORP_ACQ_PROCEEDS_OFFSET])
        ).sum(1, keepdim=True)
        seller_balance = seller_balance + (
            seller_players * player_rows[:, :, _PLAYER_CASH_OFFSET]
        ).sum(1, keepdim=True)
        prices = low + self._acq_price_offsets.float().unsqueeze(0)
        return torch.stack([prices, cash - prices, seller_balance + prices], dim=-1)

    def _acq_price_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """Jointly score offsets 0..50; no pass or special FI price slot."""
        outcomes = self._match_dtype_device(
            self._acq_price_features(ctx.raw_tokens), ctx.tokens,
        ).flatten(1)
        return self.acq_price_head(torch.cat(
            [ctx.active_player, ctx.active_corp, ctx.active_company,
             ctx.tokens[:, self._acq_price_info_idx], outcomes],
            dim=-1,
        ))

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
        invest = self._invest_logits(ctx)                                       # (B, 53)
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
                invest,                                            # INVEST           (53)
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
                num_tokens, num_tokens)`` uint8 directed relation planes
                or sparse eval-server coordinates ``(batch,
                MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH)``
                uint8. Dense rows are attention queries and columns are
                attention keys; sparse rows are ``(relation_id, query, key, value)``
                records padded with ``(0, 0, 0, 0)``. Binary values are 0/1;
                share counts are raw integers, normalized in the model by 7.

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
        expected_x_shape = (x.shape[0], self._num_tokens, self.cfg.token_dim)
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
            self._num_tokens,
            self._num_tokens,
        )
        expected_sparse_rel_shape = (
            x.shape[0],
            MAX_ATTENTION_RELATION_EDGES,
            ATTENTION_RELATION_COORD_WIDTH,
        )
        if tuple(relations.shape) == expected_dense_rel_shape:
            if relations.dtype != torch.uint8:
                raise AssertionError(
                    f"dense relation planes must be uint8; got {relations.dtype}"
                )
            relation_flags = self._normalize_dense_relations(relations, tokens)
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
        tokens = self.relation_input_mixing(
            tokens, attn_mask[:, 0, 0, :], relation_flags, sparse_relation_ctx,
            self._static_company_relations, self._company_slice,
        )

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
            tokens = block(tokens, attn_mask, relation_bias)
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

        Used as a TB diagnostic to detect pass/action logit-scale drift within
        phases that have a dedicated pass/no-op slot.

        Returns a ``(2 * len(PHASES_WITH_PASS_SLOT),)`` tensor packed as
        ``[pass_abs_p0, action_abs_p0, pass_abs_p1, action_abs_p1, ...]`` in
        ``PHASES_WITH_PASS_SLOT`` order. Phases with no rows in the batch (or
        all-illegal pass slots) return 0 for both stats — the caller is
        expected to filter via per-phase row counts.
        """
        abs_logits = policy_logits.detach().abs()
        stats: list[torch.Tensor] = []
        for phase in PHASES_WITH_PASS_SLOT:
            offset = PHASE_OFFSETS[phase]
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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT/LLaMA-style trunc-normal initialization.

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
        # Per-type additive embeddings: same small-random init.
        nn.init.trunc_normal_(self.type_embeds.weight, std=0.02)
        # Relation attention starts behavior-preserving; training can learn
        # positive or negative head/layer-specific biases from zero.
        nn.init.zeros_(self.relation_bias_mult)
        nn.init.constant_(self.relation_input_mixing.relation_gains, 0.1)


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
          f"layers={cfg.num_layers}, d_ff={_ffn_hidden_dim(cfg)}")
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
    type_params = model.type_embeds.weight.numel()
    trunk_params = (
        sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.final_norm.parameters())
    )
    policy_modules: list[nn.Module] = [
        model.invest_auction_head,
        model.invest_trade_head,
        model.invest_pass_head,
        model.closing_company_head, model.closing_pass_head,
        model.acq_company_head, model.acq_corp_head, model.acq_pass_head,
        model.ipo_corp_head, model.ipo_pass_head,
        model.bid_head,
        model.dividend_head,
        model.issue_head,
        model.acq_offer_head,
        model.acq_price_head,
        model.par_head,
    ]
    policy_params = sum(sum(p.numel() for p in m.parameters()) for m in policy_modules)
    value_params = sum(p.numel() for p in model.value_head.parameters())

    print("Parameter breakdown:")
    for name, count in [
        ("Input projections", proj_params),
        ("Corp ID embeds", corp_id_params),
        ("Type embeds", type_params),
        ("Relation input mixing", count_parameters(model.relation_input_mixing)),
        ("Relation attention bias", model.relation_bias_mult.numel()),
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
