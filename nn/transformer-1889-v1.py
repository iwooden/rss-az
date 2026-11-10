"""Transformer v1 for Shikoku 1889 AlphaZero training (1889/model-design.md
§12, 1889/v1-design.md §7).

v0 (``nn/transformer-1889.py``) on token-data-1889.md's layout 2, with three
changes:

- the projections take layout 2's widths, and heads read its new per-choice
  slots raw;
- the trunk runs on the 80 rows that are not CANDIDATE rows. TILE and ER_BONUS
  are scored by ``candidate_head`` from each candidate's own row, the selected
  hex's final row, and the final rows of the hexes across the candidate's
  exits. The input still carries all 95 rows;
- the attention bias takes 25 dynamic relations, ids 0-20 and 25-28, and the 10
  static ones. Relations 21-24 join candidate rows, which the trunk does not
  hold, so they have no channel.

The rest is v0's: per-type projections, with one tile projection shared by HEX
and CANDIDATE rows, plus identity and type embeddings feed a pre-RMSNorm trunk
whose attention takes an additive relation bias per layer and head. Policy heads
read the context ``[GLOBAL, actor]``, with the decision corp wherever the head's
phases have one, the rows the choice names, and raw per-choice slots, and fill
the 19 phase blocks of the unified 228 slots. Values are read from the player
rows.

Relations arrive as flat ``(N, 5)`` int16 records (``flatten_relation_records``),
or as dense ``(B, R, T, T)`` uint8 planes in tests, both indexed by the input's
95 rows. Every layer builds one dense bias tensor over the trunk's rows from
them, since SDPA needs one.

The file stands alone, as v3's does beside v2's and v0's beside v3's
(1889/pipeline-refactor.md §5): v0's file can go once v1 has beaten it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

import g1889.data.tables as D
from g1889.relations import RelationId, StaticRelationId, static_relation_planes
from g1889.token_data import TileBlock, TokenDataSize, TokenRow, TokenWidth2, get_num_tokens, get_token_widths
from games.g1889 import POLICY_LAYOUT
from nn.model_contract import FLAT_RELATION_RECORD_WIDTH, ModelInputSpec

if TYPE_CHECKING:
    from train.config import TrainingConfig

INPUT_LAYOUT_VERSION = 2

_GELU_APPROX = "tanh"

# Relation channels, in id order. The consumed ids are v1's own, not RelationId's:
# relations 21-24 join candidate rows, which the trunk does not hold, and later
# steps append relations past 28, which carry no weight here.
_RELATION_IDS = (
    *range(int(RelationId.REL_CANDIDATE_AT)),
    *range(int(RelationId.REL_SOLD_THIS_SR), int(RelationId.REL_ROUTE_OF) + 1),
)
_NUM_DYNAMIC_RELATIONS = len(_RELATION_IDS)
_NUM_RELATION_IDS = _RELATION_IDS[-1] + 1  # the planes a dense input must hold
_NUM_STATIC_RELATIONS = int(StaticRelationId.NUM_STATIC_RELATIONS)
_NUM_MODEL_RELATIONS = _NUM_DYNAMIC_RELATIONS + _NUM_STATIC_RELATIONS

# Rows. Players come last, so every other index is the same at every capacity.
_NUM_CORPS = int(D.NUM_CORPS)
_NUM_PRIVATES = int(D.NUM_PRIVATES)
_NUM_TRAINS = int(D.NUM_TRAIN_TYPES)
_NUM_HEXES = int(D.NUM_HEXES)
_NUM_CANDIDATES = int(D.MAX_LAY_CANDIDATES)
_NUM_ER_CANDIDATES = int(D.NUM_ER_CANDIDATES)
_NUM_SIDES = int(D.NUM_SIDES)
_NUM_PAIRS = int(D.NUM_PAIRS)
_GLOBAL = int(TokenRow.ROW_GLOBAL)
_PRICE = int(TokenRow.ROW_PRICE)
_DIVIDEND = int(TokenRow.ROW_DIVIDEND)
_CORPS = slice(int(TokenRow.ROW_CORP), int(TokenRow.ROW_PRIVATE))
_PRIVATES = slice(int(TokenRow.ROW_PRIVATE), int(TokenRow.ROW_TRAIN))
_TRAINS = slice(int(TokenRow.ROW_TRAIN), int(TokenRow.ROW_HEX))
_HEXES = slice(int(TokenRow.ROW_HEX), int(TokenRow.ROW_CANDIDATE))
_CANDIDATES = slice(int(TokenRow.ROW_CANDIDATE), int(TokenRow.ROW_PLAYER))
_PLAYERS = slice(int(TokenRow.ROW_PLAYER), None)
_PRIVATE_DR = int(D.PRIVATE_DR)
_PRIVATE_MF = int(D.PRIVATE_MF)

# Slot 0 of every row is the key mask, and the selections are is_selected slots.
_KEY_MASK = 0
_CORP_DECISION = 1
_CORP_SELLER = 2
_CORP_SOLD = 3
_PRIVATE_SELECTED = 1
_PRIVATE_TOPMOST = 2
_TRAIN_SELECTED = 1
_HEX_SELECTED = 1
_PLAYER_ACTOR = 1

# The projections' slots: a row's own, and around the shared tile block.
_FIRST_FEATURE = 1
_TILE_SLOTS = slice(int(TileBlock.TILE_BLOCK), int(TileBlock.TILE_BLOCK) + int(TileBlock.TW_TILE_BLOCK))
_HEX_OWN_SLOTS = [_FIRST_FEATURE, *range(_TILE_SLOTS.stop, int(TokenWidth2.TW2_HEX))]
_CANDIDATE_OWN_SLOTS = [_FIRST_FEATURE, *range(_TILE_SLOTS.stop, int(TokenWidth2.TW2_CANDIDATE))]
# The tile block's path mask, a slot per endpoint pair in PAIRS order (token-data-1889.md).
_TILE_PATH_SLOTS = slice(_TILE_SLOTS.start + 41, _TILE_SLOTS.start + 41 + _NUM_PAIRS)

# The heads' raw per-choice slots, in layout 2's numbers (model-design §12.2).
# Extraction exports the widths and the tile block, not the slots, so these
# follow token-data-1889.md's tables, as v0's follow layout 1's.
_PRIVATE_AUCTION_SLOTS = slice(7, 14)  # face, revenue, price; high, bids, minimum, the actor's
_PRIVATE_BUY_SLOTS = slice(16, 18)  # can_buy_private, can_exchange_dr
_PRICE_SLOTS = slice(5, 24)  # ¥ per id, the reference, the payer's funds
_PRICE_FLOAT_SLOTS = slice(24, 30)  # at PAR, per par index: shares of the float out once the actor buys
_CORP_STOCK_SLOTS = [
    *range(7, 26),  # prices, the market cell and zone, sale prices, sold out, cash, shares, tokens
    *range(50, 55),  # the SR's sales and purchases of it, the actor's largest sale
    *range(58, 61),  # the price after a payout, a withhold, and a sold-out SR
    *range(68, 72),  # the float: shares still to sell, their cost, the president and the actor can pay
    72,  # the president sold it this SR
]
_CORP_SALE_SLOTS = [*range(15, 20), 54, 55]
_CORP_BUY_TRAIN_SLOTS = [*range(15, 20), 52, 55]
_PLAYER_HOLDING_SLOTS = slice(11, 25)  # shares by corp, then sold this SR by corp
_HEX_TRACK_SLOTS = slice(92, 103)
_HEX_TOKEN_SLOTS = slice(103, 105)
_CANDIDATE_SLOTS = [1, *range(99, 109)]  # 108: the route max gained with the next train
_DIVIDEND_SLOTS = slice(1, 13)  # 7-12: the next train's price, both shortfalls, room, two route maxes
_TRAIN_DISCARD_SLOTS = slice(12, 13)
_TRAIN_BUY_SLOTS = [*range(2, 11), 14]  # 10: the route max a purchase adds; 14: it starts a phase
_TRAIN_SELECT_SLOTS = [10, 11, 13]  # 13: what the seller loses
_TRAIN_SELLER_SLOTS = [13]


def _num_slots(slots: slice | list[int]) -> int:
    return slots.stop - slots.start if isinstance(slots, slice) else len(slots)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round positive integer ``value`` up to the next ``multiple`` boundary."""
    return ((value + multiple - 1) // multiple) * multiple


def _mlp(n_in: int, d: int, n_out: int) -> nn.Sequential:
    """A head: one GELU hidden layer of the trunk's width."""
    return nn.Sequential(nn.Linear(n_in, d), nn.GELU(approximate=_GELU_APPROX), nn.Linear(d, n_out))


def _match(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Move small buffers/raw slices to the current runtime dtype and device."""
    return tensor.to(device=ref.device, dtype=ref.dtype)


def _selected_row(raw: torch.Tensor, tokens: torch.Tensor, rows: slice, slot: int) -> torch.Tensor:
    """The ``(B, d)`` final row an ``is_selected`` slot picks, zero when none does."""
    selector = _match(raw[:, rows, slot], tokens)
    return torch.bmm(selector.unsqueeze(1), tokens[:, rows]).squeeze(1)


def _selected_raw(raw: torch.Tensor, rows: slice, slot: int, slots: slice | list[int]) -> torch.Tensor:
    """``slots`` of the row an ``is_selected`` slot picks, in fp32 even under autocast."""
    type_rows = raw[:, rows]
    return (type_rows[:, :, slot:slot + 1] * type_rows[:, :, slots]).sum(1)


def _score_rows(head: nn.Module, context: torch.Tensor, rows: torch.Tensor, *raw: torch.Tensor) -> torch.Tensor:
    """A pointer head on each of ``rows``: the context, the row, and its raw slots."""
    inputs = [context[:, None, :].expand(-1, rows.shape[1], -1), rows, *[_match(r, rows) for r in raw]]
    return head(torch.cat(inputs, dim=-1))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """All dimensions parameterized. Defaults are 3-player with d_model=256."""

    # Model player-row capacity. In mixed training this is effective
    # max_players; each state still encodes its actual player count.
    num_players: int = 3  # 3-5 supported
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 15
    ff_mult: float = 3.0  # FFN inner dimension is rounded up to a multiple of 64.
    # Subtract the mean over real players from the tanh values, so each row
    # sums to zero like the terminal targets.
    zero_sum_values: bool = False

    layout_version: int = field(default=INPUT_LAYOUT_VERSION, init=False)
    token_dim: int = field(default=int(TokenDataSize.TOKEN_DIM_2), init=False)

    _num_tokens: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        assert isinstance(self.zero_sum_values, bool), (
            f"zero_sum_values must be bool, got {self.zero_sum_values!r}"
        )
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
        """Input row count: the fixed rows + player capacity."""
        return self._num_tokens

    @property
    def num_trunk_tokens(self) -> int:
        """The rows the trunk runs on: the input's but the candidates."""
        return self._num_tokens - _NUM_CANDIDATES


def _validate_layout(num_players: int) -> None:
    """Assert the row map this module slices matches ``get_token_widths``.

    The two are sources of truth for one buffer: a drift between them is
    invisible at runtime (the trunk just sees mis-sized rows), so check once at
    construction.
    """
    expected = (
        [int(TokenWidth2.TW2_GLOBAL), int(TokenWidth2.TW2_PRICE), int(TokenWidth2.TW2_DIVIDEND)]
        + [int(TokenWidth2.TW2_CORP)] * (_CORPS.stop - _CORPS.start)
        + [int(TokenWidth2.TW2_PRIVATE)] * (_PRIVATES.stop - _PRIVATES.start)
        + [int(TokenWidth2.TW2_TRAIN)] * (_TRAINS.stop - _TRAINS.start)
        + [int(TokenWidth2.TW2_HEX)] * (_HEXES.stop - _HEXES.start)
        + [int(TokenWidth2.TW2_CANDIDATE)] * (_CANDIDATES.stop - _CANDIDATES.start)
        + [int(TokenWidth2.TW2_PLAYER)] * num_players
    )
    actual = get_token_widths(num_players, INPUT_LAYOUT_VERSION).tolist()
    assert actual == expected, (
        f"token layout drift between nn/transformer-1889-v1.py and g1889/token_data.pyx "
        f"for {num_players}p: actual widths {actual} vs expected {expected}"
    )


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-RMSNorm block: RMSNorm -> MHSA -> residual, RMSNorm -> SwiGLU FFN -> residual.

    v0's block, which is v3's (see its notes on SDPA under ``torch.compile``)
    with the caller finishing the SDPA mask, relation bias and hidden keys
    together.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model {d_model} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_norm = nn.RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor, sdpa_mask: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        B, N, width = h.shape
        qkv = self.qkv_proj(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        # (3, B, heads, N, head_dim) so unbind(0) yields three
        # (B, heads, N, head_dim) tensors, packed as Inductor's SDPA expects.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).contiguous().unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask)
        # (B, heads, N, head_dim) -> (B, N, d)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, width)
        x = x + self.out_proj(attn_out)
        h = self.ffn_norm(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x + h


@dataclass(frozen=True)
class _DenseRelations:
    """A forward's dense planes, normalized once for all layers."""

    planes: torch.Tensor  # (B, 25, T', T'), over the trunk's rows


@dataclass(frozen=True)
class _FlatRelations:
    """A forward's flat records, normalized once for all layers."""

    relation_ids: torch.Tensor  # (N,)
    flat_indices: torch.Tensor  # (H * N,), into a (B, H, T', T') tensor
    weights: torch.Tensor  # (N,)


@dataclass(frozen=True)
class _PolicyContext:
    raw: torch.Tensor  # the input rows, fp32
    tokens: torch.Tensor  # the final rows, in the input's row order (``_encode``)
    context: torch.Tensor  # [g, a, c]
    actor_context: torch.Tensor  # [g, a], for the phases with no decision corp


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class G1889TransformerNet(nn.Module):
    """Transformer with actor-conditioned policy readouts and per-player values."""

    _type_ids: torch.Tensor
    _relation_scales: torch.Tensor
    _relation_channel: torch.Tensor
    _relation_ids: torch.Tensor
    _trunk_rows: torch.Tensor
    _row_map: torch.Tensor
    _static_relations: torch.Tensor
    _head_offsets: torch.Tensor
    _side_planes: torch.Tensor
    _pair_sides: torch.Tensor
    _track_rows: torch.Tensor
    _token_rows: torch.Tensor
    _discard_rows: torch.Tensor
    _trade_in_types: torch.Tensor
    _port_rows: torch.Tensor
    _port_masks: torch.Tensor

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        num_tokens = cfg.num_tokens
        trunk_tokens = cfg.num_trunk_tokens
        _validate_layout(cfg.num_players)

        # --- Input projections ---
        # Slot 0, the key mask, is not projected. HEX and CANDIDATE rows share
        # the biasless tile projection over their tile blocks (§3.2).
        self.global_proj = nn.Linear(int(TokenWidth2.TW2_GLOBAL) - _FIRST_FEATURE, d)
        self.price_proj = nn.Linear(int(TokenWidth2.TW2_PRICE) - _FIRST_FEATURE, d)
        self.dividend_proj = nn.Linear(int(TokenWidth2.TW2_DIVIDEND) - _FIRST_FEATURE, d)
        self.corp_proj = nn.Linear(int(TokenWidth2.TW2_CORP) - _FIRST_FEATURE, d)
        self.private_proj = nn.Linear(int(TokenWidth2.TW2_PRIVATE) - _FIRST_FEATURE, d)
        self.train_proj = nn.Linear(int(TokenWidth2.TW2_TRAIN) - _FIRST_FEATURE, d)
        self.tile_proj = nn.Linear(_num_slots(_TILE_SLOTS), d, bias=False)
        self.hex_proj = nn.Linear(_num_slots(_HEX_OWN_SLOTS), d)
        self.candidate_proj = nn.Linear(_num_slots(_CANDIDATE_OWN_SLOTS), d)
        self.player_proj = nn.Linear(int(TokenWidth2.TW2_PLAYER) - _FIRST_FEATURE, d)

        # Row order supplies identity without extra input data (§3.3). Candidates
        # take their hex's; players' rows are in seat order and carry the seat.
        self.hex_id_embed = nn.Embedding(_NUM_HEXES, d)
        self.corp_id_embed = nn.Embedding(_NUM_CORPS, d)
        self.private_id_embed = nn.Embedding(_NUM_PRIVATES, d)
        self.train_id_embed = nn.Embedding(_NUM_TRAINS, d)
        # Row counts per type, in row order; added to every row.
        type_counts = (
            1, 1, 1, _NUM_CORPS, _NUM_PRIVATES, _NUM_TRAINS, _NUM_HEXES, _NUM_CANDIDATES, cfg.num_players,
        )
        self.type_embeds = nn.Embedding(len(type_counts), d)
        self.register_buffer(
            "_type_ids",
            torch.repeat_interleave(torch.arange(len(type_counts)), torch.tensor(type_counts)),
            persistent=False,
        )

        # --- The trunk's rows ---
        # Every input row but the candidates, in order: the players' rows move
        # up. Records and planes name input rows, so ``_row_map`` gives each
        # one's trunk row, or -1 for a candidate's. Its 256 entries cover any
        # index a uint8 record holds.
        trunk_rows = torch.cat([torch.arange(_CANDIDATES.start), torch.arange(_CANDIDATES.stop, num_tokens)])
        assert len(trunk_rows) == trunk_tokens
        row_map = torch.full((256,), -1, dtype=torch.long)
        row_map[trunk_rows] = torch.arange(trunk_tokens)
        self.register_buffer("_trunk_rows", trunk_rows, persistent=False)
        self.register_buffer("_row_map", row_map, persistent=False)

        # --- Relations ---
        # A record's id gives its channel, or -1 for an id with none: the
        # candidates' relations and any later step's.
        relation_ids = torch.tensor(_RELATION_IDS, dtype=torch.long)
        relation_channel = torch.full((256,), -1, dtype=torch.long)
        relation_channel[relation_ids] = torch.arange(_NUM_DYNAMIC_RELATIONS)
        self.register_buffer("_relation_ids", relation_ids, persistent=False)
        self.register_buffer("_relation_channel", relation_channel, persistent=False)
        # Static planes never cross IPC and aren't checkpointed (§6.3). The
        # trunk takes them over its own rows.
        static = torch.from_numpy(static_relation_planes(cfg.num_players)).float()
        self.register_buffer(
            "_static_relations", static[:, trunk_rows][:, :, trunk_rows].clone(), persistent=False,
        )
        relation_scales = torch.ones(_NUM_DYNAMIC_RELATIONS)
        shares = relation_channel[[int(RelationId.REL_PLAYER_SHARES), int(RelationId.REL_CORP_SHARES)]]
        trains = relation_channel[[int(RelationId.REL_CORP_TRAINS), int(RelationId.REL_TRAIN_HELD_BY)]]
        relation_scales[shares] = 1 / 10
        relation_scales[trains] = 1 / 4
        self.register_buffer("_relation_scales", relation_scales, persistent=False)
        self.register_buffer(
            "_head_offsets",
            torch.arange(cfg.num_heads, dtype=torch.long)[:, None] * trunk_tokens * trunk_tokens,
            persistent=False,
        )
        self.relation_bias_mult = nn.Parameter(torch.zeros(cfg.num_layers, cfg.num_heads, _NUM_MODEL_RELATIONS))

        # --- Transformer trunk ---
        d_ff = _round_up_to_multiple(math.ceil(cfg.ff_mult * d), 64)
        self.blocks = nn.ModuleList([TransformerBlock(d, cfg.num_heads, d_ff) for _ in range(cfg.num_layers)])
        self.final_norm = nn.RMSNorm(d)

        # --- The rows the slot tables name (engine-design §12 item 8) ---
        def rows(first: int, table: list[int]) -> torch.Tensor:
            return first + torch.tensor(table, dtype=torch.long)

        self.register_buffer("_track_rows", rows(_HEXES.start, D.TRACK_HEXES), persistent=False)
        self.register_buffer("_token_rows", rows(_HEXES.start, D.TOKEN_HEXES), persistent=False)
        self.register_buffer("_discard_rows", rows(_TRAINS.start, D.DISCARD_TYPES), persistent=False)
        self.register_buffer("_trade_in_types", rows(0, D.TRADE_IN_TYPE), persistent=False)
        self.register_buffer("_port_rows", rows(_HEXES.start, D.PORT_LAY_HEX), persistent=False)
        # The port tile's path mask at each lay's rotation, in PAIRS order.
        port_masks = [D.TILE_MASK[D.PORT_TILE][rotation] for rotation in D.PORT_LAY_ROT]
        self.register_buffer(
            "_port_masks",
            torch.tensor([[mask >> pair & 1 for pair in range(D.NUM_PAIRS)] for mask in port_masks]).float(),
            persistent=False,
        )
        # For the candidate head: per side, each hex's hex across it (a one-hot
        # row, zero at the map's edge), and the sides each endpoint pair touches.
        side = int(StaticRelationId.SREL_HEX_SIDE)
        self.register_buffer(
            "_side_planes", static[side:side + _NUM_SIDES, _HEXES, _HEXES].clone(), persistent=False,
        )
        pair_sides = torch.zeros(_NUM_PAIRS, _NUM_SIDES)
        for pair, ends in enumerate(zip(D.PAIR_A, D.PAIR_B, strict=True)):
            for end in ends:
                if end < _NUM_SIDES:  # the other endpoint is the center
                    pair_sides[pair, end] = 1.0
        self.register_buffer("_pair_sides", pair_sides, persistent=False)

        # --- Policy heads (§4) ---
        # c: the decision corp's row joins the context in every head but
        # auction_private, bid, and stock_corp, whose phases have none.
        ctx, actor_ctx = 3 * d, 2 * d
        # One column each: AUCTION, STOCK, BUY_PRIVATE_PRE, TRACK, TOKEN, BUY_PRIVATE_POST, ER_BONUS.
        self.pass_head = _mlp(ctx, d, 7)
        # Per private: bid on it, buy it as the topmost.
        self.auction_private_head = _mlp(actor_ctx + d + _num_slots(_PRIVATE_AUCTION_SLOTS), d, 2)
        # Drop, then the 16 grid points.
        self.bid_head = _mlp(actor_ctx + 2 * d + _num_slots(_PRICE_SLOTS), d, 17)
        # Per corp: IPO, pool, par, sell, with the actor's shares and sales of it.
        self.stock_corp_head = _mlp(actor_ctx + d + _num_slots(_CORP_STOCK_SLOTS) + 2, d, 4)
        self.par_head = _mlp(ctx + d + _num_slots(_PRICE_SLOTS) + _num_slots(_PRICE_FLOAT_SLOTS), d, 6)
        # n - 1 shares; SELL_AMOUNT and EMERGENCY_SELL_AMOUNT.
        self.sale_head = _mlp(ctx + d + _num_slots(_CORP_SALE_SLOTS), d, 5)
        # Per private: buy it, use its ability (DR's and MF's rows only).
        self.private_head = _mlp(ctx + d + _num_slots(_PRIVATE_BUY_SLOTS), d, 2)
        self.track_hex_head = _mlp(ctx + d + _num_slots(_HEX_TRACK_SLOTS), d, 1)
        # TILE and ER_BONUS, from rows the trunk ran without the candidates
        # (``_candidate_logits``): the hex it would be laid on, the hexes across
        # its exits, and its own row as projected, normed to the final rows' scale.
        self.candidate_norm = nn.RMSNorm(d)
        self.exit_side_proj = nn.ModuleList([nn.Linear(d, d // 4) for _ in range(_NUM_SIDES)])
        self.exit_proj = nn.Linear(d // 4, d)
        self.candidate_head = nn.Sequential(
            nn.Linear(ctx + 3 * d + _num_slots(_CANDIDATE_SLOTS), d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
        )
        self.token_hex_head = _mlp(ctx + d + _num_slots(_HEX_TOKEN_SLOTS), d, 1)
        # Withhold, payout.
        self.dividend_head = _mlp(ctx + d + _num_slots(_DIVIDEND_SLOTS), d, 2)
        self.discard_head = _mlp(ctx + d + _num_slots(_TRAIN_DISCARD_SLOTS), d, 1)
        # Pass, depot, pool, bankrupt, from all six TRAIN rows.
        self.buy_train_head = _mlp(ctx + _NUM_TRAINS * (d + _num_slots(_TRAIN_BUY_SLOTS)), d, 4)
        # Per corp: its trains, an emergency sale of it.
        self.buy_train_corp_head = _mlp(ctx + d + _num_slots(_CORP_BUY_TRAIN_SLOTS), d, 2)
        # Per train type, with the seller: add one, trade one in for a D.
        self.select_train_head = _mlp(ctx + 2 * d + _num_slots(_TRAIN_SELECT_SLOTS), d, 2)
        # The price grid, with the train type, the seller, and what the seller loses.
        self.train_price_head = _mlp(
            ctx + 3 * d + _num_slots(_PRICE_SLOTS) + _num_slots(_TRAIN_SELLER_SLOTS), d, 5,
        )
        self.private_price_head = _mlp(ctx + 2 * d + _num_slots(_PRICE_SLOTS), d, 4)
        self.port_head = _mlp(ctx + d + int(D.NUM_PAIRS), d, 1)

        # --- Value head: policy-MLP hidden width, applied per player row ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
            nn.Tanh(),
        )

        self._validate_policy_layout()
        self._init_weights()

    def _validate_policy_layout(self) -> None:
        """Check each readout's block against the game's action sizes.

        A dry run on one row of zeros, so the check reads the blocks the
        forward builds, and each head's input width with them.
        """
        tokens = torch.zeros(1, self.cfg.num_tokens, self.cfg.d_model)
        x = torch.zeros(1, self.cfg.num_tokens, self.cfg.token_dim)
        with torch.no_grad():
            widths = [block.shape[1] for block in self._policy_blocks(self._policy_context(tokens, x))]
        expected = list(POLICY_LAYOUT.phase_sizes)
        assert widths == expected, f"policy block widths {widths} do not match PHASE_ACTION_SIZES {expected}"
        assert sum(widths) == POLICY_LAYOUT.unified_dim, (
            f"policy block total {sum(widths)} != unified_dim {POLICY_LAYOUT.unified_dim}"
        )

    # ------------------------------------------------------------------
    # Input projection
    # ------------------------------------------------------------------

    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw rows to d_model, with identity and type embeddings.

        Hidden rows project to their biases and embeddings; they are masked as
        keys, and no legal slot reads them.

        Args:
            x: (batch, cfg.num_tokens, token_dim) zero-padded raw features: fp16
                or fp32 under autocast on the eval path, or fp32 matching the
                projection weights on the non-autocast trainer / CPU paths.
        Returns:
            ``(batch, cfg.num_tokens, d_model)`` projected input embeddings.
        """
        tiles = self.tile_proj(x[:, _HEXES.start:_CANDIDATES.stop, _TILE_SLOTS])
        hexes = (
            tiles[:, :_NUM_HEXES]
            + self.hex_proj(x[:, _HEXES][:, :, _HEX_OWN_SLOTS])
            + _match(self.hex_id_embed.weight, tiles)
        )
        # A candidate's identity is that of the hex HEX 1 selects, zero without one.
        selected_hex = _match(x[:, _HEXES, _HEX_SELECTED], tiles) @ _match(self.hex_id_embed.weight, tiles)
        candidates = (
            tiles[:, _NUM_HEXES:]
            + self.candidate_proj(x[:, _CANDIDATES][:, :, _CANDIDATE_OWN_SLOTS])
            + selected_hex.unsqueeze(1)
        )
        tokens = torch.cat(
            [
                self.global_proj(x[:, _GLOBAL, _FIRST_FEATURE:int(TokenWidth2.TW2_GLOBAL)]).unsqueeze(1),
                self.price_proj(x[:, _PRICE, _FIRST_FEATURE:int(TokenWidth2.TW2_PRICE)]).unsqueeze(1),
                self.dividend_proj(x[:, _DIVIDEND, _FIRST_FEATURE:int(TokenWidth2.TW2_DIVIDEND)]).unsqueeze(1),
                self.corp_proj(x[:, _CORPS, _FIRST_FEATURE:int(TokenWidth2.TW2_CORP)])
                + _match(self.corp_id_embed.weight, tiles),
                self.private_proj(x[:, _PRIVATES, _FIRST_FEATURE:int(TokenWidth2.TW2_PRIVATE)])
                + _match(self.private_id_embed.weight, tiles),
                self.train_proj(x[:, _TRAINS, _FIRST_FEATURE:int(TokenWidth2.TW2_TRAIN)])
                + _match(self.train_id_embed.weight, tiles),
                hexes,
                candidates,
                self.player_proj(x[:, _PLAYERS, _FIRST_FEATURE:int(TokenWidth2.TW2_PLAYER)]),
            ],
            dim=1,
        )
        return tokens + _match(self.type_embeds(self._type_ids), tokens)

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def _prepare_relations(self, relations: torch.Tensor) -> _DenseRelations | _FlatRelations:
        """Normalize a forward's relations once for all layers.

        Round the product, not the reciprocal, when entering bf16/fp16: the
        weights stay fp32 until a layer casts them.
        """
        rows = self._trunk_rows
        if relations.ndim == 4:
            planes = relations.index_select(1, self._relation_ids).index_select(2, rows).index_select(3, rows)
            planes = planes.to(dtype=self._relation_scales.dtype)
            return _DenseRelations(planes * self._relation_scales.view(1, -1, 1, 1))
        coords = relations.long()
        batch, relation, query, key, value = coords.unbind(1)
        channel = self._relation_channel[relation.clamp(0, 255)]
        query, key = self._row_map[query], self._row_map[key]
        # Any other record gets weight 0, channel 0, and cell 0 of its state:
        # padding, a relation without a channel, or one that names a candidate
        # row. Filtering them out would make N depend on the data.
        real = (value > 0) & (channel >= 0) & (query >= 0) & (key >= 0)
        channel = channel.masked_fill(~real, 0)
        weights = value.to(dtype=self._relation_scales.dtype) * self._relation_scales[channel]
        num_tokens = self.cfg.num_trunk_tokens
        cells = (query * num_tokens + key).masked_fill(~real, 0)
        cells = batch * (self.cfg.num_heads * num_tokens * num_tokens) + cells
        return _FlatRelations(
            channel,
            (cells[None, :] + self._head_offsets).reshape(-1),
            weights.masked_fill(~real, 0),
        )

    def _attention_bias(
        self,
        relations: _DenseRelations | _FlatRelations,
        hidden_keys: torch.Tensor,
        layer_idx: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """One layer's finished SDPA mask over the trunk's rows, ``(B, H, T', T')``:
        the static bias, the relations' on top of it in place, and the hidden
        keys, one dense tensor."""
        mult = _match(self.relation_bias_mult[layer_idx], ref)
        static = torch.einsum(
            "rij,hr->hij", _match(self._static_relations, ref), mult[:, _NUM_DYNAMIC_RELATIONS:],
        )
        if isinstance(relations, _DenseRelations):
            bias = torch.einsum(
                "brij,hr->bhij", _match(relations.planes, ref), mult[:, :_NUM_DYNAMIC_RELATIONS],
            )
            bias += static
        else:
            bias = static.repeat(ref.shape[0], 1, 1, 1)
            values = mult.index_select(1, relations.relation_ids) * _match(relations.weights, ref)
            bias.reshape(-1).scatter_add_(0, relations.flat_indices, values.reshape(-1))
        return bias.masked_fill_(hidden_keys, torch.finfo(bias.dtype).min)

    def _encode(self, x: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """The trunk: projections, the relation-biased blocks, and the final norm.

        The blocks run on the rows that are not candidates. The result is in
        the input's row order, as the heads index it: the trunk's final rows
        around the candidates' projected rows, normed, which no block has seen.
        """
        embedded = self._project_tokens(x)
        first, stop = _CANDIDATES.start, _CANDIDATES.stop
        tokens = torch.cat([embedded[:, :first], embedded[:, stop:]], dim=1)
        hidden = x[:, :, _KEY_MASK] <= 0.5
        hidden_keys = torch.cat([hidden[:, :first], hidden[:, stop:]], dim=1)[:, None, None, :]
        prepared = self._prepare_relations(relations)
        for layer_idx, block in enumerate(self.blocks):
            tokens = block(tokens, self._attention_bias(prepared, hidden_keys, layer_idx, tokens))
        tokens = self.final_norm(tokens)
        candidates = self.candidate_norm(embedded[:, _CANDIDATES])
        return torch.cat([tokens[:, :first], candidates, tokens[:, first:]], dim=1)

    # ------------------------------------------------------------------
    # Unified policy: every readout runs once on the full batch
    # ------------------------------------------------------------------

    def _policy_context(self, tokens: torch.Tensor, x: torch.Tensor) -> _PolicyContext:
        """The raw rows in fp32 and the selections every head reads, computed once."""
        raw = x.float()
        actor_context = torch.cat(
            [tokens[:, _GLOBAL], _selected_row(raw, tokens, _PLAYERS, _PLAYER_ACTOR)], dim=-1,
        )
        corp = _selected_row(raw, tokens, _CORPS, _CORP_DECISION)
        return _PolicyContext(raw, tokens, torch.cat([actor_context, corp], dim=-1), actor_context)

    def _candidate_logits(self, ctx: _PolicyContext) -> torch.Tensor:
        """``(B, 15)``: TILE's and ER_BONUS's logit for each candidate row.

        A candidate is scored from the context, the final row of the hex HEX 1
        selects, the final rows of the hexes across its exits, its own row
        (projected and normed, ``_encode``), and its raw slots. A side is an
        exit if a path of the candidate's tile, as laid, touches it. Each
        side's hex goes through that side's projection, the exits' are summed,
        and one projection brings the sum to the trunk's width.
        """
        raw, tokens = ctx.raw, ctx.tokens
        hexes = tokens[:, _HEXES]
        selector = _match(raw[:, _HEXES, _HEX_SELECTED], tokens)  # (B, hexes)
        selected = torch.bmm(selector.unsqueeze(1), hexes).squeeze(1)
        # The final row of the hex across each side of the selected hex, zero with none.
        across = torch.einsum("bh,shk->bsk", selector, _match(self._side_planes, tokens))
        neighbors = torch.bmm(across, hexes)  # (B, sides, d)
        projected = torch.stack(
            [side(neighbors[:, s]) for s, side in enumerate(self.exit_side_proj)], dim=1,
        )  # (B, sides, d / 4)
        candidate_raw = raw[:, _CANDIDATES]
        paths = candidate_raw[:, :, _TILE_PATH_SLOTS]
        exits = (paths @ _match(self._pair_sides, paths) > 0.5).to(tokens.dtype)  # (B, 15, sides)
        pooled = self.exit_proj(torch.bmm(exits, projected))  # (B, 15, d)
        return _score_rows(
            self.candidate_head,
            torch.cat([ctx.context, selected], dim=-1),
            pooled,
            tokens[:, _CANDIDATES],
            candidate_raw[:, :, _CANDIDATE_SLOTS],
        ).squeeze(-1)

    def _policy_blocks(self, ctx: _PolicyContext) -> list[torch.Tensor]:
        """Run every head once on the full batch and lay out the phase blocks.

        Blocks come in DecisionPhase order, the layout's. Readouts run whatever
        phase a row is in: the caller's legal mask selects, and the extra work
        is small relative to the trunk.
        """
        raw, tokens, context, actor_context = ctx.raw, ctx.tokens, ctx.context, ctx.actor_context
        (
            pass_auction, pass_stock, pass_private_pre, pass_track, pass_token, pass_private_post, pass_er_bonus,
        ) = self.pass_head(context).split(1, dim=-1)

        # Selections and raw inputs that several heads read.
        the_private = _selected_row(raw, tokens, _PRIVATES, _PRIVATE_SELECTED)
        seller = _selected_row(raw, tokens, _CORPS, _CORP_SELLER)
        price = torch.cat([tokens[:, _PRICE], _match(raw[:, _PRICE, _PRICE_SLOTS], tokens)], dim=-1)

        # AUCTION: the head scores every private; private 0 is only ever bought.
        auction = _score_rows(
            self.auction_private_head, actor_context, tokens[:, _PRIVATES],
            raw[:, _PRIVATES, _PRIVATE_AUCTION_SLOTS],
        )
        topmost = _match(raw[:, _PRIVATES, _PRIVATE_TOPMOST], tokens)
        auction_buy = (topmost * auction[:, :, 1]).sum(1, keepdim=True)

        # BUY_PRIVATE_PRE, BUY_PRIVATE_POST, and STOCK's DR and MF slots.
        private = _score_rows(
            self.private_head, context, tokens[:, _PRIVATES], raw[:, _PRIVATES, _PRIVATE_BUY_SLOTS],
        )
        private_buy = private[:, :, 0]
        private_dr = private[:, _PRIVATE_DR, 1:]
        private_mf = private[:, _PRIVATE_MF, 1:]

        # STOCK: (B, corps, 4) laid out by kind, with the actor's holdings of each corp.
        actor_holdings = _selected_raw(raw, _PLAYERS, _PLAYER_ACTOR, _PLAYER_HOLDING_SLOTS)
        stock_corp = _score_rows(
            self.stock_corp_head, actor_context, tokens[:, _CORPS],
            raw[:, _CORPS][:, :, _CORP_STOCK_SLOTS],
            actor_holdings.reshape(-1, 2, _NUM_CORPS).transpose(1, 2),
        ).transpose(1, 2).flatten(1)

        # SELL_AMOUNT and EMERGENCY_SELL_AMOUNT.
        sale = self.sale_head(torch.cat(
            [
                context,
                _selected_row(raw, tokens, _CORPS, _CORP_SOLD),
                _match(_selected_raw(raw, _CORPS, _CORP_SOLD, _CORP_SALE_SLOTS), tokens),
            ],
            dim=-1,
        ))

        # TILE and ER_BONUS.
        candidate = self._candidate_logits(ctx)

        # BUY_TRAIN: the joint head's pass, depot, and pool, the corps, then its bankruptcy.
        buy_train = self.buy_train_head(torch.cat(
            [
                context,
                tokens[:, _TRAINS].flatten(1),
                _match(raw[:, _TRAINS][:, :, _TRAIN_BUY_SLOTS].flatten(1), tokens),
            ],
            dim=-1,
        ))
        buy_train_corp = _score_rows(
            self.buy_train_corp_head, context, tokens[:, _CORPS], raw[:, _CORPS][:, :, _CORP_BUY_TRAIN_SLOTS],
        )

        # SELECT_TRAIN: an add logit per type, a trade-in logit for the types a D takes.
        select_train = _score_rows(
            self.select_train_head,
            torch.cat([context, seller], dim=-1),
            tokens[:, _TRAINS],
            raw[:, _TRAINS][:, :, _TRAIN_SELECT_SLOTS],
        )

        return [
            torch.cat([pass_auction, auction_buy, auction[:, 1:, 0]], dim=-1),                # AUCTION (8)
            self.bid_head(torch.cat([actor_context, the_private, price], dim=-1)),            # BID (17)
            torch.cat([pass_stock, stock_corp, private_dr, private_mf], dim=-1),              # STOCK (31)
            self.par_head(torch.cat(
                [context, price, _match(raw[:, _PRICE, _PRICE_FLOAT_SLOTS], tokens)], dim=-1,
            )),                                                                               # PAR (6)
            sale,                                                                             # SELL_AMOUNT (5)
            torch.cat([pass_private_pre, private_buy, private_dr, private_mf], dim=-1),       # BUY_PRIVATE_PRE (10)
            torch.cat(
                [pass_track, _score_rows(
                    self.track_hex_head, context, tokens[:, self._track_rows],
                    raw[:, self._track_rows, _HEX_TRACK_SLOTS],
                ).squeeze(-1)],
                dim=-1,
            ),                                                                                # TRACK (46)
            candidate,                                                                        # TILE (15)
            torch.cat(
                [pass_token, _score_rows(
                    self.token_hex_head, context, tokens[:, self._token_rows],
                    raw[:, self._token_rows, _HEX_TOKEN_SLOTS],
                ).squeeze(-1)],
                dim=-1,
            ),                                                                                # TOKEN (16)
            self.dividend_head(torch.cat(
                [context, tokens[:, _DIVIDEND], _match(raw[:, _DIVIDEND, _DIVIDEND_SLOTS], tokens)],
                dim=-1,
            )),                                                                               # DIVIDEND (2)
            _score_rows(
                self.discard_head, context, tokens[:, self._discard_rows],
                raw[:, self._discard_rows, _TRAIN_DISCARD_SLOTS],
            ).squeeze(-1),                                                                    # DISCARD (3)
            torch.cat(
                [buy_train[:, :3], buy_train_corp[:, :, 0], buy_train_corp[:, :, 1], buy_train[:, 3:]],
                dim=-1,
            ),                                                                                # BUY_TRAIN (18)
            torch.cat(
                [select_train[:, :, 0], select_train[:, :, 1].index_select(1, self._trade_in_types)],
                dim=-1,
            ),                                                                                # SELECT_TRAIN (9)
            self.train_price_head(torch.cat(
                [
                    context, _selected_row(raw, tokens, _TRAINS, _TRAIN_SELECTED), seller, price,
                    _match(_selected_raw(raw, _TRAINS, _TRAIN_SELECTED, _TRAIN_SELLER_SLOTS), tokens),
                ],
                dim=-1,
            )),                                                                               # TRAIN_PRICE (5)
            sale,                                                                             # EMERGENCY_SELL_AMOUNT (5)
            torch.cat([pass_private_post, private_buy, private_dr], dim=-1),                  # BUY_PRIVATE_POST (9)
            self.private_price_head(torch.cat([context, the_private, price], dim=-1)),        # PRIVATE_PRICE (4)
            torch.cat([pass_er_bonus, candidate[:, :_NUM_ER_CANDIDATES]], dim=-1),            # ER_BONUS (9)
            _score_rows(
                self.port_head, context, tokens[:, self._port_rows],
                self._port_masks.expand(tokens.shape[0], -1, -1),
            ).squeeze(-1),                                                                    # PORT (10)
        ]

    def _unified_logits(self, tokens: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``(B, unified_dim)`` fp32 logits, before the legal mask."""
        return torch.cat(self._policy_blocks(self._policy_context(tokens, x)), dim=-1).to(torch.float32)

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
            x: ``(batch, cfg.num_tokens, cfg.token_dim)`` floating-point rows,
                zero-padded to ``cfg.token_dim``. There is no unconditional
                dtype conversion here: eval paths rely on autocast to run the
                projections/trunk in bf16/fp16, while the trainer and CPU tests
                pass fp32 directly.
            legal_mask: ``(batch, unified_dim)`` bool tensor. ``True`` marks a
                slot legal; illegal slots are masked to ``-1e9``. Only a
                caller's scratch row, whose output it drops, may be all false.
            relations: Flat ``(N, 5)`` int16 records ``(batch, relation, query,
                key, value)``, with N taken from this shape, or dense ``(batch,
                R, num_tokens, num_tokens)`` uint8 planes, R >= 29, queries by
                keys. Both name the input's rows. Only value > 0, ids 0-20 and
                25-28, and rows other than the candidates' carry weight in either.

        Returns:
            policy_logits: ``(batch, unified_dim)`` fp32 logits with illegal
                slots set to ``-1e9``.
            values: ``(batch, cfg.num_players)`` per-player expected outcomes
                in canonical order, in ``[-1, 1]``. With ``cfg.zero_sum_values``,
                the mean over real players is subtracted after tanh, so each
                row sums to zero and padded slots are zero. In mixed-count
                training this is padded to the model capacity; callers mask or
                slice slots beyond the state's actual player count.
        """
        num_tokens = self.cfg.num_tokens
        assert x.is_floating_point(), f"x must be floating-point rows; got {x.dtype}"
        assert tuple(x.shape) == (x.shape[0], num_tokens, self.cfg.token_dim), (
            f"x shape must be (batch, {num_tokens}, {self.cfg.token_dim}); got {tuple(x.shape)}"
        )
        assert legal_mask.dtype == torch.bool, (
            f"legal_mask must be bool (uint8 would make ~legal_mask a bitwise "
            f"complement, not logical NOT); got {legal_mask.dtype}"
        )
        if torch.compiler.is_compiling():
            # Eval marks each input batch dimension independently with
            # mark_unbacked. Establish their equality before Python shape
            # comparisons; otherwise Dynamo cannot decide u0 == u1.
            torch._check(legal_mask.shape[0] == x.shape[0])
            if relations.ndim == 4:
                torch._check(relations.shape[0] == x.shape[0])
        assert tuple(legal_mask.shape) == (x.shape[0], POLICY_LAYOUT.unified_dim), (
            f"legal_mask shape must be (batch, {POLICY_LAYOUT.unified_dim}); got {tuple(legal_mask.shape)}"
        )
        if relations.ndim == 4:
            assert relations.dtype == torch.uint8, f"dense relation planes must be uint8; got {relations.dtype}"
            assert relations.shape[0] == x.shape[0] and relations.shape[1] >= _NUM_RELATION_IDS, (
                f"relations must be (batch, >= {_NUM_RELATION_IDS}, T, T); got {tuple(relations.shape)}"
            )
            assert relations.shape[2:] == (num_tokens, num_tokens), (
                f"relation planes must be {num_tokens} x {num_tokens}; got {tuple(relations.shape)}"
            )
        else:
            assert relations.ndim == 2 and relations.shape[1] == FLAT_RELATION_RECORD_WIDTH, (
                f"relations must be flat (N, {FLAT_RELATION_RECORD_WIDTH}) records or dense "
                f"(batch, R, T, T) planes; got {tuple(relations.shape)}"
            )
            assert relations.dtype == torch.int16, f"flat relation records must be int16; got {relations.dtype}"

        tokens = self._encode(x, relations)

        # Cast to fp32 before the sentinel: under autocast the logits are in
        # the autocast dtype (bf16/fp16), but downstream softmax / log_softmax
        # wants stable fp32 ``-1e9`` sentinels.
        policy_logits = self._unified_logits(tokens, x).masked_fill(~legal_mask, -1e9)

        values = self.value_head(tokens[:, _PLAYERS]).squeeze(-1)
        if self.cfg.zero_sum_values:
            # Padded player rows are all-zero, including the key mask.
            present = (x[:, _PLAYERS, _KEY_MASK] > 0.5).to(values.dtype)
            count = present.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (values * present).sum(dim=1, keepdim=True) / count
            values = (values - mean) * present
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
        legal slots, a TB diagnostic for logit-scale drift.

        Returns a ``(2 * len(phases_with_pass_slot),)`` tensor packed as
        ``[pass_abs_p0, action_abs_p0, pass_abs_p1, action_abs_p1, ...]``.
        Phases with no rows in the batch (or all-illegal pass slots) return 0
        for both stats: the caller filters by per-phase row counts.
        """
        abs_logits = policy_logits.detach().abs()
        stats: list[torch.Tensor] = []
        for phase in POLICY_LAYOUT.phases_with_pass_slot:
            pass_slot = POLICY_LAYOUT.offsets[phase]
            action_start = pass_slot + 1
            action_stop = POLICY_LAYOUT.offsets[phase + 1]

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
    # Optimizer routing
    # ------------------------------------------------------------------

    def input_output_layers(self) -> list[nn.Module]:
        """Input projections and final head Linears.

        Muon targets hidden-layer matrices, so the trainer keeps these on
        AdamW. Head hidden layers stay Muon-eligible, the candidate head's
        exit projections among them.
        """
        heads = [
            self.pass_head, self.auction_private_head, self.bid_head, self.stock_corp_head,
            self.par_head, self.sale_head, self.private_head, self.track_hex_head,
            self.token_hex_head, self.dividend_head, self.discard_head,
            self.buy_train_head, self.buy_train_corp_head, self.select_train_head,
            self.train_price_head, self.private_price_head, self.port_head,
            self.value_head,
        ]
        return [
            self.global_proj, self.price_proj, self.dividend_proj, self.corp_proj,
            self.private_proj, self.train_proj, self.tile_proj, self.hex_proj,
            self.candidate_proj, self.player_proj,
            *(head[2] for head in heads),
            self.candidate_head[4],
        ]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT/LLaMA-style trunc-normal initialization, as v0 and v3."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        # Relation attention starts behavior-preserving; training can learn
        # positive or negative head/layer-specific biases from zero.
        nn.init.zeros_(self.relation_bias_mult)


def build_model(config: TrainingConfig, spec: ModelInputSpec) -> G1889TransformerNet:
    """The loader's entry point: the model of ``config``'s sizes for ``spec``."""
    assert spec.game == "1889", f"nn/transformer-1889-v1.py runs game 1889, not {spec.game!r}"
    assert spec.layout_version == INPUT_LAYOUT_VERSION, f"layout {spec.layout_version}, not {INPUT_LAYOUT_VERSION}"
    assert spec.token_dim == int(TokenDataSize.TOKEN_DIM_2), f"token_dim {spec.token_dim}"
    assert spec.num_tokens == get_num_tokens(spec.num_players), (
        f"{spec.num_tokens} tokens for {spec.num_players} players"
    )
    assert spec.policy_dim == POLICY_LAYOUT.unified_dim, f"policy_dim {spec.policy_dim}"
    assert spec.value_dim == spec.num_players, f"value_dim {spec.value_dim} for {spec.num_players} players"
    return G1889TransformerNet(TransformerConfig(
        num_players=spec.num_players,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_mult=config.ff_mult,
        zero_sum_values=config.zero_sum_values,
    ))
