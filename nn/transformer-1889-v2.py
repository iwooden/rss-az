"""Transformer v2 for Shikoku 1889 AlphaZero training (1889/v2-design.md §4-6).

Input layout 3 (token-data-1889.md, "Layout 3"): one packed float vector a
state, cut by ``g1889.token_data.get_input_parts`` into row parts, pair blocks,
and the choice table. Every offset, shape, and trunk row below comes from that
export and the engine's static tables (``get_tile_table``, ``get_choice_slots``,
``get_named_rows``); the slot numbers this module reads by meaning (slot 0, the
``is_selected`` slots, the decision phase, the tile slots, build distance's)
follow the contract's tables, as v1's raw slots follow layout 2's.

- **Rows:** GLOBAL, 7 CORP, 7 PRIVATE, 25 CENTER, and the players, 40 +
  capacity. Each type has its own projection over its own slots and its slice
  of every pair block it is in: a CORP row gets its players', privates', and
  centers' values, a CENTER row its corps' and the other centers', a PLAYER
  row its corps' and privates'. A slice over players is turned to start at the
  actor (PLAYER 2-6), five seats wide whatever the capacity. A CENTER row's
  tile code and rotation look its tile block up, through the tile projection
  the choice table's tiles share. Identity and type embeddings are v1's.
- **Attention bias:** per layer, zero but for each pair block's rectangles,
  filled both ways by an einsum of the block's channels with that layer's
  per-head weights, a weight set per direction. CENTER x CENTER is symmetric
  and its two directions are one rectangle, so it has one set. Build
  distance's channels enter the bias as a one-hot of the distance their
  closeness gives (0, 1, 2, 3, 4-5), all zero for no closeness.
- **The choice head** (v2-design §5.4) scores choice row k of the decision
  from ``h_k = slot_embed[u_k] + W[phase] f_k + named_proj(row[ptr_k]) +
  tile_proj(tile_k) + hex_embed[hex_k]`` and the final rows of GLOBAL, the
  actor, the decision corp, and the subject, each choice alone. Its scores
  land on the phase's unified slots.
- **Build distance** (token-data-1889.md, Point 5) is zero until v2-design
  §10's step 6. Its weights, in the bias, the row projections, and ``W``,
  start at zero, so they get no gradient while the inputs are zero and a
  running model can switch it on.

The trunk's block, the value head, zero-sum values, initialization, and the
optimizer routing are v1's. The file stands alone, as v1's does beside v0's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

import g1889.constants as K
import g1889.data.tables as D
from g1889.token_data import (
    ChoiceTable3,
    NamedRow3,
    get_choice_slots,
    get_input_parts,
    get_input_size,
    get_named_rows,
    get_num_trunk_rows,
    get_tile_table,
)
from games.g1889 import POLICY_LAYOUT
from nn.model_contract import ModelInputSpec

if TYPE_CHECKING:
    from train.config import TrainingConfig

INPUT_LAYOUT_VERSION = 3

_GELU_APPROX = "tanh"

# Slot meanings (token-data-1889.md, "Layout 3"). Slot 0 of every row is the
# key mask; GLOBAL 1-19 are the decision phase, one-hot, as in layout 2.
_KEY_MASK = 0
_GLOBAL_PHASE = 1
_CORP_DECISION = 1
_PRIVATE_TOPMOST = 2
_PLAYER_ACTOR = 1
_PLAYER_SEATS = slice(2, 7)  # seats after the actor, one-hot; slot 2 is the actor's own
_CENTER_TILE, _CENTER_ROTATION = 2, 3  # integers, looked up in the tile table
# The subject (v2-design §5.4's s): PRIVATE 1, CORP 2, CORP 3, or CENTER 1,
# set at disjoint decisions.
_SUBJECT = {"PRIVATE": (1,), "CORP": (2, 3), "CENTER": (1,)}

# Build distance's inputs, zero until v2-design §10's step 6: the pair-block
# channels (closeness (6 - d)/6, Point 5), CORP 70-75, and the phase blocks'
# bd slots (TRACK 26-29; TILE and ER_BONUS 28-36; TOKEN 5-13).
_BUILD_DISTANCE_CHANNELS = {"CORP_CENTER": 3, "CENTER_CENTER": 0}
_BUILD_DISTANCE_SLOTS = {"CORP": range(70, 76)}
_BUILD_DISTANCE_PHASE_SLOTS = {
    int(K.DPHASE_TRACK): range(26, 30),
    int(K.DPHASE_TILE): range(28, 37),
    int(K.DPHASE_ER_BONUS): range(28, 37),
    int(K.DPHASE_TOKEN): range(5, 14),
}
_CLOSENESS_LAYS = 6  # closeness is (6 - d)/6 for a distance of d <= 5 lays
_DISTANCE_BUCKETS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 5))  # (nearest, farthest) lays

_PLAYER = "PLAYER"
_CHOICES = "CHOICES"


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round positive integer ``value`` up to the next ``multiple`` boundary."""
    return ((value + multiple - 1) // multiple) * multiple


def _match(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Move small buffers/raw slices to the current runtime dtype and device."""
    return tensor.to(device=ref.device, dtype=ref.dtype)


def distance_buckets(closeness: torch.Tensor) -> torch.Tensor:
    """``(..., 5)``: the one-hot of the distance a closeness gives, by
    ``_DISTANCE_BUCKETS``; all zero for a closeness of 0 (none within 5 lays,
    or not filled yet)."""
    lays = _CLOSENESS_LAYS * (1 - closeness)
    buckets = torch.stack([(lays > near - 0.5) & (lays < far + 0.5) for near, far in _DISTANCE_BUCKETS], dim=-1)
    return (buckets & (closeness > 0).unsqueeze(-1)).to(closeness.dtype)


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
    _input_size: int = field(init=False, repr=False)
    _num_trunk_rows: int = field(init=False, repr=False)

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
        object.__setattr__(self, "_input_size", packed_input_size(self.num_players))
        object.__setattr__(
            self, "_num_trunk_rows", int(get_num_trunk_rows(self.num_players, layout_version=INPUT_LAYOUT_VERSION)),
        )

    @property
    def input_size(self) -> int:
        """The packed vector's length at this capacity."""
        return self._input_size

    @property
    def num_trunk_rows(self) -> int:
        """The trunk's rows: 40 + capacity."""
        return self._num_trunk_rows


def packed_input_size(max_players: int) -> int:
    """The packed vector's length at capacity ``max_players``: what
    ``nn.get_model_input_spec`` reports as the spec's ``input_size``."""
    return int(get_input_size(max_players, layout_version=INPUT_LAYOUT_VERSION))


# ---------------------------------------------------------------------------
# The layout, from the engine's exports
# ---------------------------------------------------------------------------

class _Part(NamedTuple):
    offset: int
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        return math.prod(self.shape)


class _PairSlice(NamedTuple):
    """A row type's slice of a pair block: the block, the block axis the
    row is on, and the other axis's row type."""

    block: str
    axis: int
    other: str


class _BiasSpec(NamedTuple):
    """A pair block's rectangles and their columns of ``relation_bias_mult``:
    queries of the first type and keys of the second (``forward``), and the
    other way (``reverse``, None for a block of one type)."""

    block: str
    first: tuple[int, int]
    second: tuple[int, int]
    forward: slice
    reverse: slice | None


@dataclass(frozen=True)
class _Layout:
    parts: dict[str, _Part]
    row_types: tuple[str, ...]  # in trunk order
    rows: dict[str, tuple[int, int]]  # each row type's (first trunk row, count)
    pairs: dict[str, tuple[str, str]]  # each pair block's two row types
    num_trunk_rows: int


def _layout(max_players: int) -> _Layout:
    """The parts at capacity ``max_players``, and the structure they imply:
    row parts have one trunk range, pair blocks two, the choice table none."""
    exported = get_input_parts(max_players, layout_version=INPUT_LAYOUT_VERSION)
    parts = {p.name: _Part(int(p.offset), tuple(int(n) for n in p.shape)) for p in exported}
    rows = {p.name: (int(p.rows[0][0]), int(p.rows[0][1])) for p in exported if len(p.rows) == 1}
    by_range = {first_count: name for name, first_count in rows.items()}
    pairs = {
        p.name: (by_range[int(p.rows[0][0]), int(p.rows[0][1])], by_range[int(p.rows[1][0]), int(p.rows[1][1])])
        for p in exported if len(p.rows) == 2
    }
    row_types = tuple(sorted(rows, key=lambda name: rows[name][0]))
    num_trunk_rows = int(get_num_trunk_rows(max_players, layout_version=INPUT_LAYOUT_VERSION))

    offset = 0
    for p in exported:  # the parts chain
        assert p.offset == offset, f"part {p.name} at {p.offset}, expected {offset}"
        offset += math.prod(p.shape)
        assert len(p.rows) in (0, 1, 2), f"part {p.name} has {len(p.rows)} trunk ranges"
        for (_, count), extent in zip(p.rows, p.shape, strict=False):
            assert count == extent, f"part {p.name}: {count} trunk rows for an axis of {extent}"
    assert offset == get_input_size(max_players, layout_version=INPUT_LAYOUT_VERSION), "parts do not fill the input"
    first = 0
    for name in row_types:  # the row types tile the trunk
        assert rows[name][0] == first, f"row type {name} starts at {rows[name][0]}, expected {first}"
        assert len(parts[name].shape) == 2, f"row part {name} has shape {parts[name].shape}"
        first += rows[name][1]
    assert first == num_trunk_rows, f"row types cover {first} of {num_trunk_rows} trunk rows"
    for name in pairs:
        assert len(parts[name].shape) == 3, f"pair block {name} has shape {parts[name].shape}"
    assert [p.name for p in exported if not p.rows] == [_CHOICES], "the choice table is the one part off the trunk"
    assert parts[_CHOICES].shape == (int(ChoiceTable3.CHOICE3_ROWS), int(ChoiceTable3.CHOICE3_WIDTH))
    assert int(ChoiceTable3.CHOICE3_PHASE) + int(ChoiceTable3.CHOICE3_PHASE_WIDTH) == int(ChoiceTable3.CHOICE3_WIDTH)
    for block, channel in _BUILD_DISTANCE_CHANNELS.items():
        assert channel == parts[block].shape[-1] - 1, f"build distance is {block}'s last channel"
    for row_type, slots in _BUILD_DISTANCE_SLOTS.items():
        assert slots.stop == parts[row_type].shape[-1], f"build distance's slots end {row_type}'s row"
    assert rows[_PLAYER][0] + rows[_PLAYER][1] == num_trunk_rows, "players come last"
    assert _PLAYER_SEATS.stop - _PLAYER_SEATS.start >= 5, "a seat slot for every player at capacity 5"
    return _Layout(parts, row_types, rows, pairs, num_trunk_rows)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-RMSNorm block: RMSNorm -> MHSA -> residual, RMSNorm -> SwiGLU FFN -> residual.

    v1's block: the caller finishes the SDPA mask, pair-block bias and hidden
    keys together.
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class G1889TransformerNet(nn.Module):
    """Transformer with one choice head over the choice table and per-player values."""

    _type_ids: torch.Tensor
    _tile_table: torch.Tensor
    _choice_slot_index: torch.Tensor
    _choice_slot_valid: torch.Tensor
    _named_static: torch.Tensor
    _named_topmost: torch.Tensor
    _slot_phase: torch.Tensor
    _slot_row: torch.Tensor

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        layout = _layout(cfg.num_players)
        self._layout = layout
        parts = layout.parts

        # --- Each row type's features: its own slots, then its pair slices ---
        # Slot 0 is the key mask; CENTER's tile slots are looked up instead.
        self._own_slots = {
            name: [s for s in range(1, parts[name].shape[1])
                   if not (name == "CENTER" and s in (_CENTER_TILE, _CENTER_ROTATION))]
            for name in layout.row_types
        }
        num_seats = _PLAYER_SEATS.stop - _PLAYER_SEATS.start
        self._row_pairs: dict[str, list[_PairSlice]] = {name: [] for name in layout.row_types}
        widths = {name: len(slots) for name, slots in self._own_slots.items()}
        # Each type's feature columns that build distance fills.
        self._build_distance_columns: dict[str, list[int]] = {name: [] for name in layout.row_types}
        for name, slots in _BUILD_DISTANCE_SLOTS.items():
            self._build_distance_columns[name] += [self._own_slots[name].index(s) for s in slots]
        for block, (a, b) in layout.pairs.items():
            # A block of one type is one slice: its row's, which by symmetry
            # is its column's.
            for axis, (own, other) in enumerate(((a, b), (b, a))[:1 if a == b else 2]):
                self._row_pairs[own].append(_PairSlice(block, axis, other))
                count = num_seats if other == _PLAYER else layout.rows[other][1]
                channels = parts[block].shape[-1]
                bd = _BUILD_DISTANCE_CHANNELS.get(block)
                if bd is not None:
                    self._build_distance_columns[own] += [widths[own] + o * channels + bd for o in range(count)]
                widths[own] += count * channels
        self._feature_widths = widths

        # --- Input projections, identities, and types (v1's) ---
        tile_table = torch.from_numpy(get_tile_table())
        self._num_tile_codes, self._num_rotations, tile_width = tile_table.shape
        self.register_buffer("_tile_table", tile_table.flatten(0, 1).clone(), persistent=False)
        self.tile_proj = nn.Linear(tile_width, d, bias=False)
        self.row_proj = nn.ModuleDict({name: nn.Linear(widths[name], d) for name in layout.row_types})
        # Row order supplies identity without extra input data. GLOBAL has one
        # row; players' rows are in seat order and carry the seat.
        self.row_id_embed = nn.ModuleDict({
            name: nn.Embedding(layout.rows[name][1], d) for name in layout.row_types
            if name != _PLAYER and layout.rows[name][1] > 1
        })
        self.type_embeds = nn.Embedding(len(layout.row_types), d)
        self.register_buffer(
            "_type_ids",
            torch.repeat_interleave(
                torch.arange(len(layout.row_types)), torch.tensor([layout.rows[n][1] for n in layout.row_types]),
            ),
            persistent=False,
        )

        # --- Pair-block bias: a weight set per block and direction ---
        specs, column = [], 0
        rectangles: set[tuple[int, int]] = set()
        for block, (a, b) in layout.pairs.items():
            channels = parts[block].shape[-1] + (len(_DISTANCE_BUCKETS) - 1 if block in _BUILD_DISTANCE_CHANNELS else 0)
            forward = slice(column, column + channels)
            column += channels
            reverse = None
            if a != b:
                reverse = slice(column, column + channels)
                column += channels
            specs.append(_BiasSpec(block, layout.rows[a], layout.rows[b], forward, reverse))
            directions = [(layout.rows[a], layout.rows[b])] + ([(layout.rows[b], layout.rows[a])] if a != b else [])
            for (q0, qn), (k0, kn) in directions:
                cells = {(q, k) for q in range(q0, q0 + qn) for k in range(k0, k0 + kn)}
                assert not cells & rectangles, f"pair block {block} overlaps another's rectangle"
                rectangles |= cells
        self._bias_specs = specs
        # Build distance's buckets are the last of a block's channels.
        self._build_distance_bias = [
            c for spec in specs if spec.block in _BUILD_DISTANCE_CHANNELS
            for cols in (spec.forward, spec.reverse) if cols is not None
            for c in range(cols.stop - len(_DISTANCE_BUCKETS), cols.stop)
        ]
        self.relation_bias_mult = nn.Parameter(torch.zeros(cfg.num_layers, cfg.num_heads, column))

        # --- Transformer trunk ---
        d_ff = _round_up_to_multiple(math.ceil(cfg.ff_mult * d), 64)
        self.blocks = nn.ModuleList([TransformerBlock(d, cfg.num_heads, d_ff) for _ in range(cfg.num_layers)])
        self.final_norm = nn.RMSNorm(d)

        # --- The choice head (v2-design §5.4) ---
        choice_slots = torch.from_numpy(get_choice_slots()).long()  # (phases, rows)
        named_rows = torch.from_numpy(get_named_rows()).long()  # (unified slots,)
        self._num_phases, num_rows = choice_slots.shape
        self._num_features = int(ChoiceTable3.CHOICE3_WIDTH) - int(ChoiceTable3.CHOICE3_SHARED)
        self._validate_policy_layout(choice_slots, named_rows, num_rows)
        valid = choice_slots >= 0
        self.register_buffer("_choice_slot_index", choice_slots.clamp_min(0), persistent=False)
        self.register_buffer("_choice_slot_valid", valid.float().unsqueeze(-1), persistent=False)
        # Per (phase, row): the named trunk row as a one-hot, and AUCTION's
        # buy, whose row the topmost selector picks in the forward.
        named = torch.where(valid, named_rows[self._choice_slot_index], int(NamedRow3.NAMED3_NONE))
        self.register_buffer(
            "_named_static",
            F.one_hot(named.clamp_min(0), layout.num_trunk_rows).float() * (named >= 0).float().unsqueeze(-1),
            persistent=False,
        )
        self.register_buffer("_named_topmost", (named == int(NamedRow3.NAMED3_TOPMOST)).float(), persistent=False)
        # Each unified slot's phase and choice row, for laying the scores out.
        phase_of, row_of = valid.nonzero(as_tuple=True)
        order = choice_slots[phase_of, row_of].argsort()
        self.register_buffer("_slot_phase", phase_of[order].clone(), persistent=False)
        self.register_buffer("_slot_row", row_of[order].clone(), persistent=False)

        self.slot_embed = nn.Embedding(POLICY_LAYOUT.unified_dim, d)
        # W[phase]: the first layer over the shared and phase blocks.
        self.choice_phase_weight = nn.Parameter(torch.empty(self._num_phases, self._num_features, d))
        self.named_proj = nn.Linear(d, d)
        self.hex_embed = nn.Embedding(int(D.NUM_HEXES) + 1, d, padding_idx=0)  # code 0: no hex
        # The MLP over [g, a, c, s, h_k]: one GELU hidden layer of the trunk's width.
        self.choice_hidden = nn.Linear(5 * d, d)
        self.choice_out = nn.Linear(d, 1)
        # Each phase's W rows that build distance fills.
        shared = int(ChoiceTable3.CHOICE3_PHASE) - int(ChoiceTable3.CHOICE3_SHARED)
        self._build_distance_choice_rows = {
            phase: [shared + s for s in slots] for phase, slots in _BUILD_DISTANCE_PHASE_SLOTS.items()
        }

        # --- Value head: v1's, applied per player row ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _validate_policy_layout(self, choice_slots: torch.Tensor, named_rows: torch.Tensor, num_rows: int) -> None:
        """Choice row k of phase p is unified slot ``offsets[p] + k``, every
        slot is one row's, and no slot names a player row."""
        sizes = list(POLICY_LAYOUT.phase_sizes)
        assert self._num_phases == len(sizes), f"{self._num_phases} choice-slot phases for {len(sizes)}"
        assert num_rows == int(ChoiceTable3.CHOICE3_ROWS) >= max(sizes), f"{num_rows} choice rows"
        for phase, size in enumerate(sizes):
            expected = [POLICY_LAYOUT.offsets[phase] + k for k in range(size)] + [-1] * (num_rows - size)
            assert choice_slots[phase].tolist() == expected, (
                f"choice slots of phase {phase} do not match the policy blocks of PHASE_ACTION_SIZES {sizes}"
            )
        assert len(named_rows) == POLICY_LAYOUT.unified_dim, f"{len(named_rows)} named rows"
        players = self._layout.rows[_PLAYER][0]
        assert all(r in (int(NamedRow3.NAMED3_NONE), int(NamedRow3.NAMED3_TOPMOST)) or 0 <= r < players
                   for r in named_rows.tolist()), "a named row past the fixed rows"

    # ------------------------------------------------------------------
    # The packed input
    # ------------------------------------------------------------------

    def _parts(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Each part of ``(B, input_size)`` rows, as a ``(B, *shape)`` view."""
        return {
            name: raw[:, part.offset:part.offset + part.size].unflatten(1, part.shape)
            for name, part in self._layout.parts.items()
        }

    def _row_features(self, parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """``(B, rows, width)`` features per row type: its own slots, then its
        pair slices in part order, each flattened [other row][channel]. A
        slice over players is in seats after the actor."""
        seats = parts[_PLAYER][..., _PLAYER_SEATS]  # (B, P, seats): player p is seat s after the actor
        features = {}
        for name in self._layout.row_types:
            pieces = [parts[name][..., self._own_slots[name]]]
            for block, axis, other in self._row_pairs[name]:
                values = parts[block] if axis == 0 else parts[block].transpose(1, 2)
                if other == _PLAYER:
                    values = torch.einsum("bopc,bps->bosc", values, seats)
                pieces.append(values.flatten(2))
            features[name] = torch.cat(pieces, dim=-1)
        return features

    def _tiles(self, code: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """The projected tile block of each (tile code, rotation); zero for code 0."""
        index = (
            code.round().long().clamp(0, self._num_tile_codes - 1) * self._num_rotations
            + rotation.round().long().clamp(0, self._num_rotations - 1)
        )
        return F.embedding(index, self.tile_proj(self._tile_table))

    def _embed_rows(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """``(B, T, d)``: the trunk's rows, projected, with identity and type
        embeddings. Hidden rows project to their biases and embeddings; they
        are masked as keys, and no legal choice reads them."""
        features = self._row_features(parts)
        rows = []
        for name in self._layout.row_types:
            row = self.row_proj[name](features[name])
            if name in self.row_id_embed:
                row = row + _match(cast(nn.Embedding, self.row_id_embed[name]).weight, row)
            if name == "CENTER":
                center = parts[name]
                row = row + _match(self._tiles(center[..., _CENTER_TILE], center[..., _CENTER_ROTATION]), row)
            rows.append(row)
        tokens = torch.cat(rows, dim=1)
        return tokens + _match(self.type_embeds(self._type_ids), tokens)

    def _hidden_keys(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """``(B, 1, 1, T)``: the keys slot 0 hides."""
        visible = torch.cat([parts[name][..., _KEY_MASK] for name in self._layout.row_types], dim=1)
        return (visible <= 0.5)[:, None, None, :]

    def _selector(self, parts: dict[str, torch.Tensor], picks: dict[str, tuple[int, ...]]) -> torch.Tensor:
        """``(B, T)``: the sum of ``picks``' ``is_selected`` slots, each on its
        type's trunk rows."""
        columns = []
        for name in self._layout.row_types:
            rows = parts[name]
            column = torch.zeros_like(rows[..., _KEY_MASK])
            for slot in picks.get(name, ()):
                column = column + rows[..., slot]
            columns.append(column)
        return torch.cat(columns, dim=1)

    # ------------------------------------------------------------------
    # Pair-block bias
    # ------------------------------------------------------------------

    def _bias_channels(self, parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Each pair block's ``(B, rows, rows, channels)`` bias inputs, once for
        all layers: its channels, with build distance's as distance buckets."""
        channels = {}
        for block in self._layout.pairs:
            values = parts[block]
            bd = _BUILD_DISTANCE_CHANNELS.get(block)
            if bd is not None:
                values = torch.cat(
                    [values[..., :bd], values[..., bd + 1:], distance_buckets(values[..., bd])], dim=-1,
                )
            channels[block] = values
        return channels

    def _attention_bias(
        self,
        channels: dict[str, torch.Tensor],
        hidden_keys: torch.Tensor,
        layer_idx: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """One layer's finished SDPA mask, ``(B, H, T, T)``: zero but each pair
        block's rectangles, both ways, and the hidden keys."""
        mult = _match(self.relation_bias_mult[layer_idx], ref)  # (H, columns)
        T = self._layout.num_trunk_rows
        bias = ref.new_zeros(ref.shape[0], self.cfg.num_heads, T, T)
        for block, (a0, an), (b0, bn), forward, reverse in self._bias_specs:
            values = _match(channels[block], ref)
            bias[:, :, a0:a0 + an, b0:b0 + bn] = torch.einsum("bijc,hc->bhij", values, mult[:, forward])
            if reverse is not None:
                bias[:, :, b0:b0 + bn, a0:a0 + an] = torch.einsum("bijc,hc->bhji", values, mult[:, reverse])
        return bias.masked_fill_(hidden_keys, torch.finfo(bias.dtype).min)

    def _encode(self, raw: torch.Tensor) -> torch.Tensor:
        """The trunk: projections, the pair-biased blocks, and the final norm."""
        parts = self._parts(raw)
        tokens = self._embed_rows(parts)
        hidden_keys = self._hidden_keys(parts)
        channels = self._bias_channels(parts)
        for layer_idx, block in enumerate(self.blocks):
            tokens = block(tokens, self._attention_bias(channels, hidden_keys, layer_idx, tokens))
        return self.final_norm(tokens)

    # ------------------------------------------------------------------
    # The choice head
    # ------------------------------------------------------------------

    def _named_selector(self, parts: dict[str, torch.Tensor], phase: torch.Tensor) -> torch.Tensor:
        """``(B, rows, T)``: each choice row's named trunk row as a one-hot,
        zero for none; AUCTION's buy takes the topmost selector's row."""
        topmost = self._selector(parts, {"PRIVATE": (_PRIVATE_TOPMOST,)})
        return (
            torch.einsum("bp,pkt->bkt", phase, self._named_static)
            + torch.einsum("bp,pk->bk", phase, self._named_topmost).unsqueeze(-1) * topmost.unsqueeze(1)
        )

    def _choice_logits(self, tokens: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """``(B, unified_dim)`` fp32 logits, before the legal mask.

        Every row of the choice table is scored alone, whatever the phase; a
        slot of another phase than the state's gets 0, and the legal mask
        removes it.
        """
        parts = self._parts(raw)
        phase = parts["GLOBAL"][:, 0, _GLOBAL_PHASE:_GLOBAL_PHASE + self._num_phases]  # (B, phases), one-hot
        choices = parts[_CHOICES]
        phase_t = _match(phase, tokens)

        def select(selector: torch.Tensor) -> torch.Tensor:
            return torch.bmm(_match(selector, tokens).unsqueeze(1), tokens).squeeze(1)

        global_row = self._layout.rows["GLOBAL"][0]
        context = torch.cat([
            tokens[:, global_row],
            select(self._selector(parts, {_PLAYER: (_PLAYER_ACTOR,)})),
            select(self._selector(parts, {"CORP": (_CORP_DECISION,)})),
            select(self._selector(parts, _SUBJECT)),
        ], dim=-1)

        slots = self.slot_embed.weight[self._choice_slot_index] * self._choice_slot_valid  # (phases, rows, d)
        weight = (phase_t @ _match(self.choice_phase_weight.flatten(1), tokens)).unflatten(1, (self._num_features, -1))
        named = torch.bmm(_match(self._named_selector(parts, phase), tokens), self.named_proj(tokens))
        h = (
            torch.einsum("bp,pkd->bkd", phase_t, _match(slots, tokens))
            + torch.bmm(_match(choices[..., int(ChoiceTable3.CHOICE3_SHARED):], tokens), weight)
            + named
            + _match(self._tiles(choices[..., int(ChoiceTable3.CHOICE3_TILE)],
                                 choices[..., int(ChoiceTable3.CHOICE3_ROTATION)]), tokens)
            + _match(self.hex_embed(choices[..., int(ChoiceTable3.CHOICE3_HEX)].round().long()
                                    .clamp(0, self.hex_embed.num_embeddings - 1)), tokens)
        )
        # The MLP's first layer on [context, h_k], the context's part once a state.
        split = context.shape[-1]
        hidden = (
            F.linear(context, self.choice_hidden.weight[:, :split], self.choice_hidden.bias).unsqueeze(1)
            + F.linear(h, self.choice_hidden.weight[:, split:])
        )
        scores = self.choice_out(F.gelu(hidden, approximate=_GELU_APPROX)).squeeze(-1)  # (B, rows)
        on_phase = _match(phase, scores).index_select(1, self._slot_phase)
        return (scores.index_select(1, self._slot_row) * on_phase).float()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer.

        Args:
            inputs: ``(batch, cfg.input_size)`` packed layout-3 vectors, fp16 or
                fp32. They are read as fp32, which holds fp16 exactly; under
                autocast the projections and trunk run in the autocast dtype.
            legal_mask: ``(batch, unified_dim)`` bool tensor. ``True`` marks a
                slot legal; illegal slots are masked to ``-1e9``. Only a
                caller's scratch row, whose output it drops, may be all false.

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
        assert inputs.is_floating_point(), f"inputs must be floating-point; got {inputs.dtype}"
        assert legal_mask.dtype == torch.bool, (
            f"legal_mask must be bool (uint8 would make ~legal_mask a bitwise "
            f"complement, not logical NOT); got {legal_mask.dtype}"
        )
        if torch.compiler.is_compiling():
            # Eval marks each input batch dimension independently with
            # mark_unbacked. Establish their equality before Python shape
            # comparisons; otherwise Dynamo cannot decide u0 == u1.
            torch._check(legal_mask.shape[0] == inputs.shape[0])
        assert tuple(inputs.shape) == (inputs.shape[0], self.cfg.input_size), (
            f"inputs shape must be (batch, {self.cfg.input_size}); got {tuple(inputs.shape)}"
        )
        assert tuple(legal_mask.shape) == (inputs.shape[0], POLICY_LAYOUT.unified_dim), (
            f"legal_mask shape must be (batch, {POLICY_LAYOUT.unified_dim}); got {tuple(legal_mask.shape)}"
        )
        raw = inputs.float()
        tokens = self._encode(raw)

        # Cast to fp32 before the sentinel: under autocast the logits are in
        # the autocast dtype (bf16/fp16), but downstream softmax / log_softmax
        # wants stable fp32 ``-1e9`` sentinels.
        policy_logits = self._choice_logits(tokens, raw).masked_fill(~legal_mask, -1e9)

        first, count = self._layout.rows[_PLAYER]
        values = self.value_head(tokens[:, first:first + count]).squeeze(-1)
        if self.cfg.zero_sum_values:
            # Padded player rows are all-zero, including the key mask.
            present = (self._parts(raw)[_PLAYER][..., _KEY_MASK] > 0.5).to(values.dtype)
            count_present = present.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (values * present).sum(dim=1, keepdim=True) / count_present
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
        AdamW. ``named_proj``, which reads the trunk's final rows, and the
        heads' hidden layers stay Muon-eligible. ``choice_phase_weight`` (W)
        reads raw choice features; it is 3-D, which the trainer keeps on AdamW
        with decay whatever this list holds.
        """
        return [
            *self.row_proj.values(), self.tile_proj,
            self.choice_out, self.value_head[2],
        ]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT/LLaMA-style trunc-normal initialization, as v1, with zeros
        where an input is not filled yet."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)
        nn.init.trunc_normal_(self.choice_phase_weight, std=0.02)
        # Pair-block attention starts behavior-preserving; training can learn
        # positive or negative head/layer-specific biases from zero.
        nn.init.zeros_(self.relation_bias_mult)
        self._zero_unfilled_inputs()

    @torch.no_grad()
    def _zero_unfilled_inputs(self) -> None:
        """Zero the weights of inputs that are zero until a later step:
        build distance's, in the row projections, the bias, and W, and the
        hex embedding of code 0 (no hex). While their inputs are zero these
        weights get no gradient and stay zero."""
        for name, columns in self._build_distance_columns.items():
            cast(nn.Linear, self.row_proj[name]).weight[:, columns] = 0
        self.relation_bias_mult[:, :, self._build_distance_bias] = 0
        for phase, rows in self._build_distance_choice_rows.items():
            self.choice_phase_weight[phase, rows] = 0
        self.hex_embed.weight[0] = 0


def build_model(config: TrainingConfig, spec: ModelInputSpec) -> G1889TransformerNet:
    """The loader's entry point: the model of ``config``'s sizes for ``spec``."""
    assert spec.game == "1889", f"nn/transformer-1889-v2.py runs game 1889, not {spec.game!r}"
    assert spec.layout_version == INPUT_LAYOUT_VERSION, f"layout {spec.layout_version}, not {INPUT_LAYOUT_VERSION}"
    assert spec.input_size == packed_input_size(spec.num_players), (
        f"input_size {spec.input_size} for {spec.num_players} players"
    )
    assert spec.num_tokens == 0 and spec.token_dim == 0, (
        f"a packed input has no token rows; got {spec.num_tokens} x {spec.token_dim}"
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
