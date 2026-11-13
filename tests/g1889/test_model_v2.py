"""nn/transformer-1889-v2.py, the 1889 v2 model (1889/v2-design.md §4.3, §5.4,
§6; token-data-1889.md, "Layout 3").

On layout-3 vectors from ``fill_inputs_batch`` for random-playout states of
every decision phase: capacity 3 (3 players), 4 (3 and 4), and 5 (3 to 5),
with small models whose weights and bias multipliers are scrambled. The choice
table is written by these tests, so they hold whatever the engine fills there:
random features, hex codes, and tiles on each legal row, with build
distance's slots left zero.

- **Outputs** (v1-design §7's list on layout 3): legal logits finite and the
  rest -1e9; values zero-sum where asked; fp16 inputs are the fp32 values
  they round to; a batch equals its states alone; capacity changes nothing;
  hidden rows change nothing; the forward is deterministic.
- **Rows:** each packed value reaches exactly the rows the contract gives it,
  a pair block's both of its rows; each row's pair slices hold the entity
  facts; identities, types, and the tile lookup.
- **Pair blocks:** the bias is zero at initialization and changes nothing;
  each block reaches exactly its rectangles, both ways, with its own weights;
  build distance contributes nothing and gets no gradient while zero, and
  contributes once filled.
- **The choice head:** a logit moves with its own choice row only; illegal
  and zero rows change no other logit; the phase's W is the one used; a final
  row reaches the choices that name it, and the context rows reach all; the
  tile and hex code reach the logit; AUCTION's buy follows the topmost.
- Construction, the loader's contract, parameters, checkpoints, optimizer
  groups, gradients, and tracing.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._dynamo import explain
from torch._dynamo.decorators import mark_unbacked

import g1889.constants as K
import g1889.data.tables as D
from g1889.entities.corp import CORPS
from g1889.entities.graph import GRAPH
from g1889.entities.hexmap import MAP
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN
from g1889.routes.solver import max_route_revenue, route_hexes_py
from g1889.token_data import (
    ChoiceTable3 as C,
    TileBlock,
    TrunkRow3 as TR,
    fill_inputs_batch,
    get_input_parts,
    get_input_size,
    get_named_rows,
    get_tile_table,
)
from games.g1889 import POLICY_LAYOUT
from nn import _load_model_module, create_model, get_model_input_spec
from nn.policy_layout import PolicyLayout
from train.checkpoint import load_model_from_checkpoint, save_checkpoint
from train.config import TrainingConfig
from train.trainer import Trainer

import tests.g1889.test_model as T0
from tests.g1889.test_extraction import _state
from tests.g1889.test_model import Sample

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = str(REPO / 'nn' / 'transformer-1889-v2.py')
MODEL = _load_model_module(MODEL_PATH)
V1_PARAMETERS = 18_497_528  # tests/g1889/test_model_v1.py

LAYOUT = 3
OFFSETS = POLICY_LAYOUT.offsets
UNIFIED = POLICY_LAYOUT.unified_dim
D_MODEL, NUM_HEADS, NUM_LAYERS = T0.D_MODEL, T0.NUM_HEADS, T0.NUM_LAYERS

# The setups (players, capacity) and the groups the tests run on.
SETUPS = ((3, 3), (3, 4), (4, 4), (3, 5), (4, 5), (5, 5))
GROUPS = {'capacity 3': ((3, 3),), 'capacity 4': ((3, 4), (4, 4)), 'capacity 5': ((3, 5), (4, 5), (5, 5))}

# The contract's slots (token-data-1889.md, "Layout 3").
GLOBAL, CORP, PRIVATE, CENTER, PLAYER = (int(TR.TRUNK3_GLOBAL), int(TR.TRUNK3_CORP), int(TR.TRUNK3_PRIVATE),
                                         int(TR.TRUNK3_CENTER), int(TR.TRUNK3_PLAYER))
FIRST_ROW = {'GLOBAL': GLOBAL, 'CORP': CORP, 'PRIVATE': PRIVATE, 'CENTER': CENTER, 'PLAYER': PLAYER}
PHASES = slice(1, 1 + POLICY_LAYOUT.num_phases)  # GLOBAL: the decision phase, one-hot
SEATS = range(2, 7)  # PLAYER: seats after the actor, one-hot
SELECTIONS = {'CORP': (1, 2, 3), 'PRIVATE': (1, 2), 'CENTER': (1,), 'PLAYER': (1, *SEATS)}
SUBJECT = {'PRIVATE': (1,), 'CORP': (2, 3), 'CENTER': (1,)}
CENTER_TILE = (2, 3)
PHASE_COLUMN = int(C.CHOICE3_PHASE)
FEATURES = range(int(C.CHOICE3_SHARED), int(C.CHOICE3_WIDTH))
BD_CHANNELS = {'CORP_CENTER': 3, 'CENTER_CENTER': 0}
BD_CORP_SLOTS = range(70, 76)
BD_PHASE_SLOTS = {K.DPHASE_TRACK: range(26, 30), K.DPHASE_TILE: range(28, 37),
                  K.DPHASE_ER_BONUS: range(28, 37), K.DPHASE_TOKEN: range(5, 14)}
NUM_TILE_CODES = get_tile_table().shape[0]
PAIR_BLOCKS = ('CORP_PRIVATE', 'CORP_CENTER', 'CENTER_CENTER', 'PLAYER_CORP', 'PLAYER_PRIVATE')


def _parts(capacity: int) -> dict[str, Any]:
    return {part.name: part for part in get_input_parts(capacity, layout_version=LAYOUT)}


def _part(x: torch.Tensor, capacity: int, name: str) -> torch.Tensor:
    """``name``'s ``(B, *shape)`` view of packed rows ``x``."""
    part = _parts(capacity)[name]
    return x[:, part.offset:part.offset + math.prod(part.shape)].unflatten(1, part.shape)


# =============================================================================
# STATES AND INPUTS
# =============================================================================

@dataclass
class Inputs:
    x: torch.Tensor  # (B, input_size): the engine's vectors, the choice table synthetic
    real: torch.Tensor  # the engine's vectors as filled
    mask: torch.Tensor
    phases: torch.Tensor
    samples: list[Sample]
    max_players: int

    def part(self, name: str, x: torch.Tensor | None = None) -> torch.Tensor:
        return _part(self.x if x is None else x, self.max_players, name)

    def at(self, phase: int) -> list[int]:
        return (self.phases == phase).nonzero().flatten().tolist()

    def alone(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i:i + 1], self.mask[i:i + 1]


def _synthetic_choices(samples: list[Sample], seed: int) -> np.ndarray:
    """Each legal row's hex code, tile, and features at random; build
    distance's slots, and every other row, zero."""
    rng = np.random.default_rng(seed)
    choices = np.zeros((len(samples), int(C.CHOICE3_ROWS), int(C.CHOICE3_WIDTH)), np.float32)
    for i, sample in enumerate(samples):
        for k in sample.legal:
            row = choices[i, k]
            row[int(C.CHOICE3_HEX)] = rng.integers(1, D.NUM_HEXES + 1)
            row[int(C.CHOICE3_TILE)] = rng.integers(1, NUM_TILE_CODES)
            row[int(C.CHOICE3_ROTATION)] = rng.integers(0, D.NUM_SIDES)
            row[FEATURES.start:] = rng.normal(0, 0.5, len(FEATURES))
            row[[PHASE_COLUMN + s for s in BD_PHASE_SLOTS.get(sample.phase, ())]] = 0
    return choices


def _inputs(samples: list[Sample], capacity: int, seed: int = 0) -> Inputs:
    real = np.zeros((len(samples), get_input_size(capacity, layout_version=LAYOUT)), np.float32)
    fill_inputs_batch([s.row.copy() for s in samples], real, capacity, layout_version=LAYOUT)
    real_t = torch.from_numpy(real)
    x = real_t.clone()
    _part(x, capacity, 'CHOICES')[:] = torch.from_numpy(_synthetic_choices(samples, seed))
    mask = torch.zeros(len(samples), UNIFIED, dtype=torch.bool)
    for i, sample in enumerate(samples):
        mask[i, [OFFSETS[sample.phase] + k for k in sample.legal]] = True
    return Inputs(x, real_t, mask, torch.tensor([s.phase for s in samples]), samples, capacity)


@pytest.fixture(scope='module')
def setups() -> dict[tuple[int, int], list[Sample]]:
    return {setup: T0._sample_setup(*setup) for setup in SETUPS}


@pytest.fixture(scope='module')
def groups(setups) -> dict[str, Inputs]:
    return {name: _inputs([s for setup in members for s in setups[setup]], members[0][1])
            for name, members in GROUPS.items()}


@pytest.fixture(params=GROUPS)
def inputs(request, groups) -> Inputs:
    return groups[request.param]


def _scrambled(model):
    """Weights with scale: at initialization the bias is zero and the logits
    are tiny, which would hide most of what these tests look for. Build
    distance's weights are scrambled too; the hex embedding of code 0 stays
    zero, as training leaves it (``padding_idx``)."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.15)
    nn.init.normal_(model.relation_bias_mult, std=0.5)
    nn.init.normal_(model.choice_phase_weight, std=0.15)
    with torch.no_grad():
        model.hex_embed.weight[0] = 0
    return model


def _new_model(capacity: int, zero_sum: bool = False, **sizes):
    sizes = {'d_model': D_MODEL, 'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS, **sizes}
    return MODEL.G1889TransformerNet(MODEL.TransformerConfig(
        num_players=capacity, zero_sum_values=zero_sum, **sizes,
    )).eval()


@pytest.fixture(scope='module')
def weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return _scrambled(_new_model(5)).state_dict()


@pytest.fixture
def model_for(weights):
    def build(capacity: int, zero_sum: bool = False):
        model = _new_model(capacity, zero_sum)
        model.load_state_dict(weights)
        return model
    return build


@torch.inference_mode()
def _run(model, x, mask) -> tuple[torch.Tensor, torch.Tensor]:
    return model(x, mask)


@torch.inference_mode()
def _head(model, tokens, x) -> torch.Tensor:
    return model._choice_logits(tokens, x)


@torch.inference_mode()
def _encode(model, x) -> torch.Tensor:
    return model._encode(x.float())


def _assert_same(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...], tol: float = 2e-5) -> None:
    for left, right in zip(a, b, strict=True):
        assert torch.allclose(left, right, atol=tol, rtol=tol), (left - right).abs().max()


def _real_players(inputs: Inputs, x: torch.Tensor | None = None) -> torch.Tensor:
    return inputs.part('PLAYER', x)[..., 0] > 0.5


def _moved_slots(after: torch.Tensor, before: torch.Tensor) -> list[set[int]]:
    """Per state, the unified slots whose logits differ beyond rounding."""
    moved = (after - before).abs() > 1e-6 * (1 + before.abs())
    return [set(row.nonzero().flatten().tolist()) for row in moved]


def _slot(phase: int, k: int) -> int:
    return OFFSETS[phase] + k


def _phase_slots(phase: int) -> set[int]:
    return set(range(OFFSETS[phase], OFFSETS[phase + 1]))


# =============================================================================
# OUTPUTS
# =============================================================================

def test_the_states_reach_every_phase(groups):
    for inputs in groups.values():
        assert set(inputs.phases.tolist()) == set(range(POLICY_LAYOUT.num_phases))
    assert {s.num_players for s in groups['capacity 4'].samples} == {3, 4}
    assert {s.num_players for s in groups['capacity 5'].samples} == {3, 4, 5}
    # The vectors are the engine's: the phase one-hot (GLOBAL 1-19) is the sample's.
    for inputs in groups.values():
        phase = inputs.part('GLOBAL')[:, 0, PHASES]
        assert torch.equal(phase.argmax(1), inputs.phases) and (phase.sum(1) == 1).all()


def test_the_engines_vectors_run_at_every_phase(inputs, model_for):
    """The packed vectors as filled, whatever the choice table holds."""
    logits, values = _run(model_for(inputs.max_players), inputs.real, inputs.mask)
    assert torch.isfinite(logits[inputs.mask]).all() and (logits[~inputs.mask] == -1e9).all()
    assert torch.isfinite(values).all()


def test_legal_logits_are_finite_and_the_rest_masked(inputs, model_for):
    logits, values = _run(model_for(inputs.max_players), inputs.x, inputs.mask)
    assert logits.shape == inputs.mask.shape and logits.dtype == torch.float32
    assert values.shape == (len(inputs.samples), inputs.max_players)
    assert torch.isfinite(logits[inputs.mask]).all()
    assert (logits[~inputs.mask] == -1e9).all()
    assert (values.abs() <= 1).all()
    assert logits[inputs.mask].std() > 0.1


def test_values_are_zero_sum_over_real_players(inputs, model_for):
    _, raw = _run(model_for(inputs.max_players), inputs.x, inputs.mask)
    _, values = _run(model_for(inputs.max_players, zero_sum=True), inputs.x, inputs.mask)
    real = _real_players(inputs)
    assert (real.sum(1) == torch.tensor([s.num_players for s in inputs.samples])).all()
    assert (values[~real] == 0).all()
    assert values.sum(1).abs().max() < 1e-5
    for i in range(len(inputs.samples)):
        _assert_same((values[i][real[i]] - values[i][real[i]][0],), (raw[i][real[i]] - raw[i][real[i]][0],))


def test_fp16_inputs_give_the_fp32_values_they_round_to(inputs, model_for):
    model = model_for(inputs.max_players)
    half = inputs.x.to(torch.float16)
    assert not torch.equal(half.float(), inputs.x)  # some values round
    for a, b in zip(_run(model, half, inputs.mask), _run(model, half.float(), inputs.mask), strict=True):
        assert torch.equal(a, b)


def test_autocast_with_fp16_inputs_runs_the_eval_server_s_dtypes(inputs, model_for):
    """bf16 autocast on fp16 inputs: fp32 logits close to fp32's."""
    model = model_for(inputs.max_players)
    base_logits, base_values = _run(model, inputs.x, inputs.mask)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        logits, values = model(inputs.x.to(torch.float16), inputs.mask)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits[inputs.mask]).all() and (logits[~inputs.mask] == -1e9).all()
    assert (logits - base_logits)[inputs.mask].abs().max() < 0.5
    assert (values.float() - base_values).abs().max() < 0.1


def test_a_batch_equals_its_states_alone(inputs, model_for):
    model = model_for(inputs.max_players)
    logits, values = _run(model, inputs.x, inputs.mask)
    for i in range(len(inputs.samples)):
        _assert_same(_run(model, *inputs.alone(i)), (logits[i:i + 1], values[i:i + 1]))


def test_capacity_changes_nothing_for_the_same_weights(setups, model_for):
    """The same games and weights at each capacity, the padded seats hidden."""
    for players, capacities in ((3, (3, 4, 5)), (4, (4, 5))):
        runs = []
        for capacity in capacities:
            at = _inputs(setups[players, capacity], capacity)
            logits, values = _run(model_for(capacity), at.x, at.mask)
            runs.append((logits, values[:, :players]))
            assert (values[:, players:].abs() <= 1).all()
        for run in runs[1:]:
            _assert_same(run, runs[0])


def test_hidden_rows_change_nothing(inputs, model_for):
    """Junk in every hidden row's slots, but slot 0 and the selections: no
    legal logit or real player's value moves. Made visible, it would."""
    model = model_for(inputs.max_players)
    base_logits, base_values = _run(model, inputs.x, inputs.mask)
    real = _real_players(inputs)
    junk_x = inputs.x.clone()
    rng = torch.Generator().manual_seed(3)
    hidden_rows = 0
    for name in ('PRIVATE', 'PLAYER'):
        rows = inputs.part(name, junk_x)
        hidden = rows[..., 0] <= 0.5
        hidden_rows += int(hidden.sum())
        noise = 1 + torch.rand(rows.shape, generator=rng)
        noise[..., [0, *SELECTIONS[name]]] = 0
        rows[:] = torch.where(hidden[..., None], noise, rows)
    assert hidden_rows > 0
    logits, values = _run(model, junk_x, inputs.mask)
    _assert_same((logits, values * real), (base_logits, base_values * real))
    visible = junk_x.clone()
    for name in ('PRIVATE', 'PLAYER'):
        rows = inputs.part(name, visible)
        rows[..., 0] = torch.where(rows[..., 1:].abs().sum(-1) > 0, torch.ones_like(rows[..., 0]), rows[..., 0])
    moved, _ = _run(model, visible, inputs.mask)
    assert (moved - base_logits)[inputs.mask].abs().max() > 0.1


def test_the_forward_is_deterministic(groups):
    inputs = groups['capacity 5']
    torch.manual_seed(1)
    a = _new_model(5)
    torch.manual_seed(1)
    b = _new_model(5)
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values(), strict=True))
    first = _run(_scrambled(a), inputs.x, inputs.mask)
    assert all(torch.equal(p, q) for p, q in zip(first, _run(a, inputs.x, inputs.mask), strict=True))


# =============================================================================
# ROWS
# =============================================================================

def _seat(x_row_players: torch.Tensor, i: int) -> bool:
    """Whether player row i has a seat after the actor (PLAYER 2-6)."""
    return bool(x_row_players[i, SEATS.start:SEATS.stop].sum() > 0.5)


@pytest.mark.parametrize('capacity', (3, 5))
def test_each_input_value_reaches_exactly_its_rows(capacity, groups, model_for):
    """A row slot reaches its own row, a pair-block value both of its rows (a
    player's value reaches the other row only from a seated player), and the
    choice table no row. Slot 0 reaches none. A seat slot turns the corps' and
    privates' player slices. The CENTER tile slots have their own test."""
    inputs = groups[f'capacity {capacity}']
    model = model_for(capacity)
    i = next(i for i, s in enumerate(inputs.samples) if s.num_players == 3 and s.phase == K.DPHASE_BUY_TRAIN)
    x = inputs.x[i:i + 1]
    players = inputs.part('PLAYER', x)[0]
    with torch.inference_mode():
        base = model._embed_rows(model._parts(x))
    for part in get_input_parts(capacity, layout_version=LAYOUT):
        size = math.prod(part.shape)
        edited = x.repeat(size, 1)
        edited[torch.arange(size), part.offset + torch.arange(size)] += 0.5
        with torch.inference_mode():
            moved = (model._embed_rows(model._parts(edited)) - base).abs().sum(-1) > 1e-5
        for index in range(size):
            cell = np.unravel_index(index, part.shape)
            rows = set(moved[index].nonzero().flatten().tolist())
            if part.name == 'CHOICES':
                assert not rows, (part.name, cell, rows)
                continue
            if len(part.rows) == 1:
                row, slot = part.rows[0][0] + int(cell[0]), int(cell[1])
                if (part.name, slot) in {('CENTER', s) for s in CENTER_TILE}:
                    continue
                if part.name == 'PLAYER' and slot in SEATS:
                    assert row in rows and rows - {row} <= set(range(CORP, CENTER)), (part.name, cell, rows)
                    continue
                expected = set() if slot == 0 else {row}
            else:
                (a0, _), (b0, _) = part.rows
                a, b = a0 + int(cell[0]), b0 + int(cell[1])
                seated = a0 != PLAYER or _seat(players, int(cell[0]))
                expected = {a} if a0 == b0 else {a} | ({b} if seated else set())
            assert rows == expected, (part.name, cell, rows, expected)


def test_identities_and_types_move_their_own_rows(groups, model_for):
    inputs = groups['capacity 5']
    model = model_for(5)
    x = inputs.x[:1]

    def moved_rows(embed, k):
        with torch.inference_mode():
            base = model._embed_rows(model._parts(x))
            embed.weight.data[k] += 0.5
            moved = model._embed_rows(model._parts(x))
            embed.weight.data[k] -= 0.5
        return set(((moved - base).abs().sum(-1) > 1e-5)[0].nonzero().flatten().tolist())

    assert set(model.row_id_embed) == {'CORP', 'PRIVATE', 'CENTER'}
    for name, count in (('CORP', D.NUM_CORPS), ('PRIVATE', D.NUM_PRIVATES), ('CENTER', D.NUM_RCS)):
        assert model.row_id_embed[name].num_embeddings == count
        assert moved_rows(model.row_id_embed[name], count - 1) == {FIRST_ROW[name] + count - 1}
    assert moved_rows(model.type_embeds, 0) == {GLOBAL}
    assert moved_rows(model.type_embeds, 3) == set(range(CENTER, PLAYER))
    assert moved_rows(model.type_embeds, 4) == set(range(PLAYER, PLAYER + 5))


def test_centers_and_choices_share_the_tile_lookup(groups, model_for):
    """A CENTER row's tile slots add the tile projection of the tile table's
    block, and nothing else; code 0 adds nothing. A choice's tile reaches its
    logit through the same projection."""
    inputs = groups['capacity 5']
    model = model_for(5)
    table = torch.from_numpy(get_tile_table())
    assert table.shape == (NUM_TILE_CODES, D.NUM_SIDES, int(TileBlock.TW_TILE_BLOCK)) and not table[0].any()
    assert model.tile_proj.bias is None and model.tile_proj.in_features == table.shape[-1]
    x = inputs.x[:1].clone()
    center = inputs.part('CENTER', x)
    r = 4
    with torch.inference_mode():
        center[0, r, list(CENTER_TILE)] = 0
        bare = model._embed_rows(model._parts(x))
        for code, rotation in ((7, 0), (40, 3), (92, 5)):
            center[0, r, list(CENTER_TILE)] = torch.tensor([code, rotation], dtype=x.dtype)
            moved = model._embed_rows(model._parts(x)) - bare
            assert torch.allclose(moved[0, CENTER + r], model.tile_proj(table[code, rotation]), atol=1e-5)
            assert (moved[0].abs().sum(-1) > 1e-6).nonzero().flatten().tolist() == [CENTER + r]

    # The choice side: with the tile projection zeroed, a choice's tile moves nothing.
    i = inputs.at(K.DPHASE_TILE)[0]
    x, k = inputs.x[i:i + 1].clone(), inputs.samples[i].legal[0]
    tokens = _encode(model, x)
    base = _head(model, tokens, x)
    edited = x.clone()
    inputs.part('CHOICES', edited)[0, k, int(C.CHOICE3_TILE)] = 1 + inputs.part('CHOICES', x)[0, k, 1] % 90
    assert _moved_slots(_head(model, tokens, edited), base)[0] == {_slot(K.DPHASE_TILE, k)}
    with torch.no_grad():
        model.tile_proj.weight.zero_()
    assert torch.equal(_head(model, tokens, edited), _head(model, tokens, x))


def _facts(sample: Sample, capacity: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The pair blocks from the entity APIs, and each row type's slices of
    them as the contract orders them: per CORP, its privates, its centers'
    channels, then its players' by seat after the actor; per PRIVATE, its
    corps, then its players' by seat; per CENTER, its corps' channels, then
    the other centers; per PLAYER, its corps' channels, then its privates'.
    Build distance is zero."""
    state = _state(sample.row.copy())
    n = sample.num_players
    actor = TURN.active_player(state)
    for c in range(D.NUM_CORPS):
        max_route_revenue(state, c)
    route_hexes = [set(route_hexes_py(state, c)) for c in range(D.NUM_CORPS)]
    owners = [PRIVATES[p].owner(state) for p in range(D.NUM_PRIVATES)]
    in_play = [owner not in (K.OWNER_NOT_IN_PLAY, K.OWNER_CLOSED) for owner in owners]
    player_corp = np.zeros((capacity, D.NUM_CORPS, 3))
    player_private = np.zeros((capacity, D.NUM_PRIVATES, 2))
    seated = np.zeros((5, capacity))  # [seat after the actor, player]
    for i in range(n):
        seated[(i - actor) % n, i] = 1
        for c in range(D.NUM_CORPS):
            player_corp[i, c] = (PLAYERS[i].shares(state, c) / 10, CORPS[c].president(state) == i,
                                 PLAYERS[i].sold_mask(state) >> c & 1)
        for p in range(D.NUM_PRIVATES):
            if in_play[p]:
                player_private[i, p] = (owners[p] == K.OWNER_PLAYER + i, PLAYERS[i].bid(state, p) / 100)
    corp_private = np.array([[float(in_play[p] and owners[p] == K.OWNER_CORP + c) for p in range(D.NUM_PRIVATES)]
                             for c in range(D.NUM_CORPS)])
    corp_center = np.array([[(MAP.tokens(state, h) >> c & 1, GRAPH.node_reached(state, c, h), h in route_hexes[c], 0)
                              for h in D.RC_HEX] for c in range(D.NUM_CORPS)], dtype=float)
    blocks = {'PLAYER_CORP': player_corp, 'PLAYER_PRIVATE': player_private, 'CORP_PRIVATE': corp_private,
              'CORP_CENTER': corp_center}
    slices = {
        'CORP': np.concatenate([corp_private, corp_center.reshape(D.NUM_CORPS, -1),
                                np.einsum('sp,pcx->csx', seated, player_corp).reshape(D.NUM_CORPS, -1)], axis=1),
        'PRIVATE': np.concatenate([corp_private.T,
                                   np.einsum('sp,pqx->qsx', seated, player_private).reshape(D.NUM_PRIVATES, -1)],
                                  axis=1),
        'CENTER': np.concatenate([corp_center.transpose(1, 0, 2).reshape(D.NUM_RCS, -1),
                                  np.zeros((D.NUM_RCS, D.NUM_RCS))], axis=1),
        'PLAYER': np.concatenate([player_corp.reshape(capacity, -1), player_private.reshape(capacity, -1)], axis=1),
    }
    return blocks, slices


def test_each_rows_pair_slices_are_the_entity_facts(groups):
    """On the engine's vectors: every row's features are its own slots (but
    slot 0 and CENTER's tile slots), then its slice of each pair block, the
    player axis turned to start at the actor. The facts are shares held,
    presidencies, sales, private owners and bids, tokens, reach, and routes,
    and each occurs in the states."""
    for inputs in groups.values():
        model = _new_model(inputs.max_players)
        with torch.inference_mode():
            features = model._row_features(model._parts(inputs.real))
        seen: dict[str, np.ndarray] = {}
        for i, sample in enumerate(inputs.samples):
            blocks, slices = _facts(sample, inputs.max_players)
            for name, width in (('GLOBAL', 135), ('CORP', 76), ('PRIVATE', 15), ('CENTER', 21), ('PLAYER', 20)):
                own = [s for s in range(1, width) if not (name == 'CENTER' and s in CENTER_TILE)]
                expected = inputs.part(name, inputs.real)[i][:, own].numpy()
                if name in slices:
                    expected = np.concatenate([expected, slices[name]], axis=1)
                np.testing.assert_allclose(features[name][i].numpy(), expected, atol=1e-6, err_msg=f'{name} {i}')
            for block, values in blocks.items():
                channels = np.abs(values).reshape(-1, values.shape[-1] if values.ndim == 3 else 1).sum(0)
                seen[block] = seen.get(block, 0) + (channels > 0)
        bd = BD_CHANNELS['CORP_CENTER']
        assert all(count.all() for block, count in seen.items() if block != 'CORP_CENTER'), seen
        assert np.delete(seen['CORP_CENTER'], bd).all() and seen['CORP_CENTER'][bd] == 0, seen


# =============================================================================
# PAIR BLOCKS
# =============================================================================

def _bias(model, x, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    parts = model._parts(x)
    hidden = model._hidden_keys(parts)
    with torch.inference_mode():
        ref = torch.zeros(x.shape[0], 1, model.cfg.d_model)
        return model._attention_bias(model._bias_channels(parts), hidden, layer, ref), hidden.expand(
            -1, model.cfg.num_heads, model.cfg.num_trunk_rows, -1)


def test_at_initialization_the_bias_is_zero_and_changes_nothing(groups):
    inputs = groups['capacity 5']
    model = _new_model(5)
    assert not model.relation_bias_mult.any()
    for layer in range(NUM_LAYERS):
        bias, hidden = _bias(model, inputs.x, layer)
        assert (bias[~hidden] == 0).all() and (bias[hidden] == torch.finfo(bias.dtype).min).all()
    base = _run(model, inputs.x, inputs.mask)
    plain = copy.deepcopy(model)
    plain._attention_bias = lambda channels, hidden, layer, ref: torch.zeros(  # the key mask alone
        ref.shape[0], NUM_HEADS, ref.shape[1], ref.shape[1]).masked_fill(hidden, torch.finfo(ref.dtype).min)
    assert all(torch.equal(a, b) for a, b in zip(_run(plain, inputs.x, inputs.mask), base, strict=True))
    # With weights, the bias moves every state that has a pair-block value.
    with torch.no_grad():
        model.relation_bias_mult.normal_(std=0.5)
    moved = _moved_slots(_run(model, inputs.x, inputs.mask)[0], base[0])
    paired = [any(_part(inputs.x[i:i + 1], 5, name).any() for name in PAIR_BLOCKS) for i in range(len(moved))]
    assert all(bool(slots) == has for slots, has in zip(moved, paired, strict=True))
    assert sum(paired) > len(paired) // 2


def _with_build_distance(inputs: Inputs, seed: int = 5) -> torch.Tensor:
    """Synthetic build distance: closeness of every distance in the two
    channels (CENTER x CENTER symmetric, its diagonal 0), CORP 70-75, and the
    phase blocks' bd slots."""
    rng = np.random.default_rng(seed)
    x = inputs.x.clone()
    closeness = np.array([0, *((6 - d) / 6 for d in range(6))], np.float32)
    corp_center = inputs.part('CORP_CENTER', x)
    corp_center[..., BD_CHANNELS['CORP_CENTER']] = torch.from_numpy(rng.choice(closeness, corp_center.shape[:3]))
    centers = np.triu(rng.choice(closeness, (len(x), D.NUM_RCS, D.NUM_RCS)), 1)
    inputs.part('CENTER_CENTER', x)[..., 0] = torch.from_numpy(centers + centers.transpose(0, 2, 1))
    inputs.part('CORP', x)[..., list(BD_CORP_SLOTS)] = torch.rand(len(x), D.NUM_CORPS, len(BD_CORP_SLOTS))
    choices = inputs.part('CHOICES', x)
    for i, sample in enumerate(inputs.samples):
        for s in BD_PHASE_SLOTS.get(sample.phase, ()):
            choices[i, sample.legal, PHASE_COLUMN + s] = torch.rand(len(sample.legal))
    return x


def _expected_channels(values: torch.Tensor, bd: int | None) -> torch.Tensor:
    """A block's bias inputs: its channels, build distance's as a one-hot of
    the distance, 4 and 5 lays together."""
    if bd is None:
        return values
    closeness = values[..., bd]
    lays = torch.round(6 - 6 * closeness).long().clamp(max=4)
    one_hot = torch.nn.functional.one_hot(lays, 7)[..., :5].float() * (closeness > 0).unsqueeze(-1)
    return torch.cat([values[..., :bd], values[..., bd + 1:], one_hot], dim=-1)


def test_each_block_reaches_exactly_its_rectangles_both_ways(groups, model_for):
    """With one block's weights for one direction, the bias is that block's
    channels times them in its rectangle, queries of the direction's first
    type, and zero elsewhere; each direction has its own weights. A block of
    one type is one rectangle."""
    inputs = groups['capacity 5']
    model = model_for(5)
    x = _with_build_distance(inputs)
    layer = 1
    full = model.relation_bias_mult.data.clone()
    seen = set()
    for spec in model._bias_specs:
        block = _parts(5)[spec.block]
        (a0, an), (b0, bn) = block.rows
        values = _expected_channels(_part(x, 5, spec.block), BD_CHANNELS.get(spec.block))
        assert (spec.first, spec.second) == ((a0, an), (b0, bn))
        directions = [(spec.forward, 'bijc,hc->bhij', (a0, an, b0, bn))]
        if a0 != b0:
            directions.append((spec.reverse, 'bijc,hc->bhji', (b0, bn, a0, an)))
        else:
            assert spec.reverse is None and spec.block == 'CENTER_CENTER'
        for columns, equation, (q0, qn, k0, kn) in directions:
            assert columns is not None and columns.stop - columns.start == values.shape[-1]
            model.relation_bias_mult.data.zero_()
            model.relation_bias_mult.data[layer, :, columns] = full[layer, :, columns]
            bias, hidden = _bias(model, x, layer)
            expected = torch.zeros_like(bias)
            expected[:, :, q0:q0 + qn, k0:k0 + kn] = torch.einsum(equation, values, full[layer, :, columns])
            assert torch.allclose(bias[~hidden], expected[~hidden], atol=1e-5), spec.block
            assert expected[:, :, q0:q0 + qn, k0:k0 + kn][~hidden[:, :, q0:q0 + qn, k0:k0 + kn]].abs().max() > 0.1
            seen.add((spec.block, q0))
    model.relation_bias_mult.data.copy_(full)
    assert len(seen) == 9 and {spec.block for spec in model._bias_specs} == set(BD_CHANNELS) | {
        'CORP_PRIVATE', 'PLAYER_CORP', 'PLAYER_PRIVATE'}
    # The blocks' columns tile the multiplier.
    columns = sorted(c for spec in model._bias_specs for s in (spec.forward, spec.reverse) if s
                     for c in range(s.start, s.stop))
    assert columns == list(range(model.relation_bias_mult.shape[-1])) == list(range(2 * (3 + 2 + 1 + 8) + 5))


def test_distance_buckets_are_one_hots_of_distance():
    closeness = torch.tensor([0.0, *((6 - d) / 6 for d in range(6))])
    expected = torch.tensor([[0] * 5, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=torch.float32)
    assert torch.equal(MODEL.distance_buckets(closeness), expected)
    assert torch.equal(MODEL.distance_buckets(closeness.half().float()), expected)


def _build_distance_weights(model) -> dict[str, torch.Tensor]:
    """Every weight that reads build distance: CORP's and CENTER's
    projection columns, the bias's bucket columns, and W's bd rows."""
    out = {}
    for name, columns in model._build_distance_columns.items():
        if columns:
            out[f'row_proj.{name}'] = model.row_proj[name].weight[:, columns]
    out['relation_bias_mult'] = model.relation_bias_mult[:, :, model._build_distance_bias]
    for phase, rows in model._build_distance_choice_rows.items():
        out[f'W[{phase}]'] = model.choice_phase_weight[phase, rows]
    return out


def test_build_distance_columns_are_the_contracts(model_for):
    model = model_for(5)
    columns = model._build_distance_columns
    assert set(name for name, c in columns.items() if c) == {'CORP', 'CENTER'}
    # CORP: its own 70-75, then channel 3 of each of its 25 centers (after its 7 privates).
    assert columns['CORP'] == [s - 1 for s in BD_CORP_SLOTS] + [75 + 7 + 4 * r + 3 for r in range(D.NUM_RCS)]
    # CENTER: channel 3 of each of its 7 corps, then every other center.
    assert columns['CENTER'] == [18 + 4 * c + 3 for c in range(D.NUM_CORPS)] + list(range(18 + 28, 18 + 28 + 25))
    assert {phase: [r - 7 for r in rows] for phase, rows in model._build_distance_choice_rows.items()} == {
        p: list(s) for p, s in BD_PHASE_SLOTS.items()}
    assert len(model._build_distance_bias) == 3 * 5


def test_build_distance_contributes_nothing_and_learns_nothing_while_zero(groups, model_for):
    """At initialization its weights are zero. With them scrambled, the
    engine's zeros make the outputs those of zero weights, and the weights
    get no gradient, whatever else is trained."""
    inputs = groups['capacity 5']
    assert all(not w.any() for w in _build_distance_weights(_new_model(5)).values())
    model = model_for(5)
    assert all(w.abs().min() > 0 for w in _build_distance_weights(model).values() if w.numel())
    zeroed = copy.deepcopy(model)
    zeroed._zero_unfilled_inputs()
    for a, b in zip(_run(model, inputs.x, inputs.mask), _run(zeroed, inputs.x, inputs.mask), strict=True):
        assert torch.equal(a, b)
    model.train()
    logits, values = model(inputs.x, torch.ones_like(inputs.mask))
    (logits.sum() + values.square().sum()).backward()
    grads = {
        **{f'row_proj.{name}': model.row_proj[name].weight.grad[:, columns]
           for name, columns in model._build_distance_columns.items() if columns},
        'relation_bias_mult': model.relation_bias_mult.grad[:, :, model._build_distance_bias],
        **{f'W[{phase}]': model.choice_phase_weight.grad[phase, rows]
           for phase, rows in model._build_distance_choice_rows.items()},
    }
    assert all(not g.any() for g in grads.values()), [k for k, g in grads.items() if g.any()]
    assert model.relation_bias_mult.grad.abs().sum() > 0 and model.choice_phase_weight.grad.abs().sum() > 0


def test_build_distance_contributes_once_filled(groups, model_for):
    """Filled, it moves the outputs through scrambled weights, and from
    initialization its weights get gradient."""
    inputs = groups['capacity 5']
    x = _with_build_distance(inputs)
    model = model_for(5)
    base = _run(model, inputs.x, inputs.mask)
    filled = _run(model, x, inputs.mask)
    assert (filled[0] - base[0])[inputs.mask].abs().max() > 0.1 and (filled[1] - base[1]).abs().max() > 0.01
    # Each input moves its rows: the channels both of a pair's rows, CORP 70-75 its own.
    with torch.inference_mode():
        moved = (model._embed_rows(model._parts(x)) - model._embed_rows(model._parts(inputs.x))).abs().sum(-1) > 1e-5
    assert moved[:, CORP:PRIVATE].all() and moved[:, CENTER:PLAYER].all() and not moved[:, PRIVATE:CENTER].any()
    bias, hidden = _bias(model, x, 0)
    bias_base, _ = _bias(model, inputs.x, 0)
    changed = (bias - bias_base).abs() > 1e-6
    assert changed[:, :, CORP:PRIVATE, CENTER:PLAYER].any() and changed[:, :, CENTER:PLAYER, CORP:PRIVATE].any()
    assert changed[:, :, CENTER:PLAYER, CENTER:PLAYER].any()
    assert not changed[:, :, :CORP].any() and not changed[:, :, PLAYER:].any()

    fresh = _new_model(5).train()
    logits, values = fresh(x, torch.ones_like(inputs.mask))
    (logits.sum() + values.square().sum()).backward()
    grads = {
        **{name: fresh.row_proj[name].weight.grad[:, columns]
           for name, columns in fresh._build_distance_columns.items() if columns},
        'relation_bias_mult': fresh.relation_bias_mult.grad[:, :, fresh._build_distance_bias],
        **{f'W[{phase}]': fresh.choice_phase_weight.grad[phase, rows]
           for phase, rows in fresh._build_distance_choice_rows.items()},
    }
    assert all(g.abs().sum() > 0 for g in grads.values()), [k for k, g in grads.items() if not g.any()]


# =============================================================================
# THE CHOICE HEAD
# =============================================================================

def _per_phase(inputs: Inputs, count: int = 2) -> list[int]:
    return [i for phase in range(POLICY_LAYOUT.num_phases) for i in inputs.at(phase)[:count]]


def test_a_choice_logit_moves_with_its_own_row_only(inputs, model_for):
    """Each legal row's features, edited alone, move its own slot and no other
    logit or value, through the whole forward."""
    model = model_for(inputs.max_players)
    states, rows = [], []
    for i in _per_phase(inputs):
        for k in inputs.samples[i].legal:
            states.append(i)
            rows.append(k)
    x = inputs.x[states].clone()
    mask = inputs.mask[states]
    base = _run(model, x, mask)
    choices = inputs.part('CHOICES', x)
    choices[torch.arange(len(states)), rows, FEATURES.start:] += 0.5
    logits, values = _run(model, x, mask)
    for n, (i, k) in enumerate(zip(states, rows, strict=True)):
        assert _moved_slots(logits[n:n + 1], base[0][n:n + 1])[0] == {_slot(inputs.samples[i].phase, k)}, (i, k)
    assert torch.equal(values, base[1])


def test_an_illegal_or_zero_row_changes_no_other_logit(inputs, model_for):
    """Zeroing a legal row, junk in the rows past the phase's size, or
    dropping a row from the mask: no other logit moves."""
    model = model_for(inputs.max_players)
    sizes = POLICY_LAYOUT.phase_sizes
    for i in _per_phase(inputs, 1):
        x, mask = inputs.alone(i)
        sample = inputs.samples[i]
        base, _ = _run(model, x, mask)
        k = sample.legal[-1]
        zeroed = x.clone()
        inputs.part('CHOICES', zeroed)[0, k] = 0
        assert _moved_slots(_run(model, zeroed, mask)[0], base)[0] <= {_slot(sample.phase, k)}
        junk = x.clone()
        inputs.part('CHOICES', junk)[0, sizes[sample.phase]:] = 3.0
        assert torch.equal(_run(model, junk, mask)[0], base)
        dropped = mask.clone()
        dropped[0, _slot(sample.phase, k)] = False
        logits, _ = _run(model, x, dropped)
        assert _moved_slots(logits, base)[0] == {_slot(sample.phase, k)} and logits[0, _slot(sample.phase, k)] == -1e9


def test_the_decision_phases_w_is_the_one_used(groups, model_for):
    """Moving W[p] moves the logits of the states at phase p, and those of no
    other state."""
    inputs = groups['capacity 5']
    model = model_for(5)
    tokens = _encode(model, inputs.x)
    base = _head(model, tokens, inputs.x)
    for phase in range(POLICY_LAYOUT.num_phases):
        with torch.no_grad():
            model.choice_phase_weight[phase] += 0.3
        moved = _moved_slots(_head(model, tokens, inputs.x), base)
        with torch.no_grad():
            model.choice_phase_weight[phase] -= 0.3
        for i, slots in enumerate(moved):
            if inputs.samples[i].phase == phase:
                assert slots == {_slot(phase, k) for k in inputs.samples[i].legal}, (phase, i, slots)
            else:
                assert not slots, (phase, i, slots)


def _context_rows(inputs: Inputs, i: int) -> set[int]:
    """GLOBAL, the actor, the decision corp, and the subject, from the
    state's selections."""
    def picked(name, slots):
        rows = inputs.part(name)[i]
        return {FIRST_ROW[name] + r for r in range(rows.shape[0]) if any(rows[r, s] > 0.5 for s in slots)}
    rows = {GLOBAL} | picked('PLAYER', (1,)) | picked('CORP', (1,))
    for name, slots in SUBJECT.items():
        rows |= picked(name, slots)
    return rows


def _named(inputs: Inputs, i: int) -> dict[int, set[int]]:
    """Each trunk row's choices at the state's decision, by the engine's
    named rows, AUCTION's buy by the topmost selector."""
    phase = inputs.samples[i].phase
    named_rows = get_named_rows()
    topmost = [PRIVATE + p for p in range(D.NUM_PRIVATES) if inputs.part('PRIVATE')[i, p, 2] > 0.5]
    out: dict[int, set[int]] = {}
    for k in range(POLICY_LAYOUT.phase_sizes[phase]):
        row = int(named_rows[_slot(phase, k)])
        for target in topmost if row == -2 else [row] if row >= 0 else []:
            out.setdefault(target, set()).add(_slot(phase, k))
    return out


def test_a_final_row_reaches_the_choices_that_name_it(groups, model_for):
    """With the trunk's output fixed, moving a final row moves the logits of
    the choices that name it, every logit of the decision if it is context
    (GLOBAL, the actor, the decision corp, the subject), and nothing else."""
    inputs = groups['capacity 5']
    model = model_for(5)
    num_rows = model.cfg.num_trunk_rows
    states = _per_phase(inputs, 1)
    tokens = _encode(model, inputs.x[states])
    x = inputs.x[states]
    base = _head(model, tokens, x)
    edited = tokens.repeat_interleave(num_rows, 0)
    edited[torch.arange(len(edited)), torch.arange(num_rows).repeat(len(states))] += 0.5
    moved = _moved_slots(_head(model, edited, x.repeat_interleave(num_rows, 0)), base.repeat_interleave(num_rows, 0))
    named_somewhere = 0
    for n, i in enumerate(states):
        phase = inputs.samples[i].phase
        context, named = _context_rows(inputs, i), _named(inputs, i)
        for t in range(num_rows):
            expected = _phase_slots(phase) if t in context else named.get(t, set())
            assert moved[n * num_rows + t] == expected, (phase, t, moved[n * num_rows + t], expected)
            named_somewhere += bool(named.get(t)) and t not in context
    assert named_somewhere >= 20


def test_the_tile_and_the_hex_code_reach_the_logit(groups, model_for):
    """A legal row's tile code, rotation, and hex code each move its own logit
    only. Code 0 adds nothing: the tile table's and the hex embedding's row 0
    are zero."""
    inputs = groups['capacity 5']
    model = model_for(5)
    table = get_tile_table()
    # A tile whose block turns with its rotation.
    tile = next(t for t in range(1, NUM_TILE_CODES) if (table[t, 0] != table[t, 1]).any())
    hex_column, tile_column, rotation_column = int(C.CHOICE3_HEX), int(C.CHOICE3_TILE), int(C.CHOICE3_ROTATION)
    for phase in (K.DPHASE_TRACK, K.DPHASE_TILE, K.DPHASE_STOCK):
        i = inputs.at(phase)[0]
        x = inputs.x[i:i + 1].clone()
        k = inputs.samples[i].legal[-1]
        inputs.part('CHOICES', x)[0, k, [tile_column, rotation_column]] = torch.tensor([tile, 0.0])
        tokens = _encode(model, x)
        base = _head(model, tokens, x)
        for column, value in ((hex_column, 0), (hex_column, 1 + inputs.part('CHOICES', x)[0, k, 0] % D.NUM_HEXES),
                              (tile_column, 0), (tile_column, 1 + tile % (NUM_TILE_CODES - 1)), (rotation_column, 1)):
            edited = x.clone()
            inputs.part('CHOICES', edited)[0, k, column] = value
            assert _moved_slots(_head(model, tokens, edited), base)[0] == {_slot(phase, k)}, (phase, column, value)
    assert not model.hex_embed.weight[0].any() and model.hex_embed.padding_idx == 0
    with torch.inference_mode():
        zeros = torch.zeros(1, 3)
        assert not model._tiles(zeros, zeros + 4).any()


def test_auctions_buy_follows_the_topmost_selector(groups, model_for):
    """AUCTION's buy names the PRIVATE row PRIVATE 2 selects: moving the
    selector moves the buy's logit and no other, and the buy's logit with
    private q topmost reads q's final row."""
    inputs = groups['capacity 5']
    model = model_for(5)
    i = inputs.at(K.DPHASE_AUCTION)[0]
    x = inputs.x[i:i + 1].clone()
    privates = inputs.part('PRIVATE', x)
    assert privates[0, :, 1].sum() == 0  # no subject at AUCTION
    tokens = _encode(model, x)
    buy = _slot(K.DPHASE_AUCTION, K.AUCTION_BUY)
    logits = {}
    for q in range(D.NUM_PRIVATES):
        privates[0, :, 2] = 0
        privates[0, q, 2] = 1
        selector = model._named_selector(model._parts(x), inputs.part('GLOBAL', x)[:, 0, PHASES])
        assert selector[0, K.AUCTION_BUY].nonzero().flatten().tolist() == [PRIVATE + q]
        logits[q] = _head(model, tokens, x)
        edited = tokens.clone()
        edited[0, PRIVATE + q] += 0.5
        assert buy in _moved_slots(_head(model, edited, x), logits[q])[0]
    for q in range(1, D.NUM_PRIVATES):
        assert _moved_slots(logits[q], logits[0])[0] == {buy}
    privates[0, :, 2] = 0
    selector = model._named_selector(model._parts(x), inputs.part('GLOBAL', x)[:, 0, PHASES])
    assert not selector[0, K.AUCTION_BUY].any()


def test_every_slot_is_one_choice_row_and_names_the_engines_row(model_for):
    """The head's tables: unified slot u is row k of phase p for u = offsets[p]
    + k, its slot embedding is row u, and its named row is the engine's."""
    model = model_for(5)
    named_rows = get_named_rows()
    for u in range(UNIFIED):
        phase, k = int(model._slot_phase[u]), int(model._slot_row[u])
        assert OFFSETS[phase] + k == u and model._choice_slot_index[phase, k] == u
        assert model._choice_slot_valid[phase, k] == 1
        expected = [int(named_rows[u])] if named_rows[u] >= 0 else []
        assert model._named_static[phase, k].nonzero().flatten().tolist() == expected
        assert float(model._named_topmost[phase, k]) == float(named_rows[u] == -2)
    assert int(model._choice_slot_valid.sum()) == UNIFIED and int(model._named_topmost.sum()) == 1


# =============================================================================
# CONSTRUCTION AND THE CONTRACT
# =============================================================================

def _config(capacity: int = 5, model_path: str = MODEL_PATH, **fields) -> TrainingConfig:
    sizes = {'d_model': D_MODEL, 'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS}
    return TrainingConfig(game='1889', model_path=model_path, num_players=capacity, **{**sizes, **fields})


@pytest.mark.parametrize('capacity', (3, 4, 5))
def test_the_module_contract(capacity):
    config = _config(capacity, zero_sum_values=True)
    model: Any = create_model(config)
    assert isinstance(model, MODEL.G1889TransformerNet)
    spec = get_model_input_spec(config)
    assert MODEL.INPUT_LAYOUT_VERSION == spec.layout_version == model.cfg.layout_version == 3
    size = get_input_size(capacity, layout_version=LAYOUT)
    assert (spec.input_size, spec.num_tokens, spec.token_dim) == (size, 0, 0) == (4833 + 55 * capacity, 0, 0)
    assert (spec.policy_dim, spec.value_dim, spec.num_players, spec.game) == (UNIFIED, capacity, capacity, '1889')
    assert (model.cfg.num_players, model.cfg.input_size, model.cfg.num_trunk_rows) == (capacity, size, 40 + capacity)
    assert (model.cfg.d_model, model.cfg.num_heads, model.cfg.num_layers) == (D_MODEL, NUM_HEADS, NUM_LAYERS)
    assert model.cfg.zero_sum_values is True
    assert model.relation_bias_mult.shape == (NUM_LAYERS, NUM_HEADS, 33) and not model.relation_bias_mult.any()
    # Buffers are built at construction, not checkpointed.
    assert all(key.endswith(('weight', 'bias')) or key in ('relation_bias_mult', 'choice_phase_weight')
               for key in model.state_dict())
    # A token layout's spec is as it was.
    v1_spec = get_model_input_spec(_config(capacity, model_path=str(REPO / 'nn' / 'transformer-1889-v1.py')))
    assert (v1_spec.input_size, v1_spec.num_tokens, v1_spec.token_dim) == (0, 90 + capacity, 109)


@pytest.mark.parametrize('field, value', [
    ('game', 'rss'), ('layout_version', 2), ('input_size', 5107), ('num_tokens', 95), ('token_dim', 109),
    ('policy_dim', 227), ('value_dim', 4),
])
def test_build_model_asserts_the_spec(field, value):
    config = _config()
    spec = get_model_input_spec(config)
    MODEL.build_model(config, spec)
    with pytest.raises(AssertionError):
        MODEL.build_model(config, replace(spec, **{field: value}))


def test_the_forward_checks_its_inputs(groups):
    inputs = groups['capacity 5']
    model = _new_model(5)
    with pytest.raises(AssertionError, match=r'inputs shape must be \(batch, 5108\)'):
        _run(model, inputs.x[:, :-1], inputs.mask)
    with pytest.raises(AssertionError, match='legal_mask must be bool'):
        _run(model, inputs.x, inputs.mask.to(torch.uint8))
    with pytest.raises(AssertionError, match='inputs shape must be'):
        _run(_new_model(4), inputs.x, inputs.mask)


def test_construction_checks_the_parts(monkeypatch):
    MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))
    real = MODEL.get_input_parts

    def drifted(max_players, *, layout_version):
        parts = list(real(max_players, layout_version=layout_version))
        corp = next(i for i, p in enumerate(parts) if p.name == 'CORP')
        parts[corp] = parts[corp]._replace(shape=(7, 77))
        return tuple(parts)
    monkeypatch.setattr(MODEL, 'get_input_parts', drifted)
    with pytest.raises(AssertionError, match='part PRIVATE at 667, expected 674'):
        MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))


@pytest.mark.parametrize('phase', [K.DPHASE_BID, K.DPHASE_TRACK, K.DPHASE_PORT])
def test_construction_checks_the_policy_blocks(phase, monkeypatch):
    sizes = list(POLICY_LAYOUT.phase_sizes)
    sizes[phase] += 1
    monkeypatch.setattr(MODEL, 'POLICY_LAYOUT', PolicyLayout(
        tuple(sizes), POLICY_LAYOUT.phase_names, POLICY_LAYOUT.phases_with_pass_slot,
    ))
    with pytest.raises(AssertionError, match='choice'):
        MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))


def test_the_parameters_against_v1s():
    """At 256 x 15, whatever the capacity: v1's trunk, value head, and tile
    projection; the row projections of layout 3's features; the bias
    multiplier; and one choice head where v1 had sixteen."""
    sizes = {'d_model': 256, 'num_heads': 4, 'num_layers': 15, 'ff_mult': 3.0}
    v2: Any = create_model(_config(5, **sizes))
    v1: Any = create_model(_config(5, model_path=str(REPO / 'nn' / 'transformer-1889-v1.py'), **sizes))
    total = sum(p.numel() for p in v2.parameters())
    assert total == sum(p.numel() for p in create_model(_config(3, **sizes)).parameters())
    d = 256

    def count(*modules):
        return sum(p.numel() for m in modules for p in (m.parameters() if isinstance(m, nn.Module) else [m]))

    trunk = 15 * (4 * d * d + 3 * d * 768 + 2 * d) + d
    assert count(v2.blocks, v2.final_norm) == count(v1.blocks, v1.final_norm) == trunk
    assert count(v2.value_head) == count(v1.value_head) and count(v2.tile_proj) == count(v1.tile_proj) == 76 * d
    widths = {'GLOBAL': 134, 'CORP': 197, 'PRIVATE': 31, 'CENTER': 71, 'PLAYER': 54}
    assert {name: proj.in_features for name, proj in v2.row_proj.items()} == widths
    assert count(v2.row_proj) == sum((w + 1) * d for w in widths.values())
    assert count(v2.row_id_embed, v2.type_embeds) == (7 + 7 + 25 + 5) * d
    assert count(v2.relation_bias_mult) == 15 * 4 * 33
    head = count(v2.slot_embed, v2.choice_phase_weight, v2.named_proj, v2.hex_embed, v2.choice_hidden, v2.choice_out)
    assert head == 228 * d + 19 * 44 * d + (d + 1) * d + 53 * d + (5 * d + 1) * d + d + 1
    assert total == count(v2.blocks, v2.final_norm, v2.value_head, v2.tile_proj, v2.row_proj, v2.row_id_embed,
                          v2.type_embeds, v2.relation_bias_mult) + head
    assert total == 13_692_094 and sum(p.numel() for p in v1.parameters()) == V1_PARAMETERS


def test_checkpoints_rebuild_through_build_model(tmp_path, weights, groups):
    inputs = groups['capacity 5']
    config = _config(zero_sum_values=True)
    model = create_model(config)
    model.load_state_dict(weights)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path=path, epoch=0, model=model, trainer_state={'global_step': 0}, config=config,
                    metrics={}, buffer_stats={'size': 0, 'capacity': 0})
    loaded, loaded_config, _ = load_model_from_checkpoint(path, torch.device('cpu'))
    assert loaded_config.game == '1889' and loaded_config.model_path == MODEL_PATH
    assert isinstance(loaded, MODEL.G1889TransformerNet)
    outputs = _run(model.eval(), inputs.x, inputs.mask)
    assert all(torch.equal(a, b) for a, b in zip(outputs, _run(loaded.eval(), inputs.x, inputs.mask), strict=True))
    fresh = _new_model(5, zero_sum=True)
    fresh.load_state_dict(model.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(outputs, _run(fresh, inputs.x, inputs.mask), strict=True))


@pytest.mark.parametrize('optimizer', ('muon', 'adamw'))
def test_the_optimizer_groups_hold_every_parameter_once(optimizer):
    """The trainer's own grouping: Muon takes the trunk's matrices and the
    hidden layers (named_proj and the heads' first layers); AdamW with decay
    the projections of raw inputs (W among them, being 3-D) and the heads'
    last layers; AdamW without decay the biases, norms, embeddings, and the
    bias multiplier."""
    config = _config(optimizer=optimizer, learning_rate=3e-4, weight_decay=0.01, batch_size=8, num_epochs=1,
                     training_steps_per_epoch=10, warmup_epochs=0)
    model: Any = create_model(config)
    trainer = Trainer(model, config, torch.device('cpu'))
    groups = [(type(trainer.optimizer).__name__, group) for group in trainer.optimizer.param_groups]
    if trainer._aux_optimizer is not None:
        groups += [('AdamW', group) for group in trainer._aux_optimizer.param_groups]
    names = {id(p): name for name, p in model.named_parameters()}
    placed: dict[str, tuple[str, float]] = {}
    for kind, group in groups:
        for p in group['params']:
            assert names[id(p)] not in placed, names[id(p)]
            placed[names[id(p)]] = (kind, group['weight_decay'])
    assert set(placed) == set(names.values())

    def kind_of(name: str) -> tuple[str, float]:
        if name.endswith('bias') or 'norm' in name or 'embed' in name or name == 'relation_bias_mult':
            return 'AdamW', 0.0
        hidden = name.startswith('blocks.') or name in (
            'named_proj.weight', 'choice_hidden.weight', 'value_head.0.weight')
        if hidden and optimizer == 'muon':
            return 'Muon', 0.01
        return 'AdamW', 0.01
    assert placed == {name: kind_of(name) for name in names.values()}
    assert {name for name, (kind, decay) in placed.items() if kind == 'AdamW' and decay > 0} >= {
        'choice_phase_weight', 'tile_proj.weight', 'choice_out.weight', 'value_head.2.weight',
        *(f'row_proj.{name}.weight' for name in FIRST_ROW)}


def test_every_parameter_learns(groups, model_for):
    """A training step's graph reaches every parameter: every weight but
    build distance's columns and the hex embedding's row 0 gets gradient."""
    inputs = groups['capacity 5']
    model = model_for(5, zero_sum=True).train()
    logits, values = model(inputs.x, torch.ones_like(inputs.mask))
    (logits.sum() + values.square().sum()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    columns = model.relation_bias_mult.grad.abs().sum((0, 1))
    unfilled = set(model._build_distance_bias)
    assert all((columns[c] > 0) != (c in unfilled) for c in range(len(columns)))
    assert not model.hex_embed.weight.grad[0].any() and model.hex_embed.weight.grad[1:].abs().sum(1).gt(0).sum() > 40
    assert (model.slot_embed.weight.grad.abs().sum(1) > 0).all()


def test_the_forward_traces_without_graph_breaks(groups, model_for):
    """The batch stays a dynamic dimension: no data-dependent shape or
    control flow inside the forward."""
    inputs = groups['capacity 5']
    model = model_for(5)
    x, mask = inputs.x[:4].clone(), inputs.mask[:4]
    mark_unbacked(x, 0)
    with torch.inference_mode():
        explanation = explain(model)(x, mask)
    assert explanation.graph_break_count == 0, explanation.break_reasons


def test_the_logit_diagnostics(groups, model_for):
    inputs = groups['capacity 5']
    model = model_for(5)
    logits, _ = _run(model, inputs.x, inputs.mask)
    stats = model.pass_action_logit_abs(logits, inputs.mask, inputs.phases)
    assert stats.shape == (2 * len(K.PHASES_WITH_PASS_SLOT),)
    for index, phase in enumerate(K.PHASES_WITH_PASS_SLOT):
        rows = inputs.phases == phase
        legal = inputs.mask[rows, OFFSETS[phase]]
        expected = logits[rows, OFFSETS[phase]][legal].abs().mean() if legal.any() else torch.tensor(0.0)
        assert torch.isclose(stats[2 * index], expected, atol=1e-5)
