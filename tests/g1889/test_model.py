"""nn/transformer-1889.py, the 1889 v0 model (1889/model-design.md §3-4, §8 row 8.1).

On random-playout states at every decision phase, 3-5 players, at capacities 3
and 5 and mixed, with small models whose weights and relation multipliers are
scrambled, so that relations matter and logits have scale:

- legal logits are finite and the rest -1e9, and values are in range and
  zero-sum where asked, with padded slots zero;
- flat records and dense planes give the same outputs, and neither records
  past id 24 nor value-0 records change them;
- a batch equals its states run alone, and capacity changes nothing for the
  same weights;
- permuting the visible candidates, with their records, permutes the logits;
- with the trunk's output held fixed, each logit moves with its own row's raw
  slots, exactly the ones §4.2 lists, and no other row's;
- hidden rows reach no legal slot;
- every head's outputs land on the slots §4 gives them, and every slot has one.

Construction, the loader's contract, the parameter count, and
``flatten_relation_records`` are checked directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._dynamo import explain
from torch._dynamo.decorators import mark_unbacked

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.relations import (
    RelationId as REL,
    StaticRelationId as SREL,
    get_relation_coord_data_batch,
    get_relation_data_batch,
    static_relation_planes,
)
from g1889.token_data import (
    TokenRow as R,
    TokenWidth as W,
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
)
from games.g1889 import POLICY_LAYOUT
from mcts.evaluator import _infer_model_input_spec
from games import get_game
from nn import _load_model_module, create_model, get_model_input_spec
from nn.model_contract import flatten_relation_records
from nn.policy_layout import PolicyLayout
from train.checkpoint import load_model_from_checkpoint, save_checkpoint
from train.config import TrainingConfig

from tests.g1889.phases.conftest import new_game, policy_ids
from tests.g1889.test_extraction import _choose, _state

MODEL_PATH = str(Path(__file__).resolve().parents[2] / 'nn' / 'transformer-1889.py')
MODEL = _load_model_module(MODEL_PATH)

TOKEN_DIM = 108
NUM_PHASES = POLICY_LAYOUT.num_phases
OFFSETS = POLICY_LAYOUT.offsets
TILE_BLOCK = range(2, 78)

# Decisions kept per phase and setup, from games that reach all 19 phases.
SETUPS = ((3, 3), (3, 5), (4, 5), (5, 5))
SEEDS = range(60)
MAX_DECISIONS = 3000
PER_PHASE = 3
GROUPS = {'capacity 3': ((3, 3),), 'capacity 5': ((5, 5),), 'mixed': ((3, 5), (4, 5), (5, 5))}

# Small enough to run often; the design's size is checked by its parameters.
D_MODEL, NUM_HEADS, NUM_LAYERS = 64, 4, 3

# Row types: the first row, the row count (None: one per player), the width,
# and the is_selected slots.
ROW_TYPES = {
    'GLOBAL': (R.ROW_GLOBAL, 1, W.TW_GLOBAL, ()),
    'PRICE': (R.ROW_PRICE, 1, W.TW_PRICE, ()),
    'DIVIDEND': (R.ROW_DIVIDEND, 1, W.TW_DIVIDEND, ()),
    'CORP': (R.ROW_CORP, D.NUM_CORPS, W.TW_CORP, (1, 2, 3)),
    'PRIVATE': (R.ROW_PRIVATE, D.NUM_PRIVATES, W.TW_PRIVATE, (1, 2)),
    'TRAIN': (R.ROW_TRAIN, D.NUM_TRAIN_TYPES, W.TW_TRAIN, (1,)),
    'HEX': (R.ROW_HEX, D.NUM_HEXES, W.TW_HEX, (1,)),
    'CANDIDATE': (R.ROW_CANDIDATE, D.MAX_LAY_CANDIDATES, W.TW_CANDIDATE, ()),
    'PLAYER': (R.ROW_PLAYER, None, W.TW_PLAYER, (1,)),
}


def _slot(phase, action):
    return OFFSETS[phase] + action


# =============================================================================
# STATES AND INPUTS
# =============================================================================

@dataclass
class Sample:
    row: np.ndarray
    phase: int
    legal: list[int]
    num_players: int


@dataclass
class Inputs:
    x: torch.Tensor
    mask: torch.Tensor
    records: torch.Tensor
    flat: torch.Tensor
    phases: torch.Tensor
    samples: list[Sample]
    max_players: int

    def dense(self) -> torch.Tensor:
        num_tokens = get_num_tokens(self.max_players)
        planes = np.zeros((len(self.samples), int(REL.NUM_RELATIONS), num_tokens, num_tokens), np.uint8)
        get_relation_data_batch([s.row for s in self.samples], planes, self.max_players)
        return torch.from_numpy(planes)

    def at(self, phase: int) -> list[int]:
        return (self.phases == phase).nonzero().flatten().tolist()

    def alone(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[i:i + 1], self.mask[i:i + 1], flatten_relation_records(self.records[i:i + 1])


def _sample_setup(num_players: int, max_players: int) -> list[Sample]:
    """Up to PER_PHASE decisions of each phase, uniformly over the games'."""
    pick = np.random.default_rng(num_players)
    kept: list[list[Sample]] = [[] for _ in range(NUM_PHASES)]
    seen = [0] * NUM_PHASES
    for seed in SEEDS:
        rng = np.random.default_rng(100 * num_players + seed)
        state = new_game(num_players, max_players=max_players, seed=seed, priority_player=-1,
                         auction_pass_payout_cap=1, same_president=True)
        for _ in range(MAX_DECISIONS):
            if A.is_terminal_py(state):
                break
            phase = A.get_decision_phase_py(state)
            seen[phase] += 1
            at = len(kept[phase]) if len(kept[phase]) < PER_PHASE else int(pick.integers(seen[phase]))
            if at < PER_PHASE:
                sample = Sample(state._array.copy(), phase, policy_ids(state), num_players)
                kept[phase][at:at + 1] = [sample]
            assert DRIVER.apply_action(state, _choose(state, rng, True)) != STATUS_INVALID
    return [sample for samples in kept for sample in samples]


def _inputs(samples: list[Sample], max_players: int) -> Inputs:
    rows = [s.row for s in samples]
    x = np.zeros((len(rows), get_num_tokens(max_players), TOKEN_DIM), np.float32)
    get_token_data_batch(rows, x, max_players, layout_version=1)
    records = np.zeros((len(rows), 2048, 4), np.uint8)
    get_relation_coord_data_batch(rows, records, max_players)
    mask = torch.zeros(len(rows), POLICY_LAYOUT.unified_dim, dtype=torch.bool)
    for i, sample in enumerate(samples):
        mask[i, [_slot(sample.phase, action) for action in sample.legal]] = True
    records_t = torch.from_numpy(records)
    return Inputs(
        torch.from_numpy(x), mask, records_t, flatten_relation_records(records_t),
        torch.tensor([s.phase for s in samples]), samples, max_players,
    )


@pytest.fixture(scope='module')
def setups() -> dict[tuple[int, int], list[Sample]]:
    return {setup: _sample_setup(*setup) for setup in SETUPS}


@pytest.fixture(scope='module')
def groups(setups) -> dict[str, Inputs]:
    return {name: _inputs([s for setup in members for s in setups[setup]], members[0][1])
            for name, members in GROUPS.items()}


@pytest.fixture(params=GROUPS)
def inputs(request, groups) -> Inputs:
    return groups[request.param]


@pytest.fixture(scope='module')
def weights() -> dict[str, torch.Tensor]:
    """Scrambled weights, which fit a model at any capacity: at initialization
    the relation bias is zero and the logits are tiny, which would hide most
    of what these tests look for."""
    torch.manual_seed(0)
    model = _new_model(5)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.15)
    nn.init.normal_(model.relation_bias_mult, std=0.5)
    return model.state_dict()


def _new_model(capacity: int, zero_sum: bool = False):
    return MODEL.G1889TransformerNet(MODEL.TransformerConfig(
        num_players=capacity, d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        zero_sum_values=zero_sum,
    )).eval()


@pytest.fixture
def model_for(weights):
    def build(capacity: int, zero_sum: bool = False):
        model = _new_model(capacity, zero_sum)
        model.load_state_dict(weights)
        return model
    return build


@torch.inference_mode()
def _run(model, x, mask, relations) -> tuple[torch.Tensor, torch.Tensor]:
    return model(x, mask, relations)


def _assert_same(a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...], tol: float = 2e-5) -> None:
    for left, right in zip(a, b, strict=True):
        assert torch.allclose(left, right, atol=tol, rtol=tol), (left - right).abs().max()


def _real_values(values: torch.Tensor, inputs: Inputs) -> torch.Tensor:
    return values * (inputs.x[:, R.ROW_PLAYER:, 0] > 0.5)


# =============================================================================
# OUTPUTS
# =============================================================================

def test_the_states_reach_every_phase(groups):
    for inputs in groups.values():
        assert set(inputs.phases.tolist()) == set(range(NUM_PHASES))
    assert set(s.num_players for s in groups['mixed'].samples) == {3, 4, 5}


def test_legal_logits_are_finite_and_the_rest_masked(inputs, model_for):
    logits, values = _run(model_for(inputs.max_players), inputs.x, inputs.mask, inputs.flat)
    assert logits.shape == inputs.mask.shape and logits.dtype == torch.float32
    assert values.shape == (len(inputs.samples), inputs.max_players)
    assert torch.isfinite(logits[inputs.mask]).all()
    assert (logits[~inputs.mask] == -1e9).all()
    assert (values.abs() <= 1).all()
    # Scrambled weights give logits of a useful scale.
    assert logits[inputs.mask].std() > 0.1


def test_values_are_zero_sum_over_real_players(inputs, model_for):
    _, raw = _run(model_for(inputs.max_players), inputs.x, inputs.mask, inputs.flat)
    _, values = _run(model_for(inputs.max_players, zero_sum=True), inputs.x, inputs.mask, inputs.flat)
    real = inputs.x[:, R.ROW_PLAYER:, 0] > 0.5
    assert (real.sum(1) == torch.tensor([s.num_players for s in inputs.samples])).all()
    assert (values[~real] == 0).all()
    assert values.sum(1).abs().max() < 1e-5
    # Centering moves each state's values together, keeping their differences.
    for i in range(len(inputs.samples)):
        _assert_same((values[i][real[i]] - values[i][real[i]][0],), (raw[i][real[i]] - raw[i][real[i]][0],))


# =============================================================================
# RELATIONS
# =============================================================================

def test_dense_and_flat_inputs_agree(inputs, model_for):
    model = model_for(inputs.max_players)
    flat = _run(model, inputs.x, inputs.mask, inputs.flat)
    _assert_same(_run(model, inputs.x, inputs.mask, inputs.dense()), flat)
    # The records matter: without them the outputs are far away.
    logits, _ = _run(model, inputs.x, inputs.mask, inputs.flat[:0])
    assert (logits - flat[0])[inputs.mask].abs().max() > 0.1


def test_records_past_the_relations_and_value_0_records_carry_no_weight(inputs, model_for):
    model = model_for(inputs.max_players)
    base = _run(model, inputs.x, inputs.mask, inputs.flat)
    rng = torch.Generator().manual_seed(1)
    batch, num_tokens, n = len(inputs.samples), inputs.x.shape[1], 4096
    cells = torch.stack([
        torch.randint(0, batch, (n,), generator=rng),
        torch.randint(0, 25, (n,), generator=rng),
        torch.randint(0, num_tokens, (n,), generator=rng),
        torch.randint(0, num_tokens, (n,), generator=rng),
        torch.randint(1, 11, (n,), generator=rng),
    ], dim=1).to(torch.int16)

    later = cells.clone()
    later[:, 1] = torch.randint(25, 256, (n,), generator=rng)
    zero = cells.clone()
    zero[:, 4] = 0
    for extra in (later, zero):
        _assert_same(_run(model, inputs.x, inputs.mask, torch.cat([inputs.flat, extra])), base)
    mixed = torch.cat([later, inputs.flat, zero])
    _assert_same(_run(model, inputs.x, inputs.mask, mixed[torch.randperm(len(mixed), generator=rng)]), base)
    # The same records with real ids and values would have moved the outputs.
    logits, _ = _run(model, inputs.x, inputs.mask, torch.cat([inputs.flat, cells]))
    assert (logits - base[0])[inputs.mask].abs().max() > 0.1

    dense = inputs.dense()
    junk = torch.randint(0, 11, (batch, 3, num_tokens, num_tokens), generator=rng, dtype=torch.uint8)
    _assert_same(_run(model, inputs.x, inputs.mask, torch.cat([dense, junk], dim=1)), base)


def test_the_finished_mask_scales_records_adds_static_planes_and_hides_keys(groups, model_for):
    inputs = groups['mixed']
    model = model_for(inputs.max_players)
    tokens = model._project_tokens(inputs.x)
    hidden = (inputs.x[:, :, 0] <= 0.5)[:, None, None, :]
    layer = 1
    with torch.inference_mode():
        bias = model._attention_bias(model._prepare_relations(inputs.flat), hidden, layer, tokens)

    scales = np.ones(25)
    scales[[REL.REL_PLAYER_SHARES, REL.REL_CORP_SHARES]] = 0.1
    scales[[REL.REL_CORP_TRAINS, REL.REL_TRAIN_HELD_BY]] = 0.25
    mult = model.relation_bias_mult[layer].detach().numpy()
    static = static_relation_planes(inputs.max_players).astype(np.float64)
    expected = np.zeros((len(inputs.samples), NUM_HEADS, *static.shape[1:]))
    expected += np.einsum('rij,hr->hij', static, mult[:, 25:])
    for b, relation, query, key, value in inputs.flat.tolist():
        expected[b, :, query, key] += mult[:, relation] * value * scales[relation]
    visible = ~hidden.expand_as(bias)
    assert np.allclose(bias[visible].numpy(), expected[visible.numpy()], atol=1e-5)
    assert (bias[~visible] == torch.finfo(bias.dtype).min).all()


def test_the_batch_column_gives_each_state_its_own_records(groups, model_for):
    """Each state's records handed to its neighbor change the outputs."""
    inputs = groups['mixed']
    model = model_for(inputs.max_players)
    logits, _ = _run(model, inputs.x, inputs.mask, inputs.flat)
    swapped = inputs.flat.clone()
    swapped[:, 0] = (swapped[:, 0] + 1) % len(inputs.samples)
    moved, _ = _run(model, inputs.x, inputs.mask, swapped)
    assert (moved - logits)[inputs.mask].abs().max() > 0.1


# =============================================================================
# EQUIVARIANCES AND INDEPENDENCES
# =============================================================================

def test_a_batch_equals_its_states_alone(inputs, model_for):
    model = model_for(inputs.max_players)
    logits, values = _run(model, inputs.x, inputs.mask, inputs.flat)
    for i in range(len(inputs.samples)):
        _assert_same(_run(model, *inputs.alone(i)), (logits[i:i + 1], values[i:i + 1]))


def test_capacity_changes_nothing_for_the_same_weights(setups, model_for):
    """The same weights and 3-player states at capacity 3 and at capacity 5,
    where the two padded player rows are hidden."""
    at_3, at_5 = _inputs(setups[3, 3], 3), _inputs(setups[3, 5], 5)
    logits_3, values_3 = _run(model_for(3), at_3.x, at_3.mask, at_3.flat)
    logits_5, values_5 = _run(model_for(5), at_5.x, at_5.mask, at_5.flat)
    _assert_same((logits_3, values_3), (logits_5, values_5[:, :3]))


def _candidate_decisions(inputs: Inputs):
    """(i, first candidate id, candidates) at TILE and ER_BONUS decisions with two or more."""
    for phase, first in ((K.DPHASE_TILE, 0), (K.DPHASE_ER_BONUS, K.ER_BONUS_CANDIDATE)):
        for i in inputs.at(phase):
            count = int(inputs.x[i, R.ROW_CANDIDATE:R.ROW_PLAYER, 0].sum())
            if count >= 2:
                yield i, phase, first, count


def test_permuting_the_visible_candidates_permutes_their_logits(inputs, model_for):
    model = model_for(inputs.max_players)
    rng = torch.Generator().manual_seed(2)
    spread = []
    tested = 0
    for i, phase, first, count in _candidate_decisions(inputs):
        x, mask, flat = inputs.alone(i)
        logits, values = _run(model, x, mask, flat)
        perm = torch.randperm(count, generator=rng)  # candidate k moves to row perm[k]
        where = torch.arange(x.shape[1])
        where[R.ROW_CANDIDATE:R.ROW_CANDIDATE + count] = R.ROW_CANDIDATE + perm
        moved_x = x.clone()
        moved_x[0, where] = x[0]
        moved_flat = flat.clone()
        for column in (2, 3):
            moved_flat[:, column] = where[flat[:, column].long()].to(torch.int16)
        moved_logits, moved_values = _run(model, moved_x, mask, moved_flat)
        block = _slot(phase, first)
        expected = logits.clone()
        expected[0, block + perm] = logits[0, block:block + count]
        _assert_same((moved_logits, moved_values), (expected, values))
        spread.append(logits[0, block:block + count].std())
        tested += 1
    assert tested >= 1
    assert max(spread) > 1e-2


# The raw slots each pointer and joint head reads, §4.2 and §4.3:
# (name, phase, row type, slots, the phase-local ids row k of the type moves
# given the state's selections).
def _corp_stock_ids(c, x):
    return {K.STOCK_IPO + c, K.STOCK_POOL + c, K.STOCK_PAR + c, K.STOCK_SELL + c}


def _selected_ids(type_name, slot, ids):
    """All of a joint head's ``ids``, for the row ``slot`` selects."""
    first = ROW_TYPES[type_name][0]

    def own(k, x):
        return set(ids) if x[first + k, slot] > 0.5 else set()
    return own


def _table_ids(table, first=0):
    return lambda k, x: {first + i for i, v in enumerate(table) if v == k}


READOUTS = [
    ('track', K.DPHASE_TRACK, 'HEX', range(92, 103), _table_ids(D.TRACK_HEXES, K.TRACK_HEX)),
    ('token', K.DPHASE_TOKEN, 'HEX', range(103, 105), _table_ids(D.TOKEN_HEXES, K.TOKEN_HEX)),
    ('stock', K.DPHASE_STOCK, 'CORP', [*range(7, 26), *range(44, 49)], _corp_stock_ids),
    ('stock private', K.DPHASE_STOCK, 'PRIVATE', (16, 17),
     lambda k, x: {D.PRIVATE_DR: {K.STOCK_DR}, D.PRIVATE_MF: {K.STOCK_MF}}.get(k, set())),
    ('select train', K.DPHASE_SELECT_TRAIN, 'TRAIN', (10, 11),
     lambda k, x: {K.SELECT_TRAIN_TYPE + k} | _table_ids(D.TRADE_IN_TYPE, K.SELECT_TRAIN_TRADE_IN)(k, x)),
    ('auction', K.DPHASE_AUCTION, 'PRIVATE', range(7, 14),
     lambda k, x: ({K.AUCTION_BID + k - 1} if k >= 1 else set())
     | ({K.AUCTION_BUY} if x[R.ROW_PRIVATE + k, 2] > 0.5 else set())),
    ('private pre', K.DPHASE_BUY_PRIVATE_PRE, 'PRIVATE', (16, 17),
     lambda k, x: {K.BUY_PRIVATE_PRE_PRIVATE + k}
     | {D.PRIVATE_DR: {K.BUY_PRIVATE_PRE_DR}, D.PRIVATE_MF: {K.BUY_PRIVATE_PRE_MF}}.get(k, set())),
    ('private post', K.DPHASE_BUY_PRIVATE_POST, 'PRIVATE', (16, 17),
     lambda k, x: {K.BUY_PRIVATE_POST_PRIVATE + k} | ({K.BUY_PRIVATE_POST_DR} if k == D.PRIVATE_DR else set())),
    ('discard', K.DPHASE_DISCARD, 'TRAIN', (12,), _table_ids(D.DISCARD_TYPES)),
    ('tile', K.DPHASE_TILE, 'CANDIDATE', [1, *range(99, 108)], lambda k, x: {k}),
    ('er bonus', K.DPHASE_ER_BONUS, 'CANDIDATE', [1, *range(99, 108)],
     lambda k, x: {K.ER_BONUS_CANDIDATE + k} if k < D.NUM_ER_CANDIDATES else set()),
    ('buy train corp', K.DPHASE_BUY_TRAIN, 'CORP', [*range(15, 20), 46, 49],
     lambda k, x: {K.BUY_TRAIN_CORP + k, K.BUY_TRAIN_SELL + k}),
    ('buy train', K.DPHASE_BUY_TRAIN, 'TRAIN', range(2, 10),
     lambda k, x: {K.BUY_TRAIN_PASS, K.BUY_TRAIN_DEPOT, K.BUY_TRAIN_POOL, K.BUY_TRAIN_BANKRUPT}),
    ('sale', K.DPHASE_SELL_AMOUNT, 'CORP', [*range(15, 20), 48, 49], _selected_ids('CORP', 3, range(5))),
    ('emergency sale', K.DPHASE_EMERGENCY_SELL_AMOUNT, 'CORP', [*range(15, 20), 48, 49],
     _selected_ids('CORP', 3, range(5))),
    ('bid', K.DPHASE_BID, 'PRICE', range(5, 24), lambda k, x: set(range(17))),
    ('par', K.DPHASE_PAR, 'PRICE', range(5, 24), lambda k, x: set(range(6))),
    ('train price', K.DPHASE_TRAIN_PRICE, 'PRICE', range(5, 24), lambda k, x: set(range(5))),
    ('private price', K.DPHASE_PRIVATE_PRICE, 'PRICE', range(5, 24), lambda k, x: set(range(4))),
    ('dividend', K.DPHASE_DIVIDEND, 'DIVIDEND', range(1, 8), lambda k, x: set(range(2))),
]


def _moved(after: torch.Tensor, before: torch.Tensor, phase: int) -> list[set[int]]:
    """Per row of two ``_unified_logits``, the phase's ids that differ beyond rounding."""
    block = slice(OFFSETS[phase], OFFSETS[phase + 1])
    moved = (after[:, block] - before[:, block]).abs() > 1e-6 * (1 + before[:, block].abs())
    return [set(row.nonzero().flatten().tolist()) for row in moved]


def _moved_ids(model, tokens: torch.Tensor, x: torch.Tensor, phase: int, edits) -> list[set[int]]:
    """For each edit of one input row, ``(x row, slots)``, the phase's ids whose
    logits move, with the trunk's output held fixed. Each row compares with
    itself unedited: a row's place in a batch can change its rounding."""
    edited = x.repeat(len(edits), 1, 1)
    for i, (row, slots) in enumerate(edits):
        edited[i, row, slots] += 5
    fixed = tokens.repeat(len(edits), 1, 1)
    with torch.inference_mode():
        return _moved(model._unified_logits(fixed, edited), model._unified_logits(fixed, x.repeat(len(edits), 1, 1)), phase)


@pytest.mark.parametrize('readout', READOUTS, ids=[r[0] for r in READOUTS])
def test_with_the_trunk_fixed_a_logit_moves_with_its_rows_listed_slots_only(readout, inputs, model_for):
    """Each listed slot of a row moves exactly the ids that row owns, and no
    other slot of any row of the type moves anything in the block. Selections
    are context, read through the trunk's rows, and stay out of it."""
    name, phase, type_name, slots, own = readout
    model = model_for(inputs.max_players)
    first, count, width, selections = ROW_TYPES[type_name]
    others = [s for s in range(1, int(width)) if s not in slots and s not in selections]
    tested = 0
    for i in inputs.at(phase)[:2]:
        x, _, flat = inputs.alone(i)
        with torch.inference_mode():
            tokens = model._encode(x, flat)
        for k in range(int(count or inputs.max_players)):
            row = first + k
            edits = [(row, [s]) for s in slots] + [(row, others)]
            moved = _moved_ids(model, tokens, x, phase, edits)
            expected = own(k, x[0])
            # One slot's move can round away on one id, so its ids are a subset;
            # together a row's slots reach them all, and each slot moves some.
            assert all(ids <= expected and bool(ids) == bool(expected) for ids in moved[:-1]), (name, k, moved)
            assert set().union(*moved[:-1]) == expected, (name, k, moved, expected)
            assert not moved[-1], (name, k, moved[-1])
            tested += bool(expected)
    assert tested >= 1


@pytest.mark.parametrize('phase, type_name, own', [
    (K.DPHASE_TRACK, 'HEX', _table_ids(D.TRACK_HEXES, K.TRACK_HEX)),
    (K.DPHASE_TOKEN, 'HEX', _table_ids(D.TOKEN_HEXES, K.TOKEN_HEX)),
    (K.DPHASE_STOCK, 'CORP', _corp_stock_ids),
    (K.DPHASE_SELECT_TRAIN, 'TRAIN',
     lambda k, x: {K.SELECT_TRAIN_TYPE + k} | _table_ids(D.TRADE_IN_TYPE, K.SELECT_TRAIN_TRADE_IN)(k, x)),
])
def test_a_rows_final_embedding_moves_only_its_own_pointer_logits(phase, type_name, own, inputs, model_for):
    model = model_for(inputs.max_players)
    first, count, _, _ = ROW_TYPES[type_name]
    x, _, flat = inputs.alone(inputs.at(phase)[0])
    with torch.inference_mode():
        tokens = model._encode(x, flat)
        base = model._unified_logits(tokens, x)
        for k in range(int(count)):
            edited = tokens.clone()
            edited[0, first + k] += 0.5
            moved = _moved(model._unified_logits(edited, x), base, phase)[0]
            assert moved == own(k, x[0]), (k, moved)


def test_the_actor_s_holdings_of_a_corp_move_only_its_stock_logits(inputs, model_for):
    model = model_for(inputs.max_players)
    x, _, flat = inputs.alone(inputs.at(K.DPHASE_STOCK)[0])
    with torch.inference_mode():
        tokens = model._encode(x, flat)
    players = range(R.ROW_PLAYER, x.shape[1])
    actor = next(row for row in players if x[0, row, 1] > 0.5)
    for c in range(D.NUM_CORPS):
        expected = _corp_stock_ids(c, None)
        for slot in (12 + c, 19 + c):
            edits = [(row, [slot]) for row in players]
            for row, ids in zip(players, _moved_ids(model, tokens, x, K.DPHASE_STOCK, edits), strict=True):
                if row == actor:
                    assert ids and ids <= expected, (c, slot, ids)
                else:
                    assert not ids, (c, slot, row, ids)


def test_hidden_rows_reach_no_legal_slot(inputs, model_for):
    """Junk in every hidden row's slots, but for selections: they stay masked
    as keys, so no legal logit or real player's value moves."""
    model = model_for(inputs.max_players)
    base_logits, base_values = _run(model, inputs.x, inputs.mask, inputs.flat)
    hidden = inputs.x[:, :, 0] <= 0.5
    assert hidden.any()
    noise = 1 + torch.rand(inputs.x.shape, generator=torch.Generator().manual_seed(3))
    noise[:, :, 0] = 0
    for first, count, _, selections in ROW_TYPES.values():
        for slot in selections:
            noise[:, first:None if count is None else first + count, slot] = 0
    junk = torch.where(hidden[:, :, None], noise, inputs.x)
    logits, values = _run(model, junk, inputs.mask, inputs.flat)
    _assert_same((logits, _real_values(values, inputs)), (base_logits, _real_values(base_values, inputs)))
    # Junk in the same rows, made visible, would have moved the outputs.
    visible = junk.clone()
    visible[:, :, 0] = torch.where(hidden, torch.ones_like(visible[:, :, 0]), visible[:, :, 0])
    moved, _ = _run(model, visible, inputs.mask, inputs.flat)
    assert (moved - base_logits)[inputs.mask].abs().max() > 0.1


# =============================================================================
# THE PROJECTIONS AND EMBEDDINGS
# =============================================================================

@pytest.mark.parametrize('capacity', (3, 5))
def test_each_slot_projects_into_its_own_row(capacity, model_for):
    """Slot 0 and the padding past a row's width project nowhere. Any other
    slot moves its own row and no other, but HEX 1, which also gives the
    candidates their hex's identity."""
    model = model_for(capacity)
    num_tokens = get_num_tokens(capacity)
    x = torch.rand(1, num_tokens, TOKEN_DIM, generator=torch.Generator().manual_seed(4))
    with torch.inference_mode():
        base = model._project_tokens(x)
    for name, (first, count, width, _) in ROW_TYPES.items():
        first, count, width = int(first), int(count or capacity), int(width)
        edited = x.repeat(TOKEN_DIM * count, 1, 1)
        for k in range(count):
            for slot in range(TOKEN_DIM):
                edited[k * TOKEN_DIM + slot, first + k, slot] += 0.5
        with torch.inference_mode():
            moved = (model._project_tokens(edited) - base).abs().sum(-1) > 1e-5
        for k in range(count):
            for slot in range(TOKEN_DIM):
                rows = set(moved[k * TOKEN_DIM + slot].nonzero().flatten().tolist())
                if slot == 0 or slot >= width:
                    expected = set()
                elif (name, slot) == ('HEX', 1):
                    expected = {first + k, *range(int(R.ROW_CANDIDATE), int(R.ROW_PLAYER))}
                else:
                    expected = {first + k}
                assert rows == expected, (name, k, slot, rows)


def test_hexes_and_candidates_share_the_tile_projection(model_for):
    """The same tile block projects alike in a HEX row and a CANDIDATE row,
    and the rest of each row has its own projection."""
    model = model_for(5)
    x = torch.zeros(1, 95, TOKEN_DIM)
    hex_row, candidate_row = int(R.ROW_HEX) + 3, int(R.ROW_CANDIDATE) + 2
    with torch.inference_mode():
        base = model._project_tokens(x)
    x[0, [hex_row, candidate_row], TILE_BLOCK.start:TILE_BLOCK.stop] = torch.rand(len(TILE_BLOCK))
    with torch.inference_mode():
        moved = model._project_tokens(x) - base
    assert torch.allclose(moved[0, hex_row], moved[0, candidate_row], atol=1e-6)
    assert moved[0, hex_row].abs().max() > 1e-2
    assert model.tile_proj.bias is None


def test_identities_candidates_take_their_hexes(groups, model_for):
    """An entity's identity moves only its row; a hex's also moves the
    candidates laid on it, and no others."""
    inputs = groups['mixed']
    model = model_for(inputs.max_players)
    i = next(i for i, _, _, _ in _candidate_decisions(inputs))
    x = inputs.x[i:i + 1]
    the_hex = int(x[0, R.ROW_HEX:R.ROW_CANDIDATE, 1].argmax())
    assert x[0, R.ROW_HEX + the_hex, 1] == 1

    def moved_rows(embed, k):
        with torch.inference_mode():
            base = model._project_tokens(x)
            weight = embed.weight.data[k].clone()
            embed.weight.data[k] += 0.5
            moved = model._project_tokens(x)
            embed.weight.data[k] = weight
        return set(((moved - base).abs().sum(-1) > 1e-5)[0].nonzero().flatten().tolist())

    candidates = set(range(int(R.ROW_CANDIDATE), int(R.ROW_PLAYER)))
    assert moved_rows(model.hex_id_embed, the_hex) == {int(R.ROW_HEX) + the_hex} | candidates
    other = (the_hex + 1) % D.NUM_HEXES
    assert moved_rows(model.hex_id_embed, other) == {int(R.ROW_HEX) + other}
    assert moved_rows(model.corp_id_embed, 2) == {int(R.ROW_CORP) + 2}
    assert moved_rows(model.private_id_embed, 4) == {int(R.ROW_PRIVATE) + 4}
    assert moved_rows(model.train_id_embed, 5) == {int(R.ROW_TRAIN) + 5}
    assert moved_rows(model.type_embeds, 8) == set(range(int(R.ROW_PLAYER), x.shape[1]))


def test_port_masks_are_the_lays_paths(setups, model_for):
    """A port lay's raw mask is the path mask in the hex's tile block after the lay."""
    model = model_for(5)
    tested = 0
    for sample in setups[3, 5] + setups[5, 5]:
        if sample.phase != K.DPHASE_PORT:
            continue
        for action in sample.legal:
            state = _state(sample.row.copy())
            assert DRIVER.apply_action(state, action) != STATUS_INVALID
            tokens = np.zeros((get_num_tokens(5), TOKEN_DIM), np.float32)
            get_token_data(state, tokens, 5, layout_version=1)
            row = R.ROW_HEX + D.PORT_LAY_HEX[action]
            np.testing.assert_array_equal(tokens[row, 2 + 41:2 + 62], model._port_masks[action].numpy())
            tested += 1
    assert tested >= 1


# =============================================================================
# THE LAYOUT
# =============================================================================

def _ids(phase, ids):
    return {_slot(phase, action) for action in ids}


def _expected_outputs() -> dict[tuple[str, int], set[int]]:
    """Unified slots per (head, output column): §4.2's table and §4.3."""
    corps, privates = range(D.NUM_CORPS), range(D.NUM_PRIVATES)
    pre, post = K.DPHASE_BUY_PRIVATE_PRE, K.DPHASE_BUY_PRIVATE_POST
    outputs = {
        ('auction_private_head', 0): _ids(K.DPHASE_AUCTION, [K.AUCTION_BID + p - 1 for p in privates if p >= 1]),
        ('auction_private_head', 1): _ids(K.DPHASE_AUCTION, [K.AUCTION_BUY]),
        ('private_head', 0): (_ids(pre, [K.BUY_PRIVATE_PRE_PRIVATE + p for p in privates])
                              | _ids(post, [K.BUY_PRIVATE_POST_PRIVATE + p for p in privates])),
        ('private_head', 1): (_ids(K.DPHASE_STOCK, [K.STOCK_DR, K.STOCK_MF])
                              | _ids(pre, [K.BUY_PRIVATE_PRE_DR, K.BUY_PRIVATE_PRE_MF])
                              | _ids(post, [K.BUY_PRIVATE_POST_DR])),
        ('track_hex_head', 0): _ids(K.DPHASE_TRACK, [K.TRACK_HEX + i for i in range(D.NUM_TRACK_HEXES)]),
        ('token_hex_head', 0): _ids(K.DPHASE_TOKEN, [K.TOKEN_HEX + i for i in range(D.NUM_TOKEN_HEXES)]),
        ('candidate_head', 0): (_ids(K.DPHASE_TILE, range(D.MAX_LAY_CANDIDATES))
                                | _ids(K.DPHASE_ER_BONUS,
                                       [K.ER_BONUS_CANDIDATE + k for k in range(D.NUM_ER_CANDIDATES)])),
        ('discard_head', 0): _ids(K.DPHASE_DISCARD, range(D.NUM_DISCARD_TYPES)),
        ('port_head', 0): _ids(K.DPHASE_PORT, range(D.NUM_PORT_LAYS)),
        ('buy_train_head', 0): _ids(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_PASS]),
        ('buy_train_head', 1): _ids(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_DEPOT]),
        ('buy_train_head', 2): _ids(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_POOL]),
        ('buy_train_head', 3): _ids(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_BANKRUPT]),
        ('buy_train_corp_head', 0): _ids(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_CORP + c for c in corps]),
        ('buy_train_corp_head', 1): _ids(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_SELL + c for c in corps]),
        ('select_train_head', 0): _ids(K.DPHASE_SELECT_TRAIN,
                                       [K.SELECT_TRAIN_TYPE + t for t in range(D.NUM_TRAIN_TYPES)]),
        ('select_train_head', 1): _ids(K.DPHASE_SELECT_TRAIN,
                                       [K.SELECT_TRAIN_TRADE_IN + i for i in range(D.NUM_TRADE_IN_TYPES)]),
    }
    for column, base in enumerate((K.STOCK_IPO, K.STOCK_POOL, K.STOCK_PAR, K.STOCK_SELL)):
        outputs['stock_corp_head', column] = _ids(K.DPHASE_STOCK, [base + c for c in corps])
    passes = (K.DPHASE_AUCTION, K.DPHASE_STOCK, pre, K.DPHASE_TRACK, K.DPHASE_TOKEN, post, K.DPHASE_ER_BONUS)
    for column, phase in enumerate(passes):
        outputs['pass_head', column] = _ids(phase, [0])
    for head, phase, size in (
        ('bid_head', K.DPHASE_BID, 17), ('par_head', K.DPHASE_PAR, 6), ('dividend_head', K.DPHASE_DIVIDEND, 2),
        ('train_price_head', K.DPHASE_TRAIN_PRICE, 5), ('private_price_head', K.DPHASE_PRIVATE_PRICE, 4),
    ):
        for column in range(size):
            outputs[head, column] = _ids(phase, [column])
    for column in range(5):
        outputs['sale_head', column] = _ids(K.DPHASE_SELL_AMOUNT, [column]) | _ids(K.DPHASE_EMERGENCY_SELL_AMOUNT, [column])
    return outputs


def test_every_head_output_lands_on_its_slots(model_for):
    """A head's last bias, moved by one, moves its slots by one and no others;
    the selection AUCTION's buy reads is set. Together they cover every slot."""
    model = model_for(5)
    tokens = torch.zeros(1, 95, D_MODEL)
    x = torch.zeros(1, 95, TOKEN_DIM)
    x[0, R.ROW_PRIVATE + 2, 2] = 1
    expected = _expected_outputs()
    with torch.inference_mode():
        base = model._unified_logits(tokens, x)
    for (head, column), slots in expected.items():
        bias = getattr(model, head)[2].bias
        bias.data[column] += 1
        with torch.inference_mode():
            moved = model._unified_logits(tokens, x) - base
        bias.data[column] -= 1
        assert set((moved[0].abs() > 0.5).nonzero().flatten().tolist()) == slots, (head, column)
        assert torch.allclose(moved[0, sorted(slots)], torch.ones(len(slots)), atol=1e-5), (head, column)
    assert set().union(*expected.values()) == set(range(POLICY_LAYOUT.unified_dim))
    heads = {name for name, _ in model.named_children() if name.endswith('_head')}
    assert {head for head, _ in expected} == heads - {'value_head'} and len(heads) == 19


# =============================================================================
# CONSTRUCTION AND THE CONTRACT
# =============================================================================

def _config(capacity: int = 5, **fields) -> TrainingConfig:
    sizes = {'d_model': D_MODEL, 'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS}
    return TrainingConfig(game='1889', model_path=MODEL_PATH, num_players=capacity, **{**sizes, **fields})


@pytest.mark.parametrize('capacity', (3, 4, 5))
def test_the_module_contract(capacity):
    config = _config(capacity, zero_sum_values=True)
    model: Any = create_model(config)
    assert isinstance(model, MODEL.G1889TransformerNet)
    assert MODEL.INPUT_LAYOUT_VERSION == 1 and get_model_input_spec(config).layout_version == 1
    assert (model.cfg.layout_version, model.cfg.num_players) == (1, capacity)
    assert (model.cfg.d_model, model.cfg.num_heads, model.cfg.num_layers) == (D_MODEL, NUM_HEADS, NUM_LAYERS)
    assert model.cfg.zero_sum_values is True and model.cfg.num_tokens == 90 + capacity and model.cfg.token_dim == 108
    assert model.relation_bias_mult.shape == (NUM_LAYERS, NUM_HEADS, 35) and not model.relation_bias_mult.any()
    # NNEvaluator infers its spec from the model's cfg.
    inferred = _infer_model_input_spec(get_game(config), model, capacity)
    assert (inferred.num_tokens, inferred.token_dim, inferred.policy_dim, inferred.value_dim) == (
        90 + capacity, 108, 228, capacity,
    )
    # Buffers are built at construction, not checkpointed (§6.3).
    assert all(key.endswith(('weight', 'bias')) or key == 'relation_bias_mult' for key in model.state_dict())
    assert isinstance(model.value_head, nn.Module)


@pytest.mark.parametrize('field, value', [
    ('game', 'rss'), ('layout_version', 2), ('token_dim', 107), ('num_tokens', 94),
    ('policy_dim', 227), ('value_dim', 4),
])
def test_build_model_asserts_the_spec(field, value):
    config = _config()
    spec = get_model_input_spec(config)
    MODEL.build_model(config, spec)
    with pytest.raises(AssertionError):
        MODEL.build_model(config, replace(spec, **{field: value}))


def test_construction_checks_the_row_widths(monkeypatch):
    MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))
    real = MODEL.get_token_widths

    def drifted(max_players, layout_version):
        widths = real(max_players, layout_version)
        widths[R.ROW_HEX] += 1
        return widths
    monkeypatch.setattr(MODEL, 'get_token_widths', drifted)
    with pytest.raises(AssertionError, match='token layout drift'):
        MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))


@pytest.mark.parametrize('phase', [K.DPHASE_BID, K.DPHASE_TRACK, K.DPHASE_PORT])
def test_construction_checks_each_block(phase, monkeypatch):
    sizes = list(POLICY_LAYOUT.phase_sizes)
    sizes[phase] += 1
    monkeypatch.setattr(MODEL, 'POLICY_LAYOUT', PolicyLayout(
        tuple(sizes), POLICY_LAYOUT.phase_names, POLICY_LAYOUT.phases_with_pass_slot,
    ))
    with pytest.raises(AssertionError, match='policy block widths'):
        MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))


def test_parameters_match_the_design():
    """§5.2 and §5.7: 18.2M at 256 x 15, 12.8M of them in the trunk, whatever
    the capacity."""
    sizes = {'d_model': 256, 'num_heads': 4, 'num_layers': 15, 'ff_mult': 3.0}
    model: Any = create_model(_config(5, **sizes))
    total = sum(p.numel() for p in model.parameters())
    assert total == sum(p.numel() for p in create_model(_config(3, **sizes)).parameters())
    assert abs(total - 18.2e6) < 0.05e6
    assert abs(sum(p.numel() for p in model.blocks.parameters()) - 12.8e6) < 0.05e6
    heads = [module for name, module in model.named_children() if name.endswith('_head') and name != 'value_head']
    assert abs(sum(p.numel() for head in heads for p in head.parameters()) - 5.26e6) < 0.01e6


def test_checkpoints_rebuild_through_build_model(tmp_path, weights):
    config = _config(zero_sum_values=True)
    model = create_model(config)
    model.load_state_dict(weights)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path=path, epoch=0, model=model, trainer_state={'global_step': 0}, config=config,
                    metrics={}, buffer_stats={'size': 0, 'capacity': 0})
    loaded, loaded_config, _ = load_model_from_checkpoint(path, torch.device('cpu'))
    assert loaded_config.game == '1889' and isinstance(loaded, MODEL.G1889TransformerNet)
    x = torch.rand(2, 95, TOKEN_DIM, generator=torch.Generator().manual_seed(5))
    x[:, :, 0] = 1
    mask = torch.ones(2, POLICY_LAYOUT.unified_dim, dtype=torch.bool)
    flat = torch.tensor([[0, 14, 3, 23, 1], [1, 2, 90, 3, 4]], dtype=torch.int16)
    assert all(torch.equal(a, b) for a, b in zip(_run(model.eval(), x, mask, flat), _run(loaded.eval(), x, mask, flat)))


def test_diagnostics_and_optimizer_routing(groups, model_for):
    inputs = groups['mixed']
    model = model_for(inputs.max_players)
    logits, _ = _run(model, inputs.x, inputs.mask, inputs.flat)
    stats = model.pass_action_logit_abs(logits, inputs.mask, inputs.phases)
    assert stats.shape == (2 * len(K.PHASES_WITH_PASS_SLOT),) and len(K.PHASES_WITH_PASS_SLOT) == 9
    for index, phase in enumerate(K.PHASES_WITH_PASS_SLOT):
        rows = inputs.phases == phase
        pass_slot = OFFSETS[phase]
        legal = inputs.mask[rows, pass_slot]
        expected = logits[rows, pass_slot][legal].abs().mean() if legal.any() else torch.tensor(0.0)
        assert torch.isclose(stats[2 * index], expected, atol=1e-5)
        actions = inputs.mask[rows, pass_slot + 1:OFFSETS[phase + 1]]
        assert torch.isclose(stats[2 * index + 1], logits[rows, pass_slot + 1:OFFSETS[phase + 1]][actions].abs().mean())

    layers = model.input_output_layers()
    assert len({id(layer) for layer in layers}) == len(layers) == 10 + 19
    assert model.tile_proj in layers and all(isinstance(layer, nn.Linear) for layer in layers)
    for name, module in model.named_children():
        if name.endswith('_proj'):
            assert module in layers
        elif name.endswith('_head'):
            assert module[2] in layers and module[0] not in layers


def test_every_parameter_learns_through_flat_records(groups, model_for):
    """A training step's graph: the in-place bias path backpropagates into every
    weight, and into every relation column the batch's records hold."""
    inputs = groups['mixed']
    model = model_for(inputs.max_players, zero_sum=True).train()
    logits, values = model(inputs.x, torch.ones_like(inputs.mask), inputs.flat)
    # Squared: zero-sum values have no gradient on their plain sum.
    (logits.sum() + values.square().sum()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    columns = model.relation_bias_mult.grad.abs().sum((0, 1))
    assert (columns[inputs.flat[:, 1].unique().long()] > 0).all() and (columns[25:] > 0).all()
    assert len(inputs.flat[:, 1].unique()) == 25


def test_autocast_with_fp16_tokens_runs_the_eval_server_s_dtypes(inputs, model_for):
    """bf16 autocast on fp16 tokens, as the eval server runs: fp32 logits close
    to fp32's, under bf16's and fp16's resolution."""
    model = model_for(inputs.max_players)
    base_logits, base_values = _run(model, inputs.x, inputs.mask, inputs.flat)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16):
        logits, values = model(inputs.x.to(torch.float16), inputs.mask, inputs.flat)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits[inputs.mask]).all() and (logits[~inputs.mask] == -1e9).all()
    assert (logits - base_logits)[inputs.mask].abs().max() < 0.5
    assert (values.float() - base_values).abs().max() < 0.1


def test_the_forward_traces_without_graph_breaks(groups, model_for):
    """N stays a dynamic dimension: no data-dependent shape inside the forward."""
    inputs = groups['capacity 5']
    model = model_for(inputs.max_players)
    x, mask, flat = inputs.x[:4], inputs.mask[:4], inputs.flat[inputs.flat[:, 0] < 4].clone()
    mark_unbacked(flat, 0)
    with torch.inference_mode():
        explanation = explain(model)(x, mask, flat)
    assert explanation.graph_break_count == 0, explanation.break_reasons


# =============================================================================
# FLATTENING
# =============================================================================

def test_flatten_keeps_each_batch_s_real_records_in_order():
    records = np.zeros((3, 8, 4), np.uint8)
    records[0, :2] = [[1, 2, 3, 4], [25, 200, 90, 1]]
    # The middle state has no records; a value-0 record inside a state is dropped.
    records[2, :3] = [[0, 90, 3, 10], [9, 9, 9, 0], [24, 75, 74, 255]]
    expected = [[0, 1, 2, 3, 4], [0, 25, 200, 90, 1], [2, 0, 90, 3, 10], [2, 24, 75, 74, 255]]

    flat = flatten_relation_records(records)
    assert isinstance(flat, np.ndarray) and flat.dtype == np.int16 and flat.tolist() == expected

    from_tensor = flatten_relation_records(torch.from_numpy(records))
    assert isinstance(from_tensor, torch.Tensor) and from_tensor.dtype == torch.int16
    assert from_tensor.tolist() == expected

    assert flatten_relation_records(np.zeros((2, 8, 4), np.uint8)).shape == (0, 5)
    assert flatten_relation_records(np.zeros((0, 8, 4), np.uint8)).shape == (0, 5)


def test_flatten_holds_exactly_the_dense_planes(inputs):
    """The records' cells are the dense planes' nonzero cells, with their values."""
    dense = inputs.dense().numpy()
    flat = inputs.flat.numpy()
    assert len(flat) == np.count_nonzero(dense)
    batch, relation, query, key, value = flat.T
    np.testing.assert_array_equal(dense[batch, relation, query, key], value)
    assert (flat >= 0).all() and flat[:, 0].max() == len(inputs.samples) - 1 and (np.diff(batch) >= 0).all()


def test_static_planes_are_the_layout_s():
    for capacity in (3, 4, 5):
        planes = static_relation_planes(capacity)
        num_tokens = get_num_tokens(capacity)
        assert planes.shape == (SREL.NUM_STATIC_RELATIONS, num_tokens, num_tokens)
        model = MODEL.G1889TransformerNet(MODEL.TransformerConfig(num_players=capacity, d_model=D_MODEL, num_layers=1))
        np.testing.assert_array_equal(model._static_relations.numpy(), planes)
