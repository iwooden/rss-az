"""nn/transformer-1889-v1.py, the 1889 v1 model (1889/v1-design.md §7,
1889/model-design.md §12): v0 on token layout 2, with the candidates out of
the trunk and scored by a head of their own.

On layout-2 states at every decision phase, 3-5 players, with small models
whose weights and relation multipliers are scrambled: random playouts at
capacities 3 and 5 and mixed, as v0's tests use, and human positions from the
start-position file.

- **As v0** (``tests/g1889/test_model.py``, whose tests of these properties run
  here on v1 as written, through this module's fixtures): legal logits are
  finite and the rest -1e9; values are zero-sum where asked; flat records and
  dense planes agree; a batch equals its states run alone; permuting the
  visible candidates permutes their logits; a row's final embedding moves only
  its own pointer logits; hidden rows reach no legal slot.
- **v0 inside v1:** v0's weights copied to the slots layout 2 keeps, with zeros
  on what it adds, give v0's outputs from layout 1 on every state without a
  visible candidate. So the trunk, the relation path over 80 rows, and every
  slot a head kept are v0's.
- **Relations:** ids 0-20 and 25-28 have channels; a record of another id, of
  value 0, or naming a candidate row carries no weight; the finished mask is
  the records' over the trunk's rows, the players' moved up.
- **Raw slots:** with the trunk's output held fixed, every slot of every row
  moves exactly the logits model-design §12.2 lists for it, the new slots
  among them, at scrambled weights and at initialization.
- **Candidates:** nothing but a candidate's own logit depends on its row; the
  logit follows the selected hex's final row and those of the hexes across its
  exits, which are extraction's CANDIDATE_EXIT records.
- Construction, the loader's contract, what the two layouts' models refuse,
  the parameters against v0's, and checkpoints.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.relations import (
    RelationId as REL,
    RelationSize,
    get_relation_coord_data_batch,
    static_relation_planes,
)
from g1889.token_data import (
    TokenRow as R,
    TokenWidth2 as W,
    get_num_tokens,
    get_token_data,
    get_token_data_batch,
)
from games import get_game
from games.g1889 import POLICY_LAYOUT
from games.g1889_starts import load_start_positions
from mcts.evaluator import NNEvaluator, _infer_model_input_spec
from nn import _load_model_module, create_model, get_model_input_spec
from nn.model_contract import flatten_relation_records
from nn.policy_layout import PolicyLayout
from train.checkpoint import load_model_from_checkpoint, save_checkpoint
from train.config import TrainingConfig

import tests.g1889.test_model as T0
from tests.g1889.phases.conftest import policy_ids
from tests.g1889.test_extraction import _state
# v0's tests that hold for v1 as written. pytest collects them here, where the
# fixtures below hand them v1's model and layout 2's inputs.
from tests.g1889.test_model import (
    Inputs,
    Sample,
    test_a_batch_equals_its_states_alone,
    test_a_rows_final_embedding_moves_only_its_own_pointer_logits,
    test_autocast_with_fp16_tokens_runs_the_eval_server_s_dtypes,
    test_dense_and_flat_inputs_agree,
    test_hidden_rows_reach_no_legal_slot,
    test_identities_candidates_take_their_hexes,
    test_legal_logits_are_finite_and_the_rest_masked,
    test_permuting_the_visible_candidates_permutes_their_logits,
    test_the_batch_column_gives_each_state_its_own_records,
    test_the_forward_traces_without_graph_breaks,
    test_the_states_reach_every_phase,
    test_values_are_zero_sum_over_real_players,
)

REPO = Path(__file__).resolve().parents[2]
MODEL_PATH = str(REPO / 'nn' / 'transformer-1889-v1.py')
MODEL = _load_model_module(MODEL_PATH)
V0 = T0.MODEL
HUMAN_STARTS = REPO / 'g1889' / 'data' / 'human_starts.npz'

TOKEN_DIM = 109
OFFSETS = POLICY_LAYOUT.offsets
D_MODEL, NUM_HEADS, NUM_LAYERS = T0.D_MODEL, T0.NUM_HEADS, T0.NUM_LAYERS
GROUPS = {**T0.GROUPS, 'human': None}  # the human positions are at capacity 5
PER_PHASE = 3

CANDIDATES = range(int(R.ROW_CANDIDATE), int(R.ROW_PLAYER))
CANDIDATE_PHASES = (K.DPHASE_TILE, K.DPHASE_ER_BONUS)
# The relations with a channel, in channel order, and the trunk's rows at a capacity.
CHANNEL_IDS = [*range(21), 25, 26, 27, 28]
CANDIDATE_IDS = [21, 22, 23, 24]
PATH_SLOTS = range(2 + 41, 2 + 62)  # the tile block's path mask


def _trunk_rows(capacity: int) -> list[int]:
    return [*range(CANDIDATES.start), *range(CANDIDATES.stop, get_num_tokens(capacity))]


# Row types: the first row, the row count (None: one per player), the width,
# and the is_selected slots.
ROW_TYPES = {
    name: (first, count, getattr(W, f'TW2_{name}'), selections)
    for name, (first, count, _, selections) in T0.ROW_TYPES.items()
}


# =============================================================================
# STATES AND INPUTS
# =============================================================================

def _human_samples() -> list[Sample]:
    """Up to PER_PHASE positions of each decision phase, a player count each
    where the file has one."""
    positions = load_start_positions(HUMAN_STARTS)
    assert positions.max_players == 5
    flags = json.loads(positions.engine_options)
    rng = np.random.default_rng(0)
    samples = []
    for phase in range(K.NUM_DECISION_PHASES):
        at_phase = positions.decision_phase == phase
        for num_players in (3, 4, 5)[:PER_PHASE]:
            of_count = np.flatnonzero(at_phase & (positions.num_players == num_players))
            choices = of_count if len(of_count) else np.flatnonzero(at_phase)
            at = int(rng.choice(choices))
            row = positions.rows[at].copy()
            state = _state(row)
            for name, value in flags.items():
                setattr(state, name, value)
            samples.append(Sample(row, phase, policy_ids(state), int(positions.num_players[at])))
    return samples


def _tokens(rows: list[np.ndarray], max_players: int, layout_version: int = 2) -> torch.Tensor:
    x = np.zeros((len(rows), get_num_tokens(max_players), 108 if layout_version == 1 else TOKEN_DIM), np.float32)
    get_token_data_batch([row.copy() for row in rows], x, max_players, layout_version=layout_version)
    return torch.from_numpy(x)


def _inputs(samples: list[Sample], max_players: int) -> Inputs:
    rows = [s.row for s in samples]
    records = np.zeros((len(rows), int(RelationSize.MAX_RELATION_EDGES), 4), np.uint8)
    get_relation_coord_data_batch(rows, records, max_players)
    mask = torch.zeros(len(rows), POLICY_LAYOUT.unified_dim, dtype=torch.bool)
    for i, sample in enumerate(samples):
        mask[i, [OFFSETS[sample.phase] + action for action in sample.legal]] = True
    records_t = torch.from_numpy(records)
    return Inputs(
        _tokens(rows, max_players), mask, records_t, flatten_relation_records(records_t),
        torch.tensor([s.phase for s in samples]), samples, max_players,
    )


@pytest.fixture(scope='module')
def setups() -> dict[tuple[int, int], list[Sample]]:
    return {setup: T0._sample_setup(*setup) for setup in T0.SETUPS}


@pytest.fixture(scope='module')
def groups(setups) -> dict[str, Inputs]:
    playouts = {name: _inputs([s for setup in members for s in setups[setup]], members[0][1])
                for name, members in T0.GROUPS.items()}
    return {**playouts, 'human': _inputs(_human_samples(), 5)}


@pytest.fixture(params=GROUPS)
def inputs(request, groups) -> Inputs:
    return groups[request.param]


def _scrambled(model):
    """Weights with scale: at initialization the relation bias is zero and the
    logits are tiny, which would hide most of what these tests look for."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.15)
    nn.init.normal_(model.relation_bias_mult, std=0.5)
    return model


@pytest.fixture(scope='module')
def weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return _scrambled(_new_model(5)).state_dict()


def _new_model(capacity: int, zero_sum: bool = False, module: Any = MODEL):
    return module.G1889TransformerNet(module.TransformerConfig(
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


_run, _assert_same, _real_values = T0._run, T0._assert_same, T0._real_values


def _linears(head: nn.Sequential) -> list[nn.Linear]:
    return [module for module in head if isinstance(module, nn.Linear)]


def _unified(phase: int, ids) -> set[int]:
    return {OFFSETS[phase] + action for action in ids}


def _candidate_slots() -> set[int]:
    """Every unified slot the candidate head fills."""
    return (_unified(K.DPHASE_TILE, range(D.MAX_LAY_CANDIDATES))
            | _unified(K.DPHASE_ER_BONUS, [K.ER_BONUS_CANDIDATE + k for k in range(D.NUM_ER_CANDIDATES)]))


def _candidate_ids(k: int) -> set[int]:
    """The unified slots candidate row k's logit lands on."""
    return _unified(K.DPHASE_TILE, [k]) | (
        _unified(K.DPHASE_ER_BONUS, [K.ER_BONUS_CANDIDATE + k]) if k < D.NUM_ER_CANDIDATES else set())


def _moved_slots(after: torch.Tensor, before: torch.Tensor) -> list[set[int]]:
    """Per row of two ``_unified_logits``, the unified slots that differ beyond rounding."""
    moved = (after - before).abs() > 1e-6 * (1 + before.abs())
    return [set(row.nonzero().flatten().tolist()) for row in moved]


# =============================================================================
# THE STATES
# =============================================================================

def test_the_human_positions_fill_every_slot_a_head_newly_reads(groups):
    """At its head's decisions each new raw slot is filled in some of 60 human
    positions: no head reads a slot that is always zero there."""
    assert {s.num_players for s in groups['human'].samples} == {3, 4, 5}
    positions = load_start_positions(HUMAN_STARTS)
    rng = np.random.default_rng(1)
    for phase, first, count, slots in (
        (K.DPHASE_DIVIDEND, R.ROW_DIVIDEND, 1, range(7, 13)),
        (K.DPHASE_BUY_TRAIN, R.ROW_TRAIN, D.NUM_TRAIN_TYPES, (10, 14)),
        (K.DPHASE_PAR, R.ROW_PRICE, 1, range(24, 30)),
        (K.DPHASE_STOCK, R.ROW_CORP, D.NUM_CORPS, (58, 59, 60, 68, 69, 70, 71, 72)),
        (K.DPHASE_SELECT_TRAIN, R.ROW_TRAIN, D.NUM_TRAIN_TYPES, (13,)),
        (K.DPHASE_TRAIN_PRICE, R.ROW_TRAIN, D.NUM_TRAIN_TYPES, (13,)),
        (K.DPHASE_TILE, R.ROW_CANDIDATE, D.MAX_LAY_CANDIDATES, (108,)),
    ):
        at = rng.choice(np.flatnonzero(positions.decision_phase == phase), 60, replace=False)
        block = _tokens([positions.rows[i] for i in at], 5)[:, first:first + count]
        assert all(block[:, :, slot].any() for slot in slots), (phase, [bool(block[:, :, s].any()) for s in slots])


# =============================================================================
# V0 INSIDE V1
# =============================================================================

# (v1 slot, v0 slot, v1 value / v0 value) for each slot layout 2 keeps with
# layout 1's values at every decision, as tests/g1889/test_extraction_layout2.py
# holds them. Left out, and zeroed in both models: the slots whose values
# changed (GLOBAL 39-40, CORP 15-19 and 55 from 49, TRAIN 10-11), CORP 45 from
# 39 (the same only in an OR), and the slots only one layout has.
def _same(slots, shift: int = 0, scale: float = 1.0) -> list[tuple[int, int, float]]:
    return [(slot, slot + shift, scale) for slot in slots]


KEPT = {
    'global': _same(range(1, 39)) + _same(range(41, 44)),
    'price': _same(range(1, 24)),
    'dividend': _same([1]) + _same(range(2, 7), 1),
    'corp': (_same(range(1, 15)) + _same(range(20, 38)) + _same(range(46, 49), -6) + [(49, 43, 5 / 15)]
             + _same(range(50, 55), -6) + [(56, 50, 3 / 7)]),
    'private': _same(range(1, 18)),
    'train': _same(range(1, 10)) + _same([12]),
    'player': (_same(range(1, 11)) + _same(range(11, 25), 1) + _same(range(25, 31), 2)
               + [(31, 33, 1.0), (32, 35, 1.0), (33, 36, 3 / 7), (34, 37, 1.0)]),
}


def _v0_inside_v1(v0, v1) -> None:
    """Rewrite both models' weights in place so that v1 computes on layout 2
    what v0 computes on layout 1, wherever no candidate is visible."""
    d = D_MODEL
    old, new = v0.state_dict(), v1.state_dict()

    def columns(key: str, pairs: list[tuple[int, int, float]]) -> None:
        """v1's columns from v0's, each over its scale; every other column of both is zero."""
        kept_old = torch.zeros_like(old[key])
        new[key].zero_()
        for new_column, old_column, scale in pairs:
            kept_old[:, old_column] = old[key][:, old_column]
            new[key][:, new_column] = old[key][:, old_column] / scale
        old[key].copy_(kept_old)

    for name, pairs in KEPT.items():
        columns(f'{name}_proj.weight', [(a - 1, b - 1, scale) for a, b, scale in pairs])

    def head(name: str, pairs: list[tuple[int, int]]) -> None:
        columns(f'{name}_head.0.weight', [(a, b, 1.0) for a, b in pairs])

    def prefix(n: int) -> list[tuple[int, int]]:
        return [(i, i) for i in range(n)]

    # The raw columns each head adds or loses; a head's other inputs are v0's.
    head('stock_corp', prefix(3 * d + 24) + [(3 * d + 32 + i, 3 * d + 24 + i) for i in range(2)])
    head('par', prefix(4 * d + 19))
    head('dividend', prefix(4 * d) + [(4 * d, 4 * d)] + [(4 * d + i, 4 * d + 1 + i) for i in range(1, 6)])
    head('buy_train', prefix(9 * d) + [(9 * d + 10 * t + j, 9 * d + 8 * t + j)
                                        for t in range(D.NUM_TRAIN_TYPES) for j in range(8)])
    head('select_train', prefix(5 * d))  # TRAIN 10-11 changed values
    head('train_price', prefix(6 * d + 19))
    for name in ('sale', 'buy_train_corp'):  # their last raw slot is CORP 55, whose values changed
        head(name, prefix(old[f'{name}_head.0.weight'].shape[1] - 1))

    relation_mult = torch.zeros_like(new['relation_bias_mult'])
    relation_mult[:, :, :21] = old['relation_bias_mult'][:, :, :21]
    relation_mult[:, :, 25:] = old['relation_bias_mult'][:, :, 25:]
    new['relation_bias_mult'].copy_(relation_mult)
    untouched = {f'{name}_proj.weight' for name in KEPT} | {
        f'{name}_head.0.weight' for name in ('stock_corp', 'par', 'dividend', 'buy_train', 'select_train',
                                             'train_price', 'sale', 'buy_train_corp')}
    for key, value in old.items():
        if key not in untouched and key != 'relation_bias_mult' and not key.startswith('candidate'):
            new[key].copy_(value)
    v0.load_state_dict(old)
    v1.load_state_dict(new)


def test_v0s_weights_in_v1_give_v0s_outputs_wherever_no_candidate_is_visible(inputs):
    """v0 reading layout 1 against v1 reading layout 2, with v0's weights on the
    slots both layouts hold and zeros on the rest: the same legal logits and
    values at every decision but TILE and ER_BONUS, whose candidates v0's
    trunk holds and v1's does not."""
    torch.manual_seed(0)
    v0 = T0._new_model(inputs.max_players)
    _scrambled(v0)
    v1 = _new_model(inputs.max_players)
    _v0_inside_v1(v0, v1)
    x1 = _tokens([s.row for s in inputs.samples], inputs.max_players, layout_version=1)
    plain = [i for i, phase in enumerate(inputs.phases.tolist()) if phase not in CANDIDATE_PHASES]
    assert not inputs.x[plain][:, CANDIDATES.start:CANDIDATES.stop].any()
    assert len({inputs.phases[i].item() for i in plain}) == K.NUM_DECISION_PHASES - 2
    old_logits, old_values = _run(v0, x1, inputs.mask, inputs.flat)
    new_logits, new_values = _run(v1, inputs.x, inputs.mask, inputs.flat)
    _assert_same((new_logits[plain], new_values[plain]), (old_logits[plain], old_values[plain]), tol=2e-5)
    assert old_logits[inputs.mask].std() > 0.1
    # The copied weights are needed: v1's own, untouched, are far from v0.
    other, _ = _run(_new_model(inputs.max_players), inputs.x, inputs.mask, inputs.flat)
    assert (other - old_logits)[plain].abs().max() > 0.1


# =============================================================================
# RELATIONS
# =============================================================================

def test_only_the_channels_relations_and_the_trunks_rows_carry_weight(inputs, model_for):
    model = model_for(inputs.max_players)
    base = _run(model, inputs.x, inputs.mask, inputs.flat)
    rng = torch.Generator().manual_seed(1)
    batch, num_tokens, n = len(inputs.samples), inputs.x.shape[1], 4096
    trunk = torch.tensor(_trunk_rows(inputs.max_players))

    def pick(values, count=n):
        values = torch.as_tensor(values)
        return values[torch.randint(0, len(values), (count,), generator=rng)]

    cells = torch.stack([
        torch.randint(0, batch, (n,), generator=rng), pick(CHANNEL_IDS), pick(trunk), pick(trunk),
        torch.randint(1, 11, (n,), generator=rng),
    ], dim=1).to(torch.int16)

    no_channel = cells.clone()
    no_channel[:, 1] = pick([*CANDIDATE_IDS, *range(29, 256)])
    zero = cells.clone()
    zero[:, 4] = 0
    candidate_query, candidate_key = cells.clone(), cells.clone()
    candidate_query[:, 2] = pick(list(CANDIDATES))
    candidate_key[:, 3] = pick(list(CANDIDATES))
    junk = (no_channel, zero, candidate_query, candidate_key)
    for extra in junk:
        _assert_same(_run(model, inputs.x, inputs.mask, torch.cat([inputs.flat, extra])), base)
    mixed = torch.cat([*junk, inputs.flat])
    _assert_same(_run(model, inputs.x, inputs.mask, mixed[torch.randperm(len(mixed), generator=rng)]), base)
    # The same records on the trunk's rows, with channels and values, would have moved the outputs.
    logits, _ = _run(model, inputs.x, inputs.mask, torch.cat([inputs.flat, cells]))
    assert (logits - base[0])[inputs.mask].abs().max() > 0.1

    dense = inputs.dense()
    more = torch.randint(0, 11, (batch, 3, num_tokens, num_tokens), generator=rng, dtype=torch.uint8)
    _assert_same(_run(model, inputs.x, inputs.mask, torch.cat([dense, more], dim=1)), base)


def test_the_candidates_relations_change_nothing_and_relations_25_to_28_do(inputs, model_for):
    """Extraction emits the candidates' relations (21-24) for v0, and every
    record naming a candidate row is one of them. v1's outputs are those
    without them, exactly, on flat records and on dense planes; the SR's sales
    (25-26) and the route hexes (27-28) are consumed."""
    model = model_for(inputs.max_players)
    relation, rows = inputs.flat[:, 1], inputs.flat[:, 2:4]
    of_candidates = (relation >= 21) & (relation <= 24)
    on_candidate_rows = ((rows >= CANDIDATES.start) & (rows < CANDIDATES.stop)).any(1)
    assert of_candidates.any() and (of_candidates == on_candidate_rows).all()
    with_all = _run(model, inputs.x, inputs.mask, inputs.flat)
    _assert_same(with_all, _run(model, inputs.x, inputs.mask, inputs.flat[~of_candidates]), tol=0)
    dense = inputs.dense()
    assert dense[:, 21:25].any()
    without = dense.clone()
    without[:, 21:25] = 0
    _assert_same(_run(model, inputs.x, inputs.mask, dense), _run(model, inputs.x, inputs.mask, without), tol=0)

    for first in (int(REL.REL_SOLD_THIS_SR), int(REL.REL_ROUTE)):
        pair = (relation == first) | (relation == first + 1)
        assert pair.any()
        logits, _ = _run(model, inputs.x, inputs.mask, inputs.flat[~pair])
        assert (logits - with_all[0])[inputs.mask].abs().max() > 1e-3, first


def test_the_finished_mask_is_the_records_over_the_trunks_rows(groups, model_for):
    """Scaled records on their channels, the static planes, and hidden keys,
    over the 80 rows that are not candidates: a player's records land on its
    row among them, 15 above its input row."""
    inputs = groups['mixed']
    model = model_for(inputs.max_players)
    trunk = _trunk_rows(inputs.max_players)
    assert len(trunk) == model.cfg.num_trunk_tokens == 80
    tokens = model._project_tokens(inputs.x)[:, trunk]
    hidden = (inputs.x[:, trunk, 0] <= 0.5)[:, None, None, :]
    layer = 1
    with torch.inference_mode():
        bias = model._attention_bias(model._prepare_relations(inputs.flat), hidden, layer, tokens)
    assert bias.shape == (len(inputs.samples), NUM_HEADS, 80, 80)

    scales = np.ones(29)
    scales[[REL.REL_PLAYER_SHARES, REL.REL_CORP_SHARES]] = 0.1
    scales[[REL.REL_CORP_TRAINS, REL.REL_TRAIN_HELD_BY]] = 0.25
    mult = model.relation_bias_mult[layer].detach().numpy()
    assert mult.shape == (NUM_HEADS, 35)
    static = static_relation_planes(inputs.max_players).astype(np.float64)[:, trunk][:, :, trunk]
    np.testing.assert_array_equal(model._static_relations.numpy(), static)
    expected = np.zeros((len(inputs.samples), NUM_HEADS, 80, 80))
    expected += np.einsum('rij,hr->hij', static, mult[:, 25:])
    trunk_row = {row: i for i, row in enumerate(trunk)}
    assert trunk_row[R.ROW_PLAYER] == 75 and trunk_row[R.ROW_CANDIDATE - 1] == 74
    kept = players = 0
    for b, relation, query, key, value in inputs.flat.tolist():
        if relation in CHANNEL_IDS and query in trunk_row and key in trunk_row:
            expected[b, :, trunk_row[query], trunk_row[key]] += mult[:, CHANNEL_IDS.index(relation)] * value * scales[relation]
            kept += 1
            players += max(query, key) >= R.ROW_PLAYER
    assert 0 < players < kept < len(inputs.flat)
    visible = ~hidden.expand_as(bias)
    assert np.allclose(bias[visible].numpy(), expected[visible.numpy()], atol=1e-5)
    assert (bias[~visible] == torch.finfo(bias.dtype).min).all()


def test_capacity_changes_nothing_for_the_same_weights(setups, model_for):
    """The same weights and 3-player states at capacity 3 and at capacity 5,
    where the two padded player rows are hidden."""
    at_3, at_5 = _inputs(setups[3, 3], 3), _inputs(setups[3, 5], 5)
    logits_3, values_3 = _run(model_for(3), at_3.x, at_3.mask, at_3.flat)
    logits_5, values_5 = _run(model_for(5), at_5.x, at_5.mask, at_5.flat)
    _assert_same((logits_3, values_3), (logits_5, values_5[:, :3]))


# =============================================================================
# THE RAW SLOTS EACH HEAD READS
# =============================================================================

def _stock_ids(c: int) -> set[int]:
    return _unified(K.DPHASE_STOCK, [K.STOCK_IPO + c, K.STOCK_POOL + c, K.STOCK_PAR + c, K.STOCK_SELL + c])


def _if_selected(type_name: str, slot: int, ids: set[int]):
    first = ROW_TYPES[type_name][0]
    return lambda k, x: ids if x[first + k, slot] > 0.5 else set()


def _by_table(phase: int, table, first: int = 0):
    return lambda k, x: _unified(phase, [first + i for i, v in enumerate(table) if v == k])


def _private_ids(k: int, x: torch.Tensor) -> set[int]:
    pre, post = K.DPHASE_BUY_PRIVATE_PRE, K.DPHASE_BUY_PRIVATE_POST
    ids = _unified(pre, [K.BUY_PRIVATE_PRE_PRIVATE + k]) | _unified(post, [K.BUY_PRIVATE_POST_PRIVATE + k])
    if k == D.PRIVATE_DR:
        ids |= (_unified(K.DPHASE_STOCK, [K.STOCK_DR]) | _unified(pre, [K.BUY_PRIVATE_PRE_DR])
                | _unified(post, [K.BUY_PRIVATE_POST_DR]))
    if k == D.PRIVATE_MF:
        ids |= _unified(K.DPHASE_STOCK, [K.STOCK_MF]) | _unified(pre, [K.BUY_PRIVATE_PRE_MF])
    return ids


_SALE_IDS = _unified(K.DPHASE_SELL_AMOUNT, range(5)) | _unified(K.DPHASE_EMERGENCY_SELL_AMOUNT, range(5))
_PRICE_IDS = (_unified(K.DPHASE_BID, range(17)) | _unified(K.DPHASE_PAR, range(6))
              | _unified(K.DPHASE_TRAIN_PRICE, range(5)) | _unified(K.DPHASE_PRIVATE_PRICE, range(4)))
_BUY_TRAIN_IDS = _unified(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_PASS, K.BUY_TRAIN_DEPOT, K.BUY_TRAIN_POOL,
                                               K.BUY_TRAIN_BANKRUPT])

# model-design §12.2, every raw read of every head: (row type, slots, the
# unified slots row k of the type moves through them, given the state's rows).
# New in v1: PRICE 24-29; CORP 58-60 and 68-72; DIVIDEND 7-12; TRAIN 10 and 14
# for BUY_TRAIN, 13 for SELECT_TRAIN and TRAIN_PRICE; CANDIDATE 108.
READS = [
    ('PRIVATE', range(7, 14), lambda k, x: (
        (_unified(K.DPHASE_AUCTION, [K.AUCTION_BID + k - 1]) if k >= 1 else set())
        | (_unified(K.DPHASE_AUCTION, [K.AUCTION_BUY]) if x[R.ROW_PRIVATE + k, 2] > 0.5 else set()))),
    ('PRIVATE', (16, 17), _private_ids),
    ('PRICE', range(5, 24), lambda k, x: _PRICE_IDS),
    ('PRICE', range(24, 30), lambda k, x: _unified(K.DPHASE_PAR, range(6))),
    ('CORP', [*range(7, 26), *range(50, 55), *range(58, 61), *range(68, 73)], lambda k, x: _stock_ids(k)),
    ('CORP', [*range(15, 20), 54, 55], _if_selected('CORP', 3, _SALE_IDS)),
    ('CORP', [*range(15, 20), 52, 55],
     lambda k, x: _unified(K.DPHASE_BUY_TRAIN, [K.BUY_TRAIN_CORP + k, K.BUY_TRAIN_SELL + k])),
    *(('PLAYER', (11 + c, 18 + c), _if_selected('PLAYER', 1, _stock_ids(c))) for c in range(D.NUM_CORPS)),
    ('HEX', range(92, 103), _by_table(K.DPHASE_TRACK, D.TRACK_HEXES, K.TRACK_HEX)),
    ('HEX', (103, 104), _by_table(K.DPHASE_TOKEN, D.TOKEN_HEXES, K.TOKEN_HEX)),
    ('CANDIDATE', [1, *range(99, 109)], lambda k, x: _candidate_ids(k)),
    ('DIVIDEND', range(1, 13), lambda k, x: _unified(K.DPHASE_DIVIDEND, range(2))),
    ('TRAIN', (12,), _by_table(K.DPHASE_DISCARD, D.DISCARD_TYPES)),
    ('TRAIN', [*range(2, 11), 14], lambda k, x: _BUY_TRAIN_IDS),
    ('TRAIN', (10, 11, 13), lambda k, x: (
        _unified(K.DPHASE_SELECT_TRAIN, [K.SELECT_TRAIN_TYPE + k])
        | _by_table(K.DPHASE_SELECT_TRAIN, D.TRADE_IN_TYPE, K.SELECT_TRAIN_TRADE_IN)(k, x))),
    ('TRAIN', (13,), _if_selected('TRAIN', 1, _unified(K.DPHASE_TRAIN_PRICE, range(5)))),
]
# The rows read, of the types with many: every table's rows and others.
SOME_ROWS = {'HEX': range(0, D.NUM_HEXES, 4), 'CANDIDATE': (0, 1, 7, 8, 14)}


def _check_raw_reads(model, x: torch.Tensor, flat: torch.Tensor) -> int:
    """With the trunk's output fixed, each slot of each row moves the unified
    slots READS gives it and no other, and a read's slots together reach all of
    its ids. Selections are context and stay out of it. A candidate's path
    mask is read too, for its exits: a slot can move the row's logit and no
    other (test_a_path_gives_a_candidate_its_exits). Returns the reads that
    had ids to move."""
    with torch.inference_mode():
        tokens = model._encode(x, flat)
    reached = 0
    for type_name, (first, count, width, selections) in ROW_TYPES.items():
        reads = [(slots, own) for name, slots, own in READS if name == type_name]
        slots = [s for s in range(1, int(width)) if s not in selections]
        for k in SOME_ROWS.get(type_name, range(int(count or x.shape[1] - R.ROW_PLAYER))):
            row = int(first) + k
            edited = x.repeat(len(slots), 1, 1)
            edited[torch.arange(len(slots)), row, slots] += 5
            fixed = tokens.repeat(len(slots), 1, 1)
            with torch.inference_mode():
                moved = dict(zip(slots, _moved_slots(
                    model._unified_logits(fixed, edited), model._unified_logits(fixed, x.repeat(len(slots), 1, 1)),
                ), strict=True))
            for slot in slots:
                expected = set().union(*(own(k, x[0]) for read_slots, own in reads if slot in read_slots))
                if type_name == 'CANDIDATE' and slot in PATH_SLOTS:
                    assert moved[slot] <= _candidate_ids(k), (type_name, k, slot, moved[slot])
                    continue
                # One slot's move can round away on one id, so its ids are a subset.
                assert moved[slot] <= expected and bool(moved[slot]) == bool(expected), (
                    type_name, k, slot, moved[slot], expected)
            for read_slots, own in reads:
                expected = own(k, x[0])
                assert set().union(*(moved[s] for s in read_slots)) >= expected, (type_name, k, list(read_slots))
                reached += bool(expected)
    return reached


@pytest.mark.parametrize('at_initialization', [False, True], ids=['scrambled', 'initial'])
def test_with_the_trunk_fixed_each_slot_moves_the_logits_listed_for_it_only(at_initialization, groups, weights):
    """At decisions whose selections switch on the conditional reads: a sale's
    corp (CORP 3), a priced train (TRAIN 1), and the topmost private (PRIVATE
    2). At initialization too: no head starts with zero weights on a slot."""
    inputs = groups['mixed']
    torch.manual_seed(1)
    model = _new_model(inputs.max_players)
    if not at_initialization:
        model.load_state_dict(weights)
    reached = 0
    for phase in (K.DPHASE_AUCTION, K.DPHASE_SELL_AMOUNT, K.DPHASE_TRAIN_PRICE, K.DPHASE_TILE):
        x, _, flat = inputs.alone(inputs.at(phase)[0])
        reached += _check_raw_reads(model, x, flat)
    # Each conditional read was on somewhere.
    sale = inputs.x[inputs.at(K.DPHASE_SELL_AMOUNT)[0], R.ROW_CORP:R.ROW_PRIVATE, 3]
    priced = inputs.x[inputs.at(K.DPHASE_TRAIN_PRICE)[0], R.ROW_TRAIN:R.ROW_HEX, 1]
    topmost = inputs.x[inputs.at(K.DPHASE_AUCTION)[0], R.ROW_PRIVATE:R.ROW_TRAIN, 2]
    assert sale.sum() == priced.sum() == topmost.sum() == 1
    assert reached > 4 * 60


def test_every_head_output_lands_on_its_slots(model_for):
    """A head's last bias, moved by one, moves its slots by one and no others;
    the selection AUCTION's buy reads is set. Together they cover every slot."""
    model = model_for(5)
    tokens = torch.zeros(1, 95, D_MODEL)
    x = torch.zeros(1, 95, TOKEN_DIM)
    x[0, R.ROW_PRIVATE + 2, 2] = 1
    expected = T0._expected_outputs()
    with torch.inference_mode():
        base = model._unified_logits(tokens, x)
    for (head, column), slots in expected.items():
        bias = _linears(getattr(model, head))[-1].bias
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
# CANDIDATES
# =============================================================================

def test_a_candidates_row_reaches_its_own_logit_and_nothing_else(inputs, model_for):
    """The trunk runs without the candidate rows. Whatever they hold, key mask
    included, every value and every logit outside the candidate head's slots
    is the same bit for bit, so states without a legal candidate do not depend
    on them at all; and one candidate's row moves that candidate's logit only."""
    model = model_for(inputs.max_players)
    logits, values = _run(model, inputs.x, torch.ones_like(inputs.mask), inputs.flat)
    rng = torch.Generator().manual_seed(6)
    junk = inputs.x.clone()
    junk[:, CANDIDATES.start:CANDIDATES.stop] = torch.rand(len(inputs.samples), len(CANDIDATES), TOKEN_DIM, generator=rng)
    junk_logits, junk_values = _run(model, junk, torch.ones_like(inputs.mask), inputs.flat)
    others = sorted(set(range(POLICY_LAYOUT.unified_dim)) - _candidate_slots())
    assert torch.equal(junk_values, values) and torch.equal(junk_logits[:, others], logits[:, others])
    assert (junk_logits - logits)[:, sorted(_candidate_slots())].abs().min() > 1e-4
    # So under the legal mask, nothing moves where no candidate is legal.
    plain = [i for i, phase in enumerate(inputs.phases.tolist()) if phase not in CANDIDATE_PHASES]
    masked, _ = _run(model, inputs.x, inputs.mask, inputs.flat)
    junk_masked, _ = _run(model, junk, inputs.mask, inputs.flat)
    assert torch.equal(masked[plain], junk_masked[plain])

    for i in [j for phase in CANDIDATE_PHASES for j in inputs.at(phase)[:2]]:
        x, _, flat = inputs.alone(i)
        edited = x.repeat(len(CANDIDATES), 1, 1)
        for k, row in enumerate(CANDIDATES):
            edited[k, row, 1:] += torch.rand(TOKEN_DIM - 1, generator=rng)
        batch = torch.zeros(len(CANDIDATES), dtype=torch.int16)
        records = torch.cat([flat.clone() for _ in CANDIDATES])
        records[:, 0] = torch.arange(len(CANDIDATES)).repeat_interleave(len(flat)).to(batch.dtype)
        ones = torch.ones(len(CANDIDATES), POLICY_LAYOUT.unified_dim, dtype=torch.bool)
        before, _ = _run(model, x.repeat(len(CANDIDATES), 1, 1), ones, records)
        after, _ = _run(model, edited, ones, records)
        for k, moved in enumerate(_moved_slots(after, before)):
            assert moved == _candidate_ids(k), (i, k, moved)


def _candidate_geometry(inputs: Inputs, i: int) -> tuple[int, dict[int, set[int]]]:
    """The selected hex's row, and for each hex row the candidates with an exit
    toward it, from the state's CANDIDATE_AT and CANDIDATE_EXIT records."""
    records = inputs.flat[inputs.flat[:, 0] == i].tolist()
    at = {key for _, relation, _, key, _ in records if relation == REL.REL_CANDIDATE_AT}
    (selected,) = at
    toward: dict[int, set[int]] = {}
    for _, relation, query, key, _ in records:
        if relation == REL.REL_CANDIDATE_EXIT:
            toward.setdefault(key, set()).add(query - CANDIDATES.start)
    return selected, toward


def test_a_candidates_logit_follows_the_selected_hex_and_the_hexes_across_its_exits(inputs, model_for):
    """With the final rows edited one hex at a time: the selected hex moves every
    candidate's logit, a hex across an exit moves the logits of the candidates
    that exit toward it, as CANDIDATE_EXIT records them, and any other hex
    moves none."""
    model = model_for(inputs.max_players)
    hexes = range(int(R.ROW_HEX), int(R.ROW_CANDIDATE))
    block = sorted(_candidate_slots())
    tested = exits = 0
    for i in [j for phase in CANDIDATE_PHASES for j in inputs.at(phase)]:
        x, _, flat = inputs.alone(i)
        selected, toward = _candidate_geometry(inputs, i)
        assert x[0, selected, 1] == 1 and x[0, hexes.start:hexes.stop, 1].sum() == 1
        with torch.inference_mode():
            tokens = model._encode(x, flat)
            base = model._unified_logits(tokens, x)
            edited = tokens.repeat(len(hexes), 1, 1)
            edited[torch.arange(len(hexes)), list(hexes)] += 0.5
            after = model._unified_logits(edited, x.repeat(len(hexes), 1, 1))
        for row, moved in zip(hexes, _moved_slots(after[:, block], base[:, block].expand(len(hexes), -1)), strict=True):
            moved = {block[j] for j in moved}
            if row == selected:
                expected = _candidate_slots()
            else:
                expected = set().union(*(_candidate_ids(k) for k in toward.get(row, ())))
            assert moved == expected, (i, row, moved, expected)
        tested += 1
        exits += len(toward)
    assert tested >= 2 and exits >= 2


def test_a_path_gives_a_candidate_its_exits(inputs, model_for):
    """With the trunk fixed, one path set on a candidate row without any: the
    row's logit moves if a side the path touches has a hex across it from the
    selected hex, and nothing moves otherwise."""
    model = model_for(inputs.max_players)
    # Biases, which a missing hex's zero row would project to, are zero in these weights.
    assert not any(side.bias.any() for side in model.exit_side_proj)
    i = inputs.at(K.DPHASE_TILE)[0]
    x, _, flat = inputs.alone(i)
    selected, _ = _candidate_geometry(inputs, i)
    k = len(CANDIDATES) - 1
    assert not x[0, CANDIDATES.start + k].any()
    with torch.inference_mode():
        tokens = model._encode(x, flat)
        edited = x.repeat(D.NUM_PAIRS, 1, 1)
        edited[torch.arange(D.NUM_PAIRS), CANDIDATES.start + k, list(PATH_SLOTS)] = 1
        after = model._unified_logits(tokens.repeat(D.NUM_PAIRS, 1, 1), edited)
        before = model._unified_logits(tokens.repeat(D.NUM_PAIRS, 1, 1), x.repeat(D.NUM_PAIRS, 1, 1))
    across = D.HEX_NEIGHBOR[selected - R.ROW_HEX]
    seen = set()
    for pair, moved in enumerate(_moved_slots(after, before)):
        sides = [end for end in (D.PAIR_A[pair], D.PAIR_B[pair]) if end < D.NUM_SIDES]
        has_exit = any(across[side] >= 0 for side in sides)
        assert moved == (_candidate_ids(k) if has_exit else set()), (pair, moved)
        seen.add(has_exit)
    assert True in seen


def test_exits_are_the_sides_the_candidates_paths_touch(model_for):
    """The head's static tables: a pair's sides, and the hex across each side."""
    model = model_for(5)
    assert model._pair_sides.shape == (D.NUM_PAIRS, D.NUM_SIDES)
    for pair, (a, b) in enumerate(zip(D.PAIR_A, D.PAIR_B, strict=True)):
        assert set(model._pair_sides[pair].nonzero().flatten().tolist()) == {end for end in (a, b) if end < D.NUM_SIDES}
    for side in range(D.NUM_SIDES):
        for h in range(D.NUM_HEXES):
            across = D.HEX_NEIGHBOR[h][side]
            assert model._side_planes[side, h].nonzero().flatten().tolist() == ([across] if across >= 0 else [])


# =============================================================================
# THE PROJECTIONS
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
    assert base.shape == (1, num_tokens, D_MODEL)
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
                    expected = {first + k, *CANDIDATES}
                else:
                    expected = {first + k}
                assert rows == expected, (name, k, slot, rows)


def test_hexes_and_candidates_share_the_tile_projection(model_for):
    model = model_for(5)
    x = torch.zeros(1, 95, TOKEN_DIM)
    hex_row, candidate_row = int(R.ROW_HEX) + 3, int(R.ROW_CANDIDATE) + 2
    with torch.inference_mode():
        base = model._project_tokens(x)
    x[0, [hex_row, candidate_row], T0.TILE_BLOCK.start:T0.TILE_BLOCK.stop] = torch.rand(len(T0.TILE_BLOCK))
    with torch.inference_mode():
        moved = model._project_tokens(x) - base
    assert torch.allclose(moved[0, hex_row], moved[0, candidate_row], atol=1e-6)
    assert moved[0, hex_row].abs().max() > 1e-2
    assert model.tile_proj.bias is None


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
            get_token_data(state, tokens, 5, layout_version=2)
            row = R.ROW_HEX + D.PORT_LAY_HEX[action]
            np.testing.assert_array_equal(tokens[row, PATH_SLOTS.start:PATH_SLOTS.stop], model._port_masks[action].numpy())
            tested += 1
    assert tested >= 1


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
    assert MODEL.INPUT_LAYOUT_VERSION == 2 and get_model_input_spec(config).layout_version == 2
    assert (model.cfg.layout_version, model.cfg.num_players) == (2, capacity)
    assert (model.cfg.d_model, model.cfg.num_heads, model.cfg.num_layers) == (D_MODEL, NUM_HEADS, NUM_LAYERS)
    assert model.cfg.zero_sum_values is True and model.cfg.token_dim == 109
    assert (model.cfg.num_tokens, model.cfg.num_trunk_tokens) == (90 + capacity, 75 + capacity)
    # 25 dynamic channels (ids 0-20 and 25-28) and 10 static, zero at initialization.
    assert model.relation_bias_mult.shape == (NUM_LAYERS, NUM_HEADS, 35) and not model.relation_bias_mult.any()
    assert model._relation_ids.tolist() == CHANNEL_IDS
    assert model._relation_channel.tolist() == [CHANNEL_IDS.index(i) if i in CHANNEL_IDS else -1 for i in range(256)]
    assert model._trunk_rows.tolist() == _trunk_rows(capacity)
    assert model._row_map.tolist() == [
        _trunk_rows(capacity).index(row) if row in _trunk_rows(capacity) else -1 for row in range(256)]
    assert model._static_relations.shape == (10, 75 + capacity, 75 + capacity)
    # NNEvaluator infers its spec from the model's cfg: the input keeps every row.
    inferred = _infer_model_input_spec(get_game(config), model, capacity)
    assert (inferred.num_tokens, inferred.token_dim, inferred.policy_dim, inferred.value_dim) == (
        90 + capacity, 109, 228, capacity,
    )
    # Buffers are built at construction, not checkpointed.
    assert all(key.endswith(('weight', 'bias')) or key == 'relation_bias_mult' for key in model.state_dict())
    assert isinstance(model.value_head, nn.Module)


@pytest.mark.parametrize('field, value', [
    ('game', 'rss'), ('layout_version', 1), ('token_dim', 108), ('num_tokens', 94),
    ('policy_dim', 227), ('value_dim', 4),
])
def test_build_model_asserts_the_spec(field, value):
    config = _config()
    spec = get_model_input_spec(config)
    MODEL.build_model(config, spec)
    with pytest.raises(AssertionError):
        MODEL.build_model(config, replace(spec, **{field: value}))


def test_each_layouts_model_refuses_the_others_tokens_and_weights(groups):
    """A model takes its own layout's width and no other, by the forward's shape
    check; the evaluator fills each model's own layout; and neither model's
    weights load into the other."""
    inputs = groups['capacity 5']
    v0, v1 = T0._new_model(5), _new_model(5)
    x1 = _tokens([s.row for s in inputs.samples], 5, layout_version=1)
    _run(v0, x1, inputs.mask, inputs.flat)
    with pytest.raises(AssertionError, match=r'x shape must be \(batch, 95, 109\); got \(\d+, 95, 108\)'):
        _run(v1, x1, inputs.mask, inputs.flat)
    with pytest.raises(AssertionError, match=r'x shape must be \(batch, 95, 108\); got \(\d+, 95, 109\)'):
        _run(v0, inputs.x, inputs.mask, inputs.flat)
    # Each model's spec is its own layout's, and a spec of the other's is refused.
    game = get_game(_config())
    specs = {path: get_model_input_spec(_config(model_path=path)) for path in (T0.MODEL_PATH, MODEL_PATH)}
    assert [(spec.layout_version, spec.token_dim) for spec in specs.values()] == [(1, 108), (2, 109)]
    for model, own, other in ((v0, T0.MODEL_PATH, MODEL_PATH), (v1, MODEL_PATH, T0.MODEL_PATH)):
        evaluator = NNEvaluator(game, model, torch.device('cpu'), num_players=5)
        assert (evaluator.layout_version, evaluator.token_dim) == (specs[own].layout_version, specs[own].token_dim)
        with pytest.raises(ValueError, match='input_spec does not match the model input contract'):
            NNEvaluator(game, model, torch.device('cpu'), num_players=5, input_spec=specs[other])
    with pytest.raises(RuntimeError, match='size mismatch'):
        v1.load_state_dict(v0.state_dict())
    with pytest.raises(RuntimeError, match='size mismatch'):
        v0.load_state_dict(v1.state_dict())


def test_construction_checks_the_row_widths(monkeypatch):
    MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))
    real = MODEL.get_token_widths

    def drifted(max_players, layout_version):
        assert layout_version == 2
        widths = real(max_players, layout_version)
        widths[R.ROW_CORP] += 1
        return widths
    monkeypatch.setattr(MODEL, 'get_token_widths', drifted)
    with pytest.raises(AssertionError, match='token layout drift'):
        MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))


@pytest.mark.parametrize('phase', [K.DPHASE_BID, K.DPHASE_TILE, K.DPHASE_ER_BONUS, K.DPHASE_PORT])
def test_construction_checks_each_block(phase, monkeypatch):
    sizes = list(POLICY_LAYOUT.phase_sizes)
    sizes[phase] += 1
    monkeypatch.setattr(MODEL, 'POLICY_LAYOUT', PolicyLayout(
        tuple(sizes), POLICY_LAYOUT.phase_names, POLICY_LAYOUT.phases_with_pass_slot,
    ))
    with pytest.raises(AssertionError, match='policy block widths'):
        MODEL.G1889TransformerNet(MODEL.TransformerConfig(d_model=D_MODEL, num_layers=1))


def test_the_parameters_are_v0s_but_for_the_changes():
    """At 256 x 15: the trunk, the embeddings, the value head, and every head
    and projection whose inputs did not change have v0's shapes. What differs
    is the projections of the rows that grew, the first layer of the heads that
    read new slots, and the candidate head."""
    sizes = {'d_model': 256, 'num_heads': 4, 'num_layers': 15, 'ff_mult': 3.0}
    v1: Any = create_model(_config(5, **sizes))
    v0: Any = create_model(_config(5, model_path=T0.MODEL_PATH, **sizes))
    new = {name: tuple(p.shape) for name, p in v1.named_parameters()}
    old = {name: tuple(p.shape) for name, p in v0.named_parameters()}
    d = 256
    widened = {  # the first layer's inputs, v0's and v1's
        'global_proj.weight': (43, 89), 'price_proj.weight': (23, 29), 'dividend_proj.weight': (7, 12),
        'corp_proj.weight': (50, 76), 'train_proj.weight': (12, 14), 'candidate_proj.weight': (31, 32),
        'player_proj.weight': (37, 39),
        'stock_corp_head.0.weight': (3 * d + 24 + 2, 3 * d + 32 + 2), 'par_head.0.weight': (4 * d + 19, 4 * d + 25),
        'dividend_head.0.weight': (4 * d + 7, 4 * d + 12),
        'buy_train_head.0.weight': (3 * d + 6 * (d + 8), 3 * d + 6 * (d + 10)),
        'select_train_head.0.weight': (5 * d + 2, 5 * d + 3),
        'train_price_head.0.weight': (6 * d + 19, 6 * d + 20),
    }
    candidate = {
        'candidate_norm.weight': (d,), 'exit_proj.weight': (d, d // 4), 'exit_proj.bias': (d,),
        **{f'exit_side_proj.{s}.weight': (d // 4, d) for s in range(6)},
        **{f'exit_side_proj.{s}.bias': (d // 4,) for s in range(6)},
        'candidate_head.0.weight': (d, 6 * d + 11),
        'candidate_head.2.weight': (d, d), 'candidate_head.2.bias': (d,),
        'candidate_head.4.weight': (1, d), 'candidate_head.4.bias': (1,),
    }
    assert {name: shape for name, shape in new.items() if old.get(name) != shape} == {
        **{name: (d, after) for name, (_, after) in widened.items()}, **candidate,
    }
    assert {name: shape for name, shape in old.items() if new.get(name) != shape} == {
        **{name: (d, before) for name, (before, _) in widened.items()},
        'candidate_head.0.weight': (d, 5 * d + 10),
        'candidate_head.2.weight': (1, d), 'candidate_head.2.bias': (1,),
    }
    total = sum(p.numel() for p in v1.parameters())
    assert total == sum(p.numel() for p in create_model(_config(3, **sizes)).parameters())
    assert total == 18_497_016 and sum(p.numel() for p in v0.parameters()) == 18_218_872
    assert sum(p.numel() for p in v1.blocks.parameters()) == sum(p.numel() for p in v0.blocks.parameters())


def test_checkpoints_rebuild_through_build_model(tmp_path, weights, groups):
    inputs = groups['mixed']
    config = _config(zero_sum_values=True)
    model = create_model(config)
    model.load_state_dict(weights)
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path=path, epoch=0, model=model, trainer_state={'global_step': 0}, config=config,
                    metrics={}, buffer_stats={'size': 0, 'capacity': 0})
    loaded, loaded_config, _ = load_model_from_checkpoint(path, torch.device('cpu'))
    assert loaded_config.game == '1889' and isinstance(loaded, MODEL.G1889TransformerNet)
    assert loaded_config.model_path == MODEL_PATH
    outputs = _run(model.eval(), inputs.x, inputs.mask, inputs.flat)
    assert all(torch.equal(a, b) for a, b in zip(outputs, _run(loaded.eval(), inputs.x, inputs.mask, inputs.flat)))


def test_diagnostics_and_optimizer_routing(groups, model_for):
    inputs = groups['mixed']
    model = model_for(inputs.max_players)
    logits, _ = _run(model, inputs.x, inputs.mask, inputs.flat)
    stats = model.pass_action_logit_abs(logits, inputs.mask, inputs.phases)
    assert stats.shape == (2 * len(K.PHASES_WITH_PASS_SLOT),)
    for index, phase in enumerate(K.PHASES_WITH_PASS_SLOT):
        rows = inputs.phases == phase
        legal = inputs.mask[rows, OFFSETS[phase]]
        expected = logits[rows, OFFSETS[phase]][legal].abs().mean() if legal.any() else torch.tensor(0.0)
        assert torch.isclose(stats[2 * index], expected, atol=1e-5)

    # AdamW takes the projections of raw rows and each head's last layer; Muon
    # the hidden layers, the candidate head's exit projections among them.
    layers = model.input_output_layers()
    assert len({id(layer) for layer in layers}) == len(layers) == 10 + 19
    assert model.tile_proj in layers and all(isinstance(layer, nn.Linear) for layer in layers)
    for name, module in model.named_children():
        if name.endswith('_proj') and not name.startswith('exit'):
            assert module in layers
        elif name.endswith('_head'):
            *hidden, last = _linears(module)
            assert last in layers and not any(layer in layers for layer in hidden)
    assert len(_linears(model.candidate_head)) == 3
    assert model.exit_proj not in layers and not any(side in layers for side in model.exit_side_proj)


def test_every_parameter_learns_through_flat_records(groups, model_for):
    """A training step's graph: the in-place bias path backpropagates into every
    weight, the candidate head's among them, and into every relation column."""
    inputs = groups['mixed']
    model = model_for(inputs.max_players, zero_sum=True).train()
    logits, values = model(inputs.x, torch.ones_like(inputs.mask), inputs.flat)
    # Squared: zero-sum values have no gradient on their plain sum.
    (logits.sum() + values.square().sum()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    columns = model.relation_bias_mult.grad.abs().sum((0, 1))
    assert set(inputs.flat[:, 1].unique().tolist()) == set(range(29))
    assert (columns > 0).all() and len(columns) == 35
