"""Differential replay of the 1889 fixtures against Ruby (1889/engine-design.md
§10.2-3), and the initial states of fixtures and fresh games (6.2's acceptance).

Replay walks a fixture's records, maps each Ruby action to engine decisions,
and applies them through ``DRIVER`` with history. The driver auto-applies
forced decisions, which Ruby may log as actions or skip silently. The sync
rule: a Ruby action that matches a forced decision applied since the last sync,
by the same player, is consumed instead of applied; one that matches none
means Ruby skipped the pending forced decisions, so they are dropped. States
are compared whenever no forced decision is pending.

Replay stops at the first unsupported action, engine NotImplementedError, or
mismatch, and at the driver's development stop (an automated step from a later
sub-step, where the engine rests). ``REPLAY_FLOOR`` holds the last record index
each fixture must match; each sub-step raises it, so progress ratchets.
``replay_extract(..., start=i)`` starts from ``load_state`` of record i, so
later rounds replay before the engine can reach them.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.corp import CORPS
from g1889.entities.hexmap import MAP
from g1889.entities.private import PRIVATES
from g1889.entities.turn import TURN
from g1889.state import GameState, get_global_fields, get_layout

from tests.g1889.replay.action_map import Decision, ReplayEnd, UnsupportedAction, map_action
from tests.g1889.replay.extracts import Extract, FIXTURES, load_fixture_extract, load_new_game_extract
from tests.g1889.replay.snapshot import Snapshot, compare, load_state

# Sub-step 6.4: through the first SR. Its last record starts the first OR,
# where the engine rests at OR_START (the development stop).
REPLAY_FLOOR = {'314': 21, '962': 24, 'hs_jmmljkbw_17502': 26, 'hs_lhglbbiz_17502': 26}

# A forced decision the driver applied: (decision phase, id, the player it was for).
Forced = tuple[int, int, int]
ACTIVE_PLAYER_SLOT = get_layout(D.MIN_PLAYERS).global_offset + get_global_fields().active_player


@dataclass
class Progress:
    fixture: str
    records: int
    matched: int = -1              # the last record whose snapshot compared equal
    stop: str | None = None        # 'unsupported', 'engine', or 'mismatch'; None when replay ran out
    detail: str = ''

    def __str__(self) -> str:
        end = f'stopped ({self.stop}): {self.detail}' if self.stop else f'complete ({self.detail or "all records"})'
        return f'{self.fixture}: matched through record {self.matched} of {self.records - 1}; {end}'


def _phase_name(phase: int) -> str:
    return K.DecisionPhase(phase).name if phase >= 0 else 'an automated step'


def new_game(extract: Extract, max_players: int = 0) -> GameState:
    """``initialize_game`` with Ruby's first player, mapped into seat order."""
    ids = [p['id'] for p in extract['players']]
    first = extract['records'][0]['state']['game']['player_order'][0]
    state = GameState(len(ids), max_players=max_players)
    state.initialize_game(len(ids), priority_player=ids.index(first))
    return state


def consume(pending: list[Forced], decisions_in: Callable[[int], list[Decision]],
            actors: set[int]) -> list[Decision] | None:
    """The sync rule for one Ruby action by one of ``actors``. ``decisions_in(phase)``
    maps the action as taken in ``phase``. If its decisions continue the pending
    forced ones for that player from some entry on, drop the pending decisions
    through the match and return the decisions still to apply; return None when
    nothing matches."""
    for j, (phase, _, _) in enumerate(pending):
        try:
            decisions = decisions_in(phase)
        except UnsupportedAction:
            continue
        m = min(len(decisions), len(pending) - j)
        if m and all(pending[j + k] == (*decisions[k], player) and player in actors
                     for k, (*_, player) in enumerate(pending[j:j + m])):
            del pending[:j + m]
            return decisions[m:]
    return None


def _apply(state: GameState, decisions: list[Decision], pending: list[Forced]) -> str | None:
    """Apply decisions through the driver, queueing the forced ones it adds; an error message on failure."""
    for phase, action_id in decisions:
        if pending:  # forced inside one Ruby action: it must be the decision Ruby made
            if pending[0][:2] != (phase, action_id):
                return f'engine forced {pending[0][:2]} where Ruby chose {(phase, action_id)}'
            pending.pop(0)
            continue
        engine_phase = get_decision_phase_py(state)
        if engine_phase != phase:
            return f'engine is at {_phase_name(engine_phase)}, Ruby acts in {_phase_name(phase)}'
        history: list = []
        if DRIVER.apply_action(state, action_id, history) == STATUS_INVALID:
            return f'{_phase_name(phase)} id {action_id} is illegal in the engine'
        pending.extend((p, a, int(row[ACTIVE_PLAYER_SLOT])) for row, p, a in history[1:] if p >= 0)
    return None


def _at_development_stop(state: GameState) -> bool:
    """The driver rests at an automated step only at its development stop (6.9 removes it)."""
    return get_decision_phase_py(state) < 0 and TURN.round(state) != K.ROUND_GAME_OVER


# TEMPORARY, until OR_START lands (6.5): after the SR's last action the engine
# rests at OR_START, while Ruby's snapshot has started the OR. Ruby paid each
# owned private's revenue, placed the floated corps' home tokens, and gave the
# turn to the first corp (``Round::Operating#setup``). Undo the first two in
# the snapshot and skip the OR's turn fields, so the SR's end compares in full.
OR_TURN_FIELDS = ('game.step', 'game.active_players', 'game.current_entity', 'game.round_entities',
                  'game.entity_index', 'operating.')


def _before_or_start(state: GameState, snap: Snapshot) -> Snapshot:
    snap = copy.deepcopy(snap)
    for private in PRIVATES:
        owner, revenue = private.owner(state), private.revenue(state)
        if owner >= K.OWNER_CORP:
            snap['corporations'][D.CORP_SYMS[owner - K.OWNER_CORP]]['cash'] -= revenue
        elif owner >= K.OWNER_PLAYER:
            snap['players'][owner - K.OWNER_PLAYER]['cash'] -= revenue
        else:
            continue
        snap['game']['bank_cash'] += revenue
    for c, corp in enumerate(CORPS):
        home = D.CORP_HOME[c]
        if corp.floated(state) and not MAP.tokens(state, home) >> c & 1:
            ruby_hex, ruby_corp = snap['hexes'][D.HEX_IDS[home]], snap['corporations'][corp.sym]
            ruby_hex['tokens'].remove(corp.sym)
            ruby_hex['reserved'].append(corp.sym)
            ruby_corp['token_hexes'].remove(D.HEX_IDS[home])
            ruby_corp['tokens_left'] += 1
    return snap


def compare_at_rest(state: GameState, snap: Snapshot) -> list[str]:
    """``compare``, allowing for the development stop at OR_START."""
    if not (_at_development_stop(state) and TURN.step(state) == K.STEP_OR_START and snap['game']['round'] == 'OR'):
        return compare(state, snap)
    return [d for d in compare(state, _before_or_start(state, snap)) if not d.startswith(OR_TURN_FIELDS)]


def replay(fixture: str) -> Progress:
    return replay_extract(fixture, load_fixture_extract(fixture))


def replay_extract(name: str, extract: Extract, start: int = 0) -> Progress:
    """Replay ``extract`` from a new game, or from ``load_state`` of record ``start``."""
    records = extract['records']
    progress = Progress(name, len(records))
    ids = [p['id'] for p in extract['players']]
    state = load_state(records[start]['state'], len(ids)) if start else new_game(extract)
    pending: list[Forced] = []     # forced decisions applied since the last sync
    deferred = None                # the record whose snapshot waits for them

    def matches(record) -> bool:
        if diffs := compare_at_rest(state, record['state']):
            progress.stop, progress.detail = 'mismatch', f"record {record['index']}: " + '; '.join(diffs)
            return False
        progress.matched = record['index']
        return True

    if not matches(records[start]):
        return progress
    for record in records[start + 1:]:
        host = record['action']
        singles = [(host, record['pre']), *zip(host.get('auto_actions') or [], record.get('auto_pre') or [])]
        where = f"record {record['index']} ({host['type']})"
        if _at_development_stop(state):
            if pending and deferred is not None and not matches(deferred):  # Ruby skips forced passes silently
                return progress
            progress.stop, progress.detail = 'engine', f'{where}: the engine rests at its development stop, ' \
                                                       f'{K.Step(TURN.step(state)).name}'
            return progress
        try:
            for action, pre in singles:
                actors = {ids.index(player) for player in pre['active_players']}
                decisions = consume(pending, lambda phase: map_action(action, pre, phase, state), actors)
                if decisions is None:
                    if pending:  # Ruby skipped these forced decisions without logging them
                        pending.clear()
                        if deferred is not None and not matches(deferred):
                            return progress
                    decisions = map_action(action, pre, get_decision_phase_py(state), state)
                if error := _apply(state, decisions, pending):
                    progress.stop, progress.detail = 'mismatch', f'{where}: {error}'
                    return progress
        except UnsupportedAction as e:
            progress.stop, progress.detail = 'unsupported', f'{where}: {e}'
            return progress
        except NotImplementedError as e:
            progress.stop, progress.detail = 'engine', f'{where}: {e}'
            return progress
        except ReplayEnd:
            progress.detail = f'{where}: Ruby ended the game by hand'
            return progress
        deferred = record if pending else None
        if not pending and not matches(record):
            return progress
    if deferred is not None:  # no Ruby action is left to log what is pending
        matches(deferred)
    return progress


@pytest.mark.parametrize('fixture', FIXTURES)
def test_replay_reaches_its_floor(fixture):
    progress = replay(fixture)
    print(progress)
    assert progress.matched >= REPLAY_FLOOR[fixture], str(progress)
    if progress.stop == 'mismatch':
        warnings.warn(str(progress))


def _later_srs(extract: Extract) -> list[tuple[int, int]]:
    """``(first, last)`` records of each SR after the first: the snapshot that
    opens it, and its last action (the one before an ``end_game``)."""
    records = extract['records']
    spans = []
    for prev, record in zip(records, records[1:]):
        if record['state']['game']['round'] == 'SR' and prev['state']['game']['round'] == 'OR':
            first = record['index']
            last = next((r['index'] - (r['action']['type'] == 'end_game') for r in records[first + 1:]
                         if r['state']['game']['round'] != 'SR' or r['action']['type'] == 'end_game'), len(records) - 1)
            spans.append((first, last))
    return spans


@pytest.mark.parametrize('fixture', FIXTURES)
def test_every_later_sr_replays_from_its_first_snapshot(fixture):
    """Each SR the replay cannot reach yet, from ``load_state``: sales,
    presidency changes, and sold-out moves of real games, with Ruby's legal
    STOCK set checked at every turn."""
    extract = load_fixture_extract(fixture)
    spans = _later_srs(extract)
    assert len(spans) == {'314': 6, '962': 4}.get(fixture, 1)
    for first, last in spans:
        progress = replay_extract(fixture, extract, start=first)
        assert progress.matched >= last and progress.stop in (None, 'engine'), str(progress)


# Acceptance for 6.2: initialize_game equals Ruby's new game.


@pytest.mark.parametrize('fixture', FIXTURES)
def test_fixture_initial_state(fixture):
    extract = load_fixture_extract(fixture)
    assert compare(new_game(extract), extract['records'][0]['state']) == []


@pytest.mark.parametrize('num_players', range(D.MIN_PLAYERS, D.MAX_PLAYERS + 1))
def test_new_game_initial_state(num_players):
    player_ids = [100 + 7 * i for i in range(num_players)]
    for extract in (load_new_game_extract(num_players),
                    load_new_game_extract(num_players, seed=1889, player_ids=player_ids)):
        snap = extract['records'][0]['state']
        assert compare(new_game(extract), snap) == []
        assert compare(new_game(extract, max_players=D.MAX_PLAYERS), snap) == []


# The sync rule, on synthetic decisions.

PASS_AUCTION, PASS_STOCK = (K.DPHASE_AUCTION, K.AUCTION_PASS, 0), (K.DPHASE_STOCK, K.STOCK_PASS, 1)


def test_sync_consumes_logged_forced_decisions():
    pending = [PASS_AUCTION]
    assert consume(pending, lambda phase: [(phase, 0)], {0}) == [] and pending == []
    assert consume([], lambda phase: [(phase, 0)], {0}) is None


def test_sync_drops_forced_decisions_ruby_skipped_before_a_logged_one():
    pending = [(K.DPHASE_TOKEN, K.TOKEN_PASS, 0), PASS_STOCK]

    def stock_only(phase):
        if phase != K.DPHASE_STOCK:
            raise UnsupportedAction('not here')
        return [(phase, 0)]

    assert consume(pending, stock_only, {1}) == [] and pending == []


def test_sync_returns_the_rest_of_a_partly_forced_action():
    pending = [(K.DPHASE_TRACK, K.TRACK_BUY_PRIVATE + 2, 3)]
    decisions = [(K.DPHASE_TRACK, K.TRACK_BUY_PRIVATE + 2), (K.DPHASE_PRIVATE_PRICE, K.PRIVATE_PRICE_EXACT + 80)]
    assert consume(pending, lambda phase: decisions, {3}) == decisions[1:] and pending == []


def test_sync_reports_no_match():
    pending = [PASS_AUCTION]
    assert consume(pending, lambda phase: [(K.DPHASE_AUCTION, K.AUCTION_BUY)], {0}) is None
    assert consume(pending, lambda phase: [], {0}) is None  # no decision, e.g. a message
    assert pending == [PASS_AUCTION]


def test_sync_matches_only_the_forced_player():
    """After a purchase in the first SR, the driver ends the buyer's turn with a
    forced "done" that Ruby does not log; the next player's pass is their own."""
    pending = [(K.DPHASE_STOCK, K.STOCK_PASS, 0)]
    assert consume(pending, lambda phase: [(phase, 0)], {1}) is None and pending == [(K.DPHASE_STOCK, 0, 0)]
