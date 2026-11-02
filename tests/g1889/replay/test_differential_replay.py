"""Differential replay of the 1889 fixtures against Ruby (1889/engine-design.md
§10.2-3), and the initial states of fixtures and fresh games (6.2's acceptance).

Replay walks a fixture's records, maps each Ruby action to engine decisions,
and applies them through ``DRIVER`` with history. The driver auto-applies
forced decisions, which Ruby may log as actions or skip silently. The sync
rule: a Ruby action that matches a forced decision applied since the last sync
is consumed instead of applied; one that matches none means Ruby skipped the
pending forced decisions, so they are dropped. States are compared whenever no
forced decision is pending.

Replay stops at the first unsupported action, engine NotImplementedError, or
mismatch. ``REPLAY_FLOOR`` holds the last record index each fixture must
match; each sub-step raises it, so progress ratchets.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.state import GameState

from tests.g1889.replay.action_map import Decision, ReplayEnd, UnsupportedAction, map_action
from tests.g1889.replay.extracts import Extract, FIXTURES, load_fixture_extract, load_new_game_extract
from tests.g1889.replay.snapshot import compare

# Sub-step 6.2: only the initial snapshot.
REPLAY_FLOOR = {fixture: 0 for fixture in FIXTURES}


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


def consume(pending: list[Decision], decisions_in: Callable[[int], list[Decision]]) -> list[Decision] | None:
    """The sync rule for one Ruby action. ``decisions_in(phase)`` maps the action
    as taken in ``phase``. If its decisions continue the pending forced ones from
    some entry on, drop the pending decisions through the match and return the
    decisions still to apply; return None when nothing matches."""
    for j, (phase, _) in enumerate(pending):
        try:
            decisions = decisions_in(phase)
        except UnsupportedAction:
            continue
        m = min(len(decisions), len(pending) - j)
        if m and decisions[:m] == pending[j:j + m]:
            del pending[:j + m]
            return decisions[m:]
    return None


def _apply(state: GameState, decisions: list[Decision], pending: list[Decision]) -> str | None:
    """Apply decisions through the driver, queueing the forced ones it adds; an error message on failure."""
    for phase, action_id in decisions:
        if pending:  # forced inside one Ruby action: it must be the decision Ruby made
            if pending[0] != (phase, action_id):
                return f'engine forced {pending[0]} where Ruby chose {(phase, action_id)}'
            pending.pop(0)
            continue
        engine_phase = get_decision_phase_py(state)
        if engine_phase != phase:
            return f'engine is at {_phase_name(engine_phase)}, Ruby acts in {_phase_name(phase)}'
        history: list = []
        if DRIVER.apply_action(state, action_id, history) == STATUS_INVALID:
            return f'{_phase_name(phase)} id {action_id} is illegal in the engine'
        pending.extend((p, a) for _, p, a in history[1:] if p >= 0)
    return None


def replay(fixture: str) -> Progress:
    extract = load_fixture_extract(fixture)
    records = extract['records']
    progress = Progress(fixture, len(records))
    state = new_game(extract)
    pending: list[Decision] = []   # forced decisions applied since the last sync
    deferred = None                # the record whose snapshot waits for them

    def matches(record) -> bool:
        if diffs := compare(state, record['state']):
            progress.stop, progress.detail = 'mismatch', f"record {record['index']}: " + '; '.join(diffs)
            return False
        progress.matched = record['index']
        return True

    if not matches(records[0]):
        return progress
    for record in records[1:]:
        host = record['action']
        singles = [(host, record['pre']), *zip(host.get('auto_actions') or [], record.get('auto_pre') or [])]
        where = f"record {record['index']} ({host['type']})"
        try:
            for action, pre in singles:
                decisions = consume(pending, lambda phase: map_action(action, pre, phase, state))
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

PASS_AUCTION, PASS_STOCK = (K.DPHASE_AUCTION, K.AUCTION_PASS), (K.DPHASE_STOCK, K.STOCK_PASS)


def test_sync_consumes_logged_forced_decisions():
    pending = [PASS_AUCTION]
    assert consume(pending, lambda phase: [(phase, 0)]) == [] and pending == []
    assert consume([], lambda phase: [(phase, 0)]) is None


def test_sync_drops_forced_decisions_ruby_skipped_before_a_logged_one():
    pending = [(K.DPHASE_TOKEN, K.TOKEN_PASS), PASS_STOCK]

    def stock_only(phase):
        if phase != K.DPHASE_STOCK:
            raise UnsupportedAction('not here')
        return [(phase, 0)]

    assert consume(pending, stock_only) == [] and pending == []


def test_sync_returns_the_rest_of_a_partly_forced_action():
    pending = [(K.DPHASE_TRACK, K.TRACK_BUY_PRIVATE + 2)]
    decisions = [(K.DPHASE_TRACK, K.TRACK_BUY_PRIVATE + 2), (K.DPHASE_PRIVATE_PRICE, K.PRIVATE_PRICE_EXACT + 80)]
    assert consume(pending, lambda phase: decisions) == decisions[1:] and pending == []


def test_sync_reports_no_match():
    pending = [PASS_AUCTION]
    assert consume(pending, lambda phase: [(K.DPHASE_AUCTION, K.AUCTION_BUY)]) is None
    assert consume(pending, lambda phase: []) is None  # no decision, e.g. a message
    assert pending == [PASS_AUCTION]
