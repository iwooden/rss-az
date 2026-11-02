"""Differential replay of the 1889 fixtures against Ruby (1889/engine-design.md
§10.2-3), and the initial states of fixtures and fresh games (6.2's acceptance).

Replay walks a fixture's records, maps each Ruby action to engine decisions,
and applies them through ``DRIVER`` with history. The driver auto-applies
forced decisions, which Ruby may log as actions or skip silently. The sync
rule: a Ruby action that matches a forced decision applied since the last sync,
by the same player, is consumed instead of applied; one that matches none
means Ruby skipped the pending forced decisions, so they are dropped. States
are compared whenever no forced decision is pending.

Three equivalences need the harness. Track after the lay (engine-design
§10.2): Ruby keeps Track open after a corp's lay while its president owns MF,
for the port tile, and the engine moves on to TOKEN. That snapshot is
compared with the engine at TOKEN (``snapshot.track_after_lay``), and Ruby's
pass that then closes Track maps to no decision: the forced decisions the
engine applied since the lay are what Ruby does on that pass and after it,
and the sync rule takes them from there (Ruby logs a forced train purchase).
Route (§8): Ruby waits at its Route step for the president's routes, and
the engine's ROUTE step runs by itself, on to DIVIDEND. A Route snapshot is
compared with the engine's row before its last ROUTE step, and
``run_routes`` maps to no decision or to ``override_route_revenue``.
Discards (§10.2): Ruby accepts a discard from any crowded corp, and the
engine takes them in corp order, so the mapper holds a discard that comes
early (``HELD_DISCARD``) until the engine reaches its corp; the records in
between are not compared.

Replay stops at the first unsupported action, engine NotImplementedError, or
mismatch, and at the driver's development stop (an automated step from a later
sub-step, where the engine rests). When it stops at an action while forced
decisions are pending, Ruby may be waiting at one of them (BuyCompany, whose
purchases land with 6.8, when the engine forces its "done"), so the record
before is compared with the engine where Ruby rests (``_ruby_rests``); so is
a record whose forced decisions are pending when the replay runs out.
``REPLAY_FLOOR`` holds the last record index each fixture must match; each
sub-step raises it, so progress ratchets. ``replay_extract(..., start=i)``
starts from ``load_state`` of record i, so later rounds replay before the
engine can reach them.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState, get_global_fields, get_layout

from tests.g1889.replay.action_map import (
    HELD_DISCARD, ROUTE_OVERRIDE, Decision, ReplayEnd, UnsupportedAction, map_action,
)
from tests.g1889.replay.extracts import Extract, FIXTURES, load_fixture_extract, load_new_game_extract
from tests.g1889.replay.snapshot import compare, load_state, track_after_lay

# Sub-step 6.7: the hs_* fixtures to their end_game; 314 to the record before
# its first private purchase (81, buy_company), 962 to the record before MF's
# port tile (33, 6.8), which comes before its first private purchase.
REPLAY_FLOOR = {'314': 80, '962': 32, 'hs_jmmljkbw_17502': 49, 'hs_lhglbbiz_17502': 46}
END_TO_END = ('hs_jmmljkbw_17502', 'hs_lhglbbiz_17502')

# A forced decision the driver applied: (decision phase, id, the player it was
# for, and the row before it); the sync tests give only the first three.
Forced = tuple[Any, ...]
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
    for j, (phase, *_) in enumerate(pending):
        try:
            decisions = decisions_in(phase)
        except UnsupportedAction:
            continue
        m = min(len(decisions), len(pending) - j)
        if m and all(forced[:2] == decisions[k] and forced[2] in actors
                     for k, forced in enumerate(pending[j:j + m])):
            del pending[:j + m]
            return decisions[m:]
    return None


def _apply(state: GameState, decisions: list[Decision], pending: list[Forced], log: list,
           held: list[tuple[int, int]]) -> str | None:
    """Apply decisions through the driver, queueing the forced ones it adds and
    adding its history to ``log``; an error message on failure. A
    ``ROUTE_OVERRIDE`` pays a human's revenue (``override_route_revenue``),
    which records no decision of its own, and a ``HELD_DISCARD`` waits in
    ``held`` for its corp's turn to discard (``_release``)."""
    for phase, action_id in decisions:
        if phase == HELD_DISCARD:
            held.append(divmod(action_id, D.NUM_TRAIN_TYPES))
            continue
        if pending:  # forced inside one Ruby action: it must be the decision Ruby made
            if pending[0][:2] != (phase, action_id):
                return f'engine forced {pending[0][:2]} where Ruby chose {(phase, action_id)}'
            pending.pop(0)
            continue
        history: list = []
        if phase == ROUTE_OVERRIDE:
            if DRIVER.override_route_revenue(state, action_id, history) == STATUS_INVALID:
                return f'the route override to {action_id} is illegal at {K.Step(TURN.step(state)).name}'
            forced = history
        else:
            engine_phase = get_decision_phase_py(state)
            if engine_phase != phase:
                return f'engine is at {_phase_name(engine_phase)}, Ruby acts in {_phase_name(phase)}'
            if DRIVER.apply_action(state, action_id, history) == STATUS_INVALID:
                return f'{_phase_name(phase)} id {action_id} is illegal in the engine'
            forced = history[1:]
        pending.extend((p, a, int(row[ACTIVE_PLAYER_SLOT]), row) for row, p, a in forced if p >= 0)
        log += history
    return _release(state, pending, log, held)


def _discarding(state: GameState, row=None) -> int:
    """The corp the engine's DISCARD is for, in ``state`` or a history row; -1 elsewhere."""
    if row is not None:
        state = GameState.from_array(row, TURN.num_players(state), state.max_players)
    return TRAINS.first_crowded(state) if TURN.step(state) == K.STEP_DISCARD else -1


def _release(state: GameState, pending: list[Forced], log: list, held: list[tuple[int, int]]) -> str | None:
    """Apply each held discard once the engine reaches its corp: consume the
    discard the driver forced for it, or make it at the engine's DISCARD."""
    while held:
        corp, train = held[0]
        hit = next((i for i, (phase, action_id, _, row) in enumerate(pending)
                    if (phase, action_id) == (K.DPHASE_DISCARD, train) and _discarding(state, row) == corp), None)
        if hit is not None:
            del pending[hit]
        elif _discarding(state) == corp:
            held.pop(0)
            if error := _apply(state, [(K.DPHASE_DISCARD, train)], pending, log, held):
                return error
            continue
        else:
            return None
        held.pop(0)
    return None


def _ruby_rests(state: GameState, pending: list[Forced]) -> list[GameState]:
    """Where Ruby may rest after the record whose forced decisions are
    pending, when no next action tells: past them all, as the engine is, or
    waiting at one of them (``replay_extract``'s ``rests``)."""
    return [state] + [GameState.from_array(row, TURN.num_players(state), state.max_players) for *_, row in pending]


def _at_development_stop(state: GameState) -> bool:
    """The driver rests at an automated step only at its development stop (6.9 removes it)."""
    return get_decision_phase_py(state) < 0 and TURN.round(state) != K.ROUND_GAME_OVER


def _closes_track(record) -> bool:
    """Ruby's pass that closes a Track step kept open after the lay."""
    host = record['action']
    return host['type'] == 'pass' and record['pre']['step'] == 'Track' and not host.get('auto_actions')


def replay(fixture: str) -> Progress:
    return replay_extract(fixture, load_fixture_extract(fixture))


def replay_extract(name: str, extract: Extract, start: int = 0, end: int | None = None) -> Progress:
    """Replay ``extract`` from a new game, or from ``load_state`` of record
    ``start``, through record ``end`` (the last by default)."""
    records = extract['records']
    progress = Progress(name, len(records))
    ids = [p['id'] for p in extract['players']]
    state = load_state(records[start]['state'], len(ids)) if start else new_game(extract)
    pending: list[Forced] = []     # forced decisions applied since the last sync
    deferred = None                # the record whose snapshot waits for them
    held: list[tuple[int, int]] = []  # (corp, train type) of discards Ruby took early
    track_open = False             # Ruby kept Track open after the lay; the engine is past it
    route_row = None               # the engine's row at its last ROUTE step, not yet compared

    def matches(record, at: GameState | None = None) -> bool:
        nonlocal route_row
        if at is None:
            at = state
            if record['state']['game']['step'] == 'Route' and route_row is not None:  # EQUIVALENT: before ROUTE
                at, route_row = GameState.from_array(route_row, len(ids), state.max_players), None
        if diffs := compare(at, record['state']):
            progress.stop, progress.detail = 'mismatch', f"record {record['index']}: " + '; '.join(diffs)
            return False
        progress.matched = record['index']
        return True

    def rests(record) -> bool:
        """Compare a record whose forced decisions are pending, with no next
        Ruby action to tell whether Ruby took them: it matches if Ruby rests
        at the engine's state or before one of them (BuyCompany, whose
        purchases land with 6.8, when the engine forced its "done")."""
        at = next((at for at in _ruby_rests(state, pending) if not compare(at, record['state'])), state)
        return matches(record, at)

    def stop(kind: str, detail: str) -> Progress:
        """Stop at an action the engine cannot take yet, first comparing the
        record whose forced decisions are pending where Ruby rests."""
        if pending and deferred is not None and not held and not rests(deferred):
            return progress
        progress.stop, progress.detail = kind, detail
        return progress

    if not matches(records[start]):
        return progress
    for record in records[start + 1:None if end is None else end + 1]:
        host = record['action']
        singles = [(host, record['pre']), *zip(host.get('auto_actions') or [], record.get('auto_pre') or [])]
        where = f"record {record['index']} ({host['type']})"
        log: list = []
        if track_open and _closes_track(record):  # EQUIVALENT: Track after the lay; no decision of its own
            singles = []
        track_open = False
        if _at_development_stop(state):  # loaded at an automated step a later sub-step owns
            progress.stop, progress.detail = 'engine', f'{where}: the engine rests at its development stop, ' \
                                                       f'{K.Step(TURN.step(state)).name}'
            return progress
        try:
            for action, pre in singles:
                actors = {ids.index(player) for player in pre['active_players']}
                decisions = consume(pending, lambda phase: map_action(action, pre, phase, state), actors)
                if decisions is None:
                    decisions = map_action(action, pre, get_decision_phase_py(state), state)
                    if pending:  # Ruby skipped these forced decisions without logging them
                        pending.clear()
                        if deferred is not None and not held and not matches(deferred):
                            return progress
                if error := _apply(state, decisions, pending, log, held):
                    progress.stop, progress.detail = 'mismatch', f'{where}: {error}'
                    return progress
        except UnsupportedAction as e:
            return stop('unsupported', f'{where}: {e}')
        except NotImplementedError as e:
            return stop('engine', f'{where}: {e}')
        except ReplayEnd:
            progress.detail = f'{where}: Ruby ended the game by hand'
            return progress
        if _at_development_stop(state):  # the engine cannot finish Ruby's action, so its record is not compared
            progress.stop, progress.detail = 'engine', f'{where}: the engine rests at its development stop, ' \
                                                       f'{K.Step(TURN.step(state)).name}'
            return progress
        route_row = next((row for row, phase, step in log if phase < 0 and step == K.STEP_ROUTE), route_row)
        if track_after_lay(record['state']):  # compare at TOKEN: the row before a forced TOKEN, or the rest
            at_token = next((row for row, phase, _ in log if phase == K.DPHASE_TOKEN), None)
            at = state if at_token is None else GameState.from_array(at_token, len(ids), state.max_players)
            if not matches(record, at):
                return progress
            track_open, deferred = True, None
            continue
        if held:  # Ruby's state between discards in its order, which the engine never reaches
            deferred = None
            continue
        deferred = record if pending else None
        if not pending and not matches(record):
            return progress
    if held:
        progress.stop, progress.detail = 'mismatch', f'discards never reached in the engine: {held}'
    elif deferred is not None:  # no Ruby action is left to log what is pending
        rests(deferred)
    return progress


@pytest.mark.parametrize('fixture', FIXTURES)
def test_replay_reaches_its_floor(fixture):
    progress = replay(fixture)
    print(progress)
    assert progress.matched >= REPLAY_FLOOR[fixture], str(progress)
    if fixture in END_TO_END:  # every record but the closing end_game
        assert progress.stop is None and progress.matched == progress.records - 2, str(progress)
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
        progress = replay_extract(fixture, extract, start=first, end=last)
        assert progress.matched == last and progress.stop is None, str(progress)


def _or_turns(extract: Extract) -> list[tuple[int, int, bool]]:
    """``(first, last, opens_or)`` for each corp's turn in an OR: the snapshot
    where it opens, the action that hands the round on (the next turn, the next
    round, or the game's end), and whether the turn opens its OR."""
    records = extract['records']

    def turn_of(record):
        g = record['state']['game']
        if g['round'] != 'OR' or g['finished']:
            return None
        return g['turn'], g['round_num'], record['state']['operating']['operator']

    turns = []
    for prev, record in zip(records, records[1:]):
        if (key := turn_of(record)) is not None and key != (before := turn_of(prev)):
            last = next((r['index'] for r in records[record['index'] + 1:] if turn_of(r) != key), len(records) - 1)
            turns.append((record['index'], last, before is None or before[:2] != key[:2]))
    return turns


def _lays_and_tokens(records, first: int, last: int) -> list[int]:
    """The records of a corp's lays and tokens."""
    return [i for i in range(first + 1, last + 1) if records[i]['action']['type'] == 'place_token' or (
        records[i]['action']['type'] == 'lay_tile' and records[i]['action']['entity_type'] == 'corporation')]


# Per fixture: ORs and turns; the turns that stop at an action of a later
# sub-step before their last lay or token (962 record 33: MF's port tile,
# ahead of TR's lay and token); the replayed routes that pay a human's revenue
# (override_route_revenue); the turns' stops by kind: emergency money (314's
# must-buys short of cash), a private purchase or MF's port tile (6.8), and
# the game's end after the bank breaks (962's last turn); and per kind of
# turn action, how many the turns hold and how many of them replay. The
# buys left are 314's emergency purchases (350, 444, 453) and 962's after the
# port tile (36-38) and a private purchase (274).
OR_TURN_COVERAGE = {
    '314': dict(ors=10, turns=43, cut_short=0, overrides=5, stops={'engine': 4, 'unsupported': 3},
                lays_and_tokens=(46, 46), run_routes=(32, 32), dividend=(32, 32), buy_train=(30, 27),
                discard_train=(2, 2)),
    '962': dict(ors=10, turns=56, cut_short=1, overrides=1, stops={'engine': 1, 'unsupported': 4},
                lays_and_tokens=(59, 57), run_routes=(48, 48), dividend=(48, 47), buy_train=(21, 17),
                discard_train=(0, 0)),
    **{fixture: dict(ors=1, turns=1, cut_short=0, overrides=0, stops={}, lays_and_tokens=(0, 0),
                     run_routes=(0, 0), dividend=(0, 0), buy_train=(0, 0), discard_train=(0, 0))
       for fixture in END_TO_END},
}
TURN_ACTIONS = ('run_routes', 'dividend', 'buy_train', 'discard_train')


@pytest.mark.parametrize('fixture', FIXTURES)
def test_every_or_turn_replays_from_its_first_snapshot(fixture):
    """Each corp's turn in each OR, from ``load_state`` of the snapshot where
    it opens (each OR's first turn opens the OR): the lays, tokens, passes,
    routes, dividends, train purchases, and discards of real games, with
    Ruby's BUY_TRAIN choices checked at every BuyTrain snapshot. A turn
    replays until the engine rests where a later sub-step takes over
    (emergency money, the game's end) or an action of a later sub-step; it
    never mismatches."""
    extract = load_fixture_extract(fixture)
    records = extract['records']
    turns = _or_turns(extract)
    counts = {kind: [0, 0] for kind in ('lays_and_tokens', *TURN_ACTIONS)}
    cut_short = overrides = 0
    stops: dict[str, int] = {}
    for first, last, _ in turns:
        progress = replay_extract(fixture, extract, start=first, end=last)
        assert progress.stop != 'mismatch', str(progress)
        if progress.stop:
            stops[progress.stop] = stops.get(progress.stop, 0) + 1
        found = {'lays_and_tokens': _lays_and_tokens(records, first, last)}
        found.update({kind: [i for i in range(first + 1, last + 1) if records[i]['action']['type'] == kind]
                      for kind in TURN_ACTIONS})
        if found['lays_and_tokens'] and progress.matched < found['lays_and_tokens'][-1]:
            assert progress.stop == 'unsupported', str(progress)
            cut_short += 1
        for kind, indices in found.items():
            counts[kind][0] += len(indices)
            counts[kind][1] += sum(i <= progress.matched for i in indices)
        overrides += sum(records[i]['pre']['revenue'] != records[i - 1]['state']['operating']['auto_router']['revenue']
                         for i in found['run_routes'] if i <= progress.matched)
    coverage = dict(ors=sum(opens for *_, opens in turns), turns=len(turns), cut_short=cut_short,
                    overrides=overrides, stops=stops, **{kind: tuple(c) for kind, c in counts.items()})
    assert coverage == OR_TURN_COVERAGE[fixture]


# Per fixture and kind: the actions, and how many replay from the snapshot
# before them. The buys left need emergency money (314 records 350, 444, 453);
# the dividend left ends the game after the bank breaks (962 record 446).
ONE_ACTION_COVERAGE = {
    '314': {'dividend': (32, 32), 'buy_train': (30, 27), 'discard_train': (2, 2)},
    '962': {'dividend': (48, 47), 'buy_train': (21, 21), 'discard_train': (0, 0)},
}


@pytest.mark.parametrize('fixture', ('314', '962'))
def test_every_dividend_train_purchase_and_discard_replays_from_the_snapshot_before_it(fixture):
    """Each one alone, from ``load_state``, including those the turn replays
    stop short of (after MF's port tile or a private purchase): its payments,
    price move, phase change, rusting, and privates' closing (962 record 274:
    KU's first 5 closes the UTF it owns). A run of discards replays whole,
    since Ruby takes them in any order (314 records 339-340)."""
    extract = load_fixture_extract(fixture)
    records = extract['records']
    coverage = {kind: [0, 0] for kind in ONE_ACTION_COVERAGE[fixture]}
    for record in records[1:]:
        kind, i = record['action']['type'], record['index']
        if kind not in coverage or kind == 'discard_train' and records[i - 1]['action']['type'] == kind:
            continue
        end = next(j for j in range(i, len(records)) if records[j + 1]['action']['type'] != kind) \
            if kind == 'discard_train' else i
        progress = replay_extract(fixture, extract, start=i - 1, end=end)
        assert progress.stop != 'mismatch', str(progress)
        coverage[kind][0] += end - i + 1
        coverage[kind][1] += (end - i + 1) * (progress.matched == end)
    assert {kind: tuple(c) for kind, c in coverage.items()} == ONE_ACTION_COVERAGE[fixture]


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
