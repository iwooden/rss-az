"""Differential replay of the 1889 fixtures against Ruby (1889/engine-design.md
§10.2-3), and the initial states of fixtures and fresh games (6.2's acceptance).

Replay walks a fixture's records, maps each Ruby action to engine decisions,
and applies them through ``DRIVER`` with history. The driver auto-applies
forced decisions, which Ruby may log as actions or skip silently. The sync
rule: a Ruby action that matches a forced decision applied since the last sync,
by the same player, is consumed instead of applied; one that matches none
means Ruby skipped the pending forced decisions, so they are dropped. States
are compared whenever no forced decision is pending. Every decision a Ruby
action maps to, sub-phases such as PRIVATE_PRICE and PORT included, must be
the engine's active player's, one of Ruby's active players before the action;
``compare`` checks the active player at every snapshot too.

Six equivalences need the harness. Track after the lay (engine-design
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
between are not compared. SR turns Ruby skips (§10.2): the engine offers DR
and MF on every SR turn of their owner, so it stops at a turn whose only
other choice is the pass, which Ruby skips or ends without logging
anything; after each Ruby action the harness applies that pass
(``skipped_sr_turn``), and queues the forced decisions after it as usual.
BUY_PRIVATE_PRE (§5.5): Ruby has no such step, so the engine at PRE compares
with Ruby's Track, and where Ruby then does anything but PRE's own moves the
harness applies PRE's "done" first (``_leave_pre``). A human's private move at
another of Ruby's stops takes the override, which returns to the decision it
interrupted; if the driver has already forced that decision for lack of a
choice, the harness backs up to it first (``_forced_at_ruby_step``).

Replay stops at the first unsupported action or mismatch; an engine error
propagates. When it stops at an action while forced decisions are pending,
Ruby may be waiting at one of them (a Track step whose only choice is the
pass, which Ruby asks), so the record before is compared with the engine
where Ruby rests (``_ruby_rests``); so is a record whose forced decisions
are pending when the replay runs out.
Since 6.8b every fixture replays end to end, 314 to its bankruptcy and 962
to the game's end after the bank breaks, and the engine's final scores equal
Ruby's ``result`` (the ``hs_*`` fixtures' players end the game by hand,
before which their values equal Ruby's). ``replay_extract(..., start=i)``
starts from ``load_state`` of record i, so each round and turn also replays
alone.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py, is_terminal_py
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.phases.operating import setup_operating_round_py
from g1889.state import GameState, get_global_fields, get_layout

from tests.g1889.replay.action_map import (
    HELD_DISCARD, OVERRIDE_STEP_PHASE, PRIVATE_OVERRIDE, ROUTE_OVERRIDE, TOPMOST_BID_OVERRIDE, Decision, ReplayEnd,
    UnsupportedAction, map_action, skipped_sr_turn,
)
from tests.g1889.phases.conftest import end_auction, enter_stock_round, hold
from tests.g1889.phases.conftest import new_game as fresh_game
from tests.g1889.replay.extracts import Extract, FIXTURES, load_fixture_extract, load_new_game_extract
from tests.g1889.replay.snapshot import (
    CORP, END_REASONS, RUBY_STEPS, compare, engine_result, load_state, track_after_lay,
)

# How each fixture's game ends in Ruby: 314 when TR's president goes bankrupt
# (record 457), 962 after the OR set in which the bank broke (record 446), and
# the hs_* fixtures by hand (end_game), which the engine does not model.
ENDINGS = {'314': 'bankrupt', '962': 'bank', 'hs_jmmljkbw_17502': 'manually_ended',
           'hs_lhglbbiz_17502': 'manually_ended'}

# The privates' moves in a replay from the start, by path: 314 buys only at
# Ruby's blocking BuyCompany. 962 lays MF's port tile before its first lay
# (record 33), lays ER's bonus, and buys 4 at BuyCompany and 3 at BuyTrain.
PRIVATE_PATHS = {
    '314': {'BUY_PRIVATE_POST': 6},
    '962': {'BUY_PRIVATE_PRE': 1, 'BUY_PRIVATE_POST': 4, 'ER_BONUS': 1, 'override at BuyTrain': 3},
    'hs_jmmljkbw_17502': {}, 'hs_lhglbbiz_17502': {},
}

# A forced decision the driver applied: (decision phase, id, the player it was
# for, and the row before it); the sync tests give only the first three.
Forced = tuple[Any, ...]
# A discard Ruby took early: (corp, DISCARD's id, the players who may make it).
Held = tuple[int, int, set[int]]
ACTIVE_PLAYER_SLOT = get_layout(D.MIN_PLAYERS).global_offset + get_global_fields().active_player


@dataclass
class Progress:
    fixture: str
    records: int
    matched: int = -1              # the last record whose snapshot compared equal
    stop: str | None = None        # 'unsupported' or 'mismatch'; None when replay ran out
    detail: str = ''
    state: GameState | None = None  # the engine's state, where replay stopped
    private_moves: Counter = field(default_factory=Counter)  # the privates' moves Ruby logged, by the path they took

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


def _apply(state: GameState, decisions: list[Decision], actors: set[int], pending: list[Forced], log: list,
           held: list[Held]) -> str | None:
    """Apply decisions by one of ``actors`` through the driver, queueing the
    forced ones it adds and adding its history to ``log``; an error message on
    failure, or when the engine's active player is not one of ``actors``. A
    ``ROUTE_OVERRIDE`` pays a human's revenue (``override_route_revenue``) and
    a ``PRIVATE_OVERRIDE`` starts a human's private move
    (``override_private_slot``), and a ``TOPMOST_BID_OVERRIDE`` a bid on private
    0 (``override_topmost_bid``), none recording a decision of its own, and a
    ``HELD_DISCARD`` waits in ``held`` for its corp's turn to discard
    (``_release``). Then the engine passes for each SR turn Ruby skips or ends
    (``_pass_skipped_sr_turns``)."""
    for phase, action_id in decisions:
        if phase == HELD_DISCARD:  # by its corp's president, whose discard Ruby takes out of its order
            corp, slot = divmod(action_id, D.NUM_TRAIN_TYPES)
            held.append((corp, slot, {CORPS[corp].president(state)}))
            continue
        if not pending and phase != K.DPHASE_BUY_PRIVATE_PRE:
            _leave_pre(state, pending, log)
        player = pending[0][2] if pending else TURN.active_player(state)
        if player not in actors:
            return f'{_phase_name(phase)} is player {player}\'s in the engine, not one of Ruby\'s {sorted(actors)}'
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
        elif phase == PRIVATE_OVERRIDE:
            if DRIVER.override_private_slot(state, action_id, history) == STATUS_INVALID:
                return f'the private override of slot {action_id} is illegal at {K.Step(TURN.step(state)).name}'
            forced = history
        elif phase == TOPMOST_BID_OVERRIDE:
            if DRIVER.override_topmost_bid(state, history) == STATUS_INVALID:
                return f'the bid on private 0 is illegal at {K.Step(TURN.step(state)).name}'
            forced = history
        else:
            engine_phase = get_decision_phase_py(state)
            if engine_phase != phase:
                return f'engine is at {_phase_name(engine_phase)}, Ruby acts in {_phase_name(phase)}'
            if DRIVER.apply_action(state, action_id, history) == STATUS_INVALID:
                return f'{_phase_name(phase)} id {action_id} is illegal in the engine'
            forced = history[1:]
        _queue(pending, forced)
        log += history
    _pass_skipped_sr_turns(state, pending, log)
    return _release(state, pending, log, held)


def _queue(pending: list[Forced], forced: list) -> None:
    """Queue the decisions in driver history ``forced`` for the sync rule."""
    pending.extend((p, a, int(row[ACTIVE_PLAYER_SLOT]), row) for row, p, a in forced if p >= 0)


def _pass_skipped_sr_turns(state: GameState, pending: list[Forced], log: list) -> None:
    """EQUIVALENT: pass where the engine rests at an SR turn whose only other
    choices are DR's exchange and MF's port tile, which Ruby skips or ends
    without logging anything (``skipped_sr_turn``). The pass is not Ruby's
    decision, so the sync rule never takes a Ruby action for it; the forced
    decisions after it are queued as usual."""
    while decisions := skipped_sr_turn(state):
        history: list = []
        assert DRIVER.apply_action(state, decisions[0][1], history) != STATUS_INVALID
        _queue(pending, history[1:])
        log += history


def _leave_pre(state: GameState, pending: list[Forced], log: list) -> None:
    """EQUIVALENT: PRE's "done" where the engine rests at PRE and Ruby, at its
    Track step, does something else. Ruby logs no such decision, so the sync
    rule never takes a Ruby action for it; the forced decisions after it are
    queued as usual, and the action that follows may be one of them (a Track
    pass with nothing to lay)."""
    if get_decision_phase_py(state) == K.DPHASE_BUY_PRIVATE_PRE:
        history: list = []
        assert DRIVER.apply_action(state, K.BUY_PRIVATE_PRE_DONE, history) != STATUS_INVALID
        _queue(pending, history[1:])
        log += history


def _discarding(state: GameState, row=None) -> int:
    """The corp the engine's DISCARD is for, in ``state`` or a history row; -1 elsewhere."""
    if row is not None:
        state = GameState.from_array(row, TURN.num_players(state), state.max_players)
    return TRAINS.first_crowded(state) if TURN.step(state) == K.STEP_DISCARD else -1


def _release(state: GameState, pending: list[Forced], log: list, held: list[Held]) -> str | None:
    """Apply each held discard once the engine reaches its corp: consume the
    discard the driver forced for it, or make it at the engine's DISCARD."""
    while held:
        corp, slot, actors = held[0]
        hit = next((i for i, (phase, action_id, player, row) in enumerate(pending)
                    if (phase, action_id) == (K.DPHASE_DISCARD, slot) and _discarding(state, row) == corp
                    and player in actors), None)
        if hit is not None:
            del pending[hit]
        elif _discarding(state) == corp:
            held.pop(0)
            if error := _apply(state, [(K.DPHASE_DISCARD, slot)], actors, pending, log, held):
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


def _closes_track(record) -> bool:
    """Ruby's pass that closes a Track step kept open after the lay."""
    host = record['action']
    return host['type'] == 'pass' and record['pre']['step'] == 'Track' and not host.get('auto_actions')


def _is_private_move(action) -> bool:
    """A corp's purchase, DR's exchange, or a private's tile."""
    return action['type'] == 'buy_company' or (
        action['type'] in ('lay_tile', 'buy_shares') and action['entity_type'] == 'company')


def _forced_at_ruby_step(pending: list[Forced], action, pre, track_open: bool) -> int | None:
    """The pending forced decision a human's private move interrupts, or None:
    the driver forced it for lack of a choice in the engine's action space, as
    it can a must-buy's one source, where Ruby still waits and joins the
    private's move to its step. With Track kept open after the lay it is the
    first, as the engine has left TRACK."""
    if not _is_private_move(action) or pre.get('round') != 'OR' or pre['step'] not in OVERRIDE_STEP_PHASE:
        return None
    if track_open:
        return 0 if pending else None
    phase = OVERRIDE_STEP_PHASE[pre['step']]
    return next((j for j, forced in enumerate(pending) if forced[0] == phase), None)


def _private_path(phase: int, pre) -> str:
    """How a private's move replays: its slot's phase, or the override at Ruby's step."""
    return f"override at {pre['step']}" if phase == PRIVATE_OVERRIDE else _phase_name(phase).removeprefix('DPHASE_')


def replay(fixture: str) -> Progress:
    return replay_extract(fixture, load_fixture_extract(fixture))


def replay_extract(name: str, extract: Extract, start: int = 0, end: int | None = None) -> Progress:
    """Replay ``extract`` from a new game, or from ``load_state`` of record
    ``start``, through record ``end`` (the last by default)."""
    records = extract['records']
    progress = Progress(name, len(records))
    ids = [p['id'] for p in extract['players']]
    state = load_state(records[start]['state'], len(ids)) if start else new_game(extract)
    progress.state = state
    pending: list[Forced] = []     # forced decisions applied since the last sync
    deferred = None                # the record whose snapshot waits for them
    held: list[Held] = []          # discards Ruby took early
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
        at the engine's state or before one of them (a Track step whose
        only choice is the pass, which Ruby asks)."""
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
        was_open, track_open = track_open, False
        try:
            for action, pre in singles:
                actors = {ids.index(player) for player in pre['active_players']}
                decisions = consume(pending, lambda phase: map_action(action, pre, phase, state), actors)
                if decisions is None:
                    forced = _forced_at_ruby_step(pending, action, pre, was_open)
                    if forced is not None:  # EQUIVALENT: Ruby still waits there; back up to it
                        state._array[:] = pending[forced][3]
                        pending.clear()
                        if deferred is not None and not held and not matches(deferred, state):
                            return progress
                        deferred = None
                        progress.private_moves['backed up'] += 1
                    decisions = map_action(action, pre, get_decision_phase_py(state), state)
                    if pending:  # Ruby skipped these forced decisions without logging them
                        pending.clear()
                        if deferred is not None and not held and not matches(deferred):
                            return progress
                        deferred = None
                    if decisions and _is_private_move(action):
                        progress.private_moves[_private_path(decisions[0][0], pre)] += 1
                if error := _apply(state, decisions, actors, pending, log, held):
                    progress.stop, progress.detail = 'mismatch', f'{where}: {error}'
                    return progress
        except UnsupportedAction as e:
            return stop('unsupported', f'{where}: {e}')
        except ReplayEnd:
            progress.detail = f'{where}: Ruby ended the game by hand'
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
def test_replay_runs_end_to_end(fixture):
    """Every record replays and compares equal, but the hs_* fixtures'
    closing end_game; ``compare`` checks every player's value throughout and
    Ruby's ``result`` once the game is over. The engine's final scores equal
    that result: after 314's bankruptcy, after 962's last OR, and where the
    hs_* fixtures' players end the game by hand."""
    extract = load_fixture_extract(fixture)
    progress = replay_extract(fixture, extract)
    print(progress)
    if progress.stop == 'mismatch':
        warnings.warn(str(progress))
    final = extract['records'][-1]['state']['game']
    by_hand = ENDINGS[fixture] == 'manually_ended'
    assert final['end_reason'] == ENDINGS[fixture]
    assert progress.stop is None and progress.matched == progress.records - 1 - by_hand, str(progress)
    assert progress.private_moves == PRIVATE_PATHS[fixture]
    state, ids = progress.state, [p['id'] for p in extract['players']]
    assert is_terminal_py(state) != by_hand
    assert END_REASONS[TURN.end_reason(state)] == (None if by_hand else ENDINGS[fixture])
    assert engine_result(state, ids) == final['result']


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


def _private_moves(records, first: int, last: int) -> list[int]:
    """The records of the privates' own moves: MF's port tile, and ER's bonus lay or pass."""
    return [i for i in range(first + 1, last + 1) if records[i]['action']['entity_type'] == 'company']


# Per fixture: ORs and turns; the turns that stop at an action of a later
# sub-step before their last lay or token (none since 6.8a); the replayed
# routes that pay a human's revenue (override_route_revenue); the turns' stops
# by kind (none since 6.8b); and per kind of turn action, how many the turns
# hold and how many of them replay. The privates' moves are MF's port tile
# (962 record 33) and ER's bonus (314 record 90 passes, 962 record 42 lays).
# 314's emergencies: TR buys KO's 3 for ¥2 (350); SR's president pays toward
# a 6 (444); KO's sells four corps' shares and pays toward a D (449-453);
# and TR's sells KO's, then goes bankrupt (456-457), which ends the game.
# 962's last turn's dividend ends the game after the bank broke (446).
OR_TURN_COVERAGE = {
    '314': dict(ors=10, turns=43, cut_short=0, overrides=5, stops={}, lays_and_tokens=(46, 46),
                private_moves=(1, 1), run_routes=(32, 32), dividend=(32, 32), buy_train=(30, 30),
                discard_train=(2, 2), buy_company=(6, 6), sell_shares=(5, 5), bankrupt=(1, 1)),
    '962': dict(ors=10, turns=56, cut_short=0, overrides=1, stops={}, lays_and_tokens=(59, 59),
                private_moves=(2, 2), run_routes=(48, 48), dividend=(48, 48), buy_train=(21, 21),
                discard_train=(0, 0), buy_company=(7, 7), sell_shares=(0, 0), bankrupt=(0, 0)),
    **{fixture: dict(ors=1, turns=1, cut_short=0, overrides=0, stops={}, lays_and_tokens=(0, 0),
                     private_moves=(0, 0), run_routes=(0, 0), dividend=(0, 0), buy_train=(0, 0),
                     discard_train=(0, 0), buy_company=(0, 0), sell_shares=(0, 0), bankrupt=(0, 0))
       for fixture, ending in ENDINGS.items() if ending == 'manually_ended'},
}
TURN_ACTIONS = ('run_routes', 'dividend', 'buy_train', 'discard_train', 'buy_company', 'sell_shares', 'bankrupt')


@pytest.mark.parametrize('fixture', FIXTURES)
def test_every_or_turn_replays_from_its_first_snapshot(fixture):
    """Each corp's turn in each OR, from ``load_state`` of the snapshot where
    it opens (each OR's first turn opens the OR): the lays, tokens, passes,
    routes, dividends, train purchases, discards, private purchases, and
    emergencies of real games, with Ruby's BUY_TRAIN choices checked at every
    BuyTrain snapshot. Since 6.8b every turn replays to its end, the game's
    end included."""
    extract = load_fixture_extract(fixture)
    records = extract['records']
    turns = _or_turns(extract)
    counts = {kind: [0, 0] for kind in ('lays_and_tokens', 'private_moves', *TURN_ACTIONS)}
    cut_short = overrides = 0
    stops: dict[str, int] = {}
    for first, last, _ in turns:
        progress = replay_extract(fixture, extract, start=first, end=last)
        assert progress.stop != 'mismatch', str(progress)
        if progress.stop:
            stops[progress.stop] = stops.get(progress.stop, 0) + 1
        found = {'lays_and_tokens': _lays_and_tokens(records, first, last),
                 'private_moves': _private_moves(records, first, last)}
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
# before them; sell_shares counts the emergency sales at BuyTrain.
ONE_ACTION_COVERAGE = {
    '314': {'dividend': (32, 32), 'buy_train': (30, 30), 'discard_train': (2, 2), 'buy_company': (6, 6),
            'sell_shares': (5, 5), 'bankrupt': (1, 1)},
    '962': {'dividend': (48, 48), 'buy_train': (21, 21), 'discard_train': (0, 0), 'buy_company': (7, 7),
            'sell_shares': (0, 0), 'bankrupt': (0, 0)},
}


@pytest.mark.parametrize('fixture', ('314', '962'))
def test_every_dividend_train_purchase_and_discard_replays_from_the_snapshot_before_it(fixture):
    """Each one alone, from ``load_state``: its payments, price move, phase
    change, rusting, and privates' closing (962 record 274: KU's first 5
    closes the UTF it owns); each private purchase, ER's up to its seller's
    ER_BONUS; each emergency sale, purchase, and the bankruptcy (314 records
    349-457); and the dividend that ends 962. A run of discards replays
    whole, since Ruby takes them in any order (314 records 339-340)."""
    extract = load_fixture_extract(fixture)
    records = extract['records']
    coverage = {kind: [0, 0] for kind in ONE_ACTION_COVERAGE[fixture]}
    for record in records[1:]:
        kind, i = record['action']['type'], record['index']
        if kind not in coverage or kind == 'discard_train' and records[i - 1]['action']['type'] == kind:
            continue
        if kind == 'sell_shares' and record['pre']['step'] != 'BuyTrain':  # an SR's, which the SR tests replay
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
    pending = [(K.DPHASE_BUY_PRIVATE_POST, K.BUY_PRIVATE_POST_PRIVATE + 2, 3)]
    decisions = [(K.DPHASE_BUY_PRIVATE_POST, K.BUY_PRIVATE_POST_PRIVATE + 2),
                 (K.DPHASE_PRIVATE_PRICE, K.PRIVATE_PRICE_EXACT + 80)]
    assert consume(pending, lambda phase: decisions, {3}) == decisions[1:] and pending == []


def test_sync_reports_no_match():
    pending = [PASS_AUCTION]
    assert consume(pending, lambda phase: [(K.DPHASE_AUCTION, K.AUCTION_BUY)], {0}) is None
    assert consume(pending, lambda phase: [], {0}) is None  # no decision, e.g. a message
    assert pending == [PASS_AUCTION]


def test_the_harness_passes_for_an_sr_turn_ruby_skips():
    """Player 1 has ¥0 and owns DR: the engine offers it [pass, DR], where
    Ruby skips the turn and logs nothing (as 314 and 962 do in SR 1). After
    player 0's pass the harness passes for player 1, so player 2's par
    applies; the supplied pass is not queued for the sync rule."""
    state = fresh_game(3)
    end_auction(state, [1 if p == D.PRIVATE_DR else 2 for p in range(D.NUM_PRIVATES - 1)])  # no UTF at 3
    BANK.transfer(state, 1, BANK_ACCOUNT, PLAYERS[1].cash(state))
    enter_stock_round(state, 0, turn=1)
    pending: list[Forced] = []
    log: list = []
    assert _apply(state, [(K.DPHASE_STOCK, K.STOCK_PASS)], {0}, pending, log, []) is None
    assert [(phase, action_id, int(row[ACTIVE_PLAYER_SLOT])) for row, phase, action_id in log] == [
        (K.DPHASE_STOCK, K.STOCK_PASS, 0), (K.DPHASE_STOCK, K.STOCK_PASS, 1)]
    assert pending == [] and TURN.active_player(state) == 2 and skipped_sr_turn(state) == []
    assert PLAYERS[1].passed(state) and PRIVATES[D.PRIVATE_DR].owner(state) == K.OWNER_PLAYER + 1
    par = [(K.DPHASE_STOCK, K.STOCK_PAR + CORP['KO']), (K.DPHASE_PAR, 0)]
    assert _apply(state, par, {2}, pending, log, []) is None and CORPS[CORP['KO']].president(state) == 2


def test_the_harness_applies_pre_done_before_rubys_other_actions():
    """KO's president owns DR, so the turn opens at PRE. Ruby logs a lay or a
    pass at its Track step, and the harness applies "done" first, not queued
    for the sync rule; the forced decisions after it are, and a Track pass
    with nothing to lay is one of them."""
    state = fresh_game(3)
    end_auction(state, [0])
    hold(state, CORP['KO'], {0: 6})
    TURN.set_or_count(state, 1)
    setup_operating_round_py(state)
    DRIVER.advance(state)
    assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE
    pending: list[Forced] = []
    log: list = []
    assert _apply(state, [(K.DPHASE_TRACK, K.TRACK_PASS)], {0}, pending, log, []) is None
    assert [(phase, action_id) for row, phase, action_id in log][:2] == [
        (K.DPHASE_BUY_PRIVATE_PRE, K.BUY_PRIVATE_PRE_DONE), (K.DPHASE_TRACK, K.TRACK_PASS)]
    assert pending and pending[0][:2] == (K.DPHASE_TOKEN, K.TOKEN_PASS)
    assert PRIVATES[D.PRIVATE_DR].owner(state) == K.OWNER_PLAYER  # the exchange waits for Ruby to log it


def test_pre_keeps_its_own_move_and_compares_as_rubys_track():
    """A private's move while the engine is at PRE takes PRE's slot; ``compare``
    takes PRE for Ruby's Track, as ``test_replay_runs_end_to_end`` exercises."""
    state = fresh_game(3)
    end_auction(state, [0])
    hold(state, CORP['KO'], {0: 6})
    TURN.set_or_count(state, 1)
    setup_operating_round_py(state)
    DRIVER.advance(state)
    pending: list[Forced] = []
    log: list = []
    assert _apply(state, [(K.DPHASE_BUY_PRIVATE_PRE, K.BUY_PRIVATE_PRE_DR)], {0}, pending, log, []) is None
    assert [(phase, action_id) for row, phase, action_id in log] == [(K.DPHASE_BUY_PRIVATE_PRE, K.BUY_PRIVATE_PRE_DR)]
    assert TURN.step(state) == K.STEP_BUY_PRIVATE_PRE and PRIVATES[D.PRIVATE_DR].owner(state) == K.OWNER_CLOSED
    assert RUBY_STEPS[K.STEP_BUY_PRIVATE_PRE] == 'Track'


def test_sync_matches_only_the_forced_player():
    """After a purchase in the first SR, the driver ends the buyer's turn with a
    forced "done" that Ruby does not log; the next player's pass is their own."""
    pending = [(K.DPHASE_STOCK, K.STOCK_PASS, 0)]
    assert consume(pending, lambda phase: [(phase, 0)], {1}) is None and pending == [(K.DPHASE_STOCK, 0, 0)]
