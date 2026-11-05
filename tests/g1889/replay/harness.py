"""The differential replay harness for 1889 games against Ruby (1889/engine-design.md
§10.2-3): the tests of ``test_differential_replay.py`` and ``tests/g1889/games/`` run it.

Replay walks a fixture's records, maps each Ruby action to engine decisions,
and applies them through ``DRIVER`` with history. The driver auto-applies
forced decisions, which Ruby may log as actions or skip silently. The sync
rule: a snapshot equal to the engine's state means Ruby rests where the engine
does, having skipped the pending forced decisions, so they are dropped. After
that, a Ruby action that matches a forced decision applied since the last sync,
by the same player and as taken at that decision's row, is consumed instead of
applied; one that matches none means Ruby skipped the pending forced decisions,
so they are dropped too. States are compared whenever no forced decision is
pending, and a record whose forced decisions are pending is compared when the
next action settles where Ruby rested: at the row of the forced decision it
takes, or else past them all. A chat, note, or programmed move's setting (``no_decision``) leaves
the game and the sync alone, and its snapshot stands with the one before it.
Every decision a Ruby action maps to, sub-phases such as PRIVATE_PRICE and PORT
included, must be the engine's active player's, one of Ruby's active players
before the action, but for DR's exchange out of turn, which is its owner's;
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
``run_routes`` maps to no decision or to ``override_route_revenue``. A
private's move there backs up to that row and takes the override, and ROUTE
runs again on what the move left. Discards (§10.2): Ruby accepts a discard
from any crowded corp, and the engine takes them in corp order, so the mapper
holds a discard that comes early (``HELD_DISCARD``) until the engine reaches
its corp; the records in between are not compared. If the driver has already
forced the earlier corp's discard, the harness backs up to it first
(``_discard_behind_another``). SR turns Ruby skips (§10.2): the engine offers DR
and MF on every SR turn of their owner, so it stops at a turn whose only
other choice is the pass, which Ruby skips or ends without logging
anything; after each Ruby action the harness applies that pass
(``skipped_sr_turn``), and queues the forced decisions after it as usual.
BUY_PRIVATE_PRE (§5.5): Ruby has no such step, so the engine at PRE compares
with Ruby's Track, and where Ruby then does anything but PRE's own moves the
harness applies PRE's "done" first (``_leave_pre``). A human's private move at
another of Ruby's stops takes the override, which returns to the decision it
interrupted; if the driver has already forced that decision for lack of a
choice, the harness backs up to it first (``_forced_at_ruby_step``). DR's
exchange by an owner who is not acting takes ``DRIVER.apply_out_of_turn``, in
the same way.

Replay stops at the first unsupported action or mismatch; an engine error
propagates. When it stops at an action while forced decisions are pending,
Ruby may be waiting at one of them (a Track step whose only choice is the
pass, which Ruby asks), so the record before is compared with the engine
where Ruby rests (``_ruby_rests``); so is a record whose forced decisions
are pending when the replay runs out.
``replay_extract(..., start=i)`` starts from ``load_state`` of record i, so each
round and turn also replays alone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py
from g1889.driver import DRIVER, STATUS_INVALID
from g1889.entities.corp import CORPS
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState, get_global_fields, get_layout

from tests.g1889.replay.action_map import (
    HELD_DISCARD, OUT_OF_TURN, OVERRIDE_STEP_PHASE, PRIVATE_OVERRIDE, ROUTE_OVERRIDE, TOPMOST_BID_OVERRIDE, Decision,
    ReplayEnd, UnsupportedAction, map_action, no_decision, skipped_sr_turn,
)
from tests.g1889.replay.extracts import Extract, load_fixture_extract
from tests.g1889.replay.snapshot import CORP, compare, load_state, track_after_lay

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
    verified: set[int] = field(default_factory=set)  # the records whose snapshots compared equal

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


def consume(pending: list[Forced], decisions_in: Callable[[Forced], list[Decision]],
            actors: set[int], rested: list[Forced] | None = None) -> list[Decision] | None:
    """The sync rule for one Ruby action by one of ``actors``. ``decisions_in(forced)``
    maps the action as taken at that forced decision, in its phase and on its
    row; an action that cannot be taken there does not match it. If its decisions
    continue the pending forced ones for that player from some entry on, drop the
    pending decisions through the match and return the decisions still to apply,
    adding the entry the match starts at, where Ruby was resting, to ``rested``;
    return None when nothing matches."""
    for j, entry in enumerate(pending):
        try:
            decisions = decisions_in(entry)
        except (UnsupportedAction, ValueError):
            continue
        m = min(len(decisions), len(pending) - j)
        if m and all(forced[:2] == decisions[k] and forced[2] in actors
                     for k, forced in enumerate(pending[j:j + m])):
            if rested is not None:
                rested.append(pending[j])
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
    (``override_private_slot``), an ``OUT_OF_TURN`` makes DR's exchange
    (``apply_out_of_turn``), and a ``TOPMOST_BID_OVERRIDE`` a bid on private 0
    (``override_topmost_bid``), none recording a decision of its own, and a
    ``HELD_DISCARD`` waits in ``held`` for its corp's turn to discard
    (``_release``). Then the engine passes for each SR turn Ruby skips or ends
    (``_pass_skipped_sr_turns``)."""
    history: list
    for phase, action_id in decisions:
        if phase == HELD_DISCARD:  # by its corp's president, whose discard Ruby takes out of its order
            corp, slot = divmod(action_id, D.NUM_TRAIN_TYPES)
            held.append((corp, slot, {CORPS[corp].president(state)}))
            continue
        if phase == OUT_OF_TURN:  # by the private's owner, not Ruby's active player, and the decision goes on
            history = []
            if DRIVER.apply_out_of_turn(state, action_id, history) == STATUS_INVALID:
                return f'{D.PRIVATE_SYMS[action_id]} out of turn is illegal at {K.Step(TURN.step(state)).name}'
            _queue(pending, history)
            log += history
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
        history = []
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


def _forced_at_ruby_step(pending: list[Forced], action, pre, track_open: bool, route_row) -> Any:
    """The row a human's private move backs up to, or None. The driver forced
    the decision it interrupts for lack of a choice in the engine's action
    space, as it can a must-buy's one source, where Ruby still waits and joins
    the private's move to its step. With Track kept open after the lay it is
    the first pending decision, as the engine has left TRACK. At Route it is
    the ROUTE step itself: the move can change what the routes earn (ER's bonus
    lay), so ROUTE runs again after it."""
    if not _is_private_move(action) or pre.get('round') != 'OR':
        return None
    if pre['step'] == 'Route':
        return route_row
    if pre['step'] not in OVERRIDE_STEP_PHASE:
        return None
    if track_open:
        return pending[0][3] if pending else None
    phase = OVERRIDE_STEP_PHASE[pre['step']]
    return next((forced[3] for forced in pending if forced[0] == phase), None)


def _discard_behind_another(state: GameState, pending: list[Forced], action) -> Any:
    """The row of the driver's forced discard, if ``action`` is another crowded
    corp's: Ruby takes discards in any order, so it has yet to make the forced
    one, and the harness backs up to it and holds ``action`` (``_discard_train``)."""
    if action['type'] != 'discard_train':
        return None
    row = next((forced[3] for forced in pending if forced[0] == K.DPHASE_DISCARD), None)
    return None if row is None or _discarding(state, row) == CORP[action['entity']] else row


def _private_path(phase: int, pre) -> str:
    """How a private's move replays: its slot's phase, the override at Ruby's step, or out of turn."""
    if phase == OUT_OF_TURN:
        return 'out of turn'
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
    route_row = None               # the engine's row at its last ROUTE step

    def at_row(row) -> GameState:
        return GameState.from_array(row, len(ids), state.max_players)

    def matches(record, at: GameState | None = None) -> bool:
        if at is None:
            at = state
            if record['state']['game']['step'] == 'Route' and route_row is not None:  # EQUIVALENT: before ROUTE
                at = at_row(route_row)
        if diffs := compare(at, record['state']):
            progress.stop, progress.detail = 'mismatch', f"record {record['index']}: " + '; '.join(diffs)
            return False
        progress.matched = record['index']
        progress.verified.add(record['index'])
        return True

    def rests(record) -> bool:
        """Compare a record whose forced decisions are pending, with no next
        Ruby action to tell whether Ruby took them: it matches if Ruby rests
        at the engine's state or before one of them (a Track step whose
        only choice is the pass, which Ruby asks)."""
        if record['state']['game']['step'] == 'Route':
            return matches(record)
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
    if TURN.step(state) == K.STEP_ROUTE:  # load_state stops where Ruby waits; the engine runs on by itself
        route_row = state._array.copy()
        log: list = []
        DRIVER.advance(state, log)
        _queue(pending, log)
    for record in records[start + 1:None if end is None else end + 1]:
        host = record['action']
        if (no_decision(host) and not host.get('auto_actions')
                and record['state'] == records[record['index'] - 1]['state']):
            # Ruby's state is the record before's, so its compare, done or deferred, stands for this one.
            if deferred is not None:
                deferred = record
            elif not held:
                progress.matched = record['index']
            continue
        singles = [(host, record['pre']), *zip(host.get('auto_actions') or [], record.get('auto_pre') or [])]
        where = f"record {record['index']} ({host['type']})"
        log = []
        if track_open and _closes_track(record):  # EQUIVALENT: Track after the lay; no decision of its own
            singles = []
        was_open, track_open = track_open, False
        try:
            for action, pre in singles:
                if no_decision(action):
                    continue
                actors = {ids.index(player) for player in pre['active_players']}
                # EQUIVALENT: Ruby takes this discard before the one the driver forced.
                row = _discard_behind_another(state, pending, action)
                rested: list[Forced] = []
                decisions = None if row is not None else consume(
                    pending, lambda forced: map_action(action, pre, forced[0], at_row(forced[3])), actors, rested)
                if rested and deferred is not None:  # Ruby logs a forced decision, so it was resting at its row
                    if not held and not matches(deferred, at_row(rested[0][3])):
                        return progress
                    deferred = None
                if decisions is None:
                    if row is None:
                        row = _forced_at_ruby_step(pending, action, pre, was_open, route_row)
                        if row is not None:
                            progress.private_moves['backed up'] += 1
                    if row is not None:  # EQUIVALENT: Ruby still waits there; back up to it
                        state._array[:] = row
                        pending.clear()
                        if deferred is not None and not held and not matches(deferred, state):
                            return progress
                        deferred = None
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
        route_row = next((row for row, phase, step in reversed(log) if phase < 0 and step == K.STEP_ROUTE), route_row)
        if track_after_lay(record['state']):  # compare at TOKEN: the row before a forced TOKEN, or the rest
            at_token = next((row for row, phase, _ in log if phase == K.DPHASE_TOKEN), None)
            at = state if at_token is None else at_row(at_token)
            if not matches(record, at):
                return progress
            track_open, deferred = True, None
            continue
        if held:  # Ruby's state between discards in its order, which the engine never reaches
            deferred = None
            continue
        if pending and not compare(state, record['state']):  # Ruby rests where the engine does, past them all
            pending.clear()
        deferred = record if pending else None
        if not pending and not matches(record):
            return progress
    if held:
        progress.stop, progress.detail = 'mismatch', f'discards never reached in the engine: {held}'
    elif deferred is not None:  # no Ruby action is left to log what is pending
        rests(deferred)
    return progress


def unverified_records(extract: Extract, progress: Progress) -> list[int]:
    """The records a replay left uncompared, but for those it cannot compare: a
    discard Ruby takes out of the engine's order, and a snapshot equal to the
    record's before, whose compare stands for it."""
    records = extract['records']
    return [r['index'] for r in records[1:] if r['index'] not in progress.verified
            and r['action']['type'] != 'discard_train' and r['state'] != records[r['index'] - 1]['state']]


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
