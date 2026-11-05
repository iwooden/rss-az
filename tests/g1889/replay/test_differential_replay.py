"""Differential replay of the 1889 fixtures against Ruby (1889/engine-design.md
§10.2-3), and the initial states of fixtures and fresh games (6.2's acceptance).
``harness.py`` holds the replay, its sync rule, and its equivalences.

Since 6.8b every fixture replays end to end, 314 to its bankruptcy and 962
to the game's end after the bank breaks, and the engine's final scores equal
Ruby's ``result`` (the ``hs_*`` fixtures' players end the game by hand,
before which their values equal Ruby's). ``replay_extract(..., start=i)``
starts from ``load_state`` of record i, so each round and turn also replays
alone.
"""

from __future__ import annotations

import warnings

import pytest

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import is_terminal_py
from g1889.driver import DRIVER
from g1889.entities.bank import BANK, BANK_ACCOUNT
from g1889.entities.corp import CORPS
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.phases.operating import setup_operating_round_py
from g1889.state import GameState

from tests.g1889.replay.action_map import HELD_DISCARD, UnsupportedAction, map_action, skipped_sr_turn
from tests.g1889.phases.conftest import end_auction, enter_stock_round, hold
from tests.g1889.phases.conftest import new_game as fresh_game
from tests.g1889.replay.extracts import FIXTURES, load_fixture_extract, load_new_game_extract
from tests.g1889.replay.harness import (
    ACTIVE_PLAYER_SLOT, Forced, Held, _apply, _discard_behind_another, _discarding, _later_srs, _or_turns, _queue,
    consume, new_game, replay_extract,
)
from tests.g1889.replay.snapshot import CORP, END_REASONS, RUBY_STEPS, compare, engine_result

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
    assert consume(pending, lambda forced: [(forced[0], 0)], {0}) == [] and pending == []
    assert consume([], lambda forced: [(forced[0], 0)], {0}) is None


def test_sync_drops_forced_decisions_ruby_skipped_before_a_logged_one():
    pending = [(K.DPHASE_TOKEN, K.TOKEN_PASS, 0), PASS_STOCK]

    def stock_only(forced):
        if forced[0] != K.DPHASE_STOCK:
            raise UnsupportedAction('not here')
        return [(forced[0], 0)]

    assert consume(pending, stock_only, {1}) == [] and pending == []


def test_sync_returns_the_rest_of_a_partly_forced_action():
    pending = [(K.DPHASE_BUY_PRIVATE_POST, K.BUY_PRIVATE_POST_PRIVATE + 2, 3)]
    decisions = [(K.DPHASE_BUY_PRIVATE_POST, K.BUY_PRIVATE_POST_PRIVATE + 2),
                 (K.DPHASE_PRIVATE_PRICE, K.PRIVATE_PRICE_EXACT + 80)]
    assert consume(pending, lambda forced: decisions, {3}) == decisions[1:] and pending == []


def test_sync_reports_no_match():
    pending = [PASS_AUCTION]
    assert consume(pending, lambda forced: [(K.DPHASE_AUCTION, K.AUCTION_BUY)], {0}) is None
    assert consume(pending, lambda forced: [], {0}) is None  # no decision, e.g. a message
    assert pending == [PASS_AUCTION]


def test_sync_does_not_match_a_decision_the_action_cannot_be_taken_at():
    """A lay that is no candidate at the forced decision's row raises there,
    for that decision only; the one after it can still match."""
    pending = [PASS_AUCTION, PASS_STOCK]

    def lay_at_stock(forced):
        if forced[0] != K.DPHASE_STOCK:
            raise ValueError('not an engine candidate here')
        return [(forced[0], 0)]

    assert consume(pending, lay_at_stock, {1}) == [] and pending == []


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
    assert consume(pending, lambda forced: [(forced[0], 0)], {1}) is None and pending == [(K.DPHASE_STOCK, 0, 0)]


def _two_crowded_corps():
    """UR's 5 has just made AR and TR crowded at the limit of 2, as in Ruby
    both wait: AR holds three 4s, so the driver forces its discard, and TR a 3
    and two 4s, so TR's is a choice. ``pending`` holds AR's forced discard."""
    state = fresh_game(3)
    end_auction(state, [2])
    for corp, trains in ((CORP['AR'], {D.TRAIN_4: 3}), (CORP['TR'], {D.TRAIN_3: 1, D.TRAIN_4: 2})):
        hold(state, corp, {0: 6})
        for t, count in trains.items():
            TRAINS.set_corp_trains(state, corp, t, count)
    hold(state, CORP['UR'], {1: 6})
    PHASE.set_current(state, D.GAME_PHASE_5)
    TURN.set_round(state, K.ROUND_OPERATING)
    TURN.set_operating_corp(state, CORP['UR'])
    TURN.set_active_player(state, 0)
    TURN.set_step(state, K.STEP_DISCARD)
    TURN.set_return_step(state, K.STEP_BUY_TRAIN)
    before = state._array.copy()
    history: list = []
    DRIVER.advance(state, history)
    pending: list[Forced] = []
    _queue(pending, history)
    assert [forced[:2] for forced in pending] == [(K.DPHASE_DISCARD, D.TRAIN_DISCARD_SLOT[D.TRAIN_4])]
    assert TRAINS.first_crowded(state) == CORP['TR'] and TURN.step(state) == K.STEP_DISCARD
    return state, pending, before


def test_ruby_may_discard_before_the_discard_the_driver_forced():
    """TR's discard reaches the harness first: it backs up to AR's forced
    discard and holds TR's, and AR's, when Ruby logs it, releases TR's. Both
    corps end at the limit, whichever order they discard in."""
    state, pending, before = _two_crowded_corps()
    trs = {'type': 'discard_train', 'entity': 'TR'}
    row = _discard_behind_another(state, pending, trs)
    assert row is not None and (row == before).all()
    state._array[:] = row
    pending.clear()
    assert _discarding(state) == CORP['AR']
    log: list = []
    held: list[Held] = []
    trs_3 = map_action(trs, {'train_type': '3'}, K.DPHASE_DISCARD, state)
    assert trs_3 == [(HELD_DISCARD, D.NUM_TRAIN_TYPES * CORP['TR'] + D.TRAIN_DISCARD_SLOT[D.TRAIN_3])]
    assert _apply(state, trs_3, {0}, pending, log, held) is None and len(held) == 1 and not log
    ars = {'type': 'discard_train', 'entity': 'AR'}
    assert _discard_behind_another(state, pending, ars) is None
    ars_4 = map_action(ars, {'train_type': '4'}, K.DPHASE_DISCARD, state)
    assert ars_4 == [(K.DPHASE_DISCARD, D.TRAIN_DISCARD_SLOT[D.TRAIN_4])]
    assert _apply(state, ars_4, {0}, pending, log, held) is None and not held
    assert TRAINS.corp_trains(state, CORP['AR'], D.TRAIN_4) == 2
    assert TRAINS.corp_trains(state, CORP['TR'], D.TRAIN_3) == 0
    assert TRAINS.corp_trains(state, CORP['TR'], D.TRAIN_4) == 2


def test_the_forced_discard_is_consumed_when_ruby_makes_it_first():
    """Ruby's own order: AR's discard is the forced one, so no back-up, and TR's then applies."""
    state, pending, _ = _two_crowded_corps()
    action = {'type': 'discard_train', 'entity': 'AR'}
    assert _discard_behind_another(state, pending, action) is None
    decided = consume(pending, lambda forced: map_action(action, {'train_type': '4'}, forced[0], GameState.from_array(
        forced[3], 3, state.max_players)), {0})
    assert decided == [] and pending == []
