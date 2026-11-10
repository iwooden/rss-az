"""1889 self-play diagnostics (1889/model-design.md §6.6): the collector behind
``Game.new_selfplay_stats``.
"""

from __future__ import annotations

import numpy as np

import g1889.constants as K
import g1889.data.tables as D
from g1889.actions import get_decision_phase_py
from g1889.entities.corp import CORPS
from g1889.entities.phase import PHASE
from g1889.entities.player import PLAYERS
from g1889.entities.private import PRIVATES, PrivateFlag
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.state import GameState

_MF = PRIVATES[D.PRIVATE_MF]


def _trains_bought(state: GameState, train_type: int) -> int:
    """New trains of a type the corps have bought: the copies gone from the
    depot, where a rank's last copy sells before the rank that rusts it. D's
    depot is unlimited and D never rusts, so its count is the Ds in play. A
    purchase from the pool or another corp moves a train already counted."""
    if D.TRAIN_COUNT[train_type] < 0:
        return TRAINS.pool(state, train_type) + sum(
            TRAINS.corp_trains(state, c, train_type) for c in range(D.NUM_CORPS)
        )
    return D.TRAIN_COUNT[train_type] - TRAINS.depot(state, train_type)


def _players_cash(state: GameState) -> int:
    return sum(PLAYERS[i].cash(state) for i in range(TURN.num_players(state)))


def _pool_shares(state: GameState) -> int:
    return sum(CORPS[c].pool_shares(state) for c in range(D.NUM_CORPS))


def _tokens_beyond_home(state: GameState) -> int:
    """Station tokens on the map besides home tokens, a corp's first."""
    return sum(max(D.CORP_TOKENS[c] - CORPS[c].tokens_left(state) - 1, 0) for c in range(D.NUM_CORPS))


def _stranded(state: GameState) -> bool:
    """A corp is parred but not floated."""
    return any(CORPS[c].par_index(state) >= 0 and not CORPS[c].floated(state) for c in range(D.NUM_CORPS))


def _srs_over(game_round: int, turn: int) -> int:
    """SRs over by a state of this round and SR number: an OR set, or the game's
    end, comes after its SR."""
    if game_round == K.ROUND_AUCTION:
        return 0
    return turn - (game_round == K.ROUND_STOCK)


def _mf_port_laid(state: GameState) -> bool:
    return bool(_MF.flags(state) & PrivateFlag.MF_PORT_USED)


class G1889SelfPlayStats:
    """One self-play game's end, length in ORs, track passes, withholds, floats,
    final treasuries, and trains; and its auction, pars, capitalization, SR2
    sales, tokens, phase timing, and MF's port tile.

    ``finish`` reports per-game values as means over games (denominator 1), and
    the rest as rates: track passes over the group's TRACK decisions and withholds
    over its DIVIDEND decisions, those with a choice, since the driver plays a
    forced pass or withhold without a search; and final cash over floated corps.

    The driver plays forced decisions and automated steps unseen, so the later
    stats come from the state: what changed between consecutive observations,
    or by the game's end. The first observation stands for everything before
    it. A game started mid-way reports a stat only if the events it covers
    come after its first observation, and leaves the stat out otherwise, as it
    does where the game had none of them.
    """

    def __init__(self, num_players: int) -> None:
        self.track_decisions = 0
        self.track_passes = 0
        # DIVIDEND decisions and withholds, by game phase.
        self.dividends = [0] * len(D.PHASE_NAMES)
        self.withholds = [0] * len(D.PHASE_NAMES)
        # OR sets' lengths by SR number. Forced decisions pass unseen, and an OR
        # can hold none but them, so ORs are counted from these, not as seen.
        self.or_set_lengths: dict[int, int] = {}
        self.started = False
        # The previous observation's round, SR number, and OR number.
        self.game_round = self.turn = self.or_num = 0

    def _start(self, state: GameState, game_round: int, turn: int) -> None:
        """What was there before the first observation."""
        self.started = True
        # The auction, from before its first sale: each unsold private's bids
        # by player as the latest observation's choice leaves them.
        self.bids: dict[int, list[int]] | None = None
        if game_round == K.ROUND_AUCTION and all(
            PRIVATES[p].owner(state) in (K.OWNER_UNSOLD, K.OWNER_NOT_IN_PLAY) for p in range(D.NUM_PRIVATES)
        ):
            self.bids = {p: [] for p in range(D.NUM_PRIVATES) if PRIVATES[p].owner(state) == K.OWNER_UNSOLD}
        self.premium = 0
        self.sr1_cash: int | None = None
        self.parred = {c for c in range(D.NUM_CORPS) if CORPS[c].par_index(state) >= 0}
        self.floated = {c for c in range(D.NUM_CORPS) if CORPS[c].floated(state)}
        self.float_cash: list[int] = []
        self.srs_over = _srs_over(game_round, turn)
        self.srs_ended = self.srs_stranded = 0
        # SR2 from before it starts: its sales are the pool's growth across it
        # plus its pool purchases, which are never forced.
        self.before_sr2 = turn == 1
        self.pool_before_sr2 = _pool_shares(state)
        self.sr2_pool_buys = 0
        self.sr2_sold: int | None = None
        self.tokens_from_start = _tokens_beyond_home(state) == 0
        self.cash_5s: int | None = None
        self.watch_5s = TRAINS.depot(state, D.TRAIN_4) > 0
        # Each later phase's OR, as (SR number, OR number in the set).
        self.phase = PHASE.current(state)
        self.phase_starts: dict[int, tuple[int, int]] = {}
        mf_owner = _MF.owner(state)
        self.watch_mf = not _mf_port_laid(state) and (
            mf_owner == K.OWNER_UNSOLD or K.OWNER_PLAYER <= mf_owner < K.OWNER_CORP
        )
        # The SR number of the round MF's port tile was laid in, and whether it was an OR.
        self.mf_port: tuple[int, bool] | None = None

    def _changes(self, state: GameState, game_round: int, turn: int) -> None:
        """What happened since the previous observation, read from ``state``,
        the next observation or the game's end."""
        if self.bids is not None:
            sr1 = game_round == K.ROUND_STOCK and turn == 1
            if game_round == K.ROUND_AUCTION or sr1:
                # Cash cannot price a sale: forced passes can pay the privates
                # out between two sales. A private the buyer bid on sells at
                # the bid; any other at its price, face or A's less the
                # discount, which stops changing once A sells.
                for p in [p for p in self.bids if PRIVATES[p].owner(state) != K.OWNER_UNSOLD]:
                    bid = self.bids.pop(p)[PRIVATES[p].owner(state) - K.OWNER_PLAYER]
                    self.premium += (bid or PRIVATES[p].price(state)) - D.PRIVATE_VALUE[p]
            if game_round != K.ROUND_AUCTION:
                # SR1's turns before its first choice are passes, so its cash is SR1's opening cash.
                self.sr1_cash = _players_cash(state) if sr1 else None
                self.bids = None
        phase = PHASE.current(state)
        if phase != self.phase:
            # A phase starts with a train bought in an OR, at the latest
            # observation's choice or later: here, in the earliest OR that can be.
            for p in range(self.phase + 1, phase + 1):
                self.phase_starts[p] = (self.turn, self.or_num)
            self.phase = phase
        srs_over = _srs_over(game_round, turn)
        if srs_over > self.srs_over:
            # Pars and floats come from choices, so an SR's end looks as the next observation does.
            self.srs_stranded += _stranded(state) * (srs_over - self.srs_over)
            self.srs_ended += srs_over - self.srs_over
            self.srs_over = srs_over
        if self.before_sr2 and self.sr2_sold is None:
            if turn == 1:
                self.pool_before_sr2 = _pool_shares(state)
            elif srs_over >= 2:
                self.sr2_sold = _pool_shares(state) - self.pool_before_sr2 + self.sr2_pool_buys
        if self.watch_mf and _mf_port_laid(state):
            self.watch_mf = False
            # In the latest observation's round: the MF slot is a choice, and PORT follows it.
            self.mf_port = (self.turn, self.game_round == K.ROUND_OPERATING)

    def observe(
        self,
        state: GameState,
        legal_actions: np.ndarray,
        chosen_idx: int,
        priors: np.ndarray,
        visit_counts: np.ndarray,
    ) -> None:
        # or_count holds a set's length from its SR's end to the next SR's.
        game_round = TURN.round(state)
        turn = TURN.turn(state)
        if game_round == K.ROUND_OPERATING:
            self.or_set_lengths[turn] = TURN.or_count(state)
        elif game_round == K.ROUND_STOCK:
            self.or_set_lengths[turn - 1] = TURN.or_count(state)
        if self.started:
            self._changes(state, game_round, turn)
        else:
            self._start(state, game_round, turn)
        # As of the first searched decision after the event: a float's treasury
        # (10 x par until it operates), and the richest once the last 4 has sold.
        for c in range(D.NUM_CORPS):
            if c not in self.floated and CORPS[c].floated(state):
                self.floated.add(c)
                self.float_cash.append(CORPS[c].cash(state))
        if self.watch_5s and TRAINS.depot(state, D.TRAIN_4) == 0:
            self.watch_5s = False
            self.cash_5s = max((CORPS[c].cash(state) for c in self.floated), default=0)
        self.game_round, self.turn, self.or_num = game_round, turn, TURN.or_num(state)

        action = int(legal_actions[chosen_idx])
        decision_phase = get_decision_phase_py(state)
        if self.bids is not None:
            self._bids_after(state, decision_phase, action)
        if decision_phase == K.DPHASE_TRACK:
            self.track_decisions += 1
            self.track_passes += int(action == K.TRACK_PASS)
        elif decision_phase == K.DPHASE_DIVIDEND:
            game_phase = PHASE.current(state)
            self.dividends[game_phase] += 1
            self.withholds[game_phase] += int(action == K.DIVIDEND_WITHHOLD)
        elif decision_phase == K.DPHASE_STOCK and turn == 2:
            self.sr2_pool_buys += int(K.STOCK_POOL <= action < K.STOCK_POOL + D.NUM_CORPS)

    def _bids_after(self, state: GameState, decision_phase: int, action: int) -> None:
        """The unsold privates' bids once this auction decision applies. Until
        the next observation, bids change only by choices (a closed auction's
        bidder may always drop out), but for BID's one amount after AUCTION
        selects a private: the least bid."""
        assert self.bids is not None
        for p, bids in self.bids.items():
            bids[:] = [PLAYERS[i].bid(state, p) for i in range(TURN.num_players(state))]
        player = TURN.active_player(state)
        if decision_phase == K.DPHASE_BID:
            p = TURN.auction_private(state)
            p = p if p >= 0 else TURN.pending_private(state)
            if action == K.BID_DROP:
                self.bids[p][player] = 0
            elif action >= K.BID_EXACT:
                self.bids[p][player] = action - K.BID_EXACT
            else:
                self.bids[p][player] = PRIVATES[p].min_bid(state) + D.MIN_BID_INCREMENT * (action - K.BID_POINT)
        elif decision_phase == K.DPHASE_AUCTION and action >= K.AUCTION_BID:
            p = action - K.AUCTION_BID + 1
            self.bids[p][player] = PRIVATES[p].min_bid(state)

    def _or_number(self, turn: int, or_num: int) -> int:
        """The game's OR number of OR ``or_num`` in the set after SR ``turn``."""
        return or_num + sum(self.or_set_lengths.get(sr, 1) for sr in range(1, turn))

    def finish(self, state: GameState) -> dict[str, tuple[float, float]]:
        # The game ends in an OR, of the set its SR number names. A set with no
        # search in it or the next SR has at least one OR.
        ors_played = self._or_number(TURN.turn(state), TURN.or_num(state))
        end_reason = TURN.end_reason(state)
        last_phase = PHASE.current(state)
        floated = [c for c in range(D.NUM_CORPS) if CORPS[c].floated(state)]
        stats: dict[str, tuple[float, float]] = {
            "end_bank": (float(end_reason == K.END_REASON_BANK), 1),
            "end_bankruptcy": (float(end_reason == K.END_REASON_BANKRUPT), 1),
            "ors_played": (float(ors_played), 1),
            "track_pass_share": (float(self.track_passes), float(self.track_decisions)),
            "withhold_share": (float(sum(self.withholds)), float(sum(self.dividends))),
            "corps_floated": (float(len(floated)), 1),
            "corp_cash_end": (float(sum(CORPS[c].cash(state) for c in floated)), float(len(floated))),
        }
        for phase, name in enumerate(D.PHASE_NAMES):
            stats[f"last_phase_{name}"] = (float(phase == last_phase), 1)
            # Absent for a phase with no such decision: logged, a 0 would read as never withholding.
            if self.dividends[phase]:
                stats[f"withhold_share_{name}"] = (float(self.withholds[phase]), float(self.dividends[phase]))
        for train_type, name in enumerate(D.TRAIN_NAMES):
            stats[f"trains_bought_{name}"] = (float(_trains_bought(state, train_type)), 1)
        if not self.started:
            return stats

        self._changes(state, TURN.round(state), TURN.turn(state))
        if self.sr1_cash is not None:
            stats["auction_premium"] = (float(self.premium), 1)
            stats["auction_sr1_cash"] = (float(self.sr1_cash), float(TURN.num_players(state)))
        pars = [CORPS[c].par_price(state) for c in range(D.NUM_CORPS)
                if c not in self.parred and CORPS[c].par_index(state) >= 0]
        if pars:
            stats["par_price"] = (float(sum(pars)), float(len(pars)))
            stats["par_65_share"] = (float(pars.count(D.PAR_PRICE[0])), float(len(pars)))
        if self.srs_ended:
            stats["par_stranded_share"] = (float(self.srs_stranded), float(self.srs_ended))
        if self.float_cash:
            stats["corp_cash_float"] = (float(sum(self.float_cash)), float(len(self.float_cash)))
        if self.cash_5s is not None:
            stats["corp_cash_max_5s"] = (float(self.cash_5s), 1)
        if self.sr2_sold is not None:
            stats["sr2_shares_sold"] = (float(self.sr2_sold), 1)
        if self.tokens_from_start:
            stats["tokens_placed"] = (float(_tokens_beyond_home(state)), 1)
        if self.before_sr2:
            for phase, (turn, or_num) in self.phase_starts.items():
                stats[f"phase_start_or_{D.PHASE_NAMES[phase]}"] = (float(self._or_number(turn, or_num)), 1)
        if self.mf_port is not None or self.watch_mf:
            stats["mf_port_laid"] = (float(self.mf_port is not None), 1)
        if self.mf_port is not None:
            stats["mf_port_sr"] = (float(self.mf_port[0]), 1)
            stats["mf_port_in_or"] = (float(self.mf_port[1]), 1)
        return stats
