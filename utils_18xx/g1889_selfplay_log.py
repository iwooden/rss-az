"""Play one 1889 self-play game from a checkpoint and write an 18xx.games hotseat log.

Import the output with "Import hotseat game" on 18xx.games or a local instance, and
step through it to see the map and the model's lays, as ``train/analyze_game.py``
shows an RSS game in markdown.

Usage:
    .venv/bin/python -m utils_18xx.g1889_selfplay_log CHECKPOINT|latest \
        [--checkpoint-dir D] --num-players 4 --seed 1 [--simulations N] [--device D] \
        --output game.json [--no-eval] [--verify]

    # Convert a game already played, without replaying it (a game takes many minutes):
    .venv/bin/python -m utils_18xx.g1889_selfplay_log --played game.played.json --output game.json

The game is the real loop, ``train.self_play.play_game``, at the self-play settings
of the epoch the checkpoint's weights play (``compute_epoch_config(cp["epoch"] + 1)``,
as training resumes): root noise, flags, and simulations. Its moves are strict, as
in ``train/analyze_game.py``: each is the search's most visited (``strict_config``).
It evaluates in fp32 uncompiled. Playing saves ``<output>.played.json`` before
conversion starts.

Conversion: the saved decisions re-drive a fresh game with driver history, which
gives every decision's row, forced ones included. Each decision that completes a
Ruby action becomes a semantic intent (``game_intents``), read from its row. Ruby
(``g1889_compose.rb``) turns the intents into a validated log; Ruby is a
dependency of this tool only. ``--verify`` also replays the log against Ruby's state
at every record (``tests/g1889/replay/harness.py:check_export``). Ruby's refusal of
a move is reported with its context: it may be a bug in the engine, not here.

Seats: player ``i`` of the engine is ``Seat i+1`` with id ``i+1``. Ruby's first
player is the first of the log's ``players``, so the log lists them from the engine's
first player, in seat order. Everything else about a seat is by its id.

Model evaluation: each searched decision adds a ``message`` by the acting seat just
before its move, unless ``--no-eval``:

    TRACK v=+0.18 | *G8 p.41 n.63 | F9 p.22 n.21 | pass p.19 n.09

``v`` is the search's root value for the acting seat: the root's mean backed-up
value (``value_sum / visit_count``) in the value head's [-1, 1], at this decision,
after the search. ``p`` is the raw (unnoised) network prior over the searched
actions, ``n`` the share of the root's visits. The top choices come by visits, and
``*`` marks the one played, which is the first of them. The UI prints the sender, so
the line does not.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

import g1889.actions as A
import g1889.constants as K
import g1889.data.tables as D
from g1889.driver import DRIVER, STATUS_GAME_OVER, STATUS_INVALID
from g1889.entities.private import PRIVATES, topmost
from g1889.entities.trains import TRAINS
from g1889.entities.turn import TURN
from g1889.routes.solver import best_routes_py
from g1889.state import GameState
from games import get_game
from games.g1889_selfplay import G1889SelfPlayStats
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import get_model_input_spec
from train import self_play
from train.checkpoint import find_latest_checkpoint, load_model_from_checkpoint
from train.config import TrainingConfig

COMPOSER = Path(__file__).with_name("g1889_compose.rb")
TOP_CHOICES = 3

Intent = dict[str, Any]


class ComposeError(RuntimeError):
    """Ruby refused the game. ``info`` holds its context (``g1889_compose.rb``'s ``reject``)."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        try:
            self.info: dict[str, Any] = json.loads(text)
        except ValueError:
            self.info = {"message": text}


@dataclass
class PlayedGame:
    """A played game as conversion needs it: JSON-safe, so a converter bug needs no replay."""

    config_json: str
    num_players: int
    seed: int
    actions: list[int]  # the searched decisions' ids, in order
    # Per searched decision: {"value", "legal", "priors", "visits"}; empty for a game without them.
    evals: list[dict[str, Any]]
    final_row: list[int]
    net_worths: list[int]
    source: dict[str, Any]  # how it was played, for the log's description

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> PlayedGame:
        return cls(**json.loads(path.read_text()))


# =============================================================================
# PLAYING
# =============================================================================


def strict_config(config: TrainingConfig) -> TrainingConfig:
    """``config`` for a game of the search's best moves, as ``train/analyze_game.py``
    plays one. Temperature 0 makes ``play_game`` take the most visited move. Root noise
    stays. Forced playouts go, as in every search outside training: they exist for
    pruned targets, and here they would spend simulations on the moves being compared.
    """
    return replace(config, temp_initial=0.0, temp_final=0.0, forced_playouts_k=0.0, policy_target_pruning="none")


def play(
    model: torch.nn.Module,
    config: TrainingConfig,
    *,
    num_players: int,
    seed: int,
    epoch: int,
    simulations: int | None,
    device: torch.device,
    source: dict[str, Any] | None = None,
    on_decision: Callable[[int], None] | None = None,
) -> PlayedGame:
    """One ``play_game`` at ``strict_config``, recording what it decides. Its collector
    hook gives each decision and the terminal state, and a pass-through ``run_search``
    the root; neither touches the search, the generator, or ``play_game`` itself."""
    config = strict_config(config)
    game = get_game(config)
    max_players = config.effective_max_players
    epoch_config = config.compute_epoch_config(epoch)
    if simulations is not None:
        epoch_config = replace(epoch_config, num_simulations=simulations)
    evaluator = NNEvaluator(
        game, model, device, num_players=max_players, terminal_rank_weight=config.terminal_blend,
        terminal_rank_tau=config.terminal_rank_tau,
        eval_dtype=None, input_spec=get_model_input_spec(config),
    )
    capacity = 2 * (max(config.max_simulations, epoch_config.num_simulations) + 1)
    pool = StatePool(game, capacity, game.state_size(max_players))
    roots: list[Any] = [None]
    actions: list[int] = []
    evals: list[dict[str, Any]] = []
    final_rows: list[list[int]] = []

    class Recorder(G1889SelfPlayStats):
        def observe(self, state, legal_actions, chosen_idx, priors, visit_counts) -> None:
            super().observe(state, legal_actions, chosen_idx, priors, visit_counts)
            root = roots[0]
            actions.append(int(legal_actions[chosen_idx]))
            evals.append({
                "value": round(float(root.value_sum[root.active_player_id] / root.visit_count), 4),
                "legal": [int(a) for a in legal_actions],
                "priors": [round(float(p), 4) for p in priors],
                "visits": [int(v) for v in visit_counts],
            })
            if on_decision is not None:
                on_decision(len(actions))

        def finish(self, state) -> dict[str, tuple[float, float]]:
            final_rows.append([int(v) for v in state._array])
            return super().finish(state)

    run_search = self_play.run_search

    def spying_run_search(*args: Any, **kwargs: Any) -> Any:
        roots[0] = run_search(*args, **kwargs)
        return roots[0]

    spied = replace(game, new_selfplay_stats=Recorder)
    with patch.object(self_play, "get_game", lambda _: spied), patch.object(self_play, "run_search", spying_run_search):
        record = self_play.play_game(
            evaluator, config, game_seed=seed, rng=np.random.default_rng(seed), state_pool=pool,
            epoch_config=epoch_config, num_players=num_players, rng_seed=seed,
        )
    return PlayedGame(
        config.to_json(), num_players, seed, actions, evals, final_rows[-1], list(record.net_worths),
        {**(source or {}), "epoch": epoch, "simulations": epoch_config.num_simulations,
         "c_puct": epoch_config.c_puct, "duration_secs": round(record.duration_secs, 1)},
    )


# =============================================================================
# THE GAME'S EVENTS
# =============================================================================


@dataclass
class Event:
    """One entry of the driver's history: a decision (``phase`` >= 0), or an
    automated step (``phase`` -1, ``action`` the step)."""

    row: np.ndarray
    phase: int
    action: int
    searched: int | None  # the searched decision's index; None when the driver forced it


def redrive(played: PlayedGame) -> tuple[list[Event], GameState]:
    """Drive a fresh game with the saved decisions. Every row of the game comes from
    this, which must end on the state the game ended on."""
    config = TrainingConfig.from_json(played.config_json)
    game = get_game(config)
    state = game.new_game(played.num_players, config.effective_max_players, played.seed)
    events: list[Event] = []
    status = None
    for i, action in enumerate(played.actions):
        assert status != STATUS_GAME_OVER, f"decision {i} after the game's end"
        history: list[tuple[np.ndarray, int, int]] = []
        status = DRIVER.apply_action(state, action, history)
        assert status != STATUS_INVALID, f"decision {i}, id {action}, is illegal on the re-drive"
        events += [Event(row, phase, act, i if k == 0 else None) for k, (row, phase, act) in enumerate(history)]
    assert status == STATUS_GAME_OVER, "the saved decisions do not finish the game"
    assert np.array_equal(state._array, played.final_row), "the re-drive ends on another state"
    assert game.final_scores(state) == played.net_worths, "the re-drive ends on other scores"
    return events, state


def _view(row: np.ndarray, num_players: int, max_players: int) -> GameState:
    return GameState.from_array(row, num_players, max_players=max_players)


# =============================================================================
# INTENTS: WHAT THE GAME DID, FOR THE RUBY COMPOSER
# =============================================================================


def _corp_sym(corp: int) -> str:
    return D.CORP_SYMS[corp]


def _by_player(view: GameState) -> Intent:
    return {"entity": TURN.active_player(view) + 1, "entity_type": "player", "order": [TURN.turn(view), 0],
            "round": "Auction" if TURN.round(view) == K.ROUND_AUCTION else "Stock"}


def _in_turn(view: GameState, reach: str | None = None) -> Intent:
    """The fields of an intent in the operating corp's turn: where Ruby must have
    got to before it (see ``settle`` in the composer)."""
    return {"operator": _corp_sym(TURN.operating_corp(view)), "order": [TURN.turn(view), TURN.or_num(view)],
            "reach": reach}


def _by_corp(view: GameState, reach: str | None = None) -> Intent:
    corp = _corp_sym(TURN.operating_corp(view))
    return {"entity": corp, "entity_type": "corporation", **_in_turn(view, reach)}


def _by_private(view: GameState, private: int, reach: str | None = None) -> Intent:
    """A private's own move: in an SR, or the operating corp's turn."""
    where = _in_turn(view, reach) if TURN.round(view) == K.ROUND_OPERATING else {"order": [TURN.turn(view), 0]}
    return {"entity": D.PRIVATE_SYMS[private], "entity_type": "company", **where}


def _only_dr_or_mf_besides_pass(view: GameState) -> bool:
    """An SR turn the engine offers for DR or MF alone, which Ruby skips or ends
    without logging anything: its step offers nothing, not even the pass."""
    buf = np.empty(K.MAX_ACTION_SIZE, dtype=np.uint16)
    legal = buf[:A.enumerate_legal_actions_py(view, buf)]
    return all(a == K.STOCK_PASS or A.decode_action_py(K.DPHASE_STOCK, int(a)).type
               in (A.ACTION_DR_EXCHANGE, A.ACTION_MF_PORT) for a in legal)


def _lay(hex_id: int, lay: tuple[int, int], by: Intent) -> Intent:
    tile, rotation = lay
    return {"type": "lay_tile", **by, "hex": D.HEX_IDS[hex_id], "tile": D.TILE_NAMES[tile], "rotation": rotation}


def _buy_train(view: GameState, train: int, price: int, exchange: int | None = None) -> Intent:
    source = TURN.pending_train_source(view)
    sources = {K.TRAIN_SOURCE_DEPOT: "depot", K.TRAIN_SOURCE_POOL: "pool"}
    intent = {"type": "buy_train", **_by_corp(view, "BuyTrain"), "train": D.TRAIN_NAMES[train], "price": price,
              "source": sources.get(source) or _corp_sym(source - K.TRAIN_SOURCE_CORP)}
    if exchange is not None:
        intent["exchange"] = D.TRAIN_NAMES[exchange]
    return intent


def decision_intent(view: GameState, phase: int, action: int, forced: bool) -> Intent | None:
    """The intent of the decision that completes a Ruby action, or None for one that
    opens a factored choice (its sub-decision makes the intent) or has no Ruby action."""
    info = A.decode_action_py(phase, action)
    kind = info.type
    bid_private = TURN.auction_private(view) if TURN.auction_private(view) >= 0 else TURN.pending_private(view)
    if phase in (K.DPHASE_AUCTION, K.DPHASE_BID):
        if kind == A.ACTION_PASS:
            return {"type": "pass", **_by_player(view), "optional": forced}
        if kind == A.ACTION_SELECT_BID:
            return None
        if kind == A.ACTION_BUY_TOPMOST:
            private = topmost(view)
            return {"type": "bid", **_by_player(view), "company": D.PRIVATE_SYMS[private],
                    "price": PRIVATES[private].min_bid(view)}
        return {"type": "bid", **_by_player(view), "company": D.PRIVATE_SYMS[bid_private],
                "price": A.price_amount_py(view, action)}
    if phase == K.DPHASE_STOCK:
        if kind == A.ACTION_PASS:
            return {"type": "pass", **_by_player(view), "optional": forced or _only_dr_or_mf_besides_pass(view),
                    "done": bool(TURN.sr_bought(view) or TURN.sr_sold_mask(view))}
        if kind in (A.ACTION_BUY_IPO, A.ACTION_BUY_POOL):
            return {"type": "buy_shares", **_by_player(view), "corporation": _corp_sym(info.corp),
                    "source": "ipo" if kind == A.ACTION_BUY_IPO else "pool"}
    if phase == K.DPHASE_PAR:
        return {"type": "par", **_by_player(view), "corporation": _corp_sym(TURN.pending_corp(view)),
                "price": info.amount}
    if phase == K.DPHASE_SELL_AMOUNT:
        return {"type": "sell_shares", **_by_player(view), "corporation": _corp_sym(TURN.pending_corp(view)),
                "count": info.count}
    if phase == K.DPHASE_EMERGENCY_SELL_AMOUNT:
        return {"type": "sell_shares", **_by_player(view), **_in_turn(view, "BuyTrain"),
                "corporation": _corp_sym(TURN.pending_corp(view)), "count": info.count}
    if kind == A.ACTION_DR_EXCHANGE:  # its share comes from the IPO
        reach = "BuyCompany" if phase == K.DPHASE_BUY_PRIVATE_POST else None
        return {"type": "buy_shares", **_by_private(view, D.PRIVATE_DR, reach), "player": TURN.active_player(view) + 1,
                "corporation": _corp_sym(D.DR_EXCHANGE_CORP), "source": "ipo"}
    if phase == K.DPHASE_PORT:
        return _lay(info.hex, (D.PORT_TILE, info.count), _by_private(view, D.PRIVATE_MF))
    if phase == K.DPHASE_TRACK and kind == A.ACTION_PASS:
        return {"type": "pass", **_by_corp(view, "Track"), "optional": forced}
    if phase == K.DPHASE_TILE:
        corp = TURN.operating_corp(view)
        lay = A.lay_candidates_py(view, TURN.pending_hex(view), corp)[action]
        return _lay(TURN.pending_hex(view), lay, _by_corp(view, "Track"))
    if phase == K.DPHASE_TOKEN:
        if kind == A.ACTION_PASS:
            return {"type": "pass", **_by_corp(view, "Token"), "optional": forced}
        return {"type": "place_token", **_by_corp(view, "Token"), "hex": D.HEX_IDS[info.hex]}
    if phase == K.DPHASE_DIVIDEND:
        return {"type": "dividend", **_by_corp(view, "Dividend"), "optional": forced,
                "kind": "payout" if kind == A.ACTION_PAYOUT else "withhold"}
    if phase == K.DPHASE_DISCARD:
        return {"type": "discard_train", "entity": _corp_sym(TRAINS.first_crowded(view)),
                "entity_type": "corporation", "train": D.TRAIN_NAMES[info.train_type]}
    if phase == K.DPHASE_BUY_TRAIN:
        if kind == A.ACTION_PASS:
            return {"type": "pass", **_by_corp(view, "BuyTrain"), "optional": forced}
        if kind == A.ACTION_BANKRUPT:
            return {"type": "bankrupt", **_by_corp(view, "BuyTrain")}
    if phase == K.DPHASE_SELECT_TRAIN:
        if kind == A.ACTION_TRADE_IN:
            traded = info.train_type
            return _buy_train(view, D.TRAIN_D, D.TRAIN_PRICE[D.TRAIN_D] - D.TRAIN_TRADE_IN[D.TRAIN_D][traded], traded)
        if TURN.pending_train_source(view) < K.TRAIN_SOURCE_CORP:
            return _buy_train(view, info.train_type, D.TRAIN_PRICE[info.train_type])
    if phase == K.DPHASE_TRAIN_PRICE:
        return _buy_train(view, TURN.pending_train_type(view), A.price_amount_py(view, action))
    if phase == K.DPHASE_BUY_PRIVATE_POST and kind == A.ACTION_PASS:
        return {"type": "pass", **_by_corp(view, "BuyCompany"), "optional": forced}
    if phase == K.DPHASE_PRIVATE_PRICE:
        reach = "BuyCompany" if TURN.return_step(view) == K.STEP_BUY_PRIVATE_POST else None
        return {"type": "buy_company", **_by_corp(view, reach),
                "company": D.PRIVATE_SYMS[TURN.pending_private(view)], "price": A.price_amount_py(view, action)}
    if phase == K.DPHASE_ER_BONUS:
        by = _by_private(view, D.PRIVATE_ER)
        if kind == A.ACTION_PASS:
            return {"type": "pass", **by, "optional": forced}
        return _lay(D.ER_HEX, A.er_candidates_py(view)[info.count], by)
    return None


def route_intent(view: GameState) -> Intent:
    """The engine's ROUTE step as Ruby's ``run_routes``: the solver's routes on the
    row before it, each walk cut into its legs between stops."""
    corp = TURN.operating_corp(view)
    routes = []
    for route in best_routes_py(view, corp):
        stops = {D.HEX_IDS[h] for h in route["stops"]}
        hexes = [D.HEX_IDS[h] for h in route["hexes"]]
        legs: list[list[str]] = []
        leg = hexes[:1]
        for hex_id in hexes[1:]:
            leg.append(hex_id)
            if hex_id in stops:
                legs.append(leg)
                leg = [hex_id]
        routes.append({"train": D.TRAIN_NAMES[route["train"]], "revenue": route["revenue"], "connections": legs,
                       "stops": [D.HEX_IDS[h] for h in route["stops"]]})
    return {"type": "run_routes", **_by_corp(view, "Route"), "routes": routes}


def action_label(view: GameState, phase: int, action: int) -> str:
    """A short name for a choice at the decision on ``view``."""
    info = A.decode_action_py(phase, action)
    kind = info.type
    if kind == A.ACTION_PASS:
        return {K.DPHASE_BID: "drop", K.DPHASE_BUY_PRIVATE_PRE: "done", K.DPHASE_BUY_PRIVATE_POST: "done"}.get(phase, "pass")
    if kind == A.ACTION_BUY_TOPMOST:
        return f"buy {D.PRIVATE_SYMS[topmost(view)]}"
    if kind == A.ACTION_SELECT_BID:
        return f"bid {D.PRIVATE_SYMS[info.private]}"
    if kind in (A.ACTION_PRICE_POINT, A.ACTION_EXACT_PRICE):
        return f"¥{A.price_amount_py(view, action)}"
    if kind == A.ACTION_PAR:
        return f"¥{info.amount}"
    if kind == A.ACTION_SELL:
        return f"{info.count}x"
    if kind == A.ACTION_DR_EXCHANGE:
        return "DR"
    if kind == A.ACTION_MF_PORT:
        return "MF"
    if kind in (A.ACTION_SELECT_HEX, A.ACTION_PLACE_TOKEN):
        return D.HEX_IDS[info.hex]
    if kind == A.ACTION_LAY_CANDIDATE:
        candidates = (A.er_candidates_py(view) if phase == K.DPHASE_ER_BONUS
                      else A.lay_candidates_py(view, TURN.pending_hex(view), TURN.operating_corp(view)))
        tile, rotation = candidates[info.count]
        return f"{D.TILE_NAMES[tile]}r{rotation}"
    if kind == A.ACTION_PORT_LAY:
        return f"{D.HEX_IDS[info.hex]}r{info.count}"
    if kind == A.ACTION_BUY_PRIVATE:
        return f"buy {D.PRIVATE_SYMS[info.private]}"
    if kind == A.ACTION_DISCARD or kind == A.ACTION_TRAIN_TYPE:
        return D.TRAIN_NAMES[info.train_type]
    if kind == A.ACTION_TRADE_IN:
        return f"D for {D.TRAIN_NAMES[info.train_type]}"
    prefix = {A.ACTION_BUY_IPO: "ipo", A.ACTION_BUY_POOL: "pool", A.ACTION_SELECT_PAR: "par",
              A.ACTION_SELECT_SALE: "sell", A.ACTION_BUY_CORP_TRAIN: "from"}.get(kind)
    if prefix:
        return f"{prefix} {_corp_sym(info.corp)}"
    return {A.ACTION_WITHHOLD: "withhold", A.ACTION_PAYOUT: "payout", A.ACTION_BUY_DEPOT_TRAIN: "depot",
            A.ACTION_BUY_POOL_TRAIN: "pool", A.ACTION_BANKRUPT: "bankrupt"}[kind]


def eval_message(view: GameState, phase: int, played_action: int, evaluation: dict[str, Any],
                 top: int = TOP_CHOICES) -> str:
    """The chat line of a searched decision, without its sender (the UI prints it)."""
    visits = np.array(evaluation["visits"])
    legal, priors = evaluation["legal"], evaluation["priors"]
    shown = [int(i) for i in np.argsort(-visits, kind="stable")[:top]]
    chosen = legal.index(played_action)
    if chosen not in shown:
        shown.append(chosen)

    def share(x: float) -> str:
        return f"{x:.2f}".removeprefix("0")

    total = visits.sum()
    parts = [f"{'*' if i == chosen else ''}{action_label(view, phase, legal[i])} p{share(priors[i])} "
             f"n{share(visits[i] / total)}" for i in shown]
    name = K.DecisionPhase(phase).name.removeprefix("DPHASE_")
    return " | ".join([f"{name} v={evaluation['value']:+.2f}", *parts])


def game_intents(played: PlayedGame, events: list[Event], max_players: int, include_eval: bool = True,
                 top: int = TOP_CHOICES) -> list[Intent]:
    """The game as intents in order: every decision of the driver's history that
    completes a Ruby action, each ROUTE step, and the evaluations' chat lines."""
    intents: list[Intent] = []
    for event in events:
        view = _view(event.row, played.num_players, max_players)
        if event.phase < 0:
            if event.action == K.STEP_ROUTE:
                intents.append(route_intent(view))
            continue
        if include_eval and played.evals and event.searched is not None:
            line = eval_message(view, event.phase, event.action, played.evals[event.searched], top)
            intents.append({"type": "message", **_by_player(view), "message": line})
        intent = decision_intent(view, event.phase, event.action, event.searched is None)
        if intent is not None:
            intents.append(intent)
    return intents


# =============================================================================
# THE LOG
# =============================================================================


def compose_log(played: PlayedGame, *, include_eval: bool = True, top: int = TOP_CHOICES,
                created_at: int = 0) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The hotseat JSON Ruby composes for the game, and Ruby's notes."""
    events, state = redrive(played)
    first = TURN.priority_player(_view(events[0].row, played.num_players, state.max_players))
    seats = [(first + i) % played.num_players for i in range(played.num_players)]
    created_at = created_at or int(time.time())
    source = played.source
    doc = {
        "id": f"hs_selfplay_{played.seed}",
        "seed": played.seed,
        "description": (f"1889 self-play: {source['checkpoint']} (epoch {source['epoch']}), seed {played.seed}, "
                        f"{source['simulations']} sims") if "checkpoint" in source else f"1889 game, seed {played.seed}",
        "players": [{"id": seat + 1, "name": f"Seat {seat + 1}"} for seat in seats],
        "created_at": created_at,
        "created_date": time.strftime("%Y-%m-%d", time.gmtime(created_at)),
        "intents": game_intents(played, events, state.max_players, include_eval, top),
    }
    result = subprocess.run(["ruby", str(COMPOSER)], input=json.dumps(doc), capture_output=True, encoding="utf-8",
                            check=False)
    if result.returncode != 0:
        raise ComposeError(result.stderr.strip())
    composed = json.loads(result.stdout)
    return composed["game"], composed["notes"]


def compose(played: PlayedGame, **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``compose_log``, whose result must be the engine's scores by seat."""
    log, notes = compose_log(played, **kwargs)
    scores = {int(seat): value for seat, value in log["result"].items()}
    expected = {seat + 1: value for seat, value in enumerate(played.net_worths)}
    if scores != expected:
        raise ComposeError(f"Ruby's result {scores} is not the engine's scores {expected}")
    return log, notes


# =============================================================================
# THE COMMAND LINE
# =============================================================================


def _play_from_checkpoint(args: argparse.Namespace, played_path: Path) -> PlayedGame:
    if args.checkpoint == "latest":
        path = find_latest_checkpoint(Path(args.checkpoint_dir))
        if path is None:
            raise SystemExit(f"no checkpoints in {args.checkpoint_dir}")
    else:
        path = Path(args.checkpoint)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, cp = load_model_from_checkpoint(path, device)
    if config.game != "1889":
        raise SystemExit(f"{path} is a {config.game} checkpoint")
    epoch = int(cp["epoch"]) + 1  # type: ignore[arg-type]  # the epoch these weights play
    start = time.perf_counter()

    def progress(decisions: int) -> None:
        if decisions % 25 == 0:
            print(f"  {decisions} decisions, {time.perf_counter() - start:.0f}s", file=sys.stderr, flush=True)

    print(f"Playing {path} on {device} (epoch {epoch} settings), {args.num_players} players, seed {args.seed}",
          file=sys.stderr, flush=True)
    played = play(model, config, num_players=args.num_players, seed=args.seed, epoch=epoch,
                  simulations=args.simulations, device=device, source={"checkpoint": str(path)},
                  on_decision=progress)
    played.save(played_path)
    print(f"Saved the played game to {played_path}: {len(played.actions)} decisions, {played.source['simulations']} "
          f"sims, {played.source['duration_secs']}s, scores {played.net_worths}", file=sys.stderr, flush=True)
    return played


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an 18xx.games hotseat log of a 1889 self-play game.")
    parser.add_argument("checkpoint", nargs="?", help="checkpoint path, or 'latest'")
    parser.add_argument("--checkpoint-dir", default="checkpoints/1889-v0", help="where 'latest' looks")
    parser.add_argument("--num-players", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1, help="the game's and the search's seed")
    parser.add_argument("--simulations", type=int, default=None, help="override the epoch's simulations")
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-eval", action="store_true", help="omit the evaluation chat lines")
    parser.add_argument("--played", type=Path, default=None, help="convert this saved game instead of playing")
    parser.add_argument("--verify", action="store_true",
                        help="replay the log against Ruby's state after every action (tests' replay harness)")
    args = parser.parse_args()
    if args.played is None and args.checkpoint is None:
        parser.error("give a checkpoint, or --played")

    played_path = args.output.with_suffix(".played.json")
    played = PlayedGame.load(args.played) if args.played else _play_from_checkpoint(args, played_path)
    try:
        log, notes = compose(played, include_eval=not args.no_eval)
    except ComposeError as e:
        raise SystemExit(f"Ruby refused the game ({args.played or played_path} keeps what was played):\n{e}") from e
    args.output.write_text(json.dumps(log))
    print(f"Wrote {args.output}: {len(log['actions'])} actions, {log['game_end_reason']} end, result {log['result']}",
          file=sys.stderr)
    for note in notes:
        print(f"note ({note['kind']}), intent {note['intent_index']}: {note['message']}", file=sys.stderr)
    if args.verify:
        from tests.g1889.replay.harness import check_export  # test tooling, on a box with the submodule

        problems = check_export(f"selfplay-{played.num_players}p-{played.seed}", log, played.net_worths,
                                redrive(played)[1])
        print("\n".join(problems) or "Verified: Ruby's state equals the engine's at every record", file=sys.stderr)
        if problems:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
