"""The checks on one game scraped from 18xx.games, which ``test_replay.py`` runs
in-process or in a spawned worker. Each takes ``(label, extract)``, fails by
raising, and returns counts whose first says how much it saw.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, NamedTuple

from g1889.actions import is_terminal_py
from g1889.entities.turn import TURN

from tests.g1889.replay.extracts import forget_extracts, load_scraped_extract
from tests.g1889.replay.harness import _later_srs, _or_turns, replay_extract
from tests.g1889.replay.snapshot import END_REASONS, engine_result
from tests.g1889.replay.test_extraction_oracles import check_extraction
from tests.g1889.replay.test_network_oracles import check_must_buy, check_network, check_operating_corp_ids
from tests.g1889.replay.test_private_oracles import check_privates
from tests.g1889.replay.test_route_oracles import (
    check_brute_force, check_human_revenues, check_solver_matches_auto_router,
)
from tests.g1889.replay.test_train_oracles import check_buy_train_and_discard


def check_whole_game(label, extract) -> tuple[int, ...]:
    """From a new game to Ruby's natural end: every record compares equal, and
    the engine ends the same way with the same scores. Returns the records."""
    final = extract['records'][-1]['state']['game']
    assert final['end_reason'] in ('bank', 'bankrupt'), f"{label}: Ruby's end_reason is {final['end_reason']}"
    progress = replay_extract(label, extract)
    assert progress.stop is None and progress.matched == progress.records - 1, str(progress)
    state = progress.state
    assert is_terminal_py(state), f'{label}: the engine has not ended where Ruby has'
    assert END_REASONS[TURN.end_reason(state)] == final['end_reason'], f'{label}: {TURN.end_reason(state)}'
    assert engine_result(state, [p['id'] for p in extract['players']]) == final['result'], label
    return (progress.records,)


def check_turns_alone(label, extract) -> tuple[int, int]:
    """Every OR turn and later SR from ``load_state`` of its first snapshot, with
    no stop of any kind. Failures do not hide one another. Returns the turns and SRs."""
    turns = [(first, last) for first, last, _ in _or_turns(extract)]
    srs = _later_srs(extract)
    failures = []
    for first, last in (*turns, *srs):
        try:
            progress = replay_extract(label, extract, start=first, end=last)
            if progress.stop or progress.matched != last:
                failures.append(f'records {first}-{last}: {progress}')
        except Exception as e:
            failures.append(f'records {first}-{last}: {type(e).__name__}: {e}')
    assert not failures, f'{len(failures)} of {len(turns) + len(srs)} spans:\n' + '\n'.join(failures)
    return len(turns), len(srs)


def check_no_revenue_above_the_maximum(label, extract) -> tuple[int, int]:
    """Nothing a human logged earns more than the engine's maximum. Returns the
    routes run and those below it, which are allowed: humans miss the best route."""
    runs, below, above = check_human_revenues(extract)
    assert not above, f"{label}: logged revenue above the engine's maximum at action ids {sorted(above)}"
    return runs, below


REPLAYS = {'whole game': check_whole_game, 'turns alone': check_turns_alone}
ORACLES = {
    'solver': check_solver_matches_auto_router, 'human revenue': check_no_revenue_above_the_maximum,
    'brute force': check_brute_force, 'network': check_network, 'operating ids': check_operating_corp_ids,
    'must buy': check_must_buy, 'buy train': check_buy_train_and_discard, 'privates': check_privates,
    'extraction': check_extraction,
}
CHECKS = REPLAYS | ORACLES
MODEL_MIN_PLAYERS = 3  # extraction serves the model; the other checks hold for the engine's 2 players too


class Outcome(NamedTuple):
    counts: Any = None
    failure: str | None = None  # the traceback


def first_count(counts) -> int:
    while isinstance(counts, tuple):
        counts = counts[0]
    return counts


def run_game(path: Path) -> dict[str, Outcome]:
    """Every check that applies to the game in ``path``, so that one failing does
    not hide the others. A file that will not load fails them all."""
    try:
        extract = load_scraped_extract(path)
    except Exception:
        return {name: Outcome(failure=traceback.format_exc()) for name in CHECKS}
    outcomes = {}
    for name, check in CHECKS.items():
        if check is check_extraction and len(extract['players']) < MODEL_MIN_PLAYERS:
            continue
        try:
            counts = check(path.stem, extract)
            assert first_count(counts) > 0, f'{path.stem}: {name} saw nothing to check: {counts}'
            outcomes[name] = Outcome(counts)
        except Exception:
            outcomes[name] = Outcome(failure=traceback.format_exc())
    forget_extracts()  # each is used once, and a full run loads them all
    return outcomes
