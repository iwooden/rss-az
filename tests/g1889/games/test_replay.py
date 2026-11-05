"""Parity with Ruby on real games scraped from 18xx.games (1889/engine-design.md
§10.2): the whole game, each OR turn and later SR alone, and the oracles. The
game files are gitignored, so with none on disk these skip. ``checks.py`` holds
what runs per game.

By default a few smoke games run in-process. ``RSS_AZ_REPLAY_FULL=1`` runs every
file in ``data/`` (or ``RSS_AZ_REPLAY_DATA``) in a pool of ``RSS_AZ_REPLAY_JOBS``
workers, by default one per CPU. Only the selected tests' games run, so
``-k 271883`` checks one game.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from tests.g1889.games.checks import CHECKS, ORACLES, Outcome, first_count, check_whole_game, run_game
from tests.g1889.replay.extracts import load_fixture_extract, ruby

DATA_DIR = Path(os.environ.get('RSS_AZ_REPLAY_DATA') or Path(__file__).parent / 'data')
SMOKE_GAME_IDS = [
    '264180',  # ER bought at Ruby's Route step, whose bonus lay ROUTE must count
    '270059',  # DR exchanged out of turn
    '264038',  # 2 players: two crowded corps, a forced discard, and a bankruptcy
    '274060',  # the sync rule's false match
    '268833',  # 5 players to a natural end, which no fixture reaches
]
RUN_FULL_REPLAY = os.environ.get('RSS_AZ_REPLAY_FULL') == '1'


def _discover_game_ids() -> list[str]:
    discovered = sorted(path.stem for path in DATA_DIR.glob('*.json'))
    return discovered if RUN_FULL_REPLAY else [game_id for game_id in SMOKE_GAME_IDS if game_id in discovered]


game_ids = _discover_game_ids()


@pytest.fixture(scope='session')
def results(request) -> dict[str, dict[str, Outcome]]:
    """Each selected test's game, run once, for the tests to assert on."""
    if ruby() is None:  # as a cold extract would, but pytest's skip cannot cross from a worker
        pytest.skip('requires ruby')
    selected = sorted({str(item.callspec.params['game_id']) for item in request.session.items
                       if isinstance(item, pytest.Function) and item.function.__module__ == __name__
                       and 'game_id' in item.fixturenames})
    paths = [DATA_DIR / f'{game_id}.json' for game_id in selected]
    if not RUN_FULL_REPLAY:
        return dict(zip(selected, map(run_game, paths)))
    jobs = int(os.environ.get('RSS_AZ_REPLAY_JOBS') or os.cpu_count() or 1)
    with ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        return dict(zip(selected, pool.map(run_game, paths)))


def _assert_passed(outcomes: dict[str, Outcome], names) -> None:
    failures = {name: outcomes[name].failure for name in names if name in outcomes and outcomes[name].failure}
    assert not failures, f'{sorted(failures)} failed:\n' + '\n'.join(dict.fromkeys(map(str, failures.values())))


@pytest.mark.parametrize('game_id', game_ids)
def test_whole_game_replays(results, game_id):
    _assert_passed(results[game_id], ['whole game'])


@pytest.mark.parametrize('game_id', game_ids)
def test_every_or_turn_and_later_sr_replays_alone(results, game_id):
    _assert_passed(results[game_id], ['turns alone'])


@pytest.mark.parametrize('game_id', game_ids)
def test_oracles_agree(results, game_id):
    _assert_passed(results[game_id], ORACLES)


@pytest.mark.skipif(not RUN_FULL_REPLAY, reason='only a full run covers every file')
def test_the_full_run_covered_every_file(results):
    on_disk = sorted(path.stem for path in DATA_DIR.glob('*.json'))
    if not on_disk:
        pytest.skip(f'no scraped games in {DATA_DIR}')
    assert sorted(results) == on_disk
    assert all(set(CHECKS) - {'extraction'} <= set(outcomes) for outcomes in results.values())
    totals = {name: sum(first_count(outcomes[name].counts) for outcomes in results.values() if name in outcomes
                        and not outcomes[name].failure) for name in CHECKS}
    print(f'{len(results)} games, by what each check saw first: {totals}')
    assert all(totals.values())


def test_a_game_ruby_did_not_finish_naturally_fails():
    """An ``hs_*`` fixture ends by hand. A scraped game like it must not pass as parity."""
    with pytest.raises(AssertionError, match='manually_ended'):
        check_whole_game('hs_lhglbbiz_17502', load_fixture_extract('hs_lhglbbiz_17502'))


def test_a_file_that_will_not_load_fails_every_check(tmp_path):
    path = tmp_path / '1.json'
    path.write_text('{"id": 1, "actions": [')
    outcomes = run_game(path)
    assert set(outcomes) == set(CHECKS)
    assert all('JSONDecodeError' in str(outcome.failure) for outcome in outcomes.values())
