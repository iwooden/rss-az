"""Ruby state extracts for differential replay of Shikoku 1889.

``extract_states.rb``, next to this module, replays an 18xx.games game through
the Ruby engine in ``submodules/18xx`` and records Ruby's state after every
action (1889/engine-design.md section 10.2). This module runs it, caches the
output, and returns plain JSON data: dicts, lists, strings, ints, bools, None.
It is test tooling: when an extract has to be made and ``ruby`` is missing, it
calls ``pytest.skip``. Returned extracts are shared between callers, so treat
them as read-only.

Ruby is the oracle. The extractor records what Ruby does, quirks included,
and corrects nothing.

Caching
-------
Extracts are written to ``.extract_cache/`` here (gitignored), never into the
submodule. A cached extract is reused only while its key and the engine both
match:

- The file name carries a hash of the schema version, the extractor's source,
  and the input (the fixture's bytes, the ``--new-game`` arguments, or a test's
  game JSON). Editing the extractor or the input therefore makes a new
  extract, and the stale file is deleted when the new one is written.
- The extract must be newer than every engine Ruby file the 1889 game can load:
  ``lib/engine.rb`` and ``lib/engine/**/*.rb``, except other titles'
  ``game/g_*`` code. A submodule update or local edit rewrites those files, so
  their mtimes move past the cache.

Extract
-------
``schema``
    ``SCHEMA_VERSION``.
``source``
    ``kind`` (``'fixture'`` or ``'new_game'``), ``seed``, ``optional_rules``,
    and for fixtures ``file`` and ``id``.
``players``
    ``[{id, name}]`` in canonical order: the seat order of the game JSON, which
    is Ruby's initial ``@players`` order. Engine player index i is entry i.
    Ruby later rotates ``@players`` (``state['game']['player_order']`` shows
    it); every per-player list below stays in canonical order.
``raw_action_count``, ``surviving_action_count``
    Actions in the log, and those that survive undo and redo: the non-nil
    entries of Ruby's ``Game::Base.filtered_actions``, including messages,
    ``program_*`` actions, and ``end_game``.
``records``
    ``records[0]`` is ``{index: 0, action: None, state}``, the state before
    any action. Each later record is one surviving action, in log order:
    ``{index, action, pre, state}``, plus ``auto_pre`` when the action has
    ``auto_actions``.

    ``action``
        The log entry verbatim, ``auto_actions`` included.
    ``pre``
        Annotations taken just before the action ran (see Annotations).
    ``auto_pre``
        One annotation per ``action['auto_actions']`` entry, each taken just
        before that auto action ran, after the host and earlier auto actions.
    ``state``
        The snapshot after the action and all its auto actions.
``full_load``
    Fixtures only. ``processed_action_ids`` and ``state`` from an independent
    ``Engine::Game.load`` of the whole file: Ruby's own resolution, which must
    equal the record ids and the last record's state.

References (``REF``) name an entity: ``{type: 'player', id: <player id>}``,
``{type: 'corporation', id: 'AR'}``, ``{type: 'company', id: 'ER'}``, or
``{type: 'pool' | 'bank' | 'depot'}``; None means no entity. Player ids are
the game JSON's (ints here). Every other id is an 18xx string (hex ``K4``,
tile ``57``, corporation ``AR``, private ``TR``, train type ``D``, train id
``3-0``); map them through ``g1889.data.load_tables()``. Corporation TR and private TR share a symbol, so
``type`` or the enclosing key tells them apart.

Annotations (``pre``)
---------------------
Every annotation has ``PRE_KEYS``:

``type``
    The action type.
``round``, ``round_num``, ``turn``, ``step``, ``current_entity``, ``active_players``
    As in ``state['game']``, before the action.
``handler``
    The step class Ruby hands the action to. It differs from ``step`` when a
    non-blocking step acts: ``SpecialTrack`` for a private's tile lay during
    Track, ``BuyCompany`` during BuyTrain, ``Bankrupt``, ``Program`` for
    ``program_*``, and ``EndGame``.

``PRE_EXTRA_KEYS`` adds, by action type:

``bid``
    ``company``, ``price``; ``topmost``, the cheapest unsold private;
    ``auctioning``, the private in a closed auction (two or more bids), or
    None; ``min_bid`` for the bid's private; ``purchase``, True when the bid
    buys the topmost private outright.
``par``
    ``corporation``, ``par_price``.
``buy_shares``
    ``source`` (``'ipo'``, ``'pool'``, or ``'player'``), ``corporation``,
    ``percent``, ``price`` (the bundle's cost). A buy by company DR is its
    exchange.
``sell_shares``
    ``corporation``, ``percent``, ``count`` (certificates), ``price``.
``lay_tile``
    ``hex``; ``tile``, the tile name without the copy suffix of ``57-1``;
    ``rotation`` as logged (not canonical); ``old_tile``, ``old_rotation``.
``place_token``
    ``hex``; ``city``, the city's index on its tile (always 0 in 1889).
``run_routes``
    ``revenue``: what Ruby pays. ``Action::RunRoutes`` hands a route's logged
    revenue, if any, to ``Route.new``, which caches it, so Ruby checks and
    computes only routes logged without one. 314 logs ``hexes`` and
    ``revenue`` (Ruby pays the logged sum unchecked); 962 logs
    ``connections`` only (Ruby computes). ``trains``, the train ids run.
    ``recomputed_revenue`` and ``recomputed_routes`` (per route): Ruby's
    computation of the logged routes with any logged revenue dropped, or
    ``recomputed_revenue`` None with ``recomputed_error`` when it fails or a
    route has no ``connections`` to rebuild it from (314). ``max_revenue``,
    ``max_routes`` (``[{train, revenue, hexes}]``), and ``max_timed_out``:
    AutoRouter's best combination, from its Ruby path walk and a port of its
    JavaScript combination search.
``dividend``
    ``revenue``: the revenue of this turn's routes, the amount being paid or
    withheld.
``buy_train``
    ``train_type``; ``source`` (``'depot'``, ``'pool'``, or ``'corporation'``);
    ``price``; ``seller``, the selling corporation, for a corporation source;
    ``exchange_type``, the train type traded in, for an exchange.
``discard_train``
    ``train_type``.
``buy_company``
    ``company``, ``seller`` (``REF``), ``price``.
``bankrupt``
    ``president``, the bankrupt player's id.

Snapshot (``state``)
--------------------
``STATE_KEYS``:

``game`` (``GAME_KEYS``)
    ``phase`` (``'2'`` to ``'D'``), ``train_limit``, ``tile_colors``;
    ``round`` (``'ISR'`` for the private auction, ``'SR'``, ``'OR'``);
    ``round_num``, the OR's number within its set (1 outside ORs); ``turn``,
    Ruby's turn, the SR number; ``or_count``, the ORs in the current set;
    ``step``, the round's active step class (it stays set once the game is
    over); ``current_entity`` (``REF``) and ``active_players`` (player ids),
    who Ruby waits on; ``round_entities``, the ids the round iterates (players
    in the ISR and SR, corporations in operating order in an OR), and
    ``entity_index`` into them; ``priority``, Ruby's priority deal player;
    ``player_order``, Ruby's rotated ``@players``; ``bank_cash``,
    ``bank_broken``; ``finished``; ``end_reason`` (``'bankrupt'``,
    ``'bank'``, ``'manually_ended'``, or None); ``result``, ``{str(player
    id): value}`` once finished (the form of a fixture's ``result``), else None.
``players`` (``PLAYER_KEYS``), a list in canonical order
    ``id``, ``name``, ``cash``; ``value``, Ruby's ``player_value`` (cash, shares
    of IPO'd corporations at market, privates at face); ``shares``,
    ``{corporation: {percent, president}}`` for corporations held, in
    corporation order; ``privates``; ``bids``, ``{private: price}`` for open
    auction bids; ``passed`` (Ruby's pass flag, which the auction and SR use);
    ``bankrupt``. A bankrupt player keeps the shares Ruby could not sell.
``corporations`` (``CORPORATION_KEYS``), keyed by symbol in corporation order
    ``ipoed``; ``par`` and ``price`` (None until parred); ``market``, ``[row,
    column, stack]``, where ``stack`` is the corporation's index in Ruby's
    cell list (0 first in operating order among corporations at that cell);
    ``floated``; ``cash``; ``ipo_percent``, ``pool_percent``; ``president``, a
    player id or None; ``tokens_left``; ``token_hexes``, the hexes of placed
    tokens in charter order; ``trains`` (``{type: count}``) and ``train_ids``;
    ``privates``; ``operated`` (has an operating history); ``last_revenue``
    and ``last_dividend`` (``'payout'`` or ``'withhold'``) from the latest
    operating history entry, else None; ``closed``.
``privates`` (``PRIVATE_KEYS``), keyed by symbol
    ``owner`` (``REF``, None while unsold or closed), ``closed``, ``value``,
    ``revenue`` (UTF's rises to 50 in phase 5), ``discount`` (the auction
    discount), ``abilities`` (the ability types left). Privates not in play
    are absent: UTF below 4 players, SIR at 2.
``hexes`` (``HEX_KEYS``), all 52, keyed by hex id
    ``tile``, the tile name (a printed tile is named by its hex id);
    ``rotation`` as Ruby stores it (not canonical); ``tokens``, corporation
    symbols in slot order; ``reserved``, corporations whose home reservation
    is still unfilled.
``tiles``
    ``{name: copies left in the supply}`` for every supply tile name.
``depot``, ``pool``
    ``{train type: count}`` for unsold trains (Ruby keeps one D at a time and
    clones it on each sale; D is unlimited) and for the open market.
``auction``
    None outside the ISR, else ``companies`` (unsold privates, cheapest
    first), ``auctioning`` (as in ``pre``), and ``min_bids``
    (``{private: minimum bid}``).
``stock``
    None outside SRs, else ``sold`` (``{str(player id): {corporation: 'now' |
    'prev'}}``, sales this SR, ``'now'`` for this turn), ``current_actions``
    (the action types of this turn), ``bought_from_ipo``, ``last_to_act``,
    ``pass_order``, and ``legal``: the turn player's options in
    ``BuySellParShares``, or None when another step is active. ``legal`` holds
    ``actions``, Ruby's action types for the player (``['sell_shares']`` alone
    on a must-sell turn); ``buy_ipo``, ``buy_pool``, and ``par``, the
    corporations each can take; and ``sell``, ``{corporation: [percent of
    each sellable bundle]}``.
``operating``
    None outside ORs, else ``operator``; ``routes_revenue``, the revenue of
    this turn's routes once run, else None; ``ebuy_sold`` (corporations the
    president has sold in this emergency) and ``ebuy_last_share_price``.
``round_state``
    Ruby's round state variables, converted generically. Diagnostics only:
    its keys vary with the round, so compare the fields above instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

SCHEMA_VERSION = 2

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXTRACTOR = HERE / 'extract_states.rb'
CACHE_DIR = HERE / '.extract_cache'
SUBMODULE = REPO_ROOT / 'submodules' / '18xx'
FIXTURE_DIR = SUBMODULE / 'public' / 'fixtures' / '1889'
FIXTURES = ('314', '962', 'hs_jmmljkbw_17502', 'hs_lhglbbiz_17502')
TIMEOUT_SECONDS = 600

STATE_KEYS = frozenset({
    'game', 'players', 'corporations', 'privates', 'hexes', 'tiles', 'depot', 'pool', 'auction', 'stock',
    'operating', 'round_state',
})
GAME_KEYS = frozenset({
    'phase', 'train_limit', 'tile_colors', 'round', 'round_num', 'turn', 'or_count', 'step', 'current_entity',
    'active_players', 'round_entities', 'entity_index', 'priority', 'player_order', 'bank_cash', 'bank_broken',
    'finished', 'end_reason', 'result',
})
PLAYER_KEYS = frozenset({'id', 'name', 'cash', 'value', 'shares', 'privates', 'bids', 'passed', 'bankrupt'})
CORPORATION_KEYS = frozenset({
    'ipoed', 'par', 'price', 'market', 'floated', 'cash', 'ipo_percent', 'pool_percent', 'president',
    'tokens_left', 'token_hexes', 'trains', 'train_ids', 'privates', 'operated', 'last_revenue', 'last_dividend',
    'closed',
})
PRIVATE_KEYS = frozenset({'owner', 'closed', 'value', 'revenue', 'discount', 'abilities'})
HEX_KEYS = frozenset({'tile', 'rotation', 'tokens', 'reserved'})
AUCTION_KEYS = frozenset({'companies', 'auctioning', 'min_bids'})
STOCK_KEYS = frozenset({'sold', 'current_actions', 'bought_from_ipo', 'last_to_act', 'pass_order', 'legal'})
OPERATING_KEYS = frozenset({'operator', 'routes_revenue', 'ebuy_sold', 'ebuy_last_share_price'})

PRE_KEYS = frozenset({
    'type', 'round', 'round_num', 'turn', 'step', 'handler', 'current_entity', 'active_players',
})
PRE_EXTRA_KEYS: dict[str, frozenset[str]] = {
    'bid': frozenset({'company', 'price', 'topmost', 'auctioning', 'min_bid', 'purchase'}),
    'par': frozenset({'corporation', 'par_price'}),
    'buy_shares': frozenset({'source', 'corporation', 'percent', 'price'}),
    'sell_shares': frozenset({'corporation', 'percent', 'count', 'price'}),
    'lay_tile': frozenset({'hex', 'tile', 'rotation', 'old_tile', 'old_rotation'}),
    'place_token': frozenset({'hex', 'city'}),
    'run_routes': frozenset({'revenue', 'trains', 'recomputed_revenue', 'max_revenue', 'max_routes', 'max_timed_out'}),
    'dividend': frozenset({'revenue'}),
    'buy_train': frozenset({'train_type', 'source', 'price'}),
    'discard_train': frozenset({'train_type'}),
    'buy_company': frozenset({'company', 'seller', 'price'}),
    'bankrupt': frozenset({'president'}),
}
# Keys present only in some annotations of a type.
PRE_OPTIONAL_KEYS: dict[str, frozenset[str]] = {
    'run_routes': frozenset({'recomputed_routes', 'recomputed_error'}),
    'buy_train': frozenset({'seller', 'exchange_type'}),
}

Extract = dict[str, Any]

_loaded: dict[Path, Extract] = {}


def ruby() -> str | None:
    return shutil.which('ruby')


def fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f'{name}.json'


def load_fixture(name: str) -> dict[str, Any]:
    """The fixture's game JSON."""
    return json.loads(fixture_path(name).read_text())


def load_fixture_extract(name: str) -> Extract:
    """The extract of fixture ``name`` (a stem of ``FIXTURES`` or another file in ``FIXTURE_DIR``)."""
    path = fixture_path(name)
    return _load(f'fixture-{name}', [str(path)], path.read_bytes())


def load_new_game_extract(num_players: int, seed: int = 1, player_ids: Sequence[int | str] | None = None) -> Extract:
    """The extract of a fresh ``num_players`` game: one record, the initial snapshot.

    ``player_ids`` gives the seat order (default 1..num_players). Ruby's first
    player is always the first seat, so compare against
    ``initialize_game(..., priority_player=0)``.
    """
    args = ['--new-game', str(num_players), '--seed', str(seed)]
    if player_ids is not None:
        args += ['--players', ','.join(str(p) for p in player_ids)]
    inputs = ' '.join(args).encode()
    return _load(f'new-game-{num_players}-{hashlib.sha256(inputs).hexdigest()[:8]}', args, inputs)


def load_game_extract(label: str, game: dict[str, Any]) -> Extract:
    """The extract of a game JSON built by a test, such as a scripted auction,
    cached under ``label`` like a fixture's."""
    inputs = json.dumps(game, sort_keys=True).encode()
    CACHE_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix='.game.json')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(inputs)
        return _load(f'game-{label}', [tmp], inputs)
    finally:
        os.unlink(tmp)


def _load(label: str, args: list[str], inputs: bytes) -> Extract:
    """Reads ``label``'s cached extract, first running the extractor with ``args`` if the cache is stale."""
    digest = hashlib.sha256()
    for part in (str(SCHEMA_VERSION).encode(), EXTRACTOR.read_bytes(), inputs):
        digest.update(hashlib.sha256(part).digest())
    path = CACHE_DIR / f'{label}-{digest.hexdigest()[:16]}.json'

    if path in _loaded:
        return _loaded[path]
    if not (path.exists() and path.stat().st_mtime_ns >= _engine_mtime_ns()):
        _extract(label, path, args)
    _loaded[path] = extract = json.loads(path.read_text())
    return extract


def _extract(label: str, path: Path, args: list[str]) -> None:
    exe = ruby()
    if exe is None:
        pytest.skip('requires ruby')
    result = subprocess.run([exe, str(EXTRACTOR), *args], capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                            cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f'extract_states.rb {" ".join(args)} failed:\n{result.stderr}')

    CACHE_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        f.write(result.stdout)
    os.replace(tmp, path)
    for stale in CACHE_DIR.glob(f'{label}-*.json'):
        if stale != path and stale.name.rsplit('-', 1)[0] == label:
            stale.unlink()


def _engine_sources() -> Iterator[Path]:
    lib = SUBMODULE / 'lib'
    yield lib / 'engine.rb'
    engine = lib / 'engine'
    for root, dirs, files in os.walk(engine):
        if Path(root) == engine / 'game':
            dirs[:] = [d for d in dirs if not d.startswith('g_') or d == 'g_1889']
            files = [f for f in files if not f.startswith('g_') or f == 'g_1889.rb']
        yield from (Path(root) / f for f in files if f.endswith('.rb'))


@cache
def _engine_mtime_ns() -> int:
    return max(p.stat().st_mtime_ns for p in _engine_sources())
