"""Static data for the Shikoku 1889 engine.

``tables.json`` holds the map, tiles, upgrade table, and game constants. It is
generated from the 18xx.games Ruby engine in ``submodules/18xx`` by
``generate.py``, which then renders it as the C arrays and enums of
``tables.pxd``/``tables.pyx`` (module ``g1889.data.tables``). Do not edit any
of the three by hand. Regenerate with ``.venv/bin/python -m g1889.data.generate``,
or ``--cython-only`` to redo the Cython from the checked-in JSON without Ruby.
The engine cimports ``g1889.data.tables`` and never reads the JSON; Python
tests and tools use ``load_tables()``.

Conventions
-----------
Hexes
    Flat-topped, with 18xx.games ids: column letters A-L run west to east and
    row numbers 1-14 north to south. Hexes are indexed 0-51 sorted by (column,
    row): A8, A10, B3, ... ``x``/``y`` are the Ruby engine's doubled
    coordinates (``x`` = column index, ``y`` = row - 1).

Sides
    The 18xx engine's flat-layout numbering, so its tile path codes stay valid.
    Side 0 is the south edge and numbers run clockwise:

        side      0 S     1 SW     2 NW     3 N      4 NE    5 SE
        (dx, dy)  (0, 2)  (-1, 1)  (-1, -1) (0, -2)  (1, -1) (1, 1)

    Side s of a hex meets side ``(s + 3) % 6`` of the hex across it. Rotating a
    tile by r moves its side s to ``(s + r) % 6``; the node does not move.
    ``neighbors[s]`` is the hex across side s, or -1 off the map. ``exit_mask``
    has bit s set when track may leave through side s: the side is on the map
    and does not face a gray or red hex side without track (Ruby
    ``Hex#neighbors``). ``ruby_order`` is the hex's place in Ruby's
    ``Game#hexes``, the order in which its graph walks a corporation's tokens
    (``HEX_WALK_ORDER``).

Endpoints and paths
    Endpoints 0-5 are the sides and 6 (``NODE``) is the hex's revenue node.
    Every 1889 hex and tile has at most one revenue center. A path joins two
    endpoints. The 21 unordered pairs are numbered lexicographically,
    (0,1)=0, (0,2)=1, ..., (0,6)=5, (1,2)=6, ..., (5,6)=20 (``PAIRS``), and a
    tile's mask at rotation r sets bit ``PAIR_INDEX[a][b]`` for each rotated
    path. Paths are stored raw, without transitive closure: crossings (16, 19,
    20) join no tracks, and a route cannot pass from one branch of a junction
    tile (23-29, 39-47) into another, because it would reverse at the shared
    side. A tile's ``paths`` keep Ruby's order (``Tile#paths``), which its
    path walk follows; ``TILE_PATHS`` holds them as pair indices by rotation,
    padded with 255.

Tiles
    Ids 0-39 are the supply tiles in Ruby ``Tile#<=>`` order (color, then
    number). Ids 40-91 are preprinted map tiles: hex h starts with tile
    ``40 + h``. Preprinted tiles are used only at rotation 0, have count 0, and
    never return to the supply. ``revenue``/``slots`` describe the tile's
    revenue center; offboard tiles carry revenue 0, and their per-phase revenue
    is in ``revenue_centers``. ``cost`` is the terrain or upgrade cost of
    replacing the tile; ``terrain`` bits follow ``TERRAINS``. Private SMR's
    discount applies only when ``terrain`` is exactly mountain.

Revenue centers
    One row per printed revenue center (25), in hex order; a hex's ``rc``
    indexes it, or is -1. ``revenue`` is indexed by phase for trains other
    than D, and ``d_revenue`` is the D-train value. For cities and towns these
    are the printed values; the hex's current tile supplies revenue and slots
    once one is laid.

Upgrades
    ``upgrades`` rows map ``from: [tile, rotation]`` to every legal
    ``[tile, rotation]`` that may replace it on a hex whose sides are all
    open. On hex h, a target is legal only if its exits fit ``exit_mask[h]``.
    The table ignores phase colors, supply counts, connectivity, blockers, and
    cost. ``blocks_lay`` tiles (437) are laid only by a private ability. The C
    tables store it as a CSR over ``6 * tile + rotation`` (``UPG_START``,
    ``UPG_TILE``, ``UPG_ROT``) with canonical targets only: rotations with equal
    masks collapse to the lowest (``TILE_CANON_ROT``).

Other conventions
    Corporations, privates (A-G by face value), trains, and phases keep
    Ruby's order. Train ``count`` -1 means unlimited. ``rusts_on``, ``on``,
    and ``available_on`` are train or phase indices, or -1. Privates close
    through the 5-train's ``close_companies`` event, which fires with the
    first 5-train, as phase 5 begins. Private abilities are Ruby's parameters
    with hexes, tiles, and corporations as indices. ``players`` gives cash,
    certificate limit, and privates in play for 2-6 players; ``rules`` holds
    Ruby rule constants by name. Enum-valued fields are indices into the
    tuples below, which ``tables.json`` repeats under ``enums``.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

TABLES_PATH = Path(__file__).with_name('tables.json')

NUM_SIDES = 6
NODE = 6
NUM_ENDPOINTS = 7
PAIRS = tuple((a, b) for a in range(NUM_ENDPOINTS) for b in range(a + 1, NUM_ENDPOINTS))
NUM_PAIRS = len(PAIRS)
PAIR_INDEX = tuple(
    tuple(PAIRS.index((min(a, b), max(a, b))) if a != b else -1 for b in range(NUM_ENDPOINTS))
    for a in range(NUM_ENDPOINTS)
)

COLORS = ('white', 'yellow', 'green', 'brown', 'gray', 'red')
RC_TYPES = ('none', 'city', 'town', 'offboard')
LABELS = ('', 'H', 'K', 'T')
TERRAINS = ('mountain', 'water')
MARKET_TYPES = ('', 'par', 'no_cert_limit', 'unlimited')


def rotate_endpoint(endpoint: int, rotation: int) -> int:
    return endpoint if endpoint == NODE else (endpoint + rotation) % NUM_SIDES


def path_mask(paths, rotation: int = 0) -> int:
    mask = 0
    for a, b in paths:
        mask |= 1 << PAIR_INDEX[rotate_endpoint(a, rotation)][rotate_endpoint(b, rotation)]
    return mask


def mask_exits(mask: int) -> int:
    """Side bitmask of the sides any path in ``mask`` uses."""
    exits = 0
    for i, (a, b) in enumerate(PAIRS):
        if mask >> i & 1:
            exits |= (1 << a) | (1 << b if b != NODE else 0)
    return exits


@cache
def load_tables() -> dict[str, Any]:
    with TABLES_PATH.open() as f:
        return json.load(f)
