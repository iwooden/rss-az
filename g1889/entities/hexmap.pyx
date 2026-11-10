"""MAP entity; see hexmap.pxd."""

from libc.stdint cimport int16_t, uint8_t, uint32_t
from libc.string cimport memset

from g1889.constants cimport OWNER_CORP
from g1889.data.tables cimport (
    HEX_EXIT_MASK, HEX_HOME_CORP, HEX_NEIGHBOR, HEX_RC, MAX_TILE_PATHS, NODE, NUM_CORPS, NUM_HEXES, NUM_SIDES,
    NUM_SUPPLY_TILES, PAIR_A, PAIR_B, PRIVATE_SMR, RC_CITY, RC_D_REVENUE, RC_OFFBOARD, RC_REVENUE, SMR_DISCOUNT,
    TERRAIN_MOUNTAIN, TILE_CANON_ROT, TILE_COST, TILE_MASK, TILE_PATH_COUNT, TILE_PATHS, TILE_RC, TILE_REVENUE,
    TILE_SLOTS, TILE_TERRAIN, UPG_ROT, UPG_START, UPG_TILE,
)
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.corp cimport corp_token_price, corp_tokens_left, invalidate_all_routes, invalidate_routes
from g1889.entities.private cimport private_owner
from g1889.state cimport CORP_FIELDS, GLOBALS, HEX_FIELDS, LAYOUT, GameState

from g1889.entities import bank as _bank

cdef Bank BANK = _bank.BANK


cdef inline int _slot(int hex, int field) noexcept nogil:
    return LAYOUT.hexes_offset + hex * HEX_FIELDS.size + field


cdef int hex_tile(GameState state, int hex) noexcept nogil:
    return <int>state._data[_slot(hex, HEX_FIELDS.tile)]


cdef int hex_rotation(GameState state, int hex) noexcept nogil:
    return <int>state._data[_slot(hex, HEX_FIELDS.rotation)]


cdef int hex_tokens(GameState state, int hex) noexcept nogil:
    return <int>state._data[_slot(hex, HEX_FIELDS.tokens)]


cdef uint32_t hex_paths(GameState state, int hex) noexcept nogil:
    """The path mask of the hex's tile as laid (PAIR bits; data/__init__.py)."""
    return TILE_MASK[hex_tile(state, hex)][hex_rotation(state, hex)]


cdef int supply_count(GameState state, int tile) noexcept nogil:
    return <int>state._data[LAYOUT.supply_offset + tile]


cdef void invalidate_connectivity(GameState state) noexcept nogil:
    """Mark every corp's connectivity and routes stale."""
    state._data[LAYOUT.global_offset + GLOBALS.conn_dirty] = <int16_t>((1 << <int>NUM_CORPS) - 1)
    invalidate_all_routes(state)


cdef inline void _invalidate(GameState state, int mask) noexcept nogil:
    """Mark stale the connectivity and routes of the corps in ``mask``."""
    cdef int slot = LAYOUT.global_offset + GLOBALS.conn_dirty
    state._data[slot] = <int16_t>(state._data[slot] | mask)
    invalidate_routes(state, mask)


# =============================================================================
# WHICH CORPS A HEX TOUCHES
# =============================================================================
#
# A corp's connectivity (``graph.pyx``) and its routes (``routes/solver.pyx``)
# are computed from its own tokens and trains, the game phase, and the tile,
# rotation, and token mask of some hexes. ``map_touch_mask`` bounds which
# hexes: for each corp, the track joined to its tokens. It starts on every
# path at a token's city, and from a path it takes
#
# - at each of the path's sides that is open, the hex across, and every path
#   of that hex at the facing side;
# - at a revenue center the corp may pass (not an offboard, and not a city
#   whose slots hold only other corps' tokens), the center's other paths.
#
# A hex is touched by the corp when one of its paths is taken or it lies
# across an open side of a taken path. This search follows every way out of
# a path and marks nothing off, so it takes every path that GRAPH's walks
# record and that the solver's segments from the corp's reachable centers
# cross, and it reads the tile and tokens only of touched hexes:
#
# - **GRAPH** walks from each token's city, through the paths at the side it
#   crosses and past the centers that do not block, as above but skipping
#   what it has marked. Every hex whose tile or tokens it reads is a token's
#   hex, the hex of a path it walks, or the hex across a side it crosses.
# - **The solver** builds routes from the corp's token cities over segments,
#   passing a center only if it does not block. A segment on such a route
#   runs between two centers over paths joined side to side, so the search
#   above takes its paths from whichever of its ends the corp reaches. The
#   solver also finds segments elsewhere on the map, which no route of the
#   corp can use; they only shift the numbering of the corp's own, whose
#   order among themselves is set by the hexes they cross.
#
# So a change to a hex the corp does not touch, by any corp but itself,
# leaves everything its connectivity and routes are computed from as it was:
# the recomputation would read the same values at every step, touch the
# same hexes, and return the same result. That covers a lay that adds exits
# or city slots and a token that fills a city: whoever could newly reach, or
# be newly stopped at, the hex has track facing it or a path into its center
# already, and so touches it. A corp's own token changes where its walks
# start, so the corp is always marked. The mask is computed on the state
# before the change and needs no cache to be fresh.

cdef inline int _blocked_corps(GameState state, int hex, int tile) noexcept nogil:
    """The corps that may not pass the hex's center (``Node#blocks?``): every
    corp at an offboard, and at a full city those without a token there."""
    cdef int everyone = (1 << <int>NUM_CORPS) - 1
    cdef int tokens
    if TILE_RC[tile] == <int>RC_OFFBOARD:
        return everyone
    tokens = hex_tokens(state, hex)
    return everyone & ~tokens if TILE_RC[tile] == <int>RC_CITY and _count(tokens) >= TILE_SLOTS[tile] else 0


cdef int map_touch_mask(GameState state, int hex) noexcept nogil:
    """The corps whose connectivity or routes may depend on ``hex`` (above)."""
    cdef uint8_t taken[52][6]       # [NUM_HEXES][MAX_TILE_PATHS]: the corps that take each path
    cdef uint8_t touched[52]
    cdef uint8_t queued[52][6]
    cdef int16_t queue[312]         # paths to follow, a ring: each is in it once at most
    cdef int head = 0, count = 0
    cdef int h, k, k2, tile, rotation, pair, end, side, across, facing, tile_across, rotation_across, pair_across
    cdef int corps, onward, entry
    memset(taken, 0, sizeof(taken))
    memset(touched, 0, sizeof(touched))
    memset(queued, 0, sizeof(queued))
    for h in range(<int>NUM_HEXES):
        corps = hex_tokens(state, h)
        if not corps:
            continue
        touched[h] = <uint8_t>corps
        tile = hex_tile(state, h)
        rotation = hex_rotation(state, h)
        for k in range(TILE_PATH_COUNT[tile]):
            if PAIR_B[TILE_PATHS[tile][rotation][k]] == <int>NODE:
                taken[h][k] = <uint8_t>corps
                queued[h][k] = 1
                queue[(head + count) % 312] = <int16_t>(h * <int>MAX_TILE_PATHS + k)
                count += 1
    while count:
        entry = queue[head]
        head = (head + 1) % 312
        count -= 1
        h = entry // <int>MAX_TILE_PATHS
        k = entry % <int>MAX_TILE_PATHS
        queued[h][k] = 0
        corps = taken[h][k]
        tile = hex_tile(state, h)
        rotation = hex_rotation(state, h)
        pair = TILE_PATHS[tile][rotation][k]
        for end in range(2):
            side = PAIR_A[pair] if end == 0 else PAIR_B[pair]
            if side == <int>NODE:
                across, facing, onward = h, <int>NODE, corps & ~_blocked_corps(state, h, tile)
            elif HEX_EXIT_MASK[h] >> side & 1:
                across, facing, onward = HEX_NEIGHBOR[h][side], (side + 3) % <int>NUM_SIDES, corps
                touched[across] |= <uint8_t>corps
            else:
                continue
            tile_across = hex_tile(state, across)
            rotation_across = hex_rotation(state, across)
            for k2 in range(TILE_PATH_COUNT[tile_across]):
                pair_across = TILE_PATHS[tile_across][rotation_across][k2]
                if ((PAIR_A[pair_across] == facing or PAIR_B[pair_across] == facing)
                        and onward & ~taken[across][k2]):
                    taken[across][k2] |= <uint8_t>onward
                    touched[across] |= <uint8_t>onward
                    if not queued[across][k2]:
                        queued[across][k2] = 1
                        queue[(head + count) % 312] = <int16_t>(across * <int>MAX_TILE_PATHS + k2)
                        count += 1
    return touched[hex]


cdef void map_put_tile(GameState state, int hex, int tile, int rotation, int touch) noexcept nogil:
    """Lay supply tile ``tile`` on the hex at the canonical form of
    ``rotation``; the replaced tile returns to the supply unless it is printed."""
    cdef int old = hex_tile(state, hex)
    if old < <int>NUM_SUPPLY_TILES:
        state._data[LAYOUT.supply_offset + old] = <int16_t>(supply_count(state, old) + 1)
    state._data[LAYOUT.supply_offset + tile] = <int16_t>(supply_count(state, tile) - 1)
    state._data[_slot(hex, HEX_FIELDS.tile)] = <int16_t>tile
    state._data[_slot(hex, HEX_FIELDS.rotation)] = TILE_CANON_ROT[tile][rotation]
    _invalidate(state, touch)


cdef void map_put_token(GameState state, int hex, int corp, int touch) noexcept nogil:
    """Put the corp's next charter token in the hex's city: one fewer left
    (CORPS' field, which only placing a token lowers), and its bit in the mask."""
    cdef int left = LAYOUT.corps_offset + corp * CORP_FIELDS.size + CORP_FIELDS.tokens_left
    state._data[left] = <int16_t>(state._data[left] - 1)
    state._data[_slot(hex, HEX_FIELDS.tokens)] = <int16_t>(hex_tokens(state, hex) | 1 << corp)
    _invalidate(state, touch | 1 << corp)


# =============================================================================
# CITIES, REVENUE, TERRAIN
# =============================================================================

cdef inline bint _home_reserved(GameState state, int hex) noexcept nogil:
    """The hex keeps a slot for its home corp, which has not placed its token."""
    cdef int home = HEX_HOME_CORP[hex]
    return home >= 0 and not (hex_tokens(state, hex) >> home & 1)


cdef int map_token_count(GameState state, int hex) noexcept nogil:
    return _count(hex_tokens(state, hex))


cdef int map_free_slots(GameState state, int hex) noexcept nogil:
    """Slots open to any corp without a reservation there (``available_slots``):
    the city's slots less its tokens and an outstanding home reservation."""
    cdef int tile = hex_tile(state, hex)
    if TILE_RC[tile] != <int>RC_CITY:
        return 0
    return TILE_SLOTS[tile] - _count(hex_tokens(state, hex)) - _home_reserved(state, hex)


cdef bint map_has_slot_for(GameState state, int hex, int corp) noexcept nogil:
    """Whether the corp could put a token in the hex's city, connectivity and
    price aside (``City#tokenable?``): no token of its own there, and a free
    slot or its own home reservation."""
    if TILE_RC[hex_tile(state, hex)] != <int>RC_CITY or hex_tokens(state, hex) >> corp & 1:
        return False
    return map_free_slots(state, hex) > 0 or (HEX_HOME_CORP[hex] == corp and _home_reserved(state, hex))


cdef bint map_blocked_for(GameState state, int hex, int corp) noexcept nogil:
    """A city whose slots are all filled by other corps' tokens, which routes
    and connectivity may reach but not pass (``tile_blocks``)."""
    return tile_blocks(hex_tile(state, hex), hex_tokens(state, hex), corp)


cdef int map_revenue(GameState state, int hex, int phase, bint is_d) noexcept nogil:
    """The hex's revenue center value: the tile's for a city or town (0 for
    an empty printed city), the phase's for an offboard, or its D value for a
    D train (D trains run only in phase D)."""
    cdef int tile = hex_tile(state, hex)
    if TILE_RC[tile] != <int>RC_OFFBOARD:
        return TILE_REVENUE[tile]
    return RC_D_REVENUE[HEX_RC[hex]] if is_d else RC_REVENUE[HEX_RC[hex]][phase]


cdef int map_lay_cost(GameState state, int hex, int corp) noexcept nogil:
    """What the corp pays to lay on the hex: the replaced tile's terrain or
    upgrade cost, less SMR's ¥80 on pure-mountain terrain when the corp owns
    SMR (``Game::Base#upgrade_cost``, ``TileDiscount#discounts_tile?``)."""
    cdef int tile = hex_tile(state, hex)
    cdef int cost = TILE_COST[tile]
    if TILE_TERRAIN[tile] == <int>TERRAIN_MOUNTAIN and private_owner(state, PRIVATE_SMR) == <int>OWNER_CORP + corp:
        cost -= min(cost, <int>SMR_DISCOUNT)
    return cost


cdef inline int _check_hex(int hex) except -1:
    assert 0 <= hex < <int>NUM_HEXES, f"hex {hex} out of range"
    return 0


cdef inline int _check_corp(int corp) except -1:
    assert 0 <= corp < <int>NUM_CORPS, f"corp {corp} out of range"
    return 0


cdef bint _upgrades_to(int old, int old_rotation, int tile, int rotation) noexcept:
    """Whether (tile, rotation) is in the upgrade table of (old, old_rotation)."""
    cdef int k = old * <int>NUM_SIDES + old_rotation
    cdef int canon = TILE_CANON_ROT[tile][rotation]
    cdef int i
    for i in range(UPG_START[k], UPG_START[k + 1]):
        if UPG_TILE[i] == tile and UPG_ROT[i] == canon:
            return True
    return False


# =============================================================================
# HANDLE
# =============================================================================

cdef class HexMap:
    """Stateless handle over the hexes block and the tile supply."""

    cpdef int tile(self, GameState state, int hex):
        return hex_tile(state, hex)

    cpdef int rotation(self, GameState state, int hex):
        return hex_rotation(state, hex)

    cpdef void set_tile(self, GameState state, int hex, int tile, int rotation):
        """Put a tile on the hex at the canonical form of ``rotation``. The
        supply is unchanged; a printed tile only returns to its own hex.
        Every corp's caches go stale, as with ``set_tokens``."""
        _check_hex(hex)
        assert 0 <= tile < <int>NUM_SUPPLY_TILES or tile == <int>NUM_SUPPLY_TILES + hex, \
            f"tile {tile} cannot sit on hex {hex}"
        assert 0 <= rotation < <int>NUM_SIDES, f"rotation {rotation} out of range"
        state._data[_slot(hex, HEX_FIELDS.tile)] = <int16_t>tile
        state._data[_slot(hex, HEX_FIELDS.rotation)] = TILE_CANON_ROT[tile][rotation]
        invalidate_connectivity(state)

    cpdef int tokens(self, GameState state, int hex):
        return hex_tokens(state, hex)

    cpdef void set_tokens(self, GameState state, int hex, int mask):
        _check_hex(hex)
        assert 0 <= mask < 1 << <int>NUM_CORPS, f"token mask {mask} out of range"
        state._data[_slot(hex, HEX_FIELDS.tokens)] = <int16_t>mask
        invalidate_connectivity(state)

    cpdef int supply(self, GameState state, int tile):
        return supply_count(state, tile)

    cpdef void set_supply(self, GameState state, int tile, int count):
        assert 0 <= tile < <int>NUM_SUPPLY_TILES and count >= 0, f"{count} copies of tile {tile}"
        state._data[LAYOUT.supply_offset + tile] = <int16_t>count

    cpdef int conn_dirty(self, GameState state):
        """Corp mask of stale connectivity caches."""
        return <int>state._data[LAYOUT.global_offset + GLOBALS.conn_dirty]

    cpdef int touch_mask(self, GameState state, int hex):
        """Corp mask of those whose connectivity or routes may depend on the
        hex (``map_touch_mask``): the corps a lay or a token there marks stale."""
        _check_hex(hex)
        return map_touch_mask(state, hex)

    # Queries

    cpdef int free_slots(self, GameState state, int hex):
        _check_hex(hex)
        return map_free_slots(state, hex)

    cpdef bint has_slot_for(self, GameState state, int hex, int corp):
        _check_hex(hex)
        _check_corp(corp)
        return map_has_slot_for(state, hex, corp)

    cpdef bint blocked_for(self, GameState state, int hex, int corp):
        _check_hex(hex)
        _check_corp(corp)
        return map_blocked_for(state, hex, corp)

    cpdef int revenue(self, GameState state, int hex, int phase, bint is_d=False):
        _check_hex(hex)
        return map_revenue(state, hex, phase, is_d)

    cpdef int lay_cost(self, GameState state, int hex, int corp):
        _check_hex(hex)
        _check_corp(corp)
        return map_lay_cost(state, hex, corp)

    # Mutations. Upgrades keep the hex's tokens: the token mask stays, and a
    # 1889 upgrade never loses slots.

    cpdef void lay(self, GameState state, int hex, int tile, int rotation):
        """Lay supply tile ``tile`` on the hex at ``rotation`` (any rotation;
        the canonical one is stored), returning the replaced tile to the supply
        unless it is printed (``Game::Base#update_tile_lists``). The lay must
        be in the upgrade table; legality beyond it is the caller's."""
        _check_hex(hex)
        cdef int old = hex_tile(state, hex)
        assert 0 <= tile < <int>NUM_SUPPLY_TILES and supply_count(state, tile) > 0, f"tile {tile} is not in supply"
        assert 0 <= rotation < <int>NUM_SIDES and _upgrades_to(old, hex_rotation(state, hex), tile, rotation), \
            f"tile {tile} at rotation {rotation} does not upgrade hex {hex}"
        assert _count(hex_tokens(state, hex)) <= TILE_SLOTS[tile], f"tile {tile} cannot hold hex {hex}'s tokens"
        map_put_tile(state, hex, tile, rotation, map_touch_mask(state, hex))

    cpdef void lay_track(self, GameState state, int corp, int hex, int tile, int rotation):
        """The corp lays the tile and pays the lay cost, priced before the lay, to the bank."""
        _check_corp(corp)
        cdef int cost = self.lay_cost(state, hex, corp)
        self.lay(state, hex, tile, rotation)
        if cost:
            BANK.transfer(state, <int>OWNER_CORP + corp, <int>BANK_ACCOUNT, cost)

    cpdef void place_token(self, GameState state, int hex, int corp):
        """Put the corp's next charter token in the hex's city, unpaid: a slot
        must be open to it (its home reservation counts)."""
        _check_hex(hex)
        _check_corp(corp)
        assert corp_tokens_left(state, corp) > 0, f"corp {corp} has no tokens left"
        assert map_has_slot_for(state, hex, corp), f"corp {corp} cannot token hex {hex}"
        map_put_token(state, hex, corp, map_touch_mask(state, hex))

    cpdef void buy_token(self, GameState state, int hex, int corp):
        """The corp places its next token and pays its price (¥0 for the home
        token) to the bank (``Tokener#place_token``)."""
        _check_corp(corp)
        cdef int price = corp_token_price(state, corp)
        self.place_token(state, hex, corp)
        if price:
            BANK.transfer(state, <int>OWNER_CORP + corp, <int>BANK_ACCOUNT, price)


# map_touch_mask's arrays are sized with literals, and its corp masks are 8 bits.
assert <int>NUM_HEXES == 52 and <int>MAX_TILE_PATHS == 6 and <int>NUM_CORPS <= 8

MAP = HexMap()
