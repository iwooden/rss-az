"""MAP entity; see hexmap.pxd."""

from libc.stdint cimport int16_t, uint32_t

from g1889.constants cimport OWNER_CORP
from g1889.data.tables cimport (
    HEX_HOME_CORP, HEX_RC, NUM_CORPS, NUM_HEXES, NUM_SIDES, NUM_SUPPLY_TILES, PRIVATE_SMR,
    RC_CITY, RC_D_REVENUE, RC_OFFBOARD, RC_REVENUE, SMR_DISCOUNT, TERRAIN_MOUNTAIN, TILE_CANON_ROT, TILE_COST,
    TILE_MASK, TILE_RC, TILE_REVENUE, TILE_SLOTS, TILE_TERRAIN, UPG_ROT, UPG_START, UPG_TILE,
)
from g1889.entities.bank cimport BANK_ACCOUNT, Bank
from g1889.entities.corp cimport Corporation, corp_token_price, corp_tokens_left, invalidate_all_routes
from g1889.entities.private cimport private_owner
from g1889.state cimport GLOBALS, HEX_FIELDS, LAYOUT, GameState

from g1889.entities import bank as _bank, corp as _corp

cdef Bank BANK = _bank.BANK
cdef list CORPS = _corp.CORPS


cdef inline int _slot(int hex, int field) noexcept nogil:
    return LAYOUT.hexes_offset + hex * HEX_FIELDS.size + field


cdef inline int _count(int mask) noexcept nogil:
    cdef int n = 0
    while mask:
        mask &= mask - 1
        n += 1
    return n


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
    state._data[LAYOUT.global_offset + GLOBALS.conn_dirty] = <int16_t>((1 << <int>NUM_CORPS) - 1)
    invalidate_all_routes(state)


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
    and connectivity may reach but not pass (``City#blocks?``). Reservations
    never block."""
    cdef int tile = hex_tile(state, hex)
    cdef int tokens = hex_tokens(state, hex)
    return TILE_RC[tile] == <int>RC_CITY and not (tokens >> corp & 1) and _count(tokens) >= TILE_SLOTS[tile]


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
        supply is unchanged; a printed tile only returns to its own hex."""
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
        if old < <int>NUM_SUPPLY_TILES:
            self.set_supply(state, old, supply_count(state, old) + 1)
        self.set_supply(state, tile, supply_count(state, tile) - 1)
        self.set_tile(state, hex, tile, rotation)

    cpdef void lay_track(self, GameState state, int corp, int hex, int tile, int rotation):
        """The corp pays the lay cost to the bank and lays the tile."""
        _check_corp(corp)
        cdef int cost = self.lay_cost(state, hex, corp)
        if cost:
            BANK.transfer(state, <int>OWNER_CORP + corp, <int>BANK_ACCOUNT, cost)
        self.lay(state, hex, tile, rotation)

    cpdef void place_token(self, GameState state, int hex, int corp):
        """Put the corp's next charter token in the hex's city, unpaid: a slot
        must be open to it (its home reservation counts)."""
        _check_hex(hex)
        _check_corp(corp)
        cdef int left = corp_tokens_left(state, corp)
        assert left > 0, f"corp {corp} has no tokens left"
        assert map_has_slot_for(state, hex, corp), f"corp {corp} cannot token hex {hex}"
        (<Corporation>CORPS[corp]).set_tokens_left(state, left - 1)
        self.set_tokens(state, hex, hex_tokens(state, hex) | 1 << corp)

    cpdef void buy_token(self, GameState state, int hex, int corp):
        """The corp pays its next token's price (¥0 for the home token) to the
        bank and places it (``Tokener#place_token``)."""
        _check_corp(corp)
        cdef int price = corp_token_price(state, corp)
        if price:
            BANK.transfer(state, <int>OWNER_CORP + corp, <int>BANK_ACCOUNT, price)
        self.place_token(state, hex, corp)


MAP = HexMap()
