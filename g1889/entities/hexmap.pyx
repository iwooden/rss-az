"""MAP entity; see hexmap.pxd."""

from libc.stdint cimport int16_t

from g1889.data.tables cimport NUM_CORPS, NUM_HEXES, NUM_SIDES, NUM_SUPPLY_TILES, TILE_CANON_ROT
from g1889.entities.corp cimport invalidate_all_routes
from g1889.state cimport GLOBALS, HEX_FIELDS, LAYOUT, GameState


cdef inline int _slot(int hex, int field) noexcept nogil:
    return LAYOUT.hexes_offset + hex * HEX_FIELDS.size + field


cdef int hex_tile(GameState state, int hex) noexcept nogil:
    return <int>state._data[_slot(hex, HEX_FIELDS.tile)]


cdef int hex_rotation(GameState state, int hex) noexcept nogil:
    return <int>state._data[_slot(hex, HEX_FIELDS.rotation)]


cdef int hex_tokens(GameState state, int hex) noexcept nogil:
    return <int>state._data[_slot(hex, HEX_FIELDS.tokens)]


cdef int supply_count(GameState state, int tile) noexcept nogil:
    return <int>state._data[LAYOUT.supply_offset + tile]


cdef void invalidate_connectivity(GameState state) noexcept nogil:
    state._data[LAYOUT.global_offset + GLOBALS.conn_dirty] = <int16_t>((1 << <int>NUM_CORPS) - 1)
    invalidate_all_routes(state)


cdef inline int _check_hex(int hex) except -1:
    assert 0 <= hex < <int>NUM_HEXES, f"hex {hex} out of range"
    return 0


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


MAP = HexMap()
