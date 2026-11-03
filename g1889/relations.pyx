"""Relation extraction for 1889; see relations.pxd and 1889/extraction-design.md §6.

``_emit`` writes a state's records once, and each form copies them out: the
sparse fills emit into their buffer and zero its tail, and the dense fills
emit into a stack buffer and scatter it into zeroed planes. Records come
entity by entity, each pair's two directions together. No (relation, query,
key) repeats and no value is 0, so the dense planes' nonzero count is the
record count.

Relations 21-24, the candidates', are sub-step 7.2's: their ids are
reserved, and nothing emits them yet.
"""

import numpy as np

from libc.stdint cimport uint32_t
from libc.string cimport memset

from g1889.constants cimport OWNER_CORP, OWNER_PLAYER
from g1889.data.tables cimport (
    CORP_HOME, ENDPOINT_PAIRS, HEX_BLOCKER, HEX_EXIT_MASK, HEX_NEIGHBOR, MF_PORT_HEX, NUM_CORPS, NUM_HEXES,
    NUM_PORTS, NUM_PRIVATES, NUM_SIDES, NUM_TRAIN_TYPES, PRIVATE_MF,
)
from g1889.entities.corp cimport corp_president
from g1889.entities.graph cimport GRAPH_NODE, GRAPH_SIDES
from g1889.entities.hexmap cimport hex_paths, hex_tokens
from g1889.entities.player cimport player_bid, player_shares
from g1889.entities.private cimport private_owner
from g1889.entities.trains cimport corp_train_count
from g1889.entities.turn cimport turn_num_players
from g1889.state cimport GameState
from g1889.token_data cimport (
    ROW_CORP, ROW_HEX, ROW_PLAYER, ROW_PRIVATE, ROW_TRAIN, bind_row, extraction_prologue, get_num_tokens,
    graph_entry, resolve_max_players,
)


cdef struct Record:
    unsigned char relation
    unsigned char query
    unsigned char key
    unsigned char value


# =============================================================================
# PUBLIC API
# =============================================================================

cpdef void get_relation_data(GameState state, unsigned char[:, :, ::1] planes, int max_players=0):
    """Fill ``planes``, ``(NUM_RELATIONS, T, T)`` for T =
    ``get_num_tokens(max_players)``, for ``state``; ``max_players`` 0 is the
    state's player count."""
    cdef int num_players = turn_num_players(state)
    cdef int num_tokens, count
    max_players = resolve_max_players(num_players, max_players, "get_relation_data")
    num_tokens = get_num_tokens(max_players)
    assert planes.shape[0] == <int>NUM_RELATIONS and planes.shape[1] == planes.shape[2] == num_tokens, \
        f"get_relation_data: planes {(planes.shape[0], planes.shape[1], planes.shape[2])} for {num_tokens} tokens"
    with nogil:
        extraction_prologue(state)
        count = _fill_planes(state, planes, num_players)
    _check_count(count, "get_relation_data")


cpdef void get_relation_data_batch(list state_arrays, unsigned char[:, :, :, ::1] planes, int max_players=0):
    """Fill ``planes[i]`` from ``state_arrays[i]``: writable raw rows, which may
    mix player counts. ``max_players`` 0 takes the buffer's capacity. One
    scratch state is rebound across the rows, as in the token batch."""
    cdef int n = len(state_arrays)
    cdef int num_tokens, i, count
    cdef GameState scratch = None
    if n == 0:
        return
    if max_players == 0:
        max_players = <int>planes.shape[2] - <int>ROW_PLAYER
    num_tokens = get_num_tokens(max_players)
    assert (planes.shape[0] >= n and planes.shape[1] == <int>NUM_RELATIONS
            and planes.shape[2] == planes.shape[3] == num_tokens), \
        f"get_relation_data_batch: planes {(planes.shape[0], planes.shape[1], planes.shape[2], planes.shape[3])} " \
        f"for {n} rows of {num_tokens} tokens"
    for i in range(n):
        scratch = bind_row(scratch, state_arrays[i], max_players, "get_relation_data_batch")
        with nogil:
            extraction_prologue(scratch)
            count = _fill_planes(scratch, planes[i], turn_num_players(scratch))
        _check_count(count, "get_relation_data_batch")


cpdef int get_relation_coord_data(GameState state, unsigned char[:, ::1] records, int max_players=0):
    """Fill ``records``, ``(MAX_RELATION_EDGES, RELATION_RECORD_WIDTH)``, for
    ``state`` and return the count; ``max_players`` 0 is the state's player
    count. Records name tokens by row, which no capacity moves."""
    cdef int num_players = turn_num_players(state)
    cdef int count
    resolve_max_players(num_players, max_players, "get_relation_coord_data")
    assert records.shape[0] == <int>MAX_RELATION_EDGES and records.shape[1] == <int>RELATION_RECORD_WIDTH, \
        f"get_relation_coord_data: records {(records.shape[0], records.shape[1])}"
    with nogil:
        extraction_prologue(state)
        count = _fill_records(state, <Record*>&records[0, 0], num_players)
    _check_count(count, "get_relation_coord_data")
    return count


cpdef void get_relation_coord_data_batch(list state_arrays, unsigned char[:, :, ::1] records, int max_players=0):
    """Fill ``records[i]`` from ``state_arrays[i]`` as ``get_relation_coord_data``;
    ``max_players`` 0 checks each row against its own count."""
    cdef int n = len(state_arrays)
    cdef int i, count
    cdef GameState scratch = None
    if n == 0:
        return
    assert (records.shape[0] >= n and records.shape[1] == <int>MAX_RELATION_EDGES
            and records.shape[2] == <int>RELATION_RECORD_WIDTH), \
        f"get_relation_coord_data_batch: records {(records.shape[0], records.shape[1], records.shape[2])} for {n} rows"
    for i in range(n):
        scratch = bind_row(scratch, state_arrays[i], max_players, "get_relation_coord_data_batch")
        with nogil:
            extraction_prologue(scratch)
            count = _fill_records(scratch, <Record*>&records[i, 0, 0], turn_num_players(scratch))
        _check_count(count, "get_relation_coord_data_batch")


cpdef object static_relation_planes(int max_players):
    """§6.2's planes, ``(NUM_STATIC_RELATIONS, T, T)`` uint8 for T =
    ``get_num_tokens(max_players)``: the hex across each side, each corp's home,
    and the hexes of TR's and ER's blocks and MF's port lays."""
    cdef int h, s, c
    assert 3 <= max_players <= 5, f"static_relation_planes: max_players must be 3-5, got {max_players}"
    out = np.zeros((<int>NUM_STATIC_RELATIONS, get_num_tokens(max_players), get_num_tokens(max_players)), np.uint8)
    cdef unsigned char[:, :, ::1] planes = out
    for h in range(<int>NUM_HEXES):
        for s in range(<int>NUM_SIDES):
            if HEX_NEIGHBOR[h][s] >= 0:
                planes[<int>SREL_HEX_SIDE + s, <int>ROW_HEX + h, <int>ROW_HEX + HEX_NEIGHBOR[h][s]] = 1
        if HEX_BLOCKER[h] >= 0:
            _static_pair(planes, SREL_ABILITY_HEX, <int>ROW_PRIVATE + HEX_BLOCKER[h], <int>ROW_HEX + h)
    for c in range(<int>NUM_CORPS):
        _static_pair(planes, SREL_HOME, <int>ROW_CORP + c, <int>ROW_HEX + CORP_HOME[c])
    for s in range(<int>NUM_PORTS):
        _static_pair(planes, SREL_ABILITY_HEX, <int>ROW_PRIVATE + <int>PRIVATE_MF, <int>ROW_HEX + MF_PORT_HEX[s])
    return out


cdef inline void _static_pair(unsigned char[:, :, ::1] planes, int relation, int a, int b) noexcept:
    planes[relation, a, b] = planes[relation + 1, b, a] = 1


cdef int _check_count(int count, str func) except -1:
    assert count <= <int>MAX_RELATION_EDGES, \
        f"{func}: {count} relation records, above MAX_RELATION_EDGES {<int>MAX_RELATION_EDGES}"
    return 0


# =============================================================================
# THE EMITTER AND THE TWO FORMS
# =============================================================================

cdef int _fill_planes(GameState state, unsigned char[:, :, ::1] planes, int num_players) noexcept nogil:
    """The dense form: zeroed planes with every record scattered in. Returns
    the record count."""
    cdef Record records[2048]               # MAX_RELATION_EDGES
    cdef int count = _emit(state, num_players, records)
    cdef int k
    memset(&planes[0, 0, 0], 0, planes.shape[0] * planes.shape[1] * planes.shape[2])
    for k in range(min(count, <int>MAX_RELATION_EDGES)):
        planes[records[k].relation, records[k].query, records[k].key] = records[k].value
    return count


cdef int _fill_records(GameState state, Record* records, int num_players) noexcept nogil:
    """The sparse form: the records, then zeros to MAX_RELATION_EDGES. Returns
    the record count."""
    cdef int count = _emit(state, num_players, records)
    cdef int used = min(count, <int>MAX_RELATION_EDGES)
    memset(records + used, 0, (<int>MAX_RELATION_EDGES - used) * sizeof(Record))
    return count


cdef inline int _put(Record* out, int n, int relation, int query, int key, int value) noexcept nogil:
    """Append a record. Past the bound only the count grows, for the caller's assert."""
    if n < <int>MAX_RELATION_EDGES:
        out[n].relation = <unsigned char>relation
        out[n].query = <unsigned char>query
        out[n].key = <unsigned char>key
        out[n].value = <unsigned char>value
    return n + 1


cdef inline int _pair(Record* out, int n, int relation, int a, int b, int value) noexcept nogil:
    """``relation`` from token a to b, and its reverse, ``relation + 1``, from b to a."""
    return _put(out, _put(out, n, relation, a, b, value), relation + 1, b, a, value)


cdef int _emit(GameState state, int num_players, Record* out) noexcept nogil:
    """Write relations 0-20 of the state to ``out``, up to MAX_RELATION_EDGES
    records, and return how many there are. Reads the caches raw, so the
    prologue runs first."""
    cdef int n = 0
    cdef int c, i, p, t, h, s, corp, entry, owner
    cdef uint32_t paths
    for c in range(<int>NUM_CORPS):
        corp = <int>ROW_CORP + c
        for i in range(num_players):
            s = player_shares(state, i, c)
            if s > 0:
                n = _pair(out, n, REL_PLAYER_HOLDS, <int>ROW_PLAYER + i, corp, 1)
                n = _pair(out, n, REL_PLAYER_SHARES, <int>ROW_PLAYER + i, corp, s)
        i = corp_president(state, c)
        if i >= 0:
            n = _pair(out, n, REL_PRESIDES, <int>ROW_PLAYER + i, corp, 1)
        for t in range(<int>NUM_TRAIN_TYPES):
            s = corp_train_count(state, c, t)
            if s > 0:
                n = _pair(out, n, REL_CORP_TRAINS, corp, <int>ROW_TRAIN + t, s)
        for h in range(<int>NUM_HEXES):
            if hex_tokens(state, h) >> c & 1:
                n = _pair(out, n, REL_STATION, corp, <int>ROW_HEX + h, 1)
            entry = graph_entry(state, c, h)
            if entry & <int>GRAPH_SIDES:
                n = _pair(out, n, REL_REACHES_HEX, corp, <int>ROW_HEX + h, 1)
            if entry & <int>GRAPH_NODE:
                n = _pair(out, n, REL_REACHES_CENTER, corp, <int>ROW_HEX + h, 1)
    for p in range(<int>NUM_PRIVATES):
        owner = private_owner(state, p)
        if owner >= <int>OWNER_CORP:
            n = _pair(out, n, REL_CORP_OWNS, <int>ROW_CORP + owner - <int>OWNER_CORP, <int>ROW_PRIVATE + p, 1)
        elif owner >= <int>OWNER_PLAYER:
            n = _pair(out, n, REL_PLAYER_OWNS, <int>ROW_PLAYER + owner - <int>OWNER_PLAYER, <int>ROW_PRIVATE + p, 1)
        for i in range(num_players):
            if player_bid(state, i, p) > 0:
                n = _pair(out, n, REL_BIDS_ON, <int>ROW_PLAYER + i, <int>ROW_PRIVATE + p, 1)
    # One symmetric plane: each linked side is met from both of its hexes.
    for h in range(<int>NUM_HEXES):
        paths = hex_paths(state, h)
        for s in range(<int>NUM_SIDES):
            if (paths & ENDPOINT_PAIRS[s] and HEX_EXIT_MASK[h] >> s & 1
                    and hex_paths(state, HEX_NEIGHBOR[h][s]) & ENDPOINT_PAIRS[(s + 3) % <int>NUM_SIDES]):
                n = _put(out, n, REL_TRACK_LINK, <int>ROW_HEX + h, <int>ROW_HEX + HEX_NEIGHBOR[h][s], 1)
    return n


assert <int>MAX_RELATION_EDGES == 2048 and sizeof(Record) == RELATION_RECORD_WIDTH  # the stack buffer, the record cast
assert REL_TRACK_LINK + 5 == NUM_RELATIONS and SREL_ABILITY_OF + 1 == NUM_STATIC_RELATIONS
# Every open side has a hex across, so the track-link test may read it.
assert all(HEX_NEIGHBOR[h][s] >= 0 for h in range(<int>NUM_HEXES) for s in range(<int>NUM_SIDES)
           if HEX_EXIT_MASK[h] >> s & 1)
