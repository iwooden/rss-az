"""
Attention-relation extraction: compact GameState -> directed relation planes.

``get_relation_data`` fills a ``(num_relations, num_tokens, num_tokens)``
uint8 buffer. Each plane is directed: row ``i`` is the query token being
updated, column ``j`` is the key/value token it may attend to. A 1 marks that
the relation is present for that ordered token pair.

``get_relation_coord_data`` emits the same information as sparse
``(relation_id, query_token, key_token)`` uint8 triplets for the eval-server
IPC path. The model still consumes dense relation planes; the eval server
materializes the dense tensor on-device before the forward pass.

The token indices mirror ``core.token_data`` / ``nn.transformer``:
companies live at rows [1, 37), corps at rows [46, 54), players after the
fixed 54-token prefix.
"""

from libc.stddef cimport size_t
from libc.string cimport memset

from core.state cimport GameState, LAYOUT, PLAYER_FIELDS, TURN_OFFSETS
from core.relations cimport (
    REL_COMPANY_OWNED_BY_FI,
    REL_COMPANY_OWNED_BY_PLAYER,
    REL_COMPANY_OWNED_BY_CORP,
    REL_CORP_HAS_PLAYER_SHAREHOLDER,
    REL_CORP_PRESIDENT_PLAYER,
    REL_CORP_OWNS_COMPANY,
    REL_FI_OWNS_COMPANY,
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    REL_NUM_ATTENTION_RELATIONS,
    REL_PLAYER_OWNS_COMPANY,
    REL_PLAYER_OWNS_CORP_SHARES,
    REL_PLAYER_PRESIDENT_OF_CORP,
)
from entities.company cimport company_owned_by_fi, company_owned_by_player
from entities.corp cimport (
    corp_has_acquisition_company,
    corp_is_in_receivership,
    corp_owns_company,
    corp_president_id,
)


# Fixed token layout constants, matching core.token_data and nn.transformer.
DEF NUM_CORPS = 8
DEF NUM_COMPANIES = 36
DEF NUM_FIXED_TOKENS = 54
DEF TOKEN_COMPANY_START = 1
DEF TOKEN_FI = 37
DEF TOKEN_CORP_START = 46
DEF TOKEN_PLAYER_START = 54


cpdef int get_num_attention_relations() noexcept nogil:
    """Number of directed relation planes emitted by this module."""
    return <int>REL_NUM_ATTENTION_RELATIONS


cpdef int get_max_attention_relation_edges() noexcept nogil:
    """Maximum sparse relation-coordinate rows emitted per state."""
    return <int>MAX_ATTENTION_RELATION_EDGES


cpdef int get_attention_relation_coord_width() noexcept nogil:
    """Width of one sparse relation-coordinate row: relation, query, key."""
    return <int>ATTENTION_RELATION_COORD_WIDTH


cpdef void get_relation_data(GameState state, unsigned char[:, :, ::1] buffer):
    """Fill ``buffer`` with directed attention relations for ``state``.

    ``buffer`` must be a C-contiguous uint8 array with exact shape
    ``(num_relations, num_tokens, num_tokens)``. The function zeroes the full
    region before writing current-state relations, so callers can reuse shared
    slots without manually clearing stale edges.
    """
    cdef int num_players = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.num_players
    ]
    cdef int num_tokens = num_players + NUM_FIXED_TOKENS
    cdef int relation_count = <int>REL_NUM_ATTENTION_RELATIONS

    assert 3 <= num_players <= 5, \
        f"get_relation_data: num_players must be 3-5, got {num_players}"
    assert buffer.shape[0] == relation_count, \
        (
            f"get_relation_data: relation planes {buffer.shape[0]} != "
            f"{relation_count}"
        )
    assert buffer.shape[1] == num_tokens, \
        f"get_relation_data: query rows {buffer.shape[1]} != num_tokens {num_tokens}"
    assert buffer.shape[2] == num_tokens, \
        f"get_relation_data: key cols {buffer.shape[2]} != num_tokens {num_tokens}"
    assert buffer.strides[2] == 1, \
        f"get_relation_data: key stride {buffer.strides[2]} != 1"
    assert buffer.strides[1] == num_tokens, \
        f"get_relation_data: query stride {buffer.strides[1]} != {num_tokens}"
    assert buffer.strides[0] == num_tokens * num_tokens, \
        (
            f"get_relation_data: relation stride {buffer.strides[0]} != "
            f"{num_tokens * num_tokens}"
        )

    with nogil:
        _fill_relations(state, buffer, num_tokens)


cpdef void get_relation_data_batch(
    list state_arrays,
    int num_players,
    unsigned char[:, :, :, ::1] buffer,
):
    """Batched ``get_relation_data`` for compact int16 state arrays.

    Reuses one scratch ``GameState`` across all rows, matching the
    ``core.token_data.get_token_data_batch`` pattern used on the eval hot path.
    """
    cdef int n = len(state_arrays)
    cdef int num_tokens = num_players + NUM_FIXED_TOKENS
    cdef int relation_count = <int>REL_NUM_ATTENTION_RELATIONS
    cdef int i
    cdef GameState scratch_gs

    assert 3 <= num_players <= 5, \
        f"get_relation_data_batch: num_players must be 3-5, got {num_players}"
    if n == 0:
        return
    assert buffer.shape[0] >= n, \
        f"get_relation_data_batch: buffer batch {buffer.shape[0]} < n {n}"
    assert buffer.shape[1] == relation_count, \
        (
            f"get_relation_data_batch: relation planes {buffer.shape[1]} != "
            f"{relation_count}"
        )
    assert buffer.shape[2] == num_tokens, (
        f"get_relation_data_batch: query rows {buffer.shape[2]} "
        f"!= num_tokens {num_tokens}"
    )
    assert buffer.shape[3] == num_tokens, (
        f"get_relation_data_batch: key cols {buffer.shape[3]} "
        f"!= num_tokens {num_tokens}"
    )
    assert buffer.strides[3] == 1, \
        f"get_relation_data_batch: key stride {buffer.strides[3]} != 1"
    assert buffer.strides[2] == num_tokens, \
        f"get_relation_data_batch: query stride {buffer.strides[2]} != {num_tokens}"
    assert buffer.strides[1] == num_tokens * num_tokens, \
        (
            f"get_relation_data_batch: relation stride {buffer.strides[1]} != "
            f"{num_tokens * num_tokens}"
        )
    assert buffer.strides[0] == relation_count * num_tokens * num_tokens, (
        f"get_relation_data_batch: batch stride {buffer.strides[0]} != "
        f"{relation_count * num_tokens * num_tokens}"
    )

    scratch_gs = GameState.from_buffer(state_arrays[0], num_players)

    for i in range(n):
        if i > 0:
            scratch_gs.rebind(state_arrays[i], num_players)
        with nogil:
            _fill_relations(scratch_gs, buffer[i], num_tokens)


cpdef int get_relation_coord_data(GameState state, unsigned char[:, ::1] coords):
    """Fill sparse directed attention-relation coordinates for ``state``.

    ``coords`` must be C-contiguous with exact shape
    ``(MAX_ATTENTION_RELATION_EDGES, ATTENTION_RELATION_COORD_WIDTH)``.
    Each populated row is ``(relation_id, query_token, key_token)``. The
    function zeroes the full buffer before writing current-state coordinates,
    leaving unused rows as the sentinel ``(0, 0, 0)``.

    Returns the number of real coordinates written.
    """
    cdef int num_players = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.num_players
    ]
    cdef int num_tokens = num_players + NUM_FIXED_TOKENS
    cdef int count
    cdef int max_edges = <int>MAX_ATTENTION_RELATION_EDGES
    cdef int coord_width = <int>ATTENTION_RELATION_COORD_WIDTH

    assert 3 <= num_players <= 5, \
        f"get_relation_coord_data: num_players must be 3-5, got {num_players}"
    assert coords.shape[0] == max_edges, \
        (
            f"get_relation_coord_data: coord rows {coords.shape[0]} != "
            f"{max_edges}"
        )
    assert coords.shape[1] == coord_width, \
        (
            f"get_relation_coord_data: coord width {coords.shape[1]} != "
            f"{coord_width}"
        )
    assert coords.strides[1] == 1, \
        f"get_relation_coord_data: coord stride {coords.strides[1]} != 1"
    assert coords.strides[0] == coord_width, \
        (
            f"get_relation_coord_data: row stride {coords.strides[0]} != "
            f"{coord_width}"
        )

    with nogil:
        count = _fill_relation_coords(state, coords, num_tokens)
    assert count <= max_edges, (
        f"get_relation_coord_data: emitted {count} relation coordinates, "
        f"capacity is {max_edges}"
    )
    return count


cpdef void get_relation_coord_data_batch(
    list state_arrays,
    int num_players,
    unsigned char[:, :, ::1] coords,
):
    """Batched ``get_relation_coord_data`` for compact int16 state arrays."""
    cdef int n = len(state_arrays)
    cdef int num_tokens = num_players + NUM_FIXED_TOKENS
    cdef int i
    cdef int count
    cdef int max_edges = <int>MAX_ATTENTION_RELATION_EDGES
    cdef int coord_width = <int>ATTENTION_RELATION_COORD_WIDTH
    cdef GameState scratch_gs

    assert 3 <= num_players <= 5, \
        f"get_relation_coord_data_batch: num_players must be 3-5, got {num_players}"
    if n == 0:
        return
    assert coords.shape[0] >= n, \
        f"get_relation_coord_data_batch: buffer batch {coords.shape[0]} < n {n}"
    assert coords.shape[1] == max_edges, \
        (
            f"get_relation_coord_data_batch: coord rows {coords.shape[1]} != "
            f"{max_edges}"
        )
    assert coords.shape[2] == coord_width, \
        (
            f"get_relation_coord_data_batch: coord width {coords.shape[2]} != "
            f"{coord_width}"
        )
    assert coords.strides[2] == 1, \
        f"get_relation_coord_data_batch: coord stride {coords.strides[2]} != 1"
    assert coords.strides[1] == coord_width, \
        (
            f"get_relation_coord_data_batch: row stride {coords.strides[1]} != "
            f"{coord_width}"
        )
    assert coords.strides[0] == max_edges * coord_width, (
        f"get_relation_coord_data_batch: batch stride {coords.strides[0]} != "
        f"{max_edges * coord_width}"
    )

    scratch_gs = GameState.from_buffer(state_arrays[0], num_players)

    for i in range(n):
        if i > 0:
            scratch_gs.rebind(state_arrays[i], num_players)
        with nogil:
            count = _fill_relation_coords(scratch_gs, coords[i], num_tokens)
        assert count <= max_edges, (
            f"get_relation_coord_data_batch: row {i} emitted {count} "
            f"relation coordinates, capacity is {max_edges}"
        )


cdef void _fill_relations(
    GameState state,
    unsigned char[:, :, ::1] buffer,
    int num_tokens,
) noexcept nogil:
    cdef int corp_id
    cdef int company_id
    cdef int corp_tok
    cdef int company_tok
    cdef int player_id
    cdef int player_tok
    cdef int president_id

    memset(
        &buffer[0, 0, 0],
        0,
        <size_t>(
            <int>REL_NUM_ATTENTION_RELATIONS * num_tokens * num_tokens
        ) * sizeof(unsigned char),
    )

    # CORP_OWNS_COMPANY: query corp token reads key company token.
    # COMPANY_OWNED_BY_CORP: query company token reads key corp token.
    # The ownership edge includes both the corp's portfolio and its ACQ pile.
    for corp_id in range(NUM_CORPS):
        corp_tok = TOKEN_CORP_START + corp_id
        for company_id in range(NUM_COMPANIES):
            if (
                corp_owns_company(state, corp_id, company_id)
                or corp_has_acquisition_company(state, corp_id, company_id)
            ):
                company_tok = TOKEN_COMPANY_START + company_id
                buffer[<int>REL_CORP_OWNS_COMPANY, corp_tok, company_tok] = 1
                buffer[<int>REL_COMPANY_OWNED_BY_CORP, company_tok, corp_tok] = 1

    # Player/FI company ownership. These are separate relation planes from
    # corp ownership so attention heads can learn different routing priors for
    # player portfolios, FI holdings, and corp portfolios.
    for company_id in range(NUM_COMPANIES):
        company_tok = TOKEN_COMPANY_START + company_id
        if company_owned_by_fi(state, company_id):
            buffer[<int>REL_FI_OWNS_COMPANY, TOKEN_FI, company_tok] = 1
            buffer[<int>REL_COMPANY_OWNED_BY_FI, company_tok, TOKEN_FI] = 1
        else:
            for player_id in range(num_tokens - TOKEN_PLAYER_START):
                if company_owned_by_player(state, company_id, player_id):
                    player_tok = TOKEN_PLAYER_START + player_id
                    buffer[<int>REL_PLAYER_OWNS_COMPANY, player_tok, company_tok] = 1
                    buffer[<int>REL_COMPANY_OWNED_BY_PLAYER, company_tok, player_tok] = 1
                    break

    # Player/corp shareholding and presidency. Presidency is derived from
    # share ownership, but it is decision-critical enough to get a distinct
    # directed relation pair.
    for corp_id in range(NUM_CORPS):
        corp_tok = TOKEN_CORP_START + corp_id
        for player_id in range(num_tokens - TOKEN_PLAYER_START):
            player_tok = TOKEN_PLAYER_START + player_id
            if _player_shares(state, player_id, corp_id) > 0:
                buffer[<int>REL_PLAYER_OWNS_CORP_SHARES, player_tok, corp_tok] = 1
                buffer[<int>REL_CORP_HAS_PLAYER_SHAREHOLDER, corp_tok, player_tok] = 1

        if not corp_is_in_receivership(state, corp_id):
            president_id = corp_president_id(state, corp_id)
            if 0 <= president_id < num_tokens - TOKEN_PLAYER_START:
                player_tok = TOKEN_PLAYER_START + president_id
                buffer[<int>REL_PLAYER_PRESIDENT_OF_CORP, player_tok, corp_tok] = 1
                buffer[<int>REL_CORP_PRESIDENT_PLAYER, corp_tok, player_tok] = 1


cdef int _fill_relation_coords(
    GameState state,
    unsigned char[:, ::1] coords,
    int num_tokens,
) noexcept nogil:
    cdef int corp_id
    cdef int company_id
    cdef int corp_tok
    cdef int company_tok
    cdef int player_id
    cdef int player_tok
    cdef int president_id
    cdef int count = 0

    memset(
        &coords[0, 0],
        0,
        <size_t>(
            <int>MAX_ATTENTION_RELATION_EDGES
            * <int>ATTENTION_RELATION_COORD_WIDTH
        ) * sizeof(unsigned char),
    )

    # CORP_OWNS_COMPANY / COMPANY_OWNED_BY_CORP.
    for corp_id in range(NUM_CORPS):
        corp_tok = TOKEN_CORP_START + corp_id
        for company_id in range(NUM_COMPANIES):
            if (
                corp_owns_company(state, corp_id, company_id)
                or corp_has_acquisition_company(state, corp_id, company_id)
            ):
                company_tok = TOKEN_COMPANY_START + company_id
                count = _append_relation_coord(
                    coords,
                    count,
                    <int>REL_CORP_OWNS_COMPANY,
                    corp_tok,
                    company_tok,
                )
                count = _append_relation_coord(
                    coords,
                    count,
                    <int>REL_COMPANY_OWNED_BY_CORP,
                    company_tok,
                    corp_tok,
                )

    # Player/FI company ownership.
    for company_id in range(NUM_COMPANIES):
        company_tok = TOKEN_COMPANY_START + company_id
        if company_owned_by_fi(state, company_id):
            count = _append_relation_coord(
                coords,
                count,
                <int>REL_FI_OWNS_COMPANY,
                TOKEN_FI,
                company_tok,
            )
            count = _append_relation_coord(
                coords,
                count,
                <int>REL_COMPANY_OWNED_BY_FI,
                company_tok,
                TOKEN_FI,
            )
        else:
            for player_id in range(num_tokens - TOKEN_PLAYER_START):
                if company_owned_by_player(state, company_id, player_id):
                    player_tok = TOKEN_PLAYER_START + player_id
                    count = _append_relation_coord(
                        coords,
                        count,
                        <int>REL_PLAYER_OWNS_COMPANY,
                        player_tok,
                        company_tok,
                    )
                    count = _append_relation_coord(
                        coords,
                        count,
                        <int>REL_COMPANY_OWNED_BY_PLAYER,
                        company_tok,
                        player_tok,
                    )
                    break

    # Player/corp shareholding and presidency.
    for corp_id in range(NUM_CORPS):
        corp_tok = TOKEN_CORP_START + corp_id
        for player_id in range(num_tokens - TOKEN_PLAYER_START):
            player_tok = TOKEN_PLAYER_START + player_id
            if _player_shares(state, player_id, corp_id) > 0:
                count = _append_relation_coord(
                    coords,
                    count,
                    <int>REL_PLAYER_OWNS_CORP_SHARES,
                    player_tok,
                    corp_tok,
                )
                count = _append_relation_coord(
                    coords,
                    count,
                    <int>REL_CORP_HAS_PLAYER_SHAREHOLDER,
                    corp_tok,
                    player_tok,
                )

        if not corp_is_in_receivership(state, corp_id):
            president_id = corp_president_id(state, corp_id)
            if 0 <= president_id < num_tokens - TOKEN_PLAYER_START:
                player_tok = TOKEN_PLAYER_START + president_id
                count = _append_relation_coord(
                    coords,
                    count,
                    <int>REL_PLAYER_PRESIDENT_OF_CORP,
                    player_tok,
                    corp_tok,
                )
                count = _append_relation_coord(
                    coords,
                    count,
                    <int>REL_CORP_PRESIDENT_PLAYER,
                    corp_tok,
                    player_tok,
                )

    return count


cdef inline int _append_relation_coord(
    unsigned char[:, ::1] coords,
    int count,
    int relation_id,
    int query_tok,
    int key_tok,
) noexcept nogil:
    if count < <int>MAX_ATTENTION_RELATION_EDGES:
        coords[count, 0] = <unsigned char>relation_id
        coords[count, 1] = <unsigned char>query_tok
        coords[count, 2] = <unsigned char>key_tok
    return count + 1


cdef inline int _player_shares(
    GameState state,
    int player_id,
    int corp_id,
) noexcept nogil:
    return <int>state._data[
        LAYOUT.players_offset
        + PLAYER_FIELDS.size * player_id
        + PLAYER_FIELDS.owned_shares
        + corp_id
    ]
