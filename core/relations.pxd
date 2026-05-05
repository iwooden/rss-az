"""
Declaration file for Graphormer-style attention relation extraction.

Dense relation buffers are ``(num_relations, num_tokens, num_tokens)`` uint8,
where ``num_tokens`` is the requested player-token capacity plus the fixed
54-token prefix. Rows are query tokens and columns are key/value tokens,
matching PyTorch SDPA's additive attention-bias layout. Sparse coordinate
buffers carry ``(relation_id, query_token, key_token)`` uint8 triplets.
"""

from core.state cimport GameState


cpdef enum AttentionRelationIndex:
    REL_CORP_OWNS_COMPANY = 0
    REL_COMPANY_OWNED_BY_CORP = 1
    REL_PLAYER_OWNS_COMPANY = 2
    REL_COMPANY_OWNED_BY_PLAYER = 3
    REL_FI_OWNS_COMPANY = 4
    REL_COMPANY_OWNED_BY_FI = 5
    REL_PLAYER_OWNS_CORP_SHARES = 6
    REL_CORP_HAS_PLAYER_SHAREHOLDER = 7
    REL_PLAYER_PRESIDENT_OF_CORP = 8
    REL_CORP_PRESIDENT_PLAYER = 9
    REL_NUM_ATTENTION_RELATIONS = 10


cpdef enum AttentionRelationCoordSize:
    MAX_ATTENTION_RELATION_EDGES = 256
    ATTENTION_RELATION_COORD_WIDTH = 3


cpdef int get_num_attention_relations() noexcept nogil


cpdef int get_max_attention_relation_edges() noexcept nogil


cpdef int get_attention_relation_coord_width() noexcept nogil


cpdef void get_relation_data(
    GameState state, unsigned char[:, :, ::1] buffer, int max_players=*,
)


cpdef void get_relation_data_batch(
    list state_arrays, object arg2, object arg3=*, int max_players=*,
)


cpdef int get_relation_coord_data(
    GameState state, unsigned char[:, ::1] coords, int max_players=*,
)


cpdef void get_relation_coord_data_batch(
    list state_arrays, object arg2, object arg3=*, int max_players=*,
)
