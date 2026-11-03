"""Shikoku 1889 relation extraction (1889/extraction-design.md §6).

Relations are directed ``(query token, key token)`` planes over
g1889.token_data's rows. ``get_relation_data`` fills dense
``(NUM_RELATIONS, T, T)`` uint8 planes, T = ``get_num_tokens(max_players)``,
and ``get_relation_coord_data`` sparse ``(MAX_RELATION_EDGES,
RELATION_RECORD_WIDTH)`` uint8 records ``(relation, query, key, value)``, zero
past the count; the batch forms take raw rows. One emitter feeds both, so
they hold the same edges. Binary planes hold 1, and the share and train
planes raw counts, which the model scales. Each fill runs the token fills'
prologue, which writes the refreshed caches into the row (§4).

``static_relation_planes`` gives the planes that never change (§6.2), which
the model holds as buffers; they never cross IPC.
"""

from g1889.state cimport GameState

# §6.1. Each pair's reverse is the next id; TRACK_LINK is its own.
cpdef enum RelationId:
    REL_PLAYER_HOLDS = 0            # player -> corp: shares > 0
    REL_CORP_HELD_BY = 1
    REL_PLAYER_SHARES = 2           # player -> corp: shares, 1-10
    REL_CORP_SHARES = 3
    REL_PRESIDES = 4                # player -> corp
    REL_PRESIDENT = 5
    REL_PLAYER_OWNS = 6             # player -> private
    REL_OWNED_BY_PLAYER = 7
    REL_CORP_OWNS = 8               # corp -> private
    REL_OWNED_BY_CORP = 9
    REL_BIDS_ON = 10                # player -> private: a bid
    REL_BID_BY = 11
    REL_CORP_TRAINS = 12            # corp -> train type: count, 1-4
    REL_TRAIN_HELD_BY = 13
    REL_STATION = 14                # corp -> hex: a token
    REL_STATION_OF = 15
    REL_REACHES_HEX = 16            # corp -> hex: GRAPH reaches a side
    REL_HEX_REACHED_BY = 17
    REL_REACHES_CENTER = 18         # corp -> hex: GRAPH reaches its center
    REL_CENTER_REACHED_BY = 19
    REL_TRACK_LINK = 20             # hex <-> hex: track on both sides of their side
    REL_CANDIDATE_AT = 21           # 21-24: sub-step 7.2's candidate rows
    REL_CANDIDATE_OF = 22
    REL_CANDIDATE_EXIT = 23
    REL_EXIT_OF = 24
    NUM_RELATIONS = 25

cpdef enum RelationSize:
    # §6.3: the bounds sum to 1,844 records, rounded up to a power of two.
    MAX_RELATION_EDGES = 2048
    RELATION_RECORD_WIDTH = 4

# §6.2
cpdef enum StaticRelationId:
    SREL_HEX_SIDE = 0               # + side: hex -> the hex across it
    SREL_HOME = 6                   # corp -> its home hex
    SREL_HOME_OF = 7
    SREL_ABILITY_HEX = 8            # private -> hex: TR's and ER's blocked hexes, MF's ports
    SREL_ABILITY_OF = 9
    NUM_STATIC_RELATIONS = 10

cpdef void get_relation_data(GameState state, unsigned char[:, :, ::1] planes, int max_players=*)
cpdef void get_relation_data_batch(list state_arrays, unsigned char[:, :, :, ::1] planes, int max_players=*)
cpdef int get_relation_coord_data(GameState state, unsigned char[:, ::1] records, int max_players=*)
cpdef void get_relation_coord_data_batch(list state_arrays, unsigned char[:, :, ::1] records, int max_players=*)
cpdef object static_relation_planes(int max_players)
