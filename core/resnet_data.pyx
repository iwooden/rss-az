"""
Dense ResNet state-vector extraction.

This module mirrors the public, normalized semantics from ``core.token_data``
but packs them into one contiguous float32 vector. Player-indexed records and
player-id one-hots are active-relative: relative slot 0 is always the current
active player.
"""

from libc.string cimport memset

from core.state cimport (
    GameState, LAYOUT, TURN_OFFSETS, PLAYER_FIELDS, FI_OFFSETS,
    get_storage_player_capacity,
)
from core.data cimport (
    GameConstants,
    GamePhases,
    ENGINE_TO_DECISION_PHASE,
    CorpIndices,
    COMPANY_FACE_VALUE,
    COMPANY_LOW_PRICE,
    COMPANY_HIGH_PRICE,
    COMPANY_STARS,
    COMPANY_INCOME,
    MARKET_PRICES,
    PAR_PRICE_VALID,
    CASH_DIVISOR,
    NET_WORTH_DIVISOR,
    COMPANY_INCOME_DIVISOR,
    ENTITY_INCOME_DIVISOR,
    SHARE_DIVISOR,
    COMPANY_PRICE_DIVISOR,
    SHARE_PRICE_DIVISOR,
    COMPANY_STAR_DIVISOR,
    CORP_STAR_DIVISOR,
    IMPACT_DIVISOR,
    PRICE_RANGE_DIVISOR,
)
from entities.corp cimport (
    corp_is_active,
    corp_cash,
    corp_unissued_shares,
    corp_issued_shares,
    corp_bank_shares,
    corp_price_index,
    corp_share_price,
    corp_acquisition_proceeds,
    corp_is_in_receivership,
    corp_president_id,
    corp_has_passed_acq_offer,
    corp_owns_company,
    corp_has_acquisition_company,
    corp_income,
    corp_raw_revenue,
    corp_synergy_income,
    corp_coo_cost,
    corp_ability_income,
    corp_total_stars,
    corp_pending_price_move,
    corp_candidate_synergy_delta,
    _simulate_dividend_price_move,
    _simulate_float,
)
from entities.company cimport (
    LOC_AUCTION,
    LOC_REVEALED,
    LOC_PLAYER,
    LOC_FI,
    LOC_CORP,
    LOC_CORP_ACQ,
    LOC_REMOVED,
    LOC_EXCLUDED,
    company_location,
    company_owner_id,
    company_adjusted_income,
)
from entities.market cimport (
    market_find_next_higher_space,
    market_find_next_lower_space,
)
from entities.player cimport refresh_player_cache_if_dirty


DEF NUM_CORPS = 8
DEF NUM_COMPANIES = 36
DEF NUM_MARKET_SPACES = 27
DEF NUM_DECISION_PHASES = 11
DEF NUM_COO_LEVELS = 7
DEF AUCTION_CAP_INT = 15
DEF NUM_PAR_PRICES = 14
DEF MAX_DIVIDEND = 26
DEF ACQ_PRICE_OFFSETS = 51
DEF FLOAT_SHARES_MAX = 4.0
DEF ROUNDTRIP_LIMIT = 2
DEF CONSECUTIVE_PASSES_DIVISOR = 5.0
DEF OWNED_COMPANIES_DIVISOR = 10.0
DEF TOTAL_SHARES_DIVISOR = 20.0
DEF PRESIDENCIES_DIVISOR = 8.0

DEF GLOBAL_BASE = 0
DEF GLOBAL_WIDTH = 23
DEF MARKET_BASE = 23
DEF MARKET_WIDTH = 54
DEF PHASE_BASE = 77
DEF PHASE_WIDTH = 79
DEF COMPANY_BASE = 156
DEF FI_WIDTH = 39


cdef inline int _company_stride(int num_players) noexcept nogil:
    return 22 + num_players


cdef inline int _fi_base(int num_players) noexcept nogil:
    return COMPANY_BASE + NUM_COMPANIES * _company_stride(num_players)


cdef inline int _corp_stride(int num_players) noexcept nogil:
    return 89 + num_players


cdef inline int _corp_base(int num_players) noexcept nogil:
    return _fi_base(num_players) + FI_WIDTH


cdef inline int _player_stride(int num_players) noexcept nogil:
    return 56 + num_players


cdef inline int _player_base(int num_players) noexcept nogil:
    return _corp_base(num_players) + NUM_CORPS * _corp_stride(num_players)


cdef inline int _resnet_vector_size(int num_players) noexcept nogil:
    return 1699 + 100 * num_players + num_players * num_players


cdef inline int _relative_slot_for_canonical(
    int active_player,
    int player_id,
    int num_players,
) noexcept nogil:
    return (player_id - active_player + num_players) % num_players


cdef inline int _canonical_player_for_relative(
    int active_player,
    int relative_slot,
    int num_players,
) noexcept nogil:
    return (active_player + relative_slot) % num_players


cpdef int get_resnet_vector_size(int num_players):
    """Dense ResNet vector width for 3-5 player training states."""
    assert 3 <= num_players <= 5, \
        f"get_resnet_vector_size: num_players must be 3-5, got {num_players}"
    return _resnet_vector_size(num_players)


cpdef void get_resnet_data(GameState state, float[::1] buffer):
    """Fill ``buffer`` with one dense active-relative ResNet vector."""
    cdef int num_players = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.num_players
    ]
    cdef int vector_size = _resnet_vector_size(num_players)
    cdef int active_player = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_player
    ]
    cdef int i

    assert 3 <= num_players <= 5, \
        f"get_resnet_data: num_players must be 3-5, got {num_players}"
    assert 0 <= active_player < num_players, \
        f"get_resnet_data: active_player must be in [0, {num_players}), got {active_player}"
    assert buffer.shape[0] == vector_size, \
        f"get_resnet_data: buffer width {buffer.shape[0]} != {vector_size}"

    with nogil:
        for i in range(num_players):
            refresh_player_cache_if_dirty(state, i)
        for i in range(NUM_CORPS):
            if corp_is_active(state, i):
                corp_income(state, i)
        _fill_vector(state, buffer, num_players, vector_size, active_player)


cpdef void get_resnet_data_batch(
    list state_arrays,
    int num_players,
    float[:, ::1] buffer,
):
    """Batched ``get_resnet_data`` over compact int16 state arrays."""
    cdef int n = len(state_arrays)
    cdef int vector_size = _resnet_vector_size(num_players)
    cdef int i, p, c, max_players
    cdef int active_player
    cdef GameState scratch_gs

    assert 3 <= num_players <= 5, \
        f"get_resnet_data_batch: num_players must be 3-5, got {num_players}"
    if n == 0:
        return
    assert buffer.shape[0] >= n, \
        f"get_resnet_data_batch: buffer batch {buffer.shape[0]} < n {n}"
    assert buffer.shape[1] == vector_size, \
        f"get_resnet_data_batch: buffer width {buffer.shape[1]} != {vector_size}"

    max_players = get_storage_player_capacity(len(state_arrays[0]))
    scratch_gs = GameState.from_buffer(
        state_arrays[0], num_players, max_players=max_players,
    )

    for i in range(n):
        if i > 0:
            scratch_gs.rebind(
                state_arrays[i], num_players, max_players=max_players,
            )
        active_player = <int>scratch_gs._data[
            LAYOUT.turn_offset + TURN_OFFSETS.active_player
        ]
        assert 0 <= active_player < num_players, (
            "get_resnet_data_batch: active_player must be in "
            f"[0, {num_players}), got {active_player}"
        )

        with nogil:
            for p in range(num_players):
                refresh_player_cache_if_dirty(scratch_gs, p)
            for c in range(NUM_CORPS):
                if corp_is_active(scratch_gs, c):
                    corp_income(scratch_gs, c)
            _fill_vector(
                scratch_gs, buffer[i], num_players, vector_size, active_player,
            )


cdef void _fill_vector(
    GameState state,
    float[::1] buffer,
    int num_players,
    int vector_size,
    int active_player,
) noexcept nogil:
    cdef int i

    memset(&buffer[0], 0, vector_size * sizeof(float))

    _fill_global_info(state, buffer, num_players)
    _fill_market_info(state, buffer)
    _fill_phase_context(state, buffer)

    for i in range(NUM_COMPANIES):
        _fill_company_record(state, buffer, i, num_players, active_player)

    _fill_fi_record(state, buffer, num_players)

    for i in range(NUM_CORPS):
        _fill_corp_record(state, buffer, i, num_players, active_player)

    for i in range(num_players):
        _fill_player_record(state, buffer, i, num_players, active_player)


cdef void _fill_global_info(
    GameState state,
    float[::1] buffer,
    int num_players,
) noexcept nogil:
    cdef int OFF_PHASE = GLOBAL_BASE
    cdef int OFF_COO = GLOBAL_BASE + 11
    cdef int OFF_END_CARD = GLOBAL_BASE + 18
    cdef int OFF_CARDS_REM = GLOBAL_BASE + 19
    cdef int OFF_NUM_PLAYERS = GLOBAL_BASE + 20

    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int decision_phase
    cdef int coo = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]
    cdef int end_card = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.end_card_flipped
    ]
    cdef int cards_rem = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.cards_remaining
    ]

    assert 0 <= phase < <int>GameConstants.NUM_PHASES, \
        f"_fill_global_info: corrupt engine phase {phase}"
    decision_phase = ENGINE_TO_DECISION_PHASE[phase]
    assert -1 <= decision_phase < NUM_DECISION_PHASES, \
        f"_fill_global_info: decision_phase {decision_phase} out of range"
    if decision_phase >= 0:
        buffer[OFF_PHASE + decision_phase] = 1.0

    assert 1 <= coo <= NUM_COO_LEVELS, \
        f"_fill_global_info: coo_level {coo} out of [1, 7]"
    buffer[OFF_COO + (coo - 1)] = 1.0
    buffer[OFF_END_CARD] = 1.0 if end_card else 0.0
    buffer[OFF_CARDS_REM] = <float>cards_rem / <float>NUM_COMPANIES
    buffer[OFF_NUM_PLAYERS + (num_players - 3)] = 1.0


cdef void _fill_market_info(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int i

    for i in range(NUM_MARKET_SPACES):
        buffer[MARKET_BASE + i] = (
            <float>MARKET_PRICES[i] / SHARE_PRICE_DIVISOR
        )
        buffer[MARKET_BASE + NUM_MARKET_SPACES + i] = (
            <float>state._data[LAYOUT.market_offset + i]
        )


cdef void _fill_phase_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]

    if phase == <int>GamePhases.PHASE_INVEST:
        _fill_invest_context(state, buffer)
    elif phase == <int>GamePhases.PHASE_BID:
        _fill_auction_context(state, buffer)
    elif phase == <int>GamePhases.PHASE_DIVIDENDS:
        _fill_dividend_context(state, buffer)
    elif phase == <int>GamePhases.PHASE_ISSUE_SHARES:
        _fill_issue_context(state, buffer)
    elif phase == <int>GamePhases.PHASE_IPO or phase == <int>GamePhases.PHASE_PAR:
        _fill_par_context(state, buffer)
    elif phase == <int>GamePhases.PHASE_ACQ_OFFER:
        _fill_acq_offer_context(state, buffer)
    elif phase == <int>GamePhases.PHASE_ACQ_SELECT_PRICE:
        _fill_acq_price_context(state, buffer)


cdef void _fill_invest_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int passes = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.consecutive_passes
    ]
    buffer[PHASE_BASE] = <float>passes / CONSECUTIVE_PASSES_DIVISOR


cdef void _fill_auction_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int base = PHASE_BASE + 1
    cdef int auction_price = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.auction_price
    ]
    cdef int high_bidder = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder
    ]
    cdef int active_company = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_company
    ]
    cdef bint is_first_bid = high_bidder < 0
    cdef int face_value
    cdef int min_bid
    cdef int min_offset

    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_auction_context: active_company {active_company} unset"
    face_value = COMPANY_FACE_VALUE[active_company]
    min_bid = face_value if is_first_bid else auction_price + 1
    min_offset = min_bid - face_value
    buffer[base] = <float>min_offset / <float>AUCTION_CAP_INT
    buffer[base + 1] = <float>min_bid / COMPANY_PRICE_DIVISOR
    buffer[base + 2] = 1.0 if is_first_bid else 0.0


cdef void _fill_dividend_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int base = PHASE_BASE + 4
    cdef int active_corp = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_corp
    ]
    cdef int amount
    cdef int price_move

    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_dividend_context: active_corp {active_corp} unset"
    assert corp_is_active(state, active_corp), \
        f"_fill_dividend_context: active_corp {active_corp} not active"
    for amount in range(MAX_DIVIDEND):
        price_move = _simulate_dividend_price_move(state, active_corp, amount)
        buffer[base + amount] = <float>price_move / IMPACT_DIVISOR


cdef void _fill_issue_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int base = PHASE_BASE + 30
    cdef int active_corp = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_corp
    ]
    cdef int current_idx
    cdef int new_idx
    cdef int delta

    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_issue_context: active_corp {active_corp} unset"
    assert corp_is_active(state, active_corp), \
        f"_fill_issue_context: active_corp {active_corp} not active"
    if active_corp == <int>CorpIndices.CORP_SM:
        buffer[base] = 0.0
    else:
        current_idx = corp_price_index(state, active_corp)
        new_idx = market_find_next_lower_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[base] = <float>delta / IMPACT_DIVISOR


cdef void _fill_par_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int base = PHASE_BASE + 31
    cdef int active_company = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_company
    ]
    cdef int face_value
    cdef int star_tier
    cdef int par_index
    cdef int slot
    cdef int float_shares
    cdef int market_index
    cdef int player_payment
    cdef int corp_cash_result
    cdef int issued

    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_par_context: active_company {active_company} unset"
    face_value = COMPANY_FACE_VALUE[active_company]
    star_tier = COMPANY_STARS[active_company]
    assert 1 <= star_tier <= 5, \
        f"_fill_par_context: star_tier {star_tier} out of [1, 5]"
    for par_index in range(NUM_PAR_PRICES):
        if PAR_PRICE_VALID[star_tier - 1][par_index] == 0:
            continue
        (
            float_shares,
            market_index,
            player_payment,
            corp_cash_result,
            issued,
        ) = _simulate_float(face_value, par_index)
        slot = base + par_index * 3
        buffer[slot] = <float>player_payment / CASH_DIVISOR
        buffer[slot + 1] = <float>corp_cash_result / CASH_DIVISOR
        buffer[slot + 2] = <float>issued / FLOAT_SHARES_MAX


cdef void _fill_acq_offer_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int base = PHASE_BASE + 73
    cdef int offer_price = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.acq_offer_price
    ]
    cdef int active_company = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_company
    ]
    cdef int low_price
    cdef int offset

    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_acq_offer_context: active_company {active_company} unset"
    low_price = COMPANY_LOW_PRICE[active_company]
    offset = offer_price - low_price
    buffer[base] = <float>offset / <float>ACQ_PRICE_OFFSETS
    buffer[base + 1] = <float>offer_price / COMPANY_PRICE_DIVISOR
    if company_location(state, active_company) == <int>LOC_FI:
        buffer[base + 2] = 1.0


cdef void _fill_acq_price_context(
    GameState state,
    float[::1] buffer,
) noexcept nogil:
    cdef int base = PHASE_BASE + 76
    cdef int active_corp = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_corp
    ]
    cdef int active_company = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_company
    ]
    cdef int low_price
    cdef int high_price
    cdef int synergy_delta

    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_acq_price_context: active_corp {active_corp} unset"
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_acq_price_context: active_company {active_company} unset"
    assert corp_is_active(state, active_corp), \
        f"_fill_acq_price_context: active_corp {active_corp} not active"

    low_price = COMPANY_LOW_PRICE[active_company]
    high_price = COMPANY_HIGH_PRICE[active_company]
    synergy_delta = corp_candidate_synergy_delta(
        state, active_corp, active_company,
    )
    buffer[base] = <float>(high_price - low_price) / PRICE_RANGE_DIVISOR
    if company_location(state, active_company) == <int>LOC_FI:
        buffer[base + 1] = 1.0
    buffer[base + 2] = <float>synergy_delta / ENTITY_INCOME_DIVISOR


cdef void _fill_company_record(
    GameState state,
    float[::1] buffer,
    int company_id,
    int num_players,
    int active_player,
) noexcept nogil:
    cdef int base = COMPANY_BASE + company_id * _company_stride(num_players)
    cdef int loc = company_location(state, company_id)
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int active_corp
    cdef int active_company
    cdef int owner_id
    cdef int coo
    cdef int rel
    cdef int delta

    buffer[base] = 0.0
    buffer[base + 1] = <float>COMPANY_LOW_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[base + 2] = <float>COMPANY_FACE_VALUE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[base + 3] = <float>COMPANY_HIGH_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[base + 4] = (
        <float>(COMPANY_HIGH_PRICE[company_id] - COMPANY_LOW_PRICE[company_id])
        / PRICE_RANGE_DIVISOR
    )
    buffer[base + 5] = <float>COMPANY_INCOME[company_id] / COMPANY_INCOME_DIVISOR
    buffer[base + 6] = <float>COMPANY_STARS[company_id] / COMPANY_STAR_DIVISOR
    buffer[base + 7] = (
        <float>company_adjusted_income(state, company_id)
        / COMPANY_INCOME_DIVISOR
    )

    coo = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]
    if loc == <int>LOC_REMOVED:
        buffer[base + 8] = 1.0
    elif loc == <int>LOC_EXCLUDED and coo > COMPANY_STARS[company_id]:
        buffer[base + 8] = 1.0
    elif loc == <int>LOC_AUCTION:
        buffer[base + 9] = 1.0
    elif loc == <int>LOC_REVEALED:
        buffer[base + 10] = 1.0
    elif loc == <int>LOC_CORP_ACQ:
        buffer[base + 11] = 1.0
        owner_id = company_owner_id(state, company_id)
        assert 0 <= owner_id < NUM_CORPS, \
            f"_fill_company_record: LOC_CORP_ACQ owner {owner_id} out of range"
        buffer[base + 13 + owner_id] = 1.0
    elif loc == <int>LOC_CORP:
        owner_id = company_owner_id(state, company_id)
        assert 0 <= owner_id < NUM_CORPS, \
            f"_fill_company_record: LOC_CORP owner {owner_id} out of range"
        buffer[base + 13 + owner_id] = 1.0
    elif loc == <int>LOC_PLAYER:
        owner_id = company_owner_id(state, company_id)
        assert 0 <= owner_id < num_players, \
            f"_fill_company_record: LOC_PLAYER owner {owner_id} out of range"
        rel = _relative_slot_for_canonical(active_player, owner_id, num_players)
        buffer[base + 21 + rel] = 1.0
    elif loc == <int>LOC_FI:
        buffer[base + 21 + num_players] = 1.0

    if phase == <int>GamePhases.PHASE_ACQ_SELECT_COMPANY:
        active_corp = <int>state._data[
            LAYOUT.turn_offset + TURN_OFFSETS.active_corp
        ]
        assert 0 <= active_corp < NUM_CORPS, \
            f"_fill_company_record: active_corp {active_corp} unset"
        assert corp_is_active(state, active_corp), \
            f"_fill_company_record: active_corp {active_corp} not active"
        if (
            not corp_owns_company(state, active_corp, company_id)
            and not corp_has_acquisition_company(state, active_corp, company_id)
        ):
            delta = corp_candidate_synergy_delta(state, active_corp, company_id)
            buffer[base + 12] = <float>delta / ENTITY_INCOME_DIVISOR

    active_company = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_company
    ]
    assert -1 <= active_company < NUM_COMPANIES, \
        f"_fill_company_record: active_company {active_company} out of range"
    if active_company == company_id:
        buffer[base] = 1.0


cdef void _fill_fi_record(
    GameState state,
    float[::1] buffer,
    int num_players,
) noexcept nogil:
    cdef int base = _fi_base(num_players)
    cdef int cash = <int>state._data[LAYOUT.fi_offset + FI_OFFSETS.cash]
    cdef int income = <int>state._data[LAYOUT.fi_offset + FI_OFFSETS.income]
    cdef int company_id
    cdef int num_companies_owned = 0

    buffer[base] = <float>cash / CASH_DIVISOR
    buffer[base + 1] = <float>income / ENTITY_INCOME_DIVISOR
    for company_id in range(NUM_COMPANIES):
        if company_location(state, company_id) == <int>LOC_FI:
            buffer[base + 3 + company_id] = 1.0
            num_companies_owned += 1
    buffer[base + 2] = <float>num_companies_owned / OWNED_COMPANIES_DIVISOR


cdef void _fill_corp_record(
    GameState state,
    float[::1] buffer,
    int corp_id,
    int num_players,
    int active_player,
) noexcept nogil:
    cdef int base = _corp_base(num_players) + corp_id * _corp_stride(num_players)
    cdef bint active = corp_is_active(state, corp_id)
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int active_corp = <int>state._data[
        LAYOUT.turn_offset + TURN_OFFSETS.active_corp
    ]
    cdef int price_idx
    cdef int president
    cdef int rel
    cdef int company_id
    cdef int current_idx
    cdef int new_idx
    cdef int delta
    cdef int offer_corp
    cdef int num_operational = 0
    cdef int num_acq_pile = 0

    assert -1 <= active_corp < NUM_CORPS, \
        f"_fill_corp_record: active_corp {active_corp} out of range"

    buffer[base + 1] = 1.0 if active else 0.0
    buffer[base + 2] = 1.0 if corp_is_in_receivership(state, corp_id) else 0.0
    buffer[base + 3] = 1.0 if corp_has_passed_acq_offer(state, corp_id) else 0.0
    buffer[base + 4] = <float>corp_unissued_shares(state, corp_id) / SHARE_DIVISOR
    buffer[base + 5] = <float>corp_issued_shares(state, corp_id) / SHARE_DIVISOR
    buffer[base + 6] = <float>corp_bank_shares(state, corp_id) / SHARE_DIVISOR

    if active:
        price_idx = corp_price_index(state, corp_id)
        assert 0 <= price_idx < NUM_MARKET_SPACES, \
            f"_fill_corp_record: price_idx {price_idx} out of range"
        buffer[base + 7 + price_idx] = 1.0
        buffer[base + 34] = <float>corp_share_price(state, corp_id) / SHARE_PRICE_DIVISOR
        buffer[base + 35] = <float>corp_pending_price_move(state, corp_id) / IMPACT_DIVISOR
        buffer[base + 36] = <float>corp_cash(state, corp_id) / CASH_DIVISOR
        buffer[base + 37] = <float>corp_acquisition_proceeds(state, corp_id) / CASH_DIVISOR
        buffer[base + 38] = <float>corp_income(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[base + 39] = <float>corp_total_stars(state, corp_id) / CORP_STAR_DIVISOR
        buffer[base + 40] = <float>corp_raw_revenue(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[base + 41] = <float>corp_synergy_income(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[base + 42] = <float>corp_coo_cost(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[base + 43] = <float>corp_ability_income(state, corp_id) / ENTITY_INCOME_DIVISOR

        if not corp_is_in_receivership(state, corp_id):
            president = corp_president_id(state, corp_id)
            assert 0 <= president < num_players, \
                f"_fill_corp_record: president {president} out of range"
            rel = _relative_slot_for_canonical(active_player, president, num_players)
            buffer[base + 53 + rel] = 1.0

        for company_id in range(NUM_COMPANIES):
            if corp_owns_company(state, corp_id, company_id):
                buffer[base + 53 + num_players + company_id] = 1.0
                num_operational += 1
            elif corp_has_acquisition_company(state, corp_id, company_id):
                buffer[base + 53 + num_players + company_id] = 1.0
                num_acq_pile += 1
        buffer[base + 50] = <float>num_operational / OWNED_COMPANIES_DIVISOR
        buffer[base + 51] = <float>num_acq_pile / OWNED_COMPANIES_DIVISOR
        buffer[base + 52] = (
            <float>(num_operational + num_acq_pile) / OWNED_COMPANIES_DIVISOR
        )

    if phase == <int>GamePhases.PHASE_ACQ_OFFER:
        offer_corp = <int>state._data[
            LAYOUT.turn_offset + TURN_OFFSETS.acq_offer_corp
        ]
        assert 0 <= offer_corp < NUM_CORPS, \
            f"_fill_corp_record: acq_offer_corp {offer_corp} unset"
        if offer_corp == corp_id:
            buffer[base + 44] = 1.0

    if phase == <int>GamePhases.PHASE_DIVIDENDS:
        if (<int>state._data[
            LAYOUT.turn_offset + TURN_OFFSETS.dividend_remaining + corp_id
        ]) != 0:
            buffer[base + 45] = 1.0

    if phase == <int>GamePhases.PHASE_ISSUE_SHARES:
        if (<int>state._data[
            LAYOUT.turn_offset + TURN_OFFSETS.issue_remaining + corp_id
        ]) != 0:
            buffer[base + 46] = 1.0

    if phase == <int>GamePhases.PHASE_IPO or phase == <int>GamePhases.PHASE_PAR:
        if not active:
            buffer[base + 47] = 1.0

    if phase == <int>GamePhases.PHASE_INVEST and active:
        current_idx = corp_price_index(state, corp_id)
        new_idx = market_find_next_higher_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[base + 48] = <float>delta / IMPACT_DIVISOR

        new_idx = market_find_next_lower_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[base + 49] = <float>delta / IMPACT_DIVISOR

    if active_corp == corp_id:
        buffer[base] = 1.0


cdef void _fill_player_record(
    GameState state,
    float[::1] buffer,
    int relative_slot,
    int num_players,
    int active_player,
) noexcept nogil:
    cdef int player_id = _canonical_player_for_relative(
        active_player, relative_slot, num_players,
    )
    cdef int base = _player_base(num_players) + relative_slot * _player_stride(num_players)
    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int turn_order = <int>state._data[player_base + PLAYER_FIELDS.turn_order]
    cdef int active_turn_order_base = (
        LAYOUT.players_offset + active_player * PLAYER_FIELDS.size
    )
    cdef int active_turn_order = <int>state._data[
        active_turn_order_base + PLAYER_FIELDS.turn_order
    ]
    cdef int relative_turn_order
    cdef int has_passed = <int>state._data[player_base + PLAYER_FIELDS.has_passed]
    cdef int cash = <int>state._data[player_base + PLAYER_FIELDS.cash]
    cdef int net_worth = <int>state._data[player_base + PLAYER_FIELDS.net_worth]
    cdef int liquidity = <int>state._data[player_base + PLAYER_FIELDS.liquidity]
    cdef int income = <int>state._data[player_base + PLAYER_FIELDS.income]
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int high_bidder
    cdef int starter
    cdef int corp_id
    cdef int shares
    cdef int buys
    cdef int sells
    cdef int roundtrip_flag = 0
    cdef int total_shares = 0
    cdef int num_presidencies = 0
    cdef int company_id
    cdef int comp_loc
    cdef int comp_owner
    cdef int num_companies_owned = 0

    assert 0 <= turn_order < num_players, \
        f"_fill_player_record: turn_order {turn_order} out of range"
    assert 0 <= active_turn_order < num_players, \
        f"_fill_player_record: active_turn_order {active_turn_order} out of range"
    relative_turn_order = (
        turn_order - active_turn_order + num_players
    ) % num_players

    buffer[base] = 1.0 if relative_slot == 0 else 0.0
    buffer[base + 1 + relative_turn_order] = 1.0
    buffer[base + 1 + num_players] = 1.0 if has_passed else 0.0
    buffer[base + 2 + num_players] = <float>cash / CASH_DIVISOR
    buffer[base + 3 + num_players] = <float>net_worth / NET_WORTH_DIVISOR
    buffer[base + 4 + num_players] = <float>liquidity / NET_WORTH_DIVISOR
    buffer[base + 5 + num_players] = <float>income / ENTITY_INCOME_DIVISOR

    if phase == <int>GamePhases.PHASE_BID:
        high_bidder = <int>state._data[
            LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder
        ]
        starter = <int>state._data[
            LAYOUT.turn_offset + TURN_OFFSETS.auction_starter
        ]
        assert -1 <= high_bidder < num_players, \
            f"_fill_player_record: auction_high_bidder {high_bidder} out of range"
        assert 0 <= starter < num_players, \
            f"_fill_player_record: auction_starter {starter} out of range"
        if high_bidder == player_id:
            buffer[base + 6 + num_players] = 1.0
        if starter == player_id:
            buffer[base + 7 + num_players] = 1.0

    for corp_id in range(NUM_CORPS):
        shares = <int>state._data[
            player_base + PLAYER_FIELDS.owned_shares + corp_id
        ]
        buys = <int>state._data[
            player_base + PLAYER_FIELDS.share_buys + corp_id
        ]
        sells = <int>state._data[
            player_base + PLAYER_FIELDS.share_sells + corp_id
        ]
        buffer[base + 9 + num_players + corp_id] = <float>shares / SHARE_DIVISOR
        total_shares += shares

        if buys >= ROUNDTRIP_LIMIT or sells >= ROUNDTRIP_LIMIT:
            roundtrip_flag = 1
        if (
            corp_is_active(state, corp_id)
            and not corp_is_in_receivership(state, corp_id)
            and corp_president_id(state, corp_id) == player_id
        ):
            num_presidencies += 1

    buffer[base + 8 + num_players] = 1.0 if roundtrip_flag else 0.0
    buffer[base + 18 + num_players] = (
        <float>num_presidencies / PRESIDENCIES_DIVISOR
    )
    buffer[base + 19 + num_players] = (
        <float>total_shares / TOTAL_SHARES_DIVISOR
    )

    for company_id in range(NUM_COMPANIES):
        comp_loc = company_location(state, company_id)
        if comp_loc != <int>LOC_PLAYER:
            continue
        comp_owner = company_owner_id(state, company_id)
        if comp_owner == player_id:
            buffer[base + 20 + num_players + company_id] = 1.0
            num_companies_owned += 1
    buffer[base + 17 + num_players] = (
        <float>num_companies_owned / OWNED_COMPANIES_DIVISOR
    )
