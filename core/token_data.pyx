"""
Token data extraction: compact GameState -> transformer eval buffer.

``get_token_data`` is the sole engine→NN interface: it fills a
(num_tokens, TOKEN_DIM) float32 buffer with normalized per-token
features from a compact GameState. It is called once per NN evaluation,
so the trunk + MCTS throughput depends on this being fast.

Token order (matches ``nn/transformer.py``):
    # Informational tokens. Every token carries at least some dynamic
    # data, so there is no pure-static prefix to prefill once per worker.
    [market_info, companies..., FI, global_info,
    # Phase-specific tokens (left zero unless the engine is in the matching phase).
     invest, auction, dividend, issue, par,
     acq_offer, acq_price_info,
    # Corp tokens, then player tokens — players last so the buffer can be padded
    # for higher player counts later without reshuffling the fixed prefix.
     corps..., players...]

Total tokens in the buffer = num_players + 54. The model consumes exactly
these engine-side rows; no synthetic model-side tokens are appended.

Active-entity selectors (active_player / active_corp / active_company
in TURN_OFFSETS) are surfaced as ``is_selected`` scalar flags on each
entity's own token, not as standalone one-hot tokens.

Per-token feature layouts (sum of widths ≤ TOKEN_DIM = 95 = max width,
currently pinned by the Corp token):

  Every token starts with attn_mask (1), a 0/1 scalar that marks whether
                the token should be visible to model attention. Entity rows,
                MarketInfo, FI, and GlobalInfo are always visible; phase
                information rows are visible only in their matching phase.

  Relational summary scalars sit immediately before the relational tail
  on corp / player / FI tokens. The relational tail (multihots and
  one-hots) is dropped on the model side in favour of Graphormer-style
  attention biases; the summary scalars stay in the projection so the
  trunk has a direct aggregate view (count, totals) of the same data.

  Player (62):  attn_mask (1) + is_selected (1) + turn_order onehot (5) +
                has_passed (1) +
                cash (1) + net_worth (1) + liquidity (1) + income (1) +
                auction_high_bidder (1) + auction_starter (1) +
                round_trips (1) + owned_shares (8) +
                relational summary: num_owned_companies (1) +
                num_presidencies (1) + total_owned_shares (1) +
                relational tail: owned_companies (36)
  Corp   (95):  attn_mask (1) + is_selected (1) + active (1) +
                in_receivership (1) +
                passed_acq_offer (1) + unissued/issued/bank shares (3) +
                price_index onehot (27) + share_price (1) +
                pending_price_move (1) + cash (1) + acq_proceeds (1) +
                income (1) + stars (1) + raw_revenue (1) + synergy_income
                (1) + coo_cost (1) + ability_income (1) +
                acq_offer_corp (1) + dividend_remaining (1) +
                issue_remaining (1) + ipo_remaining (1) +
                buy_impact (1) + sell_impact (1) +
                relational summary: num_operational_companies (1) +
                num_acq_pile_companies (1) + num_total_companies (1) +
                relational tail: president_id onehot (5) +
                owned_companies (36). The
                ``active`` slot still means "corp is floated / operational"
                (matches ``corp_is_active``); the decision-flow selector is
                the ``is_selected`` flag.
  Company (28): attn_mask (1) + is_selected (1) + static data [low/face/high/
                low_high_diff/base_income/stars] (6) + adjusted_income (1)
                + at_removed/at_auction/at_revealed/at_corp_acq (4) +
                acq_select_synergy_delta (1) +
                relational tail: owner_corp onehot (8) + owner_player onehot
                (5, padded for num_players < 5) + owner_fi (1).
                The three ownership groups are mutually exclusive:
                LOC_CORP and LOC_CORP_ACQ set owner_corp, LOC_PLAYER sets
                owner_player, and LOC_FI sets owner_fi. Unowned locations
                (AUCTION / REVEALED / REMOVED / etc.) leave all three
                owner groups zero. ``low_high_diff`` is the ACQ_SELECT_PRICE
                offset count (``high - low + 1``), the same quantity
                the price head conditions on, normalized by
                PRICE_RANGE_DIVISOR (max is 51 for CDG). ``at_removed``
                is 1 for LOC_REMOVED, and additionally 1 for
                LOC_EXCLUDED once the CoO has advanced past the
                company's star tier — the exclusion is publicly
                observable then; setting it unconditionally would leak
                setup randomness.
  FI      (40): attn_mask (1) + cash + income +
                relational summary: num_owned_companies (1) +
                relational tail: owned_companies (36)
  MarketInfo (55): attn_mask (1) + static market-space prices normalized by
                SHARE_PRICE_DIVISOR ($0..$75 / 75) (27) + per-space
                availability flags (27). The prices half is constant
                across a game; the availability half is overwritten
                every extraction.
  GlobalInfo (24): attn_mask (1) + decision phase onehot (11) +
                CoO onehot (7) + end_card_flipped (1) +
                cards_remaining (1) + num_players onehot (3)
  Invest   (2): attn_mask (1) + consecutive_passes
  Auction  (4): attn_mask (1) + min_bid_index + min_bid_value + is_first_bid
  Divd    (27): attn_mask (1) + dividend_impacts (26)
  Issue    (2): attn_mask (1) + issue_impact
  PAR     (43): attn_mask (1) + 14 price tuples, each containing
                player_cash_required, resulting_corp_cash, and
                resulting_issued_shares. Filled
                during both PHASE_IPO and PHASE_PAR — the data is
                identical across those two phases.
  AcqOff   (4): attn_mask (1) + offer_price_index + offer_price + fi_company
  AcqPrice (4): attn_mask (1) + max_offset (ACQ offset count for target) +
                fi_flag + total_synergies (marginal synergy income the active corp
                would gain by adding the target company)

All values normalized by divisors defined in ``core.data`` (compile-time
floats inlined by the C compiler). Phase-specific tokens are zeroed out
when the current engine phase does not match. The function is designed
so the post-GIL body runs ``nogil``; a small Python-level prologue
forces per-player cache refreshes so the nogil body can read cached
net_worth / liquidity / income slots directly.
"""

import numpy as np

from libc.stdint cimport int16_t
from libc.string cimport memset

from core.state cimport (
    GameState, LAYOUT, TURN_OFFSETS, CORP_FIELDS, PLAYER_FIELDS,
    COMPANY_OFFSETS, FI_OFFSETS,
)
from core.data cimport (
    GameConstants,
    GamePhases,
    DecisionPhase,
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
    LOC_DECK, LOC_AUCTION, LOC_REVEALED, LOC_PLAYER, LOC_FI, LOC_CORP,
    LOC_CORP_ACQ, LOC_REMOVED, LOC_EXCLUDED,
    company_location,
    company_owner_id,
    company_adjusted_income,
)
from entities.market cimport (
    market_find_next_higher_space,
    market_find_next_lower_space,
)
from entities.player cimport refresh_player_cache_if_dirty


# =============================================================================
# PER-TOKEN FEATURE COUNTS (single source of truth for offset arithmetic)
# =============================================================================

# Fixed token layout constants (see nn/transformer.py). These match the
# token slicing bookkeeping in ``RSSTransformerNet.__init__`` for 3-5p.
DEF NUM_CORPS = 8
DEF NUM_COMPANIES = 36
DEF NUM_MARKET_SPACES = 27
DEF NUM_DECISION_PHASES = 11
DEF NUM_COO_LEVELS = 7
DEF MAX_MODEL_PLAYERS = 5          # 3-5p supported; one-hots are padded to 5
DEF AUCTION_CAP_INT = 15           # INVEST auction price offsets per company
DEF NUM_PAR_PRICES = 14
DEF MAX_DIVIDEND = 26              # dividend amounts 0..25 (26 slots)
DEF ACQ_PRICE_OFFSETS = 51         # acquisition price offsets (matches action encoding)
DEF FLOAT_SHARES_MAX = 4.0         # max issued shares at float (face>par → 4)
DEF ROUNDTRIP_LIMIT = 2            # share buy+sell limit per corp per turn

# Normalization constant for the invest token's consecutive_passes slot.
# Matches the max training player count (5).
DEF CONSECUTIVE_PASSES_DIVISOR = 5.0

# Relational-summary divisors. Soft empirical caps rather than hard
# upper bounds — ownership counts can in principle exceed 10, but game
# logs show that's vanishingly rare; saturating slightly past the
# divisor is fine.
DEF OWNED_COMPANIES_DIVISOR = 10.0    # corp / player / FI owned-company counts
DEF TOTAL_SHARES_DIVISOR    = 20.0    # player aggregate shares across all corps
DEF PRESIDENCIES_DIVISOR    = 8.0     # = NUM_CORPS, hard cap on player presidencies


# =============================================================================
# PUBLIC API
# =============================================================================

cpdef int get_num_tokens(int num_players) noexcept nogil:
    """Input-buffer token count for the given player count (num_players + 54).

    The model consumes exactly this many engine-side token rows.
    """
    return num_players + 54


cpdef object get_token_widths(int num_players):
    """Per-position non-padded feature widths matching ``_fill_buffer``.

    Each ``buffer[i]`` row is TOKEN_DIM wide but only the first
    ``widths[i]`` slots carry features — the rest are zero padding. The
    returned array mirrors the buffer layout (informational tokens,
    phase-specific tokens, corps, players) so the caller can slice
    ``buffer[i, :widths[i]]`` or group positions by type for per-type
    projection modules without duplicating the layout logic.

    Returns a uint8 ``(num_players + 54,)`` numpy array; all widths fit
    in a byte (max is TW_CORP = 92 < 256).
    """
    assert 3 <= num_players <= 5, \
        f"get_token_widths: num_players must be 3-5, got {num_players}"

    cdef int num_tokens = num_players + 54
    widths = np.empty(num_tokens, dtype=np.uint8)
    cdef unsigned char[::1] w = widths

    cdef int i
    cdef int tok = 0

    # Informational tokens
    w[tok] = <unsigned char>TokenWidth.TW_MARKET_INFO
    tok += 1
    for i in range(NUM_COMPANIES):
        w[tok] = <unsigned char>TokenWidth.TW_COMPANY
        tok += 1

    w[tok] = <unsigned char>TokenWidth.TW_FI
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_GLOBAL_INFO
    tok += 1

    # Phase-specific tokens — the slots exist regardless of the current
    # phase (they're just zero-filled when the phase doesn't match), so
    # the widths here are the per-phase logical widths.
    w[tok] = <unsigned char>TokenWidth.TW_INVEST
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_AUCTION
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_DIVIDEND
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ISSUE
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_PAR
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ACQ_OFFER
    tok += 1
    w[tok] = <unsigned char>TokenWidth.TW_ACQ_PRICE
    tok += 1

    # Corp tokens
    for i in range(NUM_CORPS):
        w[tok] = <unsigned char>TokenWidth.TW_CORP
        tok += 1

    # Player tokens (trailing)
    for i in range(num_players):
        w[tok] = <unsigned char>TokenWidth.TW_PLAYER
        tok += 1

    return widths


cpdef void get_token_data(GameState state, float[:, ::1] buffer):
    """Fill ``buffer`` with per-token NN features for ``state``.

    ``buffer`` must be a writable C-contiguous float32 memoryview at least
    ``(num_players + 54, TOKEN_DIM)`` in size. Training is scoped to
    3-5 players; other player counts are rejected.

    The cache-refresh prologue and ``_fill_buffer`` run in a single nogil
    block — refresh goes through the module-level
    ``refresh_player_cache_if_dirty`` helper rather than the Python-level
    ``PLAYERS[i].get_net_worth(state)`` lookup it used to.
    """
    cdef int num_players = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.num_players]
    cdef int num_tokens = num_players + 54
    cdef int i

    assert 3 <= num_players <= 5, \
        f"get_token_data: num_players must be 3-5, got {num_players}"
    assert buffer.shape[0] >= num_tokens, \
        f"get_token_data: buffer rows {buffer.shape[0]} < num_tokens {num_tokens}"
    # Exact-match on the padded width: the nogil memset in ``_fill_buffer``
    # writes ``num_tokens * TOKEN_DIM * 4`` contiguous bytes, so a wider
    # buffer would silently clobber across rows.
    assert buffer.shape[1] == <int>TokenDataSize.TOKEN_DIM, \
        f"get_token_data: buffer cols {buffer.shape[1]} != TOKEN_DIM {<int>TokenDataSize.TOKEN_DIM}"

    with nogil:
        for i in range(num_players):
            refresh_player_cache_if_dirty(state, i)
        _fill_buffer(state, buffer, num_players, num_tokens)


cpdef void get_token_data_batch(
    list state_arrays,
    int num_players,
    float[:, :, ::1] buffer,
):
    """Batched ``get_token_data``: fill ``buffer[i]`` for each ``state_arrays[i]``.

    Reuses a single scratch ``GameState`` across all rows via ``rebind``
    (zero-copy). Each row's cache refresh + fill runs in a single nogil
    block; the only GIL-held op per iteration is ``rebind`` (Python-level
    validation + ``_array`` attribute write).

    Args:
        state_arrays: List of writable C-contiguous int16 state arrays, one
            per leaf. Every entry must size-match the shared ``num_players``
            layout (same constraint as ``GameState.rebind``).
        num_players: Training player count (3-5). Applies to every state
            array in the batch — mixed-player batches are not supported.
        buffer: ``(n, num_players + 54, TOKEN_DIM)`` float32 output, C-contig.
    """
    cdef int n = len(state_arrays)
    cdef int num_tokens = num_players + 54
    cdef int i, p
    cdef GameState scratch_gs

    assert 3 <= num_players <= 5, \
        f"get_token_data_batch: num_players must be 3-5, got {num_players}"
    if n == 0:
        return
    assert buffer.shape[0] >= n, \
        f"get_token_data_batch: buffer batch {buffer.shape[0]} < n {n}"
    assert buffer.shape[1] >= num_tokens, \
        f"get_token_data_batch: buffer rows {buffer.shape[1]} < num_tokens {num_tokens}"
    # Exact-match on the padded width: see ``get_token_data`` for the same
    # constraint — memset writes assume rows are tightly packed at TOKEN_DIM.
    assert buffer.shape[2] == <int>TokenDataSize.TOKEN_DIM, \
        f"get_token_data_batch: buffer cols {buffer.shape[2]} != TOKEN_DIM {<int>TokenDataSize.TOKEN_DIM}"

    scratch_gs = GameState.from_buffer(state_arrays[0], num_players)

    for i in range(n):
        if i > 0:
            scratch_gs.rebind(state_arrays[i], num_players)

        with nogil:
            for p in range(num_players):
                refresh_player_cache_if_dirty(scratch_gs, p)
            _fill_buffer(scratch_gs, buffer[i], num_players, num_tokens)


# =============================================================================
# INTERNAL: MAIN FILL DRIVER
# =============================================================================

cdef void _fill_buffer(
    GameState state,
    float[:, ::1] buffer,
    int num_players,
    int num_tokens,
) noexcept nogil:
    cdef int i, tok
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]

    # Zero the region we'll write. Many slots are written conditionally
    # (phase-specific tokens when the phase doesn't match, company
    # owner/location groups outside the current location, ACQ_SELECT_COMPANY
    # synergy slots for companies already in the portfolio, etc.) and rely
    # on a zeroed baseline.
    memset(&buffer[0, 0], 0, num_tokens * <int>TokenDataSize.TOKEN_DIM * sizeof(float))

    tok = 0

    # --- Informational tokens ---

    _fill_market_info_token(state, buffer, tok)
    tok += 1

    for i in range(NUM_COMPANIES):
        _fill_company_token(state, buffer, tok, i, num_players)
        tok += 1

    _fill_fi_token(state, buffer, tok)
    tok += 1

    _fill_global_info_token(state, buffer, tok, num_players)
    tok += 1

    # --- Phase-specific tokens (left zero when the current phase doesn't match) ---

    if phase == <int>GamePhases.PHASE_INVEST:
        _fill_invest_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_BID:
        _fill_auction_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_DIVIDENDS:
        _fill_dividend_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_ISSUE_SHARES:
        _fill_issue_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_IPO or phase == <int>GamePhases.PHASE_PAR:
        _fill_par_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_ACQ_OFFER:
        _fill_acq_offer_token(state, buffer, tok)
    tok += 1

    if phase == <int>GamePhases.PHASE_ACQ_SELECT_PRICE:
        _fill_acq_price_info_token(state, buffer, tok)
    tok += 1

    # --- Corp tokens ---
    for i in range(NUM_CORPS):
        _fill_corp_token(state, buffer, tok, i, num_players)
        tok += 1

    # --- Player tokens (last; trailing slot makes higher-player padding easy) ---
    for i in range(num_players):
        _fill_player_token(state, buffer, tok, i, num_players)
        tok += 1


# =============================================================================
# PLAYER TOKEN
# =============================================================================

cdef void _fill_player_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int player_id,
    int num_players,
) noexcept nogil:
    # Feature offsets within the player token. ``OFF_IS_SELECTED`` is set iff
    # this player is the current active_player selector. The three
    # OFF_NUM_*/OFF_TOTAL_* slots are relational-summary scalars: aggregate
    # views of the relational-tail multihots, kept in the projection so the
    # trunk doesn't have to learn to count via attention.
    cdef int OFF_ATTN_MASK         = 0
    cdef int OFF_IS_SELECTED       = 1
    cdef int OFF_TURN_ORDER        = 2    # 5 slots
    cdef int OFF_HAS_PASSED        = 7
    cdef int OFF_CASH              = 8
    cdef int OFF_NET_WORTH         = 9
    cdef int OFF_LIQUIDITY         = 10
    cdef int OFF_INCOME            = 11
    cdef int OFF_AUC_HIGH          = 12
    cdef int OFF_AUC_STARTER       = 13
    cdef int OFF_ROUND_TRIPS       = 14
    cdef int OFF_SHARES            = 15   # 8 slots
    # --- relational summary ---
    cdef int OFF_NUM_COMPANIES     = 23
    cdef int OFF_NUM_PRESIDENCIES  = 24
    cdef int OFF_TOTAL_SHARES      = 25
    # --- relational tail ---
    cdef int OFF_COMPANIES         = 26   # 36 slots

    cdef int player_base = LAYOUT.players_offset + player_id * PLAYER_FIELDS.size
    cdef int turn_order = <int>state._data[player_base + PLAYER_FIELDS.turn_order]
    cdef int has_passed = <int>state._data[player_base + PLAYER_FIELDS.has_passed]
    cdef int cash = <int>state._data[player_base + PLAYER_FIELDS.cash]
    cdef int net_worth = <int>state._data[player_base + PLAYER_FIELDS.net_worth]
    cdef int liquidity = <int>state._data[player_base + PLAYER_FIELDS.liquidity]
    cdef int income = <int>state._data[player_base + PLAYER_FIELDS.income]
    cdef int c, shares, buys, sells, roundtrip_flag, total_shares
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int high_bidder
    cdef int starter
    cdef int num_companies_owned
    cdef int num_presidencies

    assert 0 <= player_id < MAX_MODEL_PLAYERS, \
        f"_fill_player_token: player_id {player_id} out of range"

    buffer[tok, OFF_ATTN_MASK] = 1.0

    # Turn order one-hot (padded to 5 slots)
    assert 0 <= turn_order < MAX_MODEL_PLAYERS, \
        f"_fill_player_token: turn_order {turn_order} out of range for player {player_id}"
    buffer[tok, OFF_TURN_ORDER + turn_order] = 1.0

    # Has passed
    buffer[tok, OFF_HAS_PASSED] = 1.0 if has_passed else 0.0

    # Financials
    buffer[tok, OFF_CASH] = <float>cash / CASH_DIVISOR
    buffer[tok, OFF_NET_WORTH] = <float>net_worth / NET_WORTH_DIVISOR
    buffer[tok, OFF_LIQUIDITY] = <float>liquidity / NET_WORTH_DIVISOR
    buffer[tok, OFF_INCOME] = <float>income / ENTITY_INCOME_DIVISOR

    if phase == <int>GamePhases.PHASE_BID:
        high_bidder = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder]
        starter = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_starter]

        # high_bidder legitimately == -1 on the opening bid; starter is
        # stamped whenever we reach PHASE_BID and must be a live player slot.
        assert -1 <= high_bidder < MAX_MODEL_PLAYERS, \
            f"_fill_player_token: auction_high_bidder {high_bidder} out of [-1, {MAX_MODEL_PLAYERS})"
        if high_bidder == player_id:
            buffer[tok, OFF_AUC_HIGH] = 1.0

        assert 0 <= starter < MAX_MODEL_PLAYERS, \
            f"_fill_player_token: auction_starter {starter} unset or out of range in BID"
        if starter == player_id:
            buffer[tok, OFF_AUC_STARTER] = 1.0

    # Per-corp: shares, the aggregated round-trip flag, and total-shares /
    # presidency aggregates. Presidency count mirrors the gating used by
    # the corp-token's president one-hot (active && !receivership) so the
    # scalar matches the relational-tail signal.
    roundtrip_flag = 0
    total_shares = 0
    num_presidencies = 0
    for c in range(NUM_CORPS):
        shares = <int>state._data[player_base + PLAYER_FIELDS.owned_shares + c]
        buys = <int>state._data[player_base + PLAYER_FIELDS.share_buys + c]
        sells = <int>state._data[player_base + PLAYER_FIELDS.share_sells + c]

        buffer[tok, OFF_SHARES + c] = <float>shares / SHARE_DIVISOR
        total_shares += shares

        # Round-trip threshold: once the player hits the buy+sell cap on
        # any corp, any further buy/sell in that corp is illegal this turn.
        if buys >= ROUNDTRIP_LIMIT or sells >= ROUNDTRIP_LIMIT:
            roundtrip_flag = 1

        if (
            corp_is_active(state, c)
            and not corp_is_in_receivership(state, c)
            and corp_president_id(state, c) == player_id
        ):
            num_presidencies += 1

    buffer[tok, OFF_ROUND_TRIPS] = 1.0 if roundtrip_flag else 0.0
    buffer[tok, OFF_TOTAL_SHARES] = <float>total_shares / TOTAL_SHARES_DIVISOR
    buffer[tok, OFF_NUM_PRESIDENCIES] = <float>num_presidencies / PRESIDENCIES_DIVISOR

    # Owned companies (36 flags) + count.
    cdef int comp_loc
    cdef int comp_owner
    num_companies_owned = 0
    for c in range(NUM_COMPANIES):
        comp_loc = company_location(state, c)
        if comp_loc != <int>LOC_PLAYER:
            continue
        comp_owner = company_owner_id(state, c)
        if comp_owner == player_id:
            buffer[tok, OFF_COMPANIES + c] = 1.0
            num_companies_owned += 1
    buffer[tok, OFF_NUM_COMPANIES] = <float>num_companies_owned / OWNED_COMPANIES_DIVISOR

    # Active-player selector flag.
    cdef int active_player = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_player]
    assert -1 <= active_player < MAX_MODEL_PLAYERS, \
        f"_fill_player_token: active_player {active_player} out of [-1, {MAX_MODEL_PLAYERS})"
    if active_player == player_id:
        buffer[tok, OFF_IS_SELECTED] = 1.0


# =============================================================================
# CORP TOKEN
# =============================================================================

cdef void _fill_corp_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int corp_id,
    int num_players,
) noexcept nogil:
    # Feature offsets within the corp token. Two distinct "active"-ish
    # bits: ``OFF_ACTIVE`` is the lifecycle float flag
    # (``corp_is_active``, set once the corp is floated and operational);
    # ``OFF_IS_SELECTED`` is the decision-flow selector (set iff this
    # corp is the current active_corp).
    # Corp identity is inferred from row order and added by the model as
    # ``corp_id_embed``. Relational fields are grouped at the end for
    # model-side replacement.
    cdef int OFF_ATTN_MASK         = 0
    cdef int OFF_IS_SELECTED       = 1
    cdef int OFF_ACTIVE            = 2
    cdef int OFF_IN_RECV           = 3
    cdef int OFF_PASSED_ACQ        = 4
    cdef int OFF_UNISSUED          = 5
    cdef int OFF_ISSUED            = 6
    cdef int OFF_BANK              = 7
    cdef int OFF_PRICE_IDX         = 8    # 27 slots
    cdef int OFF_SHARE_PRICE       = 35
    cdef int OFF_PENDING_MOVE      = 36
    cdef int OFF_CASH              = 37
    cdef int OFF_ACQ_PROCEEDS      = 38
    cdef int OFF_INCOME            = 39
    cdef int OFF_STARS             = 40
    cdef int OFF_RAW_REVENUE       = 41
    cdef int OFF_SYNERGY           = 42
    cdef int OFF_COO_COST          = 43
    cdef int OFF_ABILITY           = 44
    cdef int OFF_ACQ_OFFER         = 45
    cdef int OFF_DIV_REMAIN        = 46
    cdef int OFF_ISSUE_REMAIN      = 47
    cdef int OFF_IPO_REMAIN        = 48
    cdef int OFF_BUY_IMPACT        = 49
    cdef int OFF_SELL_IMPACT       = 50
    # --- relational summary ---
    cdef int OFF_NUM_OPERATIONAL   = 51
    cdef int OFF_NUM_ACQ_PILE      = 52
    cdef int OFF_NUM_TOTAL         = 53
    # --- relational tail ---
    cdef int OFF_PRESIDENT         = 54   # 5 slots
    cdef int OFF_COMPANIES         = 59   # 36 slots

    cdef bint active = corp_is_active(state, corp_id)
    cdef int price_idx, president, company_id, current_idx, new_idx, delta
    cdef int num_operational, num_acq_pile
    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int offer_corp

    assert -1 <= active_corp < NUM_CORPS, \
        f"_fill_corp_token: active_corp {active_corp} out of [-1, {NUM_CORPS})"

    # Entity tokens are always visible to model attention. Lifecycle state is
    # carried separately by OFF_ACTIVE and the other feature fields.
    buffer[tok, OFF_ATTN_MASK] = 1.0

    buffer[tok, OFF_ACTIVE] = 1.0 if active else 0.0

    # Flags and share counts are meaningful regardless of active status
    buffer[tok, OFF_IN_RECV] = 1.0 if corp_is_in_receivership(state, corp_id) else 0.0
    buffer[tok, OFF_PASSED_ACQ] = 1.0 if corp_has_passed_acq_offer(state, corp_id) else 0.0
    buffer[tok, OFF_UNISSUED] = <float>corp_unissued_shares(state, corp_id) / SHARE_DIVISOR
    buffer[tok, OFF_ISSUED] = <float>corp_issued_shares(state, corp_id) / SHARE_DIVISOR
    buffer[tok, OFF_BANK] = <float>corp_bank_shares(state, corp_id) / SHARE_DIVISOR

    if active:
        price_idx = corp_price_index(state, corp_id)
        assert 0 <= price_idx < NUM_MARKET_SPACES, \
            f"_fill_corp_token: price_idx {price_idx} out of range for active corp {corp_id}"
        buffer[tok, OFF_PRICE_IDX + price_idx] = 1.0
        buffer[tok, OFF_SHARE_PRICE] = <float>corp_share_price(state, corp_id) / SHARE_PRICE_DIVISOR
        buffer[tok, OFF_PENDING_MOVE] = <float>corp_pending_price_move(state, corp_id) / IMPACT_DIVISOR
        buffer[tok, OFF_CASH] = <float>corp_cash(state, corp_id) / CASH_DIVISOR
        buffer[tok, OFF_ACQ_PROCEEDS] = <float>corp_acquisition_proceeds(state, corp_id) / CASH_DIVISOR
        buffer[tok, OFF_INCOME] = <float>corp_income(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_STARS] = <float>corp_total_stars(state, corp_id) / CORP_STAR_DIVISOR
        buffer[tok, OFF_RAW_REVENUE] = <float>corp_raw_revenue(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_SYNERGY] = <float>corp_synergy_income(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_COO_COST] = <float>corp_coo_cost(state, corp_id) / ENTITY_INCOME_DIVISOR
        buffer[tok, OFF_ABILITY] = <float>corp_ability_income(state, corp_id) / ENTITY_INCOME_DIVISOR

        # President one-hot (inactive/receivership corps leave this zero)
        if not corp_is_in_receivership(state, corp_id):
            president = corp_president_id(state, corp_id)
            assert 0 <= president < MAX_MODEL_PLAYERS, \
                f"_fill_corp_token: president {president} out of range for active corp {corp_id}"
            buffer[tok, OFF_PRESIDENT + president] = 1.0

        # Owned companies (36 flags — owned OR in acquisition pile) plus
        # the relational-summary aggregate counts. Active-only: inactive /
        # pre-float corps own nothing, so the slots stay at 0 (matches the
        # rest of the active-gated fields above).
        num_operational = 0
        num_acq_pile = 0
        for company_id in range(NUM_COMPANIES):
            if corp_owns_company(state, corp_id, company_id):
                buffer[tok, OFF_COMPANIES + company_id] = 1.0
                num_operational += 1
            elif corp_has_acquisition_company(state, corp_id, company_id):
                buffer[tok, OFF_COMPANIES + company_id] = 1.0
                num_acq_pile += 1
        buffer[tok, OFF_NUM_OPERATIONAL] = (
            <float>num_operational / OWNED_COMPANIES_DIVISOR
        )
        buffer[tok, OFF_NUM_ACQ_PILE] = (
            <float>num_acq_pile / OWNED_COMPANIES_DIVISOR
        )
        buffer[tok, OFF_NUM_TOTAL] = (
            <float>(num_operational + num_acq_pile) / OWNED_COMPANIES_DIVISOR
        )

    if phase == <int>GamePhases.PHASE_ACQ_OFFER:
        offer_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.acq_offer_corp]
        assert 0 <= offer_corp < NUM_CORPS, \
            f"_fill_corp_token: acq_offer_corp {offer_corp} unset or out of range in ACQ_OFFER"
        if offer_corp == corp_id:
            buffer[tok, OFF_ACQ_OFFER] = 1.0

    if phase == <int>GamePhases.PHASE_DIVIDENDS:
        if (<int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.dividend_remaining + corp_id]) != 0:
            buffer[tok, OFF_DIV_REMAIN] = 1.0

    if phase == <int>GamePhases.PHASE_ISSUE_SHARES:
        if (<int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.issue_remaining + corp_id]) != 0:
            buffer[tok, OFF_ISSUE_REMAIN] = 1.0

    if phase == <int>GamePhases.PHASE_IPO or phase == <int>GamePhases.PHASE_PAR:
        if not corp_is_active(state, corp_id):
            buffer[tok, OFF_IPO_REMAIN] = 1.0

    if phase == <int>GamePhases.PHASE_INVEST and active:
        current_idx = corp_price_index(state, corp_id)

        # Buy impact: delta to the next higher available market space.
        new_idx = market_find_next_higher_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[tok, OFF_BUY_IMPACT] = <float>delta / IMPACT_DIVISOR

        # Sell impact: delta to the next lower available market space.
        new_idx = market_find_next_lower_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[tok, OFF_SELL_IMPACT] = <float>delta / IMPACT_DIVISOR

    # Active-corp selector flag (independent of the lifecycle ``OFF_ACTIVE``).
    if active_corp == corp_id:
        buffer[tok, OFF_IS_SELECTED] = 1.0


# =============================================================================
# COMPANY TOKEN
# =============================================================================

cdef void _fill_company_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int company_id,
    int num_players,
) noexcept nogil:
    # Feature offsets within the company token. Static game-setup data plus
    # non-relational dynamic fields come first; the mutually-exclusive
    # ownership groups form a relational tail.
    cdef int OFF_ATTN_MASK      = 0
    cdef int OFF_IS_SELECTED    = 1
    cdef int OFF_LOW_PRICE      = 2
    cdef int OFF_FACE_VALUE     = 3
    cdef int OFF_HIGH_PRICE     = 4
    cdef int OFF_LOW_HIGH_DIFF  = 5
    cdef int OFF_BASE_INCOME    = 6
    cdef int OFF_STARS          = 7
    cdef int OFF_ADJ_INCOME     = 8
    cdef int OFF_AT_REMOVED     = 9
    cdef int OFF_AT_AUCTION     = 10
    cdef int OFF_AT_REVEALED    = 11
    cdef int OFF_AT_CORP_ACQ    = 12
    cdef int OFF_ACQ_SYNERGY    = 13
    cdef int OFF_OWNER_CORP     = 14   # 8 slots
    cdef int OFF_OWNER_PLAYER   = 22   # 5 slots (padded for num_players < 5)
    cdef int OFF_OWNER_FI       = 27

    cdef int loc = company_location(state, company_id)
    buffer[tok, OFF_ATTN_MASK] = 1.0

    # Static data
    buffer[tok, OFF_LOW_PRICE] = <float>COMPANY_LOW_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_FACE_VALUE] = <float>COMPANY_FACE_VALUE[company_id] / COMPANY_PRICE_DIVISOR
    buffer[tok, OFF_HIGH_PRICE] = <float>COMPANY_HIGH_PRICE[company_id] / COMPANY_PRICE_DIVISOR
    # low_high_diff: count of valid ACQ_SELECT_PRICE offsets for this company
    # (high - low + 1). Matches the ``max_off`` ceiling the acq-price head
    # conditions on. Max 51 (CDG: 80 - 30 + 1).
    buffer[tok, OFF_LOW_HIGH_DIFF] = (
        <float>(COMPANY_HIGH_PRICE[company_id] - COMPANY_LOW_PRICE[company_id] + 1)
        / PRICE_RANGE_DIVISOR
    )
    buffer[tok, OFF_BASE_INCOME] = <float>COMPANY_INCOME[company_id] / COMPANY_INCOME_DIVISOR
    buffer[tok, OFF_STARS] = <float>COMPANY_STARS[company_id] / COMPANY_STAR_DIVISOR

    # CoO-adjusted income (may be negative; a company's income can be
    # pushed below zero by CoO-dependent penalties).
    buffer[tok, OFF_ADJ_INCOME] = (
        <float>company_adjusted_income(state, company_id) / COMPANY_INCOME_DIVISOR
    )

    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int active_corp
    cdef int delta
    if phase == <int>GamePhases.PHASE_ACQ_SELECT_COMPANY:
        active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
        assert 0 <= active_corp < NUM_CORPS, \
            f"_fill_company_token: active_corp {active_corp} unset or out of range in ACQ_SELECT_COMPANY"
        assert corp_is_active(state, active_corp), \
            f"_fill_company_token: active_corp {active_corp} not active in ACQ_SELECT_COMPANY"
        if (
            not corp_owns_company(state, active_corp, company_id)
            and not corp_has_acquisition_company(state, active_corp, company_id)
        ):
            delta = corp_candidate_synergy_delta(state, active_corp, company_id)
            buffer[tok, OFF_ACQ_SYNERGY] = <float>delta / ENTITY_INCOME_DIVISOR

    # Active-company selector flag.
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    assert -1 <= active_company < NUM_COMPANIES, \
        f"_fill_company_token: active_company {active_company} out of [-1, {NUM_COMPANIES})"
    if active_company == company_id:
        buffer[tok, OFF_IS_SELECTED] = 1.0

    # Location dispatch: writes exactly one flag (for the at_* group) or
    # one slot (for the owner_* group), depending on the company's
    # current location. LOC_DECK / LOC_EXCLUDED below the revealed-tier
    # threshold leave everything zero.
    #
    # LOC_REMOVED additionally absorbs LOC_EXCLUDED once CoO has advanced
    # past this company's star tier — the deck is past the colour group
    # so the exclusion is publicly observable. Flagging LOC_EXCLUDED
    # unconditionally would leak which specific cards were cut at setup
    # (information the players don't have until the deck has advanced
    # past that tier).
    cdef int coo = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]
    cdef int owner_id
    if loc == <int>LOC_REMOVED:
        buffer[tok, OFF_AT_REMOVED] = 1.0
    elif loc == <int>LOC_EXCLUDED and coo > COMPANY_STARS[company_id]:
        buffer[tok, OFF_AT_REMOVED] = 1.0
    elif loc == <int>LOC_AUCTION:
        buffer[tok, OFF_AT_AUCTION] = 1.0
    elif loc == <int>LOC_REVEALED:
        buffer[tok, OFF_AT_REVEALED] = 1.0
    elif loc == <int>LOC_CORP_ACQ:
        buffer[tok, OFF_AT_CORP_ACQ] = 1.0
        owner_id = company_owner_id(state, company_id)
        assert 0 <= owner_id < NUM_CORPS, \
            f"_fill_company_token: LOC_CORP_ACQ owner {owner_id} out of [0, {NUM_CORPS}) for company {company_id}"
        buffer[tok, OFF_OWNER_CORP + owner_id] = 1.0
    elif loc == <int>LOC_CORP:
        owner_id = company_owner_id(state, company_id)
        assert 0 <= owner_id < NUM_CORPS, \
            f"_fill_company_token: LOC_CORP owner {owner_id} out of [0, {NUM_CORPS}) for company {company_id}"
        buffer[tok, OFF_OWNER_CORP + owner_id] = 1.0
    elif loc == <int>LOC_PLAYER:
        owner_id = company_owner_id(state, company_id)
        assert 0 <= owner_id < MAX_MODEL_PLAYERS, \
            f"_fill_company_token: LOC_PLAYER owner {owner_id} out of [0, {MAX_MODEL_PLAYERS}) for company {company_id}"
        buffer[tok, OFF_OWNER_PLAYER + owner_id] = 1.0
    elif loc == <int>LOC_FI:
        buffer[tok, OFF_OWNER_FI] = 1.0


# =============================================================================
# FI TOKEN
# =============================================================================

cdef void _fill_fi_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_ATTN_MASK     = 0
    cdef int OFF_CASH          = 1
    cdef int OFF_INCOME        = 2
    # --- relational summary ---
    cdef int OFF_NUM_COMPANIES = 3
    # --- relational tail ---
    cdef int OFF_COMPANIES     = 4   # 36 slots

    cdef int cash = <int>state._data[LAYOUT.fi_offset + FI_OFFSETS.cash]
    cdef int income = <int>state._data[LAYOUT.fi_offset + FI_OFFSETS.income]
    cdef int c
    cdef int num_companies_owned = 0

    buffer[tok, OFF_ATTN_MASK] = 1.0
    buffer[tok, OFF_CASH] = <float>cash / CASH_DIVISOR
    buffer[tok, OFF_INCOME] = <float>income / ENTITY_INCOME_DIVISOR

    for c in range(NUM_COMPANIES):
        if company_location(state, c) == <int>LOC_FI:
            buffer[tok, OFF_COMPANIES + c] = 1.0
            num_companies_owned += 1
    buffer[tok, OFF_NUM_COMPANIES] = (
        <float>num_companies_owned / OWNED_COMPANIES_DIVISOR
    )


# =============================================================================
# MARKET INFO TOKEN
# =============================================================================

cdef void _fill_market_info_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # First 27 slots: static $0..$75 slot prices normalized by
    # SHARE_PRICE_DIVISOR. Next 27 slots: per-space availability flags
    # (int16 0/1 read directly from the market section; direct cast is
    # enough, no ternary needed).
    cdef int OFF_ATTN_MASK    = 0
    cdef int OFF_SLOT_PRICES  = 1                   # 27 slots
    cdef int OFF_AVAILABILITY = 1 + NUM_MARKET_SPACES   # 27 slots
    cdef int i
    buffer[tok, OFF_ATTN_MASK] = 1.0
    for i in range(NUM_MARKET_SPACES):
        buffer[tok, OFF_SLOT_PRICES + i] = (
            <float>MARKET_PRICES[i] / SHARE_PRICE_DIVISOR
        )
        buffer[tok, OFF_AVAILABILITY + i] = (
            <float>state._data[LAYOUT.market_offset + i]
        )


# =============================================================================
# GLOBAL INFO TOKEN
# =============================================================================

cdef void _fill_global_info_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
    int num_players,
) noexcept nogil:
    # Bundled game-level scalars: decision-phase one-hot, CoO one-hot,
    # end-card flag, a normalized cards-remaining scalar, and the
    # num_players one-hot (slot 0 = 3p, slot 1 = 4p, slot 2 = 5p).
    # Automated / terminal engine phases map to decision_phase == -1 and
    # leave the phase one-hot all-zero.
    cdef int OFF_ATTN_MASK    = 0
    cdef int OFF_PHASE        = 1    # 11 slots (one per DecisionPhase)
    cdef int OFF_COO          = 12   # 7 slots (CoO level 1..7 → slots 0..6)
    cdef int OFF_END_CARD     = 19
    cdef int OFF_CARDS_REM    = 20
    cdef int OFF_NUM_PLAYERS  = 21   # 3 slots (3p/4p/5p)

    cdef int phase = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.phase]
    cdef int decision_phase
    cdef int coo = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.coo_level]
    cdef int end_card = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.end_card_flipped]
    cdef int cards_rem = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.cards_remaining]

    buffer[tok, OFF_ATTN_MASK] = 1.0

    assert 0 <= phase < <int>GameConstants.NUM_PHASES, \
        f"_fill_global_info_token: corrupt engine phase {phase}"
    decision_phase = ENGINE_TO_DECISION_PHASE[phase]
    # Automated / terminal engine phases map to -1; anything else is a
    # corrupt ENGINE_TO_DECISION_PHASE entry.
    assert -1 <= decision_phase < NUM_DECISION_PHASES, \
        f"_fill_global_info_token: decision_phase {decision_phase} out of [-1, {NUM_DECISION_PHASES}) for engine phase {phase}"
    if decision_phase >= 0:
        buffer[tok, OFF_PHASE + decision_phase] = 1.0

    # CoO one-hot: levels 1-7 → slots 0-6
    assert 1 <= coo <= NUM_COO_LEVELS, \
        f"_fill_global_info_token: coo_level {coo} out of [1, 7]"
    buffer[tok, OFF_COO + (coo - 1)] = 1.0

    buffer[tok, OFF_END_CARD] = 1.0 if end_card else 0.0
    buffer[tok, OFF_CARDS_REM] = <float>cards_rem / <float>NUM_COMPANIES

    # Training scope is 3-5p; the entry-point assert already rejects
    # anything outside that range, so re-assert here rather than silently
    # leaving the slot zero for mis-sized states.
    assert 3 <= num_players <= 5, \
        f"_fill_global_info_token: num_players {num_players} out of [3, 5]"
    buffer[tok, OFF_NUM_PLAYERS + (num_players - 3)] = 1.0


# =============================================================================
# PHASE-SPECIFIC TOKENS
# =============================================================================

cdef void _fill_invest_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_ATTN_MASK = 0
    cdef int OFF_PASSES    = 1

    cdef int passes = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.consecutive_passes]
    buffer[tok, OFF_ATTN_MASK] = 1.0
    buffer[tok, OFF_PASSES] = <float>passes / CONSECUTIVE_PASSES_DIVISOR


cdef void _fill_auction_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    # Price slots carry the *minimum legal next bid* rather than the last
    # bid placed. On the opening bid (is_first_bid == 1) the minimum is
    # face_value (offset 0); afterwards it's current_bid + 1.
    cdef int OFF_ATTN_MASK     = 0
    cdef int OFF_MIN_BID_IDX   = 1
    cdef int OFF_MIN_BID_VALUE = 2
    cdef int OFF_IS_FIRST_BID  = 3

    cdef int auction_price = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_price]
    cdef int high_bidder = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.auction_high_bidder]
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef bint is_first_bid = high_bidder < 0
    cdef int face_value, min_bid, min_offset

    # active_company is seeded in INVEST before the BID transition and only
    # cleared at auction resolution; any BID state without it is a driver bug.
    buffer[tok, OFF_ATTN_MASK] = 1.0
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_auction_token: active_company {active_company} unset or out of range in BID"

    face_value = COMPANY_FACE_VALUE[active_company]
    if is_first_bid:
        min_bid = face_value
    else:
        min_bid = auction_price + 1
    min_offset = min_bid - face_value
    buffer[tok, OFF_MIN_BID_IDX] = <float>min_offset / <float>AUCTION_CAP_INT
    buffer[tok, OFF_MIN_BID_VALUE] = <float>min_bid / COMPANY_PRICE_DIVISOR

    buffer[tok, OFF_IS_FIRST_BID] = 1.0 if is_first_bid else 0.0

    assert -1 <= high_bidder < MAX_MODEL_PLAYERS, \
        f"_fill_auction_token: auction_high_bidder {high_bidder} out of [-1, {MAX_MODEL_PLAYERS})"


cdef void _fill_dividend_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_ATTN_MASK = 0
    cdef int OFF_IMPACT    = 1    # 26 slots (amounts 0..25)

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int amount, price_move

    # Dividend impacts. Driver contract for PHASE_DIVIDENDS: active_corp is
    # always a live, active corp when this token is filled.
    buffer[tok, OFF_ATTN_MASK] = 1.0
    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_dividend_token: active_corp {active_corp} unset or out of range"
    assert corp_is_active(state, active_corp), \
        f"_fill_dividend_token: active_corp {active_corp} not active"
    for amount in range(MAX_DIVIDEND):
        price_move = _simulate_dividend_price_move(state, active_corp, amount)
        buffer[tok, OFF_IMPACT + amount] = <float>price_move / IMPACT_DIVISOR


cdef void _fill_issue_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_ATTN_MASK = 0
    cdef int OFF_IMPACT    = 1

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int current_idx, new_idx, delta

    # Issue impact: issuing one share drops the corp's price like a sell,
    # except Stock Masters (SM) has no price change on issue. Driver contract
    # for PHASE_ISSUE_SHARES: active_corp is always a live, active corp.
    buffer[tok, OFF_ATTN_MASK] = 1.0
    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_issue_token: active_corp {active_corp} unset or out of range"
    assert corp_is_active(state, active_corp), \
        f"_fill_issue_token: active_corp {active_corp} not active"
    if active_corp == <int>CorpIndices.CORP_SM:
        buffer[tok, OFF_IMPACT] = 0.0
    else:
        current_idx = corp_price_index(state, active_corp)
        new_idx = market_find_next_lower_space(state, current_idx)
        delta = new_idx - current_idx
        buffer[tok, OFF_IMPACT] = <float>delta / IMPACT_DIVISOR


cdef void _fill_par_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_ATTN_MASK = 0
    cdef int OFF_PAR_DATA  = 1    # 14 tuples: player_cash, corp_cash, issued_shares
    cdef int PAR_STRIDE    = 3

    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef int face_value, star_tier, par_index, base
    cdef int float_shares, market_index, player_payment, corp_cash_result, issued

    # Driver contract for PHASE_IPO / PHASE_PAR: active_company is always
    # stamped to the selected corp's target company, and every company's
    # star tier is 1..5 by static data.
    buffer[tok, OFF_ATTN_MASK] = 1.0
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_par_token: active_company {active_company} unset or out of range"
    face_value = COMPANY_FACE_VALUE[active_company]
    star_tier = COMPANY_STARS[active_company]
    assert 1 <= star_tier <= MAX_MODEL_PLAYERS, \
        f"_fill_par_token: star_tier {star_tier} out of [1, 5] for company {active_company}"
    for par_index in range(NUM_PAR_PRICES):
        if PAR_PRICE_VALID[star_tier - 1][par_index] == 0:
            continue
        # Canonical IPO simulation — same helper used by
        # ``phases/ipo.pyx::_process_ipo``.
        (float_shares, market_index, player_payment,
         corp_cash_result, issued) = _simulate_float(face_value, par_index)

        base = OFF_PAR_DATA + par_index * PAR_STRIDE
        buffer[tok, base] = (
            <float>player_payment / CASH_DIVISOR
        )
        buffer[tok, base + 1] = (
            <float>corp_cash_result / CASH_DIVISOR
        )
        buffer[tok, base + 2] = (
            <float>issued / FLOAT_SHARES_MAX
        )


cdef void _fill_acq_offer_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    cdef int OFF_ATTN_MASK   = 0
    cdef int OFF_PRICE_IDX   = 1
    cdef int OFF_PRICE_VALUE = 2
    cdef int OFF_FI_COMPANY  = 3

    cdef int offer_price = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.acq_offer_price]
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]
    cdef int low_price, offset

    # Driver contract for PHASE_ACQ_OFFER: active_company is stamped by
    # SELECT_COMPANY. The offer_corp selector is emitted on the corp tokens.
    buffer[tok, OFF_ATTN_MASK] = 1.0
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_acq_offer_token: active_company {active_company} unset or out of range"

    low_price = COMPANY_LOW_PRICE[active_company]
    offset = offer_price - low_price
    buffer[tok, OFF_PRICE_IDX] = <float>offset / <float>ACQ_PRICE_OFFSETS
    if company_location(state, active_company) == <int>LOC_FI:
        buffer[tok, OFF_FI_COMPANY] = 1.0

    buffer[tok, OFF_PRICE_VALUE] = <float>offer_price / COMPANY_PRICE_DIVISOR

cdef void _fill_acq_price_info_token(
    GameState state,
    float[:, ::1] buffer,
    int tok,
) noexcept nogil:
    """Fill the acq_price_info token during PHASE_ACQ_SELECT_PRICE.

    Kept minimal: every (active_corp, active_company)-level scalar is
    already on the corp / company / active-entity / company-location
    tokens and reaches the price head via attention. The three slots
    here carry only what can't be read off those tokens directly:
      * max_offset     — ACQ price-offset count for the target, same as
                         the company token's ``low_high_diff`` field
                         ((high - low + 1), normalized by PRICE_RANGE_DIVISOR).
      * fi_flag        — 1 if the target company is FI-owned, else 0.
                         A hard discontinuity for the head (FI sale is a
                         single fixed-price action, no offset to pick).
      * total_synergies — marginal synergy income the corp would gain
                         by adding this company to its portfolio,
                         normalized by ENTITY_INCOME_DIVISOR.
    """
    cdef int OFF_ATTN_MASK       = 0
    cdef int OFF_MAX_OFFSET      = 1
    cdef int OFF_FI_FLAG         = 2
    cdef int OFF_TOTAL_SYNERGIES = 3

    cdef int active_corp = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_corp]
    cdef int active_company = <int>state._data[LAYOUT.turn_offset + TURN_OFFSETS.active_company]

    # Driver contract for PHASE_ACQ_SELECT_PRICE: both selectors stamped,
    # corp active. SELECT_COMPANY's enumerator also rejects receivership
    # sellers, so any LOC_CORP target reaching here has a live seller.
    buffer[tok, OFF_ATTN_MASK] = 1.0
    assert 0 <= active_corp < NUM_CORPS, \
        f"_fill_acq_price_info_token: active_corp {active_corp} unset or out of range"
    assert 0 <= active_company < NUM_COMPANIES, \
        f"_fill_acq_price_info_token: active_company {active_company} unset or out of range"
    assert corp_is_active(state, active_corp), \
        f"_fill_acq_price_info_token: active_corp {active_corp} not active"

    cdef int low_price = COMPANY_LOW_PRICE[active_company]
    cdef int high_price = COMPANY_HIGH_PRICE[active_company]
    cdef int synergy_delta = corp_candidate_synergy_delta(
        state, active_corp, active_company,
    )

    buffer[tok, OFF_MAX_OFFSET] = (
        <float>(high_price - low_price + 1) / PRICE_RANGE_DIVISOR
    )
    if company_location(state, active_company) == <int>LOC_FI:
        buffer[tok, OFF_FI_FLAG] = 1.0
    buffer[tok, OFF_TOTAL_SYNERGIES] = (
        <float>synergy_delta / ENTITY_INCOME_DIVISOR
    )
