# cython: language_level=3
"""
Cython helper module re-exports.

Allows convenient cimport of all helper structs and functions:

    from helpers cimport (
        PlayerOffsets, get_player_offsets, get_player_cash, ...
        CorpOffsets, get_corp_offsets, is_corp_active, ...
        is_market_space_available, find_next_higher_price_index, ...
        DividendTurnOffsets, get_dividend_turn_offsets, ...
    )
"""

# Re-export from player module
from helpers.player cimport (
    PlayerOffsets,
    get_player_offsets,
    get_player_cash,
    set_player_cash,
    add_player_cash,
    get_player_shares,
    set_player_shares,
    player_owns_company,
    set_player_owns_company,
    is_player_president,
    set_player_president,
    get_share_buys,
    increment_share_buys,
    get_share_sells,
    increment_share_sells,
    get_roundtrips,
    clear_roundtrip_tracking,
    calculate_player_net_worth,
    update_all_player_net_worths,
)

# Re-export from corp module
from helpers.corp cimport (
    CorpOffsets,
    get_corp_offsets,
    is_corp_active,
    set_corp_active,
    get_corp_cash,
    set_corp_cash,
    add_corp_cash,
    get_corp_issued_shares,
    set_corp_issued_shares,
    get_corp_bank_shares,
    set_corp_bank_shares,
    get_corp_unissued_shares,
    set_corp_unissued_shares,
    get_corp_share_price,
    set_corp_share_price,
    get_corp_price_index,
    set_corp_price_index,
    is_corp_in_receivership,
    set_corp_in_receivership,
    corp_owns_company,
    set_corp_owns_company,
    get_corp_company_count,
    get_president_of_corp,
    set_active_player_to_president,
    find_corp_owning_company,
    calculate_corp_company_stars,
    calculate_target_stars,
    handle_corp_bankruptcy,
)

# Re-export from market module
from helpers.market cimport (
    is_market_space_available,
    set_market_space_available,
    find_next_higher_price_index,
    find_next_lower_price_index,
    find_adjusted_price_index,
)

# Re-export from turn module
from helpers.turn cimport (
    AuctionTurnOffsets,
    DividendTurnOffsets,
    IssueTurnOffsets,
    get_auction_turn_offsets,
    get_dividend_turn_offsets,
    get_issue_turn_offsets,
)
