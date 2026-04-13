# Rough token data spec:

## Global token
- Global token:
  - Num players (one-hot, 3 slots designating 3/4/5 players)
  - Phase (one-hot, 8 slots for each decision phase)
  - CoO level (one-hot, 7 slots for each CoO level)
  - End card flipped (scalar, 0/1)
  - Cards remaining (scalar, normalized by NUM_COMPANIES)
  
## Informational tokens (1 total)
- Market token:
  - Market availability (vector, 27 slots). 1 if corresponding market space is available (i.e., not occupied by corp), 0 otherwise. Note: first and last spaces ($0 and $75) always available.
  
## Phase-specific tokens (6 total)

**Overall note** - these should all be zeroed out if the game state is not in the relevant phase.

- Invest token:
  - Consecutive passes (normalized by 5 - the number of max players).
  - Buy share invest impacts (vector, 8 slots, normalized by IMPACT_DIVISOR). The number of market price indices that the corresponding corp would move if a share was bought.
  - Sell share invest impacts (vector, 8 slots, normalized by IMPACT_DIVISOR). The number of market price indices that the corresponding corp would move if a share was sold.
- Auction token:
  - Auction price index (scalar, should match the action encoding data: represents price offset from face value, normalized by AUCTION_CAP)
  - Auction price value (scalar, current auction price as a cash value. Normalized by COMPANY_PRICE_DIVISOR)
  - Auction high bidder (one-hot, 5 slots for 5 max players (0-padded for 3/4 players))
  - Auction starter (one-hot, 5 slots for 5 max players (0-padded for 3/4 players))
- Acquisition offer token:
  - Offer price index (scalar, normalized by 51.0). Matches acquisition action encoding - represents price offset from target company's low price. 
  - Offer price (scalar, normalized by COMPANY_PRICE_DIVISOR)
  - Offer corp (one-hot, 8 slots for each corp)
  - FI company (scalar, 0/1 if target company for offer is FI-owned)
- Dividends token:
  - Dividend impact (vector, 26 slots representing dividend amounts 0-25). For each slot, report the market price index impact (up or down) that would result for the active corp if the corresponding dividend amount was paid. Each value normalized by IMPACT_DIVISOR.
  - Dividend remaining (vector, 8 slots). Represents corps that have yet to act in this dividend phase.
- Issue token:
  - Issue impact (scalar, normalized by IMPACT_DIVISOR). Represents the price impact for the active corp if they were to issue a share.
  - Issue remaining (vector, 8 slots). Represents corps that have yet to act in this issue phase.
- IPO token:
  - Player cash required (vector, 14 slots for each valid par price). Represents the amount the active player needs to pay in order to float at each par price, 0 for invalid par prices.
  - Resulting corp cash (vector, 14 slots for each valid par price). Represents the amount of cash the corp will have after floating for each par price, 0 for invalid par prices.
  - Resulting issued shares (vector, 14 slots for each valid par price). Represents the amount of shares the corp will issue for each par price, 0 for invalid par prices. In practice these values will always be 0, 2 or 4, so normalize by 4.0 as the maximum.
  - IPO remaining (vector, 8 slots). Represents corps that have yet to act in this IPO phase.
  
## FI token
- Cash (normalized by CASH_DIVISOR)
- Income (normalized by ENTITY_INCOME_DIVISOR)
- Owned companies (vector, 36 slots - 1 if corresponding company is owned, 0 otherwise)

## Company tokens (36 total)
- Active company (scalar, 1/0). Whether `active_company` in TurnStateOffsets corresponds with this company.
- Company ID (one-hot, 36 slots for each company)
- Corp owner (one-hot, 8 slots for each corp). All zero if not corp-owned. Note: if the company is in a corp's acquisition pile, that also counts for this one-hot.
- Player owner (one-hot 5 slots for max 5 players, pad with 0 for 3/4 players). All zero if not player-owned.
- FI owned (scalar, 0/1 on whether company is owned by FI)
- Location: auction (scalar, 0/1 on whether company is available for auction)
- Location: revealed (scalar, 0/1 on whether company is revealed)
- Location: acquisition pile (scalar, 0/1 on whether company is in a corp's acquisition pile)
- Location: removed (scalar, 0/1 on whether company has been closed). Note: also include companies that must have been excluded based on the current CoO level (e.g., if top card on deck is orange resulting in a CoO change, and we haven't seen 3 red companies, we know that they are removed).
- Adjusted income (scalar, normalized by COMPANY_INCOME_DIVISOR). The company's income, given the current CoO level. Note: can be negative.
- Static data: company low price (scalar, normalized by COMPANY_PRICE_DIVISOR)
- Static data: company face value (scalar, normalized by COMPANY_PRICE_DIVISOR)
- Static data: company high price (scalar, normalized by COMPANY_PRICE_DIVISOR)
- Static data: company base income (scalar, normalized by COMPANY_INCOME_DIVISOR)
- Static data: company stars (scalar, normalized by 5.0 - the max number of base company stars)
- Static data: synergies (vector, 36 slots - value of synergy with another company in either direction. Normalized by COMPANY_INCOME_DIVISOR).

## Corp tokens (8 total)
- Active corp (scalar, 1/0). Whether `active_corp` in TurnStateOffsets corresponds with this corp.
- Corp ID (one-hot, 8 slots for each corp)
- Active (scalar, 1/0)
- In receivership (scalar, 1/0)
- Passed on ACQ_OFFER (scalar, 1/0)
- Unissued shares (normalized by SHARE_DIVISOR)
- Issued shares (normalized by SHARE_DIVISOR)
- Bank shares (normalized by SHARE_DIVISOR)
- Share price index (one-hot, 27 slots representing current market price index)
- Share price (scalar, normalized by SHARE_PRICE_DIVISOR)
- Pending price move (scalar, normalized by IMPACT_DIVISOR). 
- Cash (normalized by CASH_DIVISOR)
- Acquisition proceeds (normalized by CASH_DIVISOR)
- Income (normalized by ENTITY_INCOME_DIVISOR)
- Stars (normalized by CORP_STAR_DIVISOR). Note: this is *total* stars, owned companies + cash + SI bonus.
- Raw revenue (normalized by ENTITY_INCOME_DIVISOR)
- Synergy income (normalized by ENTITY_INCOME_DIVISOR)
- CoO cost (normalized by ENTITY_INCOME_DIVISOR)
- Ability income (normalized by ENTITY_INCOME_DIVISOR)
- President ID (one-hot, 5 slots for max 5 players, pad to 0 for 3/4 players). All 0 if inactive/in receivership.
- Owned companies (vector, 36 slots)

## Player tokens (3 to 5 total)
- Active player (scalar, 1/0). Whether `active_player` in TurnStateOffsets corresponds with this player.
- Player ID (one-hot, 5 slots for 5 max players, pad with 0 for 3/4 players)
- Turn order (one-hot, 5 slots for 5 max players, pad with 0 for 3/4 players)
- Has passed (scalar, 1/0)
- Cash (normalized by CASH_DIVISOR)
- Net worth (normalized by NET_WORTH_DIVISOR)
- Liquidity (normalized by NET_WORTH_DIVISOR)
- Income (normalized by ENTITY_INCOME_DIVISOR)
- Owned shares (vector, 8 slots for 8 corps, normalized by SHARE_DIVISOR)
- Round trips (scalar, 1 if any share buy/sell would be affected by the round trip limit, 0 otherwise)
- Share buys (vector 8 slots for 8 corps, normalized by SHARE_DIVISOR)
- Share sells (vector 8 slots for 8 corps, normalized by SHARE_DIVISOR)
- Presidencies (vector 8 slots for 8 corps, 1 if president 0 otherwise)
- Owned companies (vector, 36 slots)


