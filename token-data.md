# Rough token data spec

## Token order

Matches `core/token_data.pyx::_fill_buffer` and the per-type projections in
`nn/transformer.py::RSSTransformerNet._project_tokens`:

1. **Static data** (pre-populatable; pure game-setup data):
   `market_slot_prices`, `companies` (×36)
2. **Dynamic informational**:
   `market_availability`, `company_location` (×4: REMOVED, AUCTION, REVEALED,
   CORP_ACQ), `company_adjusted_income`, `FI`, `active_player`, `active_corp`,
   `active_company`, `phase`, `num_players`, `game_progress`
3. **Phase-specific** (zero unless the engine is in the matching phase):
   `invest`, `auction`, `dividend`, `issue`, `par`, `acq_offer`,
   `acq_price_info`
4. **Corp tokens** (×8)
5. **Player tokens** (×N, N ∈ {3, 4, 5})

Total tokens = `num_players + 65` (68 / 69 / 70 for 3p / 4p / 5p). The model
adds 7 learned per-phase pass anchors after projection; they don't appear in
the engine-side buffer.

Each token is zero-padded to `TOKEN_DIM = 92` (max width across all token
types, currently pinned by the Corp token).

---

## Static data tokens

### MarketSlotPrices token
- 27 static slot prices ($0..$75), each normalized by SHARE_PRICE_DIVISOR.

### Company tokens (36 total)
- Company ID (one-hot, 36 slots for each company)
- Static data: company low price (scalar, normalized by COMPANY_PRICE_DIVISOR)
- Static data: company face value (scalar, normalized by COMPANY_PRICE_DIVISOR)
- Static data: company high price (scalar, normalized by COMPANY_PRICE_DIVISOR)
- Static data: low_high_diff (scalar, normalized by PRICE_RANGE_DIVISOR — count of valid ACQ_SELECT_PRICE offsets for this company, `high - low + 1`, max 51 for CDG). Same quantity as `max_offset` on the AcqPriceInfo token.
- Static data: company base income (scalar, normalized by COMPANY_INCOME_DIVISOR)
- Static data: company stars (scalar, normalized by 5.0 — the max number of base company stars)
- Static data: synergies (vector, 36 slots — value of synergy with each other company in either direction, normalized by COMPANY_INCOME_DIVISOR).

The company token is now pure static game-setup data. Ownership, location,
CoO-adjusted income, and active-company selection each live on their own
dedicated tokens (`FI` / corp / player for ownership; the four
`company_location` tokens for location; `company_adjusted_income` for CoO
income; `active_company` for the turn selector).

---

## Dynamic informational tokens

### MarketAvailability token
- 27 slots. 1 if the corresponding market space is available (i.e., not occupied by a corp), 0 otherwise. First and last spaces ($0 and $75) always available.

### CompanyLocation tokens (4 total)
Four 36-slot bitmaps, one per target location (in buffer order):
1. LOC_REMOVED — 1 if the company has been closed / removed (includes companies excluded based on the current CoO level, e.g., when a deck-top orange forces a CoO change and we can infer the remaining red companies are gone).
2. LOC_AUCTION — 1 if the company is currently available for auction.
3. LOC_REVEALED — 1 if the company is revealed on deck (waiting for invest / ACQ).
4. LOC_CORP_ACQ — 1 if the company is in a corp's acquisition pile.

Player / corp / FI ownership is read from the corresponding owner tokens
(player / corp `Owned companies`, FI `Owned companies`) rather than duplicated
here.

### CompanyAdjustedIncome token
- Vector, 36 slots. Per-company CoO-adjusted income normalized by COMPANY_INCOME_DIVISOR. Values may be negative (a company's income can be pushed below zero by CoO-dependent penalties).

### FI token
- Cash (scalar, normalized by CASH_DIVISOR)
- Income (scalar, normalized by ENTITY_INCOME_DIVISOR)
- Owned companies (vector, 36 slots — 1 if the corresponding company is FI-owned, 0 otherwise)

### ActivePlayer token
- 5-slot one-hot (padded to MAX_MODEL_PLAYERS = 5). 1 at the currently-selected active player, 0 elsewhere. All-zero when no active player is selected (automated / terminal phases).

### ActiveCorp token
- 8-slot one-hot. 1 at the currently-selected active corp. All-zero when unset (e.g., ACQ_SELECT_CORP, CLOSING).

### ActiveCompany token
- 36-slot one-hot. 1 at the currently-selected active company. All-zero when unset (e.g., ACQ_SELECT_CORP, ACQ_SELECT_COMPANY, CLOSING).

### Phase token
- Decision phase one-hot (11 slots — one per `DecisionPhase`). All-zero in automated / terminal engine phases.

### NumPlayers token
- 3-slot one-hot (slot 0 = 3p, slot 1 = 4p, slot 2 = 5p).

### GameProgress token
- CoO level (one-hot, 7 slots — CoO levels 1..7 → slots 0..6)
- End card flipped (scalar, 0/1)
- Cards remaining (scalar, normalized by NUM_COMPANIES)

---

## Phase-specific tokens (7 total)

**Overall note** — these are zeroed when the game state is not in the relevant
phase.

### Invest token
- Consecutive passes (scalar, normalized by 5 — the max training player count).
- Buy-share invest impacts (vector, 8 slots, normalized by IMPACT_DIVISOR). Market price index delta for buying one share of the corresponding corp.
- Sell-share invest impacts (vector, 8 slots, normalized by IMPACT_DIVISOR). Market price index delta for selling one share of the corresponding corp.

### Auction token
- Minimum legal next-bid index (scalar, offset from the current auction company's face value, normalized by AUCTION_CAP). Equals 0 on the opening bid (bid at face_value) and `current_bid - face + 1` afterwards — the model can always score BID offsets against this floor.
- Minimum legal next-bid value (scalar, dollar amount of the minimum legal next bid, normalized by COMPANY_PRICE_DIVISOR).
- Is-first-bid flag (scalar, 1.0 on the opening bid — when `auction_high_bidder == -1` — and 0.0 afterwards. Pass / leave-auction is illegal while this flag is set.)
- Auction high bidder (one-hot, 5 slots for 5 max players (0-padded for 3/4 players). All zero on the opening bid since no bid has been placed yet.)
- Auction starter (one-hot, 5 slots for 5 max players (0-padded for 3/4 players)).

### Dividend token
- Dividend impact (vector, 26 slots for dividend amounts 0..25). Market price index delta (up or down) for the active corp if the corresponding dividend amount is paid, each value normalized by IMPACT_DIVISOR.
- Dividend remaining (vector, 8 slots — 1 per corp that has yet to act in this dividend phase).

### Issue token
- Issue impact (scalar, normalized by IMPACT_DIVISOR). Price impact for the active corp if it issues a share.
- Issue remaining (vector, 8 slots — 1 per corp that has yet to act in this issue phase).

### PAR (IPO) token
Shared by PHASE_IPO and PHASE_PAR — both sub-phases reference the same
per-par-price slate.
- Player cash required (vector, 14 slots for each valid par price). Amount the active player needs to pay to float at each par price; 0 for invalid par prices.
- Resulting corp cash (vector, 14 slots). Corp cash after floating at each par price; 0 for invalid par prices.
- Resulting issued shares (vector, 14 slots). Shares issued at each par price; 0 for invalid par prices. Values are 0/2/4 in practice, normalized by 4.0 as the maximum.
- IPO remaining (vector, 8 slots — 1 per corp that is still inactive and therefore available for IPO).

### AcqOffer token
- Offer price index (scalar, normalized by 51.0). Matches acquisition action encoding — offset from the target company's low price.
- Offer price (scalar, normalized by COMPANY_PRICE_DIVISOR).
- Offer corp (one-hot, 8 slots for each corp).
- FI company (scalar, 0/1 — 1 if the target company is FI-owned).

### AcqPriceInfo token
Populated during PHASE_ACQ_SELECT_PRICE. Every (active_corp, active_company)-
level scalar is already on the corp / company / active-entity / company-
location tokens and reaches the price head via attention, so this token is
kept deliberately minimal — only what can't be read off those tokens directly:
- max_offset (scalar, normalized by PRICE_RANGE_DIVISOR). Valid ACQ price-offset count for the target company (`high - low + 1`). Same quantity as the Company token's low_high_diff — duplicated here so the head doesn't have to cross-attend through company-id lookup.
- fi_flag (scalar, 0/1). 1 if the target company is FI-owned. A hard discontinuity for the head: FI sale is a single fixed-price FI_BUY action with no offset to pick.
- total_synergies (scalar, normalized by ENTITY_INCOME_DIVISOR). Marginal synergy income the active corp would gain by adding the target company to its portfolio.

---

## Corp tokens (8 total)
- Corp ID (one-hot, 8 slots for each corp)
- Active (scalar, 1/0 — 1 if the corp has floated and is in play)
- In receivership (scalar, 1/0)
- Passed on ACQ_OFFER (scalar, 1/0)
- Unissued shares (scalar, normalized by SHARE_DIVISOR)
- Issued shares (scalar, normalized by SHARE_DIVISOR)
- Bank shares (scalar, normalized by SHARE_DIVISOR)
- Share price index (one-hot, 27 slots — current market price index)
- Share price (scalar, normalized by SHARE_PRICE_DIVISOR)
- Pending price move (scalar, normalized by IMPACT_DIVISOR)
- Cash (scalar, normalized by CASH_DIVISOR)
- Acquisition proceeds (scalar, normalized by CASH_DIVISOR)
- Income (scalar, normalized by ENTITY_INCOME_DIVISOR)
- Stars (scalar, normalized by CORP_STAR_DIVISOR). *Total* stars — owned companies + cash + SI bonus.
- Raw revenue (scalar, normalized by ENTITY_INCOME_DIVISOR)
- Synergy income (scalar, normalized by ENTITY_INCOME_DIVISOR)
- CoO cost (scalar, normalized by ENTITY_INCOME_DIVISOR)
- Ability income (scalar, normalized by ENTITY_INCOME_DIVISOR)
- President ID (one-hot, 5 slots for max 5 players, 0-padded for 3/4 players). All 0 if inactive / in receivership.
- Owned companies (vector, 36 slots). Includes companies sitting in the corp's acquisition pile.

Active-corp selection (the turn's active corp) lives in the dedicated
ActiveCorp token, not here.

---

## Player tokens (3 to 5 total)
- Player ID (one-hot, 5 slots for 5 max players, 0-padded for 3/4 players)
- Turn order (one-hot, 5 slots for 5 max players, 0-padded for 3/4 players)
- Has passed (scalar, 1/0)
- Cash (scalar, normalized by CASH_DIVISOR)
- Net worth (scalar, normalized by NET_WORTH_DIVISOR)
- Liquidity (scalar, normalized by NET_WORTH_DIVISOR)
- Income (scalar, normalized by ENTITY_INCOME_DIVISOR)
- Owned shares (vector, 8 slots for 8 corps, normalized by SHARE_DIVISOR)
- Round trips (scalar, 1 if any share buy/sell would be affected by the round-trip limit, 0 otherwise)
- Share buys (vector, 8 slots for 8 corps, normalized by SHARE_DIVISOR)
- Share sells (vector, 8 slots for 8 corps, normalized by SHARE_DIVISOR)
- Presidencies (vector, 8 slots for 8 corps — 1 if this player is president of the corresponding corp)
- Owned companies (vector, 36 slots)

Active-player selection lives in the dedicated ActivePlayer token, not here.
