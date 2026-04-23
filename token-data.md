# Rough token data spec

## Token order

Matches `core/token_data.pyx::_fill_buffer` and the per-type projections in
`nn/transformer.py::RSSTransformerNet._project_tokens`:

1. **Informational tokens** (every slot carries at least some dynamic
   content — there is no pure-static prefix):
   `market_info`, `companies` (×36), `FI`, `global_info`
2. **Phase-specific tokens** (zero unless the engine is in the matching
   phase):
   `invest`, `auction`, `dividend`, `issue`, `par`, `acq_select_company`,
   `acq_offer`, `acq_price_info`
3. **Corp tokens** (×8)
4. **Player tokens** (×N, N ∈ {3, 4, 5})

Total tokens = `num_players + 55` (58 / 59 / 60 for 3p / 4p / 5p). The model
adds 4 learned pass anchors after projection for entity-readout pass phases;
they don't appear in the engine-side buffer. BID, ISSUE, and ACQ_OFFER read
their pass logits from their phase-specific informational tokens.

Each token is zero-padded to `TOKEN_DIM = 93` (max width across all token
types, currently pinned by the Corp token). The per-type non-padded widths
live in `TokenWidth` (`core/token_data.pxd`); the model slices
`buffer[i, :TW_<type>]` before its type-specific projection so padding
positions carry no parameters.

Active-entity selection (the turn's `active_player` / `active_corp` /
`active_company`) is surfaced as a single `is_selected` bit on each
affected entity's own token rather than as standalone selector tokens.

---

## Informational tokens

### MarketInfo token (54)
- Slot prices (vector, 27 slots — static $0..$75 market-space prices, each
  normalized by SHARE_PRICE_DIVISOR). Constant across a game.
- Availability (vector, 27 slots — 1 if the corresponding market space is
  unoccupied, 0 otherwise). First and last spaces ($0 and $75) always
  available.

### Company tokens (62, ×36)
- Company ID (one-hot, 36 slots)
- Low price (scalar, normalized by COMPANY_PRICE_DIVISOR)
- Face value (scalar, normalized by COMPANY_PRICE_DIVISOR)
- High price (scalar, normalized by COMPANY_PRICE_DIVISOR)
- low_high_diff (scalar, normalized by PRICE_RANGE_DIVISOR — count of valid
  ACQ_SELECT_PRICE offsets for this company, `high - low + 1`, max 51 for
  CDG). Same quantity as `max_offset` on the AcqPriceInfo token.
- Base income (scalar, normalized by COMPANY_INCOME_DIVISOR)
- Stars (scalar, normalized by COMPANY_STAR_DIVISOR)
- Adjusted income (scalar, normalized by COMPANY_INCOME_DIVISOR). Per-company
  CoO-adjusted income; may be negative (a company's income can be pushed
  below zero by CoO-dependent penalties).
- is_selected (scalar, 0/1). 1 iff this is the current `active_company`.
- at_removed (scalar, 0/1). 1 for LOC_REMOVED; also 1 for LOC_EXCLUDED once
  the CoO has advanced past the company's star tier — the exclusion is
  publicly observable then; setting it unconditionally would leak setup
  randomness.
- at_auction (scalar, 0/1). 1 for LOC_AUCTION.
- at_revealed (scalar, 0/1). 1 for LOC_REVEALED.
- at_corp_acq (scalar, 0/1). 1 for LOC_CORP_ACQ.
- owner_corp (one-hot, 8 slots). 1 at the owning corp iff the company is at
  LOC_CORP.
- owner_player (one-hot, 5 slots, padded for num_players < 5). 1 at the
  owning player iff the company is at LOC_PLAYER.
- owner_fi (scalar, 0/1). 1 iff the company is at LOC_FI.

The three ownership groups (`owner_corp` / `owner_player` / `owner_fi`) are
mutually exclusive: only the group matching the current location is
non-zero. Companies at LOC_CORP_ACQ or any unowned location (AUCTION /
REVEALED / REMOVED / DECK / EXCLUDED) leave all three groups zero; the
`at_*` flags encode those cases.

Synergies are no longer surfaced on the Company token. The ACQ_SELECT_COMPANY
phase-specific token carries the per-candidate marginal synergy delta for the
active corp; in other phases synergies are summarized on the Corp token
(`synergy_income`) and the acquisition price head reads the
target-specific total from AcqPriceInfo.

### FI token (38)
- Cash (scalar, normalized by CASH_DIVISOR)
- Income (scalar, normalized by ENTITY_INCOME_DIVISOR)
- Owned companies (vector, 36 slots — 1 if the corresponding company is at
  LOC_FI, 0 otherwise)

### GlobalInfo token (23)
Bundled game-level scalars:
- Decision phase (one-hot, 11 slots — one per `DecisionPhase`; all-zero in
  automated / terminal engine phases).
- CoO level (one-hot, 7 slots — CoO levels 1..7 → slots 0..6).
- End card flipped (scalar, 0/1).
- Cards remaining (scalar, normalized by NUM_COMPANIES).
- num_players (one-hot, 3 slots — slot 0 = 3p, slot 1 = 4p, slot 2 = 5p).

---

## Phase-specific tokens (8 total)

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

### AcqSelectCompany token
Populated during PHASE_ACQ_SELECT_COMPANY. Per-company marginal synergy
context for the active corp's SELECT_COMPANY decision; the per-company
select logits themselves are still read from company tokens.
- Synergy delta (vector, 36 slots, normalized by ENTITY_INCOME_DIVISOR).
  Marginal synergy income the active corp would gain by acquiring the
  corresponding candidate. Slots for companies already in the active corp's
  portfolio (owned or in its acquisition pile) stay at zero — self-
  acquisition is a no-op.

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

## Corp tokens (93, ×8)
The `active` slot below is the lifecycle float flag (1 iff the corp is
floated and operational); the decision-flow selector is the separate
`is_selected` bit at the end of the token.
- Corp ID (one-hot, 8 slots for each corp; retained in the engine buffer, but
  the current transformer skips this slice and adds learned `corp_id_embed`
  identity after projecting the remaining corp features)
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
- is_selected (scalar, 0/1). 1 iff this is the current `active_corp`.

---

## Player tokens (85, ×N, N ∈ {3, 4, 5})
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
- is_selected (scalar, 0/1). 1 iff this is the current `active_player`.
