# Token Data Spec

This documents the engine-side token buffer filled by
`core/token_data.pyx:get_token_data(...)` and `get_token_data_batch(...)`.
It reflects the staged token-schema refactor in `core/token_data.{pyx,pxd}`.
The companion model/test updates live in `nn/transformer.py` and
`tests/test_transformer.py`.

## Token Order

Matches `core/token_data.pyx::_fill_buffer`:

1. Informational/entity prefix:
   `market_info`, `companies` (x36), `FI`, `global_info`
2. Phase-specific tokens:
   `invest`, `auction`, `dividend`, `issue`, `par`, `acq_offer`,
   `acq_price_info`
3. Corp tokens (x8)
4. Player tokens (xN, N in {3, 4, 5})

Fixed engine rows: 54. Total tokens = `num_players + 54`
(57 / 58 / 59 for 3p / 4p / 5p).

For 3p, row indices are:

- 0: MarketInfo
- 1..36: Company 0..35
- 37: FI
- 38: GlobalInfo
- 39: Invest
- 40: Auction
- 41: Dividend
- 42: Issue
- 43: PAR / IPO
- 44: AcqOffer
- 45: AcqPriceInfo
- 46..53: Corp 0..7
- 54..56: Player 0..2

The model consumes exactly these engine-side rows; it does not append
synthetic model-side tokens after projection.

Each token row is zero-padded to `TOKEN_DIM = 95`, currently pinned by the
Corp token. Per-type widths live in `TokenWidth`:

- `TW_MARKET_INFO = 55`
- `TW_COMPANY = 28`
- `TW_FI = 40`
- `TW_GLOBAL_INFO = 24`
- `TW_INVEST = 2`
- `TW_AUCTION = 4`
- `TW_DIVIDEND = 27`
- `TW_ISSUE = 2`
- `TW_PAR = 43`
- `TW_ACQ_OFFER = 4`
- `TW_ACQ_PRICE = 4`
- `TW_CORP = 95`
- `TW_PLAYER = 62`

**Relational summary scalars.** Corp, player, and FI tokens carry a small
group of aggregate scalars (owned-company counts, presidency count, total
shares) immediately before their relational tail. The relational tail is
dropped on the model side in favour of Graphormer-style attention biases;
the summary scalars stay in the projection so the trunk has a direct
aggregate view of the data the multihots encode.

## Attention Mask Slot

Every token type starts with:

- `attn_mask` at slot 0 (scalar, 0/1)

Rules:

- MarketInfo, GlobalInfo, FI, company, corp, and emitted player tokens always
  set `attn_mask = 1`; only live player rows for `num_players` are emitted.
- Phase-specific tokens set `attn_mask = 1` only when `_fill_buffer` calls the
  matching phase helper. The PAR token is shared and sets the mask in both
  `PHASE_IPO` and `PHASE_PAR`.

## Active Entity Selection

`active_player`, `active_corp`, and `active_company` are surfaced as
`is_selected` scalar flags on their own entity tokens. They are not standalone
selector tokens. The selected entity flags now live at slot 1 because slot 0 is
reserved for `attn_mask`.

## Informational Tokens

### MarketInfo Token (55)

- `attn_mask`
- Slot prices (27 slots). Static $0..$75 market-space prices, normalized by
  `SHARE_PRICE_DIVISOR`.
- Availability (27 slots). 1 if the corresponding market space is available,
  0 otherwise.

### Company Tokens (28, x36)

Company identity is inferred from row order.

- `attn_mask`
- `is_selected`. 1 iff this is the current `active_company`.
- Low price, normalized by `COMPANY_PRICE_DIVISOR`
- Face value, normalized by `COMPANY_PRICE_DIVISOR`
- High price, normalized by `COMPANY_PRICE_DIVISOR`
- `low_high_diff`, normalized by `PRICE_RANGE_DIVISOR`. Count of valid
  ACQ_SELECT_PRICE offsets (`high - low + 1`).
- Base income, normalized by `COMPANY_INCOME_DIVISOR`
- Stars, normalized by `COMPANY_STAR_DIVISOR`
- Adjusted income, normalized by `COMPANY_INCOME_DIVISOR`
- `at_removed`. 1 for `LOC_REMOVED`; also 1 for observable `LOC_EXCLUDED`
  once CoO has advanced past the company's star tier.
- `at_auction`. 1 for `LOC_AUCTION`.
- `at_revealed`. 1 for `LOC_REVEALED`.
- `at_corp_acq`. 1 for `LOC_CORP_ACQ`.
- `acq_select_synergy_delta`. During `PHASE_ACQ_SELECT_COMPANY`, marginal
  synergy income the active corp would gain by acquiring this company,
  normalized by `ENTITY_INCOME_DIVISOR`. Zero outside the phase and zero for
  companies already in the active corp's owned/acquisition portfolio.

Relational tail:

- `owner_corp` (8 slots). 1 at the owning corp iff location is `LOC_CORP` or
  `LOC_CORP_ACQ`.
- `owner_player` (5 slots, padded for lower player counts). 1 at the owning
  player iff location is `LOC_PLAYER`.
- `owner_fi`. 1 iff location is `LOC_FI`.

The ownership groups are mutually exclusive. `LOC_CORP_ACQ` sets both
`at_corp_acq` and the owning corp's `owner_corp` slot. Unowned locations leave
all owner groups zero.

### FI Token (40)

- `attn_mask`
- Cash, normalized by `CASH_DIVISOR`
- Income, normalized by `ENTITY_INCOME_DIVISOR`

Relational summary:

- `num_owned_companies`. Count of companies at `LOC_FI`, normalized by
  `OWNED_COMPANIES_DIVISOR` (10.0, soft empirical cap).

Relational tail:

- Owned companies (36 slots). 1 if the company is at `LOC_FI`.

### GlobalInfo Token (24)

- `attn_mask`
- Decision phase one-hot (11 slots; all-zero in automated / terminal engine
  phases)
- CoO level one-hot (7 slots)
- End card flipped
- Cards remaining, normalized by `NUM_COMPANIES`
- `num_players` one-hot (3 slots for 3p/4p/5p)

## Phase-Specific Tokens

Phase-specific token rows exist in every buffer but remain all-zero outside
their matching phase. The shared PAR token is filled in both `PHASE_IPO` and
`PHASE_PAR`.

### Invest Token (2)

- `attn_mask`
- Consecutive passes, normalized by 5

Buy/sell invest impacts moved to Corp tokens.

### Auction Token (4)

- `attn_mask`
- Minimum legal next-bid index. Offset from the current auction company's face
  value, normalized by `AUCTION_CAP`.
- Minimum legal next-bid value, normalized by `COMPANY_PRICE_DIVISOR`
- `is_first_bid`. 1 when `auction_high_bidder == -1`.

`auction_high_bidder` and `auction_starter` moved to Player tokens.

### Dividend Token (27)

- `attn_mask`
- Dividend impacts (26 slots for amounts 0..25), normalized by
  `IMPACT_DIVISOR`

`dividend_remaining` moved to Corp tokens.

### Issue Token (2)

- `attn_mask`
- Issue impact for the active corp, normalized by `IMPACT_DIVISOR`

`issue_remaining` moved to Corp tokens.

### PAR / IPO Token (43)

- `attn_mask`
- 14 par-price tuples:
  `(player_cash_required, resulting_corp_cash, resulting_issued_shares)`.
  Issued shares are normalized by `FLOAT_SHARES_MAX`.

`ipo_remaining` moved to Corp tokens. It is written for inactive corps in both
`PHASE_IPO` and `PHASE_PAR`.

### AcqOffer Token (4)

- `attn_mask`
- Offer price index, normalized by `ACQ_PRICE_OFFSETS`
- Offer price, normalized by `COMPANY_PRICE_DIVISOR`
- `fi_company`. 1 if the target company is FI-owned.

`acq_offer_corp` moved to Corp tokens.

### AcqPriceInfo Token (4)

- `attn_mask`
- `max_offset`, normalized by `PRICE_RANGE_DIVISOR`
- `fi_flag`. 1 if the target company is FI-owned.
- `total_synergies`, normalized by `ENTITY_INCOME_DIVISOR`

## Corp Tokens (95, x8)

Corp identity is inferred from row order.

- `attn_mask`
- `is_selected`. 1 iff this is the current `active_corp`.
- Active lifecycle flag. 1 if floated / operational.
- In receivership
- Passed on ACQ_OFFER
- Unissued shares, normalized by `SHARE_DIVISOR`
- Issued shares, normalized by `SHARE_DIVISOR`
- Bank shares, normalized by `SHARE_DIVISOR`
- Share price index one-hot (27 slots)
- Share price, normalized by `SHARE_PRICE_DIVISOR`
- Pending price move, normalized by `IMPACT_DIVISOR`
- Cash, normalized by `CASH_DIVISOR`
- Acquisition proceeds, normalized by `CASH_DIVISOR`
- Income, normalized by `ENTITY_INCOME_DIVISOR`
- Stars, normalized by `CORP_STAR_DIVISOR`
- Raw revenue, normalized by `ENTITY_INCOME_DIVISOR`
- Synergy income, normalized by `ENTITY_INCOME_DIVISOR`
- CoO cost, normalized by `ENTITY_INCOME_DIVISOR`
- Ability income, normalized by `ENTITY_INCOME_DIVISOR`
- `acq_offer_corp`. During `PHASE_ACQ_OFFER`, 1 on the original offer corp.
- `dividend_remaining`. During `PHASE_DIVIDENDS`, 1 if this corp still needs
  to act.
- `issue_remaining`. During `PHASE_ISSUE_SHARES`, 1 if this corp still needs
  to act.
- `ipo_remaining`. During `PHASE_IPO` / `PHASE_PAR`, 1 if this corp is
  inactive.
- Buy impact. During `PHASE_INVEST`, active corp's buy-one-share market index
  delta, normalized by `IMPACT_DIVISOR`.
- Sell impact. During `PHASE_INVEST`, active corp's sell-one-share market
  index delta, normalized by `IMPACT_DIVISOR`.

Relational summary (active corps only — inactive corps leave these zero,
matching the rest of the active-gated fields):

- `num_operational_companies`. Count of companies at `LOC_CORP` owned by
  this corp, normalized by `OWNED_COMPANIES_DIVISOR`.
- `num_acq_pile_companies`. Count of companies at `LOC_CORP_ACQ` owned by
  this corp, normalized by `OWNED_COMPANIES_DIVISOR`.
- `num_total_companies`. Sum of the two, normalized by
  `OWNED_COMPANIES_DIVISOR`. Redundant by construction; saves the
  projection from learning the addition.

Relational tail:

- President ID (5 slots). All zero if inactive / receivership.
- Owned companies (36 slots). Includes companies in the acquisition pile.

## Player Tokens (62, xN, N in {3, 4, 5})

Player identity is inferred from row order.

- `attn_mask`
- `is_selected`. 1 iff this is the current `active_player`.
- Turn order one-hot (5 slots)
- Has passed
- Cash, normalized by `CASH_DIVISOR`
- Net worth, normalized by `NET_WORTH_DIVISOR`
- Liquidity, normalized by `NET_WORTH_DIVISOR`
- Income, normalized by `ENTITY_INCOME_DIVISOR`
- `auction_high_bidder`. During `PHASE_BID`, 1 on the high bidder; all zero
  on the opening bid before a bid has been placed.
- `auction_starter`. During `PHASE_BID`, 1 on the auction starter.
- Round trips. 1 if any share buy/sell would be affected by the round-trip
  limit.
- Owned shares (8 slots), normalized by `SHARE_DIVISOR`. Per-corp share
  counts are scalar quantities, not just relation presence, so they stay
  in the projection rather than the relational tail.

Relational summary:

- `num_owned_companies`. Count of companies at `LOC_PLAYER` owned by this
  player, normalized by `OWNED_COMPANIES_DIVISOR`.
- `num_presidencies`. Count of corps where this player is president,
  gated to active && !receivership corps to match the corp-token
  president one-hot. Normalized by `PRESIDENCIES_DIVISOR` (8.0 = NUM_CORPS,
  the hard cap).
- `total_owned_shares`. Sum of the 8-slot owned-shares vector,
  normalized by `TOTAL_SHARES_DIVISOR` (20.0, soft empirical cap).

Relational tail:

- Owned companies (36 slots)
