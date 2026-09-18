# Transformer V3 Token Data Spec

This document specifies layout 3, consumed by `nn/transformer-v3.py`.
Its extractor and constants live in `core/token_data_v3.pyx/.pxd`.
The shared `core/token_data.pyx` API dispatches to it with `layout_version=3`.
Shared callers use `get_token_dim(3)` and
`get_token_widths(max_players, layout_version=3)`. The model imports its
constants directly from `core.token_data_v3`.

Use `v3_behavior=True` on GameState (or `"v3_behavior": true` in training config)
to retain trade history through later phases. Layout selection alone does not
enable that engine mode; the extractor reads the counters present in the state.

## Token Order

Matches `core/token_data_v3.pyx::_fill_buffer`:

1. Informational/entity prefix:
   `market_info`, `companies` (x36), `FI`, `global_info`
2. Phase-specific tokens:
   `invest`, `auction`, `dividend`, `issue`, `par`, `acq_offer`,
   `acq_price_info`
3. Corp tokens (x8)
4. Player tokens (xM, M in {3, 4, 5}, where M is the model/storage
   player-token capacity)

Fixed engine rows: 54. Total tokens = `max_players + 54`
(57 / 58 / 59 for max 3p / 4p / 5p). Exact-width extraction uses
`max_players = num_players`; padded extraction can emit, for example, 59 rows
for an actual 3p state in a max-5 training run.

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

For a 3p state extracted with `max_players=5`, rows 57..58 are reserved
player rows and remain all-zero.

The model consumes exactly these engine-side rows; it does not append
synthetic model-side tokens after projection.

After type-specific projection and identity/type embeddings, v3 applies one
simultaneous round of directed relation messages before the transformer trunk.
A shared RMSNorm and two-layer GELU MLP transform each source token. For each
of the twelve relations, each recipient takes a weighted sum of visible neighbors,
divides by the square root of their count (clamped to at least one), and
applies a separate bias-free linear projection. Independent learned signed
gains, initialized to 0.1, scale these relation contributions before adding
them to the original recipient embedding. Like attention relation biases,
these gains are exempt from weight decay; the MLP and relation projection
matrices receive normal weight decay. Empty neighborhoods contribute zero;
hidden/padded tokens neither send nor receive messages. There is no post-sum
normalization, so neighborhood magnitude remains available. Square-root
scaling moderates growth but does not make correlated neighbors scale-invariant.

This stage uses the same relations as attention: ten binary ownership,
shareholding, and presidency planes, plus two raw share-count planes
(`PLAYER_CORP_SHARE_COUNT` and `CORP_PLAYER_SHARE_COUNT`). Cython workers
extract the count for the actual player/corporation pair in both directions.
Dense planes contain uint8 0/1 flags or 0..7 counts. Models normalize only
the count planes by `PY_SHARE_DIVISOR` (7), once per forward, for use in
attention biases and input mixing. Presence, quantity, and presidency have
independent learned coefficients/projections. Neighborhood normalization
counts nonzero neighbors, not total share weight: doubling all quantities
at fixed connectivity doubles the quantity relation's message contribution.
The player-token share vector remains available for projections and policy
head calculations.

Sparse IPC uses uint8 `(relation_id, query_token, key_token, value)` records,
with all-zero padding. Binary records have value 1; quantity records contain
the raw share count. The 256-record capacity covers the conservative bound
of 248 edges: 72 company-ownership, 160 shareholding/count, and 16 presidency
edges. This is 1,024 bytes per state instead of 768 for the previous triplets.
The eval model consumes sparse records directly; it does not build a full
`(B, R, N, N)` tensor. Dense replay extraction uses the same Cython semantics.
Both v2 and v3 accept this shared transport. V2 ignores the new count planes
and sparse records, retains its original ten-relation parameter shape, and
loads existing v2 checkpoints without conversion. V3 uses all twelve
relations; v3 checkpoints from before this addition need updating/retraining.
Token layout and policy/value output shapes are unchanged.

Each token row is zero-padded to 98 features (`TokenDataSize.TOKEN_DIM`), the
width of the Corp token. Per-type widths live in `core.token_data_v3.TokenWidth`:

- `TW_MARKET_INFO = 55`
- `TW_COMPANY = 28`
- `TW_FI = 40`
- `TW_GLOBAL_INFO = 24`
- `TW_INVEST = 2`
- `TW_AUCTION = 4`
- `TW_DIVIDEND = 53`
- `TW_ISSUE = 2`
- `TW_PAR = 43`
- `TW_ACQ_OFFER = 4`
- `TW_ACQ_PRICE = 4`
- `TW_CORP = 98`
- `TW_PLAYER = 62`

**Relational summary scalars.** Corp, player, and FI tokens carry a small
group of aggregate scalars (owned-company counts, presidency count, total
shares) before their relational tail. Corp trade-history scalars follow the
corp summary scalars and also stay in the projection. The relational tail is
dropped on the model side in favour of Graphormer-style attention biases;
the summary scalars stay in the projection so the trunk has a direct
aggregate view of the data the multihots encode.

## Attention Mask Slot

Every token type starts with:

- `attn_mask` at slot 0 (scalar, 0/1)

Rules:

- MarketInfo, GlobalInfo, FI, company, corp, and actual player tokens always
  set `attn_mask = 1`.
- Padded player rows in `[num_players, max_players)` remain all-zero, including
  `attn_mask = 0`, so they are not visible as attention keys.
- Phase-specific tokens set `attn_mask = 1` only when `_fill_buffer` calls the
  matching phase helper. The PAR token is shared and sets the mask in both
  `PHASE_IPO` and `PHASE_PAR`.

## Active Entity Selection

`active_player`, `active_corp`, and `active_company` are surfaced as
`is_selected` scalar flags on their own entity tokens. They are not standalone
selector tokens. The selected entity flags occupy slot 1 because slot 0 is
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
- `low_high_diff`, normalized by `PRICE_RANGE_DIVISOR`. The 0-indexed maximum
  legal ACQ_SELECT_PRICE offset (`high - low`).
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

Buy/sell INVEST impacts are encoded on Corp tokens.

### Auction Token (4)

- `attn_mask`
- Minimum legal next-bid index. Offset from the current auction company's face
  value, normalized by `AUCTION_CAP`.
- Minimum legal next-bid value, normalized by `COMPANY_PRICE_DIVISOR`
- `is_first_bid`. 1 when `auction_high_bidder == -1`.

`auction_high_bidder` and `auction_starter` are encoded on Player tokens.

### Dividend Token (53)

- `attn_mask`
- Actual projected market-index movements (26 slots for amounts 0..25),
  normalized by `IMPACT_DIVISOR`. Each is resolved destination minus current
  index, using the same destination calculation as dividend execution.
  Occupied destinations are skipped in the movement direction; index 0 means
  bankruptcy and the last index is the shared $75 endpoint. Zero movement
  retains the current space. Actual movement can exceed two indices.
- Projected new share prices in dollars (26 slots for amounts 0..25), normalized
  by `CASH_DIVISOR`. Bankruptcy gives $0; the shared upper endpoint gives $75.

The v3 dividend MLP receives nine numerical features per amount: dividend per
share, total payout, corporation cash after payment, cash received by the actor,
other players and bank, actual index movement, projected dollar share price,
and the actor's immediate net-worth impact. The latter is
`cash_received + shares_owned * (new_share_price - old_share_price)`.
All monetary head features use `CASH_DIVISOR`; movement uses `IMPACT_DIVISOR`.
The current Corp share-price feature is converted from `SHARE_PRICE_DIVISOR`
before computing net-worth impact. Corporation cash left is measured before
any bankruptcy cleanup. Previews exist even for illegal amounts; legality
masking excludes them from action selection.

Previews use current occupancy, without forecasting later corporation actions.
The padded row width remains 98.

`dividend_remaining` is encoded on Corp tokens.

### Issue Token (2)

- `attn_mask`
- Actual issue movement for the active corp, normalized by `IMPACT_DIVISOR`.
  It uses the next available lower space, including bankruptcy; Stock Masters
  has zero movement.

`issue_remaining` is encoded on Corp tokens.

### PAR / IPO Token (43)

- `attn_mask`
- 14 par-price tuples:
  `(player_cash_required, resulting_corp_cash, resulting_issued_shares)`.
  Issued shares are normalized by `FLOAT_SHARES_MAX`.

`ipo_remaining` is encoded on Corp tokens. It is written for inactive corps
in both `PHASE_IPO` and `PHASE_PAR`.

### AcqOffer Token (4)

- `attn_mask`
- Offer price index, normalized by `ACQ_PRICE_OFFSETS`
- Offer price, normalized by `COMPANY_PRICE_DIVISOR`
- `fi_company`. 1 if the target company is FI-owned.

`acq_offer_corp` is encoded on Corp tokens.

### AcqPriceInfo Token (4)

- `attn_mask`
- `max_offset`, normalized by `PRICE_RANGE_DIVISOR`. The 0-indexed maximum
  legal ACQ_SELECT_PRICE offset (`high - low`).
- `fi_flag`. 1 if the target company is FI-owned.
- `total_synergies`, normalized by `ENTITY_INCOME_DIVISOR`

## Corp Tokens (98, x8)

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
- Actual projected no-dividend market-index movement, normalized by
  `IMPACT_DIVISOR`. Resolved against current occupancy each extraction;
  the engine's internal cached nominal star-based move remains in [-2, +2].
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

Buy/sell impacts use the next available space in the corresponding direction,
including occupied-space skips, bankruptcy, and the shared $75 endpoint.

Relational summary (active corps only — inactive corps leave these zero,
matching the rest of the active-gated fields):

- `num_operational_companies`. Count of companies at `LOC_CORP` owned by
  this corp, normalized by `OWNED_COMPANIES_DIVISOR`.
- `num_acq_pile_companies`. Count of companies at `LOC_CORP_ACQ` owned by
  this corp, normalized by `OWNED_COMPANIES_DIVISOR`.
- `num_total_companies`. Sum of the two, normalized by
  `OWNED_COMPANIES_DIVISOR`. Redundant by construction; saves the
  projection from learning the addition.

Trade history (raw slots 54..56, included in the corp projection):

| Slot | Feature | Value |
| --- | --- | --- |
| 54 | `actor_share_buys` | Current acting player's buys of this corp / 4 |
| 55 | `actor_share_sells` | Current acting player's sells of this corp / 4 |
| 56 | `actor_round_trip` | 1 iff that player bought AND sold this corp at least once |

These features use the current `active_player`, so their values can change
when the actor changes. They are populated in every phase, including for
inactive corps. All three are zero when there is no valid actor. Counts are
normalized without clipping: five buys is represented as 1.25.

Buy/sell counters start at zero and persist throughout the turn. They reset
when IPO finishes and the next turn's INVEST begins. Returning from BID to
INVEST does not reset them.

Relational tail:

- President ID (slots 57..61). All zero if inactive / receivership.
- Owned companies (slots 62..97). Includes companies in the acquisition pile.

## Player Tokens (62, xM, M in {3, 4, 5})

Player identity is inferred from row order.

Rows `[0, num_players)` within the player-token block are filled from the
actual game state. Rows `[num_players, max_players)` are padding rows and stay
all-zero. `get_token_widths(max_players, layout_version=3)` reports
`TW_PLAYER` for every reserved player-token row so the model projection layout
is stable.

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
- `any_round_trip` (raw slot 14). 1 iff this player has bought AND sold
  at least one share of the same corporation during the current turn's
  INVEST. Buying one corp and selling a different corp does not qualify.
  This flag uses the token's own player identity regardless of who is
  acting. It persists through every phase and resets with the counters at
  the beginning of the next turn's INVEST.
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
