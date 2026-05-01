# ResNet Data Spec

Dense residual-MLP input projection for Rolling Stock Stars.

`core/resnet_data.pyx` should fill one contiguous `float32` vector from a
canonical `GameState`. The schema below uses the same normalized values and
zeroing semantics currently emitted by `core/token_data.pyx`, but packs them
as dense stride-based records instead of a `(num_tokens, TOKEN_DIM)` token
matrix.

This is not the old visible-state layout from `old-vectors.md`. It uses the
current 11 decision phases, current split ACQ phases, current unified policy
contract, and the current compact `GameState`.

## Design Rules

- Use the same scalar values, one-hots, flags, normalization divisors, and
  phase-context zeroing rules as `core/token_data.pyx`.
- Do not include token attention masks. ResNet has no token attention, so the
  token `attn_mask`/`present` slots are omitted everywhere.
- Do not include phase-token `context_active` flags. The active phase is
  already encoded in `GlobalInfo.decision_phase`; inactive phase-context
  sub-blocks are otherwise all zero.
- Keep entity selector/lifecycle flags:
  `company.is_selected`, `corp.is_selected`, `corp.active`, and
  `player.is_selected` remain in the vector.
- Keep each entity's information in one contiguous stride:
  companies use one `(22 + N)`-wide stride, corps use one `(89 + N)`-wide
  stride, and players use one `(56 + N)`-wide stride, where
  `N = num_players`.
- Rotate all player-indexed blocks and player-id one-hots so relative slot 0
  is the active player.
- Do not rotate company or corp order. Company ids and corp ids remain
  canonical.
- Do not include hidden information: no deck order, no unobservable excluded
  company identities before they are public, no setup-only hidden data.

## Player Rotation

ResNet inputs are active-relative:

```text
canonical_player = (active_player + relative_slot) % num_players
relative_slot = (canonical_player - active_player) % num_players
```

Player records are emitted in relative-slot order:

```text
player_record[0] -> active player
player_record[k] -> canonical player (active_player + k) % num_players
```

Player-id one-hots inside non-player records are also active-relative and use
exactly `num_players` slots:

- `company.owner_player_relative[N]`
- `corp.president_relative[N]`

The player record's `turn_order` one-hot should become
`turn_order_relative[N]`, where slot 0 means the active player's current turn
order. Compute it as:

```text
relative_turn_order = (player_turn_order - active_player_turn_order) % num_players
```

## Vector Size

The vector width varies by player count:

```text
RESNET_VECTOR_SIZE(num_players) = 1699 + 100 * num_players + num_players^2
```

| Players | Width |
| --- | ---: |
| 3 | 2008 |
| 4 | 2115 |
| 5 | 2224 |

Top-level layout, with `N = num_players`:

| Offset | Block | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | GlobalInfo | 23 | Current GlobalInfo token minus `attn_mask` |
| 23 | MarketInfo | 54 | Current MarketInfo token minus `attn_mask` |
| 77 | PhaseContext | 79 | Current phase-specific tokens minus `attn_mask` flags |
| 156 | Companies | `36 * (22 + N)` | 36 company records |
| `156 + 36 * (22 + N)` | FI | 39 | Current FI token minus `attn_mask` |
| `195 + 36 * (22 + N)` | Corps | `8 * (89 + N)` | 8 corp records |
| `1699 + 44 * N` | Players | `N * (56 + N)` | Active-relative player records |

Concrete offsets:

| Players | Companies | FI | Corps | Players | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 156 | 1056 | 1095 | 1831 | 2008 |
| 4 | 156 | 1092 | 1131 | 1875 | 2115 |
| 5 | 156 | 1128 | 1167 | 1919 | 2224 |

## Normalization

Use the same constants as `core/token_data.pyx`:

| Constant | Value | Use |
| --- | ---: | --- |
| `CASH_DIVISOR` | 150.0 | player/corp/FI cash, acquisition proceeds, PAR payments |
| `NET_WORTH_DIVISOR` | 200.0 | player net worth and liquidity |
| `COMPANY_INCOME_DIVISOR` | 10.0 | company base and adjusted income |
| `ENTITY_INCOME_DIVISOR` | 80.0 | player/corp/FI aggregate income, corp revenue parts, ACQ synergies |
| `SHARE_DIVISOR` | 7.0 | share counts |
| `COMPANY_PRICE_DIVISOR` | 80.0 | company prices, bid/acquisition prices |
| `SHARE_PRICE_DIVISOR` | 75.0 | share prices and market-space prices |
| `COMPANY_STAR_DIVISOR` | 5.0 | company stars |
| `CORP_STAR_DIVISOR` | 40.0 | corp total stars |
| `IMPACT_DIVISOR` | 5.0 | price movement impacts |
| `PRICE_RANGE_DIVISOR` | 50.0 | company low/high price range and ACQ max offset |
| `OWNED_COMPANIES_DIVISOR` | 10.0 | owned-company count summaries |
| `PRESIDENCIES_DIVISOR` | 8.0 | player presidency count summary |
| `TOTAL_SHARES_DIVISOR` | 20.0 | player total shares summary |
| `CONSECUTIVE_PASSES_DIVISOR` | 5.0 | invest consecutive passes |
| `FLOAT_SHARES_MAX` | 4.0 | PAR resulting issued shares |

## GlobalInfo Block

Width 23. Same values as the current GlobalInfo token after dropping
`attn_mask`.

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `decision_phase` | 11 | One-hot over `DecisionPhase`; automated/terminal phases all zero |
| 11 | `coo_level` | 7 | CoO levels 1..7 as slots 0..6 |
| 18 | `end_card_flipped` | 1 | 0/1 |
| 19 | `cards_remaining` | 1 | `/ NUM_COMPANIES` |
| 20 | `num_players` | 3 | One-hot for 3p/4p/5p |

Decision phase slot order is the current `DecisionPhase` order:

| Slot | Decision phase |
| ---: | --- |
| 0 | `DPHASE_INVEST` |
| 1 | `DPHASE_BID` |
| 2 | `DPHASE_ACQ_SELECT_CORP` |
| 3 | `DPHASE_ACQ_OFFER` |
| 4 | `DPHASE_CLOSING` |
| 5 | `DPHASE_DIVIDENDS` |
| 6 | `DPHASE_ISSUE` |
| 7 | `DPHASE_IPO` |
| 8 | `DPHASE_PAR` |
| 9 | `DPHASE_ACQ_SELECT_COMPANY` |
| 10 | `DPHASE_ACQ_SELECT_PRICE` |

## MarketInfo Block

Width 54. Same values as the current MarketInfo token after dropping
`attn_mask`.

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `slot_prices` | 27 | `MARKET_PRICES[i] / SHARE_PRICE_DIVISOR` |
| 27 | `availability` | 27 | 1 if market space is available |

## PhaseContext Block

Width 79. This concatenates the current phase-specific token rows, in the
same order as `token_data.pyx`, after dropping each token's `attn_mask`.
Inactive phase sub-blocks are all zero. The active phase is identified by
`GlobalInfo.decision_phase`.

| Offset | Sub-block | Width | Active in |
| ---: | --- | ---: | --- |
| 0 | Invest | 1 | `PHASE_INVEST` |
| 1 | Auction | 3 | `PHASE_BID` |
| 4 | Dividend | 26 | `PHASE_DIVIDENDS` |
| 30 | Issue | 1 | `PHASE_ISSUE_SHARES` |
| 31 | PAR / IPO | 42 | `PHASE_IPO`, `PHASE_PAR` |
| 73 | AcqOffer | 3 | `PHASE_ACQ_OFFER` |
| 76 | AcqPriceInfo | 3 | `PHASE_ACQ_SELECT_PRICE` |

### Invest Sub-block

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `consecutive_passes` | 1 | `/ CONSECUTIVE_PASSES_DIVISOR` |

### Auction Sub-block

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `min_bid_index` | 1 | Minimum legal next bid offset `/ AUCTION_CAP` |
| 1 | `min_bid_value` | 1 | Minimum legal next bid value `/ COMPANY_PRICE_DIVISOR` |
| 2 | `is_first_bid` | 1 | 1 before any bid has been placed |

### Dividend Sub-block

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `dividend_impacts` | 26 | Amounts 0..25, each `/ IMPACT_DIVISOR` |

### Issue Sub-block

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `issue_impact` | 1 | Active corp issue price delta `/ IMPACT_DIVISOR` |

### PAR / IPO Sub-block

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `par_data` | 42 | 14 tuples, stride 3 |

Each `par_data[par_index]` tuple:

| Tuple offset | Field | Notes |
| ---: | --- | --- |
| 0 | `player_cash_required` | `/ CASH_DIVISOR` |
| 1 | `resulting_corp_cash` | `/ CASH_DIVISOR` |
| 2 | `resulting_issued_shares` | `/ FLOAT_SHARES_MAX` |

Invalid static par slots remain zero.

### AcqOffer Sub-block

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `offer_price_index` | 1 | `(offer_price - low_price) / ACQ_PRICE_OFFSETS` |
| 1 | `offer_price` | 1 | `/ COMPANY_PRICE_DIVISOR` |
| 2 | `fi_company` | 1 | 1 if target company is FI-owned |

### AcqPriceInfo Sub-block

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `max_offset` | 1 | `(high_price - low_price) / PRICE_RANGE_DIVISOR` |
| 1 | `fi_flag` | 1 | 1 if target company is FI-owned |
| 2 | `total_synergies` | 1 | Marginal active-corp synergy `/ ENTITY_INCOME_DIVISOR` |

## Company Records

36 records, canonical company id order, stride `22 + N`. Same values as the
current company token after dropping `attn_mask`, except `owner_player` slots
are active-relative and exactly `N` wide.

Top-level offset:

```text
company_stride(N) = 22 + N
company_base(company_id, N) = 156 + company_stride(N) * company_id
```

Stride layout:

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `is_selected` | 1 | 1 iff this is `active_company` |
| 1 | `low_price` | 1 | `/ COMPANY_PRICE_DIVISOR` |
| 2 | `face_value` | 1 | `/ COMPANY_PRICE_DIVISOR` |
| 3 | `high_price` | 1 | `/ COMPANY_PRICE_DIVISOR` |
| 4 | `low_high_diff` | 1 | `(high - low) / PRICE_RANGE_DIVISOR` |
| 5 | `base_income` | 1 | `/ COMPANY_INCOME_DIVISOR` |
| 6 | `stars` | 1 | `/ COMPANY_STAR_DIVISOR` |
| 7 | `adjusted_income` | 1 | `/ COMPANY_INCOME_DIVISOR`; may be negative |
| 8 | `at_removed` | 1 | Includes publicly observable excluded cards |
| 9 | `at_auction` | 1 | 1 for `LOC_AUCTION` |
| 10 | `at_revealed` | 1 | 1 for `LOC_REVEALED` |
| 11 | `at_corp_acq` | 1 | 1 for `LOC_CORP_ACQ` |
| 12 | `acq_select_synergy_delta` | 1 | In `PHASE_ACQ_SELECT_COMPANY`, `/ ENTITY_INCOME_DIVISOR`; zero otherwise |
| 13 | `owner_corp` | 8 | Canonical corp id one-hot for `LOC_CORP` or `LOC_CORP_ACQ` |
| 21 | `owner_player_relative` | `N` | Active-relative player one-hot for `LOC_PLAYER` |
| `21 + N` | `owner_fi` | 1 | 1 for `LOC_FI` |

`LOC_EXCLUDED` should set `at_removed` only once the CoO has advanced past the
company's star tier, matching `token_data.pyx`; do not leak setup exclusions
early.

## FI Record

Width 39. Same values as the current FI token after dropping `attn_mask`.

Top-level offset:

```text
fi_base(N) = 156 + 36 * (22 + N)
```

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `cash` | 1 | `/ CASH_DIVISOR` |
| 1 | `income` | 1 | `/ ENTITY_INCOME_DIVISOR` |
| 2 | `num_owned_companies` | 1 | `/ OWNED_COMPANIES_DIVISOR` |
| 3 | `owned_companies` | 36 | Canonical company id flags for `LOC_FI` |

## Corp Records

8 records, canonical corp id order, stride `89 + N`. Same values as the
current corp token after dropping `attn_mask`, except `president` slots are
active-relative and exactly `N` wide.

Top-level offset:

```text
corp_stride(N) = 89 + N
corp_base(corp_id, N) = 195 + 36 * (22 + N) + corp_stride(N) * corp_id
```

Stride layout:

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `is_selected` | 1 | 1 iff this is `active_corp` |
| 1 | `active` | 1 | Floated / operational lifecycle flag |
| 2 | `in_receivership` | 1 | 0/1 |
| 3 | `passed_acq_offer` | 1 | 0/1 |
| 4 | `unissued_shares` | 1 | `/ SHARE_DIVISOR` |
| 5 | `issued_shares` | 1 | `/ SHARE_DIVISOR` |
| 6 | `bank_shares` | 1 | `/ SHARE_DIVISOR` |
| 7 | `price_index` | 27 | Canonical market index one-hot, active corps only |
| 34 | `share_price` | 1 | `/ SHARE_PRICE_DIVISOR`, active corps only |
| 35 | `pending_price_move` | 1 | `/ IMPACT_DIVISOR`, active corps only |
| 36 | `cash` | 1 | `/ CASH_DIVISOR`, active corps only |
| 37 | `acq_proceeds` | 1 | `/ CASH_DIVISOR`, active corps only |
| 38 | `income` | 1 | `/ ENTITY_INCOME_DIVISOR`, active corps only |
| 39 | `stars` | 1 | `/ CORP_STAR_DIVISOR`, active corps only |
| 40 | `raw_revenue` | 1 | `/ ENTITY_INCOME_DIVISOR`, active corps only |
| 41 | `synergy_income` | 1 | `/ ENTITY_INCOME_DIVISOR`, active corps only |
| 42 | `coo_cost` | 1 | `/ ENTITY_INCOME_DIVISOR`, active corps only |
| 43 | `ability_income` | 1 | `/ ENTITY_INCOME_DIVISOR`, active corps only |
| 44 | `acq_offer_corp` | 1 | 1 on original offer corp in `PHASE_ACQ_OFFER` |
| 45 | `dividend_remaining` | 1 | 1 if still to act in `PHASE_DIVIDENDS` |
| 46 | `issue_remaining` | 1 | 1 if still to act in `PHASE_ISSUE_SHARES` |
| 47 | `ipo_remaining` | 1 | 1 if inactive in `PHASE_IPO` or `PHASE_PAR` |
| 48 | `buy_impact` | 1 | In `PHASE_INVEST`, `/ IMPACT_DIVISOR` |
| 49 | `sell_impact` | 1 | In `PHASE_INVEST`, `/ IMPACT_DIVISOR` |
| 50 | `num_operational_companies` | 1 | `/ OWNED_COMPANIES_DIVISOR`, active corps only |
| 51 | `num_acq_pile_companies` | 1 | `/ OWNED_COMPANIES_DIVISOR`, active corps only |
| 52 | `num_total_companies` | 1 | `/ OWNED_COMPANIES_DIVISOR`, active corps only |
| 53 | `president_relative` | `N` | Active-relative player one-hot; zero for inactive/receivership corps |
| `53 + N` | `owned_companies` | 36 | Canonical company id flags; includes acquisition pile |

## Player Records

`N` records, active-relative order, stride `56 + N`. Same values as the
current player token after dropping `attn_mask`, with player record order and
`turn_order` rotated.

Top-level offset:

```text
player_stride(N) = 56 + N
player_base(relative_slot, N) = 1699 + 44 * N + player_stride(N) * relative_slot
```

Stride layout:

| Offset | Field | Width | Notes |
| ---: | --- | ---: | --- |
| 0 | `is_selected` | 1 | 1 for relative slot 0 |
| 1 | `turn_order_relative` | `N` | Rotated turn-order one-hot |
| `1 + N` | `has_passed` | 1 | 0/1 |
| `2 + N` | `cash` | 1 | `/ CASH_DIVISOR` |
| `3 + N` | `net_worth` | 1 | `/ NET_WORTH_DIVISOR` |
| `4 + N` | `liquidity` | 1 | `/ NET_WORTH_DIVISOR` |
| `5 + N` | `income` | 1 | `/ ENTITY_INCOME_DIVISOR` |
| `6 + N` | `auction_high_bidder` | 1 | 1 on high bidder in `PHASE_BID` |
| `7 + N` | `auction_starter` | 1 | 1 on auction starter in `PHASE_BID` |
| `8 + N` | `round_trips` | 1 | 1 if any corp hit the buy/sell round-trip cap |
| `9 + N` | `owned_shares` | 8 | Canonical corp id order, `/ SHARE_DIVISOR` |
| `17 + N` | `num_owned_companies` | 1 | `/ OWNED_COMPANIES_DIVISOR` |
| `18 + N` | `num_presidencies` | 1 | `/ PRESIDENCIES_DIVISOR` |
| `19 + N` | `total_owned_shares` | 1 | `/ TOTAL_SHARES_DIVISOR` |
| `20 + N` | `owned_companies` | 36 | Canonical company id flags for `LOC_PLAYER` |

`auction_high_bidder` and `auction_starter` are scalar flags on the rotated
player record, so no additional one-hot remapping is needed.

## Implementation Notes

- `get_resnet_vector_size(num_players)` should return the exact width table
  above and reject player counts outside 3-5.
- `get_resnet_data(...)` should zero the destination vector before filling.
- Refresh dirty player/corp caches before reading cached financial fields,
  matching the `core/token_data.pyx` prologue.
- The implementation should fill these values directly from engine/entity
  APIs rather than materializing token data and copying it.
- Tests should compare representative fields against `core/token_data.pyx`
  after accounting for removed `attn_mask` slots and active-relative
  player-id fields, then separately verify player rotation.
- ResNet model values are active-relative. Replay remains canonical, so the
  trainer/evaluator bridge must rotate value targets/outputs as described in
  `resnet-plan.md`.
