# State Vector Refactor Tracking

Changes to the visible state vector that need doc/test updates when complete.

## Changes

### 1. Corp: `pending_price_move` (1 per corp, +8 total)

**Position:** After `price_index_norm`, before `owned_companies` in corp stride.
**Encoding:** `calculate_price_move(owned_stars, required_stars) / IMPACT_DIVISOR` — raw move in [-2, +2] assuming $0 dividend.
**Value:** 0 for inactive corps.
**Corp stride:** 47 → 48 (11 scalars + 1 new + 36 company flags)

**Update triggers:** `recalculate_stars` (cash/company changes), `set_price_index` (price changes), `set_issued_shares` (issuance).

**Code changes:**
- `core/state.pxd` — `CorpFieldOffsets.pending_price_move`
- `core/state.pyx` — corp_stride, `CorpFields`, `compute_corp_field_offsets`, `get_corp_fields`
- `entities/corp.pxd` — `_pending_price_move_offset`, `update_pending_price_move`, `calculate_price_move`
- `entities/corp.pyx` — implementation + hooked into setters
- `phases/dividends.pyx` — removed local `_calculate_price_move`, cimports from `entities.corp`
- `tests/phases/conftest.py` — invariant added to `assert_invariants`

### 2. Corp: income decomposition (4 per corp, +32 total; 4 in turn state)

**Position:** After `pending_price_move`, before `owned_companies` in corp stride. Also 4 new fields in turn state active corp section after `active_corp_share_price`, before `active_corp_companies`.
**Encoding:** All normalized by `ENTITY_INCOME_DIVISOR` (80.0).
- `raw_revenue` — sum of base company incomes (before CoO/synergy/abilities)
- `synergy_income` — synergy bonus income
- `coo_cost` — **negative** CoO cost (always <= 0)
- `ability_income` — corp-specific ability bonus (VM/PR/DA/S)

**Identity:** `raw_revenue + synergy_income + coo_cost + ability_income == income`
**Value:** All 0 for inactive corps.
**Corp stride:** 48 → 52 (16 scalars + 36 company flags)
**Turn state:** +4 fields (active_corp_raw_revenue, active_corp_synergy_income, active_corp_coo_cost, active_corp_ability_income)

**Update triggers:** `calculate_income` (called from company transfers, CoO changes, income phase, float_corp).

**Code changes:**
- `core/state.pxd` — `CorpFieldOffsets` (+4), `TurnStateOffsets` (+4)
- `core/state.pyx` — corp_stride, turn_size, all namedtuples, offset computations, `set_active_corp`/`clear_active_corp`, `LayoutInfo`
- `entities/corp.pxd` — `IncomeBreakdown` struct, 4 cached offsets, `_calculate_income_nogil` returns struct
- `entities/corp.pyx` — `_calculate_income_nogil` refactored to return breakdown, `calculate_income` stores components, derived `total_coo` from `raw_revenue - adjusted_income_sum` (removed per-company CoO calls)
- `tests/phases/conftest.py` — invariants for decomposition sum and coo_cost <= 0

### 3. Player: `liquidity` (1 per player, +num_players total)

**Position:** After `net_worth`, before `turn_order` in player stride.
**Encoding:** Iterative share liquidation value / `NET_WORTH_DIVISOR` (200.0). Cash + simulated proceeds from selling all held shares one at a time, with each sell moving the corp's price to the next lower available market space.
**Value:** Equals cash when player holds no shares.
**Player stride:** 64 + num_players → 65 + num_players

**Simulation details:** Sells simulated in corp index order (0-7). Cross-corp market effects captured — selling corp 0's shares frees/occupies spaces that affect corp 1's simulation. Uses local copy of market availability array.

**Update triggers:** Lazily updated via `update_net_worth` (same call sites as net_worth).

**Code changes:**
- `core/state.pxd` — `PlayerFieldOffsets.liquidity`
- `core/state.pyx` — player_stride, `PlayerFields`, `compute_player_field_offsets`, `get_player_fields`, `initialize_game`
- `entities/player.pxd` — `_liquidity_offset`, `_market_offset`, liquidity methods
- `entities/player.pyx` — `calculate_liquidity` with simulated market, `update_net_worth` calls it
- `tests/phases/conftest.py` — non-negative invariant
- `tests/test_mcts.py` — removed hardcoded player_stride, replaced with dynamic check

### 4. Company: `companies_acquired` (36 flags, +36 total)

**Position:** After `companies_removed`, before `company_incomes` in company locations block.
**Encoding:** Binary flag per company. 1 = in a corp's acquisition zone this phase.
**Company locations:** 3 arrays → 4 arrays (auction, revealed, removed, acquired)

**Behavior changes:**
- `transfer_to_corp_acquisition` now sets `co:acquired` visible flag AND eagerly sets `corp:owned_companies`
- `_clear_visible_flag` handles `LOC_CORP_ACQ` (clears both acquired flag and corp:owned_companies)
- `transfer_to_corp` at phase exit works idempotently (owned_companies already set, acquired flag cleared)
- Model can cross-reference `co:acquired` + `corp:owned_companies` to identify acquiring corp

**Code changes:**
- `core/state.pxd` — `StateLayout.acquired_companies_offset`
- `core/state.pyx` — companies_size (3→4 arrays), layout offset, `LayoutInfo`/`get_layout`
- `entities/company.pxd` — `_acquired_offset`, `is_acquired` method
- `entities/company.pyx` — cache offset, `is_acquired`, `_clear_visible_flag` handles `LOC_CORP_ACQ`, `transfer_to_corp_acquisition` sets visible flags
- `tests/phases/conftest.py` — invariant for `LOC_CORP_ACQ` → `co:acquired` + `corp:owned_companies`; phase invariant (acquired flags zero outside ACQ)
- `tests/phases/test_acquisition.py` — updated assertions for eager `owned_companies`; set `PHASE_ACQUISITION` before all `setup_acquisition_phase_py` calls

## Pending Updates (do when all changes are done)

- [ ] `VECTORS.md` — corp stride, field table, field offsets, turn state fields, size table
- [ ] `CLAUDE.md` — state layout summary, size table
- [ ] `tests/test_state_layout.py` — expected sizes and corp_stride assertion
- [ ] `nn/model_3p.py` — input size (if hardcoded)
