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

## Pending Updates (do when all changes are done)

- [ ] `VECTORS.md` — corp stride, field table, field offsets, turn state fields, size table
- [ ] `CLAUDE.md` — state layout summary, size table
- [ ] `tests/test_state_layout.py` — expected sizes and corp_stride assertion
- [ ] `nn/model_3p.py` — input size (if hardcoded)
