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

## Pending Updates (do when all changes are done)

- [ ] `VECTORS.md` — corp stride, field table, field offsets, size table
- [ ] `CLAUDE.md` — state layout summary, size table
- [ ] `tests/test_state_layout.py` — expected sizes and corp_stride assertion
- [ ] `nn/model_3p.py` — input size (if hardcoded)
