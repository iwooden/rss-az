---
phase: 17-offer-based-close-flow
verified: 2026-01-27T18:50:10Z
status: passed
score: 9/9 must-haves verified
---

# Phase 17: Offer-Based Close Flow Verification Report

**Phase Goal:** Players can decide to close or keep negative-income companies via offer system
**Verified:** 2026-01-27T18:50:10Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Tests verify CLO-05: only negative adjusted income companies offered | ✓ VERIFIED | test_only_negative_income_offered, test_zero_income_not_offered pass |
| 2 | Tests verify CLO-06: offers sorted by face value ascending | ✓ VERIFIED | test_offers_sorted_by_face_value_ascending passes |
| 3 | Tests verify CLO-07: player-owned privates included | ✓ VERIFIED | test_player_privates_included passes |
| 4 | Tests verify CLO-08: corp subsidiaries (same-president) included | ✓ VERIFIED | test_corp_subsidiaries_included, test_receivership_corp_excluded, test_fi_excluded pass |
| 5 | Tests verify CLO-09: corp last-company rule enforced | ✓ VERIFIED | test_corp_last_company_rule, test_corp_with_multiple_companies_can_close pass |
| 6 | Tests verify CLO-10: dynamic re-validation skips invalidated offers | ✓ VERIFIED | test_prior_acceptance_invalidates_later_offer passes (integration test) |
| 7 | Tests verify CLO-11: accept action closes company | ✓ VERIFIED | test_accept_closes_company passes |
| 8 | Tests verify CLO-12: pass action keeps company | ✓ VERIFIED | test_pass_keeps_company passes |
| 9 | Tests verify CLO-13: Junkyard Scrappers receives 2x bonus | ✓ VERIFIED | test_junkyard_scrappers_bonus_on_player_close, test_junkyard_scrappers_bonus_on_corp_close pass |

**Score:** 9/9 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/phases/test_closing.py` | Comprehensive tests for offer-based closing | ✓ VERIFIED | 660 lines, 3 test classes (TestOfferGeneration, TestOfferValidation, TestCloseActions), 14 new tests covering CLO-05 through CLO-13 |
| `tests/conftest.py` | closing_offer_state fixture exported | ✓ VERIFIED | 32 lines, imports and exports closing_offer_state from tests/phases/conftest.py |
| `tests/phases/conftest.py` | closing_offer_state fixture implementation | ✓ VERIFIED | 272 lines, fixture creates GameState with high CoO level for negative income scenarios |
| `phases/closing.pyx` | Python test wrappers | ✓ VERIFIED | 635 lines, 5 Python wrappers: apply_closing_action_py, get_close_offer_count_py, get_close_offer_index_py, get_close_offer_py, generate_close_offers_py |

**Artifact Quality Checks:**

**tests/phases/test_closing.py:**
- EXISTS: ✓ File exists
- SUBSTANTIVE: ✓ 660 lines (far exceeds 15-line minimum for test files)
- NO_STUBS: ✓ No TODO/FIXME patterns, no placeholder returns
- HAS_EXPORTS: N/A (test file, not a module)
- WIRED: ✓ Imported by pytest, uses phases.closing functions extensively (20+ import statements)

**phases/closing.pyx:**
- EXISTS: ✓ File exists
- SUBSTANTIVE: ✓ 635 lines (far exceeds minimum)
- NO_STUBS: ✓ All functions have complete implementations
- HAS_EXPORTS: ✓ 5 Python wrappers exported for testing (_py suffix functions)
- WIRED: ✓ Imported by tests/phases/test_closing.py (20+ import statements), imported by core/driver.pyx (apply_closing_action)

### Key Link Verification

| From | To | Via | Status | Details |
|------|------|-----|--------|---------|
| tests/phases/test_closing.py | phases/closing.pyx | Python wrappers | ✓ WIRED | 20+ import statements, all test functions call closing.pyx via apply_closing_auto_py, apply_closing_action_py, generate_close_offers_py, get_close_offer_count_py, get_close_offer_py |
| phases/closing.pyx | entities (turn, player, corp, company, fi) | Direct module imports | ✓ WIRED | Uses turn_module.TURN, player_module.PLAYERS, corp_module.CORPS, company_module.COMPANIES, fi_module.FI throughout implementation |
| core/driver.pyx | phases/closing.pyx | apply_closing_action import | ✓ WIRED | Line 26: "from phases.closing cimport apply_closing_auto, apply_closing_action", Line 178: "result = apply_closing_action(state, &info)" |
| tests | conftest fixtures | closing_offer_state fixture | ✓ WIRED | tests/conftest.py imports from tests/phases/conftest.py, all TestOfferGeneration/TestOfferValidation/TestCloseActions use closing_offer_state fixture |

**Key Link Details:**

**Test → Implementation:**
- Pattern: Tests use Python wrappers (_py suffix) to call internal Cython functions
- Evidence: 20+ import statements in test_closing.py importing from phases.closing
- Quality: Direct function calls with result assertions, no mocking or stubs

**Driver → Closing Phase:**
- Pattern: Driver dispatches CLOSING phase actions to apply_closing_action
- Evidence: core/driver.pyx line 26 (import), line 178 (dispatch)
- Quality: Integrated into hybrid phase detection pattern (closing_company == -1 check)

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| CLO-05: Only negative adjusted income offered | ✓ SATISFIED | None — test_only_negative_income_offered, test_zero_income_not_offered verify filtering logic |
| CLO-06: Sort by face value ascending | ✓ SATISFIED | None — test_offers_sorted_by_face_value_ascending verifies ascending order |
| CLO-07: Player privates included | ✓ SATISFIED | None — test_player_privates_included verifies player-owned companies in offers |
| CLO-08: Corp subsidiaries included | ✓ SATISFIED | None — test_corp_subsidiaries_included, test_receivership_corp_excluded verify corp ownership rules |
| CLO-09: Corp last-company rule | ✓ SATISFIED | None — test_corp_last_company_rule, test_corp_with_multiple_companies_can_close verify validation |
| CLO-10: Dynamic re-validation | ✓ SATISFIED | None — test_prior_acceptance_invalidates_later_offer verifies skipping invalidated offers |
| CLO-11: Accept closes company | ✓ SATISFIED | None — test_accept_closes_company verifies company removed from game |
| CLO-12: Pass keeps company | ✓ SATISFIED | None — test_pass_keeps_company verifies company retained |
| CLO-13: Junkyard Scrappers bonus | ✓ SATISFIED | None — test_junkyard_scrappers_bonus_on_player_close, test_junkyard_scrappers_bonus_on_corp_close verify 2x bonus |

### Anti-Patterns Found

No anti-patterns detected. All files are production-quality implementations with complete logic.

**Scan Results:**
- TODO/FIXME comments: 0 instances in modified files
- Placeholder content: 0 instances
- Empty implementations: 0 instances
- Console.log only implementations: 0 instances

### Human Verification Required

None. All requirements are verifiable programmatically through unit tests.

---

## Detailed Verification

### Truth 1: CLO-05 Negative Income Filter

**Requirement:** Only companies with negative adjusted income (income - CoO < 0) are offered.

**Test Evidence:**
- `test_only_negative_income_offered`: Player owns company 0 (income $1, CoO $6 = adjusted -$5) and company 29 (income $10, CoO $0 = adjusted $10). Only company 0 offered. ✓ PASS
- `test_zero_income_not_offered`: Company 2 (income $2, CoO $2 = adjusted $0) is NOT offered. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` line 87-93: `_has_negative_adjusted_income()` returns `(base_income - coo_value) < 0` (strict negative, not <=)
- Lines 121-122, 162-163: Filter applied in `_collect_player_close_offers()` and `_collect_corp_close_offers()`

**Status:** ✓ VERIFIED — Implementation matches requirement, tests confirm behavior.

### Truth 2: CLO-06 Face Value Ascending Sort

**Requirement:** Offers sorted by face value ascending (lowest first).

**Test Evidence:**
- `test_offers_sorted_by_face_value_ascending`: Player owns companies 0 (FV $1), 6 (FV $5), 3 (FV $3). Offers sorted: company 0, company 3, company 6. Verification: `assert get_company_face_value(cid0) < get_company_face_value(cid1) < get_company_face_value(cid2)`. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 178-213: `_sort_close_offers_by_face_value()` selection sort algorithm
- Line 192: `if curr_fv < best_fv:` ensures ascending order (lowest first)
- Lines 247-249: Sort called after collecting all offers

**Status:** ✓ VERIFIED — Implementation uses selection sort with ascending comparison, test confirms output order.

### Truth 3: CLO-07 Player Privates Included

**Requirement:** Player-owned private companies are included in offers.

**Test Evidence:**
- `test_player_privates_included`: Player 1 owns company 1. Offer generated with owner_type=0 (OWNER_PLAYER), owner_id=1, company_id=1. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 105-134: `_collect_player_close_offers()` iterates all players and their owned companies
- Line 119: `if not player_module.PLAYERS[player_id].owns_company(state, company_id): continue`
- Line 128: `owner_types[idx] = OWNER_PLAYER`

**Status:** ✓ VERIFIED — Implementation collects player-owned companies, test confirms generation.

### Truth 4: CLO-08 Corp Subsidiaries Included

**Requirement:** Corp subsidiaries (same-president) included in offers. Receivership corps excluded.

**Test Evidence:**
- `test_corp_subsidiaries_included`: Corp 1 (active, not receivership, player 0 president) owns company 2. Offer generated with owner_type=1 (OWNER_CORP), owner_id=1, company_id=2. ✓ PASS
- `test_receivership_corp_excluded`: Corp 2 (receivership) owns company 4. No offers generated. ✓ PASS
- `test_fi_excluded`: FI owns company 5. No offers generated. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 137-175: `_collect_corp_close_offers()` iterates corps
- Line 151: `if not corp_module.CORPS[corp_id].is_active(state): continue`
- Lines 155-157: `president = _get_corp_president(state, corp_id); if president < 0: continue` (excludes receivership)
- Line 169: `owner_types[idx] = OWNER_CORP`
- FI exclusion: FI companies never owned by corps, so naturally excluded

**Status:** ✓ VERIFIED — Implementation filters by active + has-president, tests confirm inclusion/exclusion rules.

### Truth 5: CLO-09 Corp Last-Company Rule

**Requirement:** Corp closing offer invalid if corp would have 0 companies after close.

**Test Evidence:**
- `test_corp_with_multiple_companies_can_close`: Corp 1 owns companies 3 AND 4. Both offered (corp won't have 0 after closing one). ✓ PASS
- `test_prior_acceptance_invalidates_later_offer`: Corp 1 owns companies 0 and 3. First offer (company 0) accepted, second offer (company 3) SKIPPED (corp now has 1 company, closing would leave 0). Phase transitions to INVEST. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 261-276: `_count_corp_companies()` counts companies excluding target
- Lines 279-306: `_is_close_offer_valid()` checks last-company rule at lines 302-304: `if _count_corp_companies(state, owner_id, company_id) < 1: return False`
- Lines 353-357: `_present_next_close_offer()` calls `_is_close_offer_valid()` and skips invalid offers

**Status:** ✓ VERIFIED — Implementation enforces rule at presentation time (dynamic validation), tests confirm skipping behavior.

### Truth 6: CLO-10 Dynamic Re-Validation

**Requirement:** Prior acceptance can invalidate later offers (corp down to 1 company).

**Test Evidence:**
- `test_prior_acceptance_invalidates_later_offer`: Corp 1 has 2 companies. First offer (company 0) accepted → company closed. Second offer (company 3) becomes invalid (corp now has 1 company). Offer skipped, phase transitions to INVEST. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 336-372: `_present_next_close_offer()` loops through offers
- Lines 353-357: Each offer validated via `_is_close_offer_valid()` before presentation
- Line 355: `index += 1` advances past invalid offers
- Dynamic validation ensures state changes (prior accepts) invalidate later offers

**Status:** ✓ VERIFIED — Implementation validates at presentation time (not generation time), test confirms invalidation cascades.

### Truth 7: CLO-11 Accept Closes Company

**Requirement:** Accept action removes company from game.

**Test Evidence:**
- `test_accept_closes_company`: Player 0 owns company 1. Accept offer. Company removed: `assert COMPANIES[1].is_removed(gs)`. Player no longer owns: `assert not PLAYERS[0].owns_company(gs, 1)`. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 482-522: `_handle_close_accept()` processes accept action
- Lines 506-515: For player-owned, clears ownership, applies JS bonus, removes from game
- Line 515: `company_module.COMPANIES[company_id].remove_from_game(state)`
- Line 517: For corp-owned, calls `_close_company()` helper (which also removes)

**Status:** ✓ VERIFIED — Implementation removes company from game, test confirms removal and ownership clearing.

### Truth 8: CLO-12 Pass Keeps Company

**Requirement:** Pass action keeps company.

**Test Evidence:**
- `test_pass_keeps_company`: Player 0 owns company 2. Pass on offer. Company NOT removed: `assert not COMPANIES[2].is_removed(gs)`. Player still owns: `assert PLAYERS[0].owns_company(gs, 2)`. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 524-533: `_handle_close_pass()` processes pass action
- Line 531: `state._data[state._layout.hidden_close_offer_index_offset] = <float>(index + 1)` advances index
- Line 532: `_present_next_close_offer(state)` moves to next offer
- No company removal or ownership change

**Status:** ✓ VERIFIED — Implementation only advances offer index, test confirms company retained.

### Truth 9: CLO-13 Junkyard Scrappers Bonus

**Requirement:** Junkyard Scrappers receives 2x printed income as bonus when closing.

**Test Evidence:**
- `test_junkyard_scrappers_bonus_on_player_close`: JS active with $100 cash. Player closes company 1 (income $1). JS receives $2 bonus. Final cash: $102. ✓ PASS
- `test_junkyard_scrappers_bonus_on_corp_close`: JS active with $50 cash. Corp closes company 0 (income $1). JS receives $2 bonus. Final cash: $52. ✓ PASS

**Implementation Evidence:**
- `phases/closing.pyx` lines 509-512 (player-owned): `printed_income = get_company_income(company_id); if corp_module.CORPS[0].is_active(state): corp_module.CORPS[0].add_cash(state, printed_income * 2)`
- Lines 71-81 (`_close_company()` for corp-owned): Same logic at lines 79-81
- Bonus = printed income * 2

**Status:** ✓ VERIFIED — Implementation applies 2x bonus for both player and corp closes, tests confirm cash amounts.

---

## Implementation Quality

### Code Organization
- **Modular design:** Offer generation, validation, presentation, and action handling are separate functions
- **Reusable helpers:** `_has_negative_adjusted_income()`, `_get_corp_president()`, `_count_corp_companies()` used across multiple functions
- **Clear separation:** Phase 16 auto-close logic separate from Phase 17 offer-based logic
- **Consistent patterns:** Follows ACQUISITION phase patterns (hidden buffer, hybrid phase detection)

### Test Coverage
- **28 total tests** in test_closing.py (14 for Phase 16 auto-close, 14 for Phase 17 offer-based)
- **3 test classes** for Phase 17: TestOfferGeneration (7 tests), TestOfferValidation (3 tests), TestCloseActions (4 tests)
- **All 9 requirements** (CLO-05 through CLO-13) have dedicated test coverage
- **Edge cases covered:** zero income, receivership exclusion, FI exclusion, corp last-company rule, dynamic re-validation
- **Integration test:** test_prior_acceptance_invalidates_later_offer exercises full flow (auto-close → offer generation → accept → re-validation → skip → transition)

### Wiring Verification
- **Driver integration:** core/driver.pyx imports apply_closing_action, dispatches CLOSING phase actions
- **Hybrid phase detection:** closing_company == -1 check in _is_non_player_phase_check()
- **Python wrappers:** 5 test wrappers enable white-box testing of internal functions
- **Fixture design:** closing_offer_state provides consistent test setup with high CoO level

### No Technical Debt
- **No TODOs or FIXMEs** in implementation files
- **No placeholder implementations** or stub patterns
- **Complete error handling:** Action validation checks for active offer before processing
- **Proper memory management:** Stack-allocated arrays for offer generation (no dynamic allocation)

---

_Verified: 2026-01-27T18:50:10Z_
_Verifier: Claude (gsd-verifier)_
