---
phase: quick
plan: 005
type: execute
wave: 1
depends_on: []
files_modified:
  - phases/closing.pyx
  - tests/phases/test_closing.py
autonomous: true

must_haves:
  truths:
    - "JS only gets scrapping bonus when JS itself closes a company it owns"
    - "JS does NOT get bonus when FI closes a company"
    - "JS does NOT get bonus when another corp closes a company"
    - "JS does NOT get bonus when a player closes their private company"
  artifacts:
    - path: "phases/closing.pyx"
      provides: "Fixed JS bonus logic in _close_company, _close_player_company, _handle_close_accept"
    - path: "tests/phases/test_closing.py"
      provides: "Corrected tests for JS bonus behavior"
  key_links:
    - from: "_close_company()"
      to: "JS bonus"
      via: "owner_id == 0 check"
      pattern: "owner_type == 5.*owner_id == 0"
---

<objective>
Fix Junkyard Scrappers (JS) scrapping bonus bug - bonus should only apply when JS closes its own company.

Purpose: The current implementation incorrectly gives JS the 2x printed income bonus for ANY company closed by anyone. Per RULES.md: "When closing a company, immediately receives 2x printed income as scrapping bonus" - this means when JS closes one of **its own** companies.

Output: Fixed closing.pyx with correct JS bonus logic, updated tests validating correct behavior
</objective>

<execution_context>
@/home/icebreaker/.claude/get-shit-done/workflows/execute-plan.md
@/home/icebreaker/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@phases/closing.pyx
@tests/phases/test_closing.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix JS bonus logic in phases/closing.pyx</name>
  <files>phases/closing.pyx</files>
  <action>
Fix three locations where JS bonus is incorrectly applied:

1. **`_close_company()` lines 80-82** - Used for FI and corp closures:
   - Current (WRONG): Always gives JS bonus if JS is active
   - Fix: Only apply bonus if `owner_type == 5` (LOC_CORP) AND `owner_id == 0` (JS)
   - Replace:
     ```cython
     # Junkyard Scrappers (corp_id 0) bonus: 2x printed income
     if corp_module.CORPS[0].is_active(state):
         corp_module.CORPS[0].add_cash(state, printed_income * 2)
     ```
   - With:
     ```cython
     # Junkyard Scrappers (corp_id 0) bonus: 2x printed income only when JS closes its own company
     if owner_type == 5 and owner_id == 0:  # LOC_CORP and JS
         corp_module.CORPS[0].add_cash(state, printed_income * 2)
     ```

2. **`_close_player_company()` lines 104-106** - Used for player mandatory close:
   - Current (WRONG): Gives JS bonus when player-owned private company is closed
   - Fix: Remove entirely - player-owned private companies can never be JS-owned
   - Delete lines 104-106 (the JS bonus block)
   - Also update docstring to remove mention of Junkyard Scrappers bonus

3. **`_handle_close_accept()` lines 590-593** - Used for player-initiated closes:
   - This is inside the `if owner_type == OWNER_PLAYER:` block (line 586)
   - Current (WRONG): Gives JS bonus for player closes
   - Fix: Remove entirely - this path is for player-owned companies which are never JS-owned
   - Delete lines 590-593 (the JS bonus block inside OWNER_PLAYER)
   - The corp path (elif owner_type == OWNER_CORP) calls `_close_company` which has the fix from step 1
  </action>
  <verify>
    python setup.py build_ext --inplace
    Verify no syntax errors in the build
  </verify>
  <done>
    - _close_company() only applies JS bonus when owner_type == 5 AND owner_id == 0
    - _close_player_company() has NO JS bonus (removed)
    - _handle_close_accept() has NO JS bonus in OWNER_PLAYER block (removed)
  </done>
</task>

<task type="auto">
  <name>Task 2: Fix and add JS bonus tests in test_closing.py</name>
  <files>tests/phases/test_closing.py</files>
  <action>
Fix existing tests and add new test to verify correct JS bonus behavior:

1. **Rename and fix `test_junkyard_scrappers_bonus_on_player_close` (line 609)**:
   - Rename to: `test_junkyard_scrappers_no_bonus_on_player_close`
   - Change assertion: JS should NOT get bonus when a player closes their own company
   - Update test to verify JS cash DOES NOT change when player accepts close offer

2. **Rename and fix `test_junkyard_scrappers_bonus_on_corp_close` (line 633)**:
   - Rename to: `test_junkyard_scrappers_no_bonus_on_other_corp_close`
   - This test has corp 1 closing a company, so JS should NOT get bonus
   - Update test to verify JS cash DOES NOT change when another corp closes

3. **Add new test `test_junkyard_scrappers_bonus_only_when_js_closes`**:
   - Add to TestCloseActions class after the two fixed tests
   - Set up: JS (corp 0) is active with president, owns 2 companies
   - Have JS close one company via accept action
   - Verify JS cash increases by 2x printed income of closed company

4. **Fix `test_mandatory_close_js_bonus` in TestMandatoryClose class (line 810)**:
   - Rename to: `test_mandatory_close_no_js_bonus`
   - Update docstring to say "Junkyard Scrappers does NOT receive bonus on mandatory player close"
   - Change assertion to verify JS cash does NOT change after mandatory close

5. **Fix `test_multi_close_cascade_js_bonus` (line 1014)**:
   - Rename to: `test_multi_close_cascade_no_js_bonus_for_player_closes`
   - Update test: Player closes companies, JS should NOT get any bonus
   - Verify JS cash remains at initial value (50)
   - Keep test structure but change expected_total to just initial_js_cash (50)
  </action>
  <verify>
    pytest tests/phases/test_closing.py -v -k "junkyard" --tb=short
    pytest tests/phases/test_closing.py -v -k "mandatory_close" --tb=short
    pytest tests/phases/test_closing.py -v -k "multi_close" --tb=short
  </verify>
  <done>
    - test_junkyard_scrappers_no_bonus_on_player_close passes (verifies no bonus for player close)
    - test_junkyard_scrappers_no_bonus_on_other_corp_close passes (verifies no bonus when other corp closes)
    - test_junkyard_scrappers_bonus_only_when_js_closes passes (verifies bonus when JS closes its own)
    - test_mandatory_close_no_js_bonus passes (verifies no bonus on mandatory close)
    - test_multi_close_cascade_no_js_bonus_for_player_closes passes (verifies no bonus accumulation)
  </done>
</task>

<task type="auto">
  <name>Task 3: Verify all closing tests pass</name>
  <files>tests/phases/test_closing.py</files>
  <action>
Run full test suite for closing phase to ensure:
1. All JS bonus tests pass with new logic
2. FI auto-close tests still pass (no JS bonus for FI closes)
3. Receivership auto-close tests still pass (no JS bonus for receivership closes)
4. All other closing tests remain green
  </action>
  <verify>
    pytest tests/phases/test_closing.py -v --tb=short
  </verify>
  <done>
    All tests in test_closing.py pass, confirming:
    - JS bonus ONLY applies when JS itself closes a company it owns
    - No regressions in other closing phase functionality
  </done>
</task>

</tasks>

<verification>
1. Build succeeds: `python setup.py build_ext --inplace`
2. All JS bonus tests pass with correct behavior
3. Full closing test suite passes: `pytest tests/phases/test_closing.py -v`
4. No regressions in related phases
</verification>

<success_criteria>
- JS bonus applies ONLY when owner_type == LOC_CORP (5) AND owner_id == 0 (JS)
- JS does NOT get bonus for FI closes (existing tests pass without change)
- JS does NOT get bonus for player closes (fixed tests verify this)
- JS does NOT get bonus for other corp closes (fixed tests verify this)
- JS DOES get bonus when JS closes its own company (new test verifies this)
- All 30+ tests in test_closing.py pass
</success_criteria>

<output>
After completion, create `.planning/quick/005-fix-js-scrapping-bonus-bug/005-SUMMARY.md`
</output>
