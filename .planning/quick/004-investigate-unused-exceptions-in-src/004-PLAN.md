---
phase: quick-004
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - core/driver.pyx
  - tests/phases/test_invest.py
  - src/exceptions.py
  - src/__init__.py
autonomous: true

must_haves:
  truths:
    - "Exceptions are defined in core/driver.pyx"
    - "Tests import exceptions from core.driver"
    - "src/ directory no longer exists"
  artifacts:
    - path: "core/driver.pyx"
      provides: "ForcedActionLoopError, ZeroLegalActionsError definitions"
      contains: "class ForcedActionLoopError"
    - path: "tests/phases/test_invest.py"
      provides: "Updated imports"
      contains: "from core.driver import"
  key_links:
    - from: "core/driver.pyx"
      to: "raises exceptions"
      via: "defined in same module"
      pattern: "raise ForcedActionLoopError|raise ZeroLegalActionsError"
---

<objective>
Move exception classes from src/exceptions.py into core/driver.pyx and remove the src/ directory.

Purpose: Eliminate unnecessary src/ directory that only contained exceptions used by driver.pyx.
Output: Cleaner codebase with exceptions co-located with their usage.
</objective>

<execution_context>
@/home/icebreaker/.claude/get-shit-done/workflows/execute-plan.md
@/home/icebreaker/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@core/driver.pyx (lines 1-30 for imports, lines 225-250 for exception usage)
@src/exceptions.py (exception definitions to move)
@tests/phases/test_invest.py (lines 1155-1180 for tests that import exceptions)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Move exceptions to driver.pyx and update imports</name>
  <files>core/driver.pyx, tests/phases/test_invest.py</files>
  <action>
1. In core/driver.pyx:
   - Remove line 27: `from src.exceptions import ForcedActionLoopError, ZeroLegalActionsError`
   - Add exception class definitions after the imports (around line 27), before the DEF statements:
     ```python
     class ForcedActionLoopError(RuntimeError):
         """Raised when forced action loop exceeds iteration limit."""
         pass

     class ZeroLegalActionsError(RuntimeError):
         """Raised when zero legal actions exist outside GAME_OVER phase."""
         pass
     ```

2. In tests/phases/test_invest.py:
   - Line 1160: Change `from src.exceptions import ZeroLegalActionsError` to `from core.driver import ZeroLegalActionsError`
   - Line 1176: Change `from src.exceptions import ForcedActionLoopError` to `from core.driver import ForcedActionLoopError`
  </action>
  <verify>
python -c "from core.driver import ForcedActionLoopError, ZeroLegalActionsError; print('OK')"
python setup.py build_ext --inplace
pytest tests/phases/test_invest.py::TestExceptionGuards -v
  </verify>
  <done>Exceptions defined in driver.pyx, tests pass with new imports</done>
</task>

<task type="auto">
  <name>Task 2: Remove src/ directory</name>
  <files>src/exceptions.py, src/__init__.py</files>
  <action>
1. Delete src/exceptions.py
2. Delete src/__init__.py
3. Delete src/__pycache__/ directory if exists
4. Remove the src/ directory

Use: rm -rf src/
  </action>
  <verify>
test ! -d src && echo "src/ removed"
pytest tests/phases/test_invest.py -v
  </verify>
  <done>src/ directory completely removed, all tests still pass</done>
</task>

</tasks>

<verification>
- `python -c "from core.driver import ForcedActionLoopError, ZeroLegalActionsError"` succeeds
- `test ! -d src` confirms directory removed
- `pytest tests/phases/test_invest.py -v` passes
- No remaining imports from src.exceptions: `grep -r "from src" .` returns nothing relevant
</verification>

<success_criteria>
- Exceptions ForcedActionLoopError and ZeroLegalActionsError are defined in core/driver.pyx
- src/ directory does not exist
- All tests pass
- Build succeeds
</success_criteria>

<output>
After completion, create `.planning/quick/004-investigate-unused-exceptions-in-src/004-SUMMARY.md`
</output>
