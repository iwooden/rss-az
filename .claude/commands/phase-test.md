Implement tests for the **$ARGUMENTS** phase in `tests/phases/`.

## Setup

1. Read the phase implementation: `phases/$ARGUMENTS.pyx` and `phases/$ARGUMENTS.pxd`. Understand every code path, branch, and edge case.
2. Read the corresponding action enumerator in `core/actions.pyx` — search for `_enumerate_$ARGUMENTS` or the relevant `DPHASE_*` constant. Understand which actions are legal and when.
3. Read the shared test infrastructure in `tests/phases/conftest.py`. Understand every fixture and helper available — you MUST reuse them. Do not duplicate logic that already exists there.
4. Read the old tests on `main` for guidance on edge cases (NOT for code to copy): `git show main:tests/phases/test_$ARGUMENTS.py 2>/dev/null`. The old tests may use a completely different API — extract only the *test ideas* (what scenarios matter, what boundaries to check).
5. Read `RULES.md` for the rules governing this phase — understand the game logic you're testing.

## Architecture rules

- **Reuse `conftest.py` fixtures and helpers.** `game_state`, `float_corp_for_test`, `setup_receivership_corp`, `apply_and_verify`, `assert_invariants`, `get_legal_actions`, `find_legal_action` — use them. If you need a new fixture or helper that would benefit multiple test files, add it to `conftest.py`, not to the test file.
- **No method-level imports.** All imports at file scope, top of file. No exceptions.
- **No code duplication.** If you find yourself writing the same setup logic in multiple tests, extract it into a fixture or helper (in conftest.py if reusable, as a module-level helper in the test file if phase-specific).
- **Test through the driver.** Use `apply_and_verify(state, action_id)` which calls `DRIVER.apply_action()` and checks invariants on every intermediate state. Do NOT call phase-internal `cdef` functions directly. The `_py` wrappers on some phases exist for ad-hoc debugging, not for tests — the driver is the contract.
- **Scope tests to the phase under test.** Phase tests should only test logic within the specific phase and its transition to the next phase. Do NOT transit through earlier phases to reach the phase under test — instead, set up state directly and call the phase's `setup_*_phase_py()` entry point. This avoids coupling to other phases' side effects and keeps tests focused.
- **Use `find_legal_action` to locate actions.** Don't hardcode action IDs. Use `find_legal_action(state, action_type=..., corp_id=..., company_id=..., amount=...)` to find the right action by its decoded semantics.
- **Use `get_legal_actions` to inspect the full legal set.** Returns list of `(action_id, decoded_info)` tuples for the current state.
- **Encode action IDs when the mapping is simple.** For phases with trivial encoding (e.g., CLOSING where action 0 = pass, action N = close company N), you can construct action IDs directly using the encode helpers from `core.actions` — but prefer `find_legal_action` for complex encodings.
- **No hacks to work around entity handle gaps.** If a test requires data or behavior that isn't accessible through the entity handles (e.g., looking up a static price, computing a derived value), do NOT work around it by temporarily mutating state, hardcoding data constants, or importing `cdef`-only symbols. Instead, stop and propose the missing method/accessor to the user. A clean entity handle addition is always preferable to a fragile test workaround.

## What to test

### Coverage targets
- **Happy path:** Each distinct action type the phase supports (pass, buy, sell, raise, close, etc.)
- **State mutations:** Verify the action correctly modifies state (cash changes, share transfers, phase transitions, entity ownership changes, etc.)
- **Phase transitions:** Verify the phase correctly transitions to the next phase when done (all corps processed, all players passed, etc.)
- **Forced actions:** If the phase has auto-applied forced actions (receivership corps, single legal action), verify they chain correctly through the driver.
- **Boundary conditions:** Test at exact thresholds (cash == cost, exactly 0 income, last company/corp, etc.)
- **Enumeration correctness:** Verify that the set of legal actions matches expectations for key scenarios (e.g., which companies are closable, which corps can issue, which par prices are valid).

### What NOT to test
- Don't test invariants directly — `apply_and_verify` already checks `assert_invariants` on every intermediate and final state.
- Don't test action encoding/decoding — that's `core/actions`'s concern.
- Don't test other phases' logic (e.g., don't test that INCOME works correctly inside a CLOSING test — just verify the phase transition happens).
- Don't write tests for scenarios that can't arise in legal play.

## Style

- Use `pytest` classes to group related tests (e.g., `TestPassAction`, `TestBuyShare`, `TestPhaseTransition`).
- Descriptive method names: `test_close_company_removes_from_game`, not `test_close_1`.
- Keep each test focused on ONE behavior. Multiple related assertions in one test are fine; testing two unrelated behaviors is not.
- Minimal state setup — use fixtures and helpers, then make the smallest possible mutation to reach the scenario under test.
- Include a brief docstring on tests where the scenario isn't obvious from the name.

## File structure

Write the test file to `tests/phases/test_$ARGUMENTS.py`. If you need new shared fixtures or helpers, add them to `tests/phases/conftest.py`.

## Verification

After writing the tests, run them:
```bash
.venv/bin/python -m pytest tests/phases/test_$ARGUMENTS.py -v 2>&1 | tail -60
```

All tests must pass. If a test fails, investigate whether it's a test bug or an implementation bug (per CLAUDE.md: "assume the implementation is broken until proven otherwise"). Fix test bugs; file a beads issue for implementation bugs.
