# Phase 8: Test Updates — Context

## Fixture Design

### apply_and_track() Interface
- **Returns**: Wrapper object (not tuple)
- **Attributes**: `.state`, `.history`, `.applied_count`
- **Helper methods**: `.get_state_at(n)`, `.get_action_at(n)`, `.last_action`, etc.
- **Indexing**: Both directions — positive from start (0, 1, 2), negative from end (-1, -2)
- **Location**: Defined in `conftest.py`
- **Availability**: Alongside direct `apply_action()` calls, not replacing them

### History Semantics
- **Convention**: `history[0]` is the user-initiated action; all subsequent entries are auto-applied
- No explicit flag needed — position determines origin

## Test Categorization

### Handling Existing Tests
- **Strategy**: Update in-place (not skip, not delete)
- Tests asserting on intermediate states get rewritten to use `apply_and_track()` and inspect history

### Assertion Patterns
- Assert **both** intermediate states in history AND final state correctness
- Explicitly assert `len(result.history) == 1` when no auto-apply is expected
- This documents intent and catches regressions

### Sequential Call Consolidation
- Tests with `apply(A); apply(B); apply(C)` where B and C were forced → consolidate to single `apply_and_track(A)`
- Keep separate calls only when intermediate states had multiple valid actions

## Edge Case Coverage

### Chain Depth
- **Decision**: Claude determines realistic max based on game rules analysis
- No artificial stress tests needed

### Infinite Loop Guard
- **Decision**: Do not test explicitly — trust the implementation
- Guard exists for safety, not as testable behavior

### Phase Transitions
- **Decision**: Dedicated boundary tests for each phase transition
- Set up boundary state, verify auto-apply advances correctly to next player choice

### Foreign Investor
- **Decision**: FI-specific tests required
- Cover FI auto-actions in Phase 3 (Acquisition), Phase 4 (Closing), Phase 6 (Dividends), Phase 8 (Issue Share)

## Test Structure

### File Organization
- **Decision**: Extend existing test files (no new `test_autoloop.py`)
- Auto-apply tests go into relevant existing files based on what they test

### Parametrization
- **Decision**: Use `@pytest.mark.parametrize` for chain scenarios
- Test IDs: Descriptive, full words (e.g., `closing_phase_with_negative_income`)

### Markers
- **Decision**: No `@pytest.mark.autoloop` marker needed
- Tests organized by file/class structure is sufficient

## Deferred Ideas

(None captured during discussion)
