# Phase 19: Testing and Integration - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Comprehensive test coverage validating CLOSING phase correctness (Phases 16-18). Unit tests for all 16 CLO requirements, integration tests for ACQUISITION -> CLOSING -> INCOME flow, and edge case coverage. No new functionality — validation only.

</domain>

<decisions>
## Implementation Decisions

### Test Organization
- Consolidate all CLOSING-specific tests in `tests/phases/test_closing.py`
- Cross-phase/integration tests extend `tests/test_integration.py`
- Do not create new files per requirement or feature group
- Use docstrings for requirement traceability (existing pattern)

### Coverage Scope
- Comprehensive edge case testing — every boundary condition, empty states, single-item cases, max capacity
- Include negative tests explicitly — invalid close attempts, wrong phase errors, error conditions
- Player count testing: use existing fixture if available, otherwise test min (3) and max (6) players
- Use `@pytest.mark.parametrize` for grouping similar test cases

### Integration Depth
- Full driver loop for ACQUISITION -> CLOSING -> INCOME flow
- Extend existing whole-turn tests that stopped at ACQUISITION
- Scripted action sequences (not random) for predictable path coverage
- Verify invariants after each action applied
- Use existing test infrastructure for state/action history interaction

### Test Data Approach
- Use existing fixtures from conftest.py as primary approach
- Add new fixtures to conftest.py when scenarios not covered (prefer fixtures over inline setup)
- Realistic game values for integration tests, simplified values for unit tests
- Reusable constants in conftest.py, test-specific constants in test file

### Claude's Discretion
- Test function naming convention (based on existing codebase patterns)
- Test grouping within file (classes vs flat functions based on existing style)
- Number of integration test scenarios (based on complexity and existing patterns)

</decisions>

<specifics>
## Specific Ideas

- "Use existing test infrastructure as much as possible in favor of creating anything new"
- Extend whole-turn tests that currently stop at ACQUISITION phase
- Follow existing docstring pattern for requirement documentation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 19-testing-and-integration*
*Context gathered: 2026-01-27*
