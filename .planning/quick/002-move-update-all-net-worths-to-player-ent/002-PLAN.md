---
phase: quick
plan: 002
type: execute
wave: 1
depends_on: []
files_modified:
  - entities/player.pyx
  - phases/invest.pyx
autonomous: true

must_haves:
  truths:
    - "Inline net worth update loops replaced with single function call"
    - "All tests still pass after refactoring"
  artifacts:
    - path: "entities/player.pyx"
      provides: "update_all_net_worths() module-level wrapper function"
      contains: "def update_all_net_worths"
    - path: "phases/invest.pyx"
      provides: "Calls to player_module.update_all_net_worths(state)"
      contains: "player_module.update_all_net_worths"
  key_links:
    - from: "phases/invest.pyx"
      to: "entities/player.pyx"
      via: "player_module.update_all_net_worths(state)"
      pattern: "player_module\\.update_all_net_worths\\(state\\)"
---

<objective>
Refactor inline net worth update loops to use centralized helper function.

Purpose: The previous refactoring removed the `_update_all_net_worths` helper from `invest.pyx` but inlined the implementation at three callsites instead of using the existing `update_all_player_net_worths` cdef function in `player.pyx`. This creates code duplication. The fix is to expose the existing function as Python-visible and call it from `invest.pyx`.

Output: Clean refactored code with single responsibility - net worth updates go through `player.pyx`.
</objective>

<execution_context>
@~/.claude/get-shit-done/workflows/execute-plan.md
@~/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@entities/player.pyx
@phases/invest.pyx
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add Python-visible wrapper and replace inline loops</name>
  <files>entities/player.pyx, phases/invest.pyx</files>
  <action>
1. In `entities/player.pyx`, add a Python-visible wrapper function after the existing `update_all_player_net_worths` cdef function (around line 222):

```python
def update_all_net_worths(GameState state):
    """Update net worth for all players. Python-visible wrapper."""
    update_all_player_net_worths(state, state._num_players)
```

This follows the existing pattern of having low-level cdef functions with high-level wrappers.

2. In `phases/invest.pyx`, replace the three inline loops with calls to the new wrapper:

Location 1 - `_handle_buy_share` (lines 237-239):
Replace:
```python
# Update net worth for all players (price movement affects all shareholders)
for i in range(state._num_players):
    player_module.PLAYERS[i].update_net_worth(state)
```
With:
```python
# Update net worth for all players (price movement affects all shareholders)
player_module.update_all_net_worths(state)
```

Location 2 - `_handle_sell_share` bankruptcy path (lines 297-299):
Replace:
```python
# Update net worth for all players (bankruptcy affects all shareholders)
for i in range(state._num_players):
    player_module.PLAYERS[i].update_net_worth(state)
```
With:
```python
# Update net worth for all players (bankruptcy affects all shareholders)
player_module.update_all_net_worths(state)
```

Location 3 - `_handle_sell_share` normal path (lines 319-321):
Replace:
```python
# Update net worth for all players (price movement affects all shareholders)
for i in range(state._num_players):
    player_module.PLAYERS[i].update_net_worth(state)
```
With:
```python
# Update net worth for all players (price movement affects all shareholders)
player_module.update_all_net_worths(state)
```

3. Remove the now-unused loop variable `i` declaration from both `_handle_buy_share` and `_handle_sell_share` cdef var blocks (the `cdef int i` lines).
  </action>
  <verify>
Build: `python setup.py build_ext --inplace`
Test: `pytest tests/ -v`
Grep: `grep -n "update_all_net_worths" phases/invest.pyx` shows 3 calls
Grep: `grep -n "for i in range.*update_net_worth" phases/invest.pyx` shows 0 matches
  </verify>
  <done>
- `player.pyx` has `def update_all_net_worths(GameState state)` wrapper
- `invest.pyx` calls `player_module.update_all_net_worths(state)` at 3 locations
- No inline loops remain for net worth updates
- All tests pass
  </done>
</task>

</tasks>

<verification>
- Build succeeds: `python setup.py build_ext --inplace`
- All tests pass: `pytest tests/ -v`
- Code pattern verified: `grep -c "player_module.update_all_net_worths" phases/invest.pyx` returns 3
- No inline loops: `grep "for i in range.*_num_players" phases/invest.pyx | grep -v "#"` returns empty
</verification>

<success_criteria>
- Single function call replaces 3 inline loops
- Existing cdef function leveraged (no new logic)
- Tests pass without modification
- Build succeeds
</success_criteria>

<output>
After completion, create `.planning/quick/002-move-update-all-net-worths-to-player-ent/002-SUMMARY.md`
</output>
