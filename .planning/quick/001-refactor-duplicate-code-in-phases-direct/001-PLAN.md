---
phase: quick
plan: 001
type: execute
wave: 1
depends_on: []
files_modified:
  - entities/turn.pyx
  - entities/turn.pxd
  - phases/invest.pyx
  - phases/bid.pyx
autonomous: true

must_haves:
  truths:
    - "No duplicate helper functions exist across phase files"
    - "Turn order navigation is centralized in TurnState entity"
    - "All existing tests pass after refactor"
  artifacts:
    - path: "entities/turn.pyx"
      provides: "Centralized turn order navigation helpers"
      contains: "find_player_at_position"
    - path: "phases/invest.pyx"
      provides: "INVEST phase handler using shared helpers"
    - path: "phases/bid.pyx"
      provides: "BID phase handler using shared helpers"
  key_links:
    - from: "phases/invest.pyx"
      to: "entities/turn.pyx"
      via: "TurnState.find_player_at_position"
    - from: "phases/bid.pyx"
      to: "entities/turn.pyx"
      via: "TurnState.find_player_at_position"
---

<objective>
Refactor duplicate code in phases directory to centralized locations.

Purpose: Eliminate code duplication between invest.pyx and bid.pyx by moving shared turn-order navigation helpers to the TurnState entity.
Output: Clean phase handlers that call shared entity methods instead of duplicating logic.
</objective>

<context>
@.planning/STATE.md
@entities/turn.pyx
@entities/turn.pxd
@phases/invest.pyx
@phases/bid.pyx
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add turn order navigation methods to TurnState</name>
  <files>entities/turn.pyx, entities/turn.pxd</files>
  <action>
Add three new methods to TurnState class that handle turn order navigation:

1. `find_player_at_position(self, GameState state, int position) -> int`
   - Find player_id with given turn order position
   - Iterate through players checking get_turn_order against position
   - Return player_id or -1 if not found

2. `advance_to_next_bidder(self, GameState state) -> void`
   - Advance active player to next non-passed bidder in turn order
   - Get current player and their position
   - Loop through positions, skipping players who have passed auction
   - Call state._set_active_player() with found candidate

3. `set_active_player_after(self, GameState state, int player_id) -> void`
   - Set active player to next player after given player in turn order
   - Get position of player_id, compute (position + 1) % num_players
   - Find player at next position, set as active

Add cpdef declarations to turn.pxd for each new method.

Import player module for get_turn_order access (existing pattern in other entities).
  </action>
  <verify>python setup.py build_ext --inplace succeeds without errors</verify>
  <done>TurnState has find_player_at_position, advance_to_next_bidder, set_active_player_after methods</done>
</task>

<task type="auto">
  <name>Task 2: Update phase handlers to use TurnState methods</name>
  <files>phases/invest.pyx, phases/bid.pyx</files>
  <action>
In invest.pyx:
1. Remove the local `_find_player_at_position` helper function (lines 182-188)
2. Remove the local `_advance_to_next_bidder` helper function (lines 200-218)
3. Remove the local `_update_all_net_worths` helper function (lines 23-33)
4. Update `_advance_active_player` to use `turn_module.TURN.find_player_at_position(state, next_position)`
5. Update `_advance_to_next_bidder` call sites to use `turn_module.TURN.advance_to_next_bidder(state)`
6. Replace `_update_all_net_worths(state)` calls with loop calling `player_module.PLAYERS[i].update_net_worth(state)` for all players (consistent with existing pattern in _resolve_auction which updates winner only)

In bid.pyx:
1. Remove the local `_find_player_at_position` helper function (lines 20-26)
2. Remove the local `_count_active_bidders` helper function (keep, it's unique to bid.pyx)
3. Remove the local `_advance_to_next_bidder` helper function (lines 39-56)
4. Remove the local `_set_active_player_after` helper function (lines 60-65)
5. Update `_advance_to_next_bidder` call sites to use `turn_module.TURN.advance_to_next_bidder(state)`
6. Update `_set_active_player_after` call site to use `turn_module.TURN.set_active_player_after(state, starter_id)`

Keep `_count_active_bidders` in bid.pyx as it's specific to auction logic.
Keep `_resolve_auction` in bid.pyx as it's specific to BID phase.
Keep `_handle_buy_share`, `_handle_sell_share`, etc in invest.pyx as phase-specific.
  </action>
  <verify>python setup.py build_ext --inplace && pytest tests/ -v</verify>
  <done>Phase handlers use centralized TurnState methods, no duplicate helper functions</done>
</task>

</tasks>

<verification>
1. Build succeeds: `python setup.py build_ext --inplace`
2. All tests pass: `pytest tests/ -v`
3. No duplicate `_find_player_at_position` definitions: `grep -r "_find_player_at_position" phases/`
4. No duplicate `_advance_to_next_bidder` definitions: `grep -r "_advance_to_next_bidder" phases/`
</verification>

<success_criteria>
- Zero duplicate helper functions between invest.pyx and bid.pyx
- TurnState entity provides find_player_at_position, advance_to_next_bidder, set_active_player_after
- All existing tests pass
- Build completes without warnings
</success_criteria>

<output>
After completion, create `.planning/quick/001-refactor-duplicate-code-in-phases-direct/001-SUMMARY.md`
</output>
