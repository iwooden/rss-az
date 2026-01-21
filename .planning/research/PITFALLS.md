# Domain Pitfalls: INVEST/BID_IN_AUCTION Phase Implementation

**Domain:** Game action dispatch and phase logic for Cython game engine
**Project:** Rolling Stock Stars (18xx-style board game engine for AlphaZero training)
**Researched:** 2026-01-20
**Focus:** Adding game actions to existing state-based game engine

---

## Critical Pitfalls

Mistakes that cause rewrites or major issues. Addressing these is mandatory for success.

---

### Pitfall 1: Legal Move Mask Out of Sync with Actual State

**What goes wrong:**
The `get_valid_action_mask()` function returns a mask that doesn't match what actions are actually legal given the current state. The neural network selects an action marked as valid, but when applied, it produces an invalid game state.

**Why it happens:**
- Mask generation logic duplicates validity checks that exist in action application code
- Mask checks one set of conditions; action application checks different conditions
- Edge cases (e.g., player has exactly enough cash for auction) handled differently
- State is modified between mask generation and action application (stale mask)

**Consequences:**
- Invalid game states during training (silent corruption)
- Training instability as network learns from illegal move executions
- Difficult to debug because the bug only manifests in specific game states

**Prevention:**
1. **Single source of truth pattern:** Extract validity logic into shared `cdef` functions that both mask generation and action application call
2. **Design invariant:** `apply_action(state, action_idx)` must be a no-op or error if `mask[action_idx] == 0.0`
3. **Test every action type:** For each action type, generate valid mask, then verify applying each valid action produces a legal state

**Detection:**
- Write tests that iterate all valid actions and verify post-apply state is consistent
- Add assertions in action application that check mask validity (debug builds only)
- Fuzz testing: random states + random valid actions should never corrupt state

**Phase to address:** First phase implementing any action dispatch (likely "Game Driver Core")

---

### Pitfall 2: Partial State Updates (Non-Atomic Actions)

**What goes wrong:**
An action modifies some state fields but fails or exits early before completing all required updates, leaving state in an inconsistent intermediate form.

**Why it happens:**
- Share purchase action updates player cash but not corporation bank shares
- Auction action sets auction_company but forgets to set auction_price
- Error path exits function before completing all field updates
- Cascading updates (bankruptcy) have early exit bugs

**Consequences:**
- Game state invariants violated (e.g., total shares don't sum correctly)
- Downstream actions see corrupted state and make invalid decisions
- Neural network trains on impossible game positions

**Prevention:**
1. **All-or-nothing pattern:** Compute all new values FIRST, then write all fields in a single block at the end
2. **State invariant assertions:** After each action, verify key invariants (share sums, cash conservation, etc.)
3. **Transaction-like pattern:** For complex actions, collect all mutations in a list, apply atomically

**Detection:**
- Post-action invariant checks (e.g., total player cash + corp cash + bank = constant)
- Property-based testing: generate random valid action sequences, verify invariants hold throughout

**Phase to address:** Every phase implementing action handlers

---

### Pitfall 3: Cascading Bankruptcy State Corruption

**What goes wrong:**
Corporation bankruptcy triggers a chain of state changes (close companies, return shares, update player net worth, possibly trigger receivership or additional bankruptcies), and somewhere in this cascade, state becomes inconsistent.

**Why it happens:**
- Bankruptcy procedure has multiple steps that must happen in correct order
- Shares returned to players may change presidency, which has its own complex rules
- Companies closed may affect multiple players' and corps' income
- Recursive bankruptcy (corp A bankruptcy causes corp B to lose value, triggering corp B bankruptcy)
- Net worth recalculation done at wrong time in cascade

**Consequences:**
- Player net worth doesn't reflect actual holdings
- President flag doesn't match player with most shares
- Closed companies still showing as owned

**Prevention:**
1. **Explicit procedure documentation:** Write out the exact step order before coding
2. **Staged processing:** Complete each stage (close companies, then return shares, then update presidencies, then recalculate net worth) before starting the next
3. **Breadth-first, not depth-first:** Collect all bankruptcy triggers first, then process in batch
4. **Test edge cases:** Corp with no president (receivership) going bankrupt, multiple corps bankrupting same turn

**Detection:**
- Explicit bankruptcy test cases covering each variation
- Invariant: sum of all shares (player + bank + unissued) = total shares for each corp
- Invariant: exactly one president per active corp (or receivership flag set)

**Phase to address:** Bankruptcy procedure implementation phase

---

### Pitfall 4: Change of Presidency Logic Errors

**What goes wrong:**
When shares change hands (buy, sell, auction, bankruptcy share return), the presidency may need to transfer. The logic for determining the new president is complex and easy to get wrong.

**Why it happens:**
- Multiple players may tie for most shares (rules specify tie-break by turn order)
- Selling president must transfer presidency before losing shares, but only if another player has equal or more
- Player selling all shares may need to find any eligible president
- Receivership (no president) has special rules when shares are bought
- 18xx games have notoriously complex presidency rules with edge cases

**Consequences:**
- Wrong player controls corporation (makes all decisions)
- Receivership flag incorrectly set or not set
- Player thinks they can sell but can't due to presidency constraint

**Prevention:**
1. **Separate presidency check function:** `recalculate_presidency(state, corp_id)` called after any share transfer
2. **Test matrix:** Cover all combinations of before/after share counts that trigger presidency changes
3. **Document the exact rule:** "President is player with most shares; ties go to earlier turn order; if no players own shares, receivership"
4. **Avoid implicit order dependencies:** Always recalculate, never assume previous state is correct

**Detection:**
- Test: after every share transaction, verify `is_president_of()` returns True for exactly one player (or receivership)
- Test: player with most shares is always president (respecting tie-break)
- Test: selling president with buyer having equal shares triggers transfer

**Phase to address:** Share transaction phases (buy share, sell share, auction win)

---

### Pitfall 5: Auction Sub-Phase State Leakage

**What goes wrong:**
Auction runs as a sub-phase within INVEST. When auction ends, auction state fields (company, price, bidders, starter, passed flags) remain set and pollute subsequent logic or the next auction.

**Why it happens:**
- Auction winner is determined, but auction state not cleared
- Phase transitions back to INVEST without resetting auction fields
- Next auction inherits previous auction's passed players
- "Clear auction state" step forgotten in one of multiple exit paths (winner, all pass, etc.)

**Consequences:**
- Second auction sees wrong passed players
- Mask generation uses stale auction data
- Company marked as "for auction" but auction state says different company

**Prevention:**
1. **Clear on enter pattern:** Clear all sub-phase state when entering sub-phase
2. **Clear on exit pattern:** Clear all sub-phase state when exiting (redundant safety)
3. **Isolate sub-phase state:** Use dedicated struct/section for auction state, clear atomically
4. **State machine diagram:** Document all transitions in/out of auction, verify each clears state

**Detection:**
- Test: run multiple auctions in sequence, verify second auction is independent
- Invariant check: if phase != BID_IN_AUCTION, all auction state fields should be cleared/default

**Phase to address:** Auction implementation phase

---

## Moderate Pitfalls

Mistakes that cause delays or technical debt.

---

### Pitfall 6: Entity Handle Initialization Order (Known from v1)

**What goes wrong:**
Entity handles (PLAYERS, CORPS, TURN, etc.) have their offsets cached during `initialize()`. If action code calls entity methods before `initialize()` is called, the offsets are zero/stale, and state is read/written at wrong positions.

**Why it happens:**
- New action handler created that uses entity handles
- Developer assumes handles are auto-initialized or forgets to call initialize
- Refactoring moves code path before entity initialization
- Test fixture forgets to initialize handles after creating GameState

**Consequences:**
- State corruption at arbitrary offsets
- Seemingly unrelated fields modified
- Tests pass locally but fail in different order

**Prevention:**
1. **Copy v1 pattern exactly:** In `initialize_game()`, initialize ALL entity handles before any state modification
2. **Document requirement:** Entity handles MUST be initialized before use
3. **Consider lazy init:** Have entity methods check if initialized and auto-init (adds overhead)
4. **Test helper:** Create fixture that always initializes all handles

**Detection:**
- Tests that create GameState without calling entity.initialize() fail predictably
- Consider adding assertion in entity methods (debug mode) that checks initialization

**Phase to address:** All phases; especially important for any new entity types

---

### Pitfall 7: Round-Trip Limit Enforcement Bugs

**What goes wrong:**
Players have a limit on buy+sell pairs per corporation per turn (round-trip limit). The limit is not enforced correctly, allowing manipulation or incorrectly blocking valid trades.

**Why it happens:**
- Round-trip counter incremented at wrong time (before or after transaction)
- Counter not reset when player's turn starts
- Limit checked for buys but not sells (or vice versa)
- Off-by-one error in limit check (2 round-trips means 4 transactions, not 2)

**Consequences:**
- Player can manipulate share prices by unlimited buying/selling
- Legal trades incorrectly blocked, limiting valid actions
- Training learns incorrect game dynamics

**Prevention:**
1. **Clear spec:** Document exactly what "round-trip" means (buy+sell pair = 1 round-trip, 2 max means 4 total transactions with same corp)
2. **Increment on both:** Increment buy counter on buy, sell counter on sell
3. **Check formula:** Round-trips = min(buys, sells); block if round-trips >= MAX
4. **Reset location:** Clear tracking at start of INVEST phase for active player

**Detection:**
- Test: player can buy 2, sell 2 (2 round-trips)
- Test: player cannot buy 3 after already selling 2 (would be 3rd round-trip)
- Test: second player has fresh counters (not affected by first player's trades)

**Phase to address:** Buy/sell share action implementation

---

### Pitfall 8: Action Decoding Offset Drift

**What goes wrong:**
The action layout computation (`compute_action_layout()`) produces offsets that don't match what `decode_action()` expects, or what mask generation uses.

**Why it happens:**
- New action type added to layout but decoding not updated
- Player count-dependent offsets (auction has num_players * 20 slots) miscalculated
- Layout struct updated but not all code paths updated
- Constants duplicated (DEF vs cdef enum) and get out of sync

**Consequences:**
- Action 137 is "leave auction" in layout but decoded as "buy share"
- Mask marks wrong indices as valid
- NN output is misinterpreted, applying wrong actions

**Prevention:**
1. **Single source of truth:** All constants come from `actions.pxd`, never duplicate
2. **Computed layout:** Use `compute_action_layout()` everywhere, never hardcode offsets
3. **Round-trip test:** Encode action -> get index -> decode action -> verify match
4. **Boundary tests:** Test first and last action of each phase

**Detection:**
- Test: for each action type, encode and decode, verify round-trip
- Test: layout.bid_start == first BID_IN_AUCTION action index
- Test: layout.total_size == 186 + (num_players * 20) for all player counts

**Phase to address:** When adding any new actions or modifying layout

---

### Pitfall 9: Net Worth Update Timing Errors

**What goes wrong:**
Player net worth must be updated after actions that change cash, shares, or share prices. Updates happen at wrong time (before vs after), are skipped, or use stale share prices.

**Why it happens:**
- Net worth updated before share transaction completes
- Buy share updates net worth but sell share doesn't
- Share price changes during dividend phase but net worth not recalculated until next turn
- Bankruptcy returns shares at old prices

**Consequences:**
- NN sees incorrect player valuations
- Game end winner determination wrong
- Training reward signals incorrect

**Prevention:**
1. **Update after all mutations:** Call `update_player_net_worth()` at END of action, not middle
2. **Batch update helper:** `update_all_player_net_worths(state)` to update everyone at once
3. **Test net worth formula:** net_worth = cash + sum(shares * share_price) + sum(company_face_values)

**Detection:**
- After every action in tests, verify net worth matches calculated value
- Test: buy share, verify net worth changed by exactly (share_value - cash_paid)

**Phase to address:** All action phases; create helper function early

---

### Pitfall 10: Float Normalization Precision Issues

**What goes wrong:**
State is stored as normalized floats (cash / 200.0, shares / 7.0). Integer values are recovered via `int(x * DIVISOR + 0.5)`. Edge cases cause off-by-one errors due to floating-point precision.

**Why it happens:**
- Storing value 14 as 14/7.0 = 2.0, retrieving as int(2.0 * 7.0 + 0.5) = 14 (OK)
- But 1/7.0 = 0.14285714... * 7 = 0.99999999... + 0.5 = 1.49999... = 1 (OK)
- Edge cases where floating point rounds wrong direction
- Using different DIVISOR values in different places

**Consequences:**
- Player has 29 cash but reads as 30
- Corporation has 3 shares but reads as 2
- Mask says player can afford action, but actual cash is 1 less

**Prevention:**
1. **Consistent DIVISOR usage:** Import from single source (`core/data.pxd`)
2. **Use round() not truncation:** `int(round(x * DIVISOR))` instead of `int(x * DIVISOR + 0.5)`
3. **Test boundary values:** Test storing and retrieving values near normalization boundaries
4. **Consider integer storage for critical values:** At cost of NN input format

**Detection:**
- Test: store every integer from 0 to MAX, retrieve, verify exact match
- Test: run 1000 games, verify no invariant violations from precision errors

**Phase to address:** When implementing any new get/set accessors

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

---

### Pitfall 11: Test Setup Boilerplate Duplication

**What goes wrong:**
Every test file duplicates the same GameState setup code, entity initialization, and fixture creation. Changes require updates in many places.

**Prevention:**
- Create shared test fixtures in `conftest.py`
- Provide helper functions: `create_initialized_game(num_players, seed=42)`
- Use pytest fixtures for common state setups

**Phase to address:** First test implementation phase

---

### Pitfall 12: Implicit Phase Transition Assumptions

**What goes wrong:**
Code assumes specific phase sequences (e.g., after INVEST always comes BID_IN_AUCTION) without checking, breaking when phase flow is more complex.

**Prevention:**
- Explicit phase state machine with documented transitions
- Never assume "current phase + 1" is next phase
- Use named constants, not magic numbers

**Phase to address:** Phase transition implementation

---

### Pitfall 13: Missing Active Player Updates

**What goes wrong:**
After an action, the active player may need to change (next bidder in auction, next player in INVEST). Forgetting to update `_set_active_player()` causes same player to act repeatedly.

**Prevention:**
- End of action handler always considers active player update
- Use helper: `advance_to_next_player(state, skip_condition)`
- Test: after player 0 passes, active player is player 1

**Phase to address:** All action handlers

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|----------------|------------|
| Game Driver Core | Legal mask / action sync | Single source of truth for validity checks |
| INVEST Pass | Consecutive passes not tracked | Update counter, check for phase end |
| Buy Share | Round-trip limits, presidency | Increment counters, recalculate presidency |
| Sell Share | Presidency dump, round-trip | Check can't sell to dump presidency illegally |
| Start Auction | Sub-phase state init | Clear all auction state on entry |
| BID_IN_AUCTION | Exit paths leave stale state | Clear on every exit path |
| Auction Win | Company transfer, auction clear | Transfer company, clear auction, return to INVEST |
| Corporation Bankruptcy | Cascading updates | Staged procedure, breadth-first |
| Change of Presidency | Tie-breaking, receivership | Exhaustive test matrix |
| Net Worth Updates | Timing relative to mutations | Always update at end of action |

---

## Sources

- Game Programming Patterns: [State Machine Pattern](https://gameprogrammingpatterns.com/state.html)
- Cython GIL and nogil: [Cython Documentation](https://cython.readthedocs.io/en/latest/src/userguide/nogil.html)
- 18XX Rules (change of presidency): [1830 Rules Clarifications](http://www.18xx.net/1830/1830f.htm)
- AlphaZero legal move masking: [PettingZoo Chess](https://pettingzoo.farama.org/environments/classic/chess/)
- State Machine Tips: [State Machines for Game Dev](https://www.numberanalytics.com/blog/state-machines-for-game-dev-success)
- Project v1 patterns: `/home/icebreaker/rss-az-cython2/.planning/PROJECT.md` (Entity initialization order, module import pattern)
- Existing codebase: `/home/icebreaker/rss-az-cython2/core/actions.pyx`, `/home/icebreaker/rss-az-cython2/core/state.pyx`

---

*Research confidence: HIGH for codebase-specific pitfalls (verified against existing code), MEDIUM for domain pitfalls (based on 18xx rule complexity and general game engine patterns)*
