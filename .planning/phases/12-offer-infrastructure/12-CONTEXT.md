# Phase 12: Offer Infrastructure - Context

**Gathered:** 2026-01-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate valid acquisition offers in sorted priority order and manage acquisition state. This phase builds the infrastructure for offer discovery, sorting, and presentation - actions and validation are handled in Phase 13.

</domain>

<decisions>
## Implementation Decisions

### Tie-breaking rules
- No tie-breaking needed for share price - share price cards are unique (except 0=bankrupt, 75=game ends)
- Face values are unique per company, so sorting is fully deterministic
- OS retains FI priority even if in receivership
- Corp-to-corp: generate offers in both directions (A→B and B→A) if same president controls both
- Player privates: all corps controlled by player can bid on each private company

### Iteration model
- Pre-compute sorted offer list at phase entry
- Fixed-size offer buffer (~250 slots, half theoretical max)
- Each slot stores: (corp_id, company_id) - derive is_fi/seller dynamically
- Current index tracks position in buffer, increments on accept/pass
- Re-validate offers before presenting (company not yet acquired, corp has cash, etc.)
- Assert/error if buffer limit exceeded (indicates bad assumptions)

### Active player tracking
- Active player = president of buying corp for current offer
- Update active_player when advancing to next offer
- For receivership corps: Claude's discretion on handling (keep previous or special value)
- Note: active_player is hidden state, internal bookkeeping only

### Phase entry setup
- WRAP_UP calls into ACQUISITION logic to set up first offer before transitioning
- Offer buffer populated at phase entry (not lazily)
- Always enter ACQUISITION phase even if no offers (detect empty, immediately transition to CLOSING)
- Clear acquisition zones at phase EXIT (after merge), not entry

### State additions needed
- Add `acquisition_proceeds` field for players (tracks cash from selling private companies)
- This requires changes to: state.pyx, VECTORS.md, entities/player.pyx
- Existing tests using hardcoded offsets will need updates

### Claude's Discretion
- Exact buffer size (suggested ~250, can adjust based on analysis)
- How to handle active_player during receivership auto-buys
- Internal implementation details of offer sorting algorithm

</decisions>

<specifics>
## Specific Ideas

- Offer buffer stores just (corp_id, company_id) tuples - minimal storage, derive other info dynamically
- Sorting order is deterministic: OS→FI, then corps by descending share price, then corp-to-corp by (buyer price, target face value), then player privates with same key
- Model never sees active_player (hidden state), sees offer details via acq_active_corp/acq_target_company/acq_is_fi_offer

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope

</deferred>

---

*Phase: 12-offer-infrastructure*
*Context gathered: 2026-01-24*
