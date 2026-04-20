# Phase Refactor — Historical Design Notes

> Status (2026-04): the structural phase split has landed.
> - INVEST still exists as `PHASE_INVEST` / `DPHASE_INVEST`; only the opening-bid price choice moved out of INVEST and into BID.
> - IPO is now `IPO -> PAR`.
> - ACQUISITION is now `ACQ_SELECT_CORP -> ACQ_SELECT_COMPANY -> ACQ_SELECT_PRICE`, with `ACQ_OFFER` still handling contested / FI-priority resolution.
> - Cross-phase policy-head sharing (`corp_select_head`, `company_select_head`, `offset_select_head`) did not ship.
>
> This document survives as a reference for the motivation, the intended shape of the refactor, and the places where the final implementation deliberately diverged from the original sketch. It is not an active work queue.

## Current shipped state

Authoritative code wins over this doc. Start here:

- `core/data.{pxd,pyx}` — engine/decision phase ids, action-space sizes, `PHASE_ACTION_SIZES`
- `core/actions.{pxd,pyx}` — legality enumeration and action encoding
- `phases/invest.pyx`, `phases/bid.pyx` — INVEST/BID split as actually shipped
- `phases/ipo.pyx`, `phases/par.pyx` — IPO/PAR flow
- `phases/acq_select_corp.pyx`, `phases/acq_select_company.pyx`, `phases/acq_select_price.pyx`, `phases/acq_offer.pyx` — ACQ flow
- `nn/transformer.py` — current policy heads and per-phase dispatch
- `tests/phases/test_bid.py`, `tests/phases/test_ipo.py`, `tests/phases/test_par.py`, `tests/phases/test_acq_select_{corp,company,price}.py`

### Landed pieces

| Area | Current state | Code |
|------|---------------|------|
| INVEST/BID | Landed. INVEST keeps company selection and share trading; BID now includes the opening bid. There is no `INVEST_SELECT` rename in code. | `phases/invest.pyx`, `phases/bid.pyx` |
| IPO/PAR | Landed. IPO selects the corp; PAR selects the par price. | `phases/ipo.pyx`, `phases/par.pyx` |
| ACQ split | Landed. ACQ now walks corp -> company -> price, then optionally into `ACQ_OFFER`. | `phases/acq_select_*.pyx`, `phases/acq_offer.pyx` |
| Action sizes | Landed. `PHASE_ACTION_SIZES = [53, 16, 9, 2, 37, 26, 2, 9, 14, 36, 52]`; `MAX_ACTION_SIZE = 53`. | `core/data.pyx`, `core/data.pxd` |
| Active-flag reuse | Landed. Existing `active_corp` / `active_company` turn-block slots carry the split-phase context; no new state fields were added. | `entities/turn.pyx`, phase handlers, `core/token_data.pyx` |
| Cross-phase head sharing | Deferred. Current model uses per-sub-phase heads rather than unified shared heads. | `nn/transformer.py` |

## Actual decision phases and action widths

The final code kept the existing INVEST / IPO names rather than renaming them to `INVEST_SELECT` / `IPO_SELECT`.

`DecisionPhase` in `core/data.pxd` is:

1. `DPHASE_INVEST` — 53
2. `DPHASE_BID` — 16
3. `DPHASE_ACQ_SELECT_CORP` — 9
4. `DPHASE_ACQ_OFFER` — 2
5. `DPHASE_CLOSING` — 37
6. `DPHASE_DIVIDENDS` — 26
7. `DPHASE_ISSUE` — 2
8. `DPHASE_IPO` — 9
9. `DPHASE_PAR` — 14
10. `DPHASE_ACQ_SELECT_COMPANY` — 36
11. `DPHASE_ACQ_SELECT_PRICE` — 52

That slightly awkward ordering is real: the two later ACQ sub-phases were appended rather than fully renumbering everything around them. Any prose that presents the phases in a more logical narrative order should not be mistaken for the enum order the model and trainer consume.

## Action-space summary: original sketch vs shipped shape

| Old phase | Original sketch in this doc | Final shipped shape |
|-----------|-----------------------------|---------------------|
| INVEST | `INVEST_SELECT` with pass + company select + corp trade | Still `INVEST`. Same effective split idea, but no rename. Action space is `53 = 1 pass + 36 auction-company selects + 8*2 corp trades`. |
| BID | `BID` extended to absorb opening bid | Shipped as described. Action space is `16 = 1 pass/leave + 15 raise offsets`; pass is illegal on the opening bid because `auction_high_bidder == -1`. |
| ACQUISITION | `ACQ_SELECT_CORP -> ACQ_SELECT_COMPANY -> ACQ_SELECT_PRICE` | Shipped as described. Action spaces are `9`, `36`, `52`. |
| ACQ_OFFER | Unchanged | Unchanged in role and width (`2`). |
| IPO | `IPO_SELECT -> PAR` | Same structural split, but the phase kept the name `IPO` rather than `IPO_SELECT`. Widths are `9` and `14`. |
| Other phases | Unchanged | Unchanged (`CLOSING=37`, `DIVIDENDS=26`, `ISSUE=2`). |

## Active-flag lifecycle as shipped

The original version of this doc understated how much `active_company` still matters outside ACQ. Current behavior is:

| Phase / transition | `active_corp` | `active_company` | Notes |
|--------------------|---------------|------------------|-------|
| ACQ setup / `PHASE_ACQ_SELECT_CORP` entry | `-1` | `-1` | `setup_acquisition_phase()` clears acquisition context. |
| `ACQ_SELECT_CORP -> ACQ_SELECT_COMPANY` | selected corp | `-1` | `apply_acq_select_corp_action()` stamps the corp and advances. |
| `ACQ_SELECT_COMPANY -> ACQ_SELECT_PRICE` | selected corp | selected company | `apply_acq_select_company_action()` stamps the company and advances. |
| Direct ACQ resolution | cleared | cleared | `apply_acq_select_price_action()` clears both and returns to `PHASE_ACQ_SELECT_CORP` on the same player. |
| ACQ offer / resume path | managed by ACQ helpers | managed by ACQ helpers | `phases.util.acq_common` owns the contested/FI-priority resume details. |
| IPO setup / `PHASE_IPO` entry | `-1` | current player-owned company | `_advance_to_next_company()` seeds the company being processed. |
| `IPO -> PAR` | selected corp | same company | `apply_ipo_action()` stamps `active_corp` and switches to `PHASE_PAR`. |
| PAR resolution | cleared | next IPO company or cleared on IPO exit | `apply_par_action()` clears `active_corp`, then `_advance_to_next_company()` either seeds the next company or `_transition_out_of_ipo()` clears `active_company` and returns to INVEST. |
| BID | `-1` | auction target company | INVEST seeds `active_company`; BID uses it as the auction target. |

## NN architecture: what actually shipped

The structural split shipped without the parameter-sharing experiment.

Current `nn/transformer.py` uses:

- `company_auction_head` for INVEST company selection
- `corp_trade_head` for INVEST corp buy/sell
- `auction_raise_head` for BID offsets
- `corp_acq_head` for `ACQ_SELECT_CORP`
- `company_acq_head` for `ACQ_SELECT_COMPANY`
- `price_acq_head` for `ACQ_SELECT_PRICE`
- `corp_ipo_head` for IPO corp selection
- `par_price_head` for PAR price selection
- existing single-purpose heads for pass / dividends / issue / acq_offer / value

Notable shipped decision:

- `ACQ_SELECT_PRICE` reads a dedicated `acq_price_info` token rather than any unified offset-token scheme. The engine populates that token from `active_corp`, `active_company`, and affordability context during `PHASE_ACQ_SELECT_PRICE`.

The original shared-head idea remains a follow-up experiment, not part of the landed refactor.

## What changed relative to the original design sketch

1. Naming stayed conservative.
   - The code did not introduce `INVEST_SELECT` or `IPO_SELECT` enum names.
   - The structural split happened through action semantics and phase transitions, not a sweeping rename pass.

2. The split landed before head sharing.
   - This is the biggest intentional divergence.
   - The final implementation isolates the rules/action-space refactor from the learning-dynamics experiment.

3. IPO context kept `active_company` live.
   - The shipped IPO/PAR flow processes one player-owned company at a time.
   - `active_company` therefore remains central during IPO, while `active_corp` is only the temporary PAR choice.

4. Enum order is implementation-driven, not narrative-driven.
   - The model/trainer-facing order is the literal `DecisionPhase` enum order from `core/data.pxd`.
   - Documentation that shows ACQ sub-phases adjacent for readability must not be treated as an ABI.

## Original motivation (preserved)

Three pressures motivated the refactor:

1. ACQ's old joint `(corp, company, price)` policy was harder to learn than the underlying decision structure warranted.
2. Several phases were learning similar entity-selection concepts in separate heads with no gradient sharing.
3. INVEST's old “open an auction at price 0” special case was representationally redundant with “place the opening bid in BID”.

The final code addressed (1) and (3) directly via structural phase splitting. It intentionally postponed the parameter-sharing part of (2).

## Historical pre-landing estimates

The older version of this document included rough pre-landing estimates for game length / throughput impact and for the hoped-for gradient-amortization benefits of shared heads.

Those numbers were design-time guesses, not post-landing measurements. No measured replacement values are recorded here, so treat them as historical rationale only, not current performance claims.

## Deferred follow-up

If the cross-phase head-sharing experiment is revived, start from the live code rather than from the original sketch in this document. In particular, reevaluate:

- whether `company_auction_head`, `company_close_head`, and `company_acq_head` should really unify,
- whether BID offsets and ACQ price selection share enough semantics to justify a common head,
- whether the extra coupling is worth the lost isolation in debugging and regression analysis.

The honest comparison is a fresh implementation on top of the shipped split phases, followed by measured training curves — not an appeal to the original design intent.

## References

- `core/data.pxd`, `core/data.pyx`
- `core/actions.pxd`, `core/actions.pyx`
- `phases/invest.pyx`, `phases/bid.pyx`
- `phases/ipo.pyx`, `phases/par.pyx`
- `phases/acq_select_corp.pyx`, `phases/acq_select_company.pyx`, `phases/acq_select_price.pyx`, `phases/acq_offer.pyx`
- `nn/transformer.py`
- `transformers.md`
- `VECTORS.md`
- `token-data.md`
- `sparse-refactor.md` — precedent for keeping a landed refactor doc as historical context rather than a live task list
