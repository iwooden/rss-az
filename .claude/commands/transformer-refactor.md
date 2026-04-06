Pick up work on the transformer refactor.

## Setup

Read `transformers.md` end-to-end before doing anything else. It is the design doc for this refactor — token decomposition, model architecture, action layout, phased rollout plan. You need the full picture in context, not a partial view.

## Important context — read before touching any code

This is a **massive, breaking refactor** of nearly every layer of the engine: state representation, action space, entity handles, model, evaluator, training loop. We are deliberately working through it slice-by-slice.

- **Build breaks and test failures are expected** in the middle of a slice. Modules outside the current slice will not compile or import until they are updated in their own slice. Do not chase those errors. Do not "helpfully" patch unrelated callsites to make the build go green.
- **Touch only the files explicitly named** in the user's instructions for this slice. If the user says "modify `core/data.*`", that is the entire scope — do not also edit `entities/*` or `phases/*` because they call removed functions.
- **Assume callsites will break.** You do not need to grep for usages of a function or field you are removing. The user already knows what they're asking for breaks downstream — fixing that is a later slice's job. The exception is anything used by `core/state.*`, which has already been refactored and must keep working.
- **No cleanup of tangential issues.** No drive-by refactors, no "while I'm here" fixes, no adding type hints or docstrings to code you didn't change. Stay narrowly inside the requested scope.
- **Do not commit anything unless explicitly asked.** Each slice may want manual review before committing.
- **Verification is per-slice.** A full `pytest tests/` and full `setup.py build_ext` will fail mid-refactor. Verify the slice in isolation when possible (e.g. `cython -3 <file>.pyx` to confirm the rewritten module compiles cleanly on its own). Do not run the full quality gates and panic at the broken state — that's the expected state.

Phase 1 of the refactor (compact state in `core/state.{pxd,pyx}`) is already done in commit `09e5048`. Subsequent slices update the rest of the engine — entities, data, actions, phases — to match the new state layout and the new action space.

## This slice

$ARGUMENTS
