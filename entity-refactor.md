# Entity Hot-Path Refactor Guide

This guide records the intended pattern for refactoring entity handles so phase
handlers and action enumeration can stay readable without paying Python dispatch
cost in hot loops.

The goal is not to remove entity handles. The goal is to give each entity module
two clean APIs:

1. A public Python-facing handle API for tests, scripts, setup code, and slow
   paths.
2. A Cython-facing primitive API for hot phase/action code.

## Core Principle

Entity modules own storage. Phase modules own rules. `core/actions.pyx` owns
encoding and sparse legal-action enumeration.

Do not move phase-specific rule logic into generic entities just because the
rule needs fast state reads. Instead:

- Put common storage/domain reads on the entity as `cdef` primitives.
- Put phase-specific legality and flow helpers in the phase module.
- Let `actions.pyx` call phase legality helpers when enumeration needs phase
  rules.

Example:

- Good entity primitive: `count_corp_companies(state, corp_id, include_acquisition)`
- Good phase helper: `_corp_closable_by_player(state, corp_id, player_id)`
- Bad entity helper: `corp_can_close_company_in_closing_phase(...)`
- Bad action-module helper: `_count_corp_owned_companies(...)`

## API Layers

Each entity should generally have three layers.

### 1. Private Storage Helpers

Private helpers live only in the `.pyx`. They are allowed to look like layout
implementation details.

Example:

```cython
cdef inline int _location_at(GameState state, int company_id) noexcept nogil:
    return <int>state._data[
        LAYOUT.companies_offset + COMPANY_OFFSETS.locations + company_id
    ]
```

These helpers should usually not be declared in the `.pxd`. They are the
implementation behind the entity, not the public Cython API.

### 2. Exported Cython Primitives

Exported primitives are declared in the entity `.pxd` and implemented in the
`.pyx`. They are the API other Cython modules should use in hot loops.

Example:

```cython
# entities/company.pxd
cdef int company_location(GameState state, int company_id) noexcept nogil
cdef int company_owner_id(GameState state, int company_id) noexcept nogil
cdef bint company_owned_by_corp(GameState state, int company_id, int corp_id) noexcept nogil
```

```cython
# entities/company.pyx
cdef int company_location(GameState state, int company_id) noexcept nogil:
    return _location_at(state, company_id)
```

This looks like duplication, but the layers serve different purposes:

- `_location_at` is storage-shaped and private.
- `company_location` is domain-shaped and exported.

The exported primitive lets callers depend on the company entity contract
without depending on current storage details.

### 3. Public Handle Methods

Existing `cpdef` handle methods should remain for Python callers and readable
slow-path code.

Prefer making them delegate to the exported primitive:

```cython
cpdef int get_location(self, GameState state):
    return company_location(state, self.company_id)
```

This keeps one implementation path and makes behavior consistent between Python
and Cython callers.

## Naming Conventions

Use names that describe domain concepts, not storage mechanics.

Prefer:

```cython
company_location(...)
company_owner_id(...)
company_owned_by_player(...)
corp_is_active(...)
player_cash(...)
turn_active_player(...)
```

Avoid exporting names like:

```cython
_location_at(...)
_slot(...)
_field_at(...)
_raw_company_location(...)
```

Private helpers may use low-level names. Exported primitives should read like
entity operations.

## When To Use Primitives

Use exported Cython primitives in:

- `core/actions.pyx` legal-action enumeration.
- Phase helpers that scan all corps/companies/players.
- Automated phase setup/cleanup loops.
- Any helper that is `noexcept nogil` or should become `noexcept nogil`.
- Repeated scalar reads inside MCTS/training hot paths.

Keep public handle methods in:

- Python tests and debug scripts.
- Rare setup or one-off code where readability is more valuable.
- Mutating operations whose side effects are centralized on the handle.

## Mutations Need Extra Care

Reads are usually safe to expose as `cdef noexcept nogil` primitives.
Mutations are different.

Many handle methods perform cache invalidation, cascading recalculation, or
assertion checks. Do not bypass those casually.

Example from `entities/company`:

- Fast reads use primitives like `company_location(...)`.
- Transfers still use handle methods like:
  - `COMPANIES[company_id].transfer_to_corp(...)`
  - `COMPANIES[company_id].transfer_to_fi(...)`
  - `COMPANIES[company_id].remove_from_game(...)`

Those methods own downstream invalidation. Replacing them with raw location
writes would be a correctness risk unless the new primitive also owns the full
semantic operation.

If a hot mutation needs a primitive, expose a semantic mutation primitive, not a
raw field setter.

Prefer:

```cython
transfer_company_to_corp(state, company_id, corp_id)
```

Avoid:

```cython
set_company_location_owner(state, company_id, LOC_CORP, corp_id)
```

unless it remains private inside `entities/company.pyx`.

## Cheap Checks Before Expensive Scans

Order legality checks by cost.

Do cheap scalar checks before O(N) scans:

```cython
cdef bint _corp_closable_by_player(GameState state, int corp_id, int player_id) noexcept nogil:
    if not corp_is_active(state, corp_id):
        return False
    if corp_is_in_receivership(state, corp_id):
        return False
    if corp_president_id(state, corp_id) != player_id:
        return False

    return count_corp_companies(state, corp_id, False) > 1
```

Avoid scanning 36 companies before discovering the corp is inactive, in
receivership, or not controlled by the active player.

## Where Rule Logic Belongs

Use this split:

```text
entities/*.pyx
  storage ownership
  common domain reads/writes
  cache invalidation for entity-owned mutations

phases/<phase>.pyx
  phase rules
  phase-specific legality helpers
  automated phase flow
  action application semantics

core/actions.pyx
  action id encoding/decoding
  sparse legal action enumeration
  calls entity primitives and phase legality helpers
```

Examples:

- `count_corp_companies` belongs in `entities/corp`.
- `_corp_closable_by_player` belongs in `phases/closing`.
- `_find_first_preemptor` belongs in `phases/acquisition`.
- `actions.pyx` should not independently reimplement these rules.

## Refactor Procedure

For each entity:

1. Inventory public handle methods.
2. Identify read methods used in phase/action hot loops.
3. Check whether equivalent `cdef noexcept nogil` primitives already exist.
4. Add missing primitives to the entity `.pxd`.
5. Implement primitives in the entity `.pyx`, delegating to private storage
   helpers where appropriate.
6. Update existing `cpdef` handle methods to delegate to the new primitives.
7. Replace hot-loop handle calls in phases/actions with cimported primitives.
8. Keep semantic mutations on handle methods unless you deliberately add a
   semantic mutation primitive with all invalidation/cascades preserved.
9. Rebuild with a clean Cython build if `.pxd` files changed.
10. Run targeted sanity checks for affected phases.

## What To Search For

Useful searches:

```bash
rg -n "ENTITY_MODULE\\.HANDLES\\[[^\\]]+\\]\\.(get_|is_|has_|count_)" phases core entities
rg -n "COMPANY_OFFSETS|companies_offset" core/actions.pyx phases entities
rg -n "PLAYER_FIELDS|players_offset" phases entities
rg -n "CORP_FIELDS|corps_offset" phases entities
```

Adjust the entity/module names as needed. The goal is not to remove all direct
layout access everywhere. The goal is to make direct layout access live in the
owning entity module unless there is a clear, documented hot-path reason.

## Cython Details

Prefer these signatures for hot read primitives:

```cython
cdef int thing_value(GameState state, int thing_id) noexcept nogil
cdef bint thing_predicate(GameState state, int thing_id) noexcept nogil
```

Use `bint` for boolean predicates.

Use `noexcept nogil` when the function cannot raise and does not need Python.

Consider `inline` for tiny primitives if profiling shows call overhead matters
or if the function sits in an extremely hot inner loop:

```cython
cdef inline int company_location(GameState state, int company_id) noexcept nogil
```

Only do this if Cython accepts the declaration/implementation cleanly and the
generated dependency behavior remains sane.

## Verification

After changing a `.pxd`, run a clean build:

```bash
.venv/bin/python setup.py clean
.venv/bin/python setup.py build_ext --inplace
```

Then run targeted checks for the affected phase. At minimum, import the touched
modules and exercise legal-action enumeration for a state that reaches the
affected phase.

For larger refactors, run:

```bash
pytest tests/
```

## Current Company Refactor Pattern

`entities/company.*` now demonstrates the intended shape:

- Private storage helpers:
  - `_location_at`
  - `_owner_at`
  - `_adjusted_income_at`
- Exported primitives:
  - `company_location`
  - `company_owner_id`
  - `company_owned_by_player`
  - `company_owned_by_fi`
  - `company_owned_by_corp`
  - `company_adjusted_income`
  - `company_face_value`
  - etc.
- Public handle methods:
  - `Company.get_location`
  - `Company.is_owned_by_corp`
  - `Company.get_face_value`
  - etc.

Phase/action code should use the exported primitives for read-heavy paths and
the handle methods for semantic mutations.
