"""Shared dense policy layout for model, search, and training boundaries."""

from __future__ import annotations

import torch

from core.data import (
    MAX_ACTION_SIZE,
    PHASE_ACTION_SIZES,
    DecisionPhase,
    GameConstants,
)


NUM_PHASES = int(GameConstants.NUM_DECISION_PHASES)

_phase_offsets = [0]
for _size in PHASE_ACTION_SIZES:
    _phase_offsets.append(_phase_offsets[-1] + int(_size))
PHASE_OFFSETS = tuple(_phase_offsets)
UNIFIED_LOGIT_DIM = PHASE_OFFSETS[-1]

# Decision phases whose phase-local slot 0 is a pass/no-op followed by one or
# more non-pass actions. The trainer uses these for logit-scale diagnostics.
PHASES_WITH_PASS_SLOT = (
    int(DecisionPhase.DPHASE_INVEST),
    int(DecisionPhase.DPHASE_BID),
    int(DecisionPhase.DPHASE_ACQ_SELECT_CORP),
    int(DecisionPhase.DPHASE_CLOSING),
    int(DecisionPhase.DPHASE_IPO),
)


def build_action_lut() -> torch.Tensor:
    """Map phase-local action ids to slots in the dense policy tensor.

    The returned tensor has shape ``(NUM_PHASES, MAX_ACTION_SIZE)``. Entries
    beyond a phase's action count are zero sentinels and must never be marked
    legal.
    """
    lut = torch.zeros(NUM_PHASES, int(MAX_ACTION_SIZE), dtype=torch.long)
    for phase in range(NUM_PHASES):
        size = int(PHASE_ACTION_SIZES[phase])
        lut[phase, :size] = PHASE_OFFSETS[phase] + torch.arange(size)
    return lut
