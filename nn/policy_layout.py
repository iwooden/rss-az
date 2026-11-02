"""Shared dense policy layout for model, search, and training boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from core.data import (
    MAX_ACTION_SIZE,
    PHASE_ACTION_SIZES,
    DecisionPhase,
    GameConstants,
)


@dataclass(frozen=True)
class PolicyLayout:
    """A game's decision phases laid end to end in one dense policy vector.

    Phase ``p``'s phase-local action ``a`` occupies unified slot
    ``offsets[p] + a``; ``offsets`` ends with ``unified_dim``.
    ``phases_with_pass_slot`` lists phases whose slot 0 is a pass/no-op
    followed by one or more non-pass actions; the trainer uses them for
    logit-scale diagnostics. ``phase_names`` label per-phase metrics.
    """

    phase_sizes: tuple[int, ...]
    phase_names: tuple[str, ...]
    phases_with_pass_slot: tuple[int, ...]
    offsets: tuple[int, ...] = field(init=False)
    unified_dim: int = field(init=False)
    max_action_size: int = field(init=False)

    def __post_init__(self) -> None:
        if len(self.phase_names) != len(self.phase_sizes):
            raise ValueError(
                f"{len(self.phase_names)} phase names for "
                f"{len(self.phase_sizes)} phases"
            )
        offsets = [0]
        for size in self.phase_sizes:
            offsets.append(offsets[-1] + size)
        object.__setattr__(self, "offsets", tuple(offsets))
        object.__setattr__(self, "unified_dim", offsets[-1])
        object.__setattr__(self, "max_action_size", max(self.phase_sizes))

    @property
    def num_phases(self) -> int:
        return len(self.phase_sizes)

    def build_action_lut(self) -> torch.Tensor:
        """Map phase-local action ids to slots in the dense policy tensor.

        The returned tensor has shape ``(num_phases, max_action_size)``.
        Entries beyond a phase's action count are zero sentinels and must
        never be marked legal.
        """
        lut = torch.zeros(self.num_phases, self.max_action_size, dtype=torch.long)
        for phase, size in enumerate(self.phase_sizes):
            lut[phase, :size] = self.offsets[phase] + torch.arange(size)
        return lut


# RSS's layout under its legacy names, for the v2 and v3 model files, which stay
# untouched (1889/pipeline-refactor.md §5). The pipeline uses ``Game.policy``.
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
