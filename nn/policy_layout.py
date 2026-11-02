"""Shared dense policy layout for model, search, and training boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch


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
# They resolve from games.rss on first use: games.rss imports this module.
_LEGACY_RSS_NAMES = {
    "NUM_PHASES": "num_phases",
    "PHASE_OFFSETS": "offsets",
    "UNIFIED_LOGIT_DIM": "unified_dim",
    "PHASES_WITH_PASS_SLOT": "phases_with_pass_slot",
    "build_action_lut": "build_action_lut",
}

if TYPE_CHECKING:
    NUM_PHASES: int
    PHASE_OFFSETS: tuple[int, ...]
    UNIFIED_LOGIT_DIM: int
    PHASES_WITH_PASS_SLOT: tuple[int, ...]

    def build_action_lut() -> torch.Tensor: ...
else:
    def __getattr__(name: str) -> Any:
        attr = _LEGACY_RSS_NAMES.get(name)
        if attr is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        from games.rss import POLICY_LAYOUT

        value = globals()[name] = getattr(POLICY_LAYOUT, attr)
        return value
