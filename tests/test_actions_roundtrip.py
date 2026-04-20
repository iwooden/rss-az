"""Encode/decode round-trip tests for the per-phase action space.

Each ``DecisionPhase`` has a phase-local integer action id. The ``encode_*``
helpers in ``core/actions.pxd`` and the ``decode_action`` dispatcher in
``core/actions.pyx`` must be exact inverses across ``[0, ACTION_SIZE[phase])``.

A module-import self-check in ``core/actions.pyx`` pins one boundary id per
phase (usually ``size - 1``). That is not enough to catch an off-by-one in
``decode_action``'s interior branches — this test walks the entire legal id
range for every phase and asserts ``encode(decode(id)) == id``.
"""
import pytest

from core.actions import decode_action_py, encode_action_py
from core.data import DecisionPhase, PHASE_ACTION_SIZES


ALL_PHASES = [p for p in DecisionPhase]


@pytest.mark.parametrize("phase", ALL_PHASES, ids=[p.name for p in ALL_PHASES])
def test_encode_decode_roundtrip(phase):
    """Every legal ``(phase, action_id)`` survives decode → encode unchanged."""
    phase_id = int(phase)
    size = PHASE_ACTION_SIZES[phase_id]
    assert size > 0, f"{phase.name}: ACTION_SIZE must be positive, got {size}"

    for action_id in range(size):
        info = decode_action_py(phase_id, action_id)
        assert info.phase == phase_id, (
            f"{phase.name} id={action_id}: decoded phase={info.phase}, "
            f"expected {phase_id}"
        )
        reencoded = encode_action_py(info)
        assert reencoded == action_id, (
            f"{phase.name} id={action_id}: decode→encode produced {reencoded} "
            f"(info={info})"
        )
