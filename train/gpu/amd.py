"""AMD ROCm GPU optimization utilities.

When running on AMD GPUs (detected via torch.version.hip), these
optimizations are applied:

1. **torch.compile default mode** (Inductor fusion, no HIP graphs):
   Removes most per-op Python dispatch overhead by fusing and codegen'ing
   Triton kernels. HIP-graph mode ("reduce-overhead") is NOT used because
   ``RSSTransformerNet.forward`` runs over a ``mark_unbacked`` batch dim
   that varies every call with MCTS workload pressure — HIP/CUDA graphs
   would require re-capture per batch size. Default mode supports
   dynamic shapes.

Note: TF32 does not exist on AMD hardware — the torch TF32 flags are
accepted but have zero effect.  TunableOp is intentionally NOT enabled:
it is irrelevant when torch.compile is active (Triton bypasses ATen
GEMMs) and conflicts with HIP graph capture.
"""

from __future__ import annotations

import os
from typing import Any

import torch


def _setdefault_env(key: str, value: str) -> str:
    """Set an environment variable if not already set. Returns the active value."""
    return os.environ.setdefault(key, value)


def apply_amd_optimizations() -> dict[str, str]:
    """Apply AMD-specific PyTorch settings and environment variables.

    Environment variables are set via ``os.environ.setdefault`` so that
    user overrides from the shell are respected.  Must be called early
    in the main process — values propagate to spawned child processes.

    Returns a dict describing what was enabled (for logging).
    """
    enabled: dict[str, str] = {}
    enabled["gpu"] = torch.cuda.get_device_name()
    hip_version = getattr(torch.version, "hip", None)
    if hip_version is not None:
        enabled["rocm"] = hip_version

    # Limit hardware compute queues to reduce GPU scheduling contention
    # when many processes (eval servers + workers) submit work concurrently.
    # Default is 4; 2 reduces queue overhead on RDNA 4.
    enabled["GPU_MAX_HW_QUEUES"] = _setdefault_env("GPU_MAX_HW_QUEUES", "2")

    # Disable the SDMA (System DMA) copy engine, forcing copies through
    # compute shaders instead.  Slightly lower bandwidth but lower latency
    # for the small H2D/D2H transfers in our pipeline (~1101 floats).
    enabled["HSA_ENABLE_SDMA"] = _setdefault_env("HSA_ENABLE_SDMA", "0")

    # Inductor max-autotune: when enabled (1), benchmarks multiple Triton
    # kernel configs per GEMM shape at first compile and caches the fastest.
    # Adds significant one-time compilation time.  Off (0) by default;
    # set to 1 in the shell to opt in.
    enabled["TORCHINDUCTOR_MAX_AUTOTUNE"] = _setdefault_env(
        "TORCHINDUCTOR_MAX_AUTOTUNE", "0",
    )

    return enabled


def get_compile_kwargs(*, for_training: bool = False) -> dict[str, Any]:
    """Return torch.compile kwargs optimized for AMD GPUs.

    Uses default (automatic-dynamic) mode. ``reduce-overhead`` / HIP
    graphs were dropped because eval_server / single-process warmup
    applies ``mark_unbacked`` to the batch dim (so a single compiled
    artifact handles every batch size 1..max_batch) — HIP graphs would
    require re-capture per batch size, defeating the point. Global
    ``dynamic=True`` is explicitly discouraged by the PyTorch docs
    (forces every dim and module parameter dynamic, error-prone, can
    cause perf regressions); ``mark_unbacked`` on just the batch dim
    is the targeted alternative.

    ``shape_padding`` is disabled. Inductor's ``pad_mm`` pass reads
    concrete ``mm`` sizes via ``is_mm_compute_bound`` to decide whether
    to zero-pad operands for better Triton tiling — this bakes batch-
    size range guards (e.g. ``3 <= x.size(0) <= 4``) into the compiled
    artifact, forcing a recompile every time the batch size crosses a
    compute-vs-memory-bound threshold. Our batch sizes vary continuously
    with MCTS workload pressure, so the pass blows the recompile limit
    on its own. ``mark_unbacked`` doesn't reach this pass (it runs
    inside Inductor, after Dynamo).

    Args:
        for_training: If True, disables Inductor's
            ``joint_graph_constant_folding`` pass. That pass calls
            ``fake_tensor.is_contiguous`` on uniform-valued nodes
            (e.g. backward-pass ``zeros_like`` tensors) and guards on
            ``Eq(K * u, 0)`` to test emptiness. The batch dim carries a
            ``mark_unbacked`` symbol, so this guard is undecidable and
            compilation crashes. The pass is a training-only concern —
            inference skips the joint-graph backward partition entirely,
            which is why eval_server is unaffected. Disabling costs only
            a small forward-graph constant-folding optimization.
    """
    options: dict[str, Any] = {
        "shape_padding": False
    }
    if for_training:
        options["joint_graph_constant_folding"] = False
    return {"options": options}
