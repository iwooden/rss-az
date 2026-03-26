"""AMD ROCm GPU optimization utilities.

When running on AMD GPUs (detected via torch.version.hip), these
optimizations are applied:

1. **torch.compile mode="reduce-overhead"** (HIP graphs): Eliminates kernel
   launch overhead by capturing the entire forward pass as a HIP graph.
   Same mechanism as CUDA graphs on NVIDIA, ~2.3x inference throughput
   improvement on RDNA 4 (gfx1201).

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

    Uses mode='reduce-overhead' which enables HIP graph capture via Inductor,
    eliminating kernel launch overhead.

    Args:
        for_training: If True, returns kwargs for training (fixed batch size).
                     If False, returns kwargs for eval servers (variable batch sizes).
    """
    if for_training:
        return {"mode": "reduce-overhead"}
    else:
        return {"mode": "reduce-overhead", "dynamic": True}
