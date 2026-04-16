"""NVIDIA GPU optimization utilities.

When running on NVIDIA hardware, these optimizations are applied:

1. **TF32 matmul precision** (Ampere+): ~2-3x speedup for float32 matmuls
   with minimal precision loss (19-bit mantissa vs 23-bit). Affects the float32
   backward pass and any non-autocast matmuls.

2. **torch.compile default mode** (Inductor fusion, no CUDA graphs):
   CUDA-graph capture (mode="reduce-overhead") is NOT used because
   ``nn/transformer.py::_policy_forward`` uses ``index_select`` with
   data-dependent per-phase row counts — CUDA graphs require static
   shapes. Default-mode Inductor still removes most Python dispatch
   overhead via Triton fusion.

3. **non_blocking H2D transfers**: Allows CPU work to overlap with GPU DMA.
   Especially beneficial on GH200 with NVLink-C2C (900 GB/s).
"""

from __future__ import annotations

from typing import Any

import torch


def get_compute_capability() -> tuple[int, int] | None:
    """Return (major, minor) compute capability, or None if not NVIDIA."""
    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None) is not None:
        return None
    return torch.cuda.get_device_capability()


def apply_nvidia_optimizations() -> dict[str, str]:
    """Apply NVIDIA-specific PyTorch settings.

    Returns a dict describing what was enabled (for logging).
    """
    enabled: dict[str, str] = {}

    # TF32 for float32 matmuls — ~2-3x speedup on Ampere+ (compute >= 8.0)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    enabled["tf32"] = "matmul + cudnn"

    cap = get_compute_capability()
    if cap is not None:
        enabled["compute_capability"] = f"{cap[0]}.{cap[1]}"
        if cap[0] >= 10:
            enabled["architecture"] = "Blackwell"
        elif cap[0] >= 9:
            enabled["architecture"] = "Hopper (GH200/H100)"
        elif cap[0] >= 8:
            enabled["architecture"] = "Ampere (A100/A10)"

    enabled["gpu"] = torch.cuda.get_device_name()

    return enabled


def get_compile_kwargs(*, for_training: bool = False) -> dict[str, Any]:
    """Return torch.compile kwargs optimized for NVIDIA GPUs.

    Uses default (automatic-dynamic) mode. ``reduce-overhead`` was
    dropped because our policy-head dispatch uses ``index_select`` with
    per-phase row counts that vary every batch — CUDA graphs require
    static shapes. Global ``dynamic=True`` is explicitly discouraged by
    the PyTorch docs (forces every dim and module parameter dynamic,
    error-prone, can cause perf regressions); the eval-server warmup
    site applies ``mark_unbacked`` on the runtime-varying dims instead.

    Args:
        for_training: Unused today; retained so callers can keep a stable API.
    """
    del for_training
    return {}
