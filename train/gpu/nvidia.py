"""NVIDIA GPU optimization utilities.

When running on NVIDIA hardware, these optimizations are applied:

1. **TF32 matmul precision** (Ampere+): ~2-3x speedup for float32 matmuls
   with minimal precision loss (19-bit mantissa vs 23-bit). Affects the float32
   backward pass and any non-autocast matmuls.

2. **torch.compile mode="reduce-overhead"** (CUDA graphs): Eliminates kernel
   launch overhead by capturing the entire forward pass as a CUDA graph.
   For our ~60-kernel MLP forward pass, this can double eval server throughput
   by replacing ~300-600us of launch overhead with ~20us graph replay.

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

    Uses mode='reduce-overhead' which enables CUDA graph capture via Inductor,
    eliminating kernel launch overhead (~60 kernels per MLP forward pass).

    Args:
        for_training: If True, returns kwargs for training (fixed batch size).
                     If False, returns kwargs for eval servers (variable batch sizes).
    """
    if for_training:
        # Fixed batch size in training — CUDA graphs without dynamic shapes.
        # The graph is captured once for batch_size and replayed every step.
        return {"mode": "reduce-overhead"}
    else:
        # Eval servers see variable batch sizes (1 to num_workers * search_batch_size).
        # The CUDA graph tree captures graphs for observed sizes and pads
        # new sizes to match existing graphs. After warmup, all common sizes
        # are cached and every call is a graph replay.
        return {"mode": "reduce-overhead", "dynamic": True}
