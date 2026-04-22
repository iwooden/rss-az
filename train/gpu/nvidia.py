"""NVIDIA GPU optimization utilities.

When running on NVIDIA hardware, these optimizations are applied:

1. **TF32 matmul precision** (Ampere+): ~2-3x speedup for float32 matmuls
   with minimal precision loss (19-bit mantissa vs 23-bit). Affects the float32
   backward pass and any non-autocast matmuls.

2. **torch.compile eval policy**:
   Dynamic eval batching stays on the no-CUDA-graphs baseline
   (``max-autotune-no-cudagraphs``) because ``mark_unbacked`` batch dims
   vary every call with MCTS workload pressure. Bucketed eval batching can
   switch to the cudagraph-enabled ``reduce-overhead`` baseline because the
   GPU-visible shapes are constrained to a small repeated set.

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

    # CUDAGraph-friendly bucketed eval can legitimately exercise more than the
    # default eight Dynamo specializations (e.g. power-of-2 buckets up to 512
    # already imply 9-10 shapes). Raise the per-process limit before any
    # torch.compile call sites run.
    torch._dynamo.config.recompile_limit = 16
    enabled["recompile_limit"] = "16"

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


def _resolve_mode_options(mode: str) -> dict[str, Any]:
    """Expand a torch.compile mode into explicit Inductor options.

    Some deployed torch versions reject specifying both ``mode`` and
    ``options`` in the same ``torch.compile(...)`` call. We still need to
    layer repo-specific options on top of the chosen NVIDIA baseline, so we
    resolve the mode eagerly and return a pure-options dict instead.
    """
    list_mode_options = getattr(torch._inductor, "list_mode_options", None)
    if list_mode_options is not None:
        resolved = list_mode_options().get(mode)
        if resolved is not None:
            return dict(resolved)

    # Fallback for older/newer torch builds where ``list_mode_options`` is
    # unavailable or the mode table changes shape.
    if mode == "max-autotune-no-cudagraphs":
        return {
            "max_autotune": True,
            "coordinate_descent_tuning": True,
        }
    if mode == "reduce-overhead":
        return {
            "triton.cudagraphs": True,
            "max_autotune": True,
            "coordinate_descent_tuning": True,
        }
    return {}


def get_compile_kwargs(
    *,
    for_training: bool = False,
    eval_batch_shape_mode: str = "dynamic",
) -> dict[str, Any]:
    """Return torch.compile kwargs optimized for NVIDIA GPUs.

    Eval compile policy depends on the GPU-visible batch-shape strategy.
    ``dynamic`` keeps the current ``max-autotune-no-cudagraphs`` baseline.
    ``bucketed`` switches eval inference to the cudagraph-enabled
    ``reduce-overhead`` baseline, assuming the caller constrains launches to
    a small fixed bucket set. Training remains on the existing dynamic-shape
    path regardless of the eval knob.

    For ``dynamic`` eval we keep the existing rationale:
    single-process warmup applies ``mark_unbacked`` to the batch dim (so
    a single compiled artifact handles every batch size 1..max_batch) —
    CUDA graphs would require re-capture per batch size, defeating the
    point. Global ``dynamic=True`` is explicitly discouraged by the
    PyTorch docs (forces every dim and module parameter dynamic,
    error-prone, can cause perf regressions); ``mark_unbacked`` on just
    the batch dim is the targeted alternative.

    ``triton.autotune_at_compile_time`` moves per-kernel Triton autotune
    from runtime to compile time. The unbacked batch symbol is
    symbolically unconstrained at the lower bound, so Inductor's runtime
    autotune can sketch a 0-batch case and divide by that dim inside its
    block/grid heuristic → ``ZeroDivisionError`` during
    ``autotune_to_one_config``. Compile-time autotune synthesizes
    benchmark inputs using a non-zero size hint for unbacked symints,
    sidestepping the crash. Observed on CUDA 12.8 / torch 2.11; ROCm
    uses a different Triton heuristic and is unaffected.

    ``shape_padding`` is disabled. Inductor's ``pad_mm`` pass reads
    concrete ``mm`` sizes via ``is_mm_compute_bound`` to decide whether
    to zero-pad operands for better tiling — this bakes batch-size
    range guards (e.g. ``3 <= x.size(0) <= 4``) into the compiled
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
    mode = "max-autotune-no-cudagraphs"
    if not for_training and eval_batch_shape_mode == "bucketed":
        mode = "reduce-overhead"
    options: dict[str, Any] = _resolve_mode_options(mode)
    options.update({
        "triton.autotune_at_compile_time": True,
    })
    if for_training:
        options["joint_graph_constant_folding"] = False
    return {"options": options}
