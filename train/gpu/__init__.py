"""GPU vendor detection and optimization dispatch.

Auto-detects NVIDIA vs AMD ROCm hardware and returns a GpuConfig that
bundles vendor-specific torch.compile kwargs, per-process optimizations,
and warmup settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GpuConfig:
    """Vendor-specific GPU optimization configuration.

    Bundles all vendor-dependent behavior into a single object:
    per-process settings, torch.compile kwargs, and warmup config.
    """

    vendor: str  # "nvidia", "amd", or "cpu"

    def apply_optimizations(self) -> dict[str, str]:
        """Apply per-process GPU settings (TF32 for NVIDIA, etc.).

        Returns a dict describing what was enabled, for logging.
        """
        if self.vendor == "nvidia":
            from train.gpu.nvidia import apply_nvidia_optimizations

            return apply_nvidia_optimizations()
        elif self.vendor == "amd":
            from train.gpu.amd import apply_amd_optimizations

            return apply_amd_optimizations()
        return {}

    def get_compile_kwargs(
        self,
        *,
        for_training: bool = False,
        eval_batch_shape_mode: str = "dynamic",
    ) -> dict[str, Any]:
        """Return torch.compile kwargs appropriate for this vendor."""
        if self.vendor == "nvidia":
            from train.gpu.nvidia import get_compile_kwargs

            return get_compile_kwargs(
                for_training=for_training,
                eval_batch_shape_mode=eval_batch_shape_mode,
            )
        elif self.vendor == "amd":
            from train.gpu.amd import get_compile_kwargs

            return get_compile_kwargs(for_training=for_training)
        return {}

    def warmup_batch_size(self, training_batch_size: int) -> int:
        """Return batch size for the torch.compile warmup pass.

        reduce-overhead mode (NVIDIA CUDA graphs / AMD HIP graphs)
        benefits from warming up at the actual training batch size so
        the graph is captured at the right dimensions.
        """
        if self.vendor in ("nvidia", "amd"):
            return training_batch_size
        return 1


def detect_gpu(device_type: str) -> GpuConfig:
    """Auto-detect GPU vendor from the active PyTorch backend.

    Args:
        device_type: ``torch.device.type`` — typically ``"cuda"`` or ``"cpu"``.
    """
    if device_type != "cuda":
        return GpuConfig(vendor="cpu")

    if getattr(torch.version, "hip", None) is None:
        return GpuConfig(vendor="nvidia")
    else:
        return GpuConfig(vendor="amd")
