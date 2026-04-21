from __future__ import annotations

import torch

from train.gpu.nvidia import apply_nvidia_optimizations, get_compile_kwargs


def test_nvidia_compile_kwargs_use_max_autotune_without_cudagraphs_for_eval() -> None:
    kwargs = get_compile_kwargs(for_training=False)

    assert kwargs["mode"] == "max-autotune-no-cudagraphs"
    assert kwargs["options"]["triton.autotune_at_compile_time"] is True
    assert kwargs["options"]["shape_padding"] is False
    assert "joint_graph_constant_folding" not in kwargs["options"]


def test_nvidia_compile_kwargs_disable_joint_graph_constant_folding_for_training() -> None:
    kwargs = get_compile_kwargs(for_training=True)

    assert kwargs["mode"] == "max-autotune-no-cudagraphs"
    assert kwargs["options"]["joint_graph_constant_folding"] is False


def test_apply_nvidia_optimizations_enables_tf32_controls(monkeypatch) -> None:
    recorded: list[str] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (12, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch,
        "set_float32_matmul_precision",
        lambda mode: recorded.append(f"precision={mode}"),
    )

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        enabled = apply_nvidia_optimizations()

        assert recorded == ["precision=high"]
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert enabled["tf32"] == "matmul + cudnn"
        assert enabled["architecture"] == "Blackwell"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
