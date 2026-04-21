from __future__ import annotations

import torch

from train.gpu import GpuConfig
from train.gpu.nvidia import apply_nvidia_optimizations, get_compile_kwargs


def test_nvidia_compile_kwargs_expand_mode_into_options_for_dynamic_eval() -> None:
    kwargs = get_compile_kwargs(for_training=False, eval_batch_shape_mode="dynamic")

    assert "mode" not in kwargs
    assert kwargs["options"]["max_autotune"] is True
    assert kwargs["options"]["coordinate_descent_tuning"] is True
    assert kwargs["options"]["triton.autotune_at_compile_time"] is True
    assert kwargs["options"]["shape_padding"] is False
    assert "joint_graph_constant_folding" not in kwargs["options"]
    assert "triton.cudagraphs" not in kwargs["options"]


def test_nvidia_compile_kwargs_enable_cudagraphs_for_bucketed_eval() -> None:
    kwargs = get_compile_kwargs(for_training=False, eval_batch_shape_mode="bucketed")

    assert "mode" not in kwargs
    assert kwargs["options"]["triton.cudagraphs"] is True
    assert kwargs["options"]["shape_padding"] is False
    assert kwargs["options"]["triton.autotune_at_compile_time"] is True
    assert "joint_graph_constant_folding" not in kwargs["options"]


def test_nvidia_compile_kwargs_disable_joint_graph_constant_folding_for_training() -> None:
    kwargs = get_compile_kwargs(for_training=True)

    assert "mode" not in kwargs
    assert kwargs["options"]["max_autotune"] is True
    assert kwargs["options"]["coordinate_descent_tuning"] is True
    assert kwargs["options"]["joint_graph_constant_folding"] is False


def test_training_compile_kwargs_ignore_eval_batch_shape_mode() -> None:
    dynamic = get_compile_kwargs(for_training=True, eval_batch_shape_mode="dynamic")
    bucketed = get_compile_kwargs(for_training=True, eval_batch_shape_mode="bucketed")

    assert dynamic == bucketed
    assert "triton.cudagraphs" not in bucketed["options"]


def test_gpu_config_threads_eval_batch_shape_mode_for_nvidia() -> None:
    kwargs = GpuConfig(vendor="nvidia").get_compile_kwargs(
        for_training=False,
        eval_batch_shape_mode="bucketed",
    )

    assert kwargs["options"]["triton.cudagraphs"] is True


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
