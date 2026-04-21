from __future__ import annotations

import numpy as np
import pytest
import torch

from train.eval_server import (
    EvaluationServer,
    RequestBatchGroup,
    SharedEvalBuffers,
    _next_power_of_two,
    _partition_request_groups,
    _resolve_actual_launch_cap,
    _resolve_launch_batch_size,
    _resolve_max_launch_batch_size,
)


def test_next_power_of_two_rounds_up() -> None:
    assert _next_power_of_two(1) == 1
    assert _next_power_of_two(2) == 2
    assert _next_power_of_two(3) == 4
    assert _next_power_of_two(5) == 8
    assert _next_power_of_two(513) == 1024


def test_next_power_of_two_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        _next_power_of_two(0)


def test_resolve_actual_launch_cap_uses_explicit_bucket_cap_when_present() -> None:
    assert _resolve_actual_launch_cap(
        max_batch=1024,
        batch_shape_mode="bucketed",
        max_batch_size=512,
    ) == 512


def test_resolve_actual_launch_cap_uses_natural_max_when_dynamic_or_zero_cap() -> None:
    assert _resolve_actual_launch_cap(
        max_batch=1024,
        batch_shape_mode="dynamic",
        max_batch_size=0,
    ) == 1024
    assert _resolve_actual_launch_cap(
        max_batch=24,
        batch_shape_mode="bucketed",
        max_batch_size=0,
    ) == 24


def test_resolve_max_launch_batch_size_rounds_bucketed_cap_up_to_next_power_of_two() -> None:
    assert _resolve_max_launch_batch_size(
        max_batch=24,
        batch_shape_mode="bucketed",
        max_batch_size=0,
    ) == 32
    assert _resolve_max_launch_batch_size(
        max_batch=1024,
        batch_shape_mode="bucketed",
        max_batch_size=512,
    ) == 512


def test_resolve_launch_batch_size_keeps_dynamic_exact_and_buckets_bucketed() -> None:
    assert _resolve_launch_batch_size(actual_n=19, batch_shape_mode="dynamic") == 19
    assert _resolve_launch_batch_size(actual_n=19, batch_shape_mode="bucketed") == 32


def test_partition_request_groups_keeps_single_group_when_within_cap() -> None:
    groups = _partition_request_groups(
        np.array([5, 4, 3], dtype=np.int32),
        n_req=3,
        actual_launch_cap=16,
    )

    assert groups == [RequestBatchGroup(start=0, end=3, actual_n=12)]


def test_partition_request_groups_splits_513_into_512_and_1() -> None:
    groups = _partition_request_groups(
        np.array([256, 256, 1], dtype=np.int32),
        n_req=3,
        actual_launch_cap=512,
    )

    assert groups == [
        RequestBatchGroup(start=0, end=2, actual_n=512),
        RequestBatchGroup(start=2, end=3, actual_n=1),
    ]


def test_partition_request_groups_splits_contiguous_requests_greedily() -> None:
    groups = _partition_request_groups(
        np.array([400, 112, 128, 60], dtype=np.int32),
        n_req=4,
        actual_launch_cap=512,
    )

    assert groups == [
        RequestBatchGroup(start=0, end=2, actual_n=512),
        RequestBatchGroup(start=2, end=4, actual_n=188),
    ]


def test_partition_request_groups_rejects_request_larger_than_cap() -> None:
    with pytest.raises(ValueError, match="exceeds actual_launch_cap"):
        _partition_request_groups(
            np.array([8, 20], dtype=np.int32),
            n_req=2,
            actual_launch_cap=16,
        )


def test_evaluation_server_threads_batch_shape_mode_and_max_batch_size_into_process_kwargs() -> None:
    model = torch.nn.Linear(4, 4)
    shared_bufs = SharedEvalBuffers(num_workers=4, batch_size=8, num_players=3)

    server = EvaluationServer(
        model,
        torch.device("cpu"),
        shared_bufs,
        batch_shape_mode="bucketed",
        max_batch_size=64,
    )

    assert server._process_kwargs["batch_shape_mode"] == "bucketed"
    assert server._process_kwargs["max_batch_size"] == 64
