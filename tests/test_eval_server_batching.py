from __future__ import annotations

import multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.attention_relations import (
    ATTENTION_RELATION_COORD_WIDTH,
    MAX_ATTENTION_RELATION_EDGES,
    NUM_ATTENTION_RELATIONS,
    AttentionRelation,
)
from core.resnet_data import get_resnet_vector_size
from core.state import GameState
from nn import get_model_input_spec
from nn.model_contract import ModelKind
from nn.transformer import UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.eval_server import (
    EvaluationServer,
    RemoteEvaluator,
    RequestBatchGroup,
    SharedEvalBuffers,
    _materialize_relation_coords_,
    _next_power_of_two,
    _partition_request_groups,
    _resolve_actual_launch_cap,
    _resolve_launch_batch_size,
    _resolve_max_launch_batch_size,
)


NUM_PLAYERS = 3
U_DIM = int(UNIFIED_LOGIT_DIM)


class SyncValueModel(torch.nn.Module):
    def __init__(self, num_players: int = NUM_PLAYERS) -> None:
        super().__init__()
        self.cfg = SimpleNamespace(num_players=num_players)
        self.values = torch.nn.Parameter(torch.zeros(num_players))

    def forward(
        self,
        tokens: torch.Tensor,
        legal_mask: torch.Tensor,
        relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del tokens, relations
        legal = legal_mask.to(torch.bool)
        logits = torch.zeros(
            legal.shape[0],
            U_DIM,
            dtype=self.values.dtype,
            device=legal.device,
        )
        logits = logits.masked_fill(~legal, -1e9)
        values = self.values.to(legal.device).expand(legal.shape[0], -1)
        return logits, values


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


def test_evaluation_server_sync_weights_updates_remote_eval_outputs() -> None:
    ctx = mp.get_context("spawn")
    model = SyncValueModel()
    shared_bufs = SharedEvalBuffers(
        num_workers=1,
        batch_size=1,
        num_players=NUM_PLAYERS,
    )
    shared_bufs.init_bitmap([(0, 1)], ctx)
    server = EvaluationServer(
        model,
        torch.device("cpu"),
        shared_bufs,
        mp_context=ctx,
        no_compile=True,
    )
    state = GameState(NUM_PLAYERS)
    state.initialize_game(NUM_PLAYERS, seed=123)
    legal_mask = np.zeros((1, U_DIM), dtype=np.uint8)
    legal_mask[0, 0] = 1

    try:
        server.start()
        assert server.wait_ready(timeout=15.0)
        evaluator = RemoteEvaluator(NUM_PLAYERS, shared_bufs, worker_idx=0)

        _priors, values = evaluator.evaluate_leaves([state._array], legal_mask)
        np.testing.assert_allclose(
            values,
            np.zeros((1, NUM_PLAYERS), dtype=np.float32),
        )

        with torch.no_grad():
            model.values.copy_(torch.tensor([0.25, -0.5, 0.75]))
        state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        server.sync_weights(state_dict, timeout=15.0)

        _priors, values = evaluator.evaluate_leaves([state._array], legal_mask)
        np.testing.assert_allclose(
            values,
            np.array([[0.25, -0.5, 0.75]], dtype=np.float32),
        )
    finally:
        server.stop()


def test_attention_relation_count_matches_enum_members() -> None:
    assert NUM_ATTENTION_RELATIONS == len(AttentionRelation)


def test_shared_eval_buffers_allocates_uint8_relation_coords() -> None:
    shared_bufs = SharedEvalBuffers(
        num_workers=2,
        batch_size=4,
        num_players=3,
    )

    assert shared_bufs.model_type == ModelKind.TRANSFORMER.value
    assert shared_bufs.uses_relations is True
    assert shared_bufs.values_are_active_relative is False
    assert shared_bufs.input_dim == 0
    assert shared_bufs._states.dtype == torch.float16
    assert tuple(shared_bufs._states.shape) == (
        2,
        4,
        shared_bufs.num_tokens,
        shared_bufs.token_dim,
    )
    assert shared_bufs.num_relations == NUM_ATTENTION_RELATIONS
    assert shared_bufs.max_relation_edges == MAX_ATTENTION_RELATION_EDGES
    assert shared_bufs.relation_coord_width == ATTENTION_RELATION_COORD_WIDTH
    assert shared_bufs._relation_coords is not None
    assert shared_bufs._relation_coords.dtype == torch.uint8
    assert tuple(shared_bufs._relation_coords.shape) == (
        2,
        4,
        MAX_ATTENTION_RELATION_EDGES,
        ATTENTION_RELATION_COORD_WIDTH,
    )

    worker_coords = shared_bufs.get_input_relation_coords_np(1)
    assert worker_coords.dtype == np.uint8
    assert worker_coords.shape == (
        4,
        MAX_ATTENTION_RELATION_EDGES,
        ATTENTION_RELATION_COORD_WIDTH,
    )

    worker_coords[0, 2] = (2, 5, 7)
    assert int(shared_bufs._relation_coords[1, 0, 2, 0].item()) == 2
    assert int(shared_bufs._relation_coords[1, 0, 2, 1].item()) == 5
    assert int(shared_bufs._relation_coords[1, 0, 2, 2].item()) == 7


def test_shared_eval_buffers_allocates_float16_vectors_for_resnet() -> None:
    config = TrainingConfig(
        model_type="resnet",
        resnet_hidden_dim=32,
        resnet_num_blocks=0,
    )
    shared_bufs = SharedEvalBuffers(
        num_workers=2,
        batch_size=4,
        num_players=3,
        input_spec=get_model_input_spec(config),
    )

    assert shared_bufs.model_type == ModelKind.RESNET.value
    assert shared_bufs.uses_relations is False
    assert shared_bufs.values_are_active_relative is True
    assert shared_bufs.input_dim == get_resnet_vector_size(3)
    assert shared_bufs.num_tokens == 0
    assert shared_bufs.token_dim == 0
    assert shared_bufs.num_relations == 0
    assert shared_bufs.max_relation_edges == 0
    assert shared_bufs.relation_coord_width == 0
    assert shared_bufs._relation_coords is None
    assert shared_bufs._states.dtype == torch.float16
    assert tuple(shared_bufs._states.shape) == (
        2,
        4,
        get_resnet_vector_size(3),
    )

    worker_vectors = shared_bufs.get_input_vectors_np(1)
    assert worker_vectors.dtype == np.float16
    assert worker_vectors.shape == (4, get_resnet_vector_size(3))

    with pytest.raises(RuntimeError, match="relation coordinates"):
        shared_bufs.get_input_relation_coords_np(1)


def test_materialize_relation_coords_fills_dense_planes_and_clears_padding() -> None:
    batch_size = 3
    num_relations = NUM_ATTENTION_RELATIONS
    num_tokens = 57
    max_edges = 4
    coords = torch.zeros(batch_size, max_edges, 3, dtype=torch.uint8)
    coords[0, 0] = torch.tensor([2, 5, 7], dtype=torch.uint8)
    coords[0, 1] = torch.tensor([0, 0, 0], dtype=torch.uint8)
    coords[1, 0] = torch.tensor([3, 6, 8], dtype=torch.uint8)
    dense = torch.full(
        (batch_size, num_relations, num_tokens, num_tokens),
        9,
        dtype=torch.uint8,
    )
    flat_idx = torch.empty(batch_size, max_edges, dtype=torch.long)
    tmp_idx = torch.empty_like(flat_idx)
    batch_offsets = (
        torch.arange(batch_size, dtype=torch.long)
        * (num_relations * num_tokens * num_tokens)
    ).reshape(batch_size, 1)

    _materialize_relation_coords_(
        dense_rel=dense,
        dense_rel_flat=dense.reshape(-1),
        relation_coords=coords,
        flat_idx=flat_idx,
        flat_idx_flat=flat_idx.reshape(-1),
        tmp_idx=tmp_idx,
        batch_offsets=batch_offsets,
        sentinel_flat=batch_offsets.reshape(-1).contiguous(),
        actual_n=2,
        launch_n=3,
        num_tokens=num_tokens,
    )

    assert int(dense[0, 2, 5, 7].item()) == 1
    assert int(dense[1, 3, 6, 8].item()) == 1
    assert int(dense[0, 0, 0, 0].item()) == 0
    assert int(dense[1, 0, 0, 0].item()) == 0
    assert int(dense[2].sum().item()) == 0
    assert int(dense.sum().item()) == 2
