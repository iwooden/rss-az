"""Relation transport per game: 1889's flat records through the pipeline
(1889/model-design.md §6.1, §6.5, §8 row 8.2).

On random-playout rows of all 19 decision phases, 3-5 players at capacity 5,
with relation multipliers scrambled so that relations matter:

- replay fills the sampled rows' records over stale scratch; flattened, they
  equal a direct fill and hold the dense planes' cells;
- a trainer step on flat records gives the loss of the same step on dense
  planes, across steps whose N differs, and overfits a fixed batch;
- ``NNEvaluator``'s three entries give the same outputs on records as on planes;
- an eval server with real worker processes gives ``NNEvaluator``'s priors and
  values for the same states, whichever workers a launch combines;
- its bucketed launches ship only B × K rows, the real records and then filler
  the model ignores, and its dynamic launches only the real records;
- the eval server's warmup leaves N dynamic: launches of any shape and real
  count run on its graphs with no graph break and no recompile;
- self-play, replay, and training run on one model, and self-play through an
  eval server and a worker.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters

from g1889.relations import get_relation_coord_data_batch, get_relation_data_batch
from g1889.token_data import get_num_tokens
from games import Game, get_game
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from nn.model_contract import FLAT_RELATION_RECORD_WIDTH, flatten_relation_records
from train import trainer as trainer_module
from train.config import TrainingConfig
from train.eval_server import (
    EvaluationServer,
    RemoteEvaluator,
    SharedEvalBuffers,
    _flat_filler_rows,
    _next_power_of_two,
    _resolve_flat_launch_rows,
    _warmup_compiled_model,
)
from train.gpu import detect_gpu
from train.replay_buffer import ReplayBuffer
from train.self_play import play_game, self_play_worker
from train.trainer import Trainer

from tests.g1889.test_extraction import _state
from tests.g1889.test_model import Sample, _sample_setup, _slot

MAX_PLAYERS = 5
TOKEN_DIM = 108
EDGES = 2048
UNIFIED = 228
NUM_TOKENS = get_num_tokens(MAX_PLAYERS)

CONFIG = TrainingConfig(
    game="1889",
    model_path="nn/transformer-1889.py",
    num_players=0,
    min_players=3,
    max_players=MAX_PLAYERS,
    d_model=64,
    num_heads=4,
    num_layers=3,
    optimizer="adamw",
    warmup_epochs=0,
    grad_clip=0,
    weight_decay=0,
    learning_rate=3e-3,
    lr_min=3e-3,
    seed=0,
)
GAME = get_game(CONFIG)
DENSE_GAME = replace(GAME, relation_format="dense")

# The GPU runs fp32 and uncompiled. Local ROCm's dense-plane einsum returns garbage
# for rows past M = 2**19, 58 of them at 95 tokens, so twins on planes stay below
# that; flat records are fine at 128 and above.
DENSE_BATCH = 48
DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]

# fp32 tolerance between a flat path and its twin. Losses after optimizer steps
# drift further, though the update starts from the same gradient.
TOLERANCE = 2e-5
LATER_STEP_TOLERANCE = 2e-4
WIRE_TOLERANCE = 2e-3  # eval-server outputs against fp32 tokens, through the fp16 wire


@pytest.fixture(scope="module")
def samples() -> list[Sample]:
    return [sample for num_players in (3, 4, 5) for sample in _sample_setup(num_players, MAX_PLAYERS)]


def _stack_rows(samples: list[Sample]) -> np.ndarray:
    return np.stack([sample.row for sample in samples])


def _masks(samples: list[Sample]) -> np.ndarray:
    masks = np.zeros((len(samples), UNIFIED), np.uint8)
    for i, sample in enumerate(samples):
        masks[i, [_slot(sample.phase, action) for action in sample.legal]] = 1
    return masks


def _buffer(game: Game, samples: list[Sample]) -> ReplayBuffer:
    """A full buffer of ``samples`` with random targets over their legal slots."""
    rng = np.random.default_rng(0)
    masks = _masks(samples)
    targets = np.zeros_like(masks, np.float32)
    values = np.zeros((len(samples), MAX_PLAYERS), np.float32)
    for i, sample in enumerate(samples):
        slots = masks[i].nonzero()[0]
        targets[i, slots] = rng.dirichlet(np.ones(len(slots)))
        values[i, :sample.num_players] = rng.uniform(-1, 1, sample.num_players)
    buffer = ReplayBuffer(
        game, len(samples), game.state_size(MAX_PLAYERS), MAX_PLAYERS,
        min_players=3, max_players=MAX_PLAYERS,
    )
    buffer.add_stacked(
        _stack_rows(samples), np.array([s.phase for s in samples], np.int8), masks, targets, values,
        player_counts=np.array([s.num_players for s in samples], np.uint8),
    )
    return buffer


def _model(device: str | torch.device = "cpu") -> torch.nn.Module:
    """A small model whose relations matter: at initialization their multipliers
    are zero and the logits are tiny, which would hide what these tests look for."""
    torch.manual_seed(0)
    model = create_model(CONFIG)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.15)
    nn.init.normal_(model.get_parameter("relation_bias_mult"), std=0.5)
    return model.to(device)


# =============================================================================
# ADAPTERS AND REPLAY
# =============================================================================

def test_each_game_declares_its_format():
    assert get_game(TrainingConfig()).relation_format == "dense"
    assert GAME.relation_format == "flat"


def test_sample_into_fills_the_sampled_rows_records_over_stale_scratch(samples):
    buffer = _buffer(GAME, samples)
    batch = 96
    states = np.empty((batch, GAME.state_size(MAX_PLAYERS)), np.int16)
    phases = np.empty(batch, np.int64)
    masks = np.empty((batch, UNIFIED), np.uint8)
    targets = np.empty((batch, UNIFIED), np.float32)
    values = np.empty((batch, MAX_PLAYERS), np.float32)
    records = np.full((batch, EDGES, 4), 7, np.uint8)

    buffer.sample_into(batch, np.random.default_rng(1), states, phases, masks, targets, values, relations_out=records)

    direct = np.empty_like(records)
    get_relation_coord_data_batch(list(states), direct, MAX_PLAYERS)
    flat = flatten_relation_records(records)
    np.testing.assert_array_equal(flat, flatten_relation_records(direct))
    assert flat.dtype == np.int16 and flat.shape[1] == FLAT_RELATION_RECORD_WIDTH
    # Every batch row has records, and they are the planes' cells for that row.
    assert set(flat[:, 0]) == set(range(batch))
    planes = np.zeros((batch, GAME.num_relations, NUM_TOKENS, NUM_TOKENS), np.uint8)
    get_relation_data_batch(list(states), planes, MAX_PLAYERS)
    row, relation, query, key, value = flat.T
    np.testing.assert_array_equal(planes[row, relation, query, key], value)
    assert len(flat) == np.count_nonzero(planes)


def test_sample_returns_records_the_caller_flattens(samples):
    buffer = _buffer(GAME, samples)
    batch = buffer.sample(32, np.random.default_rng(2))
    records = batch["relations"]
    assert records.dtype == torch.uint8 and tuple(records.shape) == (32, EDGES, 4)
    direct = np.empty((32, EDGES, 4), np.uint8)
    get_relation_coord_data_batch(list(batch["states"].numpy()), direct, MAX_PLAYERS)
    np.testing.assert_array_equal(records.numpy(), direct)


# =============================================================================
# TRAINER
# =============================================================================

def _train(
    monkeypatch: pytest.MonkeyPatch, game: Game, samples: list[Sample], device: str,
    batch: int, steps: int,
) -> list[dict[str, float]]:
    monkeypatch.setattr(trainer_module, "get_game", lambda _config: game)
    trainer = Trainer(_model(device), CONFIG, torch.device(device))
    buffer = _buffer(game, samples)
    rng = np.random.default_rng(3)
    return [trainer.train_step(buffer, batch, rng) for _ in range(steps)]


@pytest.mark.parametrize("device", DEVICES)
def test_trainer_steps_on_records_match_steps_on_planes(monkeypatch, samples, device):
    """Each batch has a different N, and the optimizer steps between them, so
    the later steps' losses compare the updates as well."""
    flat = _train(monkeypatch, GAME, samples, device, batch=DENSE_BATCH, steps=4)
    dense = _train(monkeypatch, DENSE_GAME, samples, device, batch=DENSE_BATCH, steps=4)
    for step, (flat_step, dense_step) in enumerate(zip(flat, dense, strict=True)):
        assert flat_step.keys() == dense_step.keys()
        tolerance = LATER_STEP_TOLERANCE if step else TOLERANCE
        for name in ("policy_loss", "value_loss", "total_loss", "policy_kl"):
            assert flat_step[name] == pytest.approx(dense_step[name], rel=tolerance, abs=tolerance), name
    # The comparison has power: the same step without relations is far off.
    empty = replace(GAME, fill_relations_sparse_batch=lambda rows, records, *, max_players: records.fill(0))
    without = _train(monkeypatch, empty, samples, device, batch=DENSE_BATCH, steps=1)
    assert abs(without[0]["total_loss"] - flat[0]["total_loss"]) > 100 * TOLERANCE


def test_trainer_scratch_holds_a_full_batch_of_records(monkeypatch):
    monkeypatch.setattr(trainer_module, "get_game", lambda _config: GAME)
    trainer = Trainer(_model(), CONFIG, torch.device("cpu"))
    trainer._ensure_scratch(4)
    assert trainer._rel_h.dtype == torch.int16
    assert tuple(trainer._rel_h.shape) == (4 * EDGES, FLAT_RELATION_RECORD_WIDTH)
    assert trainer._rel_d.shape == trainer._rel_h.shape


@pytest.mark.parametrize("device", DEVICES)
def test_trainer_overfits_a_fixed_batch(monkeypatch, samples, device):
    """A buffer of one batch's rows: every step trains on all of them."""
    fixed = samples[::5]
    steps = _train(monkeypatch, GAME, fixed, device, batch=len(fixed), steps=250)
    assert steps[-1]["policy_kl"] < 0.05 < steps[0]["policy_kl"]
    assert steps[-1]["value_loss"] < 0.02 < steps[0]["value_loss"]


# =============================================================================
# NNEVALUATOR
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_nn_evaluator_gives_the_same_outputs_on_records_as_on_planes(samples, device):
    model = _model(device).eval()
    on_records = NNEvaluator(GAME, model, torch.device(device), num_players=MAX_PLAYERS)
    on_planes = NNEvaluator(DENSE_GAME, model, torch.device(device), num_players=MAX_PLAYERS)

    # One state at a time, with N rising and falling across calls.
    for sample in samples[:: len(samples) // 12]:
        for flat_part, dense_part in zip(
            on_records.evaluate(_state(sample.row)), on_planes.evaluate(_state(sample.row)), strict=True,
        ):
            np.testing.assert_allclose(flat_part, dense_part, atol=TOLERANCE, rtol=TOLERANCE)

    # A batch of mixed player counts, then each count's leaves as search sends them.
    states = [_state(sample.row) for sample in samples[:: len(samples) // DENSE_BATCH][:DENSE_BATCH]]
    for flat_result, dense_result in zip(
        on_records.evaluate_batch(states), on_planes.evaluate_batch(states), strict=True,
    ):
        for flat_part, dense_part in zip(flat_result, dense_result, strict=True):
            np.testing.assert_allclose(flat_part, dense_part, atol=TOLERANCE, rtol=TOLERANCE)
    for num_players in (3, 4, 5):
        count = [sample for sample in samples if sample.num_players == num_players][:32]
        rows, masks = list(_stack_rows(count)), _masks(count)
        for flat_part, dense_part in zip(
            on_records.evaluate_leaves(rows, masks), on_planes.evaluate_leaves(rows, masks), strict=True,
        ):
            np.testing.assert_allclose(flat_part, dense_part, atol=TOLERANCE, rtol=TOLERANCE)


def test_nn_evaluator_sends_the_model_only_real_records(samples):
    """The forward sees the batch's flat records, not its scratch."""
    seen: list[torch.Tensor] = []
    model = _model()
    model.register_forward_pre_hook(lambda _module, args: seen.append(args[2].clone()))
    evaluator = NNEvaluator(GAME, model, torch.device("cpu"), num_players=MAX_PLAYERS)
    count = [sample for sample in samples if sample.num_players == 4]
    # A longer batch first, so that the shorter one's scratch holds stale rows.
    for leaves in (count[:32], count[:8]):
        evaluator.evaluate_leaves(list(_stack_rows(leaves)), _masks(leaves))
        records = np.empty((len(leaves), EDGES, 4), np.uint8)
        get_relation_coord_data_batch(list(_stack_rows(leaves)), records, MAX_PLAYERS)
        np.testing.assert_array_equal(seen[-1].numpy(), flatten_relation_records(records))


# =============================================================================
# EVAL SERVER: REAL WORKER PROCESSES
# =============================================================================

def _wire_rounded(game: Game) -> Game:
    """``game`` with its tokens rounded as workers' fp16 wire rounds them.

    Workers cast tokens to fp16 before the server sees them, so the server's
    outputs differ from fp32 tokens' by that rounding; this twin leaves only
    the transport to compare.
    """
    def rounded(fill: Any) -> Any:
        def fill_rounded(rows: Any, tokens: np.ndarray, **kwargs: Any) -> None:
            fill(rows, tokens, **kwargs)
            tokens[:] = tokens.astype(np.float16).astype(np.float32)
        return fill_rounded
    return replace(game, fill_tokens=rounded(game.fill_tokens), fill_tokens_batch=rounded(game.fill_tokens_batch))


def _eval_worker(
    config: TrainingConfig, shared_bufs: SharedEvalBuffers, worker_idx: int,
    leaves: list[tuple[np.ndarray, np.ndarray]], singles: np.ndarray, results: Any,
) -> None:
    """A self-play worker's evaluator, driven by given states."""
    torch.set_num_threads(1)
    game = get_game(config)
    evaluator = RemoteEvaluator(game, config.effective_max_players, shared_bufs, worker_idx)
    root = game.new_game(int(game.num_players_from_row(singles[0])), config.effective_max_players, 0)
    results.put((
        worker_idx,
        [evaluator.evaluate_leaves(list(rows), masks) for rows, masks in leaves],
        [evaluator.evaluate(game.scratch_state(root, row)) for row in singles],
    ))


class _LaunchRecorder(nn.Module):
    """A real model that also tells the test what each launch shipped."""

    def __init__(self, model: nn.Module, launches: Any) -> None:
        super().__init__()
        self.model = model
        self.launches = launches

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor, relations: torch.Tensor) -> Any:
        # A server that dies mustn't wait to flush launches nobody is reading.
        self.launches.cancel_join_thread()
        self.launches.put((x.shape[0], relations.cpu().numpy().copy()))
        return self.model(x, legal_mask, relations)


def _round_trip(
    samples: list[Sample], device: str, mode: str, leaves: int,
) -> tuple[dict[int, tuple[list[list[Sample]], list[Sample]]], dict[int, Any], list[tuple[int, np.ndarray]]]:
    """Three real workers, each of one player count, as search sends its states,
    through a real server. Returns each worker's requests and single states, its
    outputs, and each launch's batch size and flat rows."""
    ctx = mp.get_context("spawn")
    launch_queue = ctx.Queue()
    spec = get_model_input_spec(CONFIG)
    shared_bufs = SharedEvalBuffers(GAME, 3, batch_size=leaves, num_players=MAX_PLAYERS, input_spec=spec)
    shared_bufs.init_bitmap([(0, 3)], ctx)
    server = EvaluationServer(
        _LaunchRecorder(_model(), launch_queue).eval(), torch.device(device), shared_bufs,
        profile=True, mp_context=ctx, no_compile=True, gpu_vendor=detect_gpu(device).vendor,
        # Wait for all three workers' full requests, or a moment for short ones.
        min_batch_size=3 * leaves, min_batch_timeout_ms=500.0,
        batch_shape_mode=mode, max_batch_size=_next_power_of_two(3 * leaves),
    )

    # Long requests and short ones, in shuffled phases: N rises and falls, and a
    # short request's slot still holds a long one's records.
    sizes = [leaves, 1, leaves, 3, leaves // 2, 2]
    requests = {}
    for worker, num_players in enumerate((3, 4, 5)):
        count = [sample for sample in samples if sample.num_players == num_players]
        rng = np.random.default_rng(worker)
        count = [count[i % len(count)] for i in rng.permutation(sum(sizes) + 2)]
        stops = np.cumsum(sizes)
        requests[worker] = ([count[stop - size:stop] for stop, size in zip(stops, sizes, strict=True)], count[-2:])

    results = ctx.Queue()
    workers = [
        ctx.Process(
            target=_eval_worker,
            args=(
                CONFIG, shared_bufs, worker,
                [(_stack_rows(part), _masks(part)) for part in parts],
                _stack_rows(singles), results,
            ),
            daemon=True,
        )
        for worker, (parts, singles) in requests.items()
    ]
    received = {}
    try:
        server.start()
        assert server.wait_ready(timeout=120.0)
        for process in workers:
            process.start()
        for _ in workers:
            worker, *outputs = _get_result(results, server, workers)
            received[worker] = outputs
        stats = server.get_profile_stats()
        assert stats is not None
        launches = [launch_queue.get(timeout=30.0) for _ in range(stats.batch_count)]
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5.0)
        server.stop()
    return requests, received, launches


def _assert_bucketed_launches(launches: list[tuple[int, np.ndarray]]) -> None:
    """Shapes are B × K alone, though N varies; each launch is its real records,
    then filler that names its own states, a cell to a row."""
    shapes, real_counts = set(), set()
    for launch_n, rows in launches:
        assert launch_n == _next_power_of_two(launch_n)
        rows_per_state = len(rows) // launch_n
        assert len(rows) == launch_n * rows_per_state and rows_per_state in (256, 512, 1024, 2048)
        shapes.add((launch_n, rows_per_state))

        batch, relation, query, key, value = rows.T
        real_rows = int((value > 0).sum())
        real_counts.add(real_rows)
        assert (value[:real_rows] > 0).all() and (value[real_rows:] == 0).all()
        assert real_rows > launch_n * rows_per_state // 2 or rows_per_state == 256  # the smallest K
        assert (np.diff(batch[:real_rows]) >= 0).all()
        assert ((batch >= 0) & (batch < launch_n)).all()
        filler = rows[real_rows:]
        assert (filler[:, 1] == 0).all() and ((filler[:, 2:4] >= 0) & (filler[:, 2:4] < NUM_TOKENS)).all()
        assert len(np.unique(filler[:, [0, 2, 3]], axis=0)) == len(filler)
    assert len(real_counts) > len(shapes)


def _assert_dynamic_launches(launches: list[tuple[int, np.ndarray]]) -> None:
    """Each launch is exactly its real records, in no fixed shape."""
    for launch_n, rows in launches:
        assert (rows[:, 4] > 0).all() and ((rows[:, 0] >= 0) & (rows[:, 0] < launch_n)).all()
    assert any(len(rows) % (launch_n * 256) for launch_n, rows in launches)


@pytest.mark.parametrize(
    ("device", "mode", "leaves"),
    [
        ("cpu", "dynamic", 8),
        ("cpu", "bucketed", 8),
        # Launches of up to 120 rows in a bucket of 128, the local GPU's limit.
        pytest.param(
            "cuda", "bucketed", 40,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_eval_server_round_trip_gives_nn_evaluator_s_outputs(samples, device, mode, leaves):
    """Launches combine workers, so each state's records must find its row."""
    requests, received, launches = _round_trip(samples, device, mode, leaves)
    assert max(launch_n for launch_n, _rows in launches) > leaves
    (_assert_bucketed_launches if mode == "bucketed" else _assert_dynamic_launches)(launches)

    # The rounded twin leaves the transport; fp32 tokens also differ by the wire's rounding.
    model = _model(device).eval()
    device_t = torch.device(device)
    rounded = NNEvaluator(_wire_rounded(GAME), model, device_t, num_players=MAX_PLAYERS)
    plain = NNEvaluator(GAME, model, device_t, num_players=MAX_PLAYERS)
    for worker, (parts, singles) in requests.items():
        remote_parts, remote_singles = received[worker]
        for part, (remote_priors, remote_values) in zip(parts, remote_parts, strict=True):
            assert remote_priors.shape == (len(part), UNIFIED)
            assert remote_values.shape == (len(part), part[0].num_players)
            rows, masks = list(_stack_rows(part)), _masks(part)
            priors, values = rounded.evaluate_leaves(rows, masks)
            np.testing.assert_allclose(remote_priors, priors, atol=TOLERANCE, rtol=TOLERANCE)
            np.testing.assert_allclose(remote_values, values, atol=TOLERANCE, rtol=TOLERANCE)
            plain_priors, plain_values = plain.evaluate_leaves(rows, masks)
            np.testing.assert_allclose(remote_priors, plain_priors, atol=WIRE_TOLERANCE)
            np.testing.assert_allclose(remote_values, plain_values, atol=WIRE_TOLERANCE)
        root = GAME.new_game(singles[0].num_players, MAX_PLAYERS, 0)  # the flags the workers' states carry
        for sample, remote in zip(singles, remote_singles, strict=True):
            local = rounded.evaluate(GAME.scratch_state(root, sample.row.copy()))
            np.testing.assert_allclose(remote[0], local[0], atol=TOLERANCE, rtol=TOLERANCE)
            np.testing.assert_allclose(remote[1], local[1], atol=TOLERANCE, rtol=TOLERANCE)
            np.testing.assert_array_equal(remote[2], local[2])
            assert remote[3:] == local[3:]


def _get_result(results: Any, server: EvaluationServer, workers: list[Any]) -> Any:
    """The next worker's outputs, or a failure as soon as a process dies."""
    for _ in range(600):
        try:
            return results.get(timeout=0.5)
        except queue.Empty:
            server.check_alive()
            if any(process.exitcode not in (None, 0) for process in workers):
                raise RuntimeError("an eval worker died") from None
    raise TimeoutError("eval workers did not finish")


# =============================================================================
# ROUNDING UP A BUCKETED LAUNCH
# =============================================================================

@pytest.mark.parametrize(
    ("real_rows", "launch_n", "expected"),
    [
        (0, 1, 256), (2, 1, 256), (256, 1, 256), (257, 1, 512), (1050, 1, 2048), (2048, 1, 2048),
        (512, 2, 512), (513, 2, 1024), (58_824, 256, 65_536), (65_536, 256, 65_536), (65_537, 256, 131_072),
        (256 * 2048, 256, 256 * 2048),
    ],
)
def test_bucketed_launches_round_up_to_the_smallest_b_by_k(real_rows, launch_n, expected):
    assert _resolve_flat_launch_rows(
        real_rows=real_rows, launch_n=launch_n, batch_shape_mode="bucketed",
    ) == expected


def test_dynamic_launches_keep_their_real_rows():
    assert _resolve_flat_launch_rows(real_rows=1234, launch_n=8, batch_shape_mode="dynamic") == 1234


def test_a_launch_s_filler_prefix_is_well_formed():
    """Every prefix a launch of B states can take names only those states, and
    its rows have cells of their own, at any capacity."""
    for num_tokens in (93, 94, 95):
        table = _flat_filler_rows(256 * EDGES, EDGES, num_tokens)
        assert table.dtype == np.int16
        assert (table[:, [1, 4]] == 0).all() and (table >= 0).all() and (table[:, 2:4] < num_tokens).all()
        assert len(np.unique(table[:, [0, 2, 3]], axis=0)) == len(table)
        for launch_n in (1, 2, 8, 256):
            assert table[:launch_n * EDGES, 0].max() == launch_n - 1


def test_filler_leaves_the_forward_exactly_unchanged(samples):
    """A launch with a scratch row and filler, against the same launch without:
    the same bits, not just fp32 tolerance."""
    leaves = samples[-3:]
    model = _model().eval()
    rows = list(_stack_rows(leaves))
    tokens = torch.zeros(4, NUM_TOKENS, TOKEN_DIM)
    GAME.fill_tokens_batch(rows, tokens.numpy(), max_players=MAX_PLAYERS, layout_version=1)
    mask = torch.zeros(4, UNIFIED, dtype=torch.bool)
    mask[:3] = torch.from_numpy(_masks(leaves)).bool()
    records = np.empty((3, EDGES, 4), np.uint8)
    GAME.fill_relations_sparse_batch(rows, records, max_players=MAX_PLAYERS)
    flat = flatten_relation_records(records)
    launch_rows = _resolve_flat_launch_rows(real_rows=len(flat), launch_n=4, batch_shape_mode="bucketed")
    assert launch_rows > len(flat)
    padded = np.concatenate([flat, _flat_filler_rows(launch_rows - len(flat), EDGES, NUM_TOKENS)])
    with torch.inference_mode():
        logits, values = model(tokens, mask, torch.from_numpy(padded))
        plain_logits, plain_values = model(tokens, mask, torch.from_numpy(flat))
    assert torch.equal(logits[:3], plain_logits[:3]) and torch.equal(values[:3], plain_values[:3])


# =============================================================================
# THE EVAL SERVER'S WARMUP
# =============================================================================

def _record_count(row: np.ndarray) -> int:
    records = np.empty((1, EDGES, 4), np.uint8)
    GAME.fill_relations_sparse_batch([row], records, max_players=MAX_PLAYERS)
    return len(flatten_relation_records(records))


def test_warmup_leaves_n_dynamic_in_each_bucket(samples):
    """After the warmup every launch runs on its graphs, whatever its shape and
    real count: no graph breaks, and no recompile."""
    spec = get_model_input_spec(CONFIG)
    shared_bufs = SharedEvalBuffers(GAME, 1, batch_size=8, num_players=MAX_PLAYERS, input_spec=spec)
    model = _model().eval()
    backend = CompileCounterWithBackend("aot_eager")
    compiled = cast(torch.nn.Module, torch.compile(model, backend=backend, fullgraph=True))
    counters.clear()
    _warmup_compiled_model(
        compiled, shared_bufs, torch.device("cpu"), batch_shape_mode="bucketed", max_launch_batch=8,
        autocast_dtype=None, input_dtype=torch.float32,
    )
    warmup_frames = backend.frame_count
    assert 1 <= warmup_frames <= 4  # at most one for each of batches 1, 2, 4, and 8

    gpu_rel_flat = torch.empty(8 * EDGES, FLAT_RELATION_RECORD_WIDTH, dtype=torch.int16)
    filler = _flat_filler_rows(8 * EDGES, EDGES, NUM_TOKENS)
    # The warmup ran K = 256; the most crowded states also reach larger Ks.
    ranked = sorted(samples, key=lambda sample: _record_count(sample.row))
    launches = ((8, 8), (4, 3), (8, 5), (4, 4), (2, 2), (1, 1), (1, 1))
    picks = (ranked[:8], ranked[-3:], ranked[:5], ranked[-8:-4], ranked[-2:], ranked[-1:], ranked[0:1])
    real_counts, shapes = set(), set()
    for (launch, size), leaves in zip(launches, picks, strict=True):
        rows = list(_stack_rows(leaves))
        tokens = torch.zeros(launch, NUM_TOKENS, TOKEN_DIM)
        GAME.fill_tokens_batch(rows, tokens.numpy(), max_players=MAX_PLAYERS, layout_version=spec.layout_version)
        mask = torch.zeros(launch, UNIFIED, dtype=torch.bool)
        mask[:size] = torch.from_numpy(_masks(leaves)).bool()
        records = np.empty((size, EDGES, 4), np.uint8)
        GAME.fill_relations_sparse_batch(rows, records, max_players=MAX_PLAYERS)
        flat = torch.from_numpy(flatten_relation_records(records))
        real_counts.add(len(flat))
        launch_rows = _resolve_flat_launch_rows(
            real_rows=len(flat), launch_n=launch, batch_shape_mode="bucketed",
        )
        shapes.add((launch, launch_rows // launch))
        gpu_rel_flat[:len(flat)] = flat
        gpu_rel_flat[len(flat):launch_rows] = torch.from_numpy(filler[:launch_rows - len(flat)])
        # The serve loop's persistent context, which the warmup matches.
        with torch.inference_mode(), torch.autocast("cpu", enabled=False, cache_enabled=False):
            logits, values = compiled(tokens, mask, gpu_rel_flat[:launch_rows])
            eager_logits, eager_values = model(tokens, mask, flat)
        torch.testing.assert_close(logits[:size], eager_logits[:size], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(values[:size], eager_values[:size], rtol=1e-4, atol=1e-4)
    assert len(real_counts) == len(launches)
    assert len({k for _launch, k in shapes}) >= 2
    assert backend.frame_count == warmup_frames
    assert not counters["graph_break"]
    torch._dynamo.reset()


# =============================================================================
# DOWNSTREAM: SELF-PLAY, REPLAY, AND TRAINING
# =============================================================================

SMOKE_CONFIG = replace(
    CONFIG,
    d_model=32,
    num_layers=1,
    games_per_epoch=1,
    num_epochs=1,
    num_simulations=2,
    search_batch_size=2,
    dirichlet_epsilon=0.0,
    buffer_capacity=4096,
    min_buffer_size=16,
    batch_size=16,
    training_steps_per_epoch=1,
    value_blend_start_epoch=0,
    value_blend_end_epoch=0,
    num_eval_servers=1,
)


def _assert_records_train(config: TrainingConfig, model: torch.nn.Module, records: list[Any]) -> None:
    buffer = ReplayBuffer(
        GAME, config.buffer_capacity, GAME.state_size(MAX_PLAYERS), MAX_PLAYERS,
        min_players=3, max_players=MAX_PLAYERS,
    )
    for record in records:
        assert record.num_examples > 0
        buffer.add_stacked(
            record.states, record.phase_ids, record.legal_masks, record.policy_targets,
            record.value_targets, num_players=record.num_players,
        )
    trainer = Trainer(model, config, torch.device("cpu"))
    losses = trainer.train_step(buffer, config.batch_size, np.random.default_rng(0))
    assert np.isfinite(losses["total_loss"])


def test_one_model_plays_with_nn_evaluator_and_trains():
    """The single-process path: one model serves ``NNEvaluator`` and the trainer."""
    config = replace(SMOKE_CONFIG, num_workers=0)
    torch.manual_seed(0)
    model = create_model(config)
    evaluator = NNEvaluator(
        GAME, model, torch.device("cpu"), num_players=MAX_PLAYERS,
        terminal_rank_weight=config.terminal_blend, input_spec=get_model_input_spec(config),
    )
    pool = StatePool(GAME, 2 * (config.max_simulations + 1), GAME.state_size(MAX_PLAYERS))
    record = play_game(
        evaluator, config, game_seed=0, rng=np.random.default_rng(0), state_pool=pool,
        epoch_config=config.compute_epoch_config(0), num_players=3,
    )
    _assert_records_train(config, model, [record])


def test_a_worker_plays_through_the_eval_server_and_its_rows_train():
    config = replace(SMOKE_CONFIG, num_workers=1)
    ctx = mp.get_context("spawn")
    torch.manual_seed(0)
    model = create_model(config)
    shared_bufs = SharedEvalBuffers(
        GAME, 1, batch_size=config.search_batch_size, num_players=MAX_PLAYERS,
        input_spec=get_model_input_spec(config),
    )
    shared_bufs.init_bitmap([(0, 1)], ctx)
    server = EvaluationServer(
        model, torch.device("cpu"), shared_bufs, mp_context=ctx, no_compile=True,
        batch_shape_mode="bucketed", max_batch_size=8,
    )
    tasks, results = ctx.Queue(), ctx.Queue()
    worker = ctx.Process(target=self_play_worker, args=(tasks, results, config, shared_bufs, 0), daemon=True)
    try:
        server.start()
        assert server.wait_ready(timeout=120.0)
        worker.start()
        tasks.put((0, 1, 3, config.compute_epoch_config(0)))
        record = _get_result(results, server, [worker])
        tasks.put(None)
        worker.join(timeout=10.0)
        assert not worker.is_alive()
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)
        server.stop()
    _assert_records_train(config, model, [record])
