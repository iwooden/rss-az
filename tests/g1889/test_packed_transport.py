"""Layout 3's transport (1889/v2-design.md §7): one packed vector a state
through the pipeline, with ``nn/transformer-1889-v2.py`` as the model.

On random-playout rows of all 19 decision phases, 3-5 players at capacity 5
(``tests/g1889/test_relation_transport.py``'s samples), with a small model
whose weights are scrambled so that its inputs matter:

- **Contracts:** layout 3 is packed, with relation format ``"none"``, beside
  layouts 1 and 2's flat records; the spec, the loader's check of a module
  against its game, and the shared buffers: one fp16 vector a state, no
  relation buffer.
- **Replay and the trainer:** a step's forward takes the sampled rows' vectors
  and masks alone, over stale scratch, and reports the loss of its outputs;
  held-out scoring is a direct forward's; a fixed batch overfits.
- **NNEvaluator:** its three entries give a direct forward's outputs, and the
  forward sees only the batch's vectors.
- **The eval server, with real worker processes:** what a worker sends is what
  the model receives, bit for bit after the fp16 round trip, a launch's
  padding zero; its outputs are ``NNEvaluator``'s on the same states,
  whichever workers a launch combines, at mixed player counts; bucketed
  launches are powers of two, often with fewer real states than the bucket,
  and dynamic ones the real states; its stats count them. A worker's
  one-state fill is its batch fill.
- **The warmup:** compiled, every bucket takes any real count on the warmup's
  graphs, with no graph break and no recompile.
- **Downstream:** self-play, replay, and training on one model, in-process
  and through an eval server and a worker.

``tests/g1889/test_pipeline_v2.py`` runs ``python -m train`` on v2.
"""

from __future__ import annotations

import multiprocessing as mp
from collections import Counter
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch._dynamo.testing import CompileCounterWithBackend
from torch._dynamo.utils import counters

from g1889.token_data import fill_inputs, fill_inputs_batch
from games import get_game
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool
from nn import create_model, get_model_input_spec
from train.config import TrainingConfig
from train.eval_server import (
    EvaluationServer,
    RemoteEvaluator,
    SharedEvalBuffers,
    _next_power_of_two,
    _warmup_compiled_model,
)
from train.gpu import detect_gpu
from train.profile_stats import BUILD, FORWARD, GATHER, IDLE
from train.replay_buffer import ReplayBuffer
from train.self_play import play_game, self_play_worker
from train.trainer import Trainer, policy_row_losses, value_row_sqerr

from tests.g1889.test_model import Sample
from tests.g1889.test_relation_transport import _get_result, _masks, _stack_rows
from tests.g1889.test_relation_transport import samples  # noqa: F401  (the fixture)

MODEL_PATH = "nn/transformer-1889-v2.py"
MAX_PLAYERS = 5
UNIFIED = 228
INPUT_SIZE = 5108  # at capacity 5 (token-data-1889.md, "Layout 3")

CONFIG = TrainingConfig(
    game="1889",
    model_path=MODEL_PATH,
    num_players=0,
    min_players=3,
    max_players=MAX_PLAYERS,
    d_model=64,
    num_heads=4,
    num_layers=3,
    zero_sum_values=True,
    optimizer="adamw",
    warmup_epochs=0,
    grad_clip=0,
    weight_decay=0,
    learning_rate=3e-3,
    lr_min=3e-3,
    seed=0,
)
GAME = get_game(CONFIG)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]

# fp32 against fp32 on the same inputs, whose batches may differ in size.
TOLERANCE = 2e-5
WIRE_TOLERANCE = 2e-3  # eval-server outputs against fp32 vectors, through the fp16 wire


def _model(device: str | torch.device = "cpu") -> torch.nn.Module:
    """A small v2 whose inputs matter: at initialization its bias multipliers
    are zero and its logits tiny, which would hide what these tests look for."""
    torch.manual_seed(0)
    model = create_model(CONFIG)
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.15)
    nn.init.normal_(model.get_parameter("relation_bias_mult"), std=0.5)
    nn.init.normal_(model.get_parameter("choice_phase_weight"), std=0.15)
    return model.to(device)


def _vectors(rows: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """The engine's layout-3 vectors of raw rows, fp32, filled from copies."""
    rows = [np.array(row, copy=True) for row in rows]
    out = np.full((len(rows), INPUT_SIZE), np.nan, np.float32)
    fill_inputs_batch(rows, out, MAX_PLAYERS, layout_version=3)
    return out


def _direct(model: torch.nn.Module, rows: Any, masks: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """A direct forward's softmaxed priors and values on the rows' vectors."""
    with torch.inference_mode():
        logits, values = model(
            torch.from_numpy(_vectors(rows)).to(device), torch.from_numpy(masks).bool().to(device),
        )
    return logits.softmax(dim=1).cpu().numpy(), values.cpu().numpy()


def _buffer(samples: list[Sample]) -> ReplayBuffer:
    """A full buffer of ``samples`` with random targets over their legal slots,
    and zero-sum value targets, as the model's values are."""
    rng = np.random.default_rng(0)
    masks = _masks(samples)
    targets = np.zeros_like(masks, np.float32)
    values = np.zeros((len(samples), MAX_PLAYERS), np.float32)
    for i, sample in enumerate(samples):
        slots = masks[i].nonzero()[0]
        targets[i, slots] = rng.dirichlet(np.ones(len(slots)))
        drawn = rng.uniform(-0.6, 0.6, sample.num_players)
        values[i, :sample.num_players] = drawn - drawn.mean()
    buffer = ReplayBuffer(
        GAME, len(samples), GAME.state_size(MAX_PLAYERS), MAX_PLAYERS, min_players=3, max_players=MAX_PLAYERS,
    )
    buffer.add_stacked(
        _stack_rows(samples), np.array([s.phase for s in samples], np.int8), masks, targets, values,
        player_counts=np.array([s.num_players for s in samples], np.uint8),
    )
    return buffer


# =============================================================================
# CONTRACTS
# =============================================================================

def test_layout_3_is_packed_with_no_relations():
    spec = get_model_input_spec(CONFIG)
    assert spec.packed and spec.input_shape == (INPUT_SIZE,)
    assert (spec.layout_version, spec.num_tokens, spec.token_dim, spec.input_size) == (3, 0, 0, INPUT_SIZE)
    assert GAME.packed_layouts == {3}
    assert [GAME.relation_format_of(version) for version in (1, 2, 3)] == ["flat", "flat", "none"]
    assert GAME.input_size is not None
    assert [GAME.input_size(capacity, layout_version=3) for capacity in (3, 4, 5)] == [4998, 5053, 5108]
    rss = get_game(TrainingConfig())
    assert rss.packed_layouts == frozenset() and rss.relation_format_of(3) == "dense"
    # The evaluator's spec, read off the model, is the config's.
    for capacity in (3, 4, 5):
        players = {"num_players": 3, "min_players": 0, "max_players": 0} if capacity == 3 else {"max_players": capacity}
        config = replace(CONFIG, d_model=32, num_layers=1, **players)
        evaluator = NNEvaluator(GAME, create_model(config), torch.device("cpu"), num_players=capacity)
        assert evaluator.input_spec == get_model_input_spec(config)
        assert evaluator.input_spec.input_size == GAME.input_size(capacity, layout_version=3)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        # A packed module on a token layout, and a token module on the packed one.
        ("INPUT_LAYOUT_VERSION = 2\ndef packed_input_size(max_players):\n    return 5108\n",
         "takes 5108 inputs a state at 5 players, but layout 2 of game '1889' is a token layout"),
        ("INPUT_LAYOUT_VERSION = 3\n",
         "takes token inputs a state at 5 players, but layout 3 of game '1889' is packed, 5108 values"),
    ],
)
def test_a_module_must_take_its_layout_s_inputs(tmp_path, source, message):
    path = tmp_path / "model.py"
    path.write_text(source + "def build_model(config, spec):\n    raise AssertionError\n")
    with pytest.raises(ValueError, match=message):
        get_model_input_spec(replace(CONFIG, model_path=str(path)))


def test_shared_buffers_carry_one_vector_a_state_and_no_relations():
    spec = get_model_input_spec(CONFIG)
    shared = SharedEvalBuffers(GAME, 2, batch_size=8, num_players=MAX_PLAYERS, input_spec=spec)
    assert shared.relation_format == "none" and shared._relation_coords is None
    assert shared.input_shape == (INPUT_SIZE,)
    wire = shared.get_input_states_np(1)
    assert wire.shape == (8, INPUT_SIZE) and wire.dtype == np.float16 and wire.nbytes == 8 * 10_216
    with pytest.raises(AssertionError, match="a packed layout sends no relations"):
        shared.get_input_relation_coords_np(0)
    # v1's buffers, on the same game, still carry records.
    v1 = SharedEvalBuffers(
        GAME, 1, batch_size=8, num_players=MAX_PLAYERS,
        input_spec=get_model_input_spec(replace(CONFIG, model_path="nn/transformer-1889-v1.py")),
    )
    assert v1.relation_format == "flat" and v1.get_input_relation_coords_np(0).shape == (8, 4096, 4)
    with pytest.raises(ValueError, match=r"input shape \(5000,\) does not match layout 3's \(5108,\)"):
        SharedEvalBuffers(GAME, 1, batch_size=8, num_players=MAX_PLAYERS, input_spec=replace(spec, input_size=5000))


# =============================================================================
# REPLAY AND THE TRAINER
# =============================================================================

class _ForwardSpy:
    """Each forward's arguments and outputs, copied."""

    def __init__(self, model: nn.Module) -> None:
        self.calls: list[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, torch.Tensor]]] = []
        model.register_forward_hook(self._record)

    def _record(self, _module: nn.Module, args: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        self.calls.append((
            tuple(arg.detach().cpu().clone() for arg in args),
            (output[0].detach().cpu().clone(), output[1].detach().cpu().clone()),
        ))


@pytest.mark.parametrize("device", DEVICES)
def test_a_trainer_step_takes_the_sampled_rows_vectors_and_reports_their_loss(samples, device):
    """Two steps, the second at a smaller batch over the first's scratch."""
    buffer = _buffer(samples)
    model = _model(device)
    spy = _ForwardSpy(model)
    trainer = Trainer(model, CONFIG, torch.device(device))
    rng = np.random.default_rng(3)
    # The buffer holds one pass of the samples, so a row's index is its write order.
    rows = buffer.rows_oldest_first()
    sampled: list[np.ndarray] = []
    sample_indices = buffer._sample_indices

    def spy_indices(*args: Any, **kwargs: Any) -> Any:
        indices, weights = sample_indices(*args, **kwargs)
        sampled.append(indices.copy())
        return indices, weights

    buffer._sample_indices = spy_indices  # type: ignore[method-assign]
    for batch in (48, 16):
        losses = trainer.train_step(buffer, batch, rng)
        (inputs, masks), (logits, values) = spy.calls[-1]
        drawn = sampled[-1]
        assert inputs.dtype == torch.float32 and tuple(inputs.shape) == (batch, INPUT_SIZE)
        np.testing.assert_array_equal(inputs.numpy(), _vectors(rows["states"][drawn]))
        np.testing.assert_array_equal(masks.numpy(), rows["legal_masks"][drawn].astype(bool))
        cross_entropy, _ = policy_row_losses(logits, torch.from_numpy(rows["policy_targets"][drawn]))
        sqerr, _ = value_row_sqerr(
            values, torch.from_numpy(rows["value_targets"][drawn]), torch.from_numpy(rows["player_counts"][drawn]),
        )
        assert losses["policy_loss"] == pytest.approx(float(cross_entropy.mean()), rel=1e-5)
        assert losses["value_loss"] == pytest.approx(
            float(sqerr.sum() / rows["player_counts"][drawn].sum()), rel=1e-5,
        )
    assert trainer._rel_h.numel() == 0  # no relation scratch


def test_held_out_scores_are_a_direct_forward_s(samples):
    buffer = _buffer(samples)
    model = _model()
    trainer = Trainer(model, CONFIG, torch.device("cpu"))
    rows = buffer.rows_oldest_first()
    halves = [{name: column[part] for name, column in rows.items()} for part in (slice(0, 90), slice(90, None))]
    cross_entropy, target_entropy, sqerr = trainer.score_batches(halves)
    assert model.training  # restored
    model.eval()
    with torch.no_grad():
        logits, values = model(torch.from_numpy(_vectors(rows["states"])), torch.from_numpy(rows["legal_masks"]).bool())
    expected_ce, expected_te = policy_row_losses(logits, torch.from_numpy(rows["policy_targets"]))
    expected_sq, _ = value_row_sqerr(
        values, torch.from_numpy(rows["value_targets"]), torch.from_numpy(rows["player_counts"]),
    )
    np.testing.assert_allclose(cross_entropy, expected_ce.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(target_entropy, expected_te.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sqerr, expected_sq.numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_the_trainer_overfits_a_fixed_batch(samples, device):
    """A buffer of one batch's rows: every step trains on all of them."""
    fixed = samples[::5]
    trainer = Trainer(_model(device), CONFIG, torch.device(device))
    buffer, rng = _buffer(fixed), np.random.default_rng(3)
    steps = [trainer.train_step(buffer, len(fixed), rng) for _ in range(250)]
    assert steps[-1]["policy_kl"] < 0.05 < steps[0]["policy_kl"]
    assert steps[-1]["value_loss"] < 0.02 < steps[0]["value_loss"]


# =============================================================================
# NNEVALUATOR
# =============================================================================

@pytest.mark.parametrize("device", DEVICES)
def test_nn_evaluator_gives_a_direct_forward_s_outputs(samples, device):
    model = _model(device).eval()
    evaluator = NNEvaluator(GAME, model, torch.device(device), num_players=MAX_PLAYERS)
    root = GAME.new_game(3, MAX_PLAYERS, 0)

    # One state at a time, through states that carry the config's flags, which
    # the samples' legal actions are under.
    for sample in samples[:: len(samples) // 12]:
        priors, values, actions, count, phase = evaluator.evaluate(GAME.scratch_state(root, sample.row.copy()))
        direct_priors, direct_values = _direct(model, [sample.row], _masks([sample]), device)
        slots = [GAME.policy.offsets[sample.phase] + action for action in actions]
        assert (phase, count, sorted(actions.tolist())) == (sample.phase, len(sample.legal), sorted(sample.legal))
        np.testing.assert_allclose(priors, direct_priors[0, slots], atol=TOLERANCE, rtol=TOLERANCE)
        np.testing.assert_allclose(values, direct_values[0, :sample.num_players], atol=TOLERANCE, rtol=TOLERANCE)

    # A batch of mixed player counts, then each count's leaves as search sends them.
    mixed = samples[::3][:40]
    direct_priors, direct_values = _direct(model, _stack_rows(mixed), _masks(mixed), device)
    states = [GAME.scratch_state(root, sample.row.copy()) for sample in mixed]
    for i, (priors, values, actions, _, _) in enumerate(evaluator.evaluate_batch(states)):
        slots = [GAME.policy.offsets[mixed[i].phase] + action for action in actions]
        np.testing.assert_allclose(priors, direct_priors[i, slots], atol=TOLERANCE, rtol=TOLERANCE)
        np.testing.assert_allclose(values, direct_values[i, :mixed[i].num_players], atol=TOLERANCE, rtol=TOLERANCE)
    for num_players in (3, 4, 5):
        count = [sample for sample in samples if sample.num_players == num_players][:32]
        priors, values = evaluator.evaluate_leaves(list(_stack_rows(count)), _masks(count))
        direct_priors, direct_values = _direct(model, _stack_rows(count), _masks(count), device)
        assert values.shape == (len(count), num_players)
        np.testing.assert_allclose(priors, direct_priors, atol=TOLERANCE, rtol=TOLERANCE)
        np.testing.assert_allclose(values, direct_values[:, :num_players], atol=TOLERANCE, rtol=TOLERANCE)


def test_nn_evaluator_sends_the_model_only_the_batch_s_vectors(samples):
    """Two arguments, the batch's rows of input scratch and no stale ones."""
    model = _model()
    spy = _ForwardSpy(model)
    evaluator = NNEvaluator(GAME, model, torch.device("cpu"), num_players=MAX_PLAYERS)
    count = [sample for sample in samples if sample.num_players == 4]
    for leaves in (count[:32], count[:8]):
        evaluator.evaluate_leaves(list(_stack_rows(leaves)), _masks(leaves))
        (inputs, masks), _ = spy.calls[-1]
        np.testing.assert_array_equal(inputs.numpy(), _vectors(_stack_rows(leaves)))
        np.testing.assert_array_equal(masks.numpy(), _masks(leaves).astype(bool))
    assert evaluator._scratch_cap == 32 and evaluator._rel_h.numel() == 0


# =============================================================================
# EVAL SERVER: REAL WORKER PROCESSES
# =============================================================================

def _wire_rounded_game() -> Any:
    """``GAME`` with its vectors rounded as workers' fp16 wire rounds them: the
    twin that leaves only the transport to compare."""
    def rounded(fill: Any) -> Any:
        def fill_rounded(rows: Any, out: np.ndarray, *args: Any, **kwargs: Any) -> None:
            fill(rows, out, *args, **kwargs)
            out[:] = out.astype(np.float16).astype(np.float32)
        return fill_rounded
    assert GAME.fill_inputs is not None and GAME.fill_inputs_batch is not None
    return replace(GAME, fill_inputs=rounded(GAME.fill_inputs), fill_inputs_batch=rounded(GAME.fill_inputs_batch))


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
    """A real model that also tells the test what each launch received."""

    def __init__(self, model: nn.Module, launches: Any) -> None:
        super().__init__()
        self.model = model
        self.launches = launches

    def forward(self, *args: torch.Tensor) -> Any:
        # A server that dies mustn't wait to flush launches nobody is reading.
        self.launches.cancel_join_thread()
        self.launches.put(tuple(arg.cpu().numpy().copy() for arg in args))
        return self.model(*args)


def _round_trip(
    samples: list[Sample], device: str, mode: str, leaves: int, stats_out: list[Any] | None = None,
) -> tuple[dict[int, tuple[list[list[Sample]], list[Sample]]], dict[int, Any], list[tuple[np.ndarray, ...]]]:
    """Three real workers, each of one player count, as search sends its states,
    through a real server. Returns each worker's requests and single states, its
    outputs, and each launch's arguments; ``stats_out`` gets the server's stats."""
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

    # Long requests and short ones, in shuffled phases: a short request's slot
    # still holds a long one's vectors.
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
        if stats_out is not None:
            stats_out.append(stats)
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5.0)
        server.stop()
    return requests, received, launches


def _sent(requests: dict[int, tuple[list[list[Sample]], list[Sample]]]) -> Counter[tuple[bytes, bytes]]:
    """Every state the workers sent, as its fp16 vector's and its mask's bytes."""
    sent: Counter[tuple[bytes, bytes]] = Counter()
    for parts, singles in requests.values():
        for part in [*parts, singles]:
            for vector, mask in zip(_vectors(_stack_rows(part)).astype(np.float16), _masks(part), strict=True):
                sent[vector.tobytes(), mask.astype(bool).tobytes()] += 1
    return sent


def _assert_the_model_received_what_workers_sent(
    requests: dict[int, tuple[list[list[Sample]], list[Sample]]], launches: list[tuple[np.ndarray, ...]], mode: str,
) -> None:
    """Each launch is (inputs, mask), fp16 vectors with no relations: its real
    rows exactly the states sent, each once over the launches, and its padding
    rows zero."""
    received: Counter[tuple[bytes, bytes]] = Counter()
    short = 0
    for launch in launches:
        assert len(launch) == 2
        inputs, masks = launch
        launch_n = len(inputs)
        assert inputs.dtype == np.float16 and inputs.shape == (launch_n, INPUT_SIZE) and masks.dtype == bool
        real = np.asarray(masks.any(axis=1))
        actual_n = int(real.sum())
        assert real[:actual_n].all()  # real rows first
        assert not inputs[actual_n:].any() and not masks[actual_n:].any()
        if mode == "bucketed":
            assert launch_n == _next_power_of_two(actual_n)
        else:
            assert launch_n == actual_n
        short += actual_n < launch_n
        for vector, mask in zip(inputs[:actual_n], masks[:actual_n], strict=True):
            received[vector.tobytes(), mask.tobytes()] += 1
    assert received == _sent(requests)
    # Bucketed launches pad: some have fewer real states than their bucket.
    assert (short > 0) == (mode == "bucketed")


@pytest.mark.parametrize(
    ("device", "mode", "leaves"),
    [
        ("cpu", "dynamic", 8),
        ("cpu", "bucketed", 8),
        # Launches of up to 120 rows in a bucket of 128.
        pytest.param("cuda", "bucketed", 40, marks=CUDA),
    ],
)
def test_eval_server_round_trip_gives_nn_evaluator_s_outputs(samples, device, mode, leaves):
    """Launches combine workers of three player counts, so each state's vector
    must find its row; and what the model received is what workers sent."""
    requests, received, launches = _round_trip(samples, device, mode, leaves)
    assert max(len(launch[0]) for launch in launches) > leaves
    _assert_the_model_received_what_workers_sent(requests, launches, mode)

    # The rounded twin leaves the transport; fp32 vectors also differ by the wire's rounding.
    model = _model(device).eval()
    device_t = torch.device(device)
    rounded = NNEvaluator(_wire_rounded_game(), model, device_t, num_players=MAX_PLAYERS)
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


def test_a_worker_s_single_state_fill_is_its_batch_fill(samples):
    """``evaluate`` writes the one-state fill, ``evaluate_leaves`` the batch
    fill, into the same wire: the same fp16 bytes for the same states."""
    root = GAME.new_game(4, MAX_PLAYERS, 0)
    single = np.empty(INPUT_SIZE, np.float32)
    rows = [sample.row for sample in samples[::7]]
    for row, batched in zip(rows, _vectors(rows), strict=True):
        fill_inputs(GAME.scratch_state(root, row.copy()), single, MAX_PLAYERS, layout_version=3)
        assert single.astype(np.float16).tobytes() == batched.astype(np.float16).tobytes()


def test_eval_server_stats_count_real_and_launched_rows_and_split_the_loop_s_time(samples):
    """Each launch's real rows are the states workers sent, its launched rows
    what the model saw; the stages and idle time make up the server's time."""
    stats_out: list[Any] = []
    requests, _, launches = _round_trip(samples, "cpu", "bucketed", 8, stats_out)
    stats, = stats_out
    sent = sum(_sent(requests).values())
    launched = [len(launch[0]) for launch in launches]
    assert stats.total_states == sum(stats.launch_real_rows.values()) == sent
    assert stats.launched_rows == sum(launched) > sent
    assert stats.launch_counts == {n: launched.count(n) for n in set(launched)}
    assert all(stats.launch_real_rows[n] <= n * count for n, count in stats.launch_counts.items())
    assert all(secs >= 0 for secs in stats.stage_secs)
    assert sum(stats.stage_secs) == pytest.approx(stats.wall_secs, rel=0.01)
    assert stats.stage_secs[FORWARD] > 0 and stats.stage_secs[GATHER] > 0 and stats.stage_secs[IDLE] > 0
    assert stats.stage_secs[BUILD] > 0


# =============================================================================
# THE EVAL SERVER'S WARMUP
# =============================================================================

def test_warmup_leaves_every_bucket_on_its_graphs(samples):
    """After the bucketed warmup every launch runs on its graphs, whatever its
    bucket and real count, with the serve loop's two arguments and fp16
    inputs: no graph breaks, and no recompile.

    Dynamic mode's ``mark_unbacked`` batch does not trace on the CPU, for v1 as
    for v2: the CPU attention kernel guards on a batch of 1. 1889's runs are
    bucketed; dynamic mode is the 5090's to check.
    """
    spec = get_model_input_spec(CONFIG)
    shared_bufs = SharedEvalBuffers(GAME, 1, batch_size=8, num_players=MAX_PLAYERS, input_spec=spec)
    model = _model().eval()
    backend = CompileCounterWithBackend("aot_eager")
    compiled = cast(torch.nn.Module, torch.compile(model, backend=backend, fullgraph=True))
    counters.clear()
    try:
        _warmup_compiled_model(
            compiled, shared_bufs, torch.device("cpu"), batch_shape_mode="bucketed", max_launch_batch=8,
            autocast_dtype=None, input_dtype=torch.float16,
        )
        warmup_frames = backend.frame_count
        assert warmup_frames == 2  # batch 1, then the dynamic graph from batch 2

        # The serve loop's buffers: allocated once, outside inference mode.
        gpu_s = torch.empty(8, INPUT_SIZE, dtype=torch.float16)
        gpu_mask = torch.zeros(8, UNIFIED, dtype=torch.bool)
        for number, (launch, size) in enumerate(((8, 8), (4, 3), (8, 5), (2, 2), (1, 1), (4, 4), (2, 1))):
            picks = [samples[(7 * number + 11 * i) % len(samples)] for i in range(size)]
            gpu_s[:size] = torch.from_numpy(_vectors(_stack_rows(picks)).astype(np.float16))
            gpu_s[size:launch] = 0
            gpu_mask[:size] = torch.from_numpy(_masks(picks)).bool()
            gpu_mask[size:launch] = False
            # The serve loop's persistent context, which the warmup matches.
            with torch.inference_mode(), torch.autocast("cpu", enabled=False, cache_enabled=False):
                logits, values = compiled(gpu_s[:launch], gpu_mask[:launch])
                eager_logits, eager_values = model(gpu_s[:size].float(), gpu_mask[:size])
            torch.testing.assert_close(logits[:size], eager_logits, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(values[:size], eager_values, rtol=1e-4, atol=1e-4)
        assert backend.frame_count == warmup_frames
        assert not counters["graph_break"]
    finally:
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
        GAME, config.buffer_capacity, GAME.state_size(MAX_PLAYERS), MAX_PLAYERS, min_players=3, max_players=MAX_PLAYERS,
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
        epoch_config=config.compute_epoch_config(0), num_players=4,
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
        batch_shape_mode="bucketed", max_batch_size=2,
    )
    tasks, results = ctx.Queue(), ctx.Queue()
    worker = ctx.Process(target=self_play_worker, args=(tasks, results, config, shared_bufs, 0), daemon=True)
    try:
        server.start()
        assert server.wait_ready(timeout=120.0)
        worker.start()
        tasks.put((0, 1, 5, config.compute_epoch_config(0)))
        record = _get_result(results, server, [worker])
        tasks.put(None)
        worker.join(timeout=10.0)
        assert not worker.is_alive()
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5.0)
        server.stop()
    assert record.num_players == 5
    _assert_records_train(config, model, [record])
