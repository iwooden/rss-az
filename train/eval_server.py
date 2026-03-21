"""Centralized NN evaluation server for multi-process self-play.

Each EvaluationServer runs as a separate process with its own Python GIL,
eliminating GIL contention that limited throughput when servers were threads.
The model's GPU parameter tensors are shared zero-copy via CUDA IPC
(torch.multiprocessing), so optimizer.step() in the main process is visible
to eval servers automatically.

Communication uses shared memory (torch tensors with share_memory_()):
- Input states: float32 (workers write with pure numpy slice assignment)
- Output logits: bfloat16 (halves scatter bandwidth; workers upcast for softmax)
- Output values: float32 (server upcasts from bf16 model output during D2H;
  only 3 floats/state so bf16 savings are negligible vs per-worker conversion)

A multiprocessing.Queue carries lightweight request tuples (worker_idx,
state_count), and per-worker Events signal completion.

**Multi-server concurrency:**

Each EvaluationServer process has its own GIL, CUDA context, and default
stream. Servers race on the shared queue without a lock — after completing
a forward pass, each server eagerly drains all pending requests via
get_nowait() and immediately starts the next batch. This creates organic
alternation: one server computes while the other gathers.

Workers do zero torch operations on the write side (pure numpy).
On the read side, workers do one bf16→f32 upcast on logits for
batched mask+softmax in torch. Values are read as f32 numpy directly.

RemoteEvaluator is the worker-side proxy that implements the same interface
as NNEvaluator, writing to shared memory instead of serializing over pipes.
"""

from __future__ import annotations

import copy
import queue as _queue
from time import perf_counter
from typing import Any

import numpy as np
import torch

from mcts.evaluator import (
    compute_terminal_values,
    get_layout,
    rotate_visible_state_into,
    unrotate_values,
)
from train.profile_stats import EvalClientStats, EvalServerStats


class SharedEvalBuffers:
    """Pre-allocated shared memory for zero-copy worker <-> server communication.

    Input states are float32 so workers can write with pure numpy slice
    assignment (no torch overhead).  Output logits are bfloat16 to halve
    scatter bandwidth.  Output values are float32 — the model outputs bf16
    under autocast, but the server upcasts during D2H copy.  With only
    3 floats/state the bf16 savings are negligible, and this avoids a
    per-worker bf16→f32 conversion.

    Each worker gets a fixed slot in shared tensors. Workers write rotated
    states into their input slot; the server reads them directly. The server
    writes raw logits and values into each worker's output slot. Workers apply
    legal action masking and softmax locally after reading results.

    Memory layout (per worker):
        Input:  states  float32 (batch_size x visible_size)
        Output: logits  bfloat16 (batch_size x action_dim)
                values  float32  (batch_size x num_players)
    """

    def __init__(
        self,
        num_workers: int,
        batch_size: int,
        visible_size: int,
        action_dim: int,
        num_players: int,
    ) -> None:
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.visible_size = visible_size
        self.action_dim = action_dim
        self.num_players = num_players

        # Input: float32 — workers write with pure numpy slice assignment
        self._states = torch.zeros(
            num_workers, batch_size, visible_size, dtype=torch.float32,
        ).share_memory_()
        # Output: bfloat16 — halves scatter bandwidth
        self._logits = torch.zeros(
            num_workers, batch_size, action_dim, dtype=torch.bfloat16,
        ).share_memory_()
        self._values = torch.zeros(
            num_workers, batch_size, num_players, dtype=torch.float32,
        ).share_memory_()

    def get_input_states_np(self, worker_idx: int) -> np.ndarray:
        """Numpy view into worker's float32 input state slot.

        Creates the view on-demand rather than caching in __init__, because
        numpy views don't survive pickling across spawn boundaries — they'd
        become detached copies instead of shared memory views.
        """
        return self._states[worker_idx].numpy()

    def get_output_logits(self, worker_idx: int) -> torch.Tensor:
        """bfloat16 tensor view into worker's output logit slot."""
        return self._logits[worker_idx]

    def get_output_values(self, worker_idx: int) -> torch.Tensor:
        """float32 tensor view into worker's output value slot."""
        return self._values[worker_idx]


def _eval_server_main(
    model: torch.nn.Module,
    device: torch.device,
    shared_bufs: SharedEvalBuffers,
    request_queue: Any,
    worker_events: list[Any],
    stop_event: Any,
    stats_report_event: Any,
    stats_queue: Any,
    *,
    server_id: int,
    profile: bool,
    no_compile: bool,
) -> None:
    """Eval server process entry point.

    Runs batched GPU inference in a loop, consuming requests from the shared
    queue and writing results back to shared memory. Each process has its own
    GIL and CUDA default stream, so multiple servers truly overlap.
    """
    try:
        _eval_server_serve(
            model, device, shared_bufs, request_queue, worker_events,
            stop_event, stats_report_event, stats_queue,
            server_id=server_id,
            profile=profile, no_compile=no_compile,
        )
    except Exception:
        import traceback
        traceback.print_exc()


def _eval_server_serve(
    model: torch.nn.Module,
    device: torch.device,
    shared_bufs: SharedEvalBuffers,
    request_queue: Any,
    worker_events: list[Any],
    stop_event: Any,
    stats_report_event: Any,
    stats_queue: Any,
    *,
    server_id: int,
    profile: bool,
    no_compile: bool,
) -> None:
    """Inner serve loop for an eval server process."""
    # Prevent OpenMP oversubscription (same as worker processes)
    torch.set_num_threads(1)

    model.eval()

    use_cuda = device.type == "cuda"

    # Optionally compile the model (per-process compilation)
    if not no_compile and use_cuda:
        model = torch.compile(model, dynamic=True)  # type: ignore[assignment]
        # Warmup: compile kernels before serving real requests
        model.eval()
        vis = shared_bufs.visible_size
        max_warmup = max(1, shared_bufs.num_workers * shared_bufs.batch_size)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            for warmup_bs in (1, max_warmup):
                dummy = torch.randn(warmup_bs, vis, device=device)
                model(dummy)
                del dummy
        if use_cuda:
            torch.cuda.synchronize()

    # Allocate pinned CPU buffers and GPU tensors in this process's CUDA context
    max_batch = shared_bufs.num_workers * shared_bufs.batch_size
    vis = shared_bufs.visible_size
    act = shared_bufs.action_dim
    npl = shared_bufs.num_players

    pin_s = torch.empty(max_batch, vis, dtype=torch.float32, pin_memory=use_cuda)
    pin_s_np = pin_s.numpy()
    gpu_s = torch.empty(max_batch, vis, dtype=torch.float32, device=device)
    pin_log = torch.empty(max_batch, act, dtype=torch.bfloat16, pin_memory=use_cuda)
    pin_val = torch.empty(max_batch, npl, dtype=torch.float32, pin_memory=use_cuda)

    stats: EvalServerStats | None = EvalServerStats() if profile else None
    _tp = _ti = 0.0

    # Cache per-worker shared memory views
    w_states_np = [shared_bufs.get_input_states_np(i) for i in range(shared_bufs.num_workers)]
    w_logits = [shared_bufs.get_output_logits(i) for i in range(shared_bufs.num_workers)]
    w_values = [shared_bufs.get_output_values(i) for i in range(shared_bufs.num_workers)]

    events = worker_events

    while not stop_event.is_set():
        if stats is not None:
            _tp = perf_counter()

        # Eagerly drain all pending requests without blocking.
        # Only fall back to a blocking get() when we have nothing,
        # to avoid busy-spinning when idle.
        batch_info: list[tuple[int, int]] = []
        while len(batch_info) < max_batch:
            try:
                batch_info.append(request_queue.get_nowait())
            except _queue.Empty:
                break

        if not batch_info:
            # Nothing pending — block briefly to avoid busy-spin
            try:
                batch_info.append(request_queue.get(timeout=0.01))
            except _queue.Empty:
                if stats is not None:
                    stats.record_idle(perf_counter() - _tp)
                # Check if main process wants stats
                if stats_report_event is not None and stats_report_event.is_set():
                    stats_queue.put(copy.copy(stats))
                    if stats is not None:
                        stats.reset()
                    stats_report_event.clear()
                continue
            # Got one — drain any more that arrived
            while len(batch_info) < max_batch:
                try:
                    batch_info.append(request_queue.get_nowait())
                except _queue.Empty:
                    break

        # Gather f32 states from numpy shared memory into pinned numpy
        total_n = 0
        for widx, n in batch_info:
            pin_s_np[total_n:total_n + n] = w_states_np[widx][:n]
            total_n += n

        if stats is not None:
            _ti = perf_counter()

        # --- GPU pipeline (default stream, no stream context needed) ---
        gpu_s_batch = gpu_s[:total_n]
        gpu_s_batch.copy_(pin_s[:total_n], non_blocking=True)
        with torch.no_grad():
            if use_cuda:
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    policy_logits, values = model(gpu_s_batch)
            else:
                policy_logits, values = model(gpu_s_batch)
            pin_log[:total_n].copy_(
                policy_logits.bfloat16(), non_blocking=True,
            )
            pin_val[:total_n].copy_(values, non_blocking=True)
            if use_cuda:
                torch.cuda.synchronize()

        if stats is not None:
            stats.record_batch(total_n, perf_counter() - _ti)

        # Scatter results to per-worker shared memory and signal completion
        offset = 0
        for widx, n in batch_info:
            w_logits[widx][:n].copy_(pin_log[offset:offset + n])
            w_values[widx][:n].copy_(pin_val[offset:offset + n])
            events[widx].set()
            offset += n

        # Check if main process wants stats (also check in busy path)
        if stats_report_event is not None and stats_report_event.is_set():
            stats_queue.put(copy.copy(stats))
            if stats is not None:
                stats.reset()
            stats_report_event.clear()


class EvaluationServer:
    """Process-based centralized NN evaluator using shared memory.

    Consumes (worker_idx, state_count) requests from a shared queue,
    gathers states from shared memory, runs batched inference, writes
    results back, and signals workers via per-worker Events.

    Each EvaluationServer runs in its own process with a separate GIL,
    eliminating GIL contention between servers. The model's GPU parameter
    tensors are shared via CUDA IPC (torch.multiprocessing), so weight
    updates from the trainer are visible automatically.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        shared_bufs: SharedEvalBuffers,
        request_queue: Any,
        worker_events: list[Any],
        *,
        server_id: int = 0,
        profile: bool = False,
        mp_context: Any = None,
        no_compile: bool = False,
    ) -> None:
        import multiprocessing
        ctx = mp_context or multiprocessing
        self._stop_event = ctx.Event()
        self._stats_report_event: Any = ctx.Event() if profile else None
        self._stats_queue: Any = ctx.Queue() if profile else None
        self._process: Any | None = None
        self._process_args = (
            model, device, shared_bufs, request_queue, worker_events,
            self._stop_event, self._stats_report_event, self._stats_queue,
        )
        self._process_kwargs = {
            "server_id": server_id,
            "profile": profile,
            "no_compile": no_compile,
        }
        self._mp_context = ctx
        self._server_id = server_id

    def start(self) -> None:
        """Start the server process."""
        self._stop_event.clear()
        p = self._mp_context.Process(
            target=_eval_server_main,
            args=self._process_args,
            kwargs=self._process_kwargs,
            daemon=True,
        )
        p.start()
        self._process = p

    def stop(self) -> None:
        """Signal the server to stop and wait for it."""
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    def get_profile_stats(self) -> EvalServerStats | None:
        """Request stats from the server process and return them.

        Signals the server to report, then reads from the shared queue.
        Returns None if profiling is disabled or the server doesn't respond.
        """
        if self._stats_report_event is None:
            return None
        self._stats_report_event.set()
        try:
            return self._stats_queue.get(timeout=2.0)
        except _queue.Empty:
            return None

    def reset_profile_stats(self) -> None:
        """Reset profile stats for a new epoch.

        Stats are reset inside the server after reporting.
        Drain any leftover stats from previous collection.
        """
        if self._stats_queue is not None:
            while not self._stats_queue.empty():
                try:
                    self._stats_queue.get_nowait()
                except _queue.Empty:
                    break


class RemoteEvaluator:
    """Worker-side proxy that evaluates states via shared memory + EvaluationServer.

    Implements the same evaluate/evaluate_batch/evaluate_terminal interface
    as NNEvaluator, so it can be used as a drop-in replacement.

    Write side: pure numpy slice assignment into f32 shared memory (no torch).
    Read side: logits bf16→f32 for batched mask+softmax in torch;
    values are f32 in shared memory (server upcasts during D2H).
    A Queue carries request tuples and per-worker Events signal completion.

    Invariant: each worker may have at most one outstanding eval request at a
    time.  Output slots are keyed by worker_idx alone (no request id), so a
    second request before the first completes would overwrite the output buffer.
    The sequential clear → put → wait → read flow enforces this.
    """

    # Seconds to wait for eval server response before raising.
    _EVAL_TIMEOUT = 30.0

    def __init__(
        self,
        num_players: int,
        shared_bufs: SharedEvalBuffers,
        worker_idx: int,
        request_queue: Any,
        done_event: Any,
        *,
        profile: bool = False,
        terminal_rank_weight: float = 0.5,
    ) -> None:
        self.num_players = num_players
        self.terminal_rank_weight = terminal_rank_weight
        self.layout = get_layout(num_players)
        self._worker_idx = worker_idx
        self._in_states_np = shared_bufs.get_input_states_np(worker_idx)
        self._out_logits = shared_bufs.get_output_logits(worker_idx)
        self._out_values = shared_bufs.get_output_values(worker_idx)
        self._queue = request_queue
        self._event = done_event
        self._profile = profile
        self._stats: EvalClientStats | None = EvalClientStats() if profile else None

    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a single state via the remote server."""
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()
        # Write rotated state directly into shared memory — no intermediate
        rotate_visible_state_into(
            self._in_states_np[0], state._array, active_player, self.num_players
        )
        mask = get_valid_action_mask(state)

        self._event.clear()
        self._queue.put((self._worker_idx, 1))
        if not self._event.wait(timeout=self._EVAL_TIMEOUT):
            raise RuntimeError(
                f"Eval server did not respond within {self._EVAL_TIMEOUT}s "
                f"(worker {self._worker_idx})"
            )

        # Logits: bf16 → f32 for mask+softmax; values: already f32
        logits = self._out_logits[0].float()
        logits.masked_fill_(torch.from_numpy(mask) <= 0, -1e9)
        policy_probs = torch.softmax(logits, dim=-1).numpy()
        canonical = unrotate_values(
            self._out_values[0].numpy(), active_player,
        )
        return policy_probs, canonical, mask

    def evaluate_batch(
        self,
        states: list[Any],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate multiple states in a single round-trip to the server."""
        from core.actions import get_valid_action_mask

        n = len(states)
        if n == 0:
            return []

        active_ids = [s.get_active_player() for s in states]

        # Write rotated states directly into shared memory — no intermediates
        masks_list = []
        for i, (s, ap) in enumerate(zip(states, active_ids)):
            rotate_visible_state_into(
                self._in_states_np[i], s._array, ap, self.num_players
            )
            masks_list.append(get_valid_action_mask(s))

        self._event.clear()
        self._queue.put((self._worker_idx, n))
        if not self._event.wait(timeout=self._EVAL_TIMEOUT):
            raise RuntimeError(
                f"Eval server did not respond within {self._EVAL_TIMEOUT}s "
                f"(worker {self._worker_idx}, batch {n})"
            )

        # Logits: bf16→f32 for mask+softmax; values: already f32
        logits_f32 = self._out_logits[:n].float()
        masks_t = torch.from_numpy(np.stack(masks_list))
        logits_f32.masked_fill_(masks_t <= 0, -1e9)
        probs_batch = torch.softmax(logits_f32, dim=-1).numpy()
        values_np = self._out_values[:n].numpy()

        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical = unrotate_values(values_np[i], active_ids[i])
            results.append((probs_batch[i].copy(), canonical, masks_list[i]))
        return results

    def evaluate_leaves(
        self,
        state_arrays: list[np.ndarray],
        active_player_ids: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Evaluate pre-computed leaf data in a single round-trip to the server.

        Returns raw logits — the caller applies masked softmax using the
        legal masks it already has on each node.
        """
        n = len(state_arrays)
        if n == 0:
            return []

        _stats = self._stats
        _t0 = _t1 = _t2 = 0.0
        if _stats is not None:
            _t0 = perf_counter()

        # Write rotated states directly into shared memory — no intermediates
        in_np = self._in_states_np
        for i, (arr, ap) in enumerate(zip(state_arrays, active_player_ids)):
            rotate_visible_state_into(in_np[i], arr, ap, self.num_players)

        if _stats is not None:
            _t1 = perf_counter()
            _stats.prepare_secs += _t1 - _t0

        # Signal server and wait for completion
        self._event.clear()
        self._queue.put((self._worker_idx, n))
        if not self._event.wait(timeout=self._EVAL_TIMEOUT):
            raise RuntimeError(
                f"Eval server did not respond within {self._EVAL_TIMEOUT}s "
                f"(worker {self._worker_idx}, batch {n})"
            )

        if _stats is not None:
            _t2 = perf_counter()
            _stats.wait_secs += _t2 - _t1

        # Logits: bf16→f32; values: already f32
        logits_np = self._out_logits[:n].float().numpy()
        values_np = self._out_values[:n].numpy()

        results: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical = unrotate_values(values_np[i], active_player_ids[i])
            results.append((logits_np[i].copy(), canonical))

        if _stats is not None:
            _stats.result_secs += perf_counter() - _t2
            _stats.num_calls += 1
            _stats.total_states += n

        return results

    def reset_profile_stats(self) -> None:
        """Reset profile stats for a new game."""
        if self._profile:
            self._stats = EvalClientStats()

    def get_profile_stats(self) -> EvalClientStats | None:
        """Return accumulated profile stats (None if profiling disabled)."""
        return self._stats

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values locally (no NN needed)."""
        net_worths = [
            state.get_player_net_worth(i) for i in range(self.num_players)
        ]
        return compute_terminal_values(
            net_worths, self.num_players, self.terminal_rank_weight
        )
