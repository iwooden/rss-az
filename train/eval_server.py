"""Centralized NN evaluation server for multi-process self-play.

Each EvaluationServer runs as a separate process with its own Python GIL,
eliminating GIL contention that limited throughput when servers were threads.
The model's GPU parameter tensors are shared zero-copy via CUDA IPC
(torch.multiprocessing), so optimizer.step() in the main process is visible
to eval servers automatically.

Communication uses shared memory (torch tensors with share_memory_()):
- Input states: float32 (workers write with pure numpy slice assignment)
- Output logits: bfloat16 (halves scatter bandwidth; workers upcast for Cython softmax)
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
On the read side, workers upcast bf16→f32 logits and apply mask+softmax
via Cython (no torch on the worker hot path). Values are read as f32
numpy directly.

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
    BaseEvaluator,
    rotate_visible_state_into,
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

        # Shared-memory signaling (replaces mp.Queue)
        # flags: per-worker int32 — 0=IDLE, 1=SUBMITTED (server scans these)
        # counts: per-worker int32 — number of states in current request
        self._flags = torch.zeros(num_workers, dtype=torch.int32).share_memory_()
        self._counts = torch.zeros(num_workers, dtype=torch.int32).share_memory_()

        # Per-worker done events for efficient sleep-wake (replaces spin-wait).
        # Created lazily via init_done_events() with the correct mp context.
        self.done_events: list[Any] | None = None

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

    def init_done_events(self, mp_context: Any = None) -> None:
        """Create per-worker done Events using the given mp context.

        Must be called before passing SharedEvalBuffers to server/worker
        processes. Events are used to wake workers after inference completes
        (efficient kernel sleep instead of spin-waiting).
        """
        import multiprocessing
        ctx = mp_context or multiprocessing
        self.done_events = [ctx.Event() for _ in range(self.num_workers)]

    def get_signal_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (flags, counts) numpy int32 views for shared-memory signaling.

        Creates views on-demand (numpy views don't survive spawn pickling).
        """
        return self._flags.numpy(), self._counts.numpy()


def _eval_server_main(
    model: torch.nn.Module,
    device: torch.device,
    shared_bufs: SharedEvalBuffers,
    stop_event: Any,
    ready_event: Any,
    stats_report_event: Any,
    stats_queue: Any,
    *,
    server_id: int,
    worker_start: int,
    worker_end: int,
    profile: bool,
    no_compile: bool,
    compile_kwargs: dict[str, Any] | None = None,
) -> None:
    """Eval server process entry point.

    Runs batched GPU inference in a loop, scanning shared-memory flags for
    requests from its assigned worker partition [worker_start, worker_end).
    Each process has its own GIL and CUDA default stream, so multiple
    servers truly overlap.
    """
    try:
        _eval_server_serve(
            model, device, shared_bufs,
            stop_event, ready_event, stats_report_event, stats_queue,
            server_id=server_id,
            worker_start=worker_start, worker_end=worker_end,
            profile=profile, no_compile=no_compile,
            compile_kwargs=compile_kwargs,
        )
    except Exception:
        import traceback
        traceback.print_exc()


def _eval_server_serve(
    model: torch.nn.Module,
    device: torch.device,
    shared_bufs: SharedEvalBuffers,
    stop_event: Any,
    ready_event: Any,
    stats_report_event: Any,
    stats_queue: Any,
    *,
    server_id: int,
    worker_start: int,
    worker_end: int,
    profile: bool,
    no_compile: bool,
    compile_kwargs: dict[str, Any] | None = None,
) -> None:
    """Inner serve loop for an eval server process.

    Scans shared-memory flags for assigned worker partition [worker_start,
    worker_end) instead of using mp.Queue. Communication is fully lockfree:
    workers set flag=SUBMITTED, server scans and sets flag=DONE.
    """
    import time as _time

    # Prevent OpenMP oversubscription (same as worker processes)
    torch.set_num_threads(1)

    model.eval()

    use_cuda = device.type == "cuda"

    # Apply NVIDIA-specific per-process settings (TF32 etc.) if active.
    # Detected by checking compile_kwargs for reduce-overhead mode.
    if compile_kwargs and compile_kwargs.get("mode") == "reduce-overhead":
        from train.nvidia import apply_nvidia_optimizations
        apply_nvidia_optimizations()

    # Optionally compile the model (per-process compilation).
    if not no_compile and use_cuda:
        ckw = compile_kwargs if compile_kwargs else {"dynamic": True}
        model = torch.compile(model, **ckw)  # type: ignore[assignment]
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            dummy = torch.randn(1, shared_bufs.visible_size, device=device)
            model(dummy)
            del dummy
        torch.cuda.synchronize()

    # Signal that this server is ready to serve requests
    ready_event.set()

    # Allocate pinned CPU buffers and GPU tensors in this process's CUDA context
    partition_size = worker_end - worker_start
    max_batch = partition_size * shared_bufs.batch_size
    vis = shared_bufs.visible_size
    act = shared_bufs.action_dim
    npl = shared_bufs.num_players

    pin_s = torch.empty(max_batch, vis, dtype=torch.float32, pin_memory=use_cuda)
    pin_s_np = pin_s.numpy()
    gpu_s = torch.empty(max_batch, vis, dtype=torch.float32, device=device)
    pin_log = torch.empty(max_batch, act, dtype=torch.bfloat16, pin_memory=use_cuda)
    pin_log_bytes = pin_log.view(torch.int8).numpy()  # byte view for Cython scatter
    pin_val = torch.empty(max_batch, npl, dtype=torch.float32, pin_memory=use_cuda)
    pin_val_np = pin_val.numpy()

    stats: EvalServerStats | None = EvalServerStats() if profile else None
    _tp = _ti = 0.0

    # Contiguous 3D views for Cython gather/scatter (no per-worker Python loop)
    from mcts.mcts_core import (
        gather_states as _gather_states,
        scatter_results as _scatter_results,
        server_scan as _server_scan,
        server_signal_done as _server_signal_done,
    )
    # states: (num_workers, batch_size, visible_size) f32 numpy
    all_states_np = shared_bufs._states.numpy()
    # logits: (num_workers, batch_size, action_dim) — view as bytes for bf16
    all_logits_bytes = shared_bufs._logits.view(torch.int8).numpy()
    # values: (num_workers, batch_size, num_players) f32 numpy
    all_values_np = shared_bufs._values.numpy()
    logit_row_bytes = act * 2  # bf16 = 2 bytes per element

    # Shared-memory flag and count arrays
    sig_flags, sig_counts = shared_bufs.get_signal_arrays()
    # Per-worker done events for waking workers after inference
    assert shared_bufs.done_events is not None
    done_events = shared_bufs.done_events

    # Pre-allocate request arrays (filled each scan, avoids per-batch allocation)
    _widx_buf = np.empty(partition_size, dtype=np.int32)
    _cnt_buf = np.empty(partition_size, dtype=np.int32)

    # Number of empty scans before sleeping (avoid busy-spin when idle).
    # Each scan of 48 workers is ~1us; 50 empty scans = ~50us before sleeping.
    _EMPTY_SCAN_LIMIT = 50
    empty_scans = 0

    while not stop_event.is_set():
        if stats is not None:
            _tp = perf_counter()

        # Accumulate requests: keep scanning until a scan returns nothing,
        # then process whatever we've collected. This naturally batches
        # concurrent arrivals for better GPU utilization.
        num_requests = 0
        while num_requests < partition_size:
            n = _server_scan(
                sig_flags, sig_counts,
                _widx_buf[num_requests:], _cnt_buf[num_requests:],
                worker_start, worker_end, partition_size - num_requests,
            )
            if n == 0:
                break
            num_requests += n

        if num_requests == 0:
            empty_scans += 1
            if empty_scans >= _EMPTY_SCAN_LIMIT:
                # Brief sleep to avoid burning CPU when truly idle.
                _time.sleep(0.0001)  # 100us
                empty_scans = 0
            if stats is not None:
                stats.record_idle(perf_counter() - _tp)
            # Check if main process wants stats
            if stats_report_event is not None and stats_report_event.is_set():
                stats_queue.put(copy.copy(stats))
                if stats is not None:
                    stats.reset()
                stats_report_event.clear()
            continue

        empty_scans = 0

        # Gather f32 states from per-worker shared memory into contiguous pinned buffer.
        total_n = _gather_states(
            pin_s_np, all_states_np,
            _widx_buf[:num_requests], _cnt_buf[:num_requests], num_requests,
        )

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

        # Scatter results to per-worker shared memory (Cython nogil memcpy),
        # then signal DONE via atomic flag writes (also Cython nogil).
        _scatter_results(
            pin_log_bytes, pin_val_np,
            all_logits_bytes, all_values_np,
            _widx_buf[:num_requests], _cnt_buf[:num_requests], num_requests,
            logit_row_bytes,
        )
        _server_signal_done(sig_flags, _widx_buf[:num_requests], num_requests)
        # Wake workers via kernel event (efficient sleep, no spin-wait)
        for i in range(num_requests):
            done_events[_widx_buf[i]].set()

        # Check if main process wants stats (also check in busy path)
        if stats_report_event is not None and stats_report_event.is_set():
            stats_queue.put(copy.copy(stats))
            if stats is not None:
                stats.reset()
            stats_report_event.clear()


class EvaluationServer:
    """Process-based centralized NN evaluator using shared memory.

    Each server owns a contiguous partition of workers [worker_start,
    worker_end) and only scans those workers' flags. Communication is
    fully lockfree via shared-memory flags — no mp.Queue or mp.Event.

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
        *,
        server_id: int = 0,
        worker_start: int = 0,
        worker_end: int | None = None,
        profile: bool = False,
        mp_context: Any = None,
        no_compile: bool = False,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        import multiprocessing
        ctx = mp_context or multiprocessing
        if worker_end is None:
            worker_end = shared_bufs.num_workers
        self._stop_event = ctx.Event()
        self._ready_event = ctx.Event()
        self._stats_report_event: Any = ctx.Event() if profile else None
        self._stats_queue: Any = ctx.Queue() if profile else None
        self._process: Any | None = None
        self._process_args = (
            model, device, shared_bufs,
            self._stop_event, self._ready_event,
            self._stats_report_event, self._stats_queue,
        )
        self._process_kwargs = {
            "server_id": server_id,
            "worker_start": worker_start,
            "worker_end": worker_end,
            "profile": profile,
            "no_compile": no_compile,
            "compile_kwargs": compile_kwargs,
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

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Block until the server has finished compilation and warmup.

        Returns True if the server signaled ready, False on timeout.
        """
        return self._ready_event.wait(timeout=timeout)

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


class RemoteEvaluator(BaseEvaluator):
    """Worker-side proxy that evaluates states via shared memory + EvaluationServer.

    Implements the same evaluate/evaluate_batch/evaluate_terminal interface
    as NNEvaluator, so it can be used as a drop-in replacement.

    Write side: pure numpy slice assignment into f32 shared memory (no torch).
    Read side: logits bf16→f32 for Cython mask+softmax (no torch on
    read path); values are f32 in shared memory (server upcasts during D2H).

    Communication is lockfree via shared-memory flags: worker sets
    flag=SUBMITTED, spins until flag=DONE, then reads results and resets
    flag=IDLE. No mp.Queue or mp.Event involved.

    Invariant: each worker may have at most one outstanding eval request at a
    time. Output slots are keyed by worker_idx alone (no request id), so a
    second request before the first completes would overwrite the output buffer.
    The sequential submit → spin-wait → read → idle flow enforces this.
    """

    # Seconds to wait for eval server response before raising.
    _EVAL_TIMEOUT = 30.0

    def __init__(
        self,
        num_players: int,
        shared_bufs: SharedEvalBuffers,
        worker_idx: int,
        *,
        profile: bool = False,
        terminal_rank_weight: float = 0.5,
    ) -> None:
        super().__init__(num_players, terminal_rank_weight)
        self._worker_idx = worker_idx
        self._in_states_np = shared_bufs.get_input_states_np(worker_idx)
        self._out_logits = shared_bufs.get_output_logits(worker_idx)
        self._out_values = shared_bufs.get_output_values(worker_idx)
        self._profile = profile
        self._stats: EvalClientStats | None = EvalClientStats() if profile else None

        # Shared-memory signaling: flags for lockfree request submission,
        # mp.Event for efficient done notification (kernel sleep, no spin).
        from mcts.mcts_core import worker_signal_submit, worker_reset_idle
        self._sig_flags, self._sig_counts = shared_bufs.get_signal_arrays()
        self._signal_submit = worker_signal_submit
        self._reset_idle = worker_reset_idle
        assert shared_bufs.done_events is not None, (
            "SharedEvalBuffers.init_done_events() must be called before creating RemoteEvaluator"
        )
        self._done_event = shared_bufs.done_events[worker_idx]

    def _request_eval(self, n: int) -> None:
        """Submit request via shared-memory flag, sleep until server signals done."""
        self._done_event.clear()
        self._signal_submit(self._sig_flags, self._sig_counts, self._worker_idx, n)
        if not self._done_event.wait(timeout=self._EVAL_TIMEOUT):
            raise RuntimeError(
                f"Eval server did not respond within {self._EVAL_TIMEOUT}s "
                f"(worker {self._worker_idx}, batch {n})"
            )
        self._reset_idle(self._sig_flags, self._worker_idx)

    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a single state via the remote server."""
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()
        rotate_visible_state_into(
            self._in_states_np[0], state._array, active_player, self.num_players
        )
        mask = get_valid_action_mask(state)

        self._request_eval(1)

        logits = self._out_logits[0].float().numpy()
        values = self._out_values[0].numpy()
        return self._finalize_single(logits, values, mask, active_player)

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

        masks_list = []
        for i, (s, ap) in enumerate(zip(states, active_ids)):
            rotate_visible_state_into(
                self._in_states_np[i], s._array, ap, self.num_players
            )
            masks_list.append(get_valid_action_mask(s))

        self._request_eval(n)

        logits_np = self._out_logits[:n].float().numpy()
        values_np = self._out_values[:n].numpy()
        return self._finalize_batch(logits_np, values_np, masks_list, active_ids)

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

        in_np = self._in_states_np
        for i, (arr, ap) in enumerate(zip(state_arrays, active_player_ids)):
            rotate_visible_state_into(in_np[i], arr, ap, self.num_players)

        if _stats is not None:
            _t1 = perf_counter()
            _stats.prepare_secs += _t1 - _t0

        self._request_eval(n)

        if _stats is not None:
            _t2 = perf_counter()
            _stats.wait_secs += _t2 - _t1

        logits_np = self._out_logits[:n].float().numpy()
        values_np = self._out_values[:n].numpy()
        results = self._finalize_leaves(logits_np, values_np, active_player_ids)

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
