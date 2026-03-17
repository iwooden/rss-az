"""Centralized NN evaluation server for multi-process self-play.

The EvaluationServer runs as a thread in the main process, owning the model
and GPU. Worker processes send evaluation requests via shared memory and
receive results back. Multiple workers' requests are batched into single
GPU forward passes for throughput.

Communication uses shared memory (torch tensors with share_memory_()):
- Input states: float32 (workers write with pure numpy slice assignment)
- Output logits: bfloat16 (halves scatter bandwidth; workers upcast for softmax)
- Output values: float32 (server upcasts from bf16 model output during D2H;
  only 3 floats/state so bf16 savings are negligible vs per-worker conversion)

Workers do zero torch operations on the write side (pure numpy).
On the read side, workers do one bf16→f32 upcast on logits for
batched mask+softmax in torch. Values are read as f32 numpy directly.

RemoteEvaluator is the worker-side proxy that implements the same interface
as NNEvaluator, writing to shared memory instead of serializing over pipes.
"""

from __future__ import annotations

import threading
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from multiprocessing.connection import Connection, wait

from mcts.evaluator import (
    compute_terminal_values,
    get_layout,
    rotate_visible_state,
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


class EvaluationServer:
    """Thread-based centralized NN evaluator using shared memory.

    Aggregates requests from multiple worker processes,
    runs batched inference, and dispatches results.
    Workers signal readiness via pipes (sending integer state counts);
    actual data is exchanged through SharedEvalBuffers.

    Uses pinned CPU memory and pre-allocated GPU tensors to minimize
    host-to-device and device-to-host transfer overhead.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        worker_conns: list[Connection],
        shared_bufs: SharedEvalBuffers,
        *,
        profile: bool = False,
    ) -> None:
        self._model = model
        self._device = device
        self._conns = list(worker_conns)
        self._shared_bufs = shared_bufs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._profile = profile
        self._stats: EvalServerStats | None = EvalServerStats() if profile else None

        max_batch = shared_bufs.num_workers * shared_bufs.batch_size
        vis = shared_bufs.visible_size
        act = shared_bufs.action_dim
        npl = shared_bufs.num_players
        use_cuda = device.type == "cuda"

        # Input: pinned f32 buffer + f32 GPU tensor
        # The f32→bf16 conversion happens implicitly via autocast on CUDA
        self._pin_states = torch.empty(
            max_batch, vis, dtype=torch.float32,
            pin_memory=use_cuda,
        )
        self._pin_states_np = self._pin_states.numpy()
        self._gpu_states = torch.empty(
            max_batch, vis, dtype=torch.float32, device=device,
        )

        # Output: pinned buffers (match shared memory dtypes)
        self._pin_logits = torch.empty(
            max_batch, act, dtype=torch.bfloat16,
            pin_memory=use_cuda,
        )
        self._pin_values = torch.empty(
            max_batch, npl, dtype=torch.float32,
            pin_memory=use_cuda,
        )

    def start(self) -> None:
        """Start the server thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="eval-server"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the server to stop and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_profile_stats(self) -> EvalServerStats | None:
        """Return accumulated profile stats (None if profiling disabled)."""
        return self._stats

    def reset_profile_stats(self) -> None:
        """Reset profile stats for a new epoch."""
        if self._stats is not None:
            self._stats.reset()

    def _loop(self) -> None:
        """Main server loop: gather requests, batch evaluate, dispatch.

        Runs in a daemon thread — uncaught exceptions would be silent.
        The try/except ensures crashes are logged to stdout.
        """
        try:
            self._serve()
        except Exception:
            import traceback
            traceback.print_exc()

    def _serve(self) -> None:
        """Inner server loop (called by _loop with exception guard)."""
        conns = list(self._conns)
        conn_to_idx: dict[Connection, int] = {
            c: i for i, c in enumerate(self._conns)
        }
        bufs = self._shared_bufs
        stats = self._stats
        dev = self._device
        use_cuda = dev.type == "cuda"
        _tp = _ti = 0.0  # profile timing scratch vars

        # Local refs to pre-allocated buffers (avoid attribute lookups in hot loop)
        pin_s = self._pin_states
        pin_s_np = self._pin_states_np
        gpu_s = self._gpu_states
        pin_log = self._pin_logits
        pin_val = self._pin_values

        # Cache per-worker shared memory views
        w_states_np = [bufs.get_input_states_np(i) for i in range(bufs.num_workers)]
        w_logits = [bufs.get_output_logits(i) for i in range(bufs.num_workers)]
        w_values = [bufs.get_output_values(i) for i in range(bufs.num_workers)]

        while not self._stop.is_set() and conns:
            if stats is not None:
                _tp = perf_counter()

            try:
                ready = wait(conns, timeout=0.01)
            except (OSError, ValueError):
                break

            if not ready:
                if stats is not None:
                    stats.record_idle(perf_counter() - _tp)
                continue

            # Non-blocking poll for additional connections that became ready
            # during the wait() return. This improves batch sizes.
            ready_set: set[Connection] = set(cast(list[Connection], ready))
            remaining = [c for c in conns if c not in ready_set]
            if remaining:
                try:
                    more = wait(remaining, timeout=0)
                    if more:
                        ready_set.update(cast(list[Connection], more))
                except (OSError, ValueError):
                    pass

            # Read control messages (just integer state counts)
            batch_info: list[tuple[Connection, int, int]] = []
            for conn in ready_set:
                try:
                    n: int = conn.recv()
                    batch_info.append((conn, conn_to_idx[conn], n))
                except (EOFError, OSError):
                    if conn in conns:
                        conns.remove(conn)
                    continue

            if not batch_info:
                continue

            # Gather f32 states from numpy shared memory into pinned numpy
            total_n = 0
            for _, widx, n in batch_info:
                pin_s_np[total_n:total_n + n] = w_states_np[widx][:n]
                total_n += n

            # H2D via pinned memory (non-blocking DMA)
            if stats is not None:
                _ti = perf_counter()

            gpu_s_batch = gpu_s[:total_n]
            gpu_s_batch.copy_(pin_s[:total_n], non_blocking=True)

            # GPU forward pass — autocast handles f32→bf16 on CUDA
            with torch.no_grad():
                if use_cuda:
                    with torch.autocast(dev.type, dtype=torch.bfloat16):
                        policy_logits, values = self._model(gpu_s_batch)
                else:
                    policy_logits, values = self._model(gpu_s_batch)
                # D2H: logits stay bf16; values upcast bf16→f32 via copy_
                # (on CPU without autocast both are f32; .bfloat16() converts logits)
                pin_log[:total_n].copy_(
                    policy_logits.bfloat16(), non_blocking=True,
                )
                pin_val[:total_n].copy_(values, non_blocking=True)
                if use_cuda:
                    torch.cuda.synchronize()

            if stats is not None:
                stats.record_batch(total_n, perf_counter() - _ti)

            # Scatter results to per-worker shared memory (logits bf16, values f32)
            offset = 0
            for conn, widx, n in batch_info:
                w_logits[widx][:n].copy_(pin_log[offset:offset + n])
                w_values[widx][:n].copy_(pin_val[offset:offset + n])
                try:
                    conn.send(n)
                except (OSError, BrokenPipeError):
                    if conn in conns:
                        conns.remove(conn)
                offset += n


class RemoteEvaluator:
    """Worker-side proxy that evaluates states via shared memory + EvaluationServer.

    Implements the same evaluate/evaluate_batch/evaluate_terminal interface
    as NNEvaluator, so it can be used as a drop-in replacement.

    Write side: pure numpy slice assignment into f32 shared memory (no torch).
    Read side: logits bf16→f32 for batched mask+softmax in torch;
    values are f32 in shared memory (server upcasts during D2H).
    """

    def __init__(
        self,
        conn: Connection,
        num_players: int,
        shared_bufs: SharedEvalBuffers,
        worker_idx: int,
        *,
        profile: bool = False,
    ) -> None:
        self.conn = conn
        self.num_players = num_players
        self.layout = get_layout(num_players)
        self._in_states_np = shared_bufs.get_input_states_np(worker_idx)
        self._out_logits = shared_bufs.get_output_logits(worker_idx)
        self._out_values = shared_bufs.get_output_values(worker_idx)
        self._profile = profile
        self._stats: EvalClientStats | None = EvalClientStats() if profile else None

    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a single state via the remote server."""
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()
        # Pure numpy write — no torch overhead
        self._in_states_np[0] = rotate_visible_state(
            state._array, active_player, self.num_players
        )
        mask = get_valid_action_mask(state)

        self.conn.send(1)
        self.conn.recv()

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

        # Pure numpy writes — no torch overhead
        masks_list = []
        for i, (s, ap) in enumerate(zip(states, active_ids)):
            self._in_states_np[i] = rotate_visible_state(
                s._array, ap, self.num_players
            )
            masks_list.append(get_valid_action_mask(s))

        self.conn.send(n)
        self.conn.recv()

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
        legal_masks: list[np.ndarray],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate pre-computed leaf data in a single round-trip to the server.

        Like evaluate_batch but takes raw arrays instead of GameState objects,
        avoiding Python wrapper allocation in the MCTS hot loop.
        """
        n = len(state_arrays)
        if n == 0:
            return []

        _stats = self._stats
        _t0 = _t1 = _t2 = 0.0
        if _stats is not None:
            _t0 = perf_counter()

        # Pure numpy writes — rotate and assign directly to f32 shared memory
        in_np = self._in_states_np
        for i, (arr, ap) in enumerate(zip(state_arrays, active_player_ids)):
            in_np[i] = rotate_visible_state(arr, ap, self.num_players)

        if _stats is not None:
            _t1 = perf_counter()
            _stats.prepare_secs += _t1 - _t0

        # Signal server (just the count) and wait for completion
        self.conn.send(n)
        self.conn.recv()

        if _stats is not None:
            _t2 = perf_counter()
            _stats.wait_secs += _t2 - _t1

        # Logits: bf16→f32 for mask+softmax; values: already f32
        logits_f32 = self._out_logits[:n].float()
        masks_t = torch.from_numpy(np.stack(legal_masks))
        logits_f32.masked_fill_(masks_t <= 0, -1e9)
        probs_batch = torch.softmax(logits_f32, dim=-1).numpy()
        values_np = self._out_values[:n].numpy()

        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical = unrotate_values(values_np[i], active_player_ids[i])
            results.append((probs_batch[i].copy(), canonical, legal_masks[i]))

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
        return compute_terminal_values(net_worths, self.num_players)
