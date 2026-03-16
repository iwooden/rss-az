"""Centralized NN evaluation server for multi-process self-play.

The EvaluationServer runs as a thread in the main process, owning the model
and GPU. Worker processes send evaluation requests via shared memory and
receive results back. Multiple workers' requests are batched into single
GPU forward passes for throughput.

Communication uses shared memory (multiprocessing.RawArray) for state/logit/
value data, with pipes carrying only lightweight control messages (integer
state counts). Legal action masking and softmax are applied worker-side
after receiving raw logits from the server.

RemoteEvaluator is the worker-side proxy that implements the same interface
as NNEvaluator, writing to shared memory instead of serializing over pipes.
"""

from __future__ import annotations

import ctypes
import threading
from multiprocessing import RawArray
from multiprocessing.connection import Connection, wait
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch

from mcts.evaluator import (
    apply_mask_softmax,
    compute_terminal_values,
    get_layout,
    rotate_visible_state,
    unrotate_values,
)
from train.profile_stats import EvalClientStats, EvalServerStats


class SharedEvalBuffers:
    """Pre-allocated shared memory for zero-copy worker <-> server communication.

    Each worker gets a fixed slot in shared arrays. Workers write rotated states
    into their input slot; the server reads them directly. The server writes
    raw logits and values into each worker's output slot. Workers apply legal
    action masking and softmax locally after reading results.

    Memory layout (per worker):
        Input:  states (batch_size x visible_size)
        Output: logits (batch_size x action_dim), values (batch_size x num_players)
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

        # Input buffers (written by workers, read by server)
        self._states = RawArray(
            ctypes.c_float, num_workers * batch_size * visible_size
        )
        # Output buffers (written by server, read by workers)
        # _logits carries raw (unmasked) policy logits; workers apply mask+softmax
        self._logits = RawArray(
            ctypes.c_float, num_workers * batch_size * action_dim
        )
        self._values = RawArray(
            ctypes.c_float, num_workers * batch_size * num_players
        )

    def get_input_states(self, worker_idx: int) -> np.ndarray:
        """Numpy view into worker's input state slot (batch_size x visible_size)."""
        start = worker_idx * self.batch_size * self.visible_size
        count = self.batch_size * self.visible_size
        return np.frombuffer(
            self._states, dtype=np.float32, offset=start * 4, count=count
        ).reshape(self.batch_size, self.visible_size)

    def get_output_logits(self, worker_idx: int) -> np.ndarray:
        """Numpy view into worker's output logit slot (batch_size x action_dim)."""
        start = worker_idx * self.batch_size * self.action_dim
        count = self.batch_size * self.action_dim
        return np.frombuffer(
            self._logits, dtype=np.float32, offset=start * 4, count=count
        ).reshape(self.batch_size, self.action_dim)

    def get_output_values(self, worker_idx: int) -> np.ndarray:
        """Numpy view into worker's output value slot (batch_size x num_players)."""
        start = worker_idx * self.batch_size * self.num_players
        count = self.batch_size * self.num_players
        return np.frombuffer(
            self._values, dtype=np.float32, offset=start * 4, count=count
        ).reshape(self.batch_size, self.num_players)


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

        # Pre-allocated transfer buffers for zero-alloc H2D/D2H.
        # Max possible batch = num_workers * search_batch_size.
        max_batch = shared_bufs.num_workers * shared_bufs.batch_size
        vis = shared_bufs.visible_size
        act = shared_bufs.action_dim
        npl = shared_bufs.num_players
        use_cuda = device.type == "cuda"

        # Pinned CPU buffers for fast DMA to GPU (input: states only)
        self._pin_states = torch.empty(
            max_batch, vis, dtype=torch.float32,
            pin_memory=use_cuda,
        )
        self._pin_states_np = self._pin_states.numpy()

        # Pre-allocated GPU tensor (reused every batch)
        self._gpu_states = torch.empty(max_batch, vis, dtype=torch.float32, device=device)

        # Pinned CPU buffers for fast D2H (output: raw logits + values)
        self._pin_logits = torch.empty(
            max_batch, act, dtype=torch.float32,
            pin_memory=use_cuda,
        )
        self._pin_values = torch.empty(
            max_batch, npl, dtype=torch.float32,
            pin_memory=use_cuda,
        )
        self._pin_logits_np = self._pin_logits.numpy()
        self._pin_values_np = self._pin_values.numpy()

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
        """Main server loop: gather requests, batch evaluate, dispatch."""
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
        pin_s_np = self._pin_states_np
        pin_s = self._pin_states
        gpu_s = self._gpu_states
        pin_log = self._pin_logits
        pin_val = self._pin_values
        pin_log_np = self._pin_logits_np
        pin_val_np = self._pin_values_np

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

            # Gather states into pinned memory (no masks — applied worker-side)
            total_n = 0
            for _, widx, n in batch_info:
                pin_s_np[total_n:total_n + n] = bufs.get_input_states(widx)[:n]
                total_n += n

            # H2D via pinned memory (non-blocking DMA)
            if stats is not None:
                _ti = perf_counter()

            gpu_s_batch = gpu_s[:total_n]
            gpu_s_batch.copy_(pin_s[:total_n], non_blocking=True)

            # GPU forward pass (bfloat16 for throughput, no mask)
            with torch.no_grad():
                with torch.autocast(
                    dev.type, dtype=torch.bfloat16, enabled=use_cuda,
                ):
                    policy_logits, values = self._model(gpu_s_batch)
                # Raw logits + values to pinned memory (no softmax)
                log_f = policy_logits.float()
                val_f = values.float()
                pin_log[:total_n].copy_(log_f, non_blocking=True)
                pin_val[:total_n].copy_(val_f, non_blocking=True)
                if use_cuda:
                    torch.cuda.synchronize()

            if stats is not None:
                stats.record_batch(total_n, perf_counter() - _ti)

            # Scatter results from pinned memory to per-worker shared memory
            offset = 0
            for conn, widx, n in batch_info:
                bufs.get_output_logits(widx)[:n] = pin_log_np[offset:offset + n]
                bufs.get_output_values(widx)[:n] = pin_val_np[offset:offset + n]
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

    Data flows through SharedEvalBuffers (zero-copy shared memory);
    pipes carry only integer control messages.
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
        self._in_states = shared_bufs.get_input_states(worker_idx)
        self._out_logits = shared_bufs.get_output_logits(worker_idx)
        self._out_values = shared_bufs.get_output_values(worker_idx)
        self._profile = profile
        self._stats: EvalClientStats | None = EvalClientStats() if profile else None

    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a single state via the remote server."""
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()
        self._in_states[0] = rotate_visible_state(
            state._array, active_player, self.num_players
        )
        mask = get_valid_action_mask(state)

        self.conn.send(1)
        self.conn.recv()

        # Apply mask + softmax worker-side on raw logits from server
        policy_probs = apply_mask_softmax(self._out_logits[0], mask)
        canonical = unrotate_values(self._out_values[0].copy(), active_player)
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
        masks = []

        for i, (s, ap) in enumerate(zip(states, active_ids)):
            self._in_states[i] = rotate_visible_state(
                s._array, ap, self.num_players
            )
            masks.append(get_valid_action_mask(s))

        self.conn.send(n)
        self.conn.recv()

        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            policy_probs = apply_mask_softmax(self._out_logits[i], masks[i])
            canonical = unrotate_values(self._out_values[i].copy(), active_ids[i])
            results.append((policy_probs, canonical, masks[i]))
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

        # Write rotated states into shared memory (no masks — applied locally)
        for i, (arr, ap) in enumerate(zip(state_arrays, active_player_ids)):
            self._in_states[i] = rotate_visible_state(arr, ap, self.num_players)

        if _stats is not None:
            _t1 = perf_counter()
            _stats.prepare_secs += _t1 - _t0

        # Signal server (just the count) and wait for completion
        self.conn.send(n)
        self.conn.recv()

        if _stats is not None:
            _t2 = perf_counter()
            _stats.wait_secs += _t2 - _t1

        # Read raw logits from shared memory, apply mask + softmax locally
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            policy_probs = apply_mask_softmax(
                self._out_logits[i], legal_masks[i],
            )
            canonical = unrotate_values(
                self._out_values[i].copy(), active_player_ids[i]
            )
            results.append((policy_probs, canonical, legal_masks[i]))

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
