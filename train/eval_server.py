"""Centralized NN evaluation server for multi-process self-play.

One or more EvaluationServer threads run in the main process, sharing the
model and GPU. Worker processes send evaluation requests via a shared queue
and receive completion signals via per-worker Events. Multiple workers'
requests are batched into single GPU forward passes for throughput.

Communication uses shared memory (multiprocessing.RawArray) for state/logit/
value data. A multiprocessing.Queue carries lightweight request tuples
(worker_idx, state_count), and per-worker Events signal completion.
Legal action masking and softmax are applied worker-side after receiving
raw logits from the server.

Multiple EvaluationServer threads consuming from the same queue naturally
double-buffer GPU access: one server gathers while another is on GPU.

RemoteEvaluator is the worker-side proxy that implements the same interface
as NNEvaluator, writing to shared memory instead of serializing over pipes.
"""

from __future__ import annotations

import ctypes
import queue as _queue
import threading
from multiprocessing import RawArray
from time import perf_counter
from typing import Any

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

    All channels use bfloat16 to halve PCIe and shared memory bandwidth.
    Workers convert float32 numpy <-> bfloat16 torch at the boundary.
    The eval server operates entirely in bfloat16 with zero dtype conversions.

    Each worker gets a fixed slot in shared arrays. Workers write rotated states
    into their input slot; the server reads them directly. The server writes
    raw logits and values into each worker's output slot. Workers apply legal
    action masking and softmax locally after reading results.

    Memory layout (per worker, all bfloat16):
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

        # Shared memory as raw bytes (2 bytes per bfloat16 element)
        self._states = RawArray(
            ctypes.c_char, num_workers * batch_size * visible_size * 2
        )
        self._logits = RawArray(
            ctypes.c_char, num_workers * batch_size * action_dim * 2
        )
        self._values = RawArray(
            ctypes.c_char, num_workers * batch_size * num_players * 2
        )

    def get_input_states(self, worker_idx: int) -> torch.Tensor:
        """bfloat16 tensor view into worker's input state slot."""
        start = worker_idx * self.batch_size * self.visible_size * 2
        count = self.batch_size * self.visible_size * 2
        return torch.frombuffer(
            memoryview(self._states)[start:start + count],
            dtype=torch.bfloat16,
        ).reshape(self.batch_size, self.visible_size)

    def get_output_logits(self, worker_idx: int) -> torch.Tensor:
        """bfloat16 tensor view into worker's output logit slot."""
        start = worker_idx * self.batch_size * self.action_dim * 2
        count = self.batch_size * self.action_dim * 2
        return torch.frombuffer(
            memoryview(self._logits)[start:start + count],
            dtype=torch.bfloat16,
        ).reshape(self.batch_size, self.action_dim)

    def get_output_values(self, worker_idx: int) -> torch.Tensor:
        """bfloat16 tensor view into worker's output value slot."""
        start = worker_idx * self.batch_size * self.num_players * 2
        count = self.batch_size * self.num_players * 2
        return torch.frombuffer(
            memoryview(self._values)[start:start + count],
            dtype=torch.bfloat16,
        ).reshape(self.batch_size, self.num_players)


class EvaluationServer:
    """Thread-based centralized NN evaluator using shared memory.

    Consumes (worker_idx, state_count) requests from a shared queue,
    gathers states from shared memory, runs batched inference, writes
    results back, and signals workers via per-worker Events.

    Multiple EvaluationServer instances can share the same queue for
    natural pipeline overlap on a single GPU.

    Uses pinned CPU memory and pre-allocated GPU tensors to minimize
    host-to-device and device-to-host transfer overhead.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        shared_bufs: SharedEvalBuffers,
        request_queue: Any,
        worker_events: list[Any],
        *,
        profile: bool = False,
    ) -> None:
        self._model = model
        self._device = device
        self._shared_bufs = shared_bufs
        self._request_queue = request_queue
        self._worker_events = worker_events
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._profile = profile
        self._stats: EvalServerStats | None = EvalServerStats() if profile else None

        # Pre-allocated transfer buffers for zero-alloc H2D/D2H.
        # All bfloat16 — no dtype conversions anywhere in the server.
        max_batch = shared_bufs.num_workers * shared_bufs.batch_size
        vis = shared_bufs.visible_size
        act = shared_bufs.action_dim
        npl = shared_bufs.num_players
        use_cuda = device.type == "cuda"

        # Pinned CPU buffer for fast DMA to GPU (input: states)
        self._pin_states = torch.empty(
            max_batch, vis, dtype=torch.bfloat16,
            pin_memory=use_cuda,
        )

        # Pre-allocated GPU tensor (reused every batch)
        self._gpu_states = torch.empty(
            max_batch, vis, dtype=torch.bfloat16, device=device,
        )

        # Pinned CPU buffers for fast D2H (output: raw logits + values)
        self._pin_logits = torch.empty(
            max_batch, act, dtype=torch.bfloat16,
            pin_memory=use_cuda,
        )
        self._pin_values = torch.empty(
            max_batch, npl, dtype=torch.bfloat16,
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
        """Main server loop: gather requests, batch evaluate, dispatch."""
        bufs = self._shared_bufs
        stats = self._stats
        dev = self._device
        use_cuda = dev.type == "cuda"
        _tp = _ti = 0.0  # profile timing scratch vars

        # Local refs to pre-allocated buffers (avoid attribute lookups in hot loop)
        pin_s = self._pin_states
        gpu_s = self._gpu_states
        pin_log = self._pin_logits
        pin_val = self._pin_values

        # Cache per-worker shared memory views (avoid per-call torch.frombuffer)
        w_states = [bufs.get_input_states(i) for i in range(bufs.num_workers)]
        w_logits = [bufs.get_output_logits(i) for i in range(bufs.num_workers)]
        w_values = [bufs.get_output_values(i) for i in range(bufs.num_workers)]

        req_q = self._request_queue
        events = self._worker_events
        max_batch = bufs.num_workers * bufs.batch_size

        while not self._stop.is_set():
            if stats is not None:
                _tp = perf_counter()

            # Block for first request (with timeout to check stop flag)
            try:
                first: tuple[int, int] = req_q.get(timeout=0.01)
            except _queue.Empty:
                if stats is not None:
                    stats.record_idle(perf_counter() - _tp)
                continue

            # Drain queue greedily to build a larger batch
            batch_info: list[tuple[int, int]] = [first]
            while len(batch_info) < max_batch:
                try:
                    batch_info.append(req_q.get_nowait())
                except _queue.Empty:
                    break

            # Gather bfloat16 states into pinned memory
            total_n = 0
            for widx, n in batch_info:
                pin_s[total_n:total_n + n].copy_(w_states[widx][:n])
                total_n += n

            # H2D via pinned memory (non-blocking DMA, all bfloat16)
            if stats is not None:
                _ti = perf_counter()

            gpu_s_batch = gpu_s[:total_n]
            gpu_s_batch.copy_(pin_s[:total_n], non_blocking=True)

            # Forward pass (bfloat16 throughout via autocast on all devices)
            with torch.no_grad():
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    policy_logits, values = self._model(gpu_s_batch)
                pin_log[:total_n].copy_(policy_logits, non_blocking=True)
                pin_val[:total_n].copy_(values, non_blocking=True)
                if use_cuda:
                    torch.cuda.synchronize()

            if stats is not None:
                stats.record_batch(total_n, perf_counter() - _ti)

            # Scatter bfloat16 results to per-worker shared memory and signal
            offset = 0
            for widx, n in batch_info:
                w_logits[widx][:n].copy_(pin_log[offset:offset + n])
                w_values[widx][:n].copy_(pin_val[offset:offset + n])
                events[widx].set()
                offset += n


class RemoteEvaluator:
    """Worker-side proxy that evaluates states via shared memory + EvaluationServer.

    Implements the same evaluate/evaluate_batch/evaluate_terminal interface
    as NNEvaluator, so it can be used as a drop-in replacement.

    Data flows through SharedEvalBuffers (zero-copy shared memory);
    a Queue carries request tuples and per-worker Events signal completion.
    """

    def __init__(
        self,
        num_players: int,
        shared_bufs: SharedEvalBuffers,
        worker_idx: int,
        request_queue: Any,
        done_event: Any,
        *,
        profile: bool = False,
    ) -> None:
        self.num_players = num_players
        self.layout = get_layout(num_players)
        self._worker_idx = worker_idx
        self._in_states = shared_bufs.get_input_states(worker_idx)
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
        # f32 numpy -> bf16 shared memory (copy_ handles dtype conversion)
        self._in_states[0].copy_(torch.from_numpy(
            rotate_visible_state(state._array, active_player, self.num_players)
        ))
        mask = get_valid_action_mask(state)

        self._event.clear()
        self._queue.put((self._worker_idx, 1))
        self._event.wait()

        # bf16 shared memory -> f32 numpy, then apply mask + softmax
        policy_probs = apply_mask_softmax(
            self._out_logits[0].float().numpy(), mask,
        )
        canonical = unrotate_values(
            self._out_values[0].float().numpy(), active_player,
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
        masks = []

        for i, (s, ap) in enumerate(zip(states, active_ids)):
            self._in_states[i].copy_(torch.from_numpy(
                rotate_visible_state(s._array, ap, self.num_players)
            ))
            masks.append(get_valid_action_mask(s))

        self._event.clear()
        self._queue.put((self._worker_idx, n))
        self._event.wait()

        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            policy_probs = apply_mask_softmax(
                self._out_logits[i].float().numpy(), masks[i],
            )
            canonical = unrotate_values(
                self._out_values[i].float().numpy(), active_ids[i],
            )
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

        # Write rotated states: f32 numpy -> bf16 shared memory
        for i, (arr, ap) in enumerate(zip(state_arrays, active_player_ids)):
            self._in_states[i].copy_(torch.from_numpy(
                rotate_visible_state(arr, ap, self.num_players)
            ))

        if _stats is not None:
            _t1 = perf_counter()
            _stats.prepare_secs += _t1 - _t0

        # Signal server and wait for completion
        self._event.clear()
        self._queue.put((self._worker_idx, n))
        self._event.wait()

        if _stats is not None:
            _t2 = perf_counter()
            _stats.wait_secs += _t2 - _t1

        # Read bf16 logits+values, convert to f32 numpy, apply mask + softmax
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            policy_probs = apply_mask_softmax(
                self._out_logits[i].float().numpy(), legal_masks[i],
            )
            canonical = unrotate_values(
                self._out_values[i].float().numpy(), active_player_ids[i],
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
