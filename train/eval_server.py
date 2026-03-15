"""Centralized NN evaluation server for multi-process self-play.

The EvaluationServer runs as a thread in the main process, owning the model
and GPU. Worker processes send evaluation requests via shared memory and
receive results back. Multiple workers' requests are batched into single
GPU forward passes for throughput.

Communication uses shared memory (multiprocessing.RawArray) for state/mask/
policy/value data, with pipes carrying only lightweight control messages
(integer state counts). This eliminates pickle serialization of large numpy
arrays that was the primary throughput bottleneck.

RemoteEvaluator is the worker-side proxy that implements the same interface
as NNEvaluator, writing to shared memory instead of serializing over pipes.
"""

from __future__ import annotations

import ctypes
import threading
from multiprocessing import RawArray
from multiprocessing.connection import Connection, wait
from typing import Any, cast

import numpy as np
import torch

from mcts.evaluator import (
    compute_terminal_values,
    get_layout,
    rotate_visible_state,
    unrotate_values,
)


class SharedEvalBuffers:
    """Pre-allocated shared memory for zero-copy worker <-> server communication.

    Each worker gets a fixed slot in shared arrays. Workers write rotated states
    and legal masks into their input slot; the server reads them directly.
    The server writes policies and values into each worker's output slot.
    Pipes carry only integer state counts as control messages.

    Memory layout (per worker):
        Input:  states (batch_size x visible_size), masks (batch_size x action_dim)
        Output: policies (batch_size x action_dim), values (batch_size x num_players)
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
        self._masks = RawArray(
            ctypes.c_float, num_workers * batch_size * action_dim
        )
        # Output buffers (written by server, read by workers)
        self._policies = RawArray(
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

    def get_input_masks(self, worker_idx: int) -> np.ndarray:
        """Numpy view into worker's input mask slot (batch_size x action_dim)."""
        start = worker_idx * self.batch_size * self.action_dim
        count = self.batch_size * self.action_dim
        return np.frombuffer(
            self._masks, dtype=np.float32, offset=start * 4, count=count
        ).reshape(self.batch_size, self.action_dim)

    def get_output_policies(self, worker_idx: int) -> np.ndarray:
        """Numpy view into worker's output policy slot (batch_size x action_dim)."""
        start = worker_idx * self.batch_size * self.action_dim
        count = self.batch_size * self.action_dim
        return np.frombuffer(
            self._policies, dtype=np.float32, offset=start * 4, count=count
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
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        worker_conns: list[Connection],
        shared_bufs: SharedEvalBuffers,
    ) -> None:
        self._model = model
        self._device = device
        self._conns = list(worker_conns)
        self._shared_bufs = shared_bufs
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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

    def _loop(self) -> None:
        """Main server loop: gather requests, batch evaluate, dispatch."""
        conns = list(self._conns)
        conn_to_idx: dict[Connection, int] = {
            c: i for i, c in enumerate(self._conns)
        }
        bufs = self._shared_bufs

        while not self._stop.is_set() and conns:
            try:
                ready = wait(conns, timeout=0.01)
            except (OSError, ValueError):
                break

            if not ready:
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

            # Build batch from shared memory (concatenate worker slots)
            all_states = np.concatenate(
                [bufs.get_input_states(widx)[:n] for _, widx, n in batch_info]
            )
            all_masks = np.concatenate(
                [bufs.get_input_masks(widx)[:n] for _, widx, n in batch_info]
            )

            # Single GPU forward pass (bfloat16 for throughput)
            with torch.no_grad():
                x = torch.from_numpy(all_states).to(self._device)
                mask = torch.from_numpy(all_masks).to(self._device)
                with torch.autocast(
                    self._device.type, dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ):
                    policy_logits, values = self._model(x, legal_action_mask=mask)
                policies = torch.softmax(policy_logits.float(), dim=-1).cpu().numpy()
                values_np = values.float().cpu().numpy()

            # Write results to shared memory and signal workers
            offset = 0
            for conn, widx, n in batch_info:
                bufs.get_output_policies(widx)[:n] = policies[offset:offset + n]
                bufs.get_output_values(widx)[:n] = values_np[offset:offset + n]
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
    ) -> None:
        self.conn = conn
        self.num_players = num_players
        self.layout = get_layout(num_players)
        self._in_states = shared_bufs.get_input_states(worker_idx)
        self._in_masks = shared_bufs.get_input_masks(worker_idx)
        self._out_policies = shared_bufs.get_output_policies(worker_idx)
        self._out_values = shared_bufs.get_output_values(worker_idx)

    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a single state via the remote server."""
        from core.actions import get_valid_action_mask

        active_player = state.get_active_player()
        self._in_states[0] = rotate_visible_state(
            state._array, active_player, self.num_players
        )
        mask = get_valid_action_mask(state)
        self._in_masks[0] = mask

        self.conn.send(1)
        self.conn.recv()

        canonical = unrotate_values(self._out_values[0].copy(), active_player)
        return self._out_policies[0].copy(), canonical, mask

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

        for i, (s, ap) in enumerate(zip(states, active_ids)):
            self._in_states[i] = rotate_visible_state(
                s._array, ap, self.num_players
            )
            self._in_masks[i] = get_valid_action_mask(s)

        self.conn.send(n)
        self.conn.recv()

        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical = unrotate_values(self._out_values[i].copy(), active_ids[i])
            results.append((
                self._out_policies[i].copy(),
                canonical,
                self._in_masks[i].copy(),
            ))
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

        # Write rotated states and masks into shared memory
        for i, (arr, ap) in enumerate(zip(state_arrays, active_player_ids)):
            self._in_states[i] = rotate_visible_state(arr, ap, self.num_players)
        for i, mask in enumerate(legal_masks):
            self._in_masks[i] = mask

        # Signal server (just the count) and wait for completion
        self.conn.send(n)
        self.conn.recv()

        # Read results from shared memory (must copy — buffer reused next call)
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical = unrotate_values(
                self._out_values[i].copy(), active_player_ids[i]
            )
            results.append((self._out_policies[i].copy(), canonical, legal_masks[i]))
        return results

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values locally (no NN needed)."""
        net_worths = [
            state.get_player_net_worth(i) for i in range(self.num_players)
        ]
        return compute_terminal_values(net_worths, self.num_players)
