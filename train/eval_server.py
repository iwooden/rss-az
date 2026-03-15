"""Centralized NN evaluation server for multi-process self-play.

The EvaluationServer runs as a thread in the main process, owning the model
and GPU. Worker processes send evaluation requests via multiprocessing.Pipe
and receive results back. Multiple workers' requests are batched into single
GPU forward passes for throughput.

RemoteEvaluator is the worker-side proxy that implements the same interface
as NNEvaluator, sending requests to the server instead of running locally.
"""

from __future__ import annotations

import threading
from multiprocessing.connection import Connection, wait
from typing import Any, NamedTuple, cast

import numpy as np
import torch

from mcts.evaluator import (
    compute_terminal_values,
    get_layout,
    rotate_visible_state,
    unrotate_values,
)


class EvalRequest(NamedTuple):
    """Batch of pre-rotated states for NN evaluation."""

    states: np.ndarray  # (N, visible_size)
    masks: np.ndarray  # (N, action_dim)
    num_states: int


class EvalResponse(NamedTuple):
    """Batch of NN results (policies as softmax probs, values in active-player-first order)."""

    policies: np.ndarray  # (N, action_dim)
    values: np.ndarray  # (N, num_players)


class EvaluationServer:
    """Thread-based centralized NN evaluator.

    Aggregates requests from multiple worker processes,
    runs batched inference, and dispatches results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        worker_conns: list[Connection],
    ) -> None:
        self._model = model
        self._device = device
        self._conns = list(worker_conns)
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

        while not self._stop.is_set() and conns:
            try:
                ready = wait(conns, timeout=0.01)
            except (OSError, ValueError):
                break

            if not ready:
                continue

            # Gather all pending requests
            requests: list[tuple[Connection, EvalRequest]] = []
            for conn in cast(list[Connection], ready):
                try:
                    req = conn.recv()
                    requests.append((conn, req))
                except (EOFError, OSError):
                    if conn in conns:
                        conns.remove(conn)
                    continue

            if not requests:
                continue

            # Stack all states/masks into one batch
            all_states = np.concatenate([r.states for _, r in requests])
            all_masks = np.concatenate([r.masks for _, r in requests])

            # Single GPU forward pass
            with torch.no_grad():
                x = torch.from_numpy(all_states).to(self._device)
                mask = torch.from_numpy(all_masks).to(self._device)
                policy_logits, values = self._model(x, legal_action_mask=mask)
                policies = torch.softmax(policy_logits, dim=-1).cpu().numpy()
                values_np = values.cpu().numpy()

            # Dispatch results back to respective workers
            offset = 0
            for conn, req in requests:
                n = req.num_states
                try:
                    conn.send(
                        EvalResponse(
                            policies[offset : offset + n].copy(),
                            values_np[offset : offset + n].copy(),
                        )
                    )
                except (OSError, BrokenPipeError):
                    if conn in conns:
                        conns.remove(conn)
                offset += n


class RemoteEvaluator:
    """Worker-side proxy that sends evaluation requests to the EvaluationServer.

    Implements the same evaluate/evaluate_batch/evaluate_terminal interface
    as NNEvaluator, so it can be used as a drop-in replacement.
    """

    def __init__(self, conn: Connection, num_players: int) -> None:
        self.conn = conn
        self.num_players = num_players
        self.layout = get_layout(num_players)

    def evaluate(self, state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate a single state via the remote server."""
        return self.evaluate_batch([state])[0]

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

        # Rotate states and get masks (CPU work, done in worker process)
        rotated = np.stack(
            [
                rotate_visible_state(s._array, ap, self.num_players)
                for s, ap in zip(states, active_ids)
            ]
        )
        masks = np.stack([get_valid_action_mask(s) for s in states])

        # Send to server and block until response
        self.conn.send(EvalRequest(rotated, masks, n))
        response: EvalResponse = self.conn.recv()

        # Un-rotate values to canonical player order, return masks for reuse
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical = unrotate_values(response.values[i], active_ids[i])
            results.append((response.policies[i], canonical, masks[i]))
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

        # Rotate states (CPU work, done in worker process)
        rotated = np.stack([
            rotate_visible_state(arr, ap, self.num_players)
            for arr, ap in zip(state_arrays, active_player_ids)
        ])
        masks = np.stack(legal_masks)

        # Send to server and block until response
        self.conn.send(EvalRequest(rotated, masks, n))
        response: EvalResponse = self.conn.recv()

        # Un-rotate values to canonical player order
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for i in range(n):
            canonical = unrotate_values(response.values[i], active_player_ids[i])
            results.append((response.policies[i], canonical, legal_masks[i]))
        return results

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values locally (no NN needed)."""
        net_worths = [
            state.get_player_net_worth(i) for i in range(self.num_players)
        ]
        return compute_terminal_values(net_worths, self.num_players)
