"""NN evaluator for MCTS leaf evaluation.

Fills token buffers from compact int16 game states via
``core.token_data.get_token_data``, runs NN inference, and returns
sparse softmax priors over legal actions plus canonical-order values.
No rotation — the transformer consumes non-rotated state and emits
values in canonical player order directly.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from core.token_data import get_num_tokens, TokenDataSize, get_token_data
from core.actions import (
    get_decision_phase_py,
    enumerate_legal_actions_py,
    MAX_LEGAL_ACTIONS_PY,
)
from entities.player import PLAYERS

logger = logging.getLogger(__name__)

K_MAX = int(MAX_LEGAL_ACTIONS_PY)
TOKEN_DIM = int(TokenDataSize.TOKEN_DIM)


def fill_token_buffer(state: Any, buf2d: np.ndarray) -> None:
    """Fill a ``(num_tokens, TOKEN_DIM)`` float32 buffer from a GameState.

    Thin wrapper around ``core.token_data.get_token_data`` that isolates
    the Cython import-site so callers don't need to know where the entry
    point lives. ``num_players ∉ {3, 4, 5}`` is rejected inside
    ``get_token_data`` via assert.
    """
    get_token_data(state, buf2d)


def compute_terminal_values(
    net_worths: list[int], num_players: int, rank_weight: float = 0.5,
) -> np.ndarray:
    """Compute canonical reward values for a terminal game state.

    Blend of rank-based and zero-sum net-worth-deviation rewards. The rank
    component provides sharp signal at rank boundaries (overtaking an
    opponent matters a lot). The margin component provides continuous
    gradient within ranks (3rd place still has reason to improve).

    Both components are zero-sum across players, so the blended result is
    also zero-sum (better utilization of the tanh value head's [-1, +1]
    range). The margin uses a scale factor of n/(n-1) which guarantees
    the result stays in [-1, +1] when all net worths are non-negative
    (game rules ensure this — players cannot have negative net worth).

    When all players have zero net worth, all receive 0.0.

    Args:
        net_worths: List of net worth values per player (canonical order).
        num_players: Number of players.
        rank_weight: Weight for rank component (0.0 = pure margin,
            1.0 = pure rank). Default 0.5 (equal blend).

    Returns:
        np.ndarray of shape (num_players,) with reward values per player.
    """
    max_nw = max(net_worths)

    if max_nw == 0:
        return np.zeros(num_players, dtype=np.float32)

    # Rank component: evenly spaced from +1.0 to -1.0 by placement (zero-sum)
    rank_rewards = np.linspace(1.0, -1.0, num_players)
    sorted_indices = np.argsort(net_worths)[::-1]  # descending
    rank_values = np.zeros(num_players, dtype=np.float32)
    i = 0
    while i < num_players:
        j = i + 1
        while j < num_players and net_worths[sorted_indices[j]] == net_worths[sorted_indices[i]]:
            j += 1
        avg_reward = float(np.mean(rank_rewards[i:j]))
        for k in range(i, j):
            rank_values[sorted_indices[k]] = avg_reward
        i = j

    if rank_weight >= 1.0:
        return rank_values

    # Margin component: zero-sum net-worth deviation from mean, scaled by
    # n/(n-1) so the range is exactly [-1, +1] for any NW distribution
    mean_nw = sum(net_worths) / num_players
    scale = num_players / (num_players - 1)
    margin_values = np.array(
        [scale * (nw - mean_nw) / max_nw for nw in net_worths], dtype=np.float32
    )

    if rank_weight <= 0.0:
        return margin_values

    return rank_weight * rank_values + (1.0 - rank_weight) * margin_values


class BaseEvaluator:
    """Shared state and post-processing for MCTS evaluators.

    Subclassed by NNEvaluator (local model inference) and
    RemoteEvaluator (shared-memory IPC to eval server). Both speak
    token buffers — compact int16 state in, sparse softmaxed priors +
    canonical-order values out.
    """

    def __init__(self, num_players: int, terminal_rank_weight: float = 0.5) -> None:
        self.num_players = num_players
        self.terminal_rank_weight = terminal_rank_weight
        self.num_tokens = get_num_tokens(num_players)
        self.token_dim = TOKEN_DIM

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values from a game-over state.

        Args:
            state: GameState in GAME_OVER phase.

        Returns:
            Canonical values, shape (num_players,).
        """
        net_worths = [PLAYERS[i].get_net_worth(state) for i in range(self.num_players)]
        return compute_terminal_values(
            net_worths, self.num_players, self.terminal_rank_weight
        )

    def _check_nan(self, logits: np.ndarray, values: np.ndarray) -> None:
        """Raise if NN output contains NaN."""
        has_nan_logits = np.isnan(logits).any()
        has_nan_values = np.isnan(values).any()
        if has_nan_logits or has_nan_values:
            parts: list[str] = []
            if has_nan_logits:
                nan_count = int(np.isnan(logits).sum())
                parts.append(f"logits ({nan_count}/{logits.size} NaN)")
            if has_nan_values:
                nan_count = int(np.isnan(values).sum())
                parts.append(f"values ({nan_count}/{values.size} NaN)")
            msg = f"NaN in NN eval output: {', '.join(parts)}"
            logger.error(msg)
            raise RuntimeError(msg)


class NNEvaluator(BaseEvaluator):
    """Wraps a neural network model for in-process MCTS leaf evaluation.

    Used for single-process tests and anything that doesn't go through
    the shared-mem eval server. Fills a token buffer from compact state,
    runs inference, gathers the dense policy logits at the legal-action
    ids, and softmaxes — mirroring the GPU-side gather+softmax that the
    eval server runs for ``RemoteEvaluator``.
    """

    _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_players: int = 3, *,
                 terminal_rank_weight: float = 0.5,
                 eval_dtype: str | None = None) -> None:
        super().__init__(num_players, terminal_rank_weight)
        self.model = model
        self.device = device
        self._autocast_dtype = self._DTYPE_MAP.get(eval_dtype) if eval_dtype else None
        self.model.eval()

        # Reusable token buffer for batch evaluation, grows as needed.
        self._token_buf: np.ndarray | None = None

        # Catch model/player-count mismatch before it reaches boundscheck=False
        # Cython code. The new transformer config only exposes num_players
        # (action-space width is set by PHASE_ACTION_SIZES, shared via
        # core.data.ActionSize; value-head width always tracks num_players).
        cfg = getattr(model, "cfg", None)
        if cfg is not None:
            model_np = getattr(cfg, "num_players", num_players)
            if model_np != num_players:
                raise ValueError(
                    f"Model num_players ({model_np}) does not match "
                    f"evaluator num_players ({num_players})"
                )

    def _get_token_buf(self, n: int) -> np.ndarray:
        """Return a (n, num_tokens, token_dim) token buffer, reusing if large enough."""
        buf = self._token_buf
        if buf is None or buf.shape[0] < n:
            buf = np.empty(
                (n, self.num_tokens, self.token_dim), dtype=np.float32
            )
            self._token_buf = buf
        return buf[:n]

    @torch.inference_mode()
    def evaluate(
        self, state: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Evaluate a single game state with the neural network.

        Args:
            state: GameState object (non-terminal).

        Returns:
            Tuple of (sparse_priors, canonical_values, action_ids, n_legal, phase_id):
            - sparse_priors: shape (n_legal,) float32, softmax over legal actions.
            - canonical_values: shape (num_players,) float32, per-player values in
              canonical order (already non-rotated from the model).
            - action_ids: shape (n_legal,) uint16, phase-local legal action ids.
            - n_legal: count of legal actions at this state.
            - phase_id: decision phase id 0-7.
        """
        phase_id = get_decision_phase_py(state)

        # Fill a 1-row token buffer from the compact state.
        buf = self._get_token_buf(1)
        fill_token_buffer(state, buf[0])

        # Enumerate legal actions into a uint16 scratch buffer.
        action_ids = np.empty(K_MAX, dtype=np.uint16)
        n_legal = enumerate_legal_actions_py(state, action_ids)

        # Forward pass with autocast (if requested).
        x = torch.from_numpy(buf).to(self.device)
        phase_t = torch.tensor([phase_id], dtype=torch.long, device=self.device)
        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            policy_logits, value_output = self.model(x, phase_t)
            # Gather + softmax inside the autocast region so the 14977-wide
            # logits tensor stays in bf16/fp16 — up-casting it to f32 just
            # to softmax would undo most of the autocast win.
            idx = torch.from_numpy(
                action_ids[:n_legal].astype(np.int64)
            ).to(self.device).unsqueeze(0)  # (1, n_legal)
            gathered = policy_logits.gather(1, idx)  # (1, n_legal)
            priors = gathered.softmax(dim=1).to(torch.float32)

        # Check NaN on the full dense logits too — softmax would have
        # hidden any stray NaN outside the legal slice.
        logits_np = policy_logits.float().squeeze(0).cpu().numpy()
        values_np = value_output.float().squeeze(0).cpu().numpy()
        self._check_nan(logits_np, values_np)

        priors_np = priors.squeeze(0).cpu().numpy()
        return priors_np, values_np, action_ids[:n_legal].copy(), n_legal, phase_id

    @torch.inference_mode()
    def evaluate_batch(
        self, states: list[Any],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]]:
        """Evaluate multiple game states in a single NN forward pass.

        Args:
            states: List of GameState objects (non-terminal).

        Returns:
            List of (sparse_priors, canonical_values, action_ids, n_legal,
            phase_id) tuples, one per state. See ``evaluate`` for shapes.
        """
        n = len(states)
        if n == 0:
            return []
        if n == 1:
            return [self.evaluate(states[0])]

        phase_ids = [get_decision_phase_py(s) for s in states]
        buf = self._get_token_buf(n)
        for i, s in enumerate(states):
            fill_token_buffer(s, buf[i])

        # Zero-init — torch.gather below reads the full K_MAX width of this
        # buffer, so garbage past n_legals[i] can produce out-of-bounds
        # indices into the (batch, MAX_ACTION_SIZE) logits tensor.
        # enumerate_legal_actions_py only writes [:n_legal], so the zero
        # tail stands and is safely masked out downstream.
        action_ids_buf = np.zeros((n, K_MAX), dtype=np.uint16)
        n_legals = [0] * n
        for i, s in enumerate(states):
            n_legals[i] = enumerate_legal_actions_py(s, action_ids_buf[i])

        x = torch.from_numpy(buf).to(self.device)
        phase_t = torch.tensor(phase_ids, dtype=torch.long, device=self.device)
        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            policy_logits, value_output = self.model(x, phase_t)
            idx = torch.from_numpy(
                action_ids_buf.astype(np.int64)
            ).to(self.device)  # (n, K_MAX)
            gathered = policy_logits.gather(1, idx)  # (n, K_MAX)
            k_range = torch.arange(K_MAX, device=self.device)
            n_legals_t = torch.tensor(n_legals, dtype=torch.long, device=self.device)
            k_mask = k_range[None, :] < n_legals_t[:, None]
            gathered = gathered.masked_fill(~k_mask, -1e9)
            priors = gathered.softmax(dim=1).to(torch.float32)

        logits_np = policy_logits.float().cpu().numpy()
        values_np = value_output.float().cpu().numpy()
        self._check_nan(logits_np, values_np)
        priors_np = priors.cpu().numpy()
        return [
            (
                priors_np[i, :n_legals[i]].copy(),
                values_np[i],
                action_ids_buf[i, :n_legals[i]].copy(),
                n_legals[i],
                phase_ids[i],
            )
            for i in range(n)
        ]

    @torch.inference_mode()
    def evaluate_leaves(
        self,
        state_arrays: list[np.ndarray],
        phase_ids: list[int],
        action_ids_buf: np.ndarray,
        n_legals: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Evaluate pre-enumerated leaf data in a single NN forward pass.

        Optimized hot path for MCTS: the caller has already enumerated legal
        actions and computed phase ids during selection, so we just fill token
        buffers from the raw state arrays and forward. Mirrors the eval
        server's GPU gather + softmax, but in-process.

        Args:
            state_arrays: Raw int16 state arrays (pool row views), each
                ``(total_size,)``.
            phase_ids: Decision phase ids per leaf, length n.
            action_ids_buf: Legal phase-local action ids, shape
                ``(n, K_MAX)`` uint16. Only ``[i, :n_legals[i]]`` is read.
            n_legals: Count of legal actions per leaf, length n.

        Returns:
            List of ``(sparse_priors[:n_legal], canonical_values)`` tuples.
            Priors are softmaxed over the legal list; values are canonical.
        """
        # Deferred import: avoids circular top-level import in mcts_core
        # consumers, matches the single-import-point convention elsewhere.
        from core.state import GameState

        n = len(state_arrays)
        if n == 0:
            return []
        assert len(phase_ids) == n, f"{len(phase_ids)} phase_ids vs {n} states"
        assert len(n_legals) == n, f"{len(n_legals)} n_legals vs {n} states"
        assert action_ids_buf.shape == (n, K_MAX), \
            f"action_ids_buf shape {action_ids_buf.shape} != ({n}, {K_MAX})"

        # Fill token buffers from each state array. ``get_token_data`` needs
        # a GameState wrapper; rebind a scratch one per row (zero-copy).
        buf = self._get_token_buf(n)
        scratch_gs = GameState.from_buffer(state_arrays[0], self.num_players)
        fill_token_buffer(scratch_gs, buf[0])
        for i in range(1, n):
            scratch_gs.rebind(state_arrays[i], self.num_players)
            fill_token_buffer(scratch_gs, buf[i])

        x = torch.from_numpy(buf).to(self.device)
        phase_t = torch.tensor(phase_ids, dtype=torch.long, device=self.device)
        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            policy_logits, value_output = self.model(x, phase_t)
            idx = torch.from_numpy(
                action_ids_buf.astype(np.int64)
            ).to(self.device)  # (n, K_MAX)
            gathered = policy_logits.gather(1, idx)
            k_range = torch.arange(K_MAX, device=self.device)
            n_legals_t = torch.tensor(
                n_legals, dtype=torch.long, device=self.device,
            )
            k_mask = k_range[None, :] < n_legals_t[:, None]
            gathered = gathered.masked_fill(~k_mask, -1e9)
            priors = gathered.softmax(dim=1).to(torch.float32)

        logits_np = policy_logits.float().cpu().numpy()
        values_np = value_output.float().cpu().numpy()
        self._check_nan(logits_np, values_np)
        priors_np = priors.cpu().numpy()
        return [
            (priors_np[i, :n_legals[i]].copy(), values_np[i])
            for i in range(n)
        ]
