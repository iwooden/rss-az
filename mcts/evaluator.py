"""NN evaluator for MCTS leaf evaluation.

Fills token buffers from compact int16 game states via
``core.token_data.get_token_data``, runs NN inference, and returns
sparse softmax priors over legal actions plus canonical-order values.
No rotation — the transformer consumes non-rotated state and emits
values in canonical player order directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch._dynamo.decorators import mark_unbacked

from core.token_data import (
    get_num_tokens,
    TokenDataSize,
    get_token_data,
    get_token_data_batch,
)
from core.actions import (
    get_decision_phase_py,
    enumerate_legal_actions_py,
    MAX_LEGAL_ACTIONS_PY,
)
from entities.player import PLAYERS
from nn.transformer import NUM_PHASES

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


def fill_token_buffer_batch(
    state_arrays: list[np.ndarray], num_players: int, buf3d: np.ndarray,
) -> None:
    """Fill a ``(n, num_tokens, TOKEN_DIM)`` buffer from ``n`` state arrays.

    Thin wrapper around ``core.token_data.get_token_data_batch``. Reuses a
    single scratch ``GameState`` internally via ``rebind`` — amortizes
    per-state Python dispatch and wrapper construction into a single
    Cython entry. Mixed-player batches are not supported (``num_players``
    is shared across the batch).
    """
    get_token_data_batch(state_arrays, num_players, buf3d)


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


class NNEvaluator(BaseEvaluator):
    """Wraps a neural network model for in-process MCTS leaf evaluation.

    Used for single-process tests and anything that doesn't go through
    the shared-mem eval server. Fills a preallocated pinned-host token
    buffer from compact state, async-copies it to the device, and runs
    inference — the model returns already-gathered ``(B, K_MAX)`` sparse
    logits over each row's legal-action list, mirroring the eval server's
    GPU-side gather+softmax path.

    All per-batch scratch tensors (tokens, phase_ids, action_ids, n_legals)
    are preallocated pinned-host + device pairs, grown on demand. On a CPU
    device the "device" half aliases the host half and no copy runs.
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

        # Preallocated scratch — grows lazily via ``_ensure_scratch``.
        self._scratch_cap: int = 0
        # Persistent uint16 buffer for enumerate_legal_actions_py calls
        # (the Cython function takes a uint16 memoryview, not int64). One
        # row is enough: we enumerate per-state and copy into the batched
        # int64 action-id scratch.
        self._enum_scratch: np.ndarray = np.empty(K_MAX, dtype=np.uint16)

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

    # ------------------------------------------------------------------
    # Preallocated scratch
    # ------------------------------------------------------------------

    def _ensure_scratch(self, n: int) -> None:
        """Grow preallocated pinned-host + device scratch to fit ``n`` rows.

        Grows in powers of two so the common steady-state shrinks don't
        trigger reallocation. On CPU, the "device" tensors alias the host
        tensors and no H→D copy is needed.
        """
        if n <= self._scratch_cap:
            return
        cap = max(n, max(self._scratch_cap * 2, 1))
        pm = self.device.type == "cuda"
        nt, td = self.num_tokens, self.token_dim

        # Host (pinned on CUDA): exposed as numpy for Cython fills.
        self._tok_h = torch.empty((cap, nt, td), dtype=torch.float32, pin_memory=pm)
        self._tok_h_np = self._tok_h.numpy()
        self._phase_h = torch.empty(cap, dtype=torch.long, pin_memory=pm)
        self._phase_h_np = self._phase_h.numpy()
        self._aid_h = torch.empty((cap, K_MAX), dtype=torch.long, pin_memory=pm)
        self._aid_h_np = self._aid_h.numpy()
        self._nl_h = torch.empty(cap, dtype=torch.long, pin_memory=pm)
        self._nl_h_np = self._nl_h.numpy()

        # Device: separate on CUDA, aliased on CPU (no copy needed).
        if pm:
            self._tok_d = torch.empty((cap, nt, td), dtype=torch.float32, device=self.device)
            self._phase_d = torch.empty(cap, dtype=torch.long, device=self.device)
            self._aid_d = torch.empty((cap, K_MAX), dtype=torch.long, device=self.device)
            self._nl_d = torch.empty(cap, dtype=torch.long, device=self.device)
        else:
            self._tok_d = self._tok_h
            self._phase_d = self._phase_h
            self._aid_d = self._aid_h
            self._nl_d = self._nl_h

        self._scratch_cap = cap

    def _h2d(self, n: int) -> None:
        """Async copy preallocated host scratch [:n] into device scratch [:n].

        No-op on CPU (host and device scratch alias the same tensors).
        """
        if self.device.type != "cuda":
            return
        self._tok_d[:n].copy_(self._tok_h[:n], non_blocking=True)
        self._phase_d[:n].copy_(self._phase_h[:n], non_blocking=True)
        self._aid_d[:n].copy_(self._aid_h[:n], non_blocking=True)
        self._nl_d[:n].copy_(self._nl_h[:n], non_blocking=True)

    # ------------------------------------------------------------------
    # Evaluation API
    # ------------------------------------------------------------------

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
        n_legal = enumerate_legal_actions_py(state, self._enum_scratch)

        # Fill preallocated scratch row 0.
        self._ensure_scratch(1)
        fill_token_buffer(state, self._tok_h_np[0])
        self._phase_h_np[0] = phase_id
        self._aid_h_np[0, :n_legal] = self._enum_scratch[:n_legal]
        self._aid_h_np[0, n_legal:] = 0   # safe in-range gather indices
        self._nl_h_np[0] = n_legal

        priors_np, values_np = self._forward(1)
        return (
            priors_np[0, :n_legal].copy(),
            values_np[0],
            self._enum_scratch[:n_legal].copy(),
            n_legal,
            phase_id,
        )

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

        self._ensure_scratch(n)
        # Batched token fill — one Cython call amortizes the per-state
        # dispatch + GameState-rebind work into a single entry. Phase /
        # legal-action extraction still runs per state since they're
        # Python-side and each state has its own per-phase enumerator.
        fill_token_buffer_batch(
            [s._array for s in states], self.num_players, self._tok_h_np[:n],
        )
        phase_ids: list[int] = [0] * n
        n_legals: list[int] = [0] * n
        for i, s in enumerate(states):
            phase_ids[i] = get_decision_phase_py(s)
            nl = enumerate_legal_actions_py(s, self._enum_scratch)
            n_legals[i] = nl
            self._aid_h_np[i, :nl] = self._enum_scratch[:nl]
            self._aid_h_np[i, nl:] = 0
            self._nl_h_np[i] = nl
            self._phase_h_np[i] = phase_ids[i]

        priors_np, values_np = self._forward(n)

        # Return a per-row action_ids copy reflecting what the caller asked
        # for (each state's enumerated legals). The batched aid scratch
        # tail is masked out upstream, so copying it is equivalent.
        return [
            (
                priors_np[i, :n_legals[i]].copy(),
                values_np[i],
                # uint16 is the public contract for the legal list.
                self._aid_h_np[i, :n_legals[i]].astype(np.uint16),
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
        n = len(state_arrays)
        if n == 0:
            return []
        assert len(phase_ids) == n, f"{len(phase_ids)} phase_ids vs {n} states"
        assert len(n_legals) == n, f"{len(n_legals)} n_legals vs {n} states"
        assert action_ids_buf.shape == (n, K_MAX), \
            f"action_ids_buf shape {action_ids_buf.shape} != ({n}, {K_MAX})"

        self._ensure_scratch(n)
        # Fill token buffers in one batched Cython call — avoids the
        # per-state Python dispatch + wrapper construction that the prior
        # loop incurred. The batched entry reuses a single scratch
        # GameState internally via rebind.
        fill_token_buffer_batch(
            state_arrays, self.num_players, self._tok_h_np[:n],
        )

        # Copy phase_ids / action_ids / n_legals into pinned host scratch.
        # Numpy's assignment casts uint16→int64 for action_ids and handles
        # Python-list → int64 for phase_ids/n_legals.
        self._phase_h_np[:n] = phase_ids
        self._aid_h_np[:n] = action_ids_buf
        self._nl_h_np[:n] = n_legals

        priors_np, values_np = self._forward(n)
        return [
            (priors_np[i, :n_legals[i]].copy(), values_np[i])
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Shared forward (batched)
    # ------------------------------------------------------------------

    def _forward(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Async-copy host→device, run the model, return (priors, values) as numpy.

        Assumes host scratch rows [:n] have been filled with tokens,
        phase_ids, action_ids, and n_legals. The model returns already
        per-row gathered + masked logits of shape ``(n, K_MAX)`` — we just
        softmax and copy back to host.
        """
        self._h2d(n)
        phase_indices = self._build_phase_indices(n)

        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            logits, value_output = self.model(
                self._tok_d[:n], self._phase_d[:n],
                self._aid_d[:n], self._nl_d[:n], phase_indices,
            )
            priors = logits.softmax(dim=1).to(torch.float32)
            values = value_output.to(torch.float32)

        return priors.cpu().numpy(), values.cpu().numpy()

    def _build_phase_indices(self, n: int) -> list[torch.Tensor]:
        """Build per-phase int64 row indices on host, async-shipped to device.

        Computed from the already-filled ``_phase_h_np[:n]`` slice. Avoids
        the per-iteration host sync that boolean indexing inside the model
        would otherwise force (the masked-tensor size is data-dependent).
        On CPU device, the returned tensors share memory with the numpy
        arrays.

        Each returned tensor is marked ``unbacked`` on dim 0 so that
        ``torch.compile`` doesn't specialize on its size. Without this,
        every distinct combination of per-phase row counts (and there are
        many — up to ``C(N+7, 7)``) triggers a recompile, blowing past
        the default ``recompile_limit`` after a handful of batches.
        """
        phase_view = self._phase_h_np[:n]
        out: list[torch.Tensor] = []
        for p in range(NUM_PHASES):
            idx_np = np.nonzero(phase_view == p)[0].astype(np.int64, copy=False)
            t = torch.from_numpy(idx_np)
            if self.device.type == "cuda":
                t = t.to(self.device, non_blocking=True)
            mark_unbacked(t, 0)
            out.append(t)
        return out
