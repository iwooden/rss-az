"""NN evaluator for MCTS leaf evaluation.

Fills token buffers from compact int16 game states via
``core.token_data.get_token_data`` plus relation-attention planes via
``core.relations.get_relation_data``, runs NN inference, and returns sparse
softmax priors over legal actions plus canonical-order values. No rotation —
the transformer consumes non-rotated state and emits values in canonical
player order directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.relations import get_relation_data, get_relation_data_batch
from core.token_data import (
    get_num_tokens,
    TokenDataSize,
    get_token_data,
    get_token_data_batch,
)
from core.actions import (
    get_decision_phase_py,
    enumerate_legal_actions_py,
)
from core.data import MAX_ACTION_SIZE
from entities.player import PLAYERS
from nn.transformer import UNIFIED_LOGIT_DIM, build_action_lut

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


def fill_relation_buffer(state: Any, buf3d: np.ndarray) -> None:
    """Fill a ``(num_relations, num_tokens, num_tokens)`` uint8 relation buffer."""
    get_relation_data(state, buf3d)


def fill_relation_buffer_batch(
    state_arrays: list[np.ndarray], num_players: int, buf4d: np.ndarray,
) -> None:
    """Fill a ``(n, num_relations, num_tokens, num_tokens)`` relation buffer."""
    get_relation_data_batch(state_arrays, num_players, buf4d)


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
    buffer + ``(UNIFIED_LOGIT_DIM,)`` legal mask from compact state,
    async-copies to the device, and runs inference. The model returns
    dense ``(B, UNIFIED_LOGIT_DIM)`` logits with illegal slots at -1e9;
    we softmax on-device then gather the per-leaf legal prior slice
    using the same ``action_lut`` the mask was built from.

    All per-batch scratch tensors (tokens, relation planes, masks) are preallocated
    pinned-host + device pairs, grown on demand. On a CPU device the
    "device" half aliases the host half and no copy runs.
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
        # Persistent uint16 buffer for enumerate_legal_actions_py calls.
        self._enum_scratch: np.ndarray = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
        # Worker-local LUT mapping (phase_id, phase_local_action_id) to a
        # slot in the unified logit vector. Used in evaluate()/evaluate_batch
        # to set mask bits pre-forward and gather sparse priors post-forward;
        # the dense ``evaluate_leaves`` path bypasses it entirely.
        self._action_lut_np: np.ndarray = build_action_lut().numpy()

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
        nr = NUM_ATTENTION_RELATIONS

        # Host (pinned on CUDA): exposed as numpy for the mask scatter.
        # Mask is zero-initialized so ``_build_mask_row`` only touches
        # legal slots — the full row is reset on each fill via a
        # contiguous ``row[:] = 0`` before the scatter.
        self._tok_h = torch.empty((cap, nt, td), dtype=torch.float32, pin_memory=pm)
        self._tok_h_np = self._tok_h.numpy()
        self._rel_h = torch.empty((cap, nr, nt, nt), dtype=torch.uint8, pin_memory=pm)
        self._rel_h_np = self._rel_h.numpy()
        self._mask_h = torch.zeros(
            (cap, UNIFIED_LOGIT_DIM), dtype=torch.bool, pin_memory=pm,
        )
        self._mask_h_np = self._mask_h.numpy()

        # Device: separate on CUDA, aliased on CPU (no copy needed).
        if pm:
            self._tok_d = torch.empty(
                (cap, nt, td), dtype=torch.float32, device=self.device,
            )
            self._rel_d = torch.empty(
                (cap, nr, nt, nt), dtype=torch.uint8, device=self.device,
            )
            self._mask_d = torch.empty(
                (cap, UNIFIED_LOGIT_DIM), dtype=torch.bool, device=self.device,
            )
        else:
            self._tok_d = self._tok_h
            self._rel_d = self._rel_h
            self._mask_d = self._mask_h

        self._scratch_cap = cap

    def _build_mask_row(
        self, row_idx: int, phase_id: int, action_ids_legal: np.ndarray,
    ) -> None:
        """Zero host mask row ``row_idx``, then mark legal unified slots.

        ``action_ids_legal`` is the phase-local id slice for this leaf;
        ``action_lut[phase_id, action_ids_legal]`` gives the unified
        ``UNIFIED_LOGIT_DIM``-wide slots to flip to True.
        """
        self._mask_h_np[row_idx] = False
        slots = self._action_lut_np[phase_id, action_ids_legal]
        self._mask_h_np[row_idx, slots] = True

    def _h2d(self, n: int) -> None:
        """Async copy preallocated host scratch [:n] into device scratch [:n].

        No-op on CPU (host and device scratch alias the same tensors).
        """
        if self.device.type != "cuda":
            return
        self._tok_d[:n].copy_(self._tok_h[:n], non_blocking=True)
        self._rel_d[:n].copy_(self._rel_h[:n], non_blocking=True)
        self._mask_d[:n].copy_(self._mask_h[:n], non_blocking=True)

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
            - phase_id: decision phase id 0-10.
        """
        phase_id = get_decision_phase_py(state)
        n_legal = enumerate_legal_actions_py(state, self._enum_scratch)

        # Fill preallocated scratch row 0.
        self._ensure_scratch(1)
        fill_token_buffer(state, self._tok_h_np[0])
        fill_relation_buffer(state, self._rel_h_np[0])
        self._build_mask_row(0, phase_id, self._enum_scratch[:n_legal])

        priors_np, values_np = self._forward(1)
        slots = self._action_lut_np[phase_id, self._enum_scratch[:n_legal]]
        return (
            priors_np[0, slots].copy(),
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
        fill_relation_buffer_batch(
            [s._array for s in states], self.num_players, self._rel_h_np[:n],
        )
        phase_ids: list[int] = [0] * n
        n_legals: list[int] = [0] * n
        # Buffer to hold the enumerated ids across rows so we can gather
        # priors after forward without re-enumerating.
        all_action_ids = np.empty((n, MAX_ACTION_SIZE), dtype=np.uint16)
        for i, s in enumerate(states):
            phase_ids[i] = get_decision_phase_py(s)
            nl = enumerate_legal_actions_py(s, self._enum_scratch)
            n_legals[i] = nl
            all_action_ids[i, :nl] = self._enum_scratch[:nl]
            self._build_mask_row(i, phase_ids[i], self._enum_scratch[:nl])

        priors_np, values_np = self._forward(n)
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
        for i in range(n):
            nl = n_legals[i]
            slots = self._action_lut_np[phase_ids[i], all_action_ids[i, :nl]]
            results.append((
                priors_np[i, slots].copy(),
                values_np[i],
                all_action_ids[i, :nl].copy(),
                nl,
                phase_ids[i],
            ))
        return results

    @torch.inference_mode()
    def evaluate_leaves(
        self,
        state_arrays: list[np.ndarray],
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate pre-masked leaf data in a single NN forward pass.

        Optimized hot path for MCTS: the caller has already built the dense
        ``(n, UNIFIED_LOGIT_DIM)`` legal mask during selection, so we just
        fill token buffers from the raw state arrays, copy the mask into
        device scratch, and forward. Mirrors the eval server's GPU mask
        + softmax, but in-process.

        Args:
            state_arrays: Raw int16 state arrays (pool row views), each
                ``(total_size,)``.
            legal_mask: Dense legal slot mask, shape
                ``(n, UNIFIED_LOGIT_DIM)`` uint8 or bool. One row per leaf.

        Returns:
            ``(priors, values)`` where ``priors`` is
            ``(n, UNIFIED_LOGIT_DIM)`` float32 (softmaxed; illegal slots
            collapse to ~0) and ``values`` is ``(n, num_players)`` float32
            in canonical order. The caller gathers the per-leaf legal
            prior slice (via whatever LUT they used to build the mask).
        """
        n = len(state_arrays)
        if n == 0:
            return (
                np.empty((0, UNIFIED_LOGIT_DIM), dtype=np.float32),
                np.empty((0, self.num_players), dtype=np.float32),
            )
        assert legal_mask.shape == (n, UNIFIED_LOGIT_DIM), \
            f"legal_mask shape {legal_mask.shape} != ({n}, {UNIFIED_LOGIT_DIM})"

        self._ensure_scratch(n)
        # Fill token buffers in one batched Cython call — avoids the
        # per-state Python dispatch + wrapper construction that the prior
        # loop incurred. The batched entry reuses a single scratch
        # GameState internally via rebind.
        fill_token_buffer_batch(
            state_arrays, self.num_players, self._tok_h_np[:n],
        )
        fill_relation_buffer_batch(
            state_arrays, self.num_players, self._rel_h_np[:n],
        )
        # Copy caller's dense mask into host scratch — cheaper than
        # per-row LUT scatter because the caller built it once already.
        np.copyto(self._mask_h_np[:n], legal_mask, casting="unsafe")

        return self._forward(n)

    # ------------------------------------------------------------------
    # Shared forward (batched)
    # ------------------------------------------------------------------

    def _forward(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Async-copy host→device, run the model, return (priors, values) as numpy.

        Assumes host scratch rows [:n] have been filled with tokens and
        legal masks. The model returns dense ``(n, UNIFIED_LOGIT_DIM)``
        logits with illegal slots at -1e9; we softmax on-device so illegal
        slots collapse to ~0 and the legal slots carry a proper distribution.
        Callers that need a sparse legal slice pick it out of the returned
        numpy array via the same ``action_lut`` used to build the mask.
        """
        self._h2d(n)

        with torch.autocast(self.device.type, dtype=self._autocast_dtype,
                            enabled=self._autocast_dtype is not None):
            logits, value_output = self.model(
                self._tok_d[:n], self._mask_d[:n], self._rel_d[:n],
            )
            priors = logits.softmax(dim=1).to(torch.float32)
            values = value_output.to(torch.float32)

        return priors.cpu().numpy(), values.cpu().numpy()
