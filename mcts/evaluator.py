"""NN evaluator for MCTS leaf evaluation.

Fills token buffers and relation-attention planes from compact int16 game
states through the game's adapter (``games.Game``), runs NN inference, and
returns sparse softmax priors over the policy action set plus canonical-order
values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from nn.model_contract import ModelInputSpec, ModelKind, normalize_model_type

if TYPE_CHECKING:
    from games import Game


def _unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module for ``torch.compile`` wrappers."""
    return getattr(model, "_orig_mod", model)


def _transformer_input_spec(
    game: Game, num_players: int, layout_version: int = 2,
) -> ModelInputSpec:
    return ModelInputSpec(
        model_type=ModelKind.TRANSFORMER.value,
        num_players=num_players,
        policy_dim=game.policy.unified_dim,
        value_dim=num_players,
        num_tokens=game.num_tokens(num_players),
        token_dim=game.token_dim(layout_version),
        layout_version=layout_version,
    )


def _infer_model_input_spec(
    game: Game,
    model: torch.nn.Module,
    num_players: int,
) -> ModelInputSpec:
    """Infer the evaluator input contract from a model's config.

    Tests and utility call sites may construct ``NNEvaluator`` directly
    without a training config, so infer the player capacity from the model.
    """
    cfg = getattr(_unwrap_compiled_model(model), "cfg", None)
    if cfg is None:
        return _transformer_input_spec(game, num_players)

    model_np = int(getattr(cfg, "num_players", num_players))
    if model_np != num_players:
        raise ValueError(
            f"Model num_players ({model_np}) does not match "
            f"evaluator num_players ({num_players})"
        )

    return _transformer_input_spec(game, num_players, int(getattr(cfg, "layout_version", 2)))


def _validate_model_input_spec(
    game: Game,
    input_spec: ModelInputSpec,
    num_players: int,
) -> ModelInputSpec:
    """Validate the model/evaluator boundary before scratch allocation."""
    normalize_model_type(input_spec.model_type)
    if int(input_spec.num_players) != num_players:
        raise ValueError(
            f"input_spec num_players ({input_spec.num_players}) does not match "
            f"evaluator num_players ({num_players})"
        )
    if int(input_spec.policy_dim) != game.policy.unified_dim:
        raise ValueError(
            f"input_spec policy_dim ({input_spec.policy_dim}) does not match "
            f"the game's unified policy dim ({game.policy.unified_dim})"
        )
    if int(input_spec.value_dim) != num_players:
        raise ValueError(
            f"input_spec value_dim ({input_spec.value_dim}) does not match "
            f"num_players ({num_players})"
        )

    if int(input_spec.num_tokens) != game.num_tokens(num_players):
        raise ValueError(
            f"Transformer input_spec.num_tokens ({input_spec.num_tokens}) "
            f"does not match game.num_tokens({num_players})"
        )
    expected_dim = game.token_dim(input_spec.layout_version)
    if int(input_spec.token_dim) != expected_dim:
        raise ValueError(
            f"Transformer input_spec.token_dim ({input_spec.token_dim}) "
            f"does not match layout {input_spec.layout_version} width ({expected_dim})"
        )
    return input_spec


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
    RemoteEvaluator (shared-memory IPC to eval server). Evaluators accept
    compact int16 state and return sparse softmaxed priors plus canonical-order
    values.
    """

    def __init__(
        self, game: Game, num_players: int, terminal_rank_weight: float = 0.5,
    ) -> None:
        self.game = game
        self.num_players = num_players
        self.terminal_rank_weight = terminal_rank_weight
        self.num_tokens = game.num_tokens(num_players)
        self._num_players_from_row = game.num_players_from_row

    def _actual_num_players(self, state: Any) -> int:
        """Read the actual player count from a game state."""
        return self._num_players_from_row(state._array)

    def _actual_num_players_for_rows(self, state_arrays: list[np.ndarray]) -> int:
        """Return the shared actual count for a search batch of state rows."""
        actual = self._num_players_from_row(state_arrays[0])
        for row in state_arrays[1:]:
            row_actual = self._num_players_from_row(row)
            if row_actual != actual:
                raise ValueError(
                    "evaluate_leaves expects one actual player count per MCTS "
                    f"batch, got {actual} and {row_actual}"
                )
        return actual

    def evaluate_terminal(self, state: Any) -> np.ndarray:
        """Compute terminal values from a game-over state.

        Args:
            state: Terminal game state.

        Returns:
            Canonical values, shape (num_players,).
        """
        scores = self.game.final_scores(state)
        return compute_terminal_values(
            scores, len(scores), self.terminal_rank_weight
        )


class NNEvaluator(BaseEvaluator):
    """Wraps a neural network model for in-process MCTS leaf evaluation.

    Used for single-process tests and anything that doesn't go through
    the shared-mem eval server. Fills preallocated pinned-host model input
    token and relation buffers plus a ``(unified_dim,)`` legal mask from
    compact state, async-copies to the device, and runs inference. The model returns dense
    ``(B, unified_dim)`` logits with illegal slots at -1e9; we softmax
    on-device then gather the per-leaf legal prior slice using the same
    ``action_lut`` the mask was built from.

    All per-batch scratch tensors are preallocated pinned-host + device pairs,
    grown on demand. On a CPU device the "device" half aliases the host half
    and no copy runs.
    """

    _DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}

    def __init__(self, game: Game, model: torch.nn.Module, device: torch.device,
                 num_players: int = 3, *,
                 terminal_rank_weight: float = 0.5,
                 eval_dtype: str | None = None,
                 input_spec: ModelInputSpec | None = None) -> None:
        super().__init__(game, num_players, terminal_rank_weight)
        self.model = model
        self.device = device
        self._autocast_dtype = self._DTYPE_MAP.get(eval_dtype) if eval_dtype else None
        self.model.eval()
        inferred_spec = _infer_model_input_spec(game, model, num_players)
        if input_spec is None:
            input_spec = inferred_spec
        elif input_spec != inferred_spec:
            raise ValueError("input_spec does not match the model input contract")
        self.input_spec = _validate_model_input_spec(game, input_spec, num_players)
        self.token_dim = input_spec.token_dim
        self.layout_version = input_spec.layout_version
        self._unified_dim = game.policy.unified_dim

        # Preallocated scratch — grows lazily via ``_ensure_scratch``.
        self._scratch_cap: int = 0
        # Persistent uint16 buffer for enumerate_policy_actions calls.
        self._enum_scratch: np.ndarray = np.empty(
            game.policy.max_action_size, dtype=np.uint16,
        )
        # Worker-local LUT mapping (phase_id, phase_local_action_id) to a
        # slot in the unified logit vector. Used in evaluate()/evaluate_batch
        # to set mask bits pre-forward and gather sparse priors post-forward;
        # the dense ``evaluate_leaves`` path bypasses it entirely.
        self._action_lut_np: np.ndarray = game.policy.build_action_lut().numpy()

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
        nr = self.game.num_relations
        u_dim = self._unified_dim

        # Host (pinned on CUDA): exposed as numpy for the mask scatter.
        # Mask is zero-initialized so ``_build_mask_row`` only touches
        # legal slots — the full row is reset on each fill via a
        # contiguous ``row[:] = 0`` before the scatter.
        self._tok_h = torch.empty((cap, nt, td), dtype=torch.float32, pin_memory=pm)
        self._tok_h_np = self._tok_h.numpy()
        self._rel_h = torch.empty((cap, nr, nt, nt), dtype=torch.uint8, pin_memory=pm)
        self._rel_h_np = self._rel_h.numpy()
        self._mask_h = torch.zeros(
            (cap, u_dim), dtype=torch.bool, pin_memory=pm,
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
                (cap, u_dim), dtype=torch.bool, device=self.device,
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
        policy slots to flip to True.
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
            state: Game state (non-terminal).

        Returns:
            Tuple of (sparse_priors, canonical_values, action_ids, n_legal, phase_id):
            - sparse_priors: shape (n_legal,) float32, softmax over legal actions.
            - canonical_values: shape (num_players,) float32, per-player values in
              canonical order (already non-rotated from the model).
            - action_ids: shape (n_legal,) uint16, phase-local legal action ids
              from ``game.enumerate_policy_actions`` (the state's runtime
              flags, such as RSS's ACQ price cap, apply).
            - n_legal: count of legal actions at this state.
            - phase_id: decision phase id.
        """
        game = self.game
        actual_num_players = self._actual_num_players(state)
        phase_id = game.decision_phase(state)
        n_legal = game.enumerate_policy_actions(state, self._enum_scratch)

        # Fill preallocated scratch row 0.
        self._ensure_scratch(1)
        game.fill_tokens(
            state, self._tok_h_np[0], max_players=self.num_players,
            layout_version=self.layout_version,
        )
        game.fill_relations_dense(
            state, self._rel_h_np[0], max_players=self.num_players,
        )
        self._build_mask_row(0, phase_id, self._enum_scratch[:n_legal])

        priors_np, values_np = self._forward(1)
        slots = self._action_lut_np[phase_id, self._enum_scratch[:n_legal]]
        return (
            priors_np[0, slots].copy(),
            values_np[0, :actual_num_players].copy(),
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
            states: List of game states (non-terminal).

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
        # Batched model-input fill — one Cython call amortizes the per-state
        # dispatch + state-rebind work into a single entry. Phase /
        # legal-action extraction still runs per state since they're
        # Python-side and each state has its own per-phase enumerator.
        game = self.game
        state_arrays = [s._array for s in states]
        game.fill_tokens_batch(
            state_arrays, self._tok_h_np[:n],
            max_players=self.num_players,
            layout_version=self.layout_version,
        )
        game.fill_relations_dense_batch(
            state_arrays, self._rel_h_np[:n],
            max_players=self.num_players,
        )
        phase_ids: list[int] = [0] * n
        n_legals: list[int] = [0] * n
        # Buffer to hold the enumerated ids across rows so we can gather
        # priors after forward without re-enumerating.
        all_action_ids = np.empty((n, self._enum_scratch.shape[0]), dtype=np.uint16)
        for i, s in enumerate(states):
            phase_ids[i] = game.decision_phase(s)
            nl = game.enumerate_policy_actions(s, self._enum_scratch)
            n_legals[i] = nl
            all_action_ids[i, :nl] = self._enum_scratch[:nl]
            self._build_mask_row(i, phase_ids[i], self._enum_scratch[:nl])

        priors_np, values_np = self._forward(n)
        results: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]] = []
        for i in range(n):
            nl = n_legals[i]
            actual_num_players = self._actual_num_players(states[i])
            slots = self._action_lut_np[phase_ids[i], all_action_ids[i, :nl]]
            results.append((
                priors_np[i, slots].copy(),
                values_np[i, :actual_num_players].copy(),
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
        ``(n, unified_dim)`` legal mask during selection, so we just
        fill model input buffers from the raw state arrays, copy the mask
        into device scratch, and forward. Mirrors the eval server's GPU mask
        + softmax, but in-process.

        Args:
            state_arrays: Raw int16 state arrays (pool row views), each
                ``(total_size,)``.
            legal_mask: Dense legal slot mask, shape
                ``(n, unified_dim)`` uint8 or bool. One row per leaf.

        Returns:
            ``(priors, values)`` where ``priors`` is
            ``(n, unified_dim)`` float32 (softmaxed; illegal slots
            collapse to ~0) and ``values`` is ``(n, num_players)`` float32
            in canonical order. The caller gathers the per-leaf legal
            prior slice (via whatever LUT they used to build the mask).
        """
        n = len(state_arrays)
        u_dim = self._unified_dim
        if n == 0:
            return (
                np.empty((0, u_dim), dtype=np.float32),
                np.empty((0, self.num_players), dtype=np.float32),
            )
        actual_num_players = self._actual_num_players_for_rows(state_arrays)
        assert legal_mask.shape == (n, u_dim), \
            f"legal_mask shape {legal_mask.shape} != ({n}, {u_dim})"

        self._ensure_scratch(n)
        # Fill model input buffers in one batched Cython call — avoids the
        # per-state Python dispatch + wrapper construction that the prior
        # loop incurred. The batched entry reuses a single scratch
        # state internally via rebind.
        self.game.fill_tokens_batch(
            state_arrays, self._tok_h_np[:n],
            max_players=self.num_players,
            layout_version=self.layout_version,
        )
        self.game.fill_relations_dense_batch(
            state_arrays, self._rel_h_np[:n],
            max_players=self.num_players,
        )
        # Copy caller's dense mask into host scratch — cheaper than
        # per-row LUT scatter because the caller built it once already.
        np.copyto(self._mask_h_np[:n], legal_mask, casting="unsafe")

        priors_np, values_np = self._forward(n)
        return priors_np, values_np[:, :actual_num_players].copy()

    # ------------------------------------------------------------------
    # Shared forward (batched)
    # ------------------------------------------------------------------

    def _forward(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Async-copy host→device, run the model, return (priors, values) as numpy.

        Assumes host scratch rows [:n] have been filled with model inputs and
        legal masks. The model returns dense ``(n, unified_dim)``
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
