"""MCTS search algorithm with PUCT selection and A0GB value targets.

Implements AlphaZero-style MCTS for multiplayer games:
- Vectorized PUCT selection with FPU parent value initialization
- Lazy node expansion (children allocated on first visit only)
- Dirichlet noise at the root for exploration
- A0GB greedy backup for value targets (Willemsen et al., 2020)
- Pre-allocated state pool for zero per-node allocation
- Batched leaf evaluation with leaf-lock deduplication for GPU throughput
- Lock propagation: when all children of a node are locked, the parent edge
  is locked too, preventing wasted PUCT selections into fully-locked subtrees
- Subtree reuse: reuse the chosen child's subtree as the next search root
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from train.profile_stats import SearchStats

import numpy as np

from train.config import MCTSConfig
from mcts.node import MCTSNode
from mcts.mcts_core import (
    backup as _backup,
    increment_visits as _increment_visits,
    virtual_backup as _virtual_backup,
    all_col0_neg_inf as _all_col0_neg_inf,
    descend_path as _descend_path,
    propagate_lock as _propagate_lock,
    propagate_unlock as _propagate_unlock,
    DESCEND_NEED_NEW_CHILD,
    DESCEND_VIRTUAL_BACKUP,
    DESCEND_EXISTING_LEAF,
)
from core.actions import (
    get_decision_phase_py,
    enumerate_policy_actions_py,
)
from core.data import DecisionPhase, MAX_ACTION_SIZE, GamePhases
from core.driver import DRIVER, STATUS_GAME_OVER_PY, STATUS_INVALID_PY
from core.state import GameState, get_layout, get_storage_player_capacity
from entities.turn import TURN
from nn.transformer import UNIFIED_LOGIT_DIM, build_action_lut


U_DIM = int(UNIFIED_LOGIT_DIM)
_GREEDY_TEMPERATURE = 1e-8
_MAX_BAD_VALUES_TO_REPORT = 8


class MCTSNonFiniteError(RuntimeError):
    """Raised when MCTS observes NaN or +/-inf in values or priors."""


def _context_prefix(debug_context: str | None) -> str:
    return f"{debug_context}: " if debug_context else ""


def _array_nonfinite_summary(name: str, array: Any) -> str:
    arr = np.asarray(array)
    bad = np.argwhere(~np.isfinite(arr))
    parts = [
        f"{name} shape={arr.shape}",
        f"dtype={arr.dtype}",
        f"nonfinite={len(bad)}",
    ]
    if len(bad) > 0:
        shown = bad[:_MAX_BAD_VALUES_TO_REPORT]
        samples: list[str] = []
        for idx in shown:
            idx_tuple = tuple(int(i) for i in idx)
            samples.append(f"{idx_tuple}={arr[idx_tuple]!r}")
        parts.append(f"samples=[{', '.join(samples)}]")
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        parts.append(f"finite_min={float(finite.min()):.6g}")
        parts.append(f"finite_max={float(finite.max()):.6g}")
    return ", ".join(parts)


def _raise_if_nonfinite(
    array: Any,
    *,
    name: str,
    debug_context: str | None,
    extra: str = "",
) -> None:
    if np.isfinite(array).all():
        return
    suffix = f"; {extra}" if extra else ""
    raise MCTSNonFiniteError(
        f"{_context_prefix(debug_context)}non-finite MCTS data at {name}: "
        f"{_array_nonfinite_summary(name, array)}{suffix}"
    )


def _path_summary(path: _Path) -> str:
    if not path:
        return "path=[]"
    pieces: list[str] = []
    for depth, (node, action_idx, array_idx) in enumerate(path):
        visits = "?"
        if node.visit_counts is not None and 0 <= array_idx < len(node.visit_counts):
            visits = str(int(node.visit_counts[array_idx]))
        pieces.append(
            f"d{depth}:active=P{node.active_player_id},"
            f"action={int(action_idx)},row={int(array_idx)},visits={visits}"
        )
    return "path=[" + " -> ".join(pieces) + "]"


def _action_row_summary(
    node: MCTSNode,
    array_idx: int,
    *,
    label: str,
) -> str:
    action = "?"
    visits = "?"
    if node.legal_actions is not None and 0 <= array_idx < len(node.legal_actions):
        action = str(int(node.legal_actions[array_idx]))
    if node.visit_counts is not None and 0 <= array_idx < len(node.visit_counts):
        visits = str(int(node.visit_counts[array_idx]))
    return (
        f"{label}_active=P{node.active_player_id}, "
        f"{label}_action={action}, {label}_row={int(array_idx)}, "
        f"{label}_visits={visits}"
    )


def _raise_pending_batch_if_nonfinite(
    array: Any,
    *,
    name: str,
    pending: list[tuple[_Path, MCTSNode]],
    debug_context: str | None,
    sim: int,
) -> None:
    arr = np.asarray(array)
    if np.isfinite(arr).all():
        return

    bad = np.argwhere(~np.isfinite(arr))
    batch_row = int(bad[0][0]) if bad.size > 0 and bad.shape[1] > 0 else -1
    extra = f"sim={sim}; n_pending={len(pending)}"
    if 0 <= batch_row < len(pending):
        path, node = pending[batch_row]
        extra += (
            f"; batch_row={batch_row}; leaf_state_idx={node.state_idx}; "
            f"leaf_pending_phase={node.pending_phase}; "
            f"leaf_pending_n={node.pending_n}; {_path_summary(path)}"
        )
    raise MCTSNonFiniteError(
        f"{_context_prefix(debug_context)}non-finite MCTS data at {name}: "
        f"{_array_nonfinite_summary(name, arr)}; {extra}"
    )


class StatePool:
    """Pre-allocated matrix for MCTS node game states.

    Created once and reused across all MCTS searches for the training
    lifetime. Each row stores one node's full compact int16 game state.
    ``reset()`` is called at the start of each run_search to reuse the
    same memory. The pool also owns per-search scratch buffers
    (``_legal_scratch``, ``_pending_action_ids_buf``,
    ``_pending_legal_mask_buf``, ``_pending_n_buf``, ``_pending_phase_buf``,
    ``_saved_values_buf``, ``_path_pool``) so ``run_search`` never
    allocates per simulation. The per-search ``_action_lut_np`` is owned
    here too so the mask scatter + post-forward sparse gather don't
    re-build the LUT on every search.
    """

    __slots__ = (
        "states",
        "_next",
        "_state_size",
        "_max_players",
        "_action_lut_np",
        "_legal_scratch",
        "_pending_action_ids_buf",
        "_pending_legal_mask_buf",
        "_pending_n_buf",
        "_pending_phase_buf",
        "_saved_values_buf",
        "_path_pool",
    )

    def __init__(self, capacity: int, state_size: int) -> None:
        # Compact int16 state storage. Evaluators build model-specific
        # buffers lazily from these rows: transformer token/relation tensors
        # or dense ResNet vectors. The pool itself only owns raw canonical
        # state arrays.
        self.states = np.zeros((capacity, state_size), dtype=np.int16)
        self._next = 0
        self._state_size = state_size
        self._max_players = get_storage_player_capacity(state_size)
        # (phase_id, phase-local action id) → unified-slot LUT. Used both
        # to scatter the dense legal mask per leaf and to gather the sparse
        # prior slice out of the server's dense priors for node.expand.
        self._action_lut_np: np.ndarray = build_action_lut().numpy()
        # Per-leaf enumerate scratch — written by enumerate_policy_actions_py,
        # then copied directly into the per-leaf row of _pending_action_ids_buf
        # at child creation time (no intermediate per-node allocation).
        self._legal_scratch = np.empty(MAX_ACTION_SIZE, dtype=np.uint16)
        # Packed (batch_size, MAX_ACTION_SIZE) buffer + int32 n_legals /
        # phase arrays carried through the batch. ``_pending_legal_mask_buf``
        # is the dense (batch_size, UNIFIED_LOGIT_DIM) uint8 mask fed to
        # evaluate_leaves. Lazy-allocated because batch_size isn't known
        # at pool construction.
        self._pending_action_ids_buf: np.ndarray | None = None
        self._pending_legal_mask_buf: np.ndarray | None = None
        self._pending_n_buf: np.ndarray | None = None
        self._pending_phase_buf: np.ndarray | None = None
        # Per-batch saved-Q rows for leaf-lock unlock. One row per pending
        # slot, memcpy'd from parent.value_sums at lock time and back at
        # unlock time — no per-leaf np.copy() allocation.
        self._saved_values_buf: np.ndarray | None = None
        # Preallocated path lists indexed by pending slot. ``_descend_path``
        # writes into ``_path_pool[len(pending)]`` (cleared before use); on
        # commit, that slot is appended into ``pending`` directly — no
        # per-simulation list allocation or tuple churn.
        self._path_pool: list[list[tuple[MCTSNode, int, int]]] | None = None

    def reset(self) -> None:
        """Reset the write cursor for a new search."""
        self._next = 0

    def alloc(self, source: np.ndarray) -> int:
        """Copy source array into the next pool row, return its index."""
        idx = self._next
        self.states[idx] = source
        self._next += 1
        return idx

    def alloc_from_row(self, src_idx: int) -> int:
        """Copy an existing pool row to the next slot, return the new index."""
        idx = self._next
        self.states[idx] = self.states[src_idx]
        self._next += 1
        return idx

    def compact(self, nodes: list[MCTSNode]) -> None:
        """Compact pool to retain only states for the given nodes.

        Copies retained states to the front of the pool and updates
        each node's state_idx to its new position.

        Args:
            nodes: Nodes to retain, sorted by state_idx ascending.
                Ascending order ensures copies never overwrite unread sources.
        """
        for new_idx, node in enumerate(nodes):
            if new_idx != node.state_idx:
                self.states[new_idx] = self.states[node.state_idx]
            node.state_idx = new_idx
        self._next = len(nodes)

    def row(self, idx: int) -> np.ndarray:
        """Return a view (not copy) of the given pool row."""
        return self.states[idx]

    def ensure_pending_bufs(self, batch_size: int, num_players: int) -> None:
        """Grow the per-batch scratch buffers to fit ``batch_size`` leaves."""
        buf = self._pending_action_ids_buf
        if buf is None or buf.shape[0] < batch_size:
            self._pending_action_ids_buf = np.empty(
                (batch_size, MAX_ACTION_SIZE), dtype=np.uint16,
            )
            self._pending_legal_mask_buf = np.empty(
                (batch_size, U_DIM), dtype=np.uint8,
            )
            self._pending_n_buf = np.empty(batch_size, dtype=np.int32)
            self._pending_phase_buf = np.empty(batch_size, dtype=np.int32)
        sv = self._saved_values_buf
        if sv is None or sv.shape[0] < batch_size or sv.shape[1] < num_players:
            self._saved_values_buf = np.empty(
                (batch_size, num_players), dtype=np.float32,
            )
        pp = self._path_pool
        if pp is None:
            self._path_pool = [[] for _ in range(batch_size)]
        elif len(pp) < batch_size:
            pp.extend([] for _ in range(batch_size - len(pp)))


def _add_dirichlet_noise(
    node: MCTSNode, alpha: float, epsilon: float, rng: np.random.Generator,
) -> None:
    """Add Dirichlet noise to the root node's action priors.

    Modifies priors in-place:
        P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)
    """
    assert node.priors is not None
    noise = rng.dirichlet([alpha] * len(node.priors))
    node.priors = ((1 - epsilon) * node.priors + epsilon * noise).astype(np.float32)


def _filter_acq_price_root_priors(
    priors: np.ndarray,
    action_ids: np.ndarray,
    n_legal: int,
    phase_id: int,
    max_acq_price_actions: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply the ACQ price edge cap to an already-softmaxed root eval."""
    if (
        max_acq_price_actions <= 0
        or phase_id != int(DecisionPhase.DPHASE_ACQ_SELECT_PRICE)
        or n_legal <= max_acq_price_actions
    ):
        return priors, action_ids, n_legal

    edge_count = max_acq_price_actions // 2
    keep = np.empty(max_acq_price_actions, dtype=np.intp)
    keep[:edge_count] = np.arange(edge_count, dtype=np.intp)
    keep[edge_count:] = np.arange(
        n_legal - edge_count,
        n_legal,
        dtype=np.intp,
    )
    filtered_priors = priors[keep].astype(np.float32, copy=True)
    prior_sum = float(filtered_priors.sum())
    if prior_sum > 0.0:
        filtered_priors /= prior_sum
    else:
        filtered_priors.fill(1.0 / float(max_acq_price_actions))
    return filtered_priors, action_ids[keep].copy(), max_acq_price_actions


# Type alias for a selection path: list of (parent_node, action, array_idx)
_Path = list[tuple[MCTSNode, int, int]]


def run_search(
    root_state: Any | None,
    evaluator: Any,
    config: MCTSConfig,
    rng: np.random.Generator | None = None,
    state_pool: StatePool | None = None,
    reuse_root: MCTSNode | None = None,
    profile: SearchStats | None = None,
    debug_context: str | None = None,
) -> MCTSNode:
    """Run MCTS search from the given root state.

    Supports batched leaf evaluation: multiple leaves are selected per
    iteration and evaluated in a single NN forward pass for GPU throughput.

    Batching uses a leaf-lock mechanism to prevent duplicate evaluation:
    when a leaf is queued for evaluation, its Q value in the parent is set
    to -inf, ensuring PUCT never re-selects the same leaf within a batch.
    When all children of a node become locked, the lock propagates up to
    that node's parent edge, preventing wasted PUCT selections into
    fully-locked subtrees. If all root edges become locked, the batch is
    submitted immediately (no more useful search is possible).
    Visit counts along the selection path are incremented at selection time
    to gently nudge subsequent selections toward less-explored branches.
    After evaluation, leaf locks and any propagation locks are removed
    (order-independent) and values are backed up.

    Args:
        root_state: GameState object to search from. Required for fresh
            searches, ignored when reuse_root is provided (the reused
            subtree's pooled states are authoritative).
        evaluator: NNEvaluator (in-process) or RemoteEvaluator (shared-mem IPC).
            Both return sparse-priors 5-tuples from ``evaluate`` and sparse
            ``(priors, values)`` pairs from ``evaluate_leaves``.
        config: Search hyperparameters (search_batch_size controls batching).
        rng: Optional numpy random Generator for Dirichlet noise.
            If None, creates an unseeded generator.
        state_pool: Optional pre-allocated StatePool for node state storage.
            If None, a temporary pool is created. For training, pass a
            persistent pool to avoid per-search allocation.
            Required when reuse_root is provided.
        reuse_root: Optional pre-searched subtree root from the previous move.
            When provided, the root's visit statistics have been reset by
            prepare_reuse_root() so Dirichlet noise is fully effective.
            Actions whose children have existing visits trigger lightweight
            "virtual backups" (child Q echoed to root) until the root's
            per-action count catches up, then real search begins.
            The state_pool must already be compacted for this subtree.
        debug_context: Optional caller-supplied context included in
            fail-fast non-finite diagnostics.

    Returns:
        The root MCTSNode with search statistics populated.
    """
    num_players = config.num_players
    batch_size = config.search_batch_size
    check_nonfinite = config.check_nonfinite

    if reuse_root is not None:
        # Reuse requires the matching pool — a fresh pool would make the
        # reused nodes' state_idx values point into uninitialized memory.
        if state_pool is None:
            raise ValueError("state_pool is required when reuse_root is provided")

        # Reuse existing subtree — pool was already compacted and root
        # stats reset by prepare_reuse_root(), don't reset the pool.
        # Full sim budget: virtual backups for existing children are
        # near-free, real search runs once root catches up per action.
        if reuse_root.value_sum.shape[0] != num_players:
            raise ValueError(
                f"reuse_root value width {reuse_root.value_sum.shape[0]} "
                f"does not match search num_players {num_players}"
            )
        root = reuse_root
        num_sims = config.num_simulations
    else:
        if root_state is None:
            raise ValueError("root_state is required for fresh searches")
        actual_num_players = TURN.get_num_players(root_state)
        if actual_num_players != num_players:
            raise ValueError(
                f"root_state num_players {actual_num_players} does not match "
                f"MCTS config num_players {num_players}"
            )

        # Set up state pool
        if state_pool is None:
            max_players = root_state.max_players
            total_size = get_layout(max_players).total_size
            state_pool = StatePool(2 * (config.num_simulations + 1), total_size)
        else:
            assert root_state._array.shape[0] == state_pool._state_size, (
                f"state_pool row width {state_pool._state_size} does not match "
                f"root state width {root_state._array.shape[0]}"
            )
        # Fresh search — reset pool and build root from scratch
        state_pool.reset()

        # Check if root state is terminal. GameState doesn't expose get_phase
        # directly — the canonical accessor lives on the TURN handle.
        is_terminal = TURN.get_phase(root_state) == GamePhases.PHASE_GAME_OVER

        root = MCTSNode(
            active_player_id=TURN.get_active_player(root_state),
            num_players=num_players,
            is_terminal=is_terminal,
        )
        root.state_idx = state_pool.alloc(root_state._array)

        if is_terminal:
            values = evaluator.evaluate_terminal(root_state)
            if check_nonfinite:
                _raise_if_nonfinite(
                    values,
                    name="root_terminal_values",
                    debug_context=debug_context,
                )
            root.terminal_values = values
            root.visit_count = 1
            root.value_sum += values
            return root

        # Evaluate root with NN. Sparse-priors contract:
        #   (sparse_priors, values, action_ids, n_legal, phase_id)
        priors, root_values, action_ids, n_legal, phase_id = evaluator.evaluate(
            root_state,
        )
        if check_nonfinite:
            _raise_if_nonfinite(
                priors,
                name="root_priors.raw",
                debug_context=debug_context,
                extra=f"n_legal={n_legal}",
            )
            _raise_if_nonfinite(
                root_values,
                name="root_values",
                debug_context=debug_context,
                extra=f"n_legal={n_legal}",
            )
        priors, action_ids, n_legal = _filter_acq_price_root_priors(
            priors,
            action_ids,
            n_legal,
            phase_id,
            config.max_acq_price_actions,
        )
        if check_nonfinite:
            _raise_if_nonfinite(
                priors,
                name="root_priors.filtered",
                debug_context=debug_context,
                extra=f"n_legal={n_legal}",
            )

        # Expand root onto the sparse legal list (no dense mask).
        root.expand(
            action_ids, n_legal, priors,
            num_players=num_players, default_value=root_values,
        )

        # Backup root evaluation
        root.visit_count = 1
        root.value_sum += root_values

        num_sims = config.num_simulations

    if check_nonfinite:
        _raise_if_nonfinite(
            root.value_sum,
            name="search_root.value_sum.start",
            debug_context=debug_context,
        )
        if root.priors is not None:
            _raise_if_nonfinite(
                root.priors,
                name="search_root.priors.start",
                debug_context=debug_context,
            )
        if root.value_sums is not None:
            _raise_if_nonfinite(
                root.value_sums,
                name="search_root.value_sums.start",
                debug_context=debug_context,
            )

    # Ensure per-batch scratch buffers are sized for this search
    state_pool.ensure_pending_bufs(batch_size, num_players)
    pending_action_ids_buf = state_pool._pending_action_ids_buf
    pending_legal_mask_buf = state_pool._pending_legal_mask_buf
    pending_n_buf = state_pool._pending_n_buf
    pending_phase_buf = state_pool._pending_phase_buf
    saved_values_buf = state_pool._saved_values_buf
    path_pool = state_pool._path_pool
    assert pending_action_ids_buf is not None and pending_n_buf is not None
    assert pending_legal_mask_buf is not None and pending_phase_buf is not None
    assert saved_values_buf is not None and path_pool is not None
    saved_values_buf = saved_values_buf[:, :num_players]
    legal_scratch = state_pool._legal_scratch
    action_lut_np = state_pool._action_lut_np

    # Scratch GameState rebound to each pool row via rebind()
    max_players = state_pool._max_players
    scratch_gs = GameState.from_buffer(
        state_pool.row(0), num_players, max_players=max_players,
    )

    # Add Dirichlet noise at root (fresh noise each search)
    if rng is None:
        rng = np.random.default_rng()
    if config.dirichlet_epsilon > 0:
        assert root.priors is not None
        if config.dirichlet_dynamic:
            alpha = config.dirichlet_alpha_numerator / len(root.priors)
        else:
            alpha = config.dirichlet_alpha
        _add_dirichlet_noise(root, alpha, config.dirichlet_epsilon, rng)

    # Leaf lock sentinel: -inf Q guarantees PUCT never re-selects a locked edge
    neg_inf_row = np.full(num_players, -np.inf, dtype=np.float32)

    if profile is not None:
        profile.num_searches += 1
    _t0 = _t1 = _t2 = 0.0  # profile timing scratch vars

    # Run simulations in batches
    sim = 0
    while sim < num_sims:
        if profile is not None:
            _t0 = perf_counter()

        # Collect a batch of leaves for NN evaluation.
        pending: list[tuple[_Path, MCTSNode]] = []
        pending_ids: set[int] = set()  # safety net for single-action parents

        while len(pending) < batch_size and sim < num_sims:
            # Selection: traverse tree to find a leaf via one Cython call.
            # descend_path appends (node, action_idx, array_idx) per level
            # and returns an outcome describing why the descent stopped.
            # Reuse the preallocated path list at the next pending slot;
            # if the descent doesn't commit to pending, the same slot is
            # reused (cleared) on the next iteration.
            path: _Path = path_pool[len(pending)]
            path.clear()
            outcome, descend_node, descend_aidx, descend_arr = _descend_path(
                root, config.c_puct, path,
            )

            if outcome == DESCEND_VIRTUAL_BACKUP:
                # Reuse-root case: echo child Q to root, no further descent.
                child = descend_node.children[descend_aidx]
                if check_nonfinite:
                    _raise_if_nonfinite(
                        child.value_sum,
                        name="virtual_backup.child_value_sum",
                        debug_context=debug_context,
                        extra=(
                            f"sim={sim}; child_visit_count={child.visit_count}; "
                            f"{_action_row_summary(descend_node, descend_arr, label='parent')}"
                        ),
                    )
                _virtual_backup(descend_node, child, descend_arr)
                if check_nonfinite:
                    assert descend_node.value_sums is not None
                    _raise_if_nonfinite(
                        descend_node.value_sums[descend_arr],
                        name="virtual_backup.parent_edge_after",
                        debug_context=debug_context,
                        extra=(
                            f"sim={sim}; child_visit_count={child.visit_count}; "
                            f"{_action_row_summary(descend_node, descend_arr, label='parent')}"
                        ),
                    )
                    _raise_if_nonfinite(
                        descend_node.value_sum,
                        name="virtual_backup.parent_value_sum_after",
                        debug_context=debug_context,
                        extra=(
                            f"sim={sim}; child_visit_count={child.visit_count}; "
                            f"{_action_row_summary(descend_node, descend_arr, label='parent')}"
                        ),
                    )
                sim += 1
                if profile is not None:
                    profile.virtual_backups += 1
                continue

            if outcome == DESCEND_NEED_NEW_CHILD:
                # First visit: create child node with state from pool.
                # Rebind scratch GameState to avoid allocating a wrapper.
                parent = descend_node
                action_idx = descend_aidx
                child = MCTSNode(num_players=num_players)
                child.state_idx = state_pool.alloc_from_row(parent.state_idx)
                scratch_gs.rebind(
                    state_pool.row(child.state_idx),
                    num_players,
                    max_players=max_players,
                )
                status = DRIVER.apply_action(scratch_gs, action_idx)
                assert status != STATUS_INVALID_PY, (
                    f"MCTS expansion got STATUS_INVALID for action {action_idx}"
                )
                child.active_player_id = TURN.get_active_player(scratch_gs)
                if status == STATUS_GAME_OVER_PY:
                    child.is_terminal = True
                    child.terminal_values = evaluator.evaluate_terminal(
                        scratch_gs,
                    )
                else:
                    # Pack legal actions + dense legal mask directly into
                    # the eval batch buffers at the slot this leaf will
                    # occupy when queued. Newly created non-terminal
                    # children always queue next (no virtual-backup /
                    # terminal / dedup bailout applies), so len(pending)
                    # is the slot.
                    phase_id = get_decision_phase_py(scratch_gs)
                    child.pending_phase = phase_id
                    n = enumerate_policy_actions_py(
                        scratch_gs,
                        legal_scratch,
                        config.max_acq_price_actions,
                    )
                    child.pending_n = n
                    slot = len(pending)
                    pending_action_ids_buf[slot, :n] = legal_scratch[:n]
                    pending_n_buf[slot] = n
                    pending_phase_buf[slot] = phase_id
                    # Dense mask for evaluate_leaves: zero the row, then
                    # flip legal unified-logit slots via the LUT.
                    pending_legal_mask_buf[slot] = 0
                    pending_legal_mask_buf[
                        slot, action_lut_np[phase_id, legal_scratch[:n]]
                    ] = 1
                parent.children[action_idx] = child
                node = child
            else:
                assert outcome == DESCEND_EXISTING_LEAF
                # Landed on a terminal or unexpanded (queued-for-eval) node.
                node = descend_node

            # Terminal node: increment visits along path and backup values
            if node.is_terminal:
                assert node.terminal_values is not None
                if check_nonfinite:
                    _raise_if_nonfinite(
                        node.terminal_values,
                        name="terminal_values",
                        debug_context=debug_context,
                        extra=f"sim={sim}; {_path_summary(path)}",
                    )
                sim += 1
                _increment_visits(path, node)
                _backup(path, node, node.terminal_values)
                continue

            # Non-terminal leaf: queue for batch evaluation.
            # If selection lands on an already-pending leaf, stop filling
            # this batch. `continue` would loop forever: neither sim nor
            # visit counts were updated, so PUCT would make identical
            # choices. Instead, evaluate the partial batch (unlocking
            # leaves), then the next iteration has fresh frontier.
            nid = id(node)
            if nid in pending_ids:
                break

            sim += 1
            _increment_visits(path, node)

            # Lock parent edge and queue for batch eval. Row-memcpy into
            # the preallocated saved_values_buf — the slot index is the
            # current pending length (append happens below).
            parent, _, parent_aidx = path[-1]
            assert parent.value_sums is not None
            if check_nonfinite:
                _raise_if_nonfinite(
                    parent.value_sums[parent_aidx],
                    name="leaf_lock.parent_edge_before_lock",
                    debug_context=debug_context,
                    extra=f"sim={sim}; {_path_summary(path)}",
                )
            saved_values_buf[len(pending)] = parent.value_sums[parent_aidx]
            parent.value_sums[parent_aidx] = neg_inf_row

            # Propagate lock up when all sibling edges at a node are locked
            _propagate_lock(path)

            pending.append((path, node))
            pending_ids.add(nid)

            # Early termination: all root edges locked, no more useful search
            assert root.value_sums is not None
            if _all_col0_neg_inf(root.value_sums):
                break

        # Batch evaluate all pending leaves using raw arrays (no GameState needed)
        if not pending:
            if profile is not None:
                profile.selection_secs += perf_counter() - _t0
            continue

        if profile is not None:
            _t1 = perf_counter()
            profile.selection_secs += _t1 - _t0

        # pending_action_ids_buf / pending_legal_mask_buf / pending_n_buf /
        # pending_phase_buf are already packed per-leaf at child-creation
        # time; no extra pass needed here.
        n_pending = len(pending)

        leaf_arrays = [state_pool.row(node.state_idx) for _, node in pending]
        priors_dense, values_batch = evaluator.evaluate_leaves(
            leaf_arrays,
            pending_legal_mask_buf[:n_pending],
        )
        if check_nonfinite:
            _raise_pending_batch_if_nonfinite(
                priors_dense,
                name="evaluate_leaves.priors_dense",
                pending=pending,
                debug_context=debug_context,
                sim=sim,
            )
            _raise_pending_batch_if_nonfinite(
                values_batch,
                name="evaluate_leaves.values",
                pending=pending,
                debug_context=debug_context,
                sim=sim,
            )

        if profile is not None:
            _t2 = perf_counter()
            profile.eval_secs += _t2 - _t1

        for i, (path, node) in enumerate(pending):
            # Unlock parent edge: restore saved Q values
            parent, _, parent_aidx = path[-1]
            assert parent.value_sums is not None
            parent.value_sums[parent_aidx] = saved_values_buf[i]

            # Recursively unlock propagation-locked ancestor edges
            _propagate_unlock(path)

            # Gather per-leaf sparse prior slice out of the dense
            # server response using the same LUT the mask was built
            # from. Dense priors are softmaxed on the server / inside
            # NNEvaluator, so illegal slots carry ~0 mass and the
            # gathered slice is already a valid distribution.
            n = node.pending_n
            phase_id = int(pending_phase_buf[i])
            action_ids = pending_action_ids_buf[i, :n]
            slots = action_lut_np[phase_id, action_ids]
            priors_legal = priors_dense[i, slots].copy()
            values = values_batch[i]
            if check_nonfinite:
                _raise_if_nonfinite(
                    priors_legal,
                    name="evaluate_leaves.priors_legal",
                    debug_context=debug_context,
                    extra=(
                        f"sim={sim}; batch_row={i}; phase_id={phase_id}; "
                        f"n_legal={n}; action_ids={action_ids.tolist()}; "
                        f"{_path_summary(path)}"
                    ),
                )
                _raise_if_nonfinite(
                    values,
                    name="evaluate_leaves.values_row",
                    debug_context=debug_context,
                    extra=(
                        f"sim={sim}; batch_row={i}; phase_id={phase_id}; "
                        f"n_legal={n}; {_path_summary(path)}"
                    ),
                )

            node.expand(
                action_ids, n, priors_legal,
                num_players=num_players, default_value=values,
            )
            # Clear per-leaf pending state — consumed by expansion.
            node.pending_n = 0
            node.pending_phase = -1

            # Backup values (visit counts already incremented at selection time)
            _backup(path, node, values)

        if profile is not None:
            profile.backup_secs += perf_counter() - _t2
            profile.num_eval_batches += 1
            profile.total_leaves += len(pending)

    if check_nonfinite:
        _raise_if_nonfinite(
            root.value_sum,
            name="search_root.value_sum.end",
            debug_context=debug_context,
        )
        if root.value_sums is not None:
            _raise_if_nonfinite(
                root.value_sums,
                name="search_root.value_sums.end",
                debug_context=debug_context,
            )

    return root


def get_action_probabilities(
    root: MCTSNode,
    temperature: float,
) -> np.ndarray:
    """Convert root visit counts to action probabilities.

    Returns a dense ``(MAX_ACTION_SIZE,)`` distribution with zeros outside
    the legal list — callers in self_play sample from it directly. A later
    pass (rss-az-phli.1) will switch to sparse ``(action_ids, probs)``
    once self_play moves to sparse policy targets.

    Args:
        root: Root node after search.
        temperature: Controls exploration.
            temperature=1.0: proportional to visit counts.
            temperature->0: deterministic (argmax).

    Returns:
        Probability distribution over actions, shape ``(MAX_ACTION_SIZE,)``.
    """
    probs = np.zeros(MAX_ACTION_SIZE, dtype=np.float32)

    if root.legal_actions is None:
        return probs
    assert root.visit_counts is not None

    scaled = scale_visit_counts_by_temperature(root.visit_counts, temperature)
    probs[root.legal_actions] = scaled

    return probs


def scale_visit_counts_by_temperature(
    counts: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return a sparse probability row from MCTS visits.

    Directly computing ``counts ** (1 / temperature)`` overflows quickly for
    low temperatures: with float32 visits and ``temperature=0.01``, even
    ``3 ** 100`` is already above the finite range. Log-space softmax keeps the
    exact same distribution without producing ``inf / inf``.
    """
    counts64 = np.asarray(counts, dtype=np.float64)
    probs = np.zeros(counts64.shape, dtype=np.float32)
    if counts64.size == 0:
        return probs

    if not np.isfinite(temperature):
        raise ValueError(f"temperature must be finite, got {temperature}")

    if temperature < _GREEDY_TEMPERATURE:
        best_idx = int(np.argmax(counts64))
        probs[best_idx] = 1.0
        return probs

    positive = counts64 > 0.0
    if not np.any(positive):
        return probs

    log_weights = np.log(counts64[positive]) / float(temperature)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    total = float(weights.sum())
    if total > 0.0:
        probs[positive] = (weights / total).astype(np.float32)

    return probs


def _get_greedy_leaf_node_and_depth(root: MCTSNode) -> tuple[MCTSNode, int]:
    """Return the A0GB greedy leaf and its decision-edge depth from root."""
    node = root
    depth = 0

    while node.expanded() and not node.is_terminal:
        assert node.visit_counts is not None and node.legal_actions is not None
        best_idx = int(np.argmax(node.visit_counts))
        if node.visit_counts[best_idx] == 0:
            # All children are unvisited — this node is the tree-edge leaf.
            # Its value_sum / visit_count equals V_NN from its single evaluation.
            break
        best_action = int(node.legal_actions[best_idx])
        child = node.children.get(best_action)
        if child is None:
            break
        node = child
        depth += 1

    return node, depth


def get_greedy_leaf_depth(root: MCTSNode) -> int:
    """Return the A0GB greedy leaf depth in MCTS decision edges.

    Depth is counted from the current search root. A root value has depth 0;
    each followed child edge increments the depth by 1. Automated engine
    transitions inside ``DRIVER.apply_action`` are not counted separately.
    """
    _, depth = _get_greedy_leaf_node_and_depth(root)
    return depth


def get_greedy_leaf_value(root: MCTSNode, num_players: int) -> np.ndarray:
    """Compute the A0GB greedy backup value target.

    Starting from the root, follow the child with the highest visit count
    (greedy policy) at each level until reaching a leaf node (no children)
    or a terminal node. Return that node's value.

    At a leaf node with a single visit, this equals V_NN (the neural network's
    evaluation). At a terminal node, this equals the game outcome.

    Reference: Willemsen et al., "Value targets in off-policy AlphaZero:
    a new greedy backup" (ALA 2020 / Neural Computing and Applications, 2022).

    Args:
        root: Root node after search.
        num_players: Number of players in the game.

    Returns:
        Canonical per-player values at the greedy leaf, shape (num_players,).
    """
    node, _ = _get_greedy_leaf_node_and_depth(root)

    # Return the mean value at this node
    if node.visit_count == 0:
        return np.zeros(num_players, dtype=np.float32)

    return node.value_sum / node.visit_count


def _collect_subtree_nodes(root: MCTSNode) -> list[MCTSNode]:
    """Collect all nodes in a subtree that have pool-allocated states.

    Returns nodes sorted by state_idx (ascending) for safe in-place
    compaction — copies never overwrite unread source rows.
    """
    nodes: list[MCTSNode] = []
    stack: list[MCTSNode] = [root]
    while stack:
        node = stack.pop()
        if node.state_idx >= 0:
            nodes.append(node)
        stack.extend(node.children.values())
    nodes.sort(key=lambda n: n.state_idx)
    return nodes


def _reset_root_for_reuse(node: MCTSNode) -> None:
    """Reset root-level visit stats for zero-visit-root reuse.

    Preserves children, priors, legal_actions, and default_value.
    Resets visit_counts, value_sums, visit_count, and value_sum
    so that Dirichlet noise has full effect on PUCT selection.

    During subsequent search, actions whose children have existing
    visit counts trigger "virtual backups" — the child's mean Q is
    backed up without tree traversal until the root's per-action
    visit count catches up to the child's visit count.
    """
    assert node.expanded()
    assert node.legal_actions is not None
    assert node.default_value is not None
    n = len(node.legal_actions)
    num_players = node.value_sum.shape[0]
    node.visit_count = 1
    node.value_sum = node.default_value.copy()
    node.visit_counts = np.zeros(n, dtype=np.int32)
    node.value_sums = np.broadcast_to(
        node.default_value, (n, num_players)
    ).astype(np.float32).copy()


def prepare_reuse_root(
    root: MCTSNode,
    action_idx: int,
    state_pool: StatePool,
) -> MCTSNode | None:
    """Extract the chosen child as a reuse root for the next search.

    Compacts the state pool to retain only the child's subtree states,
    then resets the child's root-level visit statistics so that
    Dirichlet noise has full effect during the next search.

    Args:
        root: The current search root.
        action_idx: The action that was chosen (played in the real game).
        state_pool: The state pool used during search.

    Returns:
        The child node ready for reuse, or None if the action has no
        child in the tree or the child is terminal.
    """
    child = root.children.get(action_idx)
    if child is None or child.is_terminal:
        return None

    # All non-terminal children are expanded after search completes
    assert child.expanded(), "Reuse root must be expanded"

    # Compact pool to retain only this subtree's states
    retained = _collect_subtree_nodes(child)
    state_pool.compact(retained)

    # Reset root-level stats so Dirichlet noise is effective
    _reset_root_for_reuse(child)

    return child
