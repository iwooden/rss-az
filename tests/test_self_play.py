"""End-to-end smoke tests for ``train/self_play.py::play_game``.

Covers the full loop: 3-player ``GameState`` + ``initialize_game(seed=0)``
→ MCTS search via in-process ``NNEvaluator`` (bypasses ``eval_server``) →
sparse policy targets + A0GB value targets → game terminates at
``STATUS_GAME_OVER``. A second test probes subtree-reuse correctness
after move 1: ``_reset_root_for_reuse`` rebuilds ``value_sums`` from
``default_value`` on the reuse root, and the caught-up virtual-backup
path inside ``run_search`` echoes each child's mean Q onto the root's
per-action value_sums.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.actions import MAX_LEGAL_ACTIONS_PY
from core.driver import DRIVER
from core.state import GameState, get_layout
from mcts.evaluator import NNEvaluator
from mcts.search import StatePool, prepare_reuse_root, run_search
from nn.transformer import RSSTransformerNet, TransformerConfig
from train.config import TrainingConfig
from train.self_play import play_game


K_MAX = int(MAX_LEGAL_ACTIONS_PY)
NUM_PLAYERS = 3


@pytest.fixture(scope="module")
def model():
    """Random-init transformer shared across tests in this module."""
    torch.manual_seed(0)
    cfg = TransformerConfig(num_players=NUM_PLAYERS)
    return RSSTransformerNet(cfg).to(torch.device("cpu"))


@pytest.fixture(scope="module")
def evaluator(model):
    """In-process NNEvaluator — bypasses the shared-mem eval server path."""
    return NNEvaluator(model, torch.device("cpu"), num_players=NUM_PLAYERS)


def test_play_game_full_3p_game_produces_valid_record(evaluator):
    """A full 3p self-play game completes and emits well-formed examples.

    play_game's loop only exits on STATUS_GAME_OVER (or raises), so
    returning a record at all proves terminal status was reached. Sparse
    invariants (policy sum, padding, n_legal bound, value range) are
    checked per-move across the whole game.
    """
    config = TrainingConfig(num_players=NUM_PLAYERS, num_simulations=64)
    rng = np.random.default_rng(0)

    record = play_game(evaluator, config, game_seed=0, rng=rng)

    assert record.total_moves > 0
    assert record.num_examples == record.total_moves
    assert len(record.net_worths) == NUM_PLAYERS
    # Game-over invariant: player net worth is non-negative (RULES.md).
    assert all(nw >= 0 for nw in record.net_worths)

    # Sparse policy invariants per move.
    assert record.n_legals.shape == (record.num_examples,)
    assert record.policy_targets.shape == (record.num_examples, K_MAX)
    for i in range(record.num_examples):
        n = int(record.n_legals[i])
        assert 0 < n <= K_MAX, f"move {i}: n_legal={n} out of range"
        row = record.policy_targets[i, :n]
        assert float(row.sum()) == pytest.approx(1.0, abs=1e-5), (
            f"move {i}: policy_target sum={float(row.sum())}"
        )
        # Zero-padded tail past n_legal.
        assert not record.policy_targets[i, n:].any(), (
            f"move {i}: policy_target nonzero past n_legal"
        )

    # Value targets per-player, clamped to the tanh head's [-1, +1] range.
    vt = record.value_targets
    assert vt.shape == (record.num_examples, NUM_PLAYERS)
    assert (vt >= -1.0 - 1e-6).all(), f"min value_target={vt.min()}"
    assert (vt <= 1.0 + 1e-6).all(), f"max value_target={vt.max()}"


def test_play_game_subtree_reuse_produces_sensible_q_after_move_1(evaluator):
    """Virtual backup through subtree reuse echoes child Q onto the root.

    Mirrors play_game's per-move flow out-of-line so we can inspect the
    reuse root between searches: run_search → prepare_reuse_root →
    DRIVER.apply_action → run_search(reuse_root=...). After reset the
    root's value_sums are rebroadcast from default_value; during the
    second search, virtual backups on caught-up children restore each
    per-action edge's Q to match its child's own mean Q.
    """
    config = TrainingConfig(num_players=NUM_PLAYERS, num_simulations=64)
    mcts_cfg = config.to_mcts_config()
    total_size = get_layout(NUM_PLAYERS).total_size
    pool = StatePool(2 * (mcts_cfg.num_simulations + 1), total_size)

    state = GameState(NUM_PLAYERS)
    state.initialize_game(NUM_PLAYERS, seed=0)

    root = run_search(
        state, evaluator, mcts_cfg,
        rng=np.random.default_rng(0), state_pool=pool,
    )
    assert root.visit_counts is not None and root.legal_actions is not None
    best_idx = int(np.argmax(root.visit_counts))
    best_action = int(root.legal_actions[best_idx])

    reuse = prepare_reuse_root(root, best_action, pool)
    assert reuse is not None, "reuse should be available for the top action"

    # _reset_root_for_reuse contract: visits zeroed, value_sums broadcast
    # from default_value (FPU), children preserved.
    assert reuse.visit_count == 1
    assert reuse.visit_counts is not None and (reuse.visit_counts == 0).all()
    assert reuse.value_sums is not None and reuse.default_value is not None
    assert reuse.legal_actions is not None
    for i in range(len(reuse.legal_actions)):
        np.testing.assert_array_equal(
            reuse.value_sums[i], reuse.default_value,
        )

    # Snapshot per-action child Q before the reuse search so we can
    # compare after virtual backups catch up each edge.
    pre_child_qs = {
        int(a): c.value_sum.copy() / c.visit_count
        for a, c in reuse.children.items() if c.visit_count > 0
    }
    assert pre_child_qs, "reused subtree had no visited children to catch up"

    # Apply the chosen action on the real state and run search again with reuse.
    DRIVER.apply_action(state, best_action)
    root2 = run_search(
        state, evaluator, mcts_cfg,
        rng=np.random.default_rng(1), state_pool=pool, reuse_root=reuse,
    )
    assert root2 is reuse
    # Full sim budget absorbed via virtual backups + real search.
    assert root2.visit_count >= mcts_cfg.num_simulations

    # For each action that had real visits before the reuse search, the
    # root's caught-up edge Q should match the child's mean Q.
    assert root2.visit_counts is not None and root2.value_sums is not None
    assert root2.legal_actions is not None
    matched = 0
    for i, action in enumerate(root2.legal_actions):
        a = int(action)
        if a not in pre_child_qs or root2.visit_counts[i] == 0:
            continue
        child = root2.children.get(a)
        assert child is not None
        root_q = root2.value_sums[i] / root2.visit_counts[i]
        child_q = child.value_sum / child.visit_count
        np.testing.assert_allclose(root_q, child_q, atol=0.02)
        matched += 1
    assert matched > 0, "no caught-up edges observed — virtual backup path idle"
