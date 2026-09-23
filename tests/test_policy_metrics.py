"""Diagnostics must describe search without changing it or its denominators."""

import pickle

import numpy as np
import pytest

from train.policy_metrics import PolicyMetrics


def observe(metrics, prior, visits, target=None, phase=0, move=0):
    visits = np.asarray(visits, dtype=np.int32)
    if target is None:
        target = visits / visits.sum()
    metrics.observe(np.asarray(prior), visits, np.asarray(target), phase, move, (2, 4))


def test_confidence_search_changes_and_kl_partition():
    stats = PolicyMetrics()
    # Overconfident agreement, underconfident agreement, then overturned prior.
    observe(stats, [.98, .02], [8, 2])
    observe(stats, [.6, .4], [8, 2])
    observe(stats, [.99, .01], [2, 8])
    s = stats.scalars()
    prefix = "policy/all/"
    assert s[prefix + "decision_count"] == 3
    assert s[prefix + "prior_top1_mean"] == pytest.approx((.98 + .6 + .99) / 3)
    assert s[prefix + "prior_above_95_fraction"] == pytest.approx(2 / 3)
    assert s[prefix + "prior_above_98_fraction"] == pytest.approx(1 / 3)
    assert s[prefix + "search_top1_changed_fraction"] == pytest.approx(1 / 3)
    assert s[prefix + "search_changed_choice_prior_mean"] == pytest.approx(.01)
    assert s[prefix + "search_changed_from_above_95_fraction"] == .5
    assert s[prefix + "search_changed_from_above_98_fraction"] == 1
    contributions = [s[prefix + "target_kl_" + name + "_contribution"] for name in (
        "different_top", "same_top_prior_sharper", "same_top_prior_not_sharper",
    )]
    assert all(value > 0 for value in contributions)
    assert sum(contributions) == pytest.approx(s[prefix + "target_kl_to_prior_mean"])
    assert s[prefix + "same_top_prior_sharper_fraction"] == pytest.approx(1 / 3)


def test_forced_moves_ties_and_empty_conditional_populations():
    stats = PolicyMetrics()
    observe(stats, [1.], [10], phase=1)
    # Neither a uniform prior nor a tied visit winner is an overturn.
    observe(stats, [.5, .5], [1, 9])
    observe(stats, [.1, .9], [5, 5])
    s = stats.scalars()
    assert s["policy/all/decision_count"] == 2
    assert s["policy/all/forced_count"] == 1
    assert s["policy/all/prior_top1_mean"] == pytest.approx(.7)
    assert s["policy/all/search_top1_changed_fraction"] == 0
    assert "policy/all/search_changed_choice_prior_mean" not in s
    assert "policy/all/search_changed_from_above_95_fraction" not in s
    assert "policy/phase/bid/prior_top1_mean" not in s


def test_temperature_effect_zero_probabilities_and_stage_boundaries():
    stats = PolicyMetrics()
    observe(stats, [.5, .5, 0], [8, 2, 0], [.94, .06, 0], move=2)
    observe(stats, [1., 0], [1, 1], move=3)
    observe(stats, [.7, .3], [7, 3], move=4)
    s = stats.scalars()
    for stage in ("pre_anneal", "anneal", "post_anneal"):
        assert s[f"policy/stage/{stage}/decision_count"] == 1
    assert all(np.isfinite(value) for value in s.values())
    assert s["policy/stage/pre_anneal/target_kl_to_prior_mean"] > s[
        "policy/stage/pre_anneal/search_kl_to_prior_mean"
    ]
    assert s["policy/stage/pre_anneal/search_unvisited_fraction_mean"] == pytest.approx(1 / 3)


def test_epoch_aggregation_weights_decisions_and_survives_worker_serialization(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tests.test_self_play_metrics import _fake_record
    from train.logging import TrainingLogger
    from train.main import _SelfPlayMetricAccumulator, _build_epoch_self_play_scalars

    first = _fake_record(3, [100, 200, 300])
    observe(first.policy_metrics, [.6, .4], [6, 4], phase=0)
    second = _fake_record(5, [100, 200, 300, 400, 500])
    for _ in range(3):
        observe(second.policy_metrics, [.99, .01], [1, 9], phase=1, move=5)
    accumulator = _SelfPlayMetricAccumulator()
    accumulator.add_record(pickle.loads(pickle.dumps(first)))
    accumulator.add_record(pickle.loads(pickle.dumps(second)))
    scalars = _build_epoch_self_play_scalars(accumulator)
    tag = "self_play_aggregate/policy/all/prior_top1_mean"
    assert scalars[tag] == pytest.approx((.6 + 3 * .99) / 4)
    assert scalars["self_play_3p/policy/all/decision_count"] == 1
    assert scalars["self_play_5p/policy/phase/bid/decision_count"] == 3
    assert scalars["self_play_aggregate/policy/stage/post_anneal/decision_count"] == 3
    assert first.policy_metrics.scalars()["policy/all/decision_count"] == 1
    logger = TrainingLogger(str(tmp_path))
    try:
        logger.log_scalars(1, scalars)
    finally:
        logger.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    assert events.Scalars(tag)[0].value == pytest.approx(scalars[tag])


@pytest.mark.parametrize("model_path", ["nn/transformer-v2.py", "nn/transformer-v3.py"])
def test_prior_capture_preserves_fresh_and_reused_search_without_extra_evaluations(model_path):
    import torch
    from core.state import GameState, get_layout
    from mcts.evaluator import NNEvaluator
    from mcts.search import StatePool, prepare_reuse_root, run_search
    from nn import create_model
    from train.config import TrainingConfig

    config = TrainingConfig(
        num_players=3, model_path=model_path,
        d_model=32, d_proj=8, num_heads=4, num_layers=1,
        num_simulations=16, dirichlet_epsilon=.25,
    )
    torch.manual_seed(2)
    model = create_model(config).eval()

    class CountingEvaluator(NNEvaluator):
        calls = 0

        def evaluate(self, state):
            self.calls += 1
            return super().evaluate(state)

        def evaluate_leaves(self, *args, **kwargs):
            self.calls += 1
            return super().evaluate_leaves(*args, **kwargs)

    state = GameState(3)
    state.initialize_game(3, seed=17)
    evaluators = [CountingEvaluator(model, torch.device("cpu"), 3) for _ in range(2)]
    pools = [StatePool(100, get_layout(3).total_size) for _ in range(2)]
    rngs = [np.random.default_rng(8) for _ in range(2)]
    captured = []
    roots = [None, None]
    expected_priors = evaluators[0].evaluate(state)[0].copy()
    evaluators[0].calls = 0
    for step in range(2):
        for i in range(2):
            roots[i] = run_search(
                state if step == 0 else None, evaluators[i], config.to_mcts_config(),
                rng=rngs[i], state_pool=pools[i], reuse_root=roots[i],
                root_priors_out=captured if i else None,
            )
        assert len(captured) == 1
        np.testing.assert_array_equal(captured[0], expected_priors)
        baseline, observed = roots
        assert baseline is not None and observed is not None
        assert observed.priors is not None
        assert not np.array_equal(captured[0], observed.priors)
        assert evaluators[0].calls == evaluators[1].calls
        assert rngs[0].bit_generator.state == rngs[1].bit_generator.state
        for attr in ("priors", "legal_actions", "visit_counts", "value_sums", "value_sum"):
            np.testing.assert_array_equal(getattr(baseline, attr), getattr(observed, attr))
        if step == 0:
            action = next(a for a, child in baseline.children.items() if not child.is_terminal)
            child_priors = baseline.children[action].priors
            assert child_priors is not None
            expected_priors = child_priors.copy()
            roots = [
                prepare_reuse_root(baseline, action, pools[0]),
                prepare_reuse_root(observed, action, pools[1]),
            ]
            assert all(root is not None for root in roots)
