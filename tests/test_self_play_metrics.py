"""Self-play metric aggregation and display tests."""

from __future__ import annotations

from types import SimpleNamespace

from rich.console import Console

from train.logging import TrainingLogger
from train.main import (
    _SelfPlayMetricAccumulator,
    _build_epoch_self_play_scalars,
)


def _fake_record(
    num_players: int,
    net_worths: list[int],
    *,
    examples: int = 10,
    moves: int = 7,
    duration: float = 2.0,
    target_entropy: float = 1.5,
    target_top1: float = 0.6,
    sample_entropy: float = 1.0,
    sample_top1: float = 0.4,
    avg_active_corp_price: float = 20.0,
    corps_in_receivership: int = 1,
    has_max_price_corp: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        num_players=num_players,
        num_examples=examples,
        total_moves=moves,
        duration_secs=duration,
        net_worths=net_worths,
        shares_per_player=[player + 1 for player in range(num_players)],
        companies_per_player=[player + 2 for player in range(num_players)],
        pres_share_values=[float((player + 1) * 10) for player in range(num_players)],
        nw_cash_pct=[0.2 for _ in range(num_players)],
        nw_companies_pct=[0.3 for _ in range(num_players)],
        nw_shares_pct=[0.5 for _ in range(num_players)],
        avg_active_corp_price=avg_active_corp_price,
        corps_in_receivership=corps_in_receivership,
        has_max_price_corp=has_max_price_corp,
        policy_target_entropy_mean=target_entropy,
        policy_target_top1_fraction=target_top1,
        sample_policy_entropy_mean=sample_entropy,
        sample_top1_action_fraction=sample_top1,
    )


def test_self_play_metric_accumulator_handles_missing_player_count() -> None:
    metrics = _SelfPlayMetricAccumulator()
    metrics.add_record(_fake_record(3, [100, 300, 200], examples=12, moves=8))
    metrics.add_record(_fake_record(5, [50, 500, 300, 200, 100], examples=20, moves=11))

    aggregate = metrics.aggregate_snapshot()
    by_count = metrics.count_snapshots()

    assert set(by_count) == {3, 5}
    assert aggregate["games"] == 2.0
    assert aggregate["examples"] == 32.0
    assert aggregate["avg_moves"] == 9.5
    assert by_count[3]["rank_net_worths"] == [300.0, 200.0, 100.0]
    assert by_count[5]["rank_net_worths"] == [500.0, 300.0, 200.0, 100.0, 50.0]
    assert by_count[3]["games"] == 1.0
    assert by_count[3]["total_net_worth"] == 600.0
    assert by_count[5]["games"] == 1.0
    assert by_count[5]["total_net_worth"] == 1150.0
    assert len(aggregate["rank_net_worths"]) == 5


def test_self_play_tensorboard_scalar_prefixes_for_mixed_counts() -> None:
    metrics = _SelfPlayMetricAccumulator()
    metrics.add_record(_fake_record(3, [100, 300, 200]))
    metrics.add_record(_fake_record(5, [50, 500, 300, 200, 100]))

    scalars = _build_epoch_self_play_scalars(metrics)

    assert "self_play_aggregate/game_length_mean" in scalars
    assert "self_play_aggregate/corps_in_receivership" in scalars
    assert "self_play_3p/policy_target_entropy_mean" in scalars
    assert "self_play_5p/net_worth_5th" in scalars
    assert not any(key.startswith("self_play/") for key in scalars)
    assert not any(key.startswith("self_play_4p/") for key in scalars)


def test_self_play_panel_formats_net_worth_by_player_count(tmp_path) -> None:
    logger = TrainingLogger(str(tmp_path))
    try:
        logger.update_self_play(
            games_done=2,
            total_examples=32,
            avg_moves=9.5,
            target_entropy=1.5,
            target_top1_frac=0.6,
            sample_entropy=1.0,
            sample_top1_frac=0.4,
            count_rank_net_worths={
                3: [300.0, 200.0, 100.0],
                5: [500.0, 300.0, 200.0, 100.0, 50.0],
            },
            count_rank_mins={
                3: [300.0, 200.0, 100.0],
                5: [500.0, 300.0, 200.0, 100.0, 50.0],
            },
            count_rank_maxs={
                3: [300.0, 200.0, 100.0],
                5: [500.0, 300.0, 200.0, 100.0, 50.0],
            },
            count_games={3: 1, 5: 1},
            count_avg_moves={3: 8.0, 5: 11.0},
            count_total_net_worths={3: 600.0, 5: 1150.0},
        )

        panel = logger._build_self_play_panel()
        text = panel.renderable.plain
    finally:
        logger.close()

    assert "1st=$300" in text
    assert "Examples: 32" in text
    assert "Avg moves/game: 9.5    3p=8.0  5p=11.0" in text
    assert "3p: num games=1, total net worth=$600" in text
    assert "1st=$500" in text
    assert "5p: num games=1, total net worth=$1,150" in text
    assert "5th=$50" in text
    assert (
        "Policy: target H=1.500 nats, top-1=60.0% | "
        "sample H=1.000 nats, top-1=40.0%"
    ) in text
    assert "Target policy:" not in text
    assert "Sample policy:" not in text


def test_epoch_summary_formats_net_worth_by_player_count(tmp_path) -> None:
    logger = TrainingLogger(str(tmp_path))
    logger.console = Console(record=True, width=120)
    try:
        logger.log_epoch_summary(
            epoch=1,
            num_epochs=3,
            self_play_stats={
                "games": 2.0,
                "examples": 32.0,
                "avg_moves": 9.5,
                "avg_duration": 2.0,
                "policy_target_entropy": 1.5,
                "policy_target_top1_frac": 0.6,
                "sample_policy_entropy": 1.0,
                "sample_top1_frac": 0.4,
                "by_player_count": {
                    3: {
                        "games": 1.0,
                        "total_net_worth": 600.0,
                        "rank_net_worths": [300.0, 200.0, 100.0],
                        "rank_net_worths_min": [300.0, 200.0, 100.0],
                        "rank_net_worths_max": [300.0, 200.0, 100.0],
                    },
                    5: {
                        "games": 1.0,
                        "total_net_worth": 1150.0,
                        "rank_net_worths": [500.0, 300.0, 200.0, 100.0, 50.0],
                        "rank_net_worths_min": [500.0, 300.0, 200.0, 100.0, 50.0],
                        "rank_net_worths_max": [500.0, 300.0, 200.0, 100.0, 50.0],
                    },
                },
            },
            train_stats={},
            buffer_size=32,
            buffer_capacity=100,
            epoch_duration=5.0,
        )
        text = logger.console.export_text()
    finally:
        logger.close()

    assert "Net worth by count:" in text
    assert "1st=$300" in text
    assert "3p: num games=1, total net worth=$600" in text
    assert "1st=$500" in text
    assert "5p: num games=1, total net worth=$1,150" in text
    assert "5th=$50" in text
    assert (
        "Policy: target H=1.500 nats, top-1=60.0% | "
        "sample H=1.000 nats, top-1=40.0%"
    ) in text
    assert "Target policy:" not in text
    assert "Sample policy:" not in text


def test_training_panel_formats_loss_by_player_count(tmp_path) -> None:
    logger = TrainingLogger(str(tmp_path))
    try:
        logger.update_training(
            step=1,
            losses={
                "policy_loss": 1.0,
                "value_loss": 0.5,
                "total_loss": 1.5,
                "policy_target_entropy": 0.7,
                "policy_kl": 0.3,
                "policy_loss_3p": 0.9,
                "value_loss_3p": 0.4,
                "policy_loss_5p": 1.1,
                "value_loss_5p": 0.6,
            },
            lr=1e-3,
        )

        panel = logger._build_training_panel()
        text = panel.renderable.plain
    finally:
        logger.close()

    assert "By players:" in text
    assert "3p: policy=0.900 value=0.400" in text
    assert "5p: policy=1.100 value=0.600" in text


def test_epoch_summary_formats_training_loss_by_player_count(tmp_path) -> None:
    logger = TrainingLogger(str(tmp_path))
    logger.console = Console(record=True, width=120)
    try:
        logger.log_epoch_summary(
            epoch=1,
            num_epochs=3,
            self_play_stats={
                "games": 1.0,
                "examples": 10.0,
                "avg_moves": 7.0,
                "avg_duration": 2.0,
            },
            train_stats={
                "steps": 2.0,
                "policy_loss": 1.0,
                "value_loss": 0.5,
                "total_loss": 1.5,
                "policy_target_entropy": 0.7,
                "policy_kl": 0.3,
                "lr": 1e-3,
                "policy_loss_3p": 0.9,
                "value_loss_3p": 0.4,
                "policy_loss_5p": 1.1,
                "value_loss_5p": 0.6,
            },
            buffer_size=32,
            buffer_capacity=100,
            epoch_duration=5.0,
        )
        text = logger.console.export_text()
    finally:
        logger.close()

    assert "Training by count:" in text
    assert "3p: policy=0.900 value=0.400" in text
    assert "5p: policy=1.100 value=0.600" in text
