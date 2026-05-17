from pathlib import Path

import torch

from nn import create_model
from train.config import TrainingConfig
from utils_18xx.analyze_replay import analyze_replay


def test_analyze_replay_outputs_grouped_markdown_for_bid_fixture() -> None:
    config = TrainingConfig(num_players=3)
    model = create_model(config).to(torch.device("cpu"))
    model.eval()

    markdown = analyze_replay(
        Path("tests/games_18xx/data/224967.json"),
        model,
        torch.device("cpu"),
        config,
        checkpoint_path="new",
        top_n=3,
        output_format="markdown",
    )

    assert "# 18xx Replay Analysis: game 224967" in markdown
    assert "_Grouped compound 18xx action" in markdown
    assert "Recorded engine action: AUCTION slot 2 (MHE, face $8)" in markdown
    assert "Recorded engine action: BID $10" in markdown
    assert "NN Values:" in markdown
    assert "## Final State" in markdown


def test_analyze_replay_outputs_html_summary_tables() -> None:
    config = TrainingConfig(num_players=3)
    model = create_model(config).to(torch.device("cpu"))
    model.eval()

    html = analyze_replay(
        Path("tests/games_18xx/data/224967.json"),
        model,
        torch.device("cpu"),
        config,
        checkpoint_path="new",
        top_n=3,
        output_format="html",
    )

    assert "<canvas id=\"valueChart\"" in html
    assert "Top 10 Value Swings" in html
    assert "Top 10 Policy Entropy Moves" in html
    assert "BID $10 on MHE" in html
    assert "chartData" in html
