"""MCTS search benchmark.

Times run_search() with a real NN model (random weights) to measure
end-to-end performance including neural network inference.
"""

from __future__ import annotations

import statistics
import time

import numpy as np
import torch

from core.actions import get_total_action_count
from core.state import GameState, get_layout
from train.config import MCTSConfig
from mcts.evaluator import NNEvaluator
from mcts.search import run_search
from nn.model_3p import RSSAlphaZeroNet, RSSModelConfig


def run_mcts_benchmark(
    num_simulations: int = 800,
    num_runs: int = 10,
    num_players: int = 3,
    device: str = "cpu",
    search_batch_size: int = 1,
) -> None:
    """Run MCTS benchmark and print timing results.

    Creates a fresh game state and NN model with random weights,
    then times num_runs independent MCTS searches.
    """
    torch_device = torch.device(device)

    # Build model with random weights
    _layout = get_layout(num_players)
    config = RSSModelConfig(input_dim=_layout.visible_size, action_dim=get_total_action_count(num_players), value_dim=num_players)
    model = RSSAlphaZeroNet(config)
    model.to(torch_device)

    evaluator = NNEvaluator(model, torch_device, num_players=num_players)
    mcts_config = MCTSConfig(
        num_simulations=num_simulations, num_players=num_players,
        search_batch_size=search_batch_size,
    )

    # Create base game state
    base_state = GameState(num_players=num_players)
    base_state.initialize_game(seed=42)

    header = (
        f"MCTS Benchmark ({num_simulations} simulations, "
        f"{num_runs} runs, {num_players} players, device={device}, "
        f"batch={search_batch_size})"
    )
    print(header)
    print("=" * len(header))

    times: list[float] = []
    for i in range(num_runs):
        rng = np.random.default_rng(seed=i)

        t0 = time.perf_counter()
        run_search(base_state, evaluator, mcts_config, rng=rng)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        sims_per_sec = num_simulations / elapsed
        print(f"Run {i + 1:>2}: {elapsed:.2f}s ({sims_per_sec:.1f} sims/sec)")

    print("-" * len(header))
    mean = statistics.mean(times)
    median = statistics.median(times)
    print(f"Min:    {min(times):>6.2f}s  |  Max: {max(times):.2f}s")
    print(f"Mean:   {mean:>6.2f}s  |  Median: {median:.2f}s")
    print(f"Avg sims/sec: {num_simulations / mean:.1f}")
