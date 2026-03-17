"""Data collection and model loading utilities for interpretability experiments.

Common workflow:
    from interp.utils import load_model, collect_states, InterpDataset

    model, config, device, epoch = load_model()
    dataset = collect_states(model, config, device, num_games=50)
    dataset.save("interp/data/states.npz")

    # Later:
    dataset = InterpDataset.load("interp/data/states.npz")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.data import GamePhases
from core.driver import DRIVER, STATUS_GAME_OVER_PY
from core.state import GameState
from mcts.evaluator import NNEvaluator, rotate_visible_state
from nn import create_model
from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import TrainingConfig


@dataclass
class InterpDataset:
    """Collected game states for interpretability analysis.

    States are pre-rotated so the active player is at slot 0, exactly
    as the NN sees them during training/inference.
    """

    states: np.ndarray  # (N, visible_size) float32
    legal_masks: np.ndarray  # (N, action_dim) float32
    phases: np.ndarray  # (N,) int32
    active_players: np.ndarray  # (N,) int32
    num_games: int
    checkpoint_path: str
    seed: int

    @property
    def num_states(self) -> int:
        return self.states.shape[0]

    def save(self, path: str | Path) -> None:
        """Save dataset to compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states,
            legal_masks=self.legal_masks,
            phases=self.phases,
            active_players=self.active_players,
            meta=np.array([self.num_games, self.seed]),
            checkpoint_path=np.array(self.checkpoint_path),
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Saved {self.num_states} states to {path} ({size_mb:.1f} MB)")

    @staticmethod
    def load(path: str | Path) -> InterpDataset:
        """Load dataset from .npz file."""
        data = np.load(path, allow_pickle=True)
        meta = data["meta"]
        return InterpDataset(
            states=data["states"],
            legal_masks=data["legal_masks"],
            phases=data["phases"],
            active_players=data["active_players"],
            num_games=int(meta[0]),
            checkpoint_path=str(data["checkpoint_path"]),
            seed=int(meta[1]),
        )


def load_model(
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str = "checkpoints",
    device: str | None = None,
) -> tuple[torch.nn.Module, TrainingConfig, torch.device, int]:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to .pt file, or None to use latest.
        checkpoint_dir: Directory to search when checkpoint_path is None.
        device: "cuda", "cpu", or None for auto-detect.

    Returns:
        (model, config, device, epoch)
    """
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if checkpoint_path is None:
        cp_path = find_latest_checkpoint(Path(checkpoint_dir))
        if cp_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    else:
        cp_path = Path(checkpoint_path)

    print(f"Loading checkpoint: {cp_path}")
    cp = load_checkpoint(cp_path, dev)
    config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]

    model = create_model(
        config.model_arch,
        input_dim=config.visible_size,
        action_dim=config.action_dim,
        value_dim=config.num_players,
    ).to(dev)
    model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
    model.eval()

    epoch = int(cp.get("epoch", -1))  # type: ignore[arg-type]
    print(f"Epoch {epoch}, device={dev}")
    return model, config, dev, epoch


def collect_states(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    num_games: int = 50,
    seed: int = 0,
    checkpoint_path: str = "",
    max_moves_per_game: int = 1000,
) -> InterpDataset:
    """Play fast games via policy sampling to collect diverse game states.

    No MCTS search — just a single NN forward pass per decision point,
    then sample from the policy. Much faster than full self-play.

    Args:
        model: Trained model (eval mode).
        config: Training config from checkpoint.
        device: Torch device.
        num_games: Number of games to play.
        seed: Base seed (game i uses seed + i).
        checkpoint_path: Recorded in metadata for provenance.
        max_moves_per_game: Safety cap to avoid infinite loops.

    Returns:
        InterpDataset with all decision points across all games.
    """
    evaluator = NNEvaluator(model, device, num_players=config.num_players)
    rng = np.random.default_rng(seed)

    all_states: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_phases: list[int] = []
    all_active: list[int] = []

    t0 = time.perf_counter()
    for game_idx in range(num_games):
        game_seed = seed + game_idx
        state = GameState(config.num_players)
        state.initialize_game(seed=game_seed)
        moves = 0

        while state.get_phase() != GamePhases.PHASE_GAME_OVER:
            active_player = state.get_active_player()
            phase = state.get_phase()

            policy_probs, _, legal_mask = evaluator.evaluate(state)

            rotated = rotate_visible_state(
                state._array, active_player, config.num_players
            )
            all_states.append(rotated)
            all_masks.append(legal_mask)
            all_phases.append(phase)
            all_active.append(active_player)

            action = int(rng.choice(config.action_dim, p=policy_probs))
            status = DRIVER.apply_action(state, action)
            moves += 1

            if status == STATUS_GAME_OVER_PY or moves >= max_moves_per_game:
                break

        if (game_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  Game {game_idx + 1}/{num_games} "
                f"({elapsed:.1f}s, {len(all_states)} states)"
            )

    elapsed = time.perf_counter() - t0
    print(f"Collected {len(all_states)} states from {num_games} games in {elapsed:.1f}s")

    return InterpDataset(
        states=np.array(all_states, dtype=np.float32),
        legal_masks=np.array(all_masks, dtype=np.float32),
        phases=np.array(all_phases, dtype=np.int32),
        active_players=np.array(all_active, dtype=np.int32),
        num_games=num_games,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )
