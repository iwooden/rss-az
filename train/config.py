"""Training configuration for AlphaZero self-play."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class MCTSConfig:
    """Hyperparameters for MCTS search."""

    num_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    num_players: int = 3
    action_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.action_dim = 186 + self.num_players * 20


@dataclass
class TrainingConfig:
    """All hyperparameters for the self-play training loop."""

    # --- Game ---
    num_players: int = 3

    # --- Self-Play ---
    games_per_epoch: int = 1000
    num_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # --- Temperature Schedule ---
    # temp_initial for the first `temp_threshold` decision points (MCTS searches),
    # then drops to temp_final. Measured in total game decisions, not per-player.
    temp_threshold: int = 60
    temp_initial: float = 1.0
    temp_final: float = 0.1

    # --- Replay Buffer ---
    buffer_capacity: int = 500_000
    min_buffer_size: int = 10_000

    # --- Training ---
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    training_steps_per_epoch: int = 1000

    # --- LR Schedule ---
    # Cosine annealing with linear warmup
    warmup_steps: int = 500
    lr_min: float = 1e-4

    # --- Loss ---
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    # --- Checkpointing ---
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 5
    keep_last_n: int = 10

    # --- Logging ---
    tensorboard_dir: str = "runs"
    log_interval: int = 100

    # --- Overall ---
    num_epochs: int = 100
    seed: int = 42

    # --- Computed ---
    action_dim: int = field(init=False)
    visible_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.action_dim = 186 + self.num_players * 20

        from core.state import get_layout

        self.visible_size = get_layout(self.num_players).visible_size

        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")
        if self.buffer_capacity <= self.min_buffer_size:
            raise ValueError(
                f"buffer_capacity ({self.buffer_capacity}) must exceed "
                f"min_buffer_size ({self.min_buffer_size})"
            )

    def to_mcts_config(self) -> MCTSConfig:
        """Create an MCTSConfig from the relevant training fields."""
        return MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            temperature=self.temp_initial,
            num_players=self.num_players,
        )

    def to_json(self) -> str:
        """Serialize to JSON for checkpoint storage."""
        d = asdict(self)
        # Remove computed fields — they'll be recomputed on load
        d.pop("action_dim", None)
        d.pop("visible_size", None)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> TrainingConfig:
        """Deserialize from JSON."""
        d = json.loads(json_str)
        # Drop computed fields if present (they're recomputed in __post_init__)
        d.pop("action_dim", None)
        d.pop("visible_size", None)
        return cls(**d)
