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
    num_players: int = 3
    search_batch_size: int = 1
    action_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.action_dim = 186 + self.num_players * 20
        self.validate()

    def validate(self) -> None:
        """Validate all fields. Called from __post_init__ and after CLI overrides."""
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.search_batch_size < 1:
            raise ValueError(
                f"search_batch_size must be >= 1, got {self.search_batch_size}"
            )
        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")
        if self.c_puct < 0:
            raise ValueError(f"c_puct must be >= 0, got {self.c_puct}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(
                f"dirichlet_alpha must be > 0, got {self.dirichlet_alpha}"
            )
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError(
                f"dirichlet_epsilon must be in [0, 1], got {self.dirichlet_epsilon}"
            )



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
    search_batch_size: int = 1
    num_workers: int = 4

    # --- Temperature Schedule ---
    # temp_initial for the first `temp_threshold` decision points (MCTS searches),
    # then drops to temp_final. Measured in total game decisions, not per-player.
    temp_threshold: int = 30
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

    # --- Profiling (operational, not checkpointed) ---
    profile: bool = False

    # --- Computed ---
    action_dim: int = field(init=False)
    visible_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.action_dim = 186 + self.num_players * 20

        from core.state import get_layout

        self.visible_size = get_layout(self.num_players).visible_size
        self.validate()

    def validate(self) -> None:
        """Validate all fields. Called from __post_init__ and after CLI overrides."""
        # MCTS fields
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.search_batch_size < 1:
            raise ValueError(
                f"search_batch_size must be >= 1, got {self.search_batch_size}"
            )
        if self.c_puct < 0:
            raise ValueError(f"c_puct must be >= 0, got {self.c_puct}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(
                f"dirichlet_alpha must be > 0, got {self.dirichlet_alpha}"
            )
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError(
                f"dirichlet_epsilon must be in [0, 1], got {self.dirichlet_epsilon}"
            )

        # Game fields
        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")

        # Training fields
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.buffer_capacity <= self.min_buffer_size:
            raise ValueError(
                f"buffer_capacity ({self.buffer_capacity}) must exceed "
                f"min_buffer_size ({self.min_buffer_size})"
            )
        if self.min_buffer_size < self.batch_size:
            raise ValueError(
                f"min_buffer_size ({self.min_buffer_size}) must be >= "
                f"batch_size ({self.batch_size})"
            )

    def to_mcts_config(self) -> MCTSConfig:
        """Create an MCTSConfig from the relevant training fields."""
        return MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=self.c_puct,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            num_players=self.num_players,
            search_batch_size=self.search_batch_size,
        )

    def to_json(self) -> str:
        """Serialize to JSON for checkpoint storage."""
        d = asdict(self)
        # Remove computed fields — they'll be recomputed on load
        d.pop("action_dim", None)
        d.pop("visible_size", None)
        d.pop("profile", None)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> TrainingConfig:
        """Deserialize from JSON."""
        d = json.loads(json_str)
        # Drop computed fields if present (they're recomputed in __post_init__)
        d.pop("action_dim", None)
        d.pop("visible_size", None)
        return cls(**d)
