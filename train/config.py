"""Training configuration for AlphaZero self-play."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class MCTSConfig:
    """Hyperparameters for MCTS search."""

    num_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.8
    dirichlet_epsilon: float = 0.25
    dirichlet_dynamic: bool = False
    dirichlet_alpha_numerator: float = 10.0
    num_players: int = 3
    search_batch_size: int = 1
    action_dim: int = field(init=False)

    def __post_init__(self) -> None:
        from core.actions import get_total_action_count

        self.action_dim = get_total_action_count(self.num_players)
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
        if self.dirichlet_alpha_numerator <= 0:
            raise ValueError(
                f"dirichlet_alpha_numerator must be > 0, got {self.dirichlet_alpha_numerator}"
            )


@dataclass
class EpochConfig:
    """Per-epoch dynamic parameters sent to workers via task queue.

    These values change each epoch based on annealing schedules.
    Computed by TrainingConfig.compute_epoch_config() in the main process,
    then sent to workers alongside each game task.
    """

    c_puct: float
    value_blend_alpha: float  # 0.0 = pure game outcome, 1.0 = pure A0GB


@dataclass
class TrainingConfig:
    """All hyperparameters for the self-play training loop."""

    # --- Model ---
    model_arch: str = "v1"  # "v1" (model_3p) or "v2" (model_3p_2)

    # --- Game ---
    num_players: int = 3

    # --- Self-Play ---
    games_per_epoch: int = 1000
    num_simulations: int = 800
    dirichlet_alpha: float = 0.8
    dirichlet_epsilon: float = 0.25
    dirichlet_dynamic: bool = False
    dirichlet_alpha_numerator: float = 10.0
    search_batch_size: int = 1
    num_workers: int = 4
    num_eval_servers: int = 1

    # --- Temperature Schedule (linear ramp) ---
    # temp_initial from move 0 to temp_anneal_start, then linearly decreases
    # to temp_final at temp_anneal_end. Stays at temp_final after that.
    # Measured in total game decision points (MCTS searches), not per-player.
    temp_initial: float = 1.0
    temp_anneal_start: int = 60
    temp_anneal_end: int = 120
    temp_final: float = 0.5

    # --- c_puct annealing ---
    # Linear interpolation from c_puct_initial to c_puct_final over the
    # first c_puct_anneal_epochs epochs.
    c_puct_initial: float = 3.5
    c_puct_final: float = 2.5
    c_puct_anneal_epochs: int = 20

    # --- Value target blending ---
    # Blend between game outcome (alpha=0) and A0GB (alpha=1).
    # Pure game outcome for epochs < value_blend_start_epoch,
    # linear ramp to pure A0GB by value_blend_end_epoch.
    value_blend_start_epoch: int = 10
    value_blend_end_epoch: int = 40

    # --- Terminal reward blending ---
    # Blend between rank-based rewards (1.0) and net-worth-margin rewards (0.0).
    # Default 0.5 = equal blend. Set to 1.0 for pure [-1, 0, +1] rank rewards.
    terminal_blend: float = 0.5

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
    warmup_steps: int = 1000
    lr_min: float = 1e-4
    lr_decay_end_epoch: int | None = None  # epoch where LR reaches lr_min (default: num_epochs)

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
        from core.actions import get_total_action_count
        from core.state import get_layout

        self.action_dim = get_total_action_count(self.num_players)

        self.visible_size = get_layout(self.num_players).visible_size
        self.validate()

    def validate(self) -> None:
        """Validate all fields. Called from __post_init__ and after CLI overrides."""
        # Model arch
        if self.model_arch not in ("v1", "v2"):
            raise ValueError(f"model_arch must be 'v1' or 'v2', got '{self.model_arch}'")

        # MCTS fields
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.search_batch_size < 1:
            raise ValueError(
                f"search_batch_size must be >= 1, got {self.search_batch_size}"
            )
        if self.dirichlet_alpha <= 0:
            raise ValueError(
                f"dirichlet_alpha must be > 0, got {self.dirichlet_alpha}"
            )
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError(
                f"dirichlet_epsilon must be in [0, 1], got {self.dirichlet_epsilon}"
            )
        if self.dirichlet_alpha_numerator <= 0:
            raise ValueError(
                f"dirichlet_alpha_numerator must be > 0, got {self.dirichlet_alpha_numerator}"
            )

        # Temperature schedule
        if self.temp_anneal_start > self.temp_anneal_end:
            raise ValueError(
                f"temp_anneal_start ({self.temp_anneal_start}) must be <= "
                f"temp_anneal_end ({self.temp_anneal_end})"
            )

        # c_puct annealing
        if self.c_puct_initial < 0:
            raise ValueError(f"c_puct_initial must be >= 0, got {self.c_puct_initial}")
        if self.c_puct_final < 0:
            raise ValueError(f"c_puct_final must be >= 0, got {self.c_puct_final}")
        if self.c_puct_anneal_epochs < 0:
            raise ValueError(
                f"c_puct_anneal_epochs must be >= 0, got {self.c_puct_anneal_epochs}"
            )

        # Value blend
        if self.value_blend_start_epoch > self.value_blend_end_epoch:
            raise ValueError(
                f"value_blend_start_epoch ({self.value_blend_start_epoch}) must be <= "
                f"value_blend_end_epoch ({self.value_blend_end_epoch})"
            )

        # Terminal blend
        if not 0.0 <= self.terminal_blend <= 1.0:
            raise ValueError(
                f"terminal_blend must be in [0, 1], got {self.terminal_blend}"
            )

        # Game fields
        if self.num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {self.num_players}")

        # Training fields
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.num_eval_servers < 1:
            raise ValueError(
                f"num_eval_servers must be >= 1, got {self.num_eval_servers}"
            )
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

    def compute_epoch_config(self, epoch: int) -> EpochConfig:
        """Compute per-epoch dynamic values for annealing schedules.

        Args:
            epoch: Zero-indexed epoch number.
        """
        # c_puct: linear anneal over first N epochs
        if self.c_puct_anneal_epochs <= 0 or epoch >= self.c_puct_anneal_epochs:
            c_puct = self.c_puct_final
        else:
            t = epoch / self.c_puct_anneal_epochs
            c_puct = self.c_puct_initial + t * (self.c_puct_final - self.c_puct_initial)

        # Value blend alpha: 0.0 = pure game outcome, 1.0 = pure A0GB
        if epoch < self.value_blend_start_epoch:
            value_blend_alpha = 0.0
        elif epoch >= self.value_blend_end_epoch:
            value_blend_alpha = 1.0
        else:
            span = self.value_blend_end_epoch - self.value_blend_start_epoch
            value_blend_alpha = (epoch - self.value_blend_start_epoch) / max(span, 1)

        return EpochConfig(
            c_puct=c_puct,
            value_blend_alpha=value_blend_alpha,
        )

    def to_mcts_config(self, c_puct_override: float | None = None) -> MCTSConfig:
        """Create an MCTSConfig from the relevant training fields.

        Args:
            c_puct_override: If provided, use this c_puct instead of c_puct_final.
        """
        return MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=c_puct_override if c_puct_override is not None else self.c_puct_final,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            dirichlet_dynamic=self.dirichlet_dynamic,
            dirichlet_alpha_numerator=self.dirichlet_alpha_numerator,
            num_players=self.num_players,
            search_batch_size=self.search_batch_size,
        )

    def to_json(self) -> str:
        """Serialize to JSON for checkpoint storage."""
        d = asdict(self)
        # Remove computed/operational fields — they'll be recomputed on load
        d.pop("action_dim", None)
        d.pop("visible_size", None)
        d.pop("profile", None)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> TrainingConfig:
        """Deserialize from JSON, with backward compatibility for old configs."""
        d = json.loads(json_str)
        # Drop computed fields if present (they're recomputed in __post_init__)
        d.pop("action_dim", None)
        d.pop("visible_size", None)
        d.pop("profile", None)

        # Backward compat: map old field names to new
        if "temp_threshold" in d:
            if "temp_anneal_start" not in d:
                d["temp_anneal_start"] = d.pop("temp_threshold")
            else:
                d.pop("temp_threshold")
        if "c_puct" in d:
            if "c_puct_final" not in d:
                d["c_puct_final"] = d.pop("c_puct")
            else:
                d.pop("c_puct")

        # Drop any fields not in the dataclass (future-proofing)
        valid = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in valid}

        return cls(**d)
