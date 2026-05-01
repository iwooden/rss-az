"""Training configuration for AlphaZero self-play."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from nn.model_contract import ModelKind, normalize_model_type


@dataclass
class MCTSConfig:
    """Hyperparameters for MCTS search."""

    num_simulations: int = 800
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.8
    dirichlet_epsilon: float = 0.25
    dirichlet_dynamic: bool = True
    dirichlet_alpha_numerator: float = 10.0
    num_players: int = 3
    search_batch_size: int = 8
    action_dim: int = field(init=False)

    def __post_init__(self) -> None:
        from core.data import MAX_ACTION_SIZE

        # Post-refactor: action space is per-phase sparse with a fixed dense
        # pad width (MAX_ACTION_SIZE, max over all phases — currently INVEST
        # at 53 after the ACQ split) that doesn't vary with player count.
        # get_action_probabilities returns a (MAX_ACTION_SIZE,) dense distribution.
        self.action_dim = int(MAX_ACTION_SIZE)
        self.validate()

    def validate(self) -> None:
        """Validate all fields. Called from __post_init__ and after CLI overrides."""
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        if self.search_batch_size < 1:
            raise ValueError(
                f"search_batch_size must be >= 1, got {self.search_batch_size}"
            )
        if not 3 <= self.num_players <= 5:
            raise ValueError(f"num_players must be 3-5, got {self.num_players}")
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
    num_simulations: int = 0  # 0 = use config default


@dataclass
class TrainingConfig:
    """All hyperparameters for the self-play training loop."""

    # --- Game ---
    num_players: int = 3

    # --- Inference ---
    eval_dtype: str | None = None  # None = no autocast; "bfloat16" or "float16"

    # --- Model ---
    model_type: str = ModelKind.TRANSFORMER.value
    # Per-block adaLN-Zero conditioning on the active decision phase.
    phase_conditioning: bool = True
    # Price-like policy slots blend fixed Fourier projections for smoothness
    # with learned per-slot embeddings for slot identity.
    price_slot_fourier_bands: int = 4
    # 0.0 = pure Fourier projection, 1.0 = pure learned slot embedding.
    price_slot_residual_scale: float = 1.0
    # Residual MLP model hyperparameters.
    resnet_hidden_dim: int = 256
    resnet_num_blocks: int = 10

    # --- Self-Play ---
    games_per_epoch: int = 500
    num_simulations: int = 800
    dirichlet_alpha: float = 0.8
    dirichlet_epsilon: float = 0.25
    dirichlet_dynamic: bool = True
    dirichlet_alpha_numerator: float = 10.0
    search_batch_size: int = 8
    num_workers: int = 4
    num_eval_servers: int = 1
    # When > 0, eval servers accumulate drained requests until the total
    # number of pending states reaches this floor before launching a GPU
    # forward pass. 0 (default) = greedy: submit on every non-empty drain,
    # no matter how small. Flush timeout below is only consulted when
    # this is non-zero.
    eval_min_batch_size: int = 0
    # Short timeout (ms) the min-batch loop waits on the doorbell when
    # the accumulator is non-empty but below the floor. On timeout, the
    # partial accumulation is submitted — anti-starvation for epoch-end
    # and single-root-eval cases.
    eval_min_batch_timeout_ms: float = 10.0
    # Eval batch-shape policy. "dynamic" preserves the current fully variable
    # batch path. "bucketed" will round launches to a small fixed bucket set
    # for CUDA-graph-friendly eval inference.
    eval_batch_shape_mode: str = "dynamic"
    # In bucketed mode, optional ceiling on actual states per launch before
    # power-of-2 padding. 0 means use the partition's natural max batch.
    # This is distinct from eval_min_batch_size, which controls when to launch.
    eval_max_batch_size: int = 0

    # --- Temperature Schedule (linear ramp) ---
    # temp_initial from move 0 to temp_anneal_start, then linearly decreases
    # to temp_final at temp_anneal_end. Stays at temp_final after that.
    # Measured in total game decision points (MCTS searches), not per-player.
    temp_initial: float = 1.0
    temp_anneal_start: int = 60
    temp_anneal_end: int = 120
    temp_final: float = 0.5

    # --- MCTS simulation ramp (linear) ---
    # When set, num_simulations ramps linearly from mcts_sims_start to
    # mcts_sims_end between mcts_ramp_start_epoch and mcts_ramp_end_epoch.
    # Before start epoch: use mcts_sims_start. After end epoch: use mcts_sims_end.
    # When None, num_simulations is used as a fixed value throughout training.
    mcts_sims_start: int | None = None
    mcts_sims_end: int | None = None
    mcts_ramp_start_epoch: int | None = None
    mcts_ramp_end_epoch: int | None = None

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
    value_blend_end_epoch: int = 200

    # --- Terminal reward blending ---
    # Blend between rank-based rewards (1.0) and net-worth-margin rewards (0.0).
    # Default 0.5 = equal blend. Set to 1.0 for pure [-1, 0, +1] rank rewards.
    terminal_blend: float = 0.75

    # --- Replay Buffer ---
    buffer_capacity: int = 500_000
    min_buffer_size: int = 10_000

    # --- Training ---
    batch_size: int = 256
    optimizer: str = "muon"         # "adamw" or "muon"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0          # global-norm clip; 0 disables
    training_steps_per_epoch: int = 1000

    # --- LR Schedule ---
    # Cosine annealing with linear warmup. Measured in training epochs so
    # changing training_steps_per_epoch preserves the same schedule shape.
    warmup_epochs: float = 1.0
    lr_min: float = 1e-4
    lr_decay_end_epoch: int | None = 200  # epoch where LR reaches lr_min (default: num_epochs)

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
    num_epochs: int = 500
    seed: int = 42

    # --- Profiling (operational, not checkpointed) ---
    profile: bool = False

    # --- Computed ---
    action_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate all fields and recompute derived state.

        Called from __post_init__ and after CLI/JSON overrides.
        """
        from core.data import MAX_ACTION_SIZE

        # Post-refactor: dense action pad width is player-count independent.
        # The transformer eval buffer is keyed off core.token_data's
        # (num_tokens, token_dim) — there is no flat state width.
        self.action_dim = int(MAX_ACTION_SIZE)

        self.model_type = normalize_model_type(self.model_type).value

        # Eval dtype
        if self.eval_dtype is not None and self.eval_dtype not in ("bfloat16", "float16"):
            raise ValueError(
                f"eval_dtype must be None, 'bfloat16', or 'float16', "
                f"got {self.eval_dtype!r}"
            )
        if not isinstance(self.phase_conditioning, bool):
            raise ValueError(
                f"phase_conditioning must be bool, got {self.phase_conditioning!r}"
            )
        if self.price_slot_fourier_bands < 0:
            raise ValueError(
                "price_slot_fourier_bands must be >= 0, "
                f"got {self.price_slot_fourier_bands}"
            )
        if not 0.0 <= self.price_slot_residual_scale <= 1.0:
            raise ValueError(
                "price_slot_residual_scale must be in [0, 1], "
                f"got {self.price_slot_residual_scale}"
            )
        if self.resnet_hidden_dim < 1:
            raise ValueError(
                f"resnet_hidden_dim must be >= 1, got {self.resnet_hidden_dim}"
            )
        if self.resnet_num_blocks < 0:
            raise ValueError(
                f"resnet_num_blocks must be >= 0, got {self.resnet_num_blocks}"
            )

        # MCTS fields
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations must be >= 1, got {self.num_simulations}")
        # Sim ramp: all four fields must be set together, or all None
        ramp_fields = (self.mcts_sims_start, self.mcts_sims_end,
                       self.mcts_ramp_start_epoch, self.mcts_ramp_end_epoch)
        ramp_set = sum(f is not None for f in ramp_fields)
        if ramp_set not in (0, 4):
            raise ValueError(
                "mcts_sims_start, mcts_sims_end, mcts_ramp_start_epoch, and "
                "mcts_ramp_end_epoch must all be set or all be None"
            )
        if ramp_set == 4:
            assert self.mcts_sims_start is not None and self.mcts_sims_end is not None
            assert self.mcts_ramp_start_epoch is not None and self.mcts_ramp_end_epoch is not None
            if self.mcts_sims_start < 1:
                raise ValueError(f"mcts_sims_start must be >= 1, got {self.mcts_sims_start}")
            if self.mcts_sims_end < 1:
                raise ValueError(f"mcts_sims_end must be >= 1, got {self.mcts_sims_end}")
            if self.mcts_ramp_start_epoch > self.mcts_ramp_end_epoch:
                raise ValueError(
                    f"mcts_ramp_start_epoch ({self.mcts_ramp_start_epoch}) must be <= "
                    f"mcts_ramp_end_epoch ({self.mcts_ramp_end_epoch})"
                )
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
        if not 3 <= self.num_players <= 5:
            raise ValueError(f"num_players must be 3-5, got {self.num_players}")

        # Optimizer
        if self.optimizer not in ("adamw", "muon"):
            raise ValueError(
                f"optimizer must be 'adamw' or 'muon', got {self.optimizer!r}"
            )
        if self.grad_clip < 0:
            raise ValueError(
                f"grad_clip must be >= 0 (0 disables), got {self.grad_clip}"
            )
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be >= 0, got {self.warmup_epochs}"
            )

        # Training fields
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.training_steps_per_epoch < 1:
            raise ValueError(
                f"training_steps_per_epoch must be >= 1, "
                f"got {self.training_steps_per_epoch}"
            )
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.num_eval_servers < 1:
            raise ValueError(
                f"num_eval_servers must be >= 1, got {self.num_eval_servers}"
            )
        if self.num_workers > 0 and self.num_eval_servers > self.num_workers:
            raise ValueError(
                f"num_eval_servers ({self.num_eval_servers}) must be <= "
                f"num_workers ({self.num_workers})"
            )
        if self.eval_min_batch_size < 0:
            raise ValueError(
                f"eval_min_batch_size must be >= 0 (0 disables), "
                f"got {self.eval_min_batch_size}"
            )
        if self.num_workers > 0 and self.eval_min_batch_size > 0:
            max_partition = -(-self.num_workers // self.num_eval_servers)
            max_batch = max_partition * self.search_batch_size
            if self.eval_min_batch_size > max_batch:
                raise ValueError(
                    f"eval_min_batch_size ({self.eval_min_batch_size}) "
                    f"must be <= partition_size * search_batch_size "
                    f"({max_batch})"
                )
        if self.eval_min_batch_timeout_ms < 0:
            raise ValueError(
                f"eval_min_batch_timeout_ms must be >= 0, "
                f"got {self.eval_min_batch_timeout_ms}"
            )
        if self.eval_batch_shape_mode not in {"dynamic", "bucketed"}:
            raise ValueError(
                "eval_batch_shape_mode must be 'dynamic' or 'bucketed', "
                f"got {self.eval_batch_shape_mode!r}"
            )
        if self.eval_max_batch_size < 0:
            raise ValueError(
                f"eval_max_batch_size must be >= 0, got {self.eval_max_batch_size}"
            )
        if self.eval_batch_shape_mode == "dynamic":
            if self.eval_max_batch_size != 0:
                raise ValueError(
                    "eval_max_batch_size must be 0 when eval_batch_shape_mode is "
                    f"'dynamic', got {self.eval_max_batch_size}"
                )
        elif self.eval_max_batch_size > 0:
            if self.eval_max_batch_size & (self.eval_max_batch_size - 1):
                raise ValueError(
                    "eval_max_batch_size must be a power of 2 in bucketed mode, "
                    f"got {self.eval_max_batch_size}"
                )
            if self.num_workers > 0:
                max_partition = -(-self.num_workers // self.num_eval_servers)
                max_batch = max_partition * self.search_batch_size
                if self.eval_max_batch_size > max_batch:
                    raise ValueError(
                        f"eval_max_batch_size ({self.eval_max_batch_size}) must be <= "
                        f"partition_size * search_batch_size ({max_batch})"
                    )
            if self.eval_min_batch_size > 0 and self.eval_max_batch_size < self.eval_min_batch_size:
                raise ValueError(
                    f"eval_max_batch_size ({self.eval_max_batch_size}) must be >= "
                    f"eval_min_batch_size ({self.eval_min_batch_size}) in bucketed mode"
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

    @property
    def max_simulations(self) -> int:
        """Maximum possible num_simulations (for StatePool sizing)."""
        if self.mcts_sims_end is not None:
            return max(self.num_simulations, self.mcts_sims_end)
        return self.num_simulations

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

        # MCTS simulation count ramp
        if (self.mcts_sims_start is not None and self.mcts_sims_end is not None
                and self.mcts_ramp_start_epoch is not None
                and self.mcts_ramp_end_epoch is not None):
            if epoch <= self.mcts_ramp_start_epoch:
                num_sims = self.mcts_sims_start
            elif epoch >= self.mcts_ramp_end_epoch:
                num_sims = self.mcts_sims_end
            else:
                span = self.mcts_ramp_end_epoch - self.mcts_ramp_start_epoch
                t = (epoch - self.mcts_ramp_start_epoch) / max(span, 1)
                num_sims = round(
                    self.mcts_sims_start + t * (self.mcts_sims_end - self.mcts_sims_start)
                )
        else:
            num_sims = self.num_simulations

        return EpochConfig(
            c_puct=c_puct,
            value_blend_alpha=value_blend_alpha,
            num_simulations=num_sims,
        )

    def to_mcts_config(
        self,
        c_puct_override: float | None = None,
        num_simulations_override: int | None = None,
    ) -> MCTSConfig:
        """Create an MCTSConfig from the relevant training fields.

        Args:
            c_puct_override: If provided, use this c_puct instead of c_puct_final.
            num_simulations_override: If provided, use this instead of self.num_simulations.
        """
        return MCTSConfig(
            num_simulations=(num_simulations_override if num_simulations_override is not None
                             else self.num_simulations),
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
        # Remove computed fields — they'll be recomputed on load
        d.pop("action_dim", None)
        return json.dumps(d, indent=2)

    @staticmethod
    def _normalize_json(d: dict[str, Any]) -> dict[str, Any]:
        """Drop computed/unknown fields so from_json + overrides stay strict."""
        valid = set(TrainingConfig.__dataclass_fields__)
        return {k: v for k, v in d.items() if k in valid}

    @classmethod
    def from_json(cls, json_str: str) -> TrainingConfig:
        """Deserialize from JSON."""
        d = cls._normalize_json(json.loads(json_str))
        return cls(**d)

    def apply_json_overrides(self, json_str: str) -> list[str]:
        """Apply fields from a JSON config on top of this config.

        Returns a list of "field = new_value (was old_value)" strings for logging.
        """
        d = self._normalize_json(json.loads(json_str))
        changes: list[str] = []
        for k, v in d.items():
            old = getattr(self, k)
            if old != v:
                setattr(self, k, v)
                changes.append(f"{k} = {v} (was {old})")
        return changes
