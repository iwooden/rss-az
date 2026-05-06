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
    min_players: int = 0
    max_players: int = 0

    # --- Inference ---
    eval_dtype: str | None = None  # None = no autocast; "bfloat16" or "float16"

    # --- Model ---
    model_type: str = ModelKind.TRANSFORMER.value
    # Optional Python module/file path that provides the implementation for
    # ``model_type``. For example: "nn/transformer-v2.py".
    model_path: str | None = None
    # Per-block adaLN-Zero conditioning on the active decision phase.
    phase_conditioning: bool = False
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
    # The scalar start/end fields act as global overrides when both are
    # non-zero. Set both to 0 to use the per-player-count arrays below, whose
    # index 0 corresponds to effective_min_players.
    temp_initial: float = 1.0
    temp_anneal_start: int = 60
    temp_anneal_end: int = 120
    temp_final: float = 0.5
    temp_anneal_starts: list[int] = field(default_factory=list)
    temp_anneal_ends: list[int] = field(default_factory=list)

    # --- Policy target temperature schedule ---
    # Independent temperature applied to the MCTS visit-count distribution
    # before storing replay policy targets. Defaults match the historical
    # action-temperature schedule used by production 3p training.
    # The scalar start/end fields use the same global-override semantics as
    # the action temperature schedule.
    policy_target_temp_initial: float = 1.0
    policy_target_temp_anneal_start: int = 60
    policy_target_temp_anneal_end: int = 120
    policy_target_temp_final: float = 0.5
    policy_target_temp_anneal_starts: list[int] = field(default_factory=list)
    policy_target_temp_anneal_ends: list[int] = field(default_factory=list)

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
        if self.model_path is not None:
            if not isinstance(self.model_path, str):
                raise ValueError(
                    f"model_path must be a string path/module name or None, got {self.model_path!r}"
                )
            stripped_model_path = self.model_path.strip()
            self.model_path = stripped_model_path or None

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

        # Game fields
        self._validate_player_count_mode()

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

        # Temperature schedules. Scalar start/end fields are global overrides
        # unless both are 0, in which case the per-player-count arrays are
        # required and validated.
        self.temp_anneal_starts, self.temp_anneal_ends = (
            self._resolve_move_schedule(
                self.temp_anneal_start,
                self.temp_anneal_end,
                self.temp_anneal_starts,
                self.temp_anneal_ends,
                "temp_anneal_start",
                "temp_anneal_end",
                "temp_anneal_starts",
                "temp_anneal_ends",
            )
        )
        (
            self.policy_target_temp_anneal_starts,
            self.policy_target_temp_anneal_ends,
        ) = self._resolve_move_schedule(
            self.policy_target_temp_anneal_start,
            self.policy_target_temp_anneal_end,
            self.policy_target_temp_anneal_starts,
            self.policy_target_temp_anneal_ends,
            "policy_target_temp_anneal_start",
            "policy_target_temp_anneal_end",
            "policy_target_temp_anneal_starts",
            "policy_target_temp_anneal_ends",
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

    @property
    def is_mixed_player_training(self) -> bool:
        """Whether this config trains over a real range of player counts."""
        return (
            self.num_players == 0
            and self.min_players != 0
            and self.max_players != 0
        )

    @property
    def effective_min_players(self) -> int:
        """Lowest actual player count covered by this training run."""
        if self.is_mixed_player_training:
            return self.min_players
        return self.num_players

    @property
    def effective_max_players(self) -> int:
        """Padded model/storage player capacity for this training run."""
        if self.is_mixed_player_training:
            return self.max_players
        return self.num_players

    def iter_player_counts(self) -> range:
        """Iterate all actual player counts this config may generate."""
        return range(self.effective_min_players, self.effective_max_players + 1)

    @staticmethod
    def _coerce_move_schedule(values: Any, name: str) -> list[int]:
        """Return a concrete integer move schedule list."""
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"{name} must be a list of move counts")
        result: list[int] = []
        for idx, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{name}[{idx}] must be an integer move count, got {value!r}"
                )
            result.append(value)
        return result

    def _resolve_move_schedule(
        self,
        scalar_start: int,
        scalar_end: int,
        starts: Any,
        ends: Any,
        scalar_start_name: str,
        scalar_end_name: str,
        starts_name: str,
        ends_name: str,
    ) -> tuple[list[int], list[int]]:
        """Resolve scalar-or-array move schedule fields to per-count arrays."""
        if isinstance(scalar_start, bool) or not isinstance(scalar_start, int):
            raise ValueError(f"{scalar_start_name} must be an integer")
        if isinstance(scalar_end, bool) or not isinstance(scalar_end, int):
            raise ValueError(f"{scalar_end_name} must be an integer")
        if scalar_start < 0 or scalar_end < 0:
            raise ValueError(
                f"{scalar_start_name}/{scalar_end_name} must be >= 0"
            )

        width = self.effective_max_players - self.effective_min_players + 1
        if scalar_start == 0 and scalar_end == 0:
            resolved_starts = self._coerce_move_schedule(starts, starts_name)
            resolved_ends = self._coerce_move_schedule(ends, ends_name)
            if len(resolved_starts) != width:
                raise ValueError(
                    f"{starts_name} must have {width} entries for player counts "
                    f"{self.effective_min_players}-{self.effective_max_players}, "
                    f"got {len(resolved_starts)}"
                )
            if len(resolved_ends) != width:
                raise ValueError(
                    f"{ends_name} must have {width} entries for player counts "
                    f"{self.effective_min_players}-{self.effective_max_players}, "
                    f"got {len(resolved_ends)}"
                )
        else:
            if scalar_start == 0 or scalar_end == 0:
                raise ValueError(
                    f"{scalar_start_name} and {scalar_end_name} must either "
                    "both be 0 to use per-player-count arrays, or both be > 0"
                )
            resolved_starts = [scalar_start] * width
            resolved_ends = [scalar_end] * width

        for idx, (start, end) in enumerate(zip(resolved_starts, resolved_ends)):
            num_players = self.effective_min_players + idx
            if start <= 0 or end <= 0:
                raise ValueError(
                    f"{starts_name}/{ends_name} entries must be > 0; "
                    f"{num_players}p has {start}-{end}"
                )
            if start > end:
                raise ValueError(
                    f"{starts_name}/{ends_name} entries must have start <= end; "
                    f"{num_players}p has {start}-{end}"
                )
        return resolved_starts, resolved_ends

    def _temperature_schedule_index(self, num_players: int) -> int:
        """Map an actual player count to its schedule array index."""
        if not (
            self.effective_min_players
            <= num_players
            <= self.effective_max_players
        ):
            raise ValueError(
                "num_players must be within the configured player range "
                f"{self.effective_min_players}-{self.effective_max_players}, "
                f"got {num_players}"
            )
        return num_players - self.effective_min_players

    def temp_anneal_window(self, num_players: int) -> tuple[int, int]:
        """Action-sampling temperature anneal window for a player count."""
        idx = self._temperature_schedule_index(num_players)
        return self.temp_anneal_starts[idx], self.temp_anneal_ends[idx]

    def policy_target_temp_anneal_window(
        self, num_players: int,
    ) -> tuple[int, int]:
        """Policy-target temperature anneal window for a player count."""
        idx = self._temperature_schedule_index(num_players)
        return (
            self.policy_target_temp_anneal_starts[idx],
            self.policy_target_temp_anneal_ends[idx],
        )

    def _validate_player_count_mode(self) -> None:
        """Validate single-count and mixed-count game configuration modes."""
        for name in ("num_players", "min_players", "max_players"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")

        has_single = self.num_players != 0
        has_min = self.min_players != 0
        has_max = self.max_players != 0
        has_range = has_min or has_max

        if has_single and has_range:
            raise ValueError(
                "num_players is mutually exclusive with min_players/max_players; "
                "use either single-count mode or mixed-count mode"
            )
        if not has_single and not has_range:
            raise ValueError(
                "player count config must set num_players or both min_players and max_players"
            )

        if has_single:
            if not 3 <= self.num_players <= 5:
                raise ValueError(
                    f"num_players must be 3-5, got {self.num_players}"
                )
            return

        if has_min != has_max:
            raise ValueError(
                "min_players and max_players must both be set for mixed player training"
            )
        if not 3 <= self.min_players <= 5:
            raise ValueError(
                f"min_players must be 3-5, got {self.min_players}"
            )
        if not 3 <= self.max_players <= 5:
            raise ValueError(
                f"max_players must be 3-5, got {self.max_players}"
            )
        if self.min_players >= self.max_players:
            raise ValueError(
                f"min_players must be < max_players for mixed player training, "
                f"got {self.min_players}-{self.max_players}"
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
        num_players: int | None = None,
    ) -> MCTSConfig:
        """Create an MCTSConfig from the relevant training fields.

        Args:
            c_puct_override: If provided, use this c_puct instead of c_puct_final.
            num_simulations_override: If provided, use this instead of self.num_simulations.
            num_players: Actual player count for the game/search. Required in
                mixed-count mode; optional in single-count mode.
        """
        if num_players is None:
            if self.is_mixed_player_training:
                raise ValueError(
                    "num_players must be passed to to_mcts_config() when "
                    "mixed player training is enabled"
                )
            actual_num_players = self.num_players
        else:
            actual_num_players = num_players

        if not (
            self.effective_min_players
            <= actual_num_players
            <= self.effective_max_players
        ):
            raise ValueError(
                "num_players must be within the configured player range "
                f"{self.effective_min_players}-{self.effective_max_players}, "
                f"got {actual_num_players}"
            )

        return MCTSConfig(
            num_simulations=(num_simulations_override if num_simulations_override is not None
                             else self.num_simulations),
            c_puct=c_puct_override if c_puct_override is not None else self.c_puct_final,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            dirichlet_dynamic=self.dirichlet_dynamic,
            dirichlet_alpha_numerator=self.dirichlet_alpha_numerator,
            num_players=actual_num_players,
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
