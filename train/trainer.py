"""AlphaZero training step with policy and value losses."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F

from core.state import LayoutInfo
from train.augment import apply_player_permutation, random_player_permutation
from train.config import TrainingConfig

_PHASE_NAMES = ["invest", "bid", "acq", "close", "div", "issue", "ipo", "par"]


class Trainer:
    """Handles optimizer, LR schedule, and training steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        device: torch.device,
        layout: LayoutInfo | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self._global_step = 0

        # Layout for player-slot permutation augmentation.
        if layout is None:
            from core.state import get_layout
            layout = get_layout(config.num_players)
        self._layout: LayoutInfo = layout

        # --- Optimizer setup ---
        if config.optimizer == "muon":
            self._setup_muon(model, config)
        else:
            self._setup_adamw(model, config)

        # --- LR schedule (shared shape for all optimizers) ---
        decay_epochs = config.lr_decay_end_epoch or config.num_epochs
        total_steps = decay_epochs * config.training_steps_per_epoch
        warmup_steps = config.warmup_steps
        lr_min_ratio = config.lr_min / config.learning_rate

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_decay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        if self._aux_optimizer is not None:
            self._aux_scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
                torch.optim.lr_scheduler.LambdaLR(self._aux_optimizer, lr_lambda)
            )
        else:
            self._aux_scheduler = None

    def _setup_adamw(
        self, model: torch.nn.Module, config: TrainingConfig
    ) -> None:
        """Standard AdamW with decay/no-decay param groups."""
        decay_params = []
        no_decay_params = []
        for module in model.modules():
            for pname, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if isinstance(module, torch.nn.LayerNorm) or pname == "bias":
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        self.optimizer: torch.optim.Optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.learning_rate)
        self._aux_optimizer: torch.optim.Optimizer | None = None

    def _setup_muon(
        self, model: torch.nn.Module, config: TrainingConfig
    ) -> None:
        """Muon for 2D weights, auxiliary AdamW for 1D params.

        Uses adjust_lr_fn="match_rms_adamw" (Moonshot) so Muon reuses
        the same LR and weight decay as AdamW — no separate tuning needed.
        """
        muon_params: list[torch.nn.Parameter] = []
        adam_decay: list[torch.nn.Parameter] = []
        adam_no_decay: list[torch.nn.Parameter] = []
        for module in model.modules():
            for pname, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if param.ndim >= 2 and not isinstance(module, torch.nn.LayerNorm):
                    muon_params.append(param)
                elif isinstance(module, torch.nn.LayerNorm) or pname == "bias":
                    adam_no_decay.append(param)
                else:
                    adam_decay.append(param)

        self.optimizer = torch.optim.Muon(
            muon_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )

        aux_groups: list[dict[str, object]] = []
        if adam_decay:
            aux_groups.append({
                "params": adam_decay, "weight_decay": config.weight_decay,
            })
        if adam_no_decay:
            aux_groups.append({
                "params": adam_no_decay, "weight_decay": 0.0,
            })
        self._aux_optimizer = (
            torch.optim.AdamW(aux_groups, lr=config.learning_rate)
            if aux_groups else None
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: dict with "states", "legal_masks", "policy_targets",
                   "value_targets". Tensors on CPU; moved to device here.

        Returns:
            dict with "policy_loss", "value_loss", "total_loss" as floats.
        """
        # non_blocking=True allows CPU work to overlap with DMA transfer.
        # On NVIDIA GH200 with NVLink-C2C this is especially beneficial.
        nb = self.device.type == "cuda"
        states = batch["states"].to(self.device, non_blocking=nb)
        legal_masks = batch["legal_masks"].to(self.device, non_blocking=nb)
        policy_targets = batch["policy_targets"].to(self.device, non_blocking=nb)
        value_targets = batch["value_targets"].to(self.device, non_blocking=nb)

        # Player-slot permutation augmentation: randomly shuffle inactive
        # player slots (1..N-1) to teach slot-order invariance.
        perm = random_player_permutation(
            self._layout.num_players, states.device
        )
        apply_player_permutation(states, value_targets, perm, self._layout)

        # Forward + loss in float32 (inference uses bfloat16 autocast separately)
        policy_logits, values = self.model(states)
        policy_logits.masked_fill_(legal_masks <= 0, -1e9)

        # Policy loss: cross-entropy with MCTS target distribution
        log_probs = F.log_softmax(policy_logits, dim=-1)
        per_example_policy_loss = -(policy_targets * log_probs).sum(dim=-1)
        policy_loss = per_example_policy_loss.mean()

        # Value loss: MSE
        value_loss = F.mse_loss(values, value_targets)

        # Combined loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        if torch.isnan(total_loss):
            raise RuntimeError(
                f"NaN loss at step {self._global_step}: "
                f"policy={policy_loss.item()}, value={value_loss.item()}"
            )

        # Backward + optimize
        self.optimizer.zero_grad()
        if self._aux_optimizer is not None:
            self._aux_optimizer.zero_grad()

        total_loss.backward()

        self.optimizer.step()
        self.scheduler.step()
        if self._aux_optimizer is not None:
            self._aux_optimizer.step()
        if self._aux_scheduler is not None:
            self._aux_scheduler.step()

        self._global_step += 1

        # Per-phase policy loss (detached, no grad impact)
        phases = states[:, :8].argmax(dim=-1)
        per_phase = per_example_policy_loss.detach()
        result: dict[str, float] = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }
        for phase_idx, name in enumerate(_PHASE_NAMES):
            mask = phases == phase_idx
            if mask.any():
                result[f"policy_loss_{name}"] = per_phase[mask].mean().item()

        return result

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @property
    def aux_lr(self) -> float | None:
        if self._aux_optimizer is not None:
            return self._aux_optimizer.param_groups[0]["lr"]
        return None

    def state_dict(self) -> dict[str, object]:
        """Return state for checkpointing."""
        state: dict[str, object] = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self._global_step,
        }
        if self._aux_optimizer is not None:
            state["aux_optimizer"] = self._aux_optimizer.state_dict()
        if self._aux_scheduler is not None:
            state["aux_scheduler"] = self._aux_scheduler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore from checkpoint.  Tolerates missing or mismatched
        optimizer/scheduler (e.g. after optimizer type change) — they
        just start fresh with a warning."""
        if "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
            except (ValueError, KeyError):
                warnings.warn(
                    "Could not restore optimizer state (optimizer type may have "
                    "changed). Starting optimizer from scratch."
                )
        if "scheduler" in state:
            try:
                self.scheduler.load_state_dict(state["scheduler"])  # type: ignore[arg-type]
            except (ValueError, KeyError):
                warnings.warn(
                    "Could not restore scheduler state. "
                    "Starting scheduler from scratch."
                )
        if self._aux_optimizer is not None and "aux_optimizer" in state:
            try:
                self._aux_optimizer.load_state_dict(state["aux_optimizer"])  # type: ignore[arg-type]
            except (ValueError, KeyError):
                warnings.warn(
                    "Could not restore aux optimizer state. "
                    "Starting aux optimizer from scratch."
                )
        if self._aux_scheduler is not None and "aux_scheduler" in state:
            try:
                self._aux_scheduler.load_state_dict(state["aux_scheduler"])  # type: ignore[arg-type]
            except (ValueError, KeyError):
                warnings.warn(
                    "Could not restore aux scheduler state. "
                    "Starting aux scheduler from scratch."
                )
        assert isinstance(state["global_step"], int)
        self._global_step = state["global_step"]
