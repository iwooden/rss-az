"""AlphaZero training step with policy and value losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from nn.model_3p import RSSAlphaZeroNet
from train.config import TrainingConfig


class Trainer:
    """Handles optimizer, LR schedule, and training steps."""

    def __init__(
        self,
        model: RSSAlphaZeroNet,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self._global_step = 0

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # total_steps assumes all epochs train. If early epochs skip training
        # (buffer not ready), the cosine decay stretches slightly — acceptable
        # since at most 1 epoch is skipped in practice.
        total_steps = config.num_epochs * config.training_steps_per_epoch
        warmup_steps = config.warmup_steps
        lr_min_ratio = config.lr_min / config.learning_rate

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_min_ratio + (1.0 - lr_min_ratio) * cosine_decay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: dict with "states", "legal_masks", "policy_targets",
                   "value_targets". Tensors on CPU; moved to device here.

        Returns:
            dict with "policy_loss", "value_loss", "total_loss" as floats.
        """
        states = batch["states"].to(self.device)
        legal_masks = batch["legal_masks"].to(self.device)
        policy_targets = batch["policy_targets"].to(self.device)
        value_targets = batch["value_targets"].to(self.device)

        # Forward pass with legal action masking
        policy_logits, values = self.model(states, legal_action_mask=legal_masks)

        # Policy loss: cross-entropy with MCTS target distribution
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(values, value_targets)

        # Combined loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        # Backward + optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()

        self._global_step += 1

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict[str, object]:
        """Return state for checkpointing."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self._global_step,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore from checkpoint."""
        self.optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        self.scheduler.load_state_dict(state["scheduler"])  # type: ignore[arg-type]
        assert isinstance(state["global_step"], int)
        self._global_step = state["global_step"]
