"""AlphaZero training step with policy and value losses.

Sparse post-refactor contract: batches carry compact int16 game states and
per-row sparse legal-action data (`phase_id`, `n_legal`, `action_ids`,
`policy_target`). The trainer materializes the
``(batch, num_tokens, token_dim)`` float32 token buffer at training time via
``core.token_data.get_token_data`` and runs policy CE over the legal set only
— no dense ``-1e9`` mask, no state rotation, no per-player augmentation.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from core.state import GameState, get_layout
from core.token_data import TokenDataSize, get_num_tokens, get_token_data
from train.config import TrainingConfig

TOKEN_DIM = int(TokenDataSize.TOKEN_DIM)

# Matches core.data.DecisionPhase order (DPHASE_INVEST..DPHASE_IPO, 8 entries).
# PAR was folded into IPO; ACQ_OFFER is a first-class decision phase.
_PHASE_NAMES = [
    "invest", "bid", "acq", "acq_offer",
    "close", "div", "issue", "ipo",
]


class Trainer:
    """Handles optimizer, LR schedule, and training steps."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self._global_step = 0

        self._num_players = config.num_players
        self._num_tokens = get_num_tokens(config.num_players)
        self._state_size = get_layout(config.num_players).total_size

        # Scratch GameState rebinds across batch rows so each train_step
        # avoids allocating a wrapper per row. Default ctor seeds the
        # canonical num_players slot that rebind() validates against;
        # the backing array gets overwritten on first rebind().
        self._scratch_state: GameState = GameState(config.num_players)

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

    def _build_token_batch(
        self, states_cpu: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize the (B, num_tokens, TOKEN_DIM) f32 token buffer.

        ``states_cpu`` is a contiguous int16 tensor of shape
        ``(batch, state_size)``. For each row we rebind the scratch
        GameState onto the row's backing memory and fill one slice of
        the output buffer via the nogil ``get_token_data`` path.
        """
        batch = states_cpu.shape[0]
        token_buf = np.empty(
            (batch, self._num_tokens, TOKEN_DIM), dtype=np.float32,
        )
        states_np = states_cpu.numpy()  # zero-copy view of the int16 tensor
        scratch = self._scratch_state
        num_players = self._num_players
        for i in range(batch):
            # rebind zero-copies onto row i; canonical num_players slot is
            # assumed present (replay buffer stored raw state arrays
            # produced by self_play, which always carry it).
            scratch.rebind(states_np[i], num_players)
            get_token_data(scratch, token_buf[i])
        return torch.from_numpy(token_buf)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: dict with keys ``states`` (int16 ``(B, state_size)``),
                ``phase_ids`` (int8 ``(B,)``), ``n_legals`` (int16 ``(B,)``),
                ``action_ids`` (int16/uint16 reinterp ``(B, K_MAX)``),
                ``policy_targets`` (float32 ``(B, K_MAX)`` zero-padded past
                ``n_legal``), and ``value_targets`` (float32 ``(B, N)``).
                Tensors on CPU; moved to device here.

        Returns:
            dict with ``policy_loss``, ``value_loss``, ``total_loss`` as
            floats, plus ``policy_loss_<phase>`` per decision phase bucket
            present in the batch.
        """
        # Token buffer fill happens on CPU (get_token_data is a nogil Cython
        # kernel — overlaps well with DataLoader worker parallelism).
        states_cpu = batch["states"]
        token_buf_cpu = self._build_token_batch(states_cpu)

        # non_blocking=True allows CPU work to overlap with DMA transfer.
        nb = self.device.type == "cuda"
        tokens = token_buf_cpu.to(self.device, non_blocking=nb)
        phase_ids = batch["phase_ids"].to(
            self.device, non_blocking=nb, dtype=torch.long,
        )
        action_ids = batch["action_ids"].to(
            self.device, non_blocking=nb, dtype=torch.long,
        )  # (B, K_MAX) — int16 values fit losslessly in long
        n_legals = batch["n_legals"].to(
            self.device, non_blocking=nb, dtype=torch.long,
        )
        policy_targets = batch["policy_targets"].to(self.device, non_blocking=nb)
        value_targets = batch["value_targets"].to(self.device, non_blocking=nb)

        # --- Forward: sparse (B, K_MAX) logits + (B, N) values ---
        # The model gathers per-row legal slices internally and fills the
        # [n_legal:K_MAX] tail with -1e9, so we can log_softmax directly.
        policy_logits, values = self.model(tokens, phase_ids, action_ids, n_legals)

        # Policy loss: sparse cross-entropy over legal actions only.
        # log_softmax is numerically stable (x - logsumexp), so the tail
        # log_probs are finite-but-very-negative; multiplied by the zero
        # tail of policy_targets they contribute nothing.
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

        # Per-phase policy loss (detached, no grad impact).
        per_phase = per_example_policy_loss.detach()
        result: dict[str, float] = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
        }
        for phase_idx, name in enumerate(_PHASE_NAMES):
            mask = phase_ids == phase_idx
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
