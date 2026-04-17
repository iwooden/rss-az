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
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch._dynamo.decorators import mark_unbacked

from core.actions import MAX_LEGAL_ACTIONS_PY
from core.state import get_layout
from core.token_data import TokenDataSize, get_num_tokens, get_token_data_batch
from nn.transformer import NUM_PHASES
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer

TOKEN_DIM = int(TokenDataSize.TOKEN_DIM)
K_MAX = int(MAX_LEGAL_ACTIONS_PY)

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

        # Lazy pinned host + device scratch (see _ensure_scratch).
        self._scratch_cap: int = 0

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
                if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)) or pname == "bias":
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
                is_norm = isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm))
                # Muon only supports 2D matrix params. Stacked (3D+) tensors —
                # e.g. the per-offset ACQ bilinear factors — go to AdamW decay:
                # Newton-Schulz orthogonalization across the stack axis isn't
                # what Muon means by "matrix."
                if param.ndim == 2 and not is_norm:
                    muon_params.append(param)
                elif is_norm or pname == "bias":
                    adam_no_decay.append(param)
                else:
                    adam_decay.append(param)

        self.optimizer = torch.optim.Muon(
            muon_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            adjust_lr_fn="match_rms_adamw",
        )

        aux_groups: list[dict[str, Any]] = []
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

    def _ensure_scratch(self, n: int) -> None:
        """Grow preallocated pinned-host + device scratch to fit ``n`` rows.

        Grows in powers of two so steady-state shrinks don't realloc. On
        CUDA, host tensors are pinned so ``.copy_(non_blocking=True)``
        genuinely async-ships. On CPU, device tensors alias the host
        tensors and no H→D copy is needed.

        Integer fields are stored as ``torch.long`` to match the model's
        input contract — widening from the replay buffer's compact
        dtypes (int8/int16/uint16) happens once on the CPU fill.
        """
        if n <= self._scratch_cap:
            return
        cap = max(n, max(self._scratch_cap * 2, 1))
        pm = self.device.type == "cuda"
        nt, td = self._num_tokens, TOKEN_DIM
        N = self._num_players

        # Raw states: CPU-only (consumed by get_token_data), never shipped.
        self._states_np = np.empty((cap, self._state_size), dtype=np.int16)

        # Pinned host (on CUDA). Exposed as numpy for buffer fills.
        self._tok_h = torch.empty((cap, nt, td), dtype=torch.float32, pin_memory=pm)
        self._tok_h_np = self._tok_h.numpy()
        self._phase_h = torch.empty(cap, dtype=torch.long, pin_memory=pm)
        self._phase_h_np = self._phase_h.numpy()
        self._aid_h = torch.empty((cap, K_MAX), dtype=torch.long, pin_memory=pm)
        self._aid_h_np = self._aid_h.numpy()
        self._nl_h = torch.empty(cap, dtype=torch.long, pin_memory=pm)
        self._nl_h_np = self._nl_h.numpy()
        self._pt_h = torch.empty((cap, K_MAX), dtype=torch.float32, pin_memory=pm)
        self._pt_h_np = self._pt_h.numpy()
        self._vt_h = torch.empty((cap, N), dtype=torch.float32, pin_memory=pm)
        self._vt_h_np = self._vt_h.numpy()

        if pm:
            self._tok_d = torch.empty(
                (cap, nt, td), dtype=torch.float32, device=self.device,
            )
            self._aid_d = torch.empty(
                (cap, K_MAX), dtype=torch.long, device=self.device,
            )
            self._nl_d = torch.empty(cap, dtype=torch.long, device=self.device)
            self._pt_d = torch.empty(
                (cap, K_MAX), dtype=torch.float32, device=self.device,
            )
            self._vt_d = torch.empty((cap, N), dtype=torch.float32, device=self.device)
        else:
            self._tok_d = self._tok_h
            self._aid_d = self._aid_h
            self._nl_d = self._nl_h
            self._pt_d = self._pt_h
            self._vt_d = self._vt_h

        self._scratch_cap = cap

    def _h2d(self, n: int) -> None:
        """Async copy pinned host scratch [:n] into device scratch [:n].

        No-op on CPU (host and device scratch alias the same tensors).
        """
        if self.device.type != "cuda":
            return
        self._tok_d[:n].copy_(self._tok_h[:n], non_blocking=True)
        self._aid_d[:n].copy_(self._aid_h[:n], non_blocking=True)
        self._nl_d[:n].copy_(self._nl_h[:n], non_blocking=True)
        self._pt_d[:n].copy_(self._pt_h[:n], non_blocking=True)
        self._vt_d[:n].copy_(self._vt_h[:n], non_blocking=True)

    def _fill_token_batch(self, n: int) -> None:
        """Fill ``_tok_h_np[:n]`` from ``_states_np[:n]`` via the batched
        Cython entry. One Python dispatch for the whole batch; rows are
        filled in a nogil loop with a shared rebindable scratch state.
        """
        get_token_data_batch(
            [self._states_np[i] for i in range(n)],
            self._num_players,
            self._tok_h_np[:n],
        )

    def train_step(
        self,
        buffer: ReplayBuffer,
        batch_size: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        """Sample a batch from ``buffer`` and execute one training step.

        Samples directly into the trainer's pinned host scratch so the
        subsequent H→D copies are genuinely async (non_blocking=True
        silently degrades to a blocking copy on pageable source memory).
        Grows scratch on first call / when ``batch_size`` increases.

        Returns:
            dict with ``policy_loss``, ``value_loss``, ``total_loss`` as
            floats, plus ``policy_loss_<phase>`` per decision phase
            bucket present in the batch.
        """
        self._ensure_scratch(batch_size)
        B = batch_size

        # Fill pinned host scratch directly from the replay buffer.
        buffer.sample_into(
            B, rng,
            self._states_np[:B],
            self._phase_h_np[:B],
            self._nl_h_np[:B],
            self._aid_h_np[:B],
            self._pt_h_np[:B],
            self._vt_h_np[:B],
        )

        # NaN in any training target is a self-play inference bug —
        # fail loudly before it corrupts gradients.
        for name, arr in (
            ("policy_targets", self._pt_h_np[:B]),
            ("value_targets", self._vt_h_np[:B]),
        ):
            if np.isnan(arr).any():
                raise RuntimeError(
                    f"NaN in sampled '{name}' at step {self._global_step}"
                )

        # Tokenize states into pinned host tokens buffer.
        self._fill_token_batch(B)

        # Async H→D on CUDA; aliased no-op on CPU.
        self._h2d(B)

        tokens = self._tok_d[:B]
        action_ids = self._aid_d[:B]
        n_legals = self._nl_d[:B]
        policy_targets = self._pt_d[:B]
        value_targets = self._vt_d[:B]

        # Build per-phase row indices on host so the model's policy gather
        # uses index_select / index_copy_ instead of boolean masking
        # (which forces one H←D sync per phase × 8 phases per forward).
        # mark_unbacked keeps torch.compile from specializing on per-phase
        # row counts (which would blow the recompile limit).
        phase_view = self._phase_h_np[:B]
        phase_indices: list[torch.Tensor] = []
        phase_counts: list[int] = []
        for _p in range(NUM_PHASES):
            _idx_np = np.nonzero(phase_view == _p)[0].astype(np.int64, copy=False)
            _t = torch.from_numpy(_idx_np)
            if self.device.type == "cuda":
                _t = _t.to(self.device, non_blocking=True)
            mark_unbacked(_t, 0)
            phase_indices.append(_t)
            phase_counts.append(int(_idx_np.shape[0]))

        # --- Forward: sparse (B, K_MAX) logits + (B, N) values ---
        # The model gathers per-row legal slices internally and fills the
        # [n_legal:K_MAX] tail with -1e9, so we can log_softmax directly.
        policy_logits, values = self.model(
            tokens, action_ids, n_legals, phase_indices,
        )

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

        # Per-phase policy loss: computed on device using the same
        # phase_indices as the model's policy gather. Each phase does
        # one index_select + mean (no host sync); empty buckets get a
        # zero placeholder that we filter out on the host side using
        # phase_counts (already known from the numpy nonzero build).
        per_phase = per_example_policy_loss.detach()
        device = per_phase.device
        per_phase_means = torch.zeros(NUM_PHASES, device=device, dtype=per_phase.dtype)
        for p in range(NUM_PHASES):
            if phase_counts[p] > 0:
                per_phase_means[p] = per_phase.index_select(0, phase_indices[p]).mean()

        # Pack every scalar we want to read back into one tensor so the
        # host read is a single H←D sync instead of 3 + 8 separate .item()
        # calls. Order: policy_loss, value_loss, total_loss, *per-phase.
        all_scalars = torch.cat([
            torch.stack([
                policy_loss.detach(), value_loss.detach(), total_loss.detach(),
            ]),
            per_phase_means,
        ])

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

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip,
            )

        self.optimizer.step()
        self.scheduler.step()
        if self._aux_optimizer is not None:
            self._aux_optimizer.step()
        if self._aux_scheduler is not None:
            self._aux_scheduler.step()

        self._global_step += 1

        # Single H←D sync for all scalars.
        scalars = all_scalars.tolist()
        result: dict[str, float] = {
            "policy_loss": scalars[0],
            "value_loss": scalars[1],
            "total_loss": scalars[2],
        }
        for phase_idx, name in enumerate(_PHASE_NAMES):
            if phase_counts[phase_idx] > 0:
                result[f"policy_loss_{name}"] = scalars[3 + phase_idx]

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

    def state_dict(self) -> dict[str, Any]:
        """Return state for checkpointing."""
        state: dict[str, Any] = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self._global_step,
        }
        if self._aux_optimizer is not None:
            state["aux_optimizer"] = self._aux_optimizer.state_dict()
        if self._aux_scheduler is not None:
            state["aux_scheduler"] = self._aux_scheduler.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
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
