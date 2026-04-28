"""AlphaZero training step with policy and value losses.

Dense unified-slot contract: batches carry compact int16 game states,
dense ``legal_mask`` + ``policy_target`` rows over the model's unified
logit space, canonical per-player ``value_target``, and a pure-reporting
``phase_id`` that the model never sees. The trainer materializes the
``(batch, num_tokens, token_dim)`` float32 token buffer and
``(batch, num_relations, num_tokens, num_tokens)`` uint8 relation planes at
training time from those compact states, then runs policy cross-entropy
directly against the dense softmax (illegal slots are masked to -1e9 by the
model and carry 0 in the target, so their log-probs contribute nothing to
the loss).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.attention_relations import NUM_ATTENTION_RELATIONS
from core.state import get_layout
from core.token_data import TokenDataSize, get_num_tokens, get_token_data_batch
from nn.transformer import NUM_PHASES, UNIFIED_LOGIT_DIM
from train.config import TrainingConfig
from train.replay_buffer import ReplayBuffer

TOKEN_DIM = int(TokenDataSize.TOKEN_DIM)
U_DIM = int(UNIFIED_LOGIT_DIM)

# Matches core.data.DecisionPhase order. DPHASE_INVEST..DPHASE_PAR occupy
# slots 0..8; DPHASE_ACQ_SELECT_COMPANY / DPHASE_ACQ_SELECT_PRICE are
# appended at slots 9 and 10 after the ACQ split. ACQ_SELECT_CORP at slot 2
# replaces the old joint ACQUISITION. ACQ_OFFER stays a first-class phase.
_PHASE_NAMES = [
    "invest", "bid", "acq_corp", "acq_offer",
    "close", "div", "issue", "ipo", "par",
    "acq_co", "acq_price",
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
        warmup_steps = math.ceil(
            config.warmup_epochs * config.training_steps_per_epoch
        )
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
        """Muon for 2D hidden-layer weights, auxiliary AdamW for other params.

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
                is_embedding = (
                    isinstance(module, torch.nn.Embedding)
                    or pname.endswith("embeds")
                )
                # Muon only supports 2D matrix params. Stacked (3D+) tensors —
                # e.g. the per-offset ACQ bilinear factors — go to AdamW decay:
                # Newton-Schulz orthogonalization across the stack axis isn't
                # what Muon means by "matrix." Embedding/anchor tables are
                # also kept on AdamW per Muon's intended hidden-layer scope.
                if param.ndim == 2 and not is_norm and not is_embedding:
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

        ``phase_ids`` widens to ``torch.long`` for ``index_add_`` on the
        TB bucketing path; it never enters the model graph.
        """
        if n <= self._scratch_cap:
            return
        cap = max(n, max(self._scratch_cap * 2, 1))
        pm = self.device.type == "cuda"
        nt, td = self._num_tokens, TOKEN_DIM
        nr = NUM_ATTENTION_RELATIONS
        N = self._num_players

        # Raw states: CPU-only (consumed by get_token_data), never shipped.
        self._states_np = np.empty((cap, self._state_size), dtype=np.int16)

        # Pinned host (on CUDA). Exposed as numpy for buffer fills.
        self._tok_h = torch.empty((cap, nt, td), dtype=torch.float32, pin_memory=pm)
        self._tok_h_np = self._tok_h.numpy()
        self._rel_h = torch.empty(
            (cap, nr, nt, nt), dtype=torch.uint8, pin_memory=pm,
        )
        self._rel_h_np = self._rel_h.numpy()
        self._phase_h = torch.empty(cap, dtype=torch.long, pin_memory=pm)
        self._phase_h_np = self._phase_h.numpy()
        self._mask_h = torch.empty((cap, U_DIM), dtype=torch.uint8, pin_memory=pm)
        self._mask_h_np = self._mask_h.numpy()
        self._pt_h = torch.empty((cap, U_DIM), dtype=torch.float32, pin_memory=pm)
        self._pt_h_np = self._pt_h.numpy()
        self._vt_h = torch.empty((cap, N), dtype=torch.float32, pin_memory=pm)
        self._vt_h_np = self._vt_h.numpy()

        if pm:
            self._tok_d = torch.empty(
                (cap, nt, td), dtype=torch.float32, device=self.device,
            )
            self._rel_d = torch.empty(
                (cap, nr, nt, nt), dtype=torch.uint8, device=self.device,
            )
            self._phase_d = torch.empty(cap, dtype=torch.long, device=self.device)
            self._mask_d = torch.empty(
                (cap, U_DIM), dtype=torch.uint8, device=self.device,
            )
            self._pt_d = torch.empty(
                (cap, U_DIM), dtype=torch.float32, device=self.device,
            )
            self._vt_d = torch.empty((cap, N), dtype=torch.float32, device=self.device)
        else:
            self._tok_d = self._tok_h
            self._rel_d = self._rel_h
            self._phase_d = self._phase_h
            self._mask_d = self._mask_h
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
        self._rel_d[:n].copy_(self._rel_h[:n], non_blocking=True)
        self._phase_d[:n].copy_(self._phase_h[:n], non_blocking=True)
        self._mask_d[:n].copy_(self._mask_h[:n], non_blocking=True)
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
            self._mask_h_np[:B],
            self._pt_h_np[:B],
            self._vt_h_np[:B],
            relations_out=self._rel_h_np[:B],
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
        relations = self._rel_d[:B]
        phase_ids = self._phase_d[:B]
        legal_masks = self._mask_d[:B].to(torch.bool)
        policy_targets = self._pt_d[:B]
        value_targets = self._vt_d[:B]

        # Per-phase row counts: needed only to filter empty buckets out of
        # the host-side per-phase loss report. Numpy nonzero on the already-
        # filled host slice; no GPU sync.
        phase_view = self._phase_h_np[:B]
        phase_counts: list[int] = [
            int((phase_view == _p).sum()) for _p in range(NUM_PHASES)
        ]

        # --- Forward: dense (B, UNIFIED_LOGIT_DIM) logits + (B, N) values ---
        # The model returns logits with illegal slots already masked to
        # -1e9 via ``legal_masks``, so log_softmax normalizes over the
        # legal set only (illegal log-probs are ~-∞ × 0 = 0 in the loss).
        policy_logits, values = self.model(tokens, legal_masks, relations)

        # Policy loss: dense cross-entropy over the unified slot space.
        # ``policy_targets`` is zero on illegal slots, so only legal slots
        # contribute; masked-to-``-1e9`` illegal log-probs multiplied by
        # zero targets vanish cleanly.
        log_probs = F.log_softmax(policy_logits, dim=-1)
        per_example_policy_loss = -(policy_targets * log_probs).sum(dim=-1)
        policy_loss = per_example_policy_loss.mean()

        # Value loss: mean squared error over the full (B, N) value tensor.
        # This keeps each sampled position at a stable scale instead of
        # magnifying the loss by the player count.
        value_loss = F.mse_loss(values, value_targets)

        # Combined loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        # Per-phase policy loss: scatter-add into a (NUM_PHASES,) bucket
        # tensor using phase_ids as the index. Single fused op replaces
        # the per-phase index_select + mean loop; empty buckets stay zero
        # and are filtered out on the host side using phase_counts.
        per_phase = per_example_policy_loss.detach()
        device = per_phase.device
        per_phase_sums = torch.zeros(NUM_PHASES, device=device, dtype=per_phase.dtype)
        per_phase_sums.index_add_(0, phase_ids, per_phase)
        # ``ones_like`` mirrors per_phase's dtype; counts in fp avoid
        # an int↔float divide guard later.
        per_phase_counts = torch.zeros(NUM_PHASES, device=device, dtype=per_phase.dtype)
        per_phase_counts.index_add_(0, phase_ids, torch.ones_like(per_phase))
        # Empty buckets divide 0 / 1 = 0 — same placeholder behavior the
        # old per-phase mean loop produced, and the host filter drops them.
        per_phase_means = per_phase_sums / per_phase_counts.clamp(min=1)

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
