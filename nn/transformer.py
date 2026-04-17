"""Transformer model for Rolling Stock Stars AlphaZero training.

Token-based architecture: each game entity is a separate input token. Type-specific
linear projections -> L pre-LN transformer blocks -> entity-readout policy heads + value head.

Key differences from the MLP model (nn/template.py):
  - Input: (batch, num_tokens, token_dim) token features, not flat state vector
  - No state rotation: active player marked with is_active flag
  - Entity-readout policy: each entity token produces its own action logits
  - Unified ACQUISITION pair-feature head (corp x company -> 52 logits)
  - Per-phase action indices (max 14,977 for ACQ), not a global action vector
  - Value read from player tokens directly (no un-rotation needed)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.actions import MAX_LEGAL_ACTIONS_PY
from core.data import (
    AUCTION_CAP,
    PHASE_ACTION_SIZES,
)
from core.token_data import TokenDataSize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Decision phases / action sizes all live in ``core.data`` and are imported
# above. This module is strictly a consumer; editing policy head widths or
# adding token types happens over there.

NUM_PHASES = 8
K_MAX = int(MAX_LEGAL_ACTIONS_PY)

_GELU_APPROX = "tanh"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """All dimensions parameterized. Defaults are 3-player with d_model=128."""

    # Core architecture
    num_players: int = 3  # 3-5 supported
    d_model: int = 128
    num_heads: int = 2
    num_layers: int = 10
    ff_mult: float = 3.0  # FFN inner dimension = ceil(ff_mult * d_model)
    acq_rank: int = 4      # Per-offset bilinear rank for the ACQ pair-feature policy head

    # Raw feature width per token (zero-padded to same size across types).
    # Sourced from core.token_data so the model and the Cython extractor
    # can't drift out of sync.
    token_dim: int = int(TokenDataSize.TOKEN_DIM)

    def __post_init__(self) -> None:
        assert 3 <= self.num_players <= 5, f"num_players must be 3-5, got {self.num_players}"

    @property
    def num_tokens(self) -> int:
        """N players + 54 fixed entity tokens."""
        return self.num_players + 54


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN transformer block: LN -> MHSA -> residual, LN -> SwiGLU FFN -> residual.

    Attention is implemented via ``F.scaled_dot_product_attention`` over a
    manually-packed QKV projection rather than ``nn.MultiheadAttention``.
    ``nn.MultiheadAttention`` dispatches to ``aten._native_multi_head_attention``,
    which doesn't support fake-tensor tracing and therefore causes a
    graph break per layer under ``torch.compile``. SDPA traces cleanly
    and Inductor fuses it into a single Triton kernel per block.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model {d_model} must be divisible by num_heads {num_heads}"
        )
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.attn_norm = nn.RMSNorm(d_model)
        # Packed Q/K/V projection. Same parameter count as
        # nn.MultiheadAttention's in_proj_weight/in_proj_bias.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        B, N, D = h.shape
        qkv = self.qkv_proj(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        # (3, B, heads, N, head_dim) so unbind(0) yields three (B, heads, N, head_dim) tensors.
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        # (B, heads, N, head_dim) -> (B, N, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        h = self.out_proj(attn_out)
        x = x + h
        h = self.ffn_norm(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x + h


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RSSTransformerNet(nn.Module):
    """Transformer with entity-readout policy heads and per-player value output."""

    # Declared for pyright: ``register_buffer`` adds dynamic attributes whose
    # type isn't otherwise visible to the checker.
    _k_range: torch.Tensor
    # Cached per-phase head dispatch tuple, indexed by phase_id 0..7.
    _dispatch: tuple[Callable[[torch.Tensor], torch.Tensor], ...]

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        np_ = cfg.num_players

        # --- Token index bookkeeping ---
        self._player_slice = slice(0, np_)
        self._corp_slice = slice(np_, np_ + 8)
        self._company_slice = slice(np_ + 8, np_ + 44)
        self._fi_idx = np_ + 44
        self._market_idx = np_ + 45
        self._global_idx = np_ + 46
        self._invest_idx = np_ + 47
        self._auction_idx = np_ + 48
        self._dividend_idx = np_ + 49
        self._issue_idx = np_ + 50
        self._par_idx = np_ + 51
        self._acq_offer_idx = np_ + 52
        self._pass_idx = np_ + 53

        # --- Type-specific input projections ---
        # All take the full zero-padded token_dim input. Weights for always-zero
        # feature positions are inert; the simplicity is worth the ~2% extra params.
        tdim = cfg.token_dim
        self.player_proj = nn.Linear(tdim, d)
        self.corp_proj = nn.Linear(tdim, d)
        self.company_proj = nn.Linear(tdim, d)
        self.fi_proj = nn.Linear(tdim, d)
        self.market_proj = nn.Linear(tdim, d)
        self.global_proj = nn.Linear(tdim, d)
        self.invest_proj = nn.Linear(tdim, d)
        self.auction_proj = nn.Linear(tdim, d)
        self.dividend_proj = nn.Linear(tdim, d)
        self.issue_proj = nn.Linear(tdim, d)
        self.par_proj = nn.Linear(tdim, d)
        self.acq_offer_proj = nn.Linear(tdim, d)
        # Pass token: no input features. Its representation is a single learned
        # vector (BERT [CLS]-style) that rides the residual stream and picks up
        # game-state context through attention. All other token types are
        # discriminated via their own per-type projection (and its bias), so a
        # shared type-embedding table would be redundant there.
        self.pass_embed = nn.Parameter(torch.empty(d))

        # --- Transformer trunk ---
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.num_heads, math.ceil(cfg.ff_mult * cfg.d_model))
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.RMSNorm(d)

        # --- Entity-readout policy heads ---
        # Shared per-entity-type heads (weight-shared across all tokens of same type)
        self.pass_head = nn.Linear(d, 1)
        self.company_auction_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, int(AUCTION_CAP)),  # AUCTION_CAP price offsets per company
        )
        self.corp_trade_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 2),  # buy, sell
        )
        self.company_close_head = nn.Linear(d, 1)  # per-company close logit

        # Phase-specific context token heads
        self.auction_raise_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, int(AUCTION_CAP) - 1),  # AUCTION_CAP-1 raise amounts
        )
        self.dividend_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 26),  # 26 dividend levels
        )
        self.issue_head = nn.Linear(d, 1)  # issue logit (pass from pass_head)
        # ACQ is still the least-settled policy design in this prototype. The
        # model now scores the full corp/company/offset action space in one
        # shot, but the remaining dense interface is still provisional pending
        # the sparse candidate path outlined in
        # /home/icebreaker/rss-az-cython2/sparse-refactor.md.
        self.acq_offer_head = nn.Linear(d, 1)  # buy logit (pass from pass_head)

        # ACQUISITION is also likely to change once the sparse candidate-scoring
        # path is implemented. For now we score the full (corp, company, offset)
        # space with a per-offset low-rank bilinear form:
        #
        #   score(c, t, p) = (corp_h[c] @ U[p]) . (comp_h[t] @ V[p])       (bilinear)
        #                  + W_corp[p] . corp_h[c]                        (corp-only)
        #                  + W_comp[p] . comp_h[t]                        (company-only)
        #
        # Each of the 52 offsets gets its own rank-r interaction pattern (U_p, V_p),
        # so offset-specific attention over corp/company features is learnable
        # instead of being crammed through a single shared GELU bottleneck. The
        # additive unary paths keep the score well-defined when either side's
        # representation is weak (pure-multiplicative scores collapse there).
        r = cfg.acq_rank
        self.acquisition_U = nn.Parameter(torch.empty(52, d, r))
        self.acquisition_V = nn.Parameter(torch.empty(52, d, r))
        self.acquisition_corp_bias = nn.Linear(d, 52)
        self.acquisition_company_bias = nn.Linear(d, 52)

        # IPO now reads par-price logits directly from each corp token. This is
        # a cleaner entity-readout signal than routing the decision through the
        # PAR token projection.
        self.corp_ipo_head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 14),  # 14 par prices per corp
        )

        # --- Value head (applied per player token) ---
        self.value_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(d // 2, 1),
            nn.Tanh(),
        )

        # Index range used by per-phase output masking. Registered as a
        # non-persistent buffer so it rides the module's device and shows
        # up in `.to(...)`, but is not saved to checkpoints.
        self.register_buffer(
            "_k_range", torch.arange(K_MAX, dtype=torch.long), persistent=False,
        )

        # Per-phase head dispatch, indexed by phase_id 0..7. Built once;
        # the bound-method refs stay valid for the module's lifetime.
        self._dispatch = (
            self._policy_invest, self._policy_bid, self._policy_acquisition,
            self._policy_acq_offer, self._policy_closing, self._policy_dividends,
            self._policy_issue, self._policy_ipo,
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Input projection
    # ------------------------------------------------------------------

    def _project_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw token features to d_model via type-specific projections.

        Args:
            x: (batch, num_tokens, token_dim) zero-padded raw features.
        Returns:
            (batch, num_tokens, d_model) embeddings ready for transformer trunk.
        """
        B = x.shape[0]

        # All projections take the full zero-padded token_dim input.
        parts: list[torch.Tensor] = [
            self.player_proj(x[:, self._player_slice]),               # (B, N, d)
            self.corp_proj(x[:, self._corp_slice]),                   # (B, 8, d)
            self.company_proj(x[:, self._company_slice]),             # (B, 36, d)
            self.fi_proj(x[:, self._fi_idx]).unsqueeze(1),            # (B, 1, d)
            self.market_proj(x[:, self._market_idx]).unsqueeze(1),
            self.global_proj(x[:, self._global_idx]).unsqueeze(1),
            self.invest_proj(x[:, self._invest_idx]).unsqueeze(1),
            self.auction_proj(x[:, self._auction_idx]).unsqueeze(1),
            self.dividend_proj(x[:, self._dividend_idx]).unsqueeze(1),
            self.issue_proj(x[:, self._issue_idx]).unsqueeze(1),
            self.par_proj(x[:, self._par_idx]).unsqueeze(1),
            self.acq_offer_proj(x[:, self._acq_offer_idx]).unsqueeze(1),
            self.pass_embed.view(1, 1, -1).expand(B, 1, -1),  # Pass: learned anchor
        ]
        tokens = torch.cat(parts, dim=1)                      # (B, num_tokens, d)
        return tokens

    # ------------------------------------------------------------------
    # Phase-specific policy heads
    # ------------------------------------------------------------------

    def _policy_invest(self, t: torch.Tensor) -> torch.Tensor:
        """INVEST: pass(1) + auction(36*AUCTION_CAP) + trade(8*2) = 557."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        auction = self.company_auction_head(t[:, self._company_slice])        # (n, 36, AUCTION_CAP)
        trade = self.corp_trade_head(t[:, self._corp_slice])                  # (n, 8, 2)
        # flatten(1) instead of reshape(n, -1): the latter is ambiguous when
        # n == 0 (empty mask in dispatch), since 0 elements / 0 rows is
        # undefined for the inferred dim.
        return torch.cat([pass_logit, auction.flatten(1), trade.flatten(1)], dim=-1)

    def _policy_bid(self, t: torch.Tensor) -> torch.Tensor:
        """BID: pass(1) + raises(AUCTION_CAP-1) = AUCTION_CAP. Pass = leave the auction."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        raises = self.auction_raise_head(t[:, self._auction_idx])             # (n, 14)
        return torch.cat([pass_logit, raises], dim=-1)

    def _policy_acquisition(self, t: torch.Tensor) -> torch.Tensor:
        """ACQUISITION: pass(1) + corp*company*offset(8*36*52=14976) = 14977."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        corp_h = t[:, self._corp_slice]                                       # (n, 8, d)
        comp_h = t[:, self._company_slice]                                    # (n, 36, d)
        # Per-offset low-rank bilinear score: factor through a rank-r bottleneck
        # instead of materializing the dense (n, 8, 36, 3*dk) pair cat. Peak
        # transient drops from ~3*dk wide down to two (n, 8|36, 52, r) projection
        # buffers and the (n, 8, 36, 52) output.
        corp_proj = torch.einsum('ncd,pdr->ncpr', corp_h, self.acquisition_U) # (n, 8, 52, r)
        comp_proj = torch.einsum('ntd,pdr->ntpr', comp_h, self.acquisition_V) # (n, 36, 52, r)
        bilin = torch.einsum('ncpr,ntpr->nctp', corp_proj, comp_proj)         # (n, 8, 36, 52)
        corp_bias = self.acquisition_corp_bias(corp_h)                        # (n, 8, 52)
        comp_bias = self.acquisition_company_bias(comp_h)                     # (n, 36, 52)
        acquisition = bilin + corp_bias.unsqueeze(2) + comp_bias.unsqueeze(1) # (n, 8, 36, 52)
        # flatten(1): reshape(n, -1) is ambiguous for empty (n=0) tensors.
        return torch.cat([pass_logit, acquisition.flatten(1)], dim=-1)

    def _policy_acq_offer(self, t: torch.Tensor) -> torch.Tensor:
        """ACQ_OFFER: pass(1) + buy(1) = 2."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        buy = self.acq_offer_head(t[:, self._acq_offer_idx])                  # (n, 1)
        return torch.cat([pass_logit, buy], dim=-1)

    def _policy_closing(self, t: torch.Tensor) -> torch.Tensor:
        """CLOSING: pass(1) + company_close(36) = 37."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        close = self.company_close_head(t[:, self._company_slice])            # (n, 36, 1)
        return torch.cat([pass_logit, close.squeeze(-1)], dim=-1)

    def _policy_dividends(self, t: torch.Tensor) -> torch.Tensor:
        """DIVIDENDS: 26 amounts."""
        return self.dividend_head(t[:, self._dividend_idx])                   # (n, 26)

    def _policy_issue(self, t: torch.Tensor) -> torch.Tensor:
        """ISSUE: pass(1) + issue(1) = 2."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        issue = self.issue_head(t[:, self._issue_idx])                        # (n, 1)
        return torch.cat([pass_logit, issue], dim=-1)

    def _policy_ipo(self, t: torch.Tensor) -> torch.Tensor:
        """IPO: pass(1) + per-corp par_price logits(8*14=112) = 113."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        ipo = self.corp_ipo_head(t[:, self._corp_slice])                      # (n, 8, 14)
        # flatten(1): reshape(n, -1) is ambiguous for empty (n=0) tensors.
        return torch.cat([pass_logit, ipo.flatten(1)], dim=-1)

    # ------------------------------------------------------------------
    # Policy dispatch (handles mixed-phase batches)
    # ------------------------------------------------------------------

    def _policy_forward(
        self, tokens: torch.Tensor,
        action_ids: torch.Tensor, n_legals: torch.Tensor,
        phase_indices: list[torch.Tensor],
    ) -> torch.Tensor:
        """Route each batch element to its phase-specific policy head and
        gather the per-row legal slice in one shot.

        The per-phase heads emit logits of shape ``(n_phase, phase_action_size)``
        — much narrower than the full ``MAX_ACTION_SIZE = 14977`` pad. Gathering
        against ``action_ids`` inside the model keeps the source tensor at
        phase-local width, so the evaluator / trainer / eval-server never have
        to allocate or scatter into the full ACQUISITION-wide scratch.

        Args:
            tokens: (batch, num_tokens, d_model) final token representations.
            action_ids: (batch, K_MAX) int tensor, phase-local legal action ids
                per row. Tail positions ([n_legals[i]:]) may hold arbitrary
                values — they are rewritten to 0 before ``gather`` and
                masked out before softmax.
            n_legals: (batch,) int tensor, legal action count per row.
            phase_indices: List of ``NUM_PHASES`` int64 tensors on
                ``tokens.device``. ``phase_indices[p]`` holds the row indices
                whose ``phase_id == p``. Dispatch uses ``index_select`` /
                ``index_copy_`` so the policy gather has no H←D sync — the
                per-iteration sync that boolean indexing would force (it has
                to read the mask's true-count to size the masked tensor)
                dominated eval latency on CPU-bound workloads. All callers
                build these on CPU from the pinned phase_ids buffer before the
                H→D copy and pass them in.

        Returns:
            (batch, K_MAX) float tensor of gathered logits, with positions
            beyond ``n_legals[i]`` filled with ``-1e9`` so they vanish after
            softmax.
        """
        B, K = action_ids.shape
        # Explicit fp32 rather than tokens.new_full: sentinels need no
        # precision, and a fixed dtype avoids a silent coupling to whatever
        # autocast leaves `tokens` as post-final_norm.
        out = torch.full((B, K), -1e9, dtype=torch.float32, device=tokens.device)

        # No `if not mask.any(): continue` early-exit: that path forces a
        # GPU→CPU sync (8 per forward) which dominates eval latency on
        # CPU-bound workloads like analyze_game. Empty masks/indices flow
        # cleanly through linear / cat / gather / masked_scatter as no-op
        # (0,*) ops, so we just dispatch all 8 phases unconditionally and
        # let the GPU pipeline absorb the extra small launches.
        # `action_ids` tail positions [n_legals[i]:] may hold stale
        # garbage (shared-memory worker slots are reused across phases
        # and only [:n_legals[i]] is written per enumeration). We clamp
        # sub_ids to the phase head width before gather so the compiled
        # kernel never indexes out of range into narrow phase heads
        # (ACQ_OFFER / ISSUE have width 2). Clamp rather than
        # ``masked_fill(invalid, 0)``: Inductor will reorder a mask
        # whose effect is "overwritten later by -1e9" past the gather,
        # since the reasoning is that the gathered value at invalid
        # positions doesn't matter. That reordering is unsafe on CUDA
        # where gather bounds-checks and device-asserts regardless of
        # what happens to the result. An unconditional clamp has no
        # such dependence and survives fusion. Output is identical:
        # invalid-position values get masked to -1e9 after gather
        # either way.
        for phase_id in range(NUM_PHASES):
            idx = phase_indices[phase_id]
            phase_logits = self._dispatch[phase_id](tokens.index_select(0, idx))
            sub_ids = action_ids.index_select(0, idx).long()
            safe_ids = sub_ids.clamp(min=0, max=phase_logits.shape[1] - 1)
            gathered = phase_logits.gather(1, safe_ids)
            invalid = self._k_range[None, :] >= n_legals.index_select(0, idx)[:, None]
            # Cast to out.dtype: under autocast, tokens (post-RMSNorm) is
            # fp32 but phase_logits (from Linear) is bf16 — index_copy_
            # requires matching dtypes. torch.compile hides the mismatch
            # via fusion; eager (NNEvaluator) surfaces it.
            out.index_copy_(0, idx, gathered.masked_fill(invalid, -1e9).to(out.dtype))

        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor,
        action_ids: torch.Tensor, n_legals: torch.Tensor,
        phase_indices: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer.

        Args:
            x: (batch, num_tokens, token_dim) token features, zero-padded.
            action_ids: (batch, K_MAX) int tensor, phase-local legal action
                ids per row. Tail ([n_legals[i]:]) may be arbitrary — sanitized
                inside ``_policy_forward`` before gather.
            n_legals: (batch,) int tensor, legal action count per row.
            phase_indices: ``NUM_PHASES``-length list of int64 row indices per
                phase; see ``_policy_forward`` for details. Required — the
                policy gather uses ``index_select`` / ``index_copy_`` to avoid
                the per-batch H←D sync that boolean indexing would force.

        Returns:
            policy_logits: (batch, K_MAX) gathered logits over the per-row
                legal-action slice. Positions beyond ``n_legals[i]`` are
                filled with ``-1e9``.
            values: (batch, num_players) per-player expected outcomes in [-1, 1].
        """
        tokens = self._project_tokens(x)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        policy_logits = self._policy_forward(
            tokens, action_ids, n_legals, phase_indices,
        )
        values = self.value_head(tokens[:, self._player_slice]).squeeze(-1)  # (B, N)
        return policy_logits, values

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT/LLaMA-style trunc-normal init, zero-init residual outputs for identity start.

        kaiming_uniform_(nonlinearity="relu") is wrong for most Linears here
        (SDPA has no ReLU, SwiGLU/GELU heads aren't ReLU, value head feeds
        Tanh) and produces bounds ~10x wider than standard transformer init.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        # Pass token: small-random learned anchor (BERT [CLS] convention).
        nn.init.trunc_normal_(self.pass_embed, std=0.02)

        # ACQ bilinear factors: same std as Linear weights. `_init_weights`
        # sweeps `nn.Linear` modules only, so raw parameters need explicit init.
        nn.init.trunc_normal_(self.acquisition_U, std=0.02)
        nn.init.trunc_normal_(self.acquisition_V, std=0.02)

        # Zero-init residual outputs so each block starts as identity
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            nn.init.zeros_(block.ffn_down.weight)
            nn.init.zeros_(block.out_proj.weight)
            nn.init.zeros_(block.out_proj.bias)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = TransformerConfig()
    model = RSSTransformerNet(cfg)
    total = count_parameters(model)

    print(f"Transformer model: {cfg.num_players}p")
    print(f"  d_model={cfg.d_model}, heads={cfg.num_heads}, "
          f"layers={cfg.num_layers}, d_ff={math.ceil(cfg.ff_mult * cfg.d_model)}, acq_rank={cfg.acq_rank}")
    print(f"  tokens={cfg.num_tokens}, token_dim={cfg.token_dim}")
    print(f"  Trainable parameters: {total:,}")
    print()

    # --- Parameter breakdown ---
    proj_modules = [
        model.player_proj, model.corp_proj, model.company_proj,
        model.fi_proj, model.market_proj, model.global_proj,
        model.invest_proj, model.auction_proj, model.dividend_proj,
        model.issue_proj, model.par_proj, model.acq_offer_proj,
    ]
    proj_params = sum(sum(p.numel() for p in m.parameters()) for m in proj_modules)
    pass_params = model.pass_embed.numel()
    trunk_params = (
        sum(p.numel() for p in model.blocks.parameters())
        + sum(p.numel() for p in model.final_norm.parameters())
    )
    policy_modules = [
        model.pass_head, model.company_auction_head, model.corp_trade_head,
        model.company_close_head, model.auction_raise_head, model.dividend_head,
        model.issue_head, model.acq_offer_head,
        model.acquisition_corp_bias, model.acquisition_company_bias,
        model.corp_ipo_head,
    ]
    policy_params = sum(sum(p.numel() for p in m.parameters()) for m in policy_modules)
    policy_params += model.acquisition_U.numel() + model.acquisition_V.numel()
    value_params = sum(p.numel() for p in model.value_head.parameters())

    print("Parameter breakdown:")
    for name, count in [
        ("Input projections", proj_params),
        ("Pass token", pass_params),
        ("Transformer trunk", trunk_params),
        ("Policy heads", policy_params),
        ("Value head", value_params),
    ]:
        print(f"  {name + ':':22s} {count:>10,}  ({count / total * 100:.1f}%)")

    phase_names = [
        "INVEST", "BID", "ACQUISITION", "ACQ_OFFER",
        "CLOSING", "DIVIDENDS", "ISSUE", "IPO",
    ]
    print()
    for name, size in zip(phase_names, PHASE_ACTION_SIZES):
        print(f"  {name:>12s}: {size:>3d} actions")

    # --- Smoke test ---
    print()
    batch_size = NUM_PHASES  # one sample per phase
    x = torch.randn(batch_size, cfg.num_tokens, cfg.token_dim)
    # Per-phase legal-action synthesis: use the first min(K_MAX, phase_size)
    # ids from each phase. Unused tail is zero-padded; n_legals masks it.
    action_ids = torch.zeros(batch_size, K_MAX, dtype=torch.long)
    n_legals = torch.zeros(batch_size, dtype=torch.long)
    for i in range(NUM_PHASES):
        n = min(K_MAX, PHASE_ACTION_SIZES[i])
        action_ids[i, :n] = torch.arange(n)
        n_legals[i] = n
    phase_indices = [torch.tensor([i], dtype=torch.int64) for i in range(NUM_PHASES)]

    policy_logits, values = model(x, action_ids, n_legals, phase_indices)

    print(f"policy_logits: {tuple(policy_logits.shape)}")
    print(f"values:        {tuple(values.shape)}")

    assert policy_logits.shape == (batch_size, K_MAX)
    assert values.shape == (batch_size, cfg.num_players)

    assert values.min() >= -1.0 and values.max() <= 1.0, "tanh output out of range"
    print("values in [-1, 1]: ok")

    for i in range(NUM_PHASES):
        n = int(n_legals[i])
        active = policy_logits[i, :n]
        inactive = policy_logits[i, n:]
        assert torch.isfinite(active).all(), f"{phase_names[i]}: non-finite logits"
        if inactive.numel() > 0:
            assert (inactive == -1e9).all(), f"{phase_names[i]}: leak beyond legal slice"
    print("per-row legal slice: ok")

    print("\nSmoke test passed.")
