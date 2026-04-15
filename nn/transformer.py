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
    d_bilinear: int = 64   # Hidden width for the unified ACQ pair-feature policy head

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
    """Pre-LN transformer block: LN -> MHSA -> residual, LN -> SwiGLU FFN -> residual."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        # We never consume attention weights in this model. Skipping them avoids
        # extra work in the forward pass and makes it easier for PyTorch to use
        # its more efficient attention kernels.
        h, _ = self.attn(h, h, h, need_weights=False)
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
        # path is implemented. For now, we build a shared feature for each
        # (corp, company) pair, then read the 52 offset/FI-buy logits from that
        # pair representation.
        #
        # A lower-rank trilinear alternative would be:
        #   score(c, t, p) = sum_r q_c[r] * k_t[r] * e_p[r]
        # with corp/company projections q_c and k_t plus a learned price
        # embedding e_p. That is smaller, but less expressive than the current
        # pair-feature head.
        dk = cfg.d_bilinear
        self.acquisition_corp_proj = nn.Linear(d, dk)
        self.acquisition_company_proj = nn.Linear(d, dk)
        self.acquisition_pair_head = nn.Sequential(
            nn.Linear(3 * dk, dk),
            nn.GELU(approximate=_GELU_APPROX),
            nn.Linear(dk, 52),  # 51 price offsets + FI buy
        )

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
        n = t.shape[0]
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        auction = self.company_auction_head(t[:, self._company_slice])        # (n, 36, AUCTION_CAP)
        trade = self.corp_trade_head(t[:, self._corp_slice])                  # (n, 8, 2)
        return torch.cat([pass_logit, auction.reshape(n, -1), trade.reshape(n, -1)], dim=-1)

    def _policy_bid(self, t: torch.Tensor) -> torch.Tensor:
        """BID: pass(1) + raises(AUCTION_CAP-1) = AUCTION_CAP. Pass = leave the auction."""
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        raises = self.auction_raise_head(t[:, self._auction_idx])             # (n, 14)
        return torch.cat([pass_logit, raises], dim=-1)

    def _policy_acquisition(self, t: torch.Tensor) -> torch.Tensor:
        """ACQUISITION: pass(1) + corp*company*offset(8*36*52=14976) = 14977."""
        n = t.shape[0]
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        corp_h = self.acquisition_corp_proj(t[:, self._corp_slice])           # (n, 8, dk)
        comp_h = self.acquisition_company_proj(t[:, self._company_slice])     # (n, 36, dk)
        corp_h = corp_h.unsqueeze(2).expand(-1, -1, 36, -1)                   # (n, 8, 36, dk)
        comp_h = comp_h.unsqueeze(1).expand(-1, 8, -1, -1)                    # (n, 8, 36, dk)
        pair_h = torch.cat([corp_h, comp_h, corp_h * comp_h], dim=-1)         # (n, 8, 36, 3*dk)
        acquisition = self.acquisition_pair_head(pair_h)                      # (n, 8, 36, 52)
        return torch.cat([pass_logit, acquisition.reshape(n, -1)], dim=-1)

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
        n = t.shape[0]
        pass_logit = self.pass_head(t[:, self._pass_idx])                     # (n, 1)
        ipo = self.corp_ipo_head(t[:, self._corp_slice])                      # (n, 8, 14)
        return torch.cat([pass_logit, ipo.reshape(n, -1)], dim=-1)

    # ------------------------------------------------------------------
    # Policy dispatch (handles mixed-phase batches)
    # ------------------------------------------------------------------

    def _policy_forward(
        self, tokens: torch.Tensor, phase_ids: torch.Tensor,
        action_ids: torch.Tensor, n_legals: torch.Tensor,
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
            phase_ids: (batch,) int tensor, phase index 0-7.
            action_ids: (batch, K_MAX) int tensor, phase-local legal action ids
                per row. Unused tail positions ([n_legals[i]:]) should be
                in-range (e.g. zero) — they are read by ``gather`` but masked
                out before softmax.
            n_legals: (batch,) int tensor, legal action count per row.

        Returns:
            (batch, K_MAX) float tensor of gathered logits, with positions
            beyond ``n_legals[i]`` filled with ``-1e9`` so they vanish after
            softmax.
        """
        B, K = action_ids.shape
        out = tokens.new_full((B, K), -1e9)

        # Local dispatch tuple: cheap to rebuild (8 bound-method refs), and
        # keeps pyright from routing a ``self._dispatch = (...)`` assignment
        # through ``nn.Module.__setattr__``.
        dispatch = (
            self._policy_invest, self._policy_bid, self._policy_acquisition,
            self._policy_acq_offer, self._policy_closing, self._policy_dividends,
            self._policy_issue, self._policy_ipo,
        )
        for phase_id in range(NUM_PHASES):
            mask = phase_ids == phase_id
            if not mask.any():
                continue
            phase_logits = dispatch[phase_id](tokens[mask])        # (n, phase_width)
            # Gather wants int64; .long() is a no-op if already long.
            sub_ids = action_ids[mask].long()                      # (n, K)
            gathered = phase_logits.gather(1, sub_ids)             # (n, K)
            invalid = self._k_range[None, :] >= n_legals[mask][:, None]
            out[mask] = gathered.masked_fill(invalid, -1e9)

        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, phase_ids: torch.Tensor,
        action_ids: torch.Tensor, n_legals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer.

        Args:
            x: (batch, num_tokens, token_dim) token features, zero-padded.
            phase_ids: (batch,) int tensor, decision phase index 0-7.
            action_ids: (batch, K_MAX) int tensor, phase-local legal action
                ids per row. Unused tail ([n_legals[i]:]) must be in-range.
            n_legals: (batch,) int tensor, legal action count per row.

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

        policy_logits = self._policy_forward(tokens, phase_ids, action_ids, n_legals)
        values = self.value_head(tokens[:, self._player_slice]).squeeze(-1)  # (B, N)
        return policy_logits, values

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming init for linears, zero-init residual outputs for identity start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                nn.init.ones_(module.weight)

        # Pass token: small-random learned anchor (BERT [CLS] convention).
        nn.init.trunc_normal_(self.pass_embed, std=0.02)

        # Zero-init residual outputs so each block starts as identity
        for block in self.blocks:
            assert isinstance(block, TransformerBlock)
            nn.init.zeros_(block.ffn_down.weight)
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)


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
          f"layers={cfg.num_layers}, d_ff={math.ceil(cfg.ff_mult * cfg.d_model)}, d_bilinear={cfg.d_bilinear}")
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
        model.acquisition_corp_proj, model.acquisition_company_proj,
        model.acquisition_pair_head,
        model.corp_ipo_head,
    ]
    policy_params = sum(sum(p.numel() for p in m.parameters()) for m in policy_modules)
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
    phase_ids = torch.arange(NUM_PHASES)
    # Per-phase legal-action synthesis: use the first min(K_MAX, phase_size)
    # ids from each phase. Unused tail is zero-padded; n_legals masks it.
    action_ids = torch.zeros(batch_size, K_MAX, dtype=torch.long)
    n_legals = torch.zeros(batch_size, dtype=torch.long)
    for i in range(NUM_PHASES):
        n = min(K_MAX, PHASE_ACTION_SIZES[i])
        action_ids[i, :n] = torch.arange(n)
        n_legals[i] = n

    policy_logits, values = model(x, phase_ids, action_ids, n_legals)

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
