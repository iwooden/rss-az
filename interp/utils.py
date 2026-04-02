"""Data collection and model loading utilities for interpretability experiments.

Common workflow:
    from interp.utils import load_model, collect_states, InterpDataset

    model, config, device, epoch = load_model()
    dataset = collect_states(model, config, device, num_games=50)
    dataset.save("interp/data/states.npz")

    # Later:
    dataset = InterpDataset.load("interp/data/states.npz")
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.data import GameConstants, GamePhases
from core.driver import DRIVER, STATUS_GAME_OVER_PY

# Decision phase ID → short display name (8 phases with player actions).
# Keyed by GamePhases constants so the mapping stays valid if enum values change.
# Auto phases (WRAP_UP, INCOME, END_CARD) and GAME_OVER are excluded — they
# never appear in collected interp data or the 8-wide visible phase one-hot.
PHASE_NAMES: dict[int, str] = {
    GamePhases.PHASE_INVEST: "INVEST",
    GamePhases.PHASE_BID_IN_AUCTION: "BID",
    GamePhases.PHASE_ACQUISITION: "ACQ",
    GamePhases.PHASE_CLOSING: "CLOSE",
    GamePhases.PHASE_DIVIDENDS: "DIV",
    GamePhases.PHASE_ISSUE_SHARES: "ISSUE",
    GamePhases.PHASE_IPO: "IPO",
    GamePhases.PHASE_PAR: "PAR",
}

# Ordered list of decision phase display names, matching model.phase_heads index order
# (sorted by raw phase ID: INVEST=0, BID=1, ACQ=3, CLOSE=4, DIV=6, ISSUE=8, IPO=9, PAR=10)
DECISION_PHASE_ORDER: list[str] = [PHASE_NAMES[pid] for pid in sorted(PHASE_NAMES)]
DECISION_PHASES: set[str] = set(DECISION_PHASE_ORDER)

from core.state import (
    GameState,
    get_corp_fields,
    get_layout,
    get_player_fields,
    get_turn_fields,
)
from entities.turn import TURN
from mcts.evaluator import NNEvaluator, rotate_visible_state
from nn import create_model
from train.checkpoint import find_latest_checkpoint, load_checkpoint
from train.config import TrainingConfig


def batch_masked_softmax(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Apply legal action mask and softmax to batched logits via torch."""
    t = torch.from_numpy(logits)
    t = t.masked_fill(torch.from_numpy(masks) <= 0, -1e9)
    return torch.softmax(t, dim=-1).numpy()


def kl_divergence_batch(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-sample KL(P || Q), shape (N,)."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=-1)


def forward_batched(
    model: torch.nn.Module,
    device: torch.device,
    states: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model forward on states in batches. Returns (logits, values)."""
    logits_list: list[np.ndarray] = []
    values_list: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for i in range(0, states.shape[0], batch_size):
            j = min(i + batch_size, states.shape[0])
            x = torch.from_numpy(states[i:j]).to(device)
            lo, va = model(x)
            logits_list.append(lo.float().cpu().numpy())
            values_list.append(va.float().cpu().numpy())

    return np.concatenate(logits_list), np.concatenate(values_list)


@dataclass
class InterpDataset:
    """Collected game states for interpretability analysis.

    States are pre-rotated so the active player is at slot 0, exactly
    as the NN sees them during training/inference.
    """

    states: np.ndarray  # (N, visible_size) float32
    legal_masks: np.ndarray  # (N, action_dim) float32
    phases: np.ndarray  # (N,) int32
    active_players: np.ndarray  # (N,) int32
    turn_numbers: np.ndarray  # (N,) int32  — game turn (1-based)
    game_indices: np.ndarray  # (N,) int32  — which game (0-based)
    num_games: int
    checkpoint_path: str
    seed: int

    @property
    def num_states(self) -> int:
        return self.states.shape[0]

    def save(self, path: str | Path) -> None:
        """Save dataset to compressed .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states,
            legal_masks=self.legal_masks,
            phases=self.phases,
            active_players=self.active_players,
            turn_numbers=self.turn_numbers,
            game_indices=self.game_indices,
            meta=np.array([self.num_games, self.seed]),
            checkpoint_path=np.array(self.checkpoint_path),
        )
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Saved {self.num_states} states to {path} ({size_mb:.1f} MB)")

    @staticmethod
    def load(path: str | Path) -> InterpDataset:
        """Load dataset from .npz file."""
        data = np.load(path, allow_pickle=True)
        meta = data["meta"]
        n = len(data["phases"])
        # Backward compat: old files lack turn_numbers/game_indices
        turn_numbers = (
            data["turn_numbers"] if "turn_numbers" in data
            else np.zeros(n, dtype=np.int32)
        )
        game_indices = (
            data["game_indices"] if "game_indices" in data
            else np.zeros(n, dtype=np.int32)
        )
        return InterpDataset(
            states=data["states"],
            legal_masks=data["legal_masks"],
            phases=data["phases"],
            active_players=data["active_players"],
            turn_numbers=turn_numbers,
            game_indices=game_indices,
            num_games=int(meta[0]),
            checkpoint_path=str(data["checkpoint_path"]),
            seed=int(meta[1]),
        )


def load_model(
    checkpoint_path: str | Path | None = None,
    checkpoint_dir: str = "checkpoints",
    device: str | None = None,
) -> tuple[torch.nn.Module, TrainingConfig, torch.device, int]:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to .pt file, or None to use latest.
        checkpoint_dir: Directory to search when checkpoint_path is None.
        device: "cuda", "cpu", or None for auto-detect.

    Returns:
        (model, config, device, epoch)
    """
    dev = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if checkpoint_path is None:
        cp_path = find_latest_checkpoint(Path(checkpoint_dir))
        if cp_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    else:
        cp_path = Path(checkpoint_path)

    print(f"Loading checkpoint: {cp_path}")
    cp = load_checkpoint(cp_path, dev)
    config = TrainingConfig.from_json(cp["config_json"])  # type: ignore[arg-type]

    model = create_model(
        config.model_path,
        input_dim=config.visible_size,
        action_dim=config.action_dim,
        value_dim=config.num_players,
    ).to(dev)
    model.load_state_dict(cp["model_state_dict"])  # type: ignore[arg-type]
    model.eval()

    epoch = int(cp.get("epoch", -1))  # type: ignore[arg-type]
    print(f"Epoch {epoch}, device={dev}")
    return model, config, dev, epoch


def collect_states(
    model: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    num_games: int = 50,
    seed: int = 0,
    checkpoint_path: str = "",
    max_moves_per_game: int = 1000,
) -> InterpDataset:
    """Play fast games via policy sampling to collect diverse game states.

    No MCTS search — just a single NN forward pass per decision point,
    then sample from the policy. Much faster than full self-play.

    Args:
        model: Trained model (eval mode).
        config: Training config from checkpoint.
        device: Torch device.
        num_games: Number of games to play.
        seed: Base seed (game i uses seed + i).
        checkpoint_path: Recorded in metadata for provenance.
        max_moves_per_game: Safety cap to avoid infinite loops.

    Returns:
        InterpDataset with all decision points across all games.
    """
    evaluator = NNEvaluator(model, device, num_players=config.num_players)
    rng = np.random.default_rng(seed)

    all_states: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_phases: list[int] = []
    all_active: list[int] = []
    all_turns: list[int] = []
    all_game_idx: list[int] = []

    t0 = time.perf_counter()
    for game_idx in range(num_games):
        game_seed = seed + game_idx
        state = GameState(config.num_players)
        state.initialize_game(seed=game_seed)
        moves = 0

        while state.get_phase() != GamePhases.PHASE_GAME_OVER:
            active_player = state.get_active_player()
            phase = state.get_phase()

            policy_probs, _, legal_mask = evaluator.evaluate(state)

            rotated = rotate_visible_state(
                state._array, active_player, config.num_players
            )
            all_states.append(rotated)
            all_masks.append(legal_mask)
            all_phases.append(phase)
            all_active.append(active_player)
            all_turns.append(TURN.get_turn_number(state))
            all_game_idx.append(game_idx)

            action = int(rng.choice(config.action_dim, p=policy_probs))
            status = DRIVER.apply_action(state, action)
            moves += 1

            if status == STATUS_GAME_OVER_PY or moves >= max_moves_per_game:
                break

        if (game_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  Game {game_idx + 1}/{num_games} "
                f"({elapsed:.1f}s, {len(all_states)} states)"
            )

    elapsed = time.perf_counter() - t0
    print(f"Collected {len(all_states)} states from {num_games} games in {elapsed:.1f}s")

    return InterpDataset(
        states=np.array(all_states, dtype=np.float32),
        legal_masks=np.array(all_masks, dtype=np.float32),
        phases=np.array(all_phases, dtype=np.int32),
        active_players=np.array(all_active, dtype=np.int32),
        turn_numbers=np.array(all_turns, dtype=np.int32),
        game_indices=np.array(all_game_idx, dtype=np.int32),
        num_games=num_games,
        checkpoint_path=checkpoint_path,
        seed=seed,
    )


def build_feature_groups(num_players: int) -> list[tuple[str, np.ndarray]]:
    """Build (name, indices) for every named field in the visible state.

    Uses get_player_fields/get_corp_fields/get_turn_fields for sub-offsets
    so this stays in sync with the layout defined in core/state.pyx.
    """
    layout = get_layout(num_players)
    pf = get_player_fields(num_players)
    cf = get_corp_fields()
    tf = get_turn_fields(num_players)
    NC = GameConstants.NUM_COMPANIES
    NK = GameConstants.NUM_CORPS
    MAX_DIV = GameConstants.MAX_DIVIDEND
    N_PAR = GameConstants.NUM_PAR_PRICES

    groups: list[tuple[str, np.ndarray]] = []

    # --- Phase & CoO ---
    groups.append(("phase", np.arange(layout.phase_offset, layout.phase_offset + layout.phase_size)))
    groups.append(("coo_level", np.arange(layout.coo_offset, layout.coo_offset + layout.coo_size)))

    # --- Player fields (aggregated across all players) ---
    _player_groups = [
        ("player:cash", pf.cash, 1),
        ("player:net_worth", pf.net_worth, 1),
        ("player:liquidity", pf.liquidity, 1),
        ("player:turn_order", pf.turn_order, num_players),
        ("player:owned_companies", pf.owned_companies, NC),
        ("player:owned_shares", pf.owned_shares, NK),
        ("player:is_president", pf.is_president, NK),
        ("player:round_trips", pf.round_trips, 1),
        ("player:income", pf.income, 1),
    ]
    for name, rel, size in _player_groups:
        idx: list[int] = []
        for p in range(num_players):
            base = layout.players_offset + p * layout.player_stride + rel
            idx.extend(range(base, base + size))
        groups.append((name, np.array(idx)))

    # --- Foreign Investor ---
    groups.append(("fi:cash", np.array([layout.fi_offset])))
    groups.append(("fi:income", np.array([layout.fi_offset + 1])))
    groups.append(("fi:companies", np.arange(layout.fi_offset + 2, layout.fi_offset + layout.fi_size)))

    # --- Company location flags ---
    groups.append(("co:for_auction", np.arange(layout.auction_companies_offset, layout.auction_companies_offset + NC)))
    groups.append(("co:revealed", np.arange(layout.revealed_companies_offset, layout.revealed_companies_offset + NC)))
    groups.append(("co:removed", np.arange(layout.removed_companies_offset, layout.removed_companies_offset + NC)))
    groups.append(("co:acquired", np.arange(layout.acquired_companies_offset, layout.acquired_companies_offset + NC)))

    # --- Company adjusted incomes ---
    groups.append(("co:adj_incomes", np.arange(layout.company_incomes_offset, layout.company_incomes_offset + layout.company_incomes_size)))

    # --- Market availability ---
    groups.append(("market:available", np.arange(layout.market_offset, layout.market_offset + layout.market_size)))

    # --- Corporation fields (aggregated across all corps) ---
    _corp_groups = [
        ("corp:active", cf.active, 1),
        ("corp:cash", cf.cash, 1),
        ("corp:unissued_shares", cf.unissued_shares, 1),
        ("corp:issued_shares", cf.issued_shares, 1),
        ("corp:bank_shares", cf.bank_shares, 1),
        ("corp:income", cf.income, 1),
        ("corp:stars", cf.stars, 1),
        ("corp:share_price", cf.share_price, 1),
        ("corp:acq_proceeds", cf.acquisition_proceeds, 1),
        ("corp:in_receivership", cf.in_receivership, 1),
        ("corp:price_index_norm", cf.price_index_norm, 1),
        ("corp:pending_price_move", cf.pending_price_move, 1),
        ("corp:raw_revenue", cf.raw_revenue, 1),
        ("corp:synergy_income", cf.synergy_income, 1),
        ("corp:coo_cost", cf.coo_cost, 1),
        ("corp:ability_income", cf.ability_income, 1),
        ("corp:owned_companies", cf.owned_companies, NC),
    ]
    for name, rel, size in _corp_groups:
        idx_list: list[int] = []
        for c in range(NK):
            base = layout.corps_offset + c * layout.corp_stride + rel
            idx_list.extend(range(base, base + size))
        groups.append((name, np.array(idx_list)))

    # --- Turn state ---
    t = layout.turn_offset
    groups.append(("turn:end_card_flipped", np.array([t + tf.end_card_flipped])))
    groups.append(("turn:consec_passes", np.array([t + tf.consecutive_passes])))

    groups.append(("turn:auction_price", np.array([t + tf.auction_price])))
    groups.append(("turn:auction_price_offset", np.array([t + tf.auction_price_offset])))
    groups.append(("turn:auction_high_bidder", np.arange(t + tf.auction_high_bidder, t + tf.auction_high_bidder + num_players)))
    groups.append(("turn:auction_starter", np.arange(t + tf.auction_starter, t + tf.auction_starter + num_players)))
    groups.append(("turn:auction_passed", np.arange(t + tf.auction_passed, t + tf.auction_passed + num_players)))

    groups.append(("turn:dividend_impact", np.arange(t + tf.dividend_impact, t + tf.dividend_impact + MAX_DIV)))
    groups.append(("turn:dividend_remaining", np.arange(t + tf.dividend_remaining, t + tf.dividend_remaining + NK)))

    groups.append(("turn:issue_remaining", np.arange(t + tf.issue_remaining, t + tf.issue_remaining + NK)))
    groups.append(("turn:issue_price_impact", np.array([t + tf.issue_price_impact])))
    groups.append(("turn:issue_cash_gain", np.array([t + tf.issue_cash_gain])))

    groups.append(("turn:acq_is_fi_offer", np.array([t + tf.acq_is_fi_offer])))
    groups.append(("turn:acq_synergy", np.arange(t + tf.acq_synergy_values, t + tf.acq_synergy_values + NC)))

    groups.append(("turn:active_company", np.arange(t + tf.active_company, t + tf.active_company + NC)))
    groups.append(("turn:active_company_stars", np.array([t + tf.active_company_stars])))
    groups.append(("turn:active_company_low_price", np.array([t + tf.active_company_low_price])))
    groups.append(("turn:active_company_face_value", np.array([t + tf.active_company_face_value])))
    groups.append(("turn:active_company_high_price", np.array([t + tf.active_company_high_price])))
    groups.append(("turn:active_company_income", np.array([t + tf.active_company_income])))

    groups.append(("turn:active_corp", np.arange(t + tf.active_corp, t + tf.active_corp + NK)))
    groups.append(("turn:active_corp_cash", np.array([t + tf.active_corp_cash])))
    groups.append(("turn:active_corp_unissued_shares", np.array([t + tf.active_corp_unissued_shares])))
    groups.append(("turn:active_corp_issued_shares", np.array([t + tf.active_corp_issued_shares])))
    groups.append(("turn:active_corp_bank_shares", np.array([t + tf.active_corp_bank_shares])))
    groups.append(("turn:active_corp_income", np.array([t + tf.active_corp_income])))
    groups.append(("turn:active_corp_stars", np.array([t + tf.active_corp_stars])))
    groups.append(("turn:active_corp_share_price", np.array([t + tf.active_corp_share_price])))
    groups.append(("turn:active_corp_acq_proceeds", np.array([t + tf.active_corp_acquisition_proceeds])))
    groups.append(("turn:active_corp_price_index_norm", np.array([t + tf.active_corp_price_index_norm])))
    groups.append(("turn:active_corp_pending_price_move", np.array([t + tf.active_corp_pending_price_move])))
    groups.append(("turn:active_corp_raw_revenue", np.array([t + tf.active_corp_raw_revenue])))
    groups.append(("turn:active_corp_synergy_income", np.array([t + tf.active_corp_synergy_income])))
    groups.append(("turn:active_corp_coo_cost", np.array([t + tf.active_corp_coo_cost])))
    groups.append(("turn:active_corp_ability_income", np.array([t + tf.active_corp_ability_income])))
    groups.append(("turn:active_corp_companies", np.arange(t + tf.active_corp_companies, t + tf.active_corp_companies + NC)))

    groups.append(("turn:cards_remaining", np.array([t + tf.cards_remaining])))

    groups.append(("turn:par_corp_treasury", np.arange(t + tf.par_corp_treasury, t + tf.par_corp_treasury + N_PAR)))
    groups.append(("turn:par_shares", np.arange(t + tf.par_shares, t + tf.par_shares + N_PAR)))

    # --- Auction slot info ---
    s = layout.auction_slot_info_offset
    groups.append(("auction_slot_info", np.arange(s, s + layout.auction_slot_info_size)))

    # --- Invest impacts ---
    groups.append(("invest:buy_impact", np.arange(layout.invest_impacts_offset, layout.invest_impacts_offset + NK)))
    groups.append(("invest:sell_impact", np.arange(layout.invest_impacts_offset + NK, layout.invest_impacts_offset + layout.invest_impacts_size)))

    # Verify full coverage
    all_idx = np.concatenate([g[1] for g in groups])
    assert len(all_idx) == layout.visible_size, (
        f"Coverage {len(all_idx)} != visible_size {layout.visible_size}"
    )
    assert len(set(all_idx)) == layout.visible_size, "Overlapping indices"

    return groups
