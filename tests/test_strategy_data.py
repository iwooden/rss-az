from __future__ import annotations

import numpy as np

from core.state import get_layout
from nn.transformer import UNIFIED_LOGIT_DIM
from train.self_play import GameRecord, StrategyTrace
from train.strategy_data import _StrategyShardWriter


U_DIM = int(UNIFIED_LOGIT_DIM)


def _empty_events(width: int) -> np.ndarray:
    return np.empty((0, width), dtype=np.int32)


def _fake_trace(num_examples: int, num_players: int) -> StrategyTrace:
    return StrategyTrace(
        nn_policy_pct=np.zeros((num_examples, U_DIM), dtype=np.float32),
        nn_values=np.zeros((num_examples, num_players), dtype=np.float32),
        mcts_visit_counts=np.zeros((num_examples, U_DIM), dtype=np.int32),
        a0gb_values=np.zeros((num_examples, num_players), dtype=np.float32),
        mcts_root_values=np.zeros((num_examples, num_players), dtype=np.float32),
        selected_action_ids=np.zeros(num_examples, dtype=np.uint16),
        selected_unified_slots=np.zeros(num_examples, dtype=np.int16),
        action_types=np.zeros(num_examples, dtype=np.int16),
        action_corps=np.full(num_examples, -1, dtype=np.int16),
        action_companies=np.full(num_examples, -1, dtype=np.int16),
        action_amounts=np.full(num_examples, -1, dtype=np.int16),
        engine_phase_ids=np.zeros(num_examples, dtype=np.int8),
        active_players=np.zeros(num_examples, dtype=np.int8),
        active_corps=np.full(num_examples, -1, dtype=np.int8),
        active_companies=np.full(num_examples, -1, dtype=np.int8),
        turn_numbers=np.ones(num_examples, dtype=np.int16),
        coo_levels=np.ones(num_examples, dtype=np.int8),
        cards_remaining=np.zeros(num_examples, dtype=np.int8),
        auction_prices=np.zeros(num_examples, dtype=np.int16),
        auction_high_bidders=np.full(num_examples, -1, dtype=np.int8),
        auction_starters=np.full(num_examples, -1, dtype=np.int8),
        acq_offer_prices=np.zeros(num_examples, dtype=np.int16),
        acq_offer_corps=np.full(num_examples, -1, dtype=np.int8),
        target_temperatures=np.ones(num_examples, dtype=np.float32),
        sample_temperatures=np.ones(num_examples, dtype=np.float32),
        greedy_leaf_depths=np.zeros(num_examples, dtype=np.int16),
        root_visit_counts=np.ones(num_examples, dtype=np.int32),
        player_cash=np.zeros((num_examples, num_players), dtype=np.int16),
        player_net_worth=np.zeros((num_examples, num_players), dtype=np.int16),
        player_liquidity=np.zeros((num_examples, num_players), dtype=np.int16),
        player_income=np.zeros((num_examples, num_players), dtype=np.int16),
        player_shares=np.zeros((num_examples, num_players, 8), dtype=np.int8),
        corp_active=np.zeros((num_examples, 8), dtype=np.int8),
        corp_prices=np.zeros((num_examples, 8), dtype=np.int16),
        corp_cash=np.zeros((num_examples, 8), dtype=np.int16),
        corp_income=np.zeros((num_examples, 8), dtype=np.int16),
        corp_presidents=np.full((num_examples, 8), -1, dtype=np.int8),
        corp_issued_shares=np.zeros((num_examples, 8), dtype=np.int8),
        corp_bank_shares=np.zeros((num_examples, 8), dtype=np.int8),
        corp_unissued_shares=np.zeros((num_examples, 8), dtype=np.int8),
        corp_receivership=np.zeros((num_examples, 8), dtype=np.int8),
        company_locations=np.zeros((num_examples, 36), dtype=np.int8),
        company_owners=np.full((num_examples, 36), -1, dtype=np.int8),
        company_adjusted_income=np.zeros((num_examples, 36), dtype=np.int16),
        auction_events=np.asarray([[0, 1, 2, 0, 7, 0, 0, 4, 1]], dtype=np.int32),
        ipo_events=_empty_events(11),
        acquisition_events=_empty_events(12),
        share_trade_events=_empty_events(12),
        dividend_events=_empty_events(9),
        issue_events=_empty_events(12),
        close_events=_empty_events(8),
    )


def _fake_record(game_id: int, num_examples: int = 2) -> GameRecord:
    num_players = 3
    state_size = get_layout(num_players).total_size
    policy = np.zeros((num_examples, U_DIM), dtype=np.float32)
    policy[:, 0] = 1.0
    return GameRecord(
        states=np.zeros((num_examples, state_size), dtype=np.int16),
        phase_ids=np.zeros(num_examples, dtype=np.int8),
        legal_masks=np.ones((num_examples, U_DIM), dtype=np.uint8),
        policy_targets=policy,
        value_targets=np.zeros((num_examples, num_players), dtype=np.float32),
        num_players=num_players,
        num_examples=num_examples,
        total_moves=num_examples,
        net_worths=[10, 20, 30],
        shares_per_player=[1, 2, 3],
        companies_per_player=[0, 1, 2],
        pres_share_values=[0.0, 0.0, 0.0],
        nw_cash_pct=[0.0, 0.0, 0.0],
        nw_companies_pct=[0.0, 0.0, 0.0],
        nw_shares_pct=[0.0, 0.0, 0.0],
        avg_active_corp_price=0.0,
        corps_in_receivership=0,
        has_max_price_corp=False,
        duration_secs=0.1,
        game_id=game_id,
        game_seed=100 + game_id,
        rng_seed=200 + game_id,
        final_state=np.zeros(state_size, dtype=np.int16),
        strategy_trace=_fake_trace(num_examples, num_players),
    )


def test_strategy_shard_writer_emits_analysis_arrays(tmp_path) -> None:
    writer = _StrategyShardWriter(tmp_path, games_per_shard=2, compress=False)
    writer.add(_fake_record(10))
    writer.add(_fake_record(11, num_examples=3))

    assert writer.files == ["strategy_3p_shard_00000.npz"]
    with np.load(tmp_path / writer.files[0]) as data:
        assert data["states"].shape[0] == 5
        assert data["game_start_offsets"].tolist() == [0, 2]
        assert data["game_num_examples"].tolist() == [2, 3]
        assert data["game_ids"].tolist() == [10, 10, 11, 11, 11]
        assert data["mcts_policy_pct"].shape == (5, U_DIM)
        assert data["auction_events"].shape == (2, 10)
        assert data["auction_events"][:, 0].tolist() == [10, 11]
