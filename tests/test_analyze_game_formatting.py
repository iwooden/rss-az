import json
from pathlib import Path
from types import SimpleNamespace

import torch

import train.analyze_game as analyze_game_module
from core.actions import (
    ACTION_AUCTION_PY as ACTION_AUCTION,
    ACTION_ISSUE_PY as ACTION_ISSUE,
    ACTION_IPO_PY as ACTION_IPO,
    get_decision_phase_py,
)
from core.state import GameState
from core.data import COMPANY_NAMES, CORP_NAMES
from entities.company import COMPANIES
from entities.turn import TURN
from nn import create_model
from phases.closing import setup_closing_phase_py
from phases.issue import setup_issue_phase_py
from phases.ipo import setup_ipo_phase_py
from tests.phases.conftest import float_corp_for_test, find_legal_action_with_info
from tests.phases.helpers.ownership import give_company_to_corp, give_company_to_player
from train.analyze_game import (
    Initial18xxSetup,
    _resolve_eval_dtype_override,
    analyze_game,
    format_action,
    format_phase_context,
    format_state_full,
    format_token_dump,
)
from train.config import TrainingConfig


def _make_state(num_players: int = 3, seed: int = 42) -> GameState:
    state = GameState(num_players)
    state.initialize_game(num_players, seed=seed)
    # Formatting tests exercise the full legality surface; flip on the
    # compatibility flag so CLOSING doesn't short-circuit to INCOME when
    # the setup only provides positive-income companies.
    state.allow_positive_income_closing = True
    return state


def test_format_state_full_restores_old_trace_sections() -> None:
    state = _make_state()

    rendered = format_state_full(state)

    assert "Phase: INVEST" in rendered
    assert "Turn: 1" in rendered
    assert "**Players**" in rendered
    assert "**FI**:" in rendered
    assert "**Auction Row** [3]:" in rendered
    assert "**Deck**:" in rendered


def test_format_action_invest_auction_uses_old_slot_bid_style() -> None:
    state = _make_state()
    phase_id = get_decision_phase_py(state)
    action_id, info = find_legal_action_with_info(state, action_type=ACTION_AUCTION)

    rendered = format_action(phase_id, action_id, state)

    assert rendered.startswith("AUCTION slot ")
    assert COMPANY_NAMES[info.company_id] in rendered
    assert "face $" in rendered


def test_format_action_issue_uses_active_corp_name() -> None:
    state = _make_state()
    float_corp_for_test(state, corp_id=0, company_id=0, player_id=0, par_index=10)
    setup_issue_phase_py(state)
    phase_id = get_decision_phase_py(state)
    action_id, _ = find_legal_action_with_info(state, action_type=ACTION_ISSUE)

    rendered = format_action(phase_id, action_id, state)

    assert CORP_NAMES[0] in rendered
    assert CORP_NAMES[-1] not in rendered


def test_format_action_ipo_includes_company_and_corp() -> None:
    state = _make_state()
    give_company_to_player(state, 14, 0)
    setup_ipo_phase_py(state)
    phase_id = get_decision_phase_py(state)
    action_id, _ = find_legal_action_with_info(state, action_type=ACTION_IPO, corp_id=0)

    rendered = format_action(phase_id, action_id, state)

    assert COMPANY_NAMES[14] in rendered
    assert CORP_NAMES[0] in rendered
    assert "float" in rendered


def test_format_phase_context_closing_lists_closable_targets() -> None:
    state = _make_state()
    float_corp_for_test(state, corp_id=0, company_id=0, player_id=0, par_index=10)
    give_company_to_corp(state, 1, 0)
    setup_closing_phase_py(state)

    rendered = format_phase_context(state)

    assert "**Closing**" in rendered
    assert COMPANY_NAMES[0] in rendered or COMPANY_NAMES[1] in rendered


def test_format_phase_context_acq_offer_describes_original_and_offered_corp() -> None:
    state = _make_state()
    COMPANIES[7].transfer_to_fi(state)
    TURN.enter_acq_offer(
        state,
        offered_corp=4,
        company_id=7,
        price=33,
        original_corp=2,
        deciding_player=1,
    )

    rendered = format_phase_context(state)

    assert "**Acquisition Offer**" in rendered
    assert CORP_NAMES[4] in rendered
    assert CORP_NAMES[2] in rendered
    assert COMPANY_NAMES[7] in rendered
    assert "$33" in rendered


def test_format_token_dump_denormalizes_rows_into_compact_table() -> None:
    state = _make_state()

    rendered = format_token_dump(state)

    market_row = next(line for line in rendered.splitlines() if "| market_info |" in line)
    company_row = next(line for line in rendered.splitlines() if "| company[0] |" in line)
    global_row = next(line for line in rendered.splitlines() if "| global_info |" in line)
    company0 = COMPANIES[0]

    assert "idx | token" in rendered
    assert market_row.startswith("00 | market_info | 55 | attn=1 prices=[0, 5, 6, 7,")
    assert (
        f"low={company0.get_low_price()} face={company0.get_face_value()} "
        f"high={company0.get_high_price()}"
    ) in company_row
    assert f"cards_remaining={TURN.get_cards_remaining(state)}" in global_row


def test_analyze_game_token_dump_flag_includes_token_tables() -> None:
    torch.manual_seed(0)
    config = TrainingConfig(num_players=3)
    model = create_model(config).to(torch.device("cpu"))
    model.eval()

    rendered = analyze_game(
        model,
        torch.device("cpu"),
        config,
        seed=1,
        num_simulations=1,
        top_n=1,
        checkpoint_path="new",
        token_dump=True,
    )

    assert "## Token Dump" in rendered
    assert "idx | token" in rendered
    assert "market_info" in rendered


def test_analyze_game_eval_dtype_override_allows_float32() -> None:
    config = TrainingConfig(num_players=3, eval_dtype="bfloat16")

    assert _resolve_eval_dtype_override(config, None) == "bfloat16"
    assert _resolve_eval_dtype_override(config, "float16") == "float16"
    assert _resolve_eval_dtype_override(config, "bfloat16") == "bfloat16"
    assert _resolve_eval_dtype_override(config, "float32") is None


def test_analyze_game_token_dump_flag_appends_normalization_report() -> None:
    torch.manual_seed(0)
    config = TrainingConfig(num_players=3)
    model = create_model(config).to(torch.device("cpu"))
    model.eval()

    rendered = analyze_game(
        model,
        torch.device("cpu"),
        config,
        seed=1,
        num_simulations=1,
        top_n=1,
        checkpoint_path="new",
        token_dump=True,
    )

    assert "## Token Normalization Report" in rendered
    assert "### Threshold Summary" in rendered
    assert "threshold | fields_exceeding | worst_abs | worst_field" in rendered
    assert "> 1.00 |" in rendered
    assert "> 1.10 |" in rendered
    assert "> 1.25 |" in rendered
    assert "token | field | min | max | avg" in rendered
    assert "market_info | market_price[0] |" in rendered
    assert "company[0] | low_price |" in rendered
    assert "global_info | cards_remaining |" in rendered


def test_analyze_game_mixed_config_uses_requested_actual_player_count() -> None:
    torch.manual_seed(0)
    config = TrainingConfig(
        num_players=0,
        min_players=3,
        max_players=5,
        num_simulations=1,
        search_batch_size=1,
        dirichlet_epsilon=0.0,
    )
    model = create_model(config).to(torch.device("cpu"))
    model.eval()

    rendered = analyze_game(
        model,
        torch.device("cpu"),
        config,
        seed=1,
        num_simulations=1,
        search_batch_size=1,
        top_n=1,
        num_players=3,
        checkpoint_path=Path(analyze_game_module.__file__).resolve().parents[1]
        / "checkpoints"
        / "example.pt",
    )

    assert "# Self-Play Analysis" in rendered
    assert "# Checkpoint: checkpoints/example.pt" in rendered
    assert "P2: net worth $" in rendered
    assert "P3: net worth $" not in rendered
    assert "A0GB Value:" in rendered
    assert "(depth:" in rendered


def test_analyze_game_player_names_relabel_output() -> None:
    torch.manual_seed(0)
    config = TrainingConfig(
        num_players=3,
        num_simulations=1,
        search_batch_size=1,
        dirichlet_epsilon=0.0,
    )
    model = create_model(config).to(torch.device("cpu"))
    model.eval()

    rendered = analyze_game(
        model,
        torch.device("cpu"),
        config,
        seed=1,
        num_simulations=1,
        search_batch_size=1,
        top_n=1,
        checkpoint_path="new",
        player_names=["yernab", "jmzed3", "enjoilife"],
    )

    assert "### Step 0: yernab [INVEST]" in rendered
    assert "Active Player: yernab" in rendered
    assert "NN Values: yernab=" in rendered
    assert "jmzed3=" in rendered
    assert "enjoilife=" in rendered
    assert "A0GB Value: yernab=" in rendered
    assert "P0" not in rendered
    assert "P1" not in rendered
    assert "P2" not in rendered


def test_extract_18xx_seed_setup_builds_minimal_rss_game(monkeypatch) -> None:
    seen = {}

    def fake_run(cmd, **kwargs):
        game_path = cmd[2]
        game_data = json.loads(Path(game_path).read_text())
        seen["cmd"] = cmd
        seen["cwd"] = kwargs["cwd"]
        seen["game_data"] = game_data
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([
                {
                    "deck_order": ["BPM", "MS"],
                    "initial_offering": ["KME", "MHE", "BSE"],
                    "cost_level": 1,
                }
            ]),
            stderr="",
        )

    monkeypatch.setattr(analyze_game_module.subprocess, "run", fake_run)

    setup = analyze_game_module._extract_18xx_seed_setup(560979115, 3)

    assert setup == Initial18xxSetup(
        deck_order=["BPM", "MS"],
        initial_offering=["KME", "MHE", "BSE"],
        cost_level=1,
    )
    assert seen["cmd"][0] == "ruby"
    assert seen["game_data"]["title"] == "Rolling Stock Stars"
    assert seen["game_data"]["settings"]["seed"] == 560979115
    assert len(seen["game_data"]["players"]) == 3


def test_analyze_game_18xx_seed_logs_applied_initial_setup(monkeypatch) -> None:
    torch.manual_seed(0)
    config = TrainingConfig(
        num_players=3,
        num_simulations=1,
        search_batch_size=1,
        dirichlet_epsilon=0.0,
    )
    model = create_model(config).to(torch.device("cpu"))
    model.eval()
    calls: list[tuple[int, int]] = []

    def fake_apply(state, seed, num_players):
        del state
        calls.append((seed, num_players))
        return Initial18xxSetup(
            deck_order=["BPM", "MS"],
            initial_offering=["KME", "MHE", "BSE"],
            cost_level=1,
        )

    monkeypatch.setattr(analyze_game_module, "_apply_18xx_seed_setup", fake_apply)

    rendered = analyze_game(
        model,
        torch.device("cpu"),
        config,
        seed=1,
        seed_18xx=560979115,
        num_simulations=1,
        search_batch_size=1,
        top_n=1,
        checkpoint_path="checkpoints/seeded.pt",
    )

    assert calls == [(560979115, 3)]
    assert rendered.startswith("# Self-Play Analysis: 18xx seed=560979115, 1 simulations/move")
    assert "# Self-Play Analysis: seed=1" not in rendered
    assert "# Checkpoint: checkpoints/seeded.pt" in rendered
    assert "# 18xx seed: 560979115" in rendered
    assert "initial offering: KME, MHE, BSE" in rendered
    assert "remaining deck: 2" in rendered
