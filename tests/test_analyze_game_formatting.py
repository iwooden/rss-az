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
from phases.closing import setup_closing_phase_py
from phases.issue import setup_issue_phase_py
from phases.ipo import setup_ipo_phase_py
from tests.phases.conftest import float_corp_for_test, find_legal_action_with_info
from tests.phases.helpers.ownership import give_company_to_corp, give_company_to_player
from train.analyze_game import format_action, format_phase_context, format_state_full


def _make_state(num_players: int = 3, seed: int = 42) -> GameState:
    state = GameState(num_players)
    state.initialize_game(num_players, seed=seed)
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
    assert "bid $" in rendered


def test_format_action_issue_uses_active_corp_name() -> None:
    state = _make_state()
    float_corp_for_test(state, corp_id=0, company_id=0, player_id=0, par_index=10)
    setup_issue_phase_py(state)
    phase_id = get_decision_phase_py(state)
    action_id, _ = find_legal_action_with_info(state, action_type=ACTION_ISSUE)

    rendered = format_action(phase_id, action_id, state)

    assert CORP_NAMES[0] in rendered
    assert CORP_NAMES[-1] not in rendered


def test_format_action_ipo_includes_company_corp_and_par() -> None:
    state = _make_state()
    give_company_to_player(state, 14, 0)
    setup_ipo_phase_py(state)
    phase_id = get_decision_phase_py(state)
    action_id, _ = find_legal_action_with_info(state, action_type=ACTION_IPO, corp_id=0)

    rendered = format_action(phase_id, action_id, state)

    assert COMPANY_NAMES[14] in rendered
    assert CORP_NAMES[0] in rendered
    assert "@$" in rendered


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
