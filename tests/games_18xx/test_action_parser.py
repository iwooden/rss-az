from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils_18xx import action_parser


def test_map_dividend_action_ignores_non_active_corp(monkeypatch):
    action = {"type": "dividend", "entity": "OS", "amount": 0}

    monkeypatch.setattr(action_parser, "CORP_NAME_TO_ID", {"OS": 2})
    monkeypatch.setattr(action_parser, "TURN", SimpleNamespace(get_active_corp=lambda state: 5))

    def unexpected_find_legal_action(state, **kwargs):
        raise AssertionError("non-active corporation should not map to a legal action")

    monkeypatch.setattr(action_parser, "find_legal_action", unexpected_find_legal_action)

    assert action_parser.map_dividend_action(object(), action, None) is None


def test_map_dividend_action_uses_active_corp(monkeypatch):
    action = {"type": "dividend", "entity": "OS", "amount": 3}
    calls: list[dict] = []

    monkeypatch.setattr(action_parser, "CORP_NAME_TO_ID", {"OS": 2})
    monkeypatch.setattr(action_parser, "TURN", SimpleNamespace(get_active_corp=lambda state: 2))
    monkeypatch.setattr(
        action_parser,
        "find_legal_action",
        lambda state, **kwargs: calls.append(kwargs) or 123,
    )

    assert action_parser.map_dividend_action(object(), action, None) == 123
    assert calls == [{"action_type": action_parser.ACTION_DIVIDEND, "amount": 3}]


def test_map_issue_action_ignores_non_active_corp(monkeypatch):
    action = {"type": "sell_shares", "entity": "OS"}

    monkeypatch.setattr(action_parser, "CORP_NAME_TO_ID", {"OS": 2})
    monkeypatch.setattr(action_parser, "TURN", SimpleNamespace(get_active_corp=lambda state: 5))

    def unexpected_find_legal_action(state, **kwargs):
        raise AssertionError("non-active corporation should not map to a legal action")

    monkeypatch.setattr(action_parser, "find_legal_action", unexpected_find_legal_action)

    assert action_parser.map_issue_action(object(), action, None) is None


def test_map_issue_action_uses_active_corp(monkeypatch):
    action = {"type": "sell_shares", "entity": "OS"}
    calls: list[dict] = []

    monkeypatch.setattr(action_parser, "CORP_NAME_TO_ID", {"OS": 2})
    monkeypatch.setattr(action_parser, "TURN", SimpleNamespace(get_active_corp=lambda state: 2))
    monkeypatch.setattr(
        action_parser,
        "find_legal_action",
        lambda state, **kwargs: calls.append(kwargs) or 123,
    )

    assert action_parser.map_issue_action(object(), action, None) == 123
    assert calls == [{"action_type": action_parser.ACTION_ISSUE}]


def test_map_ipo_pass_ignores_non_active_company_entity(monkeypatch):
    action = {"type": "pass", "entity": "HH", "entity_type": "company"}

    monkeypatch.setattr(action_parser, "COMPANY_NAME_TO_ID", {"HH": 29, "E": 28})
    monkeypatch.setattr(action_parser, "TURN", SimpleNamespace(get_active_company=lambda state: 28))

    def unexpected_find_legal_action(state, **kwargs):
        raise AssertionError("non-active IPO company pass should not map")

    monkeypatch.setattr(action_parser, "find_legal_action", unexpected_find_legal_action)

    assert action_parser.map_ipo_action(object(), action, None) is None


def test_map_ipo_par_ignores_non_active_company_entity(monkeypatch):
    action = {
        "type": "par",
        "entity": "E",
        "entity_type": "company",
        "corporation": "S",
        "share_price": "24,0,15",
    }

    monkeypatch.setattr(action_parser, "COMPANY_NAME_TO_ID", {"SZD": 22, "E": 28})
    monkeypatch.setattr(action_parser, "TURN", SimpleNamespace(get_active_company=lambda state: 22))

    def unexpected_find_legal_action(state, **kwargs):
        raise AssertionError("non-active IPO company par should not map")

    monkeypatch.setattr(action_parser, "find_legal_action", unexpected_find_legal_action)

    assert action_parser.map_ipo_action(object(), action, None) is None


def test_map_par_action_ignores_non_active_company_entity(monkeypatch):
    action = {
        "type": "par",
        "entity": "E",
        "entity_type": "company",
        "share_price": "24,0,15",
    }

    monkeypatch.setattr(action_parser, "COMPANY_NAME_TO_ID", {"SZD": 22, "E": 28})
    monkeypatch.setattr(action_parser, "TURN", SimpleNamespace(get_active_company=lambda state: 22))

    def unexpected_find_legal_action(state, **kwargs):
        raise AssertionError("non-active PAR company price should not map")

    monkeypatch.setattr(action_parser, "find_legal_action", unexpected_find_legal_action)

    assert action_parser.map_par_action(object(), action, None) is None


def test_map_acquisition_action_uses_select_company_for_fi_target(monkeypatch):
    action = {"type": "offer", "company": "KK", "corporation": "OS", "price": 40}
    calls: list[dict] = []

    monkeypatch.setattr(action_parser, "COMPANY_NAME_TO_ID", {"KK": 7})
    monkeypatch.setattr(action_parser, "CORP_NAME_TO_ID", {"OS": 2})
    monkeypatch.setattr(
        action_parser,
        "COMPANIES",
        {7: SimpleNamespace(get_location=lambda state: action_parser.LOC_FI)},
    )
    monkeypatch.setattr(
        action_parser,
        "find_legal_action",
        lambda state, **kwargs: calls.append(kwargs) or 123,
    )

    result = action_parser.map_acquisition_action(
        object(),
        action,
        action_parser.PHASE_ACQ_SELECT_COMPANY,
        None,
    )

    assert result == 123
    assert calls == [{"action_type": action_parser.ACTION_ACQ_SELECT_COMPANY, "company_id": 7}]


def test_map_acquisition_action_rejects_fi_target_in_select_price(monkeypatch):
    action = {"type": "offer", "company": "KK", "corporation": "OS", "price": 40}

    monkeypatch.setattr(action_parser, "COMPANY_NAME_TO_ID", {"KK": 7})
    monkeypatch.setattr(action_parser, "CORP_NAME_TO_ID", {"OS": 2})
    monkeypatch.setattr(
        action_parser,
        "COMPANIES",
        {7: SimpleNamespace(get_location=lambda state: action_parser.LOC_FI)},
    )
    monkeypatch.setattr(
        action_parser,
        "TURN",
        SimpleNamespace(
            get_active_corp=lambda state: 2,
            get_active_company=lambda state: 7,
        ),
    )

    with pytest.raises(ValueError, match="FI target should execute during ACQ_SELECT_COMPANY"):
        action_parser.map_acquisition_action(
            object(),
            action,
            action_parser.PHASE_ACQ_SELECT_PRICE,
            None,
        )
