from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils_18xx import action_parser


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

    with pytest.raises(ValueError, match="FI target should execute during ACQ_SELECT_COMPANY"):
        action_parser.map_acquisition_action(
            object(),
            action,
            action_parser.PHASE_ACQ_SELECT_PRICE,
            None,
        )
