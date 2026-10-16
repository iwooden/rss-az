import json
import logging
import urllib.error

import pytest

from utils_18xx import api_client
from utils_18xx.api_client import ApiClient, RequestAborted


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode()


def test_api_client_throttles_request_starts(monkeypatch):
    now = [100.0]
    sleeps = []
    requested = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def fake_urlopen(req, timeout):
        requested.append((req.full_url, timeout))
        return _Response({"games": []})

    monkeypatch.setattr(api_client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(api_client.time, "sleep", fake_sleep)
    monkeypatch.setattr(api_client.urllib.request, "urlopen", fake_urlopen)

    client = ApiClient(
        "https://18xx.games",
        request_timeout=3.0,
        min_request_interval=10.0,
    )

    client.fetch_game("1", "token")
    client.fetch_game("1", "token")

    assert sleeps == [10.0]
    assert len(requested) == 2
    assert requested[0][1] == 3.0


def test_api_client_throttles_retry_attempts(monkeypatch):
    now = [100.0]
    sleeps = []
    calls = 0

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def fake_urlopen(req, timeout):
        nonlocal calls
        del req, timeout
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary")
        return _Response({"ok": True})

    monkeypatch.setattr(api_client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(api_client.time, "sleep", fake_sleep)
    monkeypatch.setattr(api_client.urllib.request, "urlopen", fake_urlopen)

    client = ApiClient("https://18xx.games", min_request_interval=10.0)

    assert client._request("GET", "/api/test", "token", retries=2) == {"ok": True}
    assert sleeps == [1.0, 9.0]


def test_api_client_checks_abort_after_throttling(monkeypatch):
    now = [100.0]
    sleeps = []
    requested = []
    checked_at = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    def fake_urlopen(req, timeout):
        del timeout
        requested.append(req.get_method())
        return _Response({})

    def game_changed():
        checked_at.append(now[0])
        return True

    monkeypatch.setattr(api_client.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(api_client.time, "sleep", fake_sleep)
    monkeypatch.setattr(api_client.urllib.request, "urlopen", fake_urlopen)

    client = ApiClient("https://18xx.games", min_request_interval=10.0)
    client.fetch_game("1", "token")

    # The game can change while the post waits its turn, so the check runs
    # after the throttle wait; an aborted post sends nothing.
    with pytest.raises(RequestAborted):
        client.post_action("1", {"type": "pass"}, "token", abort_if=game_changed)
    assert sleeps == [10.0]
    assert checked_at == [110.0]
    assert requested == ["GET"]

    client.post_action("1", {"type": "pass"}, "token", abort_if=lambda: False)
    assert requested == ["GET", "POST"]


def test_api_client_logs_method_and_route_without_payload(monkeypatch, caplog):
    def fake_urlopen(req, timeout):
        del req, timeout
        return _Response({"ok": True})

    monkeypatch.setattr(api_client.urllib.request, "urlopen", fake_urlopen)
    caplog.set_level(logging.INFO, logger="utils_18xx.api_client")

    client = ApiClient("https://18xx.games")
    client.post_action("123", {"type": "pass", "secret": "payload"}, "token")

    log_text = caplog.text
    assert "18xx API request: POST /api/game/123/action" in log_text
    assert "payload" not in log_text
    assert "token" not in log_text
