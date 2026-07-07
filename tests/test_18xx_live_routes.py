import io
import json
import queue

from utils_18xx.live import (
    AcqOfferTracker,
    EvalRequest,
    GameBlacklist,
    WebhookHandler,
    default_api_min_interval,
    is_local_request_host,
    is_turn_webhook_text,
    parse_eval_request,
    parse_poke_game_id,
)


def test_parse_poke_game_id_from_path():
    assert parse_poke_game_id("/poke/12345") == "12345"


def test_parse_poke_game_id_from_query():
    assert parse_poke_game_id("/poke?game_id=12345") == "12345"
    assert parse_poke_game_id("/poke?game=abc") == "abc"


def test_parse_poke_game_id_rejects_other_paths():
    assert parse_poke_game_id("/webhook/rss-az-1") is None
    assert parse_poke_game_id("/poke") is None


def test_parse_eval_request_from_path():
    assert parse_eval_request(
        "/eval/12345?player_index=1&bot=rss-az-2"
    ) == EvalRequest(
        game_id="12345",
        player_index=1,
        bot_name="rss-az-2",
    )


def test_parse_eval_request_from_query():
    assert parse_eval_request(
        "/eval?game_id=12345&player=Alice&player_id=101"
    ) == EvalRequest(
        game_id="12345",
        player="Alice",
        player_id="101",
    )


def test_parse_eval_request_rejects_other_paths():
    assert parse_eval_request("/webhook/rss-az-1") is None
    assert parse_eval_request("/eval") is None
    assert parse_eval_request("/eval/123/extra") is None


def test_turn_webhook_text_is_case_insensitive():
    assert is_turn_webhook_text(
        '<@rss-az-2> Your Turn in Rolling Stock Stars "" (Investment 1)'
    )
    assert is_turn_webhook_text(
        '<@rss-az-2> Your turn in Rolling Stock Stars "" (Investment 1)'
    )
    assert is_turn_webhook_text(
        '<@rss-az-2> YOUR TURN in Rolling Stock Stars "" (Investment 1)'
    )


def test_turn_webhook_text_rejects_non_turn_notifications():
    assert not is_turn_webhook_text(
        '<@rss-az-2> Game Finished in Rolling Stock Stars "" (Issue Shares 13)'
    )


def test_public_18xx_host_defaults_to_conservative_api_throttle():
    assert default_api_min_interval("https://18xx.games") == 10.0
    assert default_api_min_interval("https://www.18xx.games") == 10.0
    assert default_api_min_interval("http://localhost:9292") == 0.0


def test_poke_endpoint_host_check_allows_only_loopback():
    assert is_local_request_host("127.0.0.1")
    assert is_local_request_host("::1")
    assert is_local_request_host("localhost")
    assert not is_local_request_host("192.168.1.10")
    assert not is_local_request_host("203.0.113.7")


def test_game_blacklist_loads_json_list(tmp_path):
    path = tmp_path / "blacklisted_games.json"
    blacklist = GameBlacklist(path)

    assert not blacklist.contains("254153")

    path.write_text(json.dumps([254153, "abc"]))

    assert blacklist.contains("254153")
    assert blacklist.contains(254153)
    assert blacklist.contains("abc")
    assert not blacklist.contains("999")


def test_acq_offer_tracker_records_and_resets_by_turn(tmp_path):
    path = tmp_path / "acq_offer_tracking.json"
    tracker = AcqOfferTracker(path)

    assert tracker.rejection_counts("256285", 3) == {}
    assert tracker.record_rejection("256285", 3, 13748) == 1
    assert tracker.record_rejection("256285", 3, 13748) == 2
    assert tracker.record_rejection("256285", 3, 8923) == 1

    assert tracker.rejection_counts("256285", 3) == {
        "13748": 2,
        "8923": 1,
    }
    assert tracker.rejection_counts("256285", 4) == {}

    assert tracker.record_rejection("256285", 4, 13748) == 1
    assert tracker.rejection_counts("256285", 4) == {"13748": 1}
    assert json.loads(path.read_text()) == {
        "256285": {
            "rejections": {"13748": 1},
            "turn": 4,
        }
    }


def _make_handler(path: str, body: str = ""):
    handler = WebhookHandler.__new__(WebhookHandler)
    handler.path = path
    handler.headers = {"Content-Length": str(len(body.encode()))}
    handler.rfile = io.BytesIO(body.encode())
    handler.wfile = io.BytesIO()
    handler.client_address = ("127.0.0.1", 12345)
    handler.responses = []
    handler.sent_headers = []

    def send_response(status):
        handler.responses.append(status)

    def send_header(key, value):
        handler.sent_headers.append((key, value))

    def end_headers():
        pass

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = end_headers
    return handler


def _turn_webhook_body(game_id: str) -> str:
    return json.dumps({
        "text": (
            f"<@rss-az-1> Your turn in "
            f"https://18xx.games/game/{game_id} (Acquisition 8)"
        ),
    })


def test_webhook_ignores_blacklisted_game(tmp_path, monkeypatch):
    path = tmp_path / "blacklisted_games.json"
    path.write_text(json.dumps([254153]))
    work_queue = queue.Queue()

    monkeypatch.setattr(WebhookHandler, "work_queue", work_queue, raising=False)
    monkeypatch.setattr(
        WebhookHandler,
        "auth",
        {"rss-az-1": {"token": "token"}},
        raising=False,
    )
    monkeypatch.setattr(
        WebhookHandler,
        "game_blacklist",
        GameBlacklist(path),
        raising=False,
    )

    handler = _make_handler(
        "/webhook/rss-az-1",
        _turn_webhook_body("254153"),
    )

    handler.do_POST()

    assert handler.responses == [200]
    assert work_queue.empty()


def test_manual_poke_bypasses_blacklist(tmp_path, monkeypatch):
    path = tmp_path / "blacklisted_games.json"
    path.write_text(json.dumps([254153]))
    work_queue = queue.Queue()

    monkeypatch.setattr(WebhookHandler, "work_queue", work_queue, raising=False)
    monkeypatch.setattr(
        WebhookHandler,
        "auth",
        {"rss-az-1": {"token": "token"}},
        raising=False,
    )
    monkeypatch.setattr(
        WebhookHandler,
        "game_blacklist",
        GameBlacklist(path),
        raising=False,
    )

    handler = _make_handler("/poke/254153")

    handler.do_GET()

    assert handler.responses == [202]
    assert work_queue.get_nowait() == ("rss-az-1", "254153")


def test_manual_eval_queues_eval_request(monkeypatch):
    work_queue = queue.Queue()

    monkeypatch.setattr(WebhookHandler, "work_queue", work_queue, raising=False)
    monkeypatch.setattr(
        WebhookHandler,
        "auth",
        {"rss-az-1": {"token": "token"}},
        raising=False,
    )

    handler = _make_handler("/eval/254153?player_index=1&bot=rss-az-1")

    handler.do_GET()

    assert handler.responses == [202]
    assert work_queue.get_nowait() == EvalRequest(
        game_id="254153",
        player_index=1,
        bot_name="rss-az-1",
    )


def test_manual_eval_is_local_only(monkeypatch):
    work_queue = queue.Queue()

    monkeypatch.setattr(WebhookHandler, "work_queue", work_queue, raising=False)

    handler = _make_handler("/eval/254153")
    handler.client_address = ("203.0.113.7", 12345)

    handler.do_GET()

    assert handler.responses == [403]
    assert work_queue.empty()
