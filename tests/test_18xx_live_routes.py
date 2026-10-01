import io
import json
import queue
from http.client import HTTPMessage

from utils_18xx.live import (
    AcqOfferTracker,
    ControlHandler,
    EvalRequest,
    FeedRequest,
    GameBlacklist,
    default_api_min_interval,
    is_local_request_host,
    parse_eval_request,
    parse_feed_request,
    parse_poke_game_id,
)


def test_parse_poke_game_id_from_path():
    assert parse_poke_game_id("/poke/12345") == "12345"


def test_parse_poke_game_id_from_query():
    assert parse_poke_game_id("/poke?game_id=12345") == "12345"
    assert parse_poke_game_id("/poke?game=abc") == "abc"


def test_parse_poke_game_id_rejects_other_paths():
    assert parse_poke_game_id("/listen/12345") is None
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


def test_parse_eval_request_from_tmp_file_path():
    assert parse_eval_request(
        "/eval/file/game%20256285.json?player=Alice"
    ) == EvalRequest(
        player="Alice",
        filename="game 256285.json",
    )


def test_parse_eval_request_rejects_other_paths():
    assert parse_eval_request("/listen/12345") is None
    assert parse_eval_request("/eval") is None
    assert parse_eval_request("/eval/123/extra") is None
    assert parse_eval_request("/eval/file") is None
    assert parse_eval_request("/eval/file/..%2Fsecret.json") is None


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


class _RecordingHandler(ControlHandler):
    """Exercise request handling without opening a network connection."""

    def __init__(self, path: str, body: str = "") -> None:
        self.path = path
        self.headers = HTTPMessage()
        self.headers["Content-Length"] = str(len(body.encode()))
        self.rfile = io.BytesIO(body.encode())
        self.response_body = io.BytesIO()
        self.wfile = self.response_body
        self.client_address = ("127.0.0.1", 12345)
        self.status_codes: list[int] = []
        self.sent_headers: list[tuple[str, str]] = []

    def send_response(self, code: int, message: str | None = None) -> None:
        self.status_codes.append(code)

    def send_header(self, keyword: str, value: str) -> None:
        self.sent_headers.append((keyword, value))

    def end_headers(self) -> None:
        pass


def _make_handler(path: str, body: str = "") -> _RecordingHandler:
    return _RecordingHandler(path, body)


def test_manual_poke_bypasses_blacklist(tmp_path, monkeypatch):
    path = tmp_path / "blacklisted_games.json"
    path.write_text(json.dumps([254153]))
    work_queue = queue.Queue()

    monkeypatch.setattr(ControlHandler, "work_queue", work_queue, raising=False)
    monkeypatch.setattr(
        ControlHandler,
        "auth",
        {"rss-az-1": {"token": "token"}},
        raising=False,
    )
    monkeypatch.setattr(
        ControlHandler,
        "game_blacklist",
        GameBlacklist(path),
        raising=False,
    )

    handler = _make_handler("/poke/254153")

    handler.do_GET()

    assert handler.status_codes == [202]
    assert work_queue.get_nowait() == ("rss-az-1", "254153")


def test_manual_eval_queues_eval_request(monkeypatch):
    work_queue = queue.Queue()

    monkeypatch.setattr(ControlHandler, "work_queue", work_queue, raising=False)
    monkeypatch.setattr(
        ControlHandler,
        "auth",
        {"rss-az-1": {"token": "token"}},
        raising=False,
    )

    handler = _make_handler("/eval/254153?player_index=1&bot=rss-az-1")

    handler.do_GET()

    assert handler.status_codes == [202]
    assert work_queue.get_nowait() == EvalRequest(
        game_id="254153",
        player_index=1,
        bot_name="rss-az-1",
    )


def test_manual_file_eval_queues_eval_request(monkeypatch):
    work_queue = queue.Queue()

    monkeypatch.setattr(ControlHandler, "work_queue", work_queue, raising=False)

    handler = _make_handler("/eval/file/256285.json?player_index=1")

    handler.do_GET()

    assert handler.status_codes == [202]
    assert work_queue.get_nowait() == EvalRequest(
        player_index=1,
        filename="256285.json",
    )
    assert json.loads(handler.response_body.getvalue())["filename"] == "256285.json"


def test_manual_eval_is_local_only(monkeypatch):
    work_queue = queue.Queue()

    monkeypatch.setattr(ControlHandler, "work_queue", work_queue, raising=False)

    handler = _make_handler("/eval/254153")
    handler.client_address = ("203.0.113.7", 12345)

    handler.do_GET()

    assert handler.status_codes == [403]
    assert work_queue.empty()


def test_parse_feed_request():
    assert parse_feed_request("/listen") == FeedRequest("status")
    assert parse_feed_request("/listen/12345") == FeedRequest(
        "listen", game_id="12345",
    )
    assert parse_feed_request("/listen/12345?resync=1") == FeedRequest(
        "listen", game_id="12345", resync=True,
    )
    assert parse_feed_request("/unlisten/12345") == FeedRequest(
        "unlisten", game_id="12345",
    )
    assert parse_feed_request("/unlisten") is None
    assert parse_feed_request("/listen/..%2Fx") is None
    assert parse_feed_request("/listen/12/extra") is None
    assert parse_feed_request("/poke/12345") is None


class _RecordingFeeds:
    def __init__(self, known=()):
        self.calls = []
        self.known = set(known)

    def listen(self, game_id, *, resync=False):
        self.calls.append(("listen", game_id, resync))

    def unlisten(self, game_id):
        self.calls.append(("unlisten", game_id))

    def has_feed(self, game_id):
        return game_id in self.known

    def status(self):
        return [{"game_id": game_id} for game_id in sorted(self.known)]


def _use_feeds(monkeypatch, feeds, work_queue=None):
    monkeypatch.setattr(ControlHandler, "game_feeds", feeds, raising=False)
    monkeypatch.setattr(
        ControlHandler, "work_queue", work_queue or queue.Queue(), raising=False,
    )
    monkeypatch.setattr(
        ControlHandler, "auth", {"rss-az-1": {"token": "token"}}, raising=False,
    )
    monkeypatch.setattr(ControlHandler, "game_blacklist", None, raising=False)


def test_listen_routes_drive_game_feeds(monkeypatch):
    feeds = _RecordingFeeds(known={"254153"})
    _use_feeds(monkeypatch, feeds)

    listen = _make_handler("/listen/254153?resync=1")
    listen.do_POST()
    unlisten = _make_handler("/unlisten/254153")
    unlisten.do_GET()
    status = _make_handler("/listen")
    status.do_GET()

    assert listen.status_codes == [202]
    assert unlisten.status_codes == [202]
    assert status.status_codes == [200]
    assert json.loads(status.response_body.getvalue()) == {
        "games": [{"game_id": "254153"}],
    }
    assert feeds.calls == [
        ("listen", "254153", True),
        ("unlisten", "254153"),
    ]


def test_unlisten_unknown_game_is_404(monkeypatch):
    feeds = _RecordingFeeds()
    _use_feeds(monkeypatch, feeds)

    handler = _make_handler("/unlisten/254153")
    handler.do_POST()

    assert handler.status_codes == [404]
    assert feeds.calls == []


def test_listen_routes_are_local_only(monkeypatch):
    feeds = _RecordingFeeds()
    _use_feeds(monkeypatch, feeds)
    remote = _make_handler("/listen/254153")
    remote.client_address = ("203.0.113.7", 12345)
    remote.do_POST()
    assert remote.status_codes == [403]
    assert feeds.calls == []


def test_unknown_routes_are_404(monkeypatch):
    feeds = _RecordingFeeds()
    work_queue = queue.Queue()
    _use_feeds(monkeypatch, feeds, work_queue)

    handler = _make_handler("/webhook/rss-az-1", '{"text": "Your Turn"}')
    handler.do_POST()

    assert handler.status_codes == [404]
    assert feeds.calls == []
    assert work_queue.empty()
