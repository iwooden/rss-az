import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from utils_18xx.message_bus import MessageBusClient, MessageBusError


class _FakeMessageBus:
    """Minimal server-side MessageBus semantics (message_bus 4.2 + 18xx.games).

    One retained message per channel, ``/__status`` for -1 and for clients
    ahead of the bus, ``dlp=t`` short polls, and long polls held until a
    publish or ``hold_secs``.
    """

    def __init__(self, hold_secs=5.0):
        self.hold_secs = hold_secs
        self.cond = threading.Condition()
        self.positions: dict[str, int] = {}
        self.latest: dict[str, dict] = {}
        self.requests: list[dict] = []
        self.held = 0
        self.fail_next = 0
        bus = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                bus.requests.append({
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": body,
                })
                if bus.fail_next:
                    bus.fail_next -= 1
                    self._reply(500, b"boom")
                    return
                subscriptions = {
                    k: int(v) for k, v in body.items() if k != "__seq"
                }
                messages = bus.backlog(subscriptions)
                if not messages and "dlp=t" not in self.path:
                    messages = bus.hold(subscriptions)
                self._reply(200, json.dumps(messages).encode())

            def _reply(self, status, payload):
                try:
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass  # client interrupted the poll

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def publish(self, channel, data):
        with self.cond:
            position = self.positions.get(channel, 0) + 1
            self.positions[channel] = position
            self.latest[channel] = {
                "global_id": position,
                "message_id": position,
                "channel": channel,
                "data": data,
            }
            self.cond.notify_all()

    def backlog(self, subscriptions):
        messages = []
        status = {}
        with self.cond:
            for channel, last_id in subscriptions.items():
                bus_id = self.positions.get(channel, 0)
                if last_id == -1 or last_id > bus_id:
                    status[channel] = bus_id
                elif last_id < bus_id:
                    messages.append(self.latest[channel])
        if status:
            messages.append({
                "global_id": -1,
                "message_id": -1,
                "channel": "/__status",
                "data": status,
            })
        return messages

    def hold(self, subscriptions):
        deadline = time.monotonic() + self.hold_secs
        with self.cond:
            self.held += 1
            try:
                while True:
                    messages = self.backlog(subscriptions)
                    remaining = deadline - time.monotonic()
                    if messages or remaining <= 0:
                        return messages
                    self.cond.wait(remaining)
            finally:
                self.held -= 1


class _Listener:
    def __init__(self):
        self.cond = threading.Condition()
        self.events: list[tuple] = []

    def on_bus_message(self, channel, message_id, data):
        self._add(("message", channel, message_id, data))

    def on_bus_status(self, channel, position, solicited):
        self._add(("status", channel, position, solicited))

    def on_bus_synced(self, channels):
        self._add(("synced", frozenset(channels)))

    def _add(self, event):
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def wait_for(self, event, timeout=5.0):
        with self.cond:
            assert self.cond.wait_for(lambda: event in self.events, timeout), (
                f"missing {event}; got {self.events}"
            )


@pytest.fixture
def fake_bus():
    bus = _FakeMessageBus()
    yield bus
    bus.close()


def test_status_then_long_polled_messages(fake_bus):
    fake_bus.publish("/game/1", {"id": 1})
    listener = _Listener()
    client = MessageBusClient(fake_bus.base_url, listener, client_id="abc")
    client.start()
    try:
        client.subscribe("/game/1")
        listener.wait_for(("status", "/game/1", 1, True))
        listener.wait_for(("synced", frozenset({"/game/1"})))

        fake_bus.publish("/game/1", {"id": 2, "type": "pass"})
        listener.wait_for(("message", "/game/1", 2, {"id": 2, "type": "pass"}))
    finally:
        client.stop()

    first, second = fake_bus.requests[:2]
    assert first["path"] == "/message-bus/abc/poll?dlp=t"
    assert first["body"] == {"/game/1": -1, "__seq": 1}
    assert first["headers"]["Dont-Chunk"] == "true"
    assert second["path"] == "/message-bus/abc/poll"
    assert second["body"] == {"/game/1": 1, "__seq": 2}
    assert client.position("/game/1") == 2


def test_subscribe_interrupts_held_long_poll(fake_bus):
    fake_bus.hold_secs = 30.0
    listener = _Listener()
    client = MessageBusClient(fake_bus.base_url, listener)
    client.start()
    try:
        client.subscribe("/game/1")
        listener.wait_for(("synced", frozenset({"/game/1"})))
        deadline = time.monotonic() + 5
        while fake_bus.held == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fake_bus.held == 1

        started = time.monotonic()
        client.subscribe("/game/2", 0)
        listener.wait_for(("synced", frozenset({"/game/2"})), timeout=5)
        assert time.monotonic() - started < 5
        assert fake_bus.requests[-1]["body"] == {
            "/game/1": 0, "/game/2": 0, "__seq": 3,
        }
    finally:
        client.stop()


def test_one_message_backlog_skips_to_latest(fake_bus):
    listener = _Listener()
    client = MessageBusClient(fake_bus.base_url, listener)
    client.subscribe("/game/1")
    assert client.poll_once() == (False, 0)

    fake_bus.publish("/game/1", {"id": 1})
    fake_bus.publish("/game/1", {"id": 2})
    assert client.poll_once() == (True, 1)

    assert listener.events[-1] == ("message", "/game/1", 2, {"id": 2})
    assert client.position("/game/1") == 2


def test_client_ahead_of_bus_gets_unsolicited_status(fake_bus):
    fake_bus.publish("/game/1", {"id": 7})
    listener = _Listener()
    client = MessageBusClient(fake_bus.base_url, listener)
    client.subscribe("/game/1", 50)
    client.poll_once()

    assert ("status", "/game/1", 1, False) in listener.events
    assert client.position("/game/1") == 1


def test_failed_poll_keeps_new_channels_pending(fake_bus):
    fake_bus.fail_next = 1
    listener = _Listener()
    client = MessageBusClient(fake_bus.base_url, listener)
    client.subscribe("/game/1")

    with pytest.raises(MessageBusError):
        client.poll_once()
    assert listener.events == []

    assert client.poll_once() == (False, 0)
    assert fake_bus.requests[-1]["path"].endswith("?dlp=t")
    assert listener.events == [
        ("status", "/game/1", 0, True),
        ("synced", frozenset({"/game/1"})),
    ]


def test_unsubscribed_channel_messages_are_dropped(fake_bus):
    listener = _Listener()
    client = MessageBusClient(fake_bus.base_url, listener)
    client.subscribe("/game/1", 0)
    client.subscribe("/game/2", 0)
    client.poll_once()
    client.unsubscribe("/game/2")
    fake_bus.publish("/game/1", {"id": 1})
    fake_bus.publish("/game/2", {"id": 1})

    client.poll_once()

    assert [e for e in listener.events if e[0] == "message"] == [
        ("message", "/game/1", 1, {"id": 1}),
    ]
