"""Long-poll client for the 18xx.games MessageBus endpoint.

18xx.games publishes every accepted game action to the MessageBus channel
``/game/<id>`` (see API.md). This client follows the browser's message-bus.js
loop: one client id and one outstanding ``POST /message-bus/<client>/poll``
covering every subscribed channel. Game channels need no authentication.

Poll body: ``{"<channel>": <last seen message id>, "__seq": <n>}``. A last id
of -1 asks only for the channel's current position, which the server returns
as a ``/__status`` message. The server holds long polls for up to 25 seconds
and answers with a JSON list of ``{global_id, message_id, channel, data}``.
Each channel keeps a backlog of one message, so a poller that falls behind by
two or more messages only receives the latest; consumers must detect gaps.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import sys
import threading
import time
import uuid
from typing import Protocol
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

STATUS_CHANNEL = "/__status"
# The server holds long polls for 25s; nginx's default proxy_read_timeout is 60s.
POLL_REQUEST_TIMEOUT = 60.0
# message-bus.js minPollInterval.
MIN_POLL_INTERVAL = 0.1
# Spacing for polls the server answers without holding (long polling disabled
# server-side). The browser would use 18xx's 1s callbackInterval.
SHORT_POLL_INTERVAL = 5.0
MAX_ERROR_BACKOFF = 60.0
# 18xx.games' nginx often answers polls with brief 502s; log failures quietly
# until this many in a row (about a minute of backoff), then warn.
FAILURE_WARN_COUNT = 7
_USER_AGENT = f"Python-urllib/{sys.version_info.major}.{sys.version_info.minor}"


class MessageBusError(Exception):
    """A poll failed or returned an unusable response."""


class _PollInterrupted(Exception):
    """The subscription set changed while a poll was being prepared."""


class MessageBusListener(Protocol):
    def on_bus_message(self, channel: str, message_id: int, data) -> None: ...

    def on_bus_status(
        self, channel: str, position: int, solicited: bool,
    ) -> None: ...

    def on_bus_synced(self, channels: set[str]) -> None: ...


class MessageBusClient:
    """Background long-poller that dispatches channel messages to a listener.

    Listener callbacks run on the poll thread and should return quickly: the
    one-message backlog means slow re-polling loses messages.

    ``on_bus_status`` reports a channel position the server sent: ``solicited``
    is True when the subscription asked for it (last id -1) and False when the
    server reset a subscription that was ahead of the bus (the channel expired
    after two idle days and restarted its ids).

    ``on_bus_synced`` fires after the first completed poll that included newly
    subscribed channels, once any backlog for them has been dispatched.
    """

    def __init__(
        self,
        base_url: str,
        listener: MessageBusListener,
        *,
        client_id: str | None = None,
        request_timeout: float = POLL_REQUEST_TIMEOUT,
    ):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Unsupported MessageBus base URL: {base_url!r}")
        self._https = parsed.scheme == "https"
        self._host = parsed.hostname
        self._port = parsed.port
        self._base_path = parsed.path.rstrip("/")
        self._listener = listener
        self.client_id = client_id or uuid.uuid4().hex
        self._request_timeout = request_timeout

        self._cond = threading.Condition()
        self._positions: dict[str, int] = {}
        self._unsynced: set[str] = set()
        self._seq = 0
        self._conn: http.client.HTTPConnection | None = None
        self._interrupted = False
        self._stopped = False
        self._thread: threading.Thread | None = None

    # -- subscriptions -----------------------------------------------------

    def subscribe(self, channel: str, position: int = -1) -> None:
        """Watch ``channel`` from ``position`` (-1: from now, via status)."""
        with self._cond:
            self._positions[channel] = int(position)
            self._unsynced.add(channel)
            self._interrupt_locked()
            self._cond.notify_all()

    def unsubscribe(self, channel: str) -> None:
        with self._cond:
            self._positions.pop(channel, None)
            self._unsynced.discard(channel)

    def position(self, channel: str) -> int | None:
        with self._cond:
            return self._positions.get(channel)

    # -- thread control ----------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="message-bus",
        )
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._interrupt_locked()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        logger.info("MessageBus poller started (client %s)", self.client_id)
        failures = 0
        while True:
            started = time.monotonic()
            try:
                result = self.poll_once()
            except Exception as exc:
                if self._consume_interrupt():
                    continue
                failures += 1
                delay = min(MAX_ERROR_BACKOFF, 2.0 ** (failures - 1))
                logger.log(
                    logging.WARNING
                    if failures >= FAILURE_WARN_COUNT
                    else logging.DEBUG,
                    "MessageBus poll failed (%d in a row): %s; retrying in %.0fs",
                    failures,
                    exc,
                    delay,
                )
                self._pause(delay)
                continue
            if result is None:
                return
            if failures >= FAILURE_WARN_COUNT:
                logger.info("MessageBus poll recovered after %d failures", failures)
            failures = 0
            long_poll, message_count = result
            elapsed = time.monotonic() - started
            if message_count:
                delay = MIN_POLL_INTERVAL
            elif long_poll:
                # An empty long poll that returns early means the server did
                # not hold it; fall back to short-poll spacing.
                delay = max(0.0, SHORT_POLL_INTERVAL - elapsed)
            else:
                delay = 0.0
            self._pause(delay)

    def _pause(self, delay: float) -> None:
        """Sleep, waking early for stop or new subscriptions."""
        deadline = time.monotonic() + delay
        with self._cond:
            while not self._stopped and not self._unsynced:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(remaining)

    def _consume_interrupt(self) -> bool:
        with self._cond:
            interrupted = self._interrupted
            self._interrupted = False
            return interrupted

    def _interrupt_locked(self) -> None:
        self._interrupted = True
        conn = self._conn
        sock = conn.sock if conn is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    # -- polling -----------------------------------------------------------

    def poll_once(self) -> tuple[bool, int] | None:
        """Run one poll; return (was_long_poll, message_count), None if stopped.

        Blocks while nothing is subscribed. Raises on request failure, with
        newly subscribed channels kept pending for the next attempt.
        """
        with self._cond:
            while not self._positions and not self._stopped:
                self._cond.wait()
            if self._stopped:
                return None
            self._interrupted = False
            body: dict[str, int] = dict(self._positions)
            new_channels = set(self._unsynced)
            self._unsynced.clear()
            self._seq += 1
            seq = self._seq

        # Short-poll when subscriptions changed so their status/backlog comes
        # back immediately instead of after the long-poll hold.
        long_poll = not new_channels
        try:
            messages = self._request(body, seq, long_poll=long_poll)
        except BaseException:
            with self._cond:
                self._unsynced |= new_channels & self._positions.keys()
            raise

        count = self._dispatch(messages, body)
        if new_channels:
            with self._cond:
                synced = new_channels & self._positions.keys()
            if synced:
                self._listener.on_bus_synced(synced)
        return long_poll, count

    def _request(self, body: dict[str, int], seq: int, *, long_poll: bool) -> list:
        path = f"{self._base_path}/message-bus/{self.client_id}/poll"
        if not long_poll:
            path += "?dlp=t"
        payload = json.dumps({**body, "__seq": seq}).encode()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Dont-Chunk": "true",
            "X-SILENCE-LOGGER": "true",
            "User-Agent": _USER_AGENT,
        }
        conn_cls = (
            http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        )
        conn = conn_cls(self._host, self._port, timeout=self._request_timeout)
        try:
            conn.connect()
            with self._cond:
                if self._interrupted or self._stopped:
                    raise _PollInterrupted()
                self._conn = conn
            conn.request("POST", path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
        finally:
            with self._cond:
                self._conn = None
            conn.close()

        if resp.status != 200:
            detail = f"HTTP {resp.status} {resp.reason}"
            # Skip proxy error pages (nginx's 502 HTML).
            if not (resp.getheader("Content-Type") or "").startswith("text/html"):
                detail += f": {raw[:200].decode(errors='replace')}"
            raise MessageBusError(detail)
        text = raw.decode().strip()
        messages = json.loads(text) if text else []
        if not isinstance(messages, list):
            raise MessageBusError(f"Unexpected poll response: {text[:200]!r}")
        return messages

    def _dispatch(self, messages: list, body: dict[str, int]) -> int:
        count = 0
        for message in messages:
            if not isinstance(message, dict):
                continue
            channel = message.get("channel")
            data = message.get("data")
            if channel == STATUS_CHANNEL:
                if isinstance(data, dict):
                    self._dispatch_status(data, body)
                continue
            message_id = message.get("message_id")
            if not isinstance(message_id, int):
                continue
            with self._cond:
                if channel not in self._positions:
                    continue
                self._positions[channel] = max(
                    self._positions[channel], message_id,
                )
            count += 1
            self._listener.on_bus_message(channel, message_id, data)
        return count

    def _dispatch_status(self, data: dict, body: dict[str, int]) -> None:
        for channel, position in data.items():
            try:
                position = int(position)
            except (TypeError, ValueError):
                continue
            with self._cond:
                if channel not in self._positions:
                    continue
                self._positions[channel] = position
            self._listener.on_bus_status(
                channel, position, solicited=body.get(channel) == -1,
            )
