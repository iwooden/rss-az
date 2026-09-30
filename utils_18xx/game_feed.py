"""Incrementally maintained 18xx.games game data for live play.

A listened game is downloaded once (``GET /api/game/:id``) and then kept
current from the actions 18xx.games publishes to the MessageBus channel
``/game/<id>``. The fields the server derives from its engine after each
action (``acting``, ``round``, ``turn``, ``status``, ``result``) are
recomputed with the local Ruby engine (``game_status.rb``).

Bots are queued for the move worker when they newly appear in ``acting``,
mirroring the "Your Turn" webhooks 18xx.games sends (``acting - prev``), and
when another player's action leaves them in ``acting``, which happens in the
simultaneous Acquisition and Closing rounds (e.g. an answer to the bot's
offer) and gets no webhook. Turns that continue through the bot's own actions
(for example finishing one phase and opening the next) are handled by the
move worker's own loop, which waits here for its posted actions instead of
re-downloading the game.

Re-downloads happen only when the bus stream cannot be trusted: an action id
gap that is not filled in shortly (the server keeps one message per channel,
and concurrent posts can publish out of order), a channel reset, a failed
local status computation, or a webhook the bus never confirmed.

Feeds persist to ``<runtime>/game_feeds/<id>.json`` so restarts resume from
the saved game and bus position without downloading.
"""

from __future__ import annotations

import heapq
import itertools
import json
import logging
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .api_client import ApiClient, PermanentError, TransientError
from .message_bus import MessageBusClient

logger = logging.getLogger(__name__)

GAME_STATUS_PATH = Path(__file__).with_name("game_status.rb")
FEED_DIR_NAME = "game_feeds"
FEED_FILE_VERSION = 1
# Concurrent actions publish outside the server's action lock, so a later id
# can arrive first; give the missing one this long before re-downloading.
GAP_GRACE_SECS = 5.0
RESYNC_COOLDOWN_SECS = 30.0
DOWNLOAD_RETRY_SECS = 60.0
# A webhook whose turn the bus has not shown by then triggers a re-download.
WEBHOOK_GRACE_SECS = 15.0
# A turn edge this recent already covers a (late) webhook for the same bot.
WEBHOOK_MATCH_SECS = 120.0
ECHO_TIMEOUT_SECS = 30.0
READY_TIMEOUT_SECS = 60.0
# Longest per-action acting history computed for one update.
MAX_STATUS_HISTORY = 200

_STOP = object()

StatusFn = Callable[[dict, "int | None"], list[dict]]


class _Blacklist(Protocol):
    def contains(self, game_id) -> bool: ...


def compute_game_status(
    game_data: dict,
    from_count: int | None = None,
    *,
    timeout: float = 60.0,
) -> list[dict]:
    """Return derived game states for action prefixes ``from_count..N``.

    Each state has ``action_count``, ``round``, ``turn``, ``acting`` (user
    ids), ``finished`` and ``result``. Without ``from_count`` only the state
    after all actions is returned.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(game_data, f)
        game_path = f.name

    args = ["ruby", str(GAME_STATUS_PATH), game_path]
    if from_count is not None:
        args.append(str(from_count))
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
        )
    finally:
        Path(game_path).unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"game_status.rb failed (rc={result.returncode}): "
            f"{result.stderr.strip()[:500]}"
        )
    states = json.loads(result.stdout)
    if not isinstance(states, list) or not states:
        raise RuntimeError("game_status.rb returned no states")
    return states


def is_valid_game_id(game_id) -> bool:
    return str(game_id).isdigit()


def _action_ids(game: dict) -> list[int]:
    return [
        action["id"]
        for action in game.get("actions") or []
        if isinstance(action.get("id"), int)
    ]


def _last_action_id(game: dict) -> int:
    return max(_action_ids(game), default=0)


def _acting_ids(game: dict | None) -> set[str]:
    if game is None:
        return set()
    return {str(user_id) for user_id in game.get("acting") or []}


def _turn_user_ids(
    before: set[str],
    steps: list[tuple[dict | None, set[str]]],
) -> set[str]:
    """User ids that may have something to do after ``steps``.

    A user qualifies when an action makes them newly acting (what 18xx's
    "Your Turn" webhook reports), or when another player's game action
    leaves them acting. The latter only happens in simultaneous rounds
    (Acquisition, Closing), e.g. a response to an offer the user made, and
    18xx sends no webhook for it. Unknown actions count as another player's.
    """
    user_ids: set[str] = set()
    previous = before
    for action, acting in steps:
        actor = None if action is None else str(action.get("user"))
        chat = action is not None and action.get("type") == "message"
        for user_id in acting:
            if user_id not in previous or (not chat and user_id != actor):
                user_ids.add(user_id)
        previous = acting
    return user_ids


def _extends(old: dict, new: dict) -> bool:
    """Return whether ``new``'s actions start with ``old``'s."""
    old_ids = _action_ids(old)
    return _action_ids(new)[: len(old_ids)] == old_ids


def _apply_status_fields(game: dict, state: dict) -> None:
    """Store server-equivalent derived fields in ``game`` (routes/game.rb)."""
    order = {
        str(player.get("id")): idx
        for idx, player in enumerate(game.get("players") or [])
    }
    finished = bool(state.get("finished"))
    game["acting"] = sorted(
        state.get("acting") or [],
        key=lambda user_id: order.get(str(user_id), len(order)),
    )
    game["round"] = state.get("round")
    game["turn"] = state.get("turn")
    game["status"] = "finished" if finished else "active"
    game["result"] = (state.get("result") or {}) if finished else {}


@dataclass
class _Feed:
    game_id: str
    listening: bool = True
    game: dict | None = None
    bus_position: int | None = None
    pending: dict[int, dict] = field(default_factory=dict)
    gap_since: float | None = None
    needs_download: bool = False
    download_scheduled: bool = False
    last_download_at: float | None = None
    downloads: int = 0
    # Whether acting bots were queued since this feed (re)started listening.
    announced: bool = False
    error: str | None = None
    enqueued_at_count: dict[str, int] = field(default_factory=dict)
    # Bots with a move-worker run queued but not yet started.
    queued: set[str] = field(default_factory=set)
    edge_at: dict[str, float] = field(default_factory=dict)
    webhook_due: dict[str, float] = field(default_factory=dict)
    webhook_bots: set[str] = field(default_factory=set)

    @property
    def channel(self) -> str:
        return f"/game/{self.game_id}"

    @property
    def action_count(self) -> int:
        return len(self.game.get("actions") or []) if self.game else 0

    @property
    def last_action_id(self) -> int:
        return _last_action_id(self.game) if self.game else 0

    @property
    def finished(self) -> bool:
        return bool(self.game) and self.game.get("status") == "finished"

    @property
    def ready(self) -> bool:
        return (
            self.game is not None
            and not self.pending
            and not self.needs_download
        )

    @property
    def state(self) -> str:
        if self.finished:
            return "finished"
        if not self.listening:
            return "stopped"
        if self.needs_download:
            return "downloading"
        if self.game is None:
            return "subscribing"
        if self.pending:
            return "gap"
        return "ready"


class GameFeedManager:
    """Owns the MessageBus poller and every listened game's cached data.

    Feed state changes on one manager thread (or in ``pump`` for tests);
    the public methods are safe to call from other threads.
    """

    def __init__(
        self,
        api: ApiClient,
        auth: dict[str, dict],
        feed_dir: Path,
        work_queue: queue.Queue,
        *,
        base_url: str | None = None,
        bus=None,
        game_blacklist: _Blacklist | None = None,
        status_fn: StatusFn = compute_game_status,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._api = api
        self._auth = auth
        self._feed_dir = feed_dir
        self._work_queue = work_queue
        self._blacklist = game_blacklist
        self._status_fn = status_fn
        self._clock = clock
        if bus is None:
            if base_url is None:
                raise ValueError("base_url is required without an explicit bus")
            bus = MessageBusClient(base_url, self)
        self._bus = bus

        self._cond = threading.Condition()
        self._feeds: dict[str, _Feed] = {}
        self._events: queue.Queue = queue.Queue()
        self._timers: list[tuple[float, int, tuple]] = []
        self._timer_seq = itertools.count()
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Resume persisted feeds and start the poller and manager threads."""
        self.load_persisted()
        self._bus.start()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="game-feeds",
        )
        self._thread.start()

    def stop(self) -> None:
        self._events.put(_STOP)
        self._bus.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def load_persisted(self) -> None:
        if not self._feed_dir.is_dir():
            return
        for path in sorted(self._feed_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                game = data["game"]
                game_id = str(data.get("game_id") or game["id"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping unreadable game feed %s: %s", path, exc)
                continue
            if not is_valid_game_id(game_id):
                continue
            position = data.get("bus_position")
            feed = _Feed(
                game_id,
                listening=bool(data.get("listening", True)),
                game=game,
                bus_position=int(position) if position is not None else None,
            )
            with self._cond:
                self._feeds[game_id] = feed
            if feed.listening and not feed.finished:
                logger.info(
                    "Resuming game feed %s at action %d (bus position %s)",
                    game_id,
                    feed.last_action_id,
                    feed.bus_position,
                )
                self._subscribe(feed)

    # -- public API ------------------------------------------------------------

    def listen(self, game_id, *, resync: bool = False) -> None:
        self._events.put(("listen", str(game_id), resync))

    def unlisten(self, game_id) -> None:
        self._events.put(("unlisten", str(game_id)))

    def notify_webhook(self, bot_name: str, game_id) -> None:
        self._events.put(("webhook", bot_name, str(game_id)))

    def request_resync(self, game_id) -> None:
        self._events.put(("resync", str(game_id)))

    def claim_turn(self, bot_name: str, game_id) -> None:
        """Mark a queued run as started; later changes may queue it again."""
        with self._cond:
            feed = self._feeds.get(str(game_id))
            if feed is not None:
                feed.queued.discard(bot_name)

    def has_feed(self, game_id) -> bool:
        with self._cond:
            return str(game_id) in self._feeds

    def is_listening(self, game_id) -> bool:
        with self._cond:
            feed = self._feeds.get(str(game_id))
            return feed is not None and feed.listening

    def status(self) -> list[dict]:
        with self._cond:
            return [self._describe(feed) for feed in self._feeds.values()]

    def snapshot(
        self,
        game_id,
        timeout: float = READY_TIMEOUT_SECS,
    ) -> dict | None:
        """Return current game data for a listened game, or None.

        Waits up to ``timeout`` for a listened feed to become ready (no
        download pending, no action gap).
        """
        game_id = str(game_id)
        with self._cond:
            ready = self._cond.wait_for(
                lambda: self._feed_ready_or_gone(game_id), timeout,
            )
            feed = self._feeds.get(game_id)
            if not ready or feed is None or not feed.listening or not feed.ready:
                return None
            return self._copy_game(feed)

    def wait_for_posted_actions(
        self,
        game_id,
        user_id,
        after_id: int,
        count: int,
        timeout: float = ECHO_TIMEOUT_SECS,
    ) -> dict | None:
        """Wait until ``count`` actions by ``user_id`` past ``after_id`` arrive.

        Returns the updated game data. If the bus does not deliver them in
        time, re-downloads the game and returns that instead (None if the
        feed never becomes ready).
        """
        game_id = str(game_id)

        def posted_seen() -> bool:
            feed = self._feeds.get(game_id)
            if feed is None or not feed.listening:
                return True
            if feed.game is None:
                return False
            seen = 0
            for action in reversed(feed.game.get("actions") or []):
                action_id = action.get("id")
                if isinstance(action_id, int) and action_id <= after_id:
                    break
                if str(action.get("user")) == str(user_id):
                    seen += 1
            return seen >= count

        with self._cond:
            if self._cond.wait_for(posted_seen, timeout):
                # A feed that stopped listening (e.g. our action finished the
                # game) still holds its final data.
                feed = self._feeds.get(game_id)
                if feed is None or feed.game is None:
                    return None
                return self._copy_game(feed)
            feed = self._feeds.get(game_id)
            downloads = feed.downloads if feed is not None else 0

        logger.warning(
            "Posted actions for game %s not seen on the bus after %.0fs; "
            "re-downloading",
            game_id,
            timeout,
        )
        self.request_resync(game_id)
        with self._cond:
            def downloaded() -> bool:
                feed = self._feeds.get(game_id)
                return (
                    feed is None
                    or not feed.listening
                    or (feed.downloads > downloads and feed.ready)
                )

            if not self._cond.wait_for(downloaded, READY_TIMEOUT_SECS):
                return None
            feed = self._feeds.get(game_id)
            if feed is None or not feed.listening or not feed.ready:
                return None
            return self._copy_game(feed)

    # -- MessageBus listener (poller thread) -----------------------------------

    def on_bus_message(self, channel: str, message_id: int, data) -> None:
        self._events.put(("bus_message", channel, message_id, data))

    def on_bus_status(self, channel: str, position: int, solicited: bool) -> None:
        self._events.put(("bus_status", channel, position, solicited))

    def on_bus_synced(self, channels: set[str]) -> None:
        self._events.put(("bus_synced", set(channels)))

    # -- event loop --------------------------------------------------------------

    def _run(self) -> None:
        while True:
            try:
                event = self._events.get(timeout=self._next_timer_delay())
            except queue.Empty:
                event = None
            if event is _STOP:
                return
            try:
                if event is not None:
                    self._handle(event)
                self._run_due_timers()
            except Exception:
                logger.exception("Game feed event failed: %r", event)

    def pump(self) -> None:
        """Process queued events and due timers on the calling thread."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            if event is not _STOP:
                self._handle(event)
        self._run_due_timers()

    def _next_timer_delay(self) -> float | None:
        if not self._timers:
            return None
        return max(0.0, self._timers[0][0] - self._clock())

    def _schedule(self, delay: float, event: tuple) -> None:
        heapq.heappush(
            self._timers,
            (self._clock() + delay, next(self._timer_seq), event),
        )

    def _run_due_timers(self) -> None:
        while self._timers and self._timers[0][0] <= self._clock():
            _, _, event = heapq.heappop(self._timers)
            self._handle(event)

    def _handle(self, event: tuple) -> None:
        kind, *args = event
        handler = getattr(self, f"_on_{kind}")
        handler(*args)

    # -- event handlers (manager thread) -----------------------------------------

    def _on_listen(self, game_id: str, resync: bool) -> None:
        if not is_valid_game_id(game_id):
            logger.warning("Ignoring listen for invalid game id %r", game_id)
            return
        feed = self._feeds.get(game_id)
        if feed is None:
            feed = _Feed(game_id)
            with self._cond:
                self._feeds[game_id] = feed
            logger.info("Listening to game %s", game_id)
            self._subscribe(feed)
            return
        if feed.finished and not resync:
            logger.info("Game %s is finished; not listening", game_id)
            return
        if not feed.listening:
            with self._cond:
                feed.listening = True
                feed.error = None
            logger.info("Resumed listening to game %s", game_id)
            self._subscribe(feed)
            self._persist(feed)
            return
        if resync:
            self._download(feed, "resync requested", force=True)

    def _on_unlisten(self, game_id: str) -> None:
        feed = self._feeds.get(game_id)
        if feed is None:
            return
        self._stop_listening(feed, "unlisten requested")

    def _on_webhook(self, bot_name: str, game_id: str) -> None:
        if not is_valid_game_id(game_id):
            logger.warning("Ignoring webhook for invalid game id %r", game_id)
            return
        feed = self._feeds.get(game_id)
        if feed is None or not feed.listening:
            self._on_listen(game_id, resync=False)
            feed = self._feeds.get(game_id)
            if feed is None or not feed.listening:
                # Finished game: let the worker take the webhook as before.
                self._enqueue_bot(game_id, bot_name, "webhook")
                return
            # Queued once the feed has (re)synchronized.
            feed.webhook_bots.add(bot_name)
            return
        if feed.game is None or not feed.announced:
            feed.webhook_bots.add(bot_name)
            return

        now = self._clock()
        edge_at = feed.edge_at.get(bot_name)
        if edge_at is not None and now - edge_at <= WEBHOOK_MATCH_SECS:
            logger.info(
                "Webhook for %s in game %s matches a bus turn event",
                bot_name,
                game_id,
            )
            return
        user_id = self._bots_in_game(feed.game).get(bot_name)
        if user_id is not None and user_id in _acting_ids(feed.game):
            self._enqueue(feed, bot_name, "webhook", dedupe=False)
            return
        logger.info(
            "Webhook for %s in game %s is ahead of the bus; waiting %.0fs",
            bot_name,
            game_id,
            WEBHOOK_GRACE_SECS,
        )
        feed.webhook_due[bot_name] = now + WEBHOOK_GRACE_SECS
        self._schedule(WEBHOOK_GRACE_SECS, ("webhook_due", game_id, bot_name))

    def _on_webhook_due(self, game_id: str, bot_name: str) -> None:
        feed = self._feeds.get(game_id)
        if feed is None or not feed.listening:
            return
        due = feed.webhook_due.get(bot_name)
        if due is None or due > self._clock():
            return
        del feed.webhook_due[bot_name]
        feed.webhook_bots.add(bot_name)
        self._download(feed, f"webhook for {bot_name} not seen on the bus")

    def _on_resync(self, game_id: str) -> None:
        # The move worker is blocked on this; skip the cooldown.
        feed = self._feeds.get(game_id)
        if feed is not None and feed.listening:
            self._download(feed, "posted action not seen on the bus", force=True)

    def _on_download(self, game_id: str, reason: str) -> None:
        feed = self._feeds.get(game_id)
        if feed is None or not feed.listening:
            return
        feed.download_scheduled = False
        self._download(feed, reason)

    def _on_gap_check(self, game_id: str) -> None:
        feed = self._feeds.get(game_id)
        if feed is None or not feed.listening or not feed.pending:
            return
        if feed.gap_since is None:
            return
        if self._clock() - feed.gap_since < GAP_GRACE_SECS:
            return
        logger.info(
            "Game %s: action %d missing after %.0fs (have %s); re-downloading",
            feed.game_id,
            feed.last_action_id + 1,
            GAP_GRACE_SECS,
            sorted(feed.pending),
        )
        self._download(feed, "action gap")

    def _on_bus_status(self, channel: str, position: int, solicited: bool) -> None:
        feed = self._feed_for_channel(channel)
        if feed is None or not feed.listening:
            return
        feed.bus_position = position
        if solicited:
            # Nothing before ``position`` can be replayed from the bus.
            with self._cond:
                feed.needs_download = True
            return
        if position > 0 and feed.game is not None:
            logger.info(
                "Game %s bus channel restarted at %d; re-downloading",
                feed.game_id,
                position,
            )
            with self._cond:
                feed.needs_download = True
            self._download(feed, "bus channel restarted", force=True)
        else:
            self._persist(feed)

    def _on_bus_synced(self, channels: set[str]) -> None:
        for channel in channels:
            feed = self._feed_for_channel(channel)
            if feed is None or not feed.listening:
                continue
            if feed.needs_download:
                self._download(feed, "listen", force=True)
            elif feed.game is not None and not feed.announced and not feed.pending:
                self._announce(feed)

    def _on_bus_message(self, channel: str, message_id: int, data) -> None:
        feed = self._feed_for_channel(channel)
        if feed is None or not feed.listening:
            return
        feed.bus_position = message_id
        if not isinstance(data, dict) or not isinstance(data.get("id"), int):
            logger.warning(
                "Ignoring unexpected message on %s: %r", channel, data,
            )
            return
        action = {key: value for key, value in data.items() if key != "_client_id"}
        action_id = action["id"]
        if feed.game is not None and action_id <= feed.last_action_id:
            return
        with self._cond:
            feed.pending[action_id] = action
        if feed.game is not None:
            self._apply_pending(feed)

    # -- feed updates --------------------------------------------------------------

    def _subscribe(self, feed: _Feed) -> None:
        position = feed.bus_position
        if feed.game is None or position is None:
            position = -1
        feed.announced = False
        self._bus.subscribe(feed.channel, position)

    def _stop_listening(self, feed: _Feed, reason: str) -> None:
        self._bus.unsubscribe(feed.channel)
        with self._cond:
            feed.listening = False
            feed.pending.clear()
            feed.gap_since = None
            feed.needs_download = False
            feed.webhook_due.clear()
            feed.webhook_bots.clear()
            self._cond.notify_all()
        logger.info("Stopped listening to game %s (%s)", feed.game_id, reason)
        self._persist(feed)

    def _apply_pending(self, feed: _Feed) -> None:
        assert feed.game is not None
        last_id = feed.last_action_id
        applied: list[dict] = []
        with self._cond:
            for action_id in [aid for aid in feed.pending if aid <= last_id]:
                del feed.pending[action_id]
            while last_id + 1 in feed.pending:
                last_id += 1
                applied.append(feed.pending.pop(last_id))

        if applied:
            game = dict(feed.game)
            game["actions"] = list(feed.game.get("actions") or []) + applied
            self._commit(feed, game, derive=True)

        if feed.pending:
            if feed.gap_since is None:
                feed.gap_since = self._clock()
                self._schedule(GAP_GRACE_SECS, ("gap_check", feed.game_id))
        else:
            feed.gap_since = None

    def _download(self, feed: _Feed, reason: str, *, force: bool = False) -> bool:
        """Replace the feed's game with a fresh download; False if deferred."""
        now = self._clock()
        if (
            not force
            and feed.last_download_at is not None
            and now - feed.last_download_at < RESYNC_COOLDOWN_SECS
        ):
            self._download_soon(
                feed,
                reason,
                RESYNC_COOLDOWN_SECS - (now - feed.last_download_at),
            )
            return False

        token = self._token_for(feed)
        if token is None:
            logger.error("No auth token available to download game %s", feed.game_id)
            return False

        logger.info("Downloading game %s (%s)", feed.game_id, reason)
        feed.last_download_at = now
        try:
            game = self._api.fetch_game(feed.game_id, token)
        except (TransientError, PermanentError) as exc:
            logger.error("Failed to download game %s: %s", feed.game_id, exc)
            with self._cond:
                feed.error = str(exc)
            self._download_soon(feed, reason, DOWNLOAD_RETRY_SECS)
            return False

        with self._cond:
            feed.downloads += 1
            feed.needs_download = False
            feed.error = None
            last_id = _last_action_id(game)
            for action_id in [aid for aid in feed.pending if aid <= last_id]:
                del feed.pending[action_id]
            extra: list[dict] = []
            while last_id + 1 in feed.pending:
                last_id += 1
                extra.append(feed.pending.pop(last_id))
            if feed.pending:
                # Either filtered from this download (chat hidden from
                # non-players) or superseded; a later message re-detects
                # any real gap.
                logger.info(
                    "Game %s: dropping unplaceable bus actions %s after download",
                    feed.game_id,
                    sorted(feed.pending),
                )
                feed.pending.clear()
            feed.gap_since = None

        if extra:
            game = dict(game)
            game["actions"] = list(game.get("actions") or []) + extra
        self._commit(feed, game, derive=bool(extra), downloaded=True)
        return True

    def _download_soon(self, feed: _Feed, reason: str, delay: float = 0.0) -> None:
        if not feed.download_scheduled:
            feed.download_scheduled = True
            self._schedule(delay, ("download", feed.game_id, reason))

    def _commit(
        self,
        feed: _Feed,
        game: dict,
        *,
        derive: bool,
        downloaded: bool = False,
    ) -> None:
        """Install ``game`` as the feed's data and queue newly acting bots.

        ``derive``: recompute the server-derived fields locally (the actions
        came from the bus). Downloads keep the server's fields and only use
        the local engine for per-action turn history.
        """
        old = feed.game
        old_acting = _acting_ids(old)
        history = (
            old is not None and feed.announced and _extends(old, game)
        )
        old_count = len(old.get("actions") or []) if old is not None else 0
        actions = game.get("actions") or []
        new_count = len(actions)
        # Bus actions (always the newest) lack the stored ``user``.
        from_count = new_count
        while from_count > 0 and "user" not in actions[from_count - 1]:
            from_count -= 1
        if history:
            from_count = min(from_count, old_count)
        from_count = max(from_count, new_count - MAX_STATUS_HISTORY)
        states: list[dict] | None = None
        if derive or (history and new_count > old_count):
            try:
                states = self._status_fn(game, from_count)
            except Exception as exc:
                logger.error(
                    "Local status for game %s failed: %s", feed.game_id, exc,
                )
                if derive and not downloaded:
                    # Keep the actions; refresh the derived fields remotely.
                    self._download_soon(feed, "local status failed")

        if states is not None:
            actions = list(actions)
            for state in states:
                count = int(state.get("action_count", 0))
                user_id = state.get("user")
                if (
                    user_id is not None
                    and 0 < count <= new_count
                    and "user" not in actions[count - 1]
                ):
                    actions[count - 1] = dict(actions[count - 1], user=user_id)
            game["actions"] = actions
        if states is not None and derive:
            _apply_status_fields(game, states[-1])
        elif states is not None and downloaded:
            local = {str(user_id) for user_id in states[-1].get("acting") or []}
            if local != _acting_ids(game):
                logger.warning(
                    "Game %s: local acting %s differs from server acting %s",
                    feed.game_id,
                    sorted(local),
                    sorted(_acting_ids(game)),
                )

        # (action, acting after it) per new action; None: action unknown.
        steps: list[tuple[dict | None, set[str]]]
        if states is not None and history:
            steps = [
                (
                    actions[count - 1],
                    {str(user_id) for user_id in state.get("acting") or []},
                )
                for state in states
                if (count := int(state.get("action_count", 0))) > old_count
            ]
        elif not feed.announced:
            old_acting = set()
            steps = [(None, _acting_ids(game))]
        elif new_count != old_count or _acting_ids(game) != old_acting:
            steps = [(None, _acting_ids(game))]
        else:
            steps = []
        turn_ids = _turn_user_ids(old_acting, steps)

        with self._cond:
            feed.game = game
            self._cond.notify_all()
        self._persist(feed)
        if new_count != old_count or old is None:
            logger.info(
                "Game %s at action %d: round=%s turn=%s acting=%s",
                feed.game_id,
                feed.last_action_id,
                game.get("round"),
                game.get("turn"),
                game.get("acting"),
            )

        if feed.finished:
            self._stop_listening(feed, "game finished")
            return
        self._queue_turns(feed, turn_ids)

    def _announce(self, feed: _Feed) -> None:
        """Queue acting bots after (re)subscribing without new actions."""
        self._queue_turns(feed, _acting_ids(feed.game))

    def _queue_turns(self, feed: _Feed, turn_ids: set[str]) -> None:
        feed.announced = True
        assert feed.game is not None
        acting = _acting_ids(feed.game)
        now = self._clock()
        queued: set[str] = set()
        for bot_name, user_id in self._bots_in_game(feed.game).items():
            if user_id in turn_ids:
                feed.edge_at[bot_name] = now
                feed.webhook_due.pop(bot_name, None)
                self._enqueue(feed, bot_name, "turn")
                queued.add(bot_name)
            elif bot_name in feed.webhook_due and user_id in acting:
                del feed.webhook_due[bot_name]
                self._enqueue(feed, bot_name, "webhook", dedupe=False)
                queued.add(bot_name)
        for bot_name in sorted(feed.webhook_bots - queued):
            feed.webhook_due.pop(bot_name, None)
            self._enqueue(feed, bot_name, "webhook", dedupe=False)
        feed.webhook_bots.clear()

    def _enqueue(
        self,
        feed: _Feed,
        bot_name: str,
        reason: str,
        *,
        dedupe: bool = True,
    ) -> None:
        """Queue a bot unless a run for it is already waiting.

        A waiting run reads the latest game data when it starts. Turn events
        also queue each bot at most once per action count; webhooks skip that
        check: like before the feed existed, the worker's replay decides
        whether there is anything to do.
        """
        count = feed.action_count
        with self._cond:
            if bot_name in feed.queued:
                return
            if dedupe and feed.enqueued_at_count.get(bot_name) == count:
                return
            if not self._enqueue_bot(feed.game_id, bot_name, reason):
                return
            feed.queued.add(bot_name)
            feed.enqueued_at_count[bot_name] = count

    def _enqueue_bot(self, game_id: str, bot_name: str, reason: str) -> bool:
        if self._blacklist is not None and self._blacklist.contains(game_id):
            logger.info(
                "Not queueing %s for blacklisted game %s", bot_name, game_id,
            )
            return False
        logger.info("Queueing %s for game %s (%s)", bot_name, game_id, reason)
        self._work_queue.put((bot_name, game_id))
        return True

    # -- helpers ---------------------------------------------------------------------

    def _feed_for_channel(self, channel: str) -> _Feed | None:
        prefix = "/game/"
        if not channel.startswith(prefix):
            return None
        return self._feeds.get(channel[len(prefix):])

    def _feed_ready_or_gone(self, game_id: str) -> bool:
        feed = self._feeds.get(game_id)
        return feed is None or not feed.listening or feed.ready

    def _bots_in_game(self, game: dict) -> dict[str, str]:
        """Map configured bot names to their user ids in ``game``."""
        players = game.get("players") or []
        player_ids = {str(player.get("id")) for player in players}
        bots: dict[str, str] = {}
        for bot_name, info in self._auth.items():
            user_id = info.get("user_id")
            if user_id is not None:
                if str(user_id) in player_ids:
                    bots[bot_name] = str(user_id)
                continue
            for player in players:
                if player.get("name") == bot_name:
                    bots[bot_name] = str(player.get("id"))
                    break
        return bots

    def _token_for(self, feed: _Feed) -> str | None:
        """Prefer a bot playing the game: 18xx hides chat from non-players."""
        candidates: list[str] = []
        if feed.game is not None:
            candidates.extend(self._bots_in_game(feed.game))
        candidates.extend(sorted(feed.webhook_bots))
        candidates.extend(self._auth)
        for bot_name in candidates:
            token = self._auth.get(bot_name, {}).get("token")
            if token:
                return token
        return None

    def _copy_game(self, feed: _Feed) -> dict:
        assert feed.game is not None
        game = dict(feed.game)
        game["actions"] = list(feed.game.get("actions") or [])
        return game

    def _describe(self, feed: _Feed) -> dict:
        game = feed.game or {}
        return {
            "game_id": feed.game_id,
            "state": feed.state,
            "listening": feed.listening,
            "actions": feed.action_count,
            "last_action_id": feed.last_action_id,
            "bus_position": feed.bus_position,
            "round": game.get("round"),
            "turn": game.get("turn"),
            "acting": game.get("acting"),
            "status": game.get("status"),
            "downloads": feed.downloads,
            "pending_actions": sorted(feed.pending),
            "error": feed.error,
        }

    def _persist(self, feed: _Feed) -> None:
        if feed.game is None:
            return
        path = self._feed_dir / f"{feed.game_id}.json"
        tmp_path = path.with_name(f".{path.name}.tmp")
        data = {
            "version": FEED_FILE_VERSION,
            "game_id": feed.game_id,
            "listening": feed.listening,
            "bus_position": feed.bus_position,
            "game": feed.game,
        }
        try:
            self._feed_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(data, f)
            tmp_path.replace(path)
        except OSError as exc:
            logger.warning("Failed to save game feed %s: %s", path, exc)
