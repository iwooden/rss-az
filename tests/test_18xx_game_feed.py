import copy
import json
import queue
import shutil
import threading
from pathlib import Path
from typing import cast

import pytest

from utils_18xx import game_feed
from utils_18xx.api_client import ApiClient, TransientError
from utils_18xx.game_feed import GameFeedManager, compute_game_status

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "submodules/18xx/public/fixtures/RollingStockStars/83092.json"
)
CHANNEL = "/game/42"
BOT = {"bot": {"token": "bot-token", "user_id": 1}}


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _FakeBus:
    def __init__(self):
        self.subscriptions: dict[str, int] = {}
        self.calls: list[tuple] = []

    def subscribe(self, channel, position=-1):
        self.subscriptions[channel] = position
        self.calls.append(("subscribe", channel, position))

    def unsubscribe(self, channel):
        self.subscriptions.pop(channel, None)
        self.calls.append(("unsubscribe", channel))

    def start(self):
        pass

    def stop(self):
        pass


class _FakeApi:
    def __init__(self, game):
        self.game = game
        self.fetches: list[tuple] = []
        self.fail = False

    def fetch_game(self, game_id, token):
        self.fetches.append((str(game_id), token))
        if self.fail:
            raise TransientError("down")
        return copy.deepcopy(self.game)


class _Blacklist:
    def __init__(self, game_ids):
        self.game_ids = set(game_ids)

    def contains(self, game_id):
        return str(game_id) in self.game_ids


def _action(action_id, acting_after, *, user=2, finishes=False):
    """A stored (GET-shaped) action; ``_deliver`` sends the bus shape."""
    return {
        "type": "pass",
        "entity": user,
        "entity_type": "player",
        "id": action_id,
        "user": user,
        "created_at": 0,
        "acting_after": acting_after,
        "finishes": finishes,
    }


def _game(actions):
    return {
        "id": 42,
        "title": "Rolling Stock Stars",
        "players": [{"id": 1, "name": "bot"}, {"id": 2, "name": "human"}],
        "actions": actions,
        "acting": actions[-1]["acting_after"],
        "round": "Investment",
        "turn": 1,
        "status": "active",
        "result": {},
    }


def _fake_status(game, from_count=None):
    """Stand-in for game_status.rb: actions carry their resulting acting set,
    and the stored user is the (player) entity."""
    actions = game["actions"]
    counts = (
        range(from_count, len(actions) + 1)
        if from_count is not None
        else [len(actions)]
    )
    states = []
    for count in counts:
        last = actions[count - 1]
        state = {
            "action_count": count,
            "round": "Investment",
            "turn": 1 + count,
            "acting": last["acting_after"],
            "finished": last["finishes"],
            "result": {"1": 10, "2": 5} if last["finishes"] else {},
        }
        if from_count is not None and count > from_count:
            state["user"] = last.get("user", last["entity"])
        states.append(state)
    return states


def _manager(tmp_path, api, *, auth=None, clock=None, blacklist=None,
             status_fn=_fake_status):
    bus = _FakeBus()
    work: queue.Queue = queue.Queue()
    manager = GameFeedManager(
        cast(ApiClient, api),
        auth or BOT,
        tmp_path / "feeds",
        work,
        bus=bus,
        game_blacklist=blacklist,
        status_fn=status_fn,
        clock=clock or _Clock(),
    )
    return manager, bus, work


def _bootstrap(manager, channel=CHANNEL, position=10):
    manager.listen(channel.rsplit("/", 1)[1])
    manager.pump()
    manager.on_bus_status(channel, position, True)
    manager.on_bus_synced({channel})
    manager.pump()


def _deliver(manager, position, action, channel=CHANNEL):
    """Publish like 18xx.games: no stored ``user``, plus ``_client_id``."""
    data = {key: value for key, value in action.items() if key != "user"}
    manager.on_bus_message(channel, position, dict(data, _client_id=None))
    manager.pump()


def _drain(work, manager=None):
    """Take queued runs; like the move worker, mark them started."""
    items = []
    while not work.empty():
        bot_name, game_id = work.get_nowait()
        if manager is not None:
            manager.claim_turn(bot_name, game_id)
        items.append((bot_name, game_id))
    return items


def _feed_ids(manager, game_id="42"):
    game = manager.snapshot(game_id, timeout=0)
    return [action["id"] for action in game["actions"]]


def test_listen_subscribes_before_downloading_and_queues_acting_bot(tmp_path):
    api = _FakeApi(_game([_action(1, [1])]))
    manager, bus, work = _manager(tmp_path, api)

    manager.listen("42")
    manager.pump()

    # The download waits for the bus position so no action can slip between.
    assert bus.calls == [("subscribe", CHANNEL, -1)]
    assert api.fetches == []

    manager.on_bus_status(CHANNEL, 7, True)
    manager.on_bus_synced({CHANNEL})
    manager.pump()

    assert api.fetches == [("42", "bot-token")]
    assert _drain(work, manager) == [("bot", "42")]
    [status] = manager.status()
    assert status["state"] == "ready"
    assert status["bus_position"] == 7


def test_bus_actions_update_game_and_queue_turn_edges(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, work = _manager(tmp_path, api)
    _bootstrap(manager)
    assert _drain(work, manager) == []

    _deliver(manager, 11, _action(2, [1]))
    game = manager.snapshot("42", timeout=0)
    assert game is not None
    assert game["actions"] == [_action(1, [2]), _action(2, [1])]
    assert game["acting"] == [1]
    assert game["turn"] == 3
    assert _drain(work, manager) == [("bot", "42")]

    # Still acting after its own action: the move worker's loop continues.
    _deliver(manager, 12, _action(3, [1], user=1))
    assert _drain(work, manager) == []

    # Duplicates are ignored.
    _deliver(manager, 13, _action(3, [1], user=1))
    assert _feed_ids(manager) == [1, 2, 3]
    assert api.fetches == [("42", "bot-token")]


def test_simultaneous_round_requeues_acting_bot_after_others_act(tmp_path):
    # Acquisition/Closing: everyone acts at once, so an answer to the bot's
    # offer leaves it acting without any "newly acting" transition.
    api = _FakeApi(_game([_action(1, [1, 2])]))
    manager, _, work = _manager(tmp_path, api)
    _bootstrap(manager)
    assert _drain(work, manager) == [("bot", "42")]

    _deliver(manager, 11, _action(2, [1, 2], user=1))  # the bot's own offer
    assert _drain(work, manager) == []

    _deliver(manager, 12, _action(3, [1, 2], user=2))  # the other player
    assert _drain(work, manager) == [("bot", "42")]

    chat = dict(_action(4, [1, 2], user=2), type="message", message="hi")
    _deliver(manager, 13, chat)
    assert _drain(work, manager) == []


def test_waiting_run_is_not_queued_again(tmp_path):
    api = _FakeApi(_game([_action(1, [1, 2])]))
    manager, _, work = _manager(tmp_path, api)
    _bootstrap(manager)
    _deliver(manager, 11, _action(2, [1, 2]))
    _deliver(manager, 12, _action(3, [1, 2]))
    # One waiting run covers both actions: it reads the latest data.
    assert _drain(work, manager) == [("bot", "42")]

    _deliver(manager, 13, _action(4, [1, 2]))
    assert _drain(work, manager) == [("bot", "42")]


def test_out_of_order_actions_fill_in_without_download(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, work = _manager(tmp_path, api)
    _bootstrap(manager)

    _deliver(manager, 12, _action(3, [2]))
    assert manager.snapshot("42", timeout=0) is None  # gap pending
    [status] = manager.status()
    assert status["state"] == "gap"

    # The bot was acting only between the two actions; the per-action
    # history still reports the turn.
    _deliver(manager, 11, _action(2, [1]))
    assert _feed_ids(manager) == [1, 2, 3]
    assert _drain(work, manager) == [("bot", "42")]
    assert len(api.fetches) == 1


def test_unfilled_gap_downloads_after_grace(tmp_path):
    clock = _Clock()
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, work = _manager(tmp_path, api, clock=clock)
    _bootstrap(manager)
    clock.now += game_feed.RESYNC_COOLDOWN_SECS

    # Action 2 was lost (one-message backlog); 3 arrives.
    _deliver(manager, 12, _action(3, [2]))
    clock.now += game_feed.GAP_GRACE_SECS - 1
    manager.pump()
    assert len(api.fetches) == 1

    api.game = _game([_action(1, [2]), _action(2, [2]), _action(3, [1])])
    clock.now += 1
    manager.pump()

    assert len(api.fetches) == 2
    assert _feed_ids(manager) == [1, 2, 3]
    assert _drain(work, manager) == [("bot", "42")]


def test_download_keeps_newer_bus_actions_and_respects_cooldown(tmp_path):
    clock = _Clock()
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, _ = _manager(tmp_path, api, clock=clock)
    _bootstrap(manager)
    clock.now += game_feed.RESYNC_COOLDOWN_SECS

    _deliver(manager, 12, _action(3, [2]))
    # A later action arrives before the gap download; it survives it.
    _deliver(manager, 13, _action(4, [2]))
    api.game = _game([_action(1, [2]), _action(2, [2]), _action(3, [2])])
    clock.now += game_feed.GAP_GRACE_SECS
    manager.pump()
    assert _feed_ids(manager) == [1, 2, 3, 4]

    # Another gap right away waits for the download cooldown.
    _deliver(manager, 15, _action(6, [2]))
    clock.now += game_feed.GAP_GRACE_SECS
    manager.pump()
    assert len(api.fetches) == 2
    api.game = _game([_action(i, [2]) for i in range(1, 7)])
    clock.now += game_feed.RESYNC_COOLDOWN_SECS
    manager.pump()
    assert len(api.fetches) == 3
    assert _feed_ids(manager) == [1, 2, 3, 4, 5, 6]


def test_feed_resumes_from_disk_without_download(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, _ = _manager(tmp_path, api)
    _bootstrap(manager, position=10)
    _deliver(manager, 11, _action(2, [1]))

    saved = json.loads((tmp_path / "feeds" / "42.json").read_text())
    assert saved["bus_position"] == 11
    assert saved["listening"] is True

    restarted, bus, work = _manager(tmp_path, api)
    restarted.load_persisted()
    assert bus.calls == [("subscribe", CHANNEL, 11)]

    # First poll: nothing new on the channel.
    restarted.on_bus_synced({CHANNEL})
    restarted.pump()

    assert len(api.fetches) == 1
    assert _feed_ids(restarted) == [1, 2]
    # The bot was mid-turn when the server stopped.
    assert _drain(work, restarted) == [("bot", "42")]


def test_feed_resume_applies_backlog_or_downloads_on_gap(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, _ = _manager(tmp_path, api)
    _bootstrap(manager, position=10)

    # One action while stopped: the one-message backlog delivers it.
    restarted, _, work = _manager(tmp_path, api)
    restarted.load_persisted()
    _deliver(restarted, 11, _action(2, [1]))
    restarted.on_bus_synced({CHANNEL})
    restarted.pump()
    assert _feed_ids(restarted) == [1, 2]
    assert _drain(work, restarted) == [("bot", "42")]
    assert len(api.fetches) == 1

    # Several actions while stopped: only the latest survives, so download.
    clock = _Clock()
    again, _, _ = _manager(tmp_path, api, clock=clock)
    again.load_persisted()
    _deliver(again, 14, _action(5, [2]))
    again.on_bus_synced({CHANNEL})
    again.pump()
    api.game = _game([_action(i, [2]) for i in range(1, 6)])
    clock.now += game_feed.GAP_GRACE_SECS
    again.pump()
    assert len(api.fetches) == 2
    assert _feed_ids(again) == [1, 2, 3, 4, 5]


def test_channel_restart_downloads(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, _ = _manager(tmp_path, api)
    _bootstrap(manager, position=10)

    # Channel expired and restarted with one new message we cannot replay.
    api.game = _game([_action(1, [2]), _action(2, [2])])
    manager.on_bus_status(CHANNEL, 1, False)
    manager.pump()

    assert len(api.fetches) == 2
    assert _feed_ids(manager) == [1, 2]


def test_finished_game_stops_listening(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, bus, work = _manager(tmp_path, api)
    _bootstrap(manager)

    _deliver(manager, 11, _action(2, [1, 2], finishes=True))

    assert bus.calls[-1] == ("unsubscribe", CHANNEL)
    assert not manager.is_listening("42")
    assert _drain(work, manager) == []
    saved = json.loads((tmp_path / "feeds" / "42.json").read_text())
    assert saved["listening"] is False
    assert saved["game"]["status"] == "finished"
    assert saved["game"]["result"] == {"1": 10, "2": 5}


def test_blacklisted_game_is_tracked_but_not_queued(tmp_path):
    api = _FakeApi(_game([_action(1, [1])]))
    manager, _, work = _manager(
        tmp_path, api, blacklist=_Blacklist({"42"}),
    )
    _bootstrap(manager)
    _deliver(manager, 11, _action(2, [2]))
    _deliver(manager, 12, _action(3, [1]))

    assert _feed_ids(manager) == [1, 2, 3]
    assert _drain(work, manager) == []


def test_webhook_for_new_game_listens_and_queues_after_download(tmp_path):
    # Server acting disagrees with the webhook; the worker's replay decides.
    api = _FakeApi(_game([_action(1, [2])]))
    manager, bus, work = _manager(tmp_path, api)

    manager.notify_webhook("bot", "42")
    manager.pump()
    assert bus.calls == [("subscribe", CHANNEL, -1)]
    assert _drain(work, manager) == []

    manager.on_bus_status(CHANNEL, 3, True)
    manager.on_bus_synced({CHANNEL})
    manager.pump()
    assert _drain(work, manager) == [("bot", "42")]


def test_webhook_matching_bus_turn_is_not_requeued(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, work = _manager(tmp_path, api)
    _bootstrap(manager)
    _deliver(manager, 11, _action(2, [1]))
    assert _drain(work, manager) == [("bot", "42")]

    manager.notify_webhook("bot", "42")
    manager.pump()
    assert _drain(work, manager) == []
    assert len(api.fetches) == 1


def test_webhook_ahead_of_bus_waits_then_downloads(tmp_path):
    clock = _Clock()
    api = _FakeApi(_game([_action(1, [2])]))
    manager, _, work = _manager(tmp_path, api, clock=clock)
    _bootstrap(manager)

    manager.notify_webhook("bot", "42")
    manager.pump()
    assert _drain(work, manager) == []

    # The bus catches up within the grace period: one queue entry, no download.
    _deliver(manager, 11, _action(2, [1]))
    assert _drain(work, manager) == [("bot", "42")]
    clock.now += game_feed.WEBHOOK_GRACE_SECS
    manager.pump()
    assert _drain(work, manager) == []
    assert len(api.fetches) == 1

    # Later turn whose action never reaches us: download after the grace.
    clock.now += game_feed.WEBHOOK_MATCH_SECS + 1
    _deliver(manager, 12, _action(3, [2]))
    manager.notify_webhook("bot", "42")
    manager.pump()
    api.game = _game([_action(1, [2]), _action(2, [1]), _action(3, [2]),
                      _action(4, [1])])
    clock.now += game_feed.WEBHOOK_GRACE_SECS
    manager.pump()
    assert len(api.fetches) == 2
    assert _feed_ids(manager) == [1, 2, 3, 4]
    assert _drain(work, manager) == [("bot", "42")]


def test_failed_listen_download_retries(tmp_path):
    clock = _Clock()
    api = _FakeApi(_game([_action(1, [1])]))
    api.fail = True
    manager, _, work = _manager(tmp_path, api, clock=clock)
    _bootstrap(manager)
    assert manager.status()[0]["state"] == "downloading"

    api.fail = False
    clock.now += game_feed.DOWNLOAD_RETRY_SECS
    manager.pump()
    assert manager.status()[0]["state"] == "ready"
    assert _drain(work, manager) == [("bot", "42")]


def test_unlisten_and_relisten(tmp_path):
    api = _FakeApi(_game([_action(1, [2])]))
    manager, bus, _ = _manager(tmp_path, api)
    _bootstrap(manager)

    manager.unlisten("42")
    manager.pump()
    assert bus.calls[-1] == ("unsubscribe", CHANNEL)
    assert manager.snapshot("42", timeout=0) is None

    manager.listen("42")
    manager.pump()
    assert bus.calls[-1] == ("subscribe", CHANNEL, 10)


def test_wait_for_posted_actions_returns_after_echo(tmp_path):
    api = _FakeApi(_game([_action(1, [1])]))
    manager, _, _ = _manager(tmp_path, api)
    _bootstrap(manager)

    # Another player's action arrives first; it does not count as ours.
    _deliver(manager, 11, _action(2, [1], user=2))
    posted = threading.Thread(
        target=lambda: (
            _deliver(manager, 12, _action(3, [2], user=1))
        ),
    )
    posted.start()
    game = manager.wait_for_posted_actions("42", 1, after_id=1, count=1, timeout=5)
    posted.join()

    assert game is not None
    assert [action["id"] for action in game["actions"]] == [1, 2, 3]


def test_wait_for_posted_actions_downloads_when_echo_is_lost(tmp_path):
    api = _FakeApi(_game([_action(1, [1])]))
    manager, _, _ = _manager(tmp_path, api, clock=_Clock())
    _bootstrap(manager)
    api.game = _game([_action(1, [1]), _action(2, [2], user=1)])

    stop = threading.Event()

    def pump_until_stopped():
        while not stop.is_set():
            manager.pump()
            stop.wait(0.01)

    pumper = threading.Thread(target=pump_until_stopped)
    pumper.start()
    try:
        game = manager.wait_for_posted_actions(
            "42", 1, after_id=1, count=1, timeout=0.05,
        )
    finally:
        stop.set()
        pumper.join()

    assert game is not None
    assert [action["id"] for action in game["actions"]] == [1, 2]
    assert len(api.fetches) == 2


@pytest.mark.skipif(shutil.which("ruby") is None, reason="requires ruby")
def test_bus_replay_matches_18xx_server_fields(tmp_path):
    """Replaying a real RSS game's tail from the bus reproduces the server's
    stored fields and queues bots exactly when they newly become active."""
    fixture = json.loads(FIXTURE.read_text())
    bootstrap_count = 529
    server_game = copy.deepcopy(fixture)
    server_game["actions"] = fixture["actions"][:bootstrap_count]
    state = compute_game_status(server_game)[-1]
    game_feed._apply_status_fields(server_game, state)
    assert server_game["round"] == "Closing"

    api = _FakeApi(server_game)
    auth = {
        "p11742": {"token": "t1", "user_id": 11742},
        "p11808": {"token": "t2", "user_id": 11808},
    }
    manager, _, work = _manager(
        tmp_path, api, auth=auth, status_fn=compute_game_status,
    )
    channel = f"/game/{fixture['id']}"
    _bootstrap(manager, channel=channel, position=100)
    # Both bots are closing companies when we start listening.
    assert _drain(work, manager) == [("p11742", "83092"), ("p11808", "83092")]

    rest = fixture["actions"][bootstrap_count:]
    # In order, with a duplicate and one out-of-order pair (536 before 535).
    deliveries = [rest[0], rest[1], rest[1], rest[2], rest[3], rest[4],
                  rest[6], rest[5], rest[7], rest[8]]
    queued = []
    for position, action in enumerate(deliveries, start=101):
        _deliver(manager, position, action, channel=channel)
        queued.extend(bot for bot, _ in _drain(work, manager))

    # Closing is simultaneous: other players' passes at 530 and 531 leave
    # p11808 acting. Dividends: p11742 acts at 533, p11808 at 534 and 536
    # (537 is p11808's own action). The game finishes at 538.
    assert queued == ["p11808", "p11808", "p11742", "p11808", "p11808"]
    assert len(api.fetches) == 1
    [status] = manager.status()
    assert status["state"] == "finished"

    saved = json.loads((tmp_path / "feeds" / "83092.json").read_text())["game"]
    # Bus copies carry no stored user (nor does this fixture's tail); the feed
    # derives it as the acting entity's player.
    player_ids = {player["id"] for player in fixture["players"]}
    assert len(saved["actions"]) == len(fixture["actions"])
    for action, expected in zip(saved["actions"], fixture["actions"]):
        if "user" in expected:
            assert action == expected
            continue
        assert {k: v for k, v in action.items() if k != "user"} == expected
        if "user" in action:
            assert action["user"] in player_ids
            if action["entity_type"] == "player":
                assert action["user"] == action["entity"]
    assert all("user" in action for action in saved["actions"][bootstrap_count:])
    assert saved["round"] == fixture["round"]
    assert saved["turn"] == fixture["turn"]
    assert saved["status"] == fixture["status"]
    assert saved["result"] == fixture["result"]
    assert set(saved["acting"]) == set(fixture["acting"])
