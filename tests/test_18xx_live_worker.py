from typing import cast

import copy
import json
import queue

from utils_18xx import live
from utils_18xx.api_client import ApiClient, PermanentError
from utils_18xx.game_feed import GameFeedManager
from utils_18xx.live import AcqOfferTracker, EvalRequest, ModelRegistry, MoveWorker


class _FakeApi:
    def __init__(self):
        self.fetches = 0
        self.posts = []
        self.calls = []
        self.updated_at = None

    def fetch_game(self, game_id, token):
        del token
        self.calls.append(("GET", str(game_id)))
        self.fetches += 1
        if self.fetches == 1:
            actions = [{"id": 1}]
            acting = [1]
            self.updated_at = 10
        else:
            actions = [{"id": 1}, {"id": 2}, {"id": 3}]
            acting = []
            self.updated_at = 30
        return {
            "id": game_id,
            "players": [{"id": 1, "name": "bot"}],
            "acting": acting,
            "actions": actions,
            "updated_at": self.updated_at,
        }

    def post_action(self, game_id, action, token):
        del token
        self.calls.append(("POST", str(game_id), action.get("n")))
        self.posts.append(action)
        return {"id": game_id, "players": [{"id": 1, "name": "bot"}], "actions": []}


class _FakeEngine:
    def process_turn(
        self,
        game_data,
        bot_player_idx,
        bot_user_id=None,
        bot_user_ids=None,
    ):
        assert bot_user_ids == {1}
        return [{"type": "pass", "n": 1}, {"type": "pass", "n": 2}]


class _SingleActionEngine:
    def __init__(self):
        self.calls = 0

    def process_turn(
        self,
        game_data,
        bot_player_idx,
        bot_user_id=None,
        bot_user_ids=None,
    ):
        del game_data, bot_player_idx, bot_user_id, bot_user_ids
        self.calls += 1
        return [{"type": "pass", "attempt": self.calls}]


class _FakeRegistry:
    def __init__(self, engine=None):
        self.engine = engine or _FakeEngine()

    def get_engine(self, num_players):
        del num_players
        return self.engine


class _SingleActionApi:
    def __init__(self):
        self.fetches = 0
        self.posts = []

    def fetch_game(self, game_id, token):
        del token
        self.fetches += 1
        if self.fetches == 1:
            acting = [1]
        else:
            acting = []
        return {
            "id": game_id,
            "players": [{"id": 1, "name": "bot"}],
            "acting": acting,
            "actions": [],
            "updated_at": self.fetches,
        }

    def post_action(self, game_id, action, token):
        del game_id, token
        self.posts.append(action)
        return {}


class _StaleActingApi:
    def fetch_game(self, game_id, token):
        return {
            "id": game_id,
            "players": [
                {"id": 1, "name": "bot"},
                {"id": 2, "name": "other"},
            ],
            "acting": [2],
            "actions": [],
        }


class _RecordingEngine:
    def __init__(self):
        self.calls = 0

    def process_turn(
        self,
        game_data,
        bot_player_idx,
        bot_user_id=None,
        bot_user_ids=None,
    ):
        del game_data, bot_player_idx, bot_user_id, bot_user_ids
        self.calls += 1
        return []


class _RecordingRegistry:
    def __init__(self, engine):
        self.engine = engine

    def get_engine(self, num_players):
        del num_players
        return self.engine


class _EvalApi:
    def __init__(self):
        self.fetches = []
        self.posts = []

    def fetch_game(self, game_id, token):
        self.fetches.append((game_id, token))
        return {
            "id": game_id,
            "players": [
                {"id": 1, "name": "bot"},
                {"id": 2, "name": "other"},
            ],
            "acting": [2],
            "actions": [],
        }

    def post_action(self, game_id, action, token):
        self.posts.append((game_id, action, token))
        raise AssertionError("eval requests must not post actions")


class _EvalEngine:
    def __init__(self):
        self.calls = []

    def evaluate_turn(self, game_data, request):
        self.calls.append((game_data, request))
        return True


class _TrackedRejectEngine:
    def __init__(self):
        self.calls = []

    def process_turn(
        self,
        game_data,
        bot_player_idx,
        bot_user_id=None,
        bot_user_ids=None,
        acq_offer_rejection_counts=None,
    ):
        self.calls.append(
            (
                game_data,
                bot_player_idx,
                bot_user_id,
                bot_user_ids,
                acq_offer_rejection_counts,
            )
        )
        return [{
            "type": "respond",
            "entity": bot_user_id,
            "entity_type": "player",
            "corporation": "SI",
            "company": "KME",
            "accept": "false",
            "_acq_offer_rejection_increment": True,
            "_acq_offer_proposer_id": 202,
            "_acq_offer_turn": 3,
        }]


class _TrackedRejectApi:
    def __init__(self, *, fail_post: bool = False):
        self.fetches = 0
        self.posts = []
        self.fail_post = fail_post

    def fetch_game(self, game_id, token):
        del token
        self.fetches += 1
        return {
            "id": game_id,
            "turn": 3,
            "players": [{"id": 1, "name": "bot"}],
            "acting": [1] if self.fetches == 1 else [],
            "actions": [],
        }

    def post_action(self, game_id, action, token):
        del token
        self.posts.append((game_id, action))
        if self.fail_post:
            raise PermanentError("rejected")
        return {}


def test_worker_posts_batched_actions_without_intermediate_fetch(monkeypatch):
    api = _FakeApi()
    seen_action_counts = []

    def fake_attach(game_data, action):
        seen_action_counts.append(len(game_data["actions"]))
        return action

    monkeypatch.setattr(live, "attach_expected_auto_actions", fake_attach)
    monkeypatch.setattr(live.time, "sleep", lambda seconds: None)

    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, api),
        {"bot": {"token": "token", "user_id": 1}},
        cast(ModelRegistry, _FakeRegistry()),
    )

    worker._process("bot", "1")

    assert seen_action_counts == [1, 2]
    assert len(api.posts) == 2
    assert api.fetches == 2
    assert api.calls == [
        ("GET", "1"),
        ("POST", "1", 1),
        ("POST", "1", 2),
        ("GET", "1"),
    ]


def test_worker_posts_without_home_game_freshness_call(monkeypatch):
    api = _SingleActionApi()
    engine = _SingleActionEngine()

    monkeypatch.setattr(live, "attach_expected_auto_actions", lambda game_data, action: action)
    monkeypatch.setattr(live.time, "sleep", lambda seconds: None)

    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, api),
        {"bot": {"token": "token", "user_id": 1}},
        cast(ModelRegistry, _FakeRegistry(engine)),
    )

    worker._process("bot", "1")

    assert engine.calls == 1
    assert api.fetches == 2
    assert api.posts == [{"type": "pass", "attempt": 1}]


def test_worker_records_acq_offer_rejection_after_successful_post(
    tmp_path,
    monkeypatch,
):
    api = _TrackedRejectApi()
    engine = _TrackedRejectEngine()
    tracker = AcqOfferTracker(tmp_path / "acq_offer_tracking.json")
    seen_post_action = []

    def fake_attach(game_data, action):
        del game_data
        assert not any(key.startswith("_") for key in action)
        seen_post_action.append(action)
        return action

    monkeypatch.setattr(live, "attach_expected_auto_actions", fake_attach)
    monkeypatch.setattr(live.time, "sleep", lambda seconds: None)

    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, api),
        {"bot": {"token": "token", "user_id": 1}},
        cast(ModelRegistry, _RecordingRegistry(engine)),
        acq_offer_tracker=tracker,
    )

    worker._process("bot", "256285")

    assert seen_post_action == [{
        "type": "respond",
        "entity": 1,
        "entity_type": "player",
        "corporation": "SI",
        "company": "KME",
        "accept": "false",
    }]
    assert api.posts == [("256285", seen_post_action[0])]
    assert tracker.rejection_counts("256285", 3) == {"202": 1}
    assert engine.calls[0][4] == {}


def test_worker_does_not_record_acq_offer_rejection_after_post_error(
    tmp_path,
    monkeypatch,
):
    api = _TrackedRejectApi(fail_post=True)
    engine = _TrackedRejectEngine()
    tracker = AcqOfferTracker(tmp_path / "acq_offer_tracking.json")

    monkeypatch.setattr(live, "attach_expected_auto_actions", lambda game_data, action: action)

    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, api),
        {"bot": {"token": "token", "user_id": 1}},
        cast(ModelRegistry, _RecordingRegistry(engine)),
        acq_offer_tracker=tracker,
    )

    worker._process("bot", "256285")

    assert len(api.posts) == 1
    assert tracker.rejection_counts("256285", 3) == {}
    assert api.fetches == 1


def test_worker_lets_replay_check_stale_top_level_acting():
    engine = _RecordingEngine()
    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, _StaleActingApi()),
        {"bot": {"token": "token", "user_id": 1}},
        cast(ModelRegistry, _RecordingRegistry(engine)),
    )

    worker._process("bot", "1")

    assert engine.calls == 1


def test_worker_process_eval_fetches_and_evaluates_without_posting():
    api = _EvalApi()
    engine = _EvalEngine()
    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, api),
        {
            "bot": {"token": "token-1", "user_id": 1},
            "analyst": {"token": "token-2", "user_id": 2},
        },
        cast(ModelRegistry, _RecordingRegistry(engine)),
    )
    request = EvalRequest(game_id="254153", player_id="2", bot_name="analyst")

    worker._process_eval(request)

    assert api.fetches == [("254153", "token-2")]
    assert api.posts == []
    assert len(engine.calls) == 1
    assert engine.calls[0][0]["id"] == "254153"
    assert engine.calls[0][1] == request


def test_worker_process_eval_loads_tmp_file_without_fetching(tmp_path, monkeypatch):
    game_data = {
        "id": 256285,
        "title": "Rolling Stock Stars",
        "players": [
            {"id": 1, "name": "bot"},
            {"id": 2, "name": "other"},
        ],
        "acting": [2],
        "actions": [],
    }
    (tmp_path / "game.json").write_text(json.dumps(game_data))
    monkeypatch.setattr(live, "EVAL_FILE_DIR", tmp_path)

    api = _EvalApi()
    engine = _EvalEngine()
    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, api),
        {},
        cast(ModelRegistry, _RecordingRegistry(engine)),
    )
    request = EvalRequest(filename="game.json", player_id="2")

    worker._process_eval(request)

    assert api.fetches == []
    assert api.posts == []
    assert engine.calls == [(game_data, request)]


def test_worker_file_eval_rejects_symlink_outside_eval_dir(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    outside_file = tmp_path / "outside.json"
    outside_file.write_text("{}")
    (eval_dir / "game.json").symlink_to(outside_file)
    monkeypatch.setattr(live, "EVAL_FILE_DIR", eval_dir)

    api = _EvalApi()
    engine = _EvalEngine()
    worker = MoveWorker(
        queue.Queue(),
        cast(ApiClient, api),
        {},
        cast(ModelRegistry, _RecordingRegistry(engine)),
    )

    worker._process_eval(EvalRequest(filename="game.json"))

    assert api.fetches == []
    assert engine.calls == []


class _EchoFeedApi:
    """18xx API whose accepted posts come back through the game feed's bus."""

    def __init__(self):
        self.fetches = 0
        self.posts = []
        self.manager: GameFeedManager | None = None
        self.game = {
            "id": 42,
            "players": [{"id": 1, "name": "bot"}, {"id": 2, "name": "other"}],
            "acting": [1],
            "actions": [{"id": 1, "user": 2, "acting_after": [1]}],
        }
        # Acting after each accepted post: the first leaves the bot acting
        # (it opens the next phase), the second passes the turn on.
        self.acting_after_posts = [[1], [2]]

    def fetch_game(self, game_id, token):
        del game_id, token
        self.fetches += 1
        return copy.deepcopy(self.game)

    def post_action(self, game_id, action, token):
        del token
        assert self.manager is not None
        self.posts.append(action)
        action_id = len(self.posts) + 1
        # Published like 18xx.games: without the stored ``user``.
        echoed = dict(
            action,
            id=action_id,
            entity=1,
            created_at=0,
            acting_after=self.acting_after_posts[len(self.posts) - 1],
            _client_id=None,
        )
        self.manager.on_bus_message(f"/game/{game_id}", 100 + action_id, echoed)
        self.manager.pump()
        return {}


def _acting_status(game, from_count=None):
    actions = game["actions"]
    counts = range(from_count, len(actions) + 1) if from_count is not None else [
        len(actions)
    ]
    return [
        {
            "action_count": count,
            "round": "Investment",
            "turn": 1,
            "acting": actions[count - 1]["acting_after"],
            "finished": False,
            "result": {},
            "user": actions[count - 1].get("entity"),
        }
        for count in counts
    ]


class _PhaseEngine:
    def __init__(self):
        self.seen_action_ids = []

    def process_turn(self, game_data, bot_player_idx, bot_user_id=None,
                     bot_user_ids=None):
        del bot_player_idx, bot_user_id, bot_user_ids
        self.seen_action_ids.append([a["id"] for a in game_data["actions"]])
        if len(self.seen_action_ids) <= 2:
            return [{"type": "pass", "n": len(self.seen_action_ids)}]
        return []


def _listened_feed(tmp_path, api, work_queue):
    manager = GameFeedManager(
        cast(ApiClient, api),
        {"bot": {"token": "token", "user_id": 1}},
        tmp_path / "feeds",
        work_queue,
        bus=_NullBus(),
        status_fn=_acting_status,
    )
    api.manager = manager
    manager.listen("42")
    manager.pump()
    manager.on_bus_status("/game/42", 100, True)
    manager.on_bus_synced({"/game/42"})
    manager.pump()
    return manager


class _NullBus:
    def subscribe(self, channel, position=-1):
        pass

    def unsubscribe(self, channel):
        pass


def test_worker_uses_feed_and_waits_for_posted_actions(tmp_path, monkeypatch):
    api = _EchoFeedApi()
    engine = _PhaseEngine()
    work_queue = queue.Queue()
    manager = _listened_feed(tmp_path, api, work_queue)
    assert api.fetches == 1

    monkeypatch.setattr(live, "attach_expected_auto_actions", lambda game_data, action: action)

    def no_sleep(seconds):
        raise AssertionError("feed-backed turns must not sleep and re-fetch")

    monkeypatch.setattr(live.time, "sleep", no_sleep)

    worker = MoveWorker(
        work_queue,
        cast(ApiClient, api),
        {"bot": {"token": "token", "user_id": 1}},
        cast(ModelRegistry, _FakeRegistry(engine)),
        game_feeds=manager,
    )
    worker._process("bot", "42")

    # Each decision sees the previous post; the bot kept acting after its
    # first post (no webhook would arrive) and was re-checked from the feed.
    assert engine.seen_action_ids == [[1], [1, 2], [1, 2, 3]]
    assert [post["n"] for post in api.posts] == [1, 2]
    assert api.fetches == 1


def test_worker_eval_reads_listened_game_from_feed(tmp_path):
    api = _EchoFeedApi()
    work_queue = queue.Queue()
    manager = _listened_feed(tmp_path, api, work_queue)
    engine = _EvalEngine()
    worker = MoveWorker(
        work_queue,
        cast(ApiClient, api),
        {"bot": {"token": "token", "user_id": 1}},
        cast(ModelRegistry, _RecordingRegistry(engine)),
        game_feeds=manager,
    )

    worker._process_eval(EvalRequest(game_id="42"))

    assert api.fetches == 1
    assert engine.calls[0][0]["actions"] == api.game["actions"]
