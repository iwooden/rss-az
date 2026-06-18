import queue

from utils_18xx import live
from utils_18xx.live import MoveWorker


class _FakeApi:
    def __init__(self):
        self.fetches = 0
        self.posts = []
        self.updated_at = None

    def fetch_game(self, game_id, token):
        self.fetches += 1
        if self.fetches == 1:
            actions = [{"id": 1}]
            acting = [1]
            self.updated_at = 10
        elif self.fetches == 2:
            actions = [{"id": 1}, {"id": 2}]
            acting = [1]
            self.updated_at = 20
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

    def fetch_game_summary(self, game_id, token):
        return {"id": game_id, "updated_at": self.updated_at}

    def post_action(self, game_id, action, token):
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


class _StaleOnceApi:
    def __init__(self):
        self.fetches = 0
        self.summary_fetches = 0
        self.posts = []
        self.updated_at = 0

    def fetch_game(self, game_id, token):
        del token
        self.fetches += 1
        if self.fetches == 1:
            self.updated_at = 10
            acting = [1]
        elif self.fetches == 2:
            self.updated_at = 11
            acting = [1]
        else:
            self.updated_at = 12
            acting = []
        return {
            "id": game_id,
            "players": [{"id": 1, "name": "bot"}],
            "acting": acting,
            "actions": [],
            "updated_at": self.updated_at,
        }

    def fetch_game_summary(self, game_id, token):
        del token
        self.summary_fetches += 1
        if self.summary_fetches == 1:
            return {"id": game_id, "updated_at": 11}
        return {"id": game_id, "updated_at": self.updated_at}

    def post_action(self, game_id, action, token):
        del game_id, token
        self.posts.append(action)
        return {}


class _AlwaysStaleApi:
    def __init__(self):
        self.fetches = 0
        self.summary_fetches = 0
        self.posts = []
        self.updated_at = 0

    def fetch_game(self, game_id, token):
        del token
        self.fetches += 1
        self.updated_at += 1
        return {
            "id": game_id,
            "players": [{"id": 1, "name": "bot"}],
            "acting": [1],
            "actions": [],
            "updated_at": self.updated_at,
        }

    def fetch_game_summary(self, game_id, token):
        del token
        self.summary_fetches += 1
        return {"id": game_id, "updated_at": self.updated_at + 1}

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


def test_worker_refetches_full_game_between_batched_posts(monkeypatch):
    api = _FakeApi()
    seen_action_counts = []

    def fake_attach(game_data, action):
        seen_action_counts.append(len(game_data["actions"]))
        return action

    monkeypatch.setattr(live, "attach_expected_auto_actions", fake_attach)
    monkeypatch.setattr(live.time, "sleep", lambda seconds: None)

    worker = MoveWorker(
        queue.Queue(),
        api,
        {"bot": {"token": "token", "user_id": 1}},
        _FakeRegistry(),
    )

    worker._process("bot", "1")

    assert seen_action_counts == [1, 2]
    assert len(api.posts) == 2


def test_worker_replans_when_game_summary_changes_before_post(monkeypatch):
    api = _StaleOnceApi()
    engine = _SingleActionEngine()

    monkeypatch.setattr(live, "attach_expected_auto_actions", lambda game_data, action: action)
    monkeypatch.setattr(live.time, "sleep", lambda seconds: None)

    worker = MoveWorker(
        queue.Queue(),
        api,
        {"bot": {"token": "token", "user_id": 1}},
        _FakeRegistry(engine),
    )

    worker._process("bot", "1")

    assert engine.calls == 2
    assert api.summary_fetches == 2
    assert api.posts == [{"type": "pass", "attempt": 2}]


def test_worker_stops_after_repeated_stale_replans(monkeypatch):
    api = _AlwaysStaleApi()
    engine = _SingleActionEngine()

    monkeypatch.setattr(live, "attach_expected_auto_actions", lambda game_data, action: action)

    worker = MoveWorker(
        queue.Queue(),
        api,
        {"bot": {"token": "token", "user_id": 1}},
        _FakeRegistry(engine),
    )

    worker._process("bot", "1")

    assert engine.calls == live.STALE_MOVE_RETRY_LIMIT + 1
    assert api.fetches == live.STALE_MOVE_RETRY_LIMIT + 1
    assert api.summary_fetches == live.STALE_MOVE_RETRY_LIMIT + 1
    assert api.posts == []


def test_worker_lets_replay_check_stale_top_level_acting():
    engine = _RecordingEngine()
    worker = MoveWorker(
        queue.Queue(),
        _StaleActingApi(),
        {"bot": {"token": "token", "user_id": 1}},
        _RecordingRegistry(engine),
    )

    worker._process("bot", "1")

    assert engine.calls == 1
