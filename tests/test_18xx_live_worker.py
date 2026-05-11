import queue

from utils_18xx import live
from utils_18xx.live import MoveWorker


class _FakeApi:
    def __init__(self):
        self.fetches = 0
        self.posts = []

    def fetch_game(self, game_id, token):
        self.fetches += 1
        if self.fetches == 1:
            actions = [{"id": 1}]
            acting = [1]
        elif self.fetches == 2:
            actions = [{"id": 1}, {"id": 2}]
            acting = [1]
        else:
            actions = [{"id": 1}, {"id": 2}, {"id": 3}]
            acting = []
        return {
            "id": game_id,
            "players": [{"id": 1, "name": "bot"}],
            "acting": acting,
            "actions": actions,
        }

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


class _FakeRegistry:
    def get_engine(self, num_players):
        return _FakeEngine()


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
