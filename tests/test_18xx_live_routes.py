from utils_18xx.live import (
    is_local_request_host,
    is_turn_webhook_text,
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


def test_poke_endpoint_host_check_allows_only_loopback():
    assert is_local_request_host("127.0.0.1")
    assert is_local_request_host("::1")
    assert is_local_request_host("localhost")
    assert not is_local_request_host("192.168.1.10")
    assert not is_local_request_host("203.0.113.7")
