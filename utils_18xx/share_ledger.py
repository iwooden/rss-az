"""Non-fungible 18xx share-id tracking for live action submission/replay."""

from __future__ import annotations

from core.data import COMPANY_NAME_TO_ID, CORP_NAME_TO_ID
from entities.company import COMPANIES
from entities.corp import CORPS


def build_share_ownership(
    game_data: dict,
    committed_ids: set,
    *,
    stop_before_id: int | None = None,
) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]]]:
    """Build share-pool and per-player ownership maps from action history."""
    share_pool, players, _ = build_share_ledger(
        game_data,
        committed_ids,
        stop_before_id=stop_before_id,
    )
    return share_pool, players


def build_share_ledger(
    game_data: dict,
    committed_ids: set,
    *,
    stop_before_id: int | None = None,
) -> tuple[dict[str, list[int]], dict[str, dict[str, list[int]]], dict[str, list[int]]]:
    """Return ``(share_pool, player_shares, treasury_shares)`` by share id."""
    share_pool: dict[str, list[int]] = {}
    players: dict[str, dict[str, list[int]]] = {}
    treasury: dict[str, list[int]] = {}
    presidents: dict[str, str] = {}
    player_order = [str(p["id"]) for p in game_data.get("players", [])]

    for player_id in player_order:
        players[player_id] = {}

    for action in game_data.get("actions", []):
        action_id = action.get("id")
        if (
            stop_before_id is not None
            and action_id is not None
            and int(action_id) >= stop_before_id
        ):
            break
        if action_id is not None and action_id not in committed_ids:
            continue

        atype = action.get("type")
        if atype == "par":
            corp = action["corporation"]
            _ensure_treasury(treasury, corp)
            if _is_market_owned_corporation(players, share_pool, corp):
                _return_market_shares_to_treasury(share_pool, treasury, corp)
            num_to_buy = _ipo_player_share_count(action)
            player_bought = _take_shares(share_pool, corp, num_to_buy)
            if len(player_bought) < num_to_buy:
                player_bought.extend(
                    _take_shares(treasury, corp, num_to_buy - len(player_bought))
                )
            moved_to_pool = _take_shares(treasury, corp, num_to_buy)

            user_id = action.get("user")
            if user_id is not None:
                player_key = str(user_id)
                player_shares = players.setdefault(player_key, {}).setdefault(
                    corp,
                    [],
                )
                player_shares.extend(player_bought)
                presidents[corp] = player_key
                _sort_player_shares(players, corp)
            share_pool.setdefault(corp, []).extend(moved_to_pool)
            _sort_pool(share_pool, corp)

        elif atype == "buy_shares":
            user_id = action.get("entity")
            if not isinstance(user_id, int):
                continue
            player_key = str(user_id)
            for share_ref in action.get("shares", []):
                corp, idx = parse_share_id(share_ref)
                if corp and idx is not None:
                    _remove_share(share_pool, corp, idx)
                    _remove_share(treasury, corp, idx)
                    _remove_share_from_any_player(players, corp, idx)
                    _append_share_to_player(players, player_key, corp, idx)
                    _recalculate_president(
                        players,
                        share_pool,
                        treasury,
                        presidents,
                        player_order,
                        corp,
                    )

        elif atype == "sell_shares":
            entity = action.get("entity")
            entity_type = action.get("entity_type")

            if entity_type == "player" and isinstance(entity, int):
                for share_ref in action.get("shares", []):
                    corp, idx = parse_share_id(share_ref)
                    if corp and idx is not None:
                        owner = find_share_owner(players, corp, idx) or str(entity)
                        p_shares = players.get(owner, {}).get(corp, [])
                        if idx in p_shares:
                            p_shares.remove(idx)
                        _append_share(share_pool, corp, idx)
                        _sort_player_shares(players, corp)
                        _recalculate_president(
                            players,
                            share_pool,
                            treasury,
                            presidents,
                            player_order,
                            corp,
                        )
            elif entity_type == "corporation" and isinstance(entity, str):
                for share_ref in action.get("shares", []):
                    corp, idx = parse_share_id(share_ref)
                    if corp and idx is not None:
                        _ensure_treasury(treasury, corp)
                        _remove_share(treasury, corp, idx)
                        _append_share(share_pool, corp, idx)

    return share_pool, players, treasury


def resolve_buyable_share(
    game_data: dict,
    corp_name: str,
    committed_ids: set,
    *,
    market_share_count: int | None = None,
    treasury_share_count: int | None = None,
) -> str:
    """Find a share available in the share pool, preferring non-president shares."""
    share_pool, _, treasury = build_share_ledger(game_data, committed_ids)
    _reconcile_share_counts(
        share_pool,
        treasury,
        corp_name,
        market_share_count=market_share_count,
        treasury_share_count=treasury_share_count,
    )
    available = sorted(share_pool.get(corp_name, []))
    if not available:
        raise ValueError(f"No shares of {corp_name} available in share pool")
    idx = next((share_idx for share_idx in available if share_idx > 0), available[0])
    return f"{corp_name}_{idx}"


def resolve_sellable_share(
    game_data: dict, corp_name: str, user_id: int, committed_ids: set,
) -> str:
    """Find a share the player can sell, preferring non-president shares."""
    _, player_shares = build_share_ownership(game_data, committed_ids)
    held = sorted(
        player_shares.get(str(user_id), {}).get(corp_name, []),
        reverse=True,
    )
    if not held:
        raise ValueError(f"User {user_id} holds no shares of {corp_name}")
    non_president = [idx for idx in held if idx > 0]
    idx = non_president[0] if non_president else held[0]
    return f"{corp_name}_{idx}"


def resolve_issuable_share(
    game_data: dict,
    corp_name: str,
    committed_ids: set,
    *,
    market_share_count: int | None = None,
    treasury_share_count: int | None = None,
) -> str:
    """Find the next corporation-held treasury share to issue."""
    share_pool, _, treasury = build_share_ledger(game_data, committed_ids)
    _reconcile_share_counts(
        share_pool,
        treasury,
        corp_name,
        market_share_count=market_share_count,
        treasury_share_count=treasury_share_count,
    )
    available = sorted(treasury.get(corp_name, []))
    if available:
        return f"{corp_name}_{available[0]}"
    raise ValueError(f"No unissued shares of {corp_name}")


def share_owner_before_action(
    game_data: dict,
    committed_ids: set,
    share_ref: str,
    action_id: int,
) -> str | None:
    """Return the player id owning ``share_ref`` immediately before action."""
    corp, idx = parse_share_id(share_ref)
    if corp is None or idx is None:
        return None
    _, players, _ = build_share_ledger(
        game_data,
        committed_ids,
        stop_before_id=action_id,
    )
    return find_share_owner(players, corp, idx)


def parse_share_id(share_ref: str) -> tuple[str | None, int | None]:
    """Parse 'IC_2' into ('IC', 2)."""
    parts = share_ref.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return None, None


def find_share_owner(
    players: dict[str, dict[str, list[int]]],
    corp_name: str,
    idx: int,
) -> str | None:
    """Return player id string for the owner of one share id."""
    for player_id, by_corp in players.items():
        if idx in by_corp.get(corp_name, []):
            return player_id
    return None


def _recalculate_president(
    players: dict[str, dict[str, list[int]]],
    share_pool: dict[str, list[int]],
    treasury: dict[str, list[int]],
    presidents: dict[str, str],
    player_order: list[str],
    corp_name: str,
) -> None:
    counts = {
        player_id: len(by_corp.get(corp_name, []))
        for player_id, by_corp in players.items()
    }
    max_count = max(counts.values(), default=0)
    current = presidents.get(corp_name)

    if max_count <= 0:
        presidents.pop(corp_name, None)
        return

    if current is not None and counts.get(current, 0) >= max_count:
        return

    ordered = player_order or list(players.keys())
    if current in ordered:
        start = ordered.index(current) + 1
        candidates = ordered[start:] + ordered[:start]
    else:
        candidates = ordered

    new_president = next(
        (player_id for player_id in candidates if counts.get(player_id, 0) == max_count),
        None,
    )
    if new_president is None:
        return

    presidents[corp_name] = new_president
    _ensure_president_share(
        players,
        share_pool,
        treasury,
        corp_name,
        new_president,
    )


def _ensure_president_share(
    players: dict[str, dict[str, list[int]]],
    share_pool: dict[str, list[int]],
    treasury: dict[str, list[int]],
    corp_name: str,
    president_id: str,
) -> None:
    if 0 in players.setdefault(president_id, {}).setdefault(corp_name, []):
        return

    current_owner = find_share_owner(players, corp_name, 0)
    source = None
    if current_owner is None:
        if 0 in share_pool.get(corp_name, []):
            source = share_pool
            share_pool[corp_name].remove(0)
        elif 0 in treasury.get(corp_name, []):
            source = treasury
            treasury[corp_name].remove(0)

    replacement = next(
        (
            idx
            for idx in sorted(players[president_id].get(corp_name, []))
            if idx != 0
        ),
        None,
    )

    if current_owner is not None:
        players[current_owner][corp_name].remove(0)
    players[president_id].setdefault(corp_name, []).append(0)

    if replacement is not None:
        players[president_id][corp_name].remove(replacement)
        if current_owner is not None:
            players[current_owner].setdefault(corp_name, []).append(replacement)
        elif source is not None:
            _append_share(source, corp_name, replacement)

    _sort_player_shares(players, corp_name)


def _ensure_treasury(treasury: dict[str, list[int]], corp_name: str) -> None:
    treasury.setdefault(corp_name, list(range(_corp_share_count(corp_name))))


def _is_market_owned_corporation(
    players: dict[str, dict[str, list[int]]],
    share_pool: dict[str, list[int]],
    corp_name: str,
) -> bool:
    if not share_pool.get(corp_name):
        return False
    return all(not by_corp.get(corp_name) for by_corp in players.values())


def _return_market_shares_to_treasury(
    share_pool: dict[str, list[int]],
    treasury: dict[str, list[int]],
    corp_name: str,
) -> None:
    market_shares = list(share_pool.get(corp_name, []))
    share_pool[corp_name] = []
    for idx in market_shares:
        _append_share(treasury, corp_name, idx)


def _reconcile_share_counts(
    share_pool: dict[str, list[int]],
    treasury: dict[str, list[int]],
    corp_name: str,
    *,
    market_share_count: int | None,
    treasury_share_count: int | None,
) -> None:
    """Adjust hidden 18xx share movements not represented as actions."""
    if market_share_count is not None:
        _ensure_treasury(treasury, corp_name)
        while len(share_pool.get(corp_name, [])) < market_share_count:
            moved = _take_shares(treasury, corp_name, 1)
            if not moved:
                break
            _append_share(share_pool, corp_name, moved[0])

        while len(share_pool.get(corp_name, [])) > market_share_count:
            shares = share_pool.get(corp_name, [])
            if not shares:
                break
            _append_share(treasury, corp_name, shares.pop())

    if treasury_share_count is not None:
        _ensure_treasury(treasury, corp_name)
        while len(treasury.get(corp_name, [])) > treasury_share_count:
            moved = _take_shares(treasury, corp_name, 1)
            if not moved:
                break
            _append_share(share_pool, corp_name, moved[0])


def _corp_share_count(corp_name: str) -> int:
    return CORPS[CORP_NAME_TO_ID[corp_name]].get_total_shares()


def _ipo_player_share_count(action: dict) -> int:
    par_price = _parse_share_price(action["share_price"])
    company_name = action.get("entity") or action.get("company")
    face_value = COMPANIES[COMPANY_NAME_TO_ID[company_name]].get_face_value()
    for count in range(1, 5):
        if par_price * count >= face_value:
            return count
    return 4


def _parse_share_price(value) -> int:
    return int(str(value).split(",", 1)[0])


def _append_share(container: dict[str, list[int]], corp_name: str, idx: int) -> None:
    shares = container.setdefault(corp_name, [])
    if idx not in shares:
        shares.append(idx)
    _sort_pool(container, corp_name)


def _append_share_to_player(
    players: dict[str, dict[str, list[int]]],
    player_id: str,
    corp_name: str,
    idx: int,
) -> None:
    shares = players.setdefault(player_id, {}).setdefault(corp_name, [])
    if idx not in shares:
        shares.append(idx)
    shares.sort()


def _remove_share(container: dict[str, list[int]], corp_name: str, idx: int) -> None:
    shares = container.get(corp_name)
    if shares and idx in shares:
        shares.remove(idx)


def _take_shares(
    container: dict[str, list[int]],
    corp_name: str,
    count: int,
) -> list[int]:
    shares = container.setdefault(corp_name, [])
    taken = shares[:count]
    del shares[:len(taken)]
    return taken


def _remove_share_from_any_player(
    players: dict[str, dict[str, list[int]]],
    corp_name: str,
    idx: int,
) -> None:
    owner = find_share_owner(players, corp_name, idx)
    if owner is not None:
        players[owner][corp_name].remove(idx)


def _sort_player_shares(
    players: dict[str, dict[str, list[int]]],
    corp_name: str,
) -> None:
    for by_corp in players.values():
        by_corp.get(corp_name, []).sort()


def _sort_pool(container: dict[str, list[int]], corp_name: str) -> None:
    container.get(corp_name, []).sort()
