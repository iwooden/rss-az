"""HTTP client for the 18xx.games REST API.

Handles game fetching, action posting, and auth via session cookies.
Distinguishes transient errors (retriable) from permanent ones.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class TransientError(Exception):
    """Retriable error (timeout, 5xx, connection refused)."""


class PermanentError(Exception):
    """Non-retriable error (4xx, invalid action)."""


class ApiClient:
    """Thin HTTP client for the 18xx.games REST API."""

    def __init__(self, base_url: str, request_timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.request_timeout = request_timeout

    def fetch_game(self, game_id: int | str, token: str) -> dict:
        """GET /api/game/:id — fetch full game data including actions."""
        return self._request("GET", f"/api/game/{game_id}", token)

    def fetch_game_summary(self, game_id: int | str, token: str) -> dict | None:
        """Fetch one no-actions game summary from the home-game API."""
        cache_buster = int(time.time() * 1000)
        response = self._request("GET", f"/api/game?_fresh={cache_buster}", token)
        for game in response.get("games", []):
            if str(game.get("id")) == str(game_id):
                return game
        return None

    def post_action(
        self, game_id: int | str, action: dict, token: str,
    ) -> dict:
        """POST /api/game/:id/action — submit a game action."""
        return self._request(
            "POST", f"/api/game/{game_id}/action", token, data=action,
        )

    def _request(
        self,
        method: str,
        path: str,
        token: str,
        data: dict | None = None,
        retries: int = 3,
        backoff: float = 1.0,
    ) -> dict:
        """Make an HTTP request with retries on transient errors."""
        url = self.base_url + path
        headers = {
            "Cookie": f"auth_token={token}",
            "Accept": "application/json",
        }
        body = None
        if data is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(data).encode()

        last_error: Exception | None = None
        for attempt in range(retries):
            if attempt > 0:
                delay = backoff * (2 ** (attempt - 1))
                logger.info(f"Retry {attempt}/{retries} after {delay:.1f}s")
                time.sleep(delay)

            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method=method,
                )
                with urllib.request.urlopen(
                    req, timeout=self.request_timeout,
                ) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                status = e.code
                try:
                    resp_body = e.read().decode()
                except Exception:
                    resp_body = ""

                if 500 <= status < 600:
                    # 500s with "Illegal action" are permanent — the
                    # 18xx engine rejected our move, retrying won't help.
                    if "Illegal action" in resp_body:
                        raise PermanentError(
                            f"HTTP {status}: {resp_body[:500]}"
                        )
                    last_error = TransientError(
                        f"HTTP {status}: {resp_body[:200]}"
                    )
                    logger.warning(
                        f"Transient error on {method} {path}: {last_error}"
                    )
                    continue
                raise PermanentError(f"HTTP {status}: {resp_body[:500]}")
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = TransientError(f"Connection error: {e}")
                logger.warning(
                    f"Transient error on {method} {path}: {last_error}"
                )
                continue

        raise last_error or TransientError("Max retries exceeded")
