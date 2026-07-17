from __future__ import annotations

import time

import pandas as pd
import requests

BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"


def _fetch_boxscore(game_pk: int, retries: int = 3) -> dict:
    last_error: Exception | None = None
    headers = {"User-Agent": "mlb-projection-live/3.0 (+GitHub Actions)"}
    for attempt in range(retries):
        try:
            response = requests.get(BOXSCORE_URL.format(game_pk=int(game_pk)), headers=headers, timeout=45)
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    if last_error:
        return {}
    return {}


def fetch_game_lineups(slate: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_pk", "side", "team", "player_id", "player_name", "batting_order", "lineup_status"]
    if slate is None or slate.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    for _, game in slate.iterrows():
        game_pk = int(game["game_pk"])
        payload = _fetch_boxscore(game_pk)
        for side in ("away", "home"):
            team_block = payload.get("teams", {}).get(side, {})
            batting_order = team_block.get("battingOrder") or []
            if len(batting_order) < 9:
                continue
            players = team_block.get("players", {})
            for position, raw_player_id in enumerate(batting_order[:9], start=1):
                try:
                    player_id = int(raw_player_id)
                except (TypeError, ValueError):
                    continue
                player = players.get(f"ID{player_id}", {})
                person = player.get("person", {})
                rows.append({
                    "game_pk": game_pk,
                    "side": side,
                    "team": game.get(f"{side}_team"),
                    "player_id": player_id,
                    "player_name": person.get("fullName") or person.get("fullNameSlug") or str(player_id),
                    "batting_order": position,
                    "lineup_status": "confirmed",
                })
    return pd.DataFrame(rows, columns=columns)
