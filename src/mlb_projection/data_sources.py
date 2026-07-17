from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

TEAM_ABBREVIATIONS = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC", 119: "LAD", 120: "WSH", 121: "NYM", 133: "ATH",
    134: "PIT", 135: "SD", 136: "SEA", 137: "SF", 138: "STL",
    139: "TB", 140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

COMPLETED_STATUSES = {"Final", "Game Over", "Completed Early"}
MODEL_GAME_TYPES = {"R", "F", "D", "L", "W"}


def _get_json(url: str, params: dict[str, Any], retries: int = 4) -> Any:
    last_error: Exception | None = None
    headers = {"User-Agent": "mlb-projection-live/2.1 (+GitHub Actions)"}
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=75)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def fetch_schedule(start_date: str, end_date: str | None = None) -> pd.DataFrame:
    payload = _get_json(
        MLB_SCHEDULE_URL,
        {
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date or start_date,
            "hydrate": "probablePitcher,team,linescore",
        },
    )
    rows: list[dict] = []
    for date_block in payload.get("dates", []):
        for game in date_block.get("games", []):
            away = game["teams"]["away"]
            home = game["teams"]["home"]
            away_id = away["team"]["id"]
            home_id = home["team"]["id"]
            rows.append({
                "game_pk": int(game["gamePk"]),
                "game_date": date_block["date"],
                "game_datetime": game.get("gameDate"),
                "game_type": game.get("gameType"),
                "status": game.get("status", {}).get("detailedState"),
                "away_team": TEAM_ABBREVIATIONS.get(away_id, away["team"].get("abbreviation", away["team"]["name"])),
                "home_team": TEAM_ABBREVIATIONS.get(home_id, home["team"].get("abbreviation", home["team"]["name"])),
                "away_team_id": away_id,
                "home_team_id": home_id,
                "away_runs": away.get("score"),
                "home_runs": home.get("score"),
                "away_probable_pitcher_id": away.get("probablePitcher", {}).get("id"),
                "home_probable_pitcher_id": home.get("probablePitcher", {}).get("id"),
                "away_probable_pitcher": away.get("probablePitcher", {}).get("fullName"),
                "home_probable_pitcher": home.get("probablePitcher", {}).get("fullName"),
                "venue": game.get("venue", {}).get("name"),
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
        frame["game_datetime"] = pd.to_datetime(frame["game_datetime"], utc=True, errors="coerce")
    return frame


def fetch_historical_games(
    start_date: str,
    end_date: str,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    cache = Path(cache_path) if cache_path else None
    cached = pd.DataFrame()
    if cache and cache.exists():
        cached = pd.read_csv(cache)
        if not cached.empty:
            cached["game_date"] = pd.to_datetime(cached["game_date"]).dt.normalize()
            cached["game_datetime"] = pd.to_datetime(cached["game_datetime"], utc=True, errors="coerce")

    parts: list[pd.DataFrame] = [cached] if not cached.empty else []
    cursor = start
    if not cached.empty:
        covered = cached[cached["game_date"].between(start, end)]
        if not covered.empty:
            cursor = covered["game_date"].max() + pd.Timedelta(days=1)

    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=30), end)
        part = fetch_schedule(cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"))
        if not part.empty:
            parts.append(part)
        cursor = chunk_end + pd.Timedelta(days=1)

    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if frame.empty:
        return frame
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    frame["game_datetime"] = pd.to_datetime(frame["game_datetime"], utc=True, errors="coerce")
    frame = frame[
        frame["status"].isin(COMPLETED_STATUSES)
        & frame["game_type"].isin(MODEL_GAME_TYPES)
        & frame["home_runs"].notna()
        & frame["away_runs"].notna()
        & frame["game_date"].between(start, end)
    ].copy()
    frame["home_runs"] = frame["home_runs"].astype(int)
    frame["away_runs"] = frame["away_runs"].astype(int)
    frame = frame.sort_values(["game_datetime", "game_pk"]).drop_duplicates("game_pk", keep="last").reset_index(drop=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    return frame
