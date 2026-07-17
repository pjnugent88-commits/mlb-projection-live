from __future__ import annotations

import math
import os
import re
import unicodedata
from typing import Any

import pandas as pd
import requests

from .odds import american_to_implied_probability, _normalize_team_name

SPORT_KEY = "baseball_mlb"
EVENTS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events"
EVENT_ODDS_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/events/{{event_id}}/odds"
PROP_MARKETS = [
    "pitcher_strikeouts", "pitcher_outs", "pitcher_hits_allowed",
    "batter_hits", "batter_total_bases", "batter_home_runs",
]


def normalize_player_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _request_json(url: str, params: dict[str, Any], timeout: int = 35) -> Any:
    response = requests.get(url, params=params, timeout=timeout)
    if response.status_code in {401, 403, 422}:
        return None
    response.raise_for_status()
    return response.json()


def fetch_player_prop_odds(api_key: str | None = None, regions: str = "us", target_date: str | None = None) -> tuple[pd.DataFrame, dict]:
    api_key = api_key or os.getenv("THE_ODDS_API_KEY")
    columns = [
        "game_pk", "market_key", "player_name", "point", "over_odds", "under_odds",
        "over_book", "under_book", "over_no_vig_probability", "under_no_vig_probability",
    ]
    metadata = {"available": False, "provider": "The Odds API", "reason": "no-api-key"}
    if not api_key:
        return pd.DataFrame(columns=columns), metadata
    params: dict[str, Any] = {"apiKey": api_key, "dateFormat": "iso"}
    if target_date:
        day = pd.Timestamp(target_date).tz_localize("America/New_York").tz_convert("UTC")
        params["commenceTimeFrom"] = day.isoformat().replace("+00:00", "Z")
        params["commenceTimeTo"] = (day + pd.Timedelta(days=1, seconds=-1)).isoformat().replace("+00:00", "Z")
    events = _request_json(EVENTS_URL, params)
    if events is None:
        metadata["reason"] = "api-tier-or-auth-does-not-include-player-props"
        return pd.DataFrame(columns=columns), metadata
    rows: list[dict] = []
    for event in events:
        payload = _request_json(EVENT_ODDS_URL.format(event_id=event["id"]), {
            "apiKey": api_key, "regions": regions, "markets": ",".join(PROP_MARKETS),
            "oddsFormat": "american", "dateFormat": "iso",
        })
        if payload is None:
            metadata["reason"] = "api-tier-does-not-include-player-props"
            break
        event_key = (str(_normalize_team_name(payload.get("away_team", ""))), str(_normalize_team_name(payload.get("home_team", ""))))
        selections: list[dict] = []
        for bookmaker in payload.get("bookmakers", []):
            book = bookmaker.get("title")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                if market_key not in PROP_MARKETS:
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("description") is None or outcome.get("point") is None or outcome.get("price") is None:
                        continue
                    selections.append({
                        "away_team": event_key[0], "home_team": event_key[1], "market_key": market_key,
                        "player_name": str(outcome["description"]), "player_key": normalize_player_name(outcome["description"]),
                        "point": float(outcome["point"]), "side": str(outcome.get("name", "")).lower(),
                        "price": float(outcome["price"]), "book": book,
                    })
        if not selections:
            continue
        raw = pd.DataFrame(selections)
        for (away, home, market_key, player_key), group in raw.groupby(["away_team", "home_team", "market_key", "player_key"]):
            line_counts = group.groupby("point").size().sort_values(ascending=False)
            point = float(line_counts.index[0])
            line_group = group[group["point"].eq(point)]
            over = line_group[line_group["side"].eq("over")].sort_values("price", ascending=False).head(1)
            under = line_group[line_group["side"].eq("under")].sort_values("price", ascending=False).head(1)
            over_odds = float(over["price"].iloc[0]) if not over.empty else math.nan
            under_odds = float(under["price"].iloc[0]) if not under.empty else math.nan
            over_implied = american_to_implied_probability(over_odds) if pd.notna(over_odds) else math.nan
            under_implied = american_to_implied_probability(under_odds) if pd.notna(under_odds) else math.nan
            total = over_implied + under_implied if pd.notna(over_implied) and pd.notna(under_implied) else math.nan
            rows.append({
                "away_team": away, "home_team": home, "market_key": market_key,
                "player_name": str(group["player_name"].iloc[0]), "player_key": player_key, "point": point,
                "over_odds": over_odds, "under_odds": under_odds,
                "over_book": None if over.empty else over["book"].iloc[0],
                "under_book": None if under.empty else under["book"].iloc[0],
                "over_no_vig_probability": over_implied / total if pd.notna(total) and total > 0 else math.nan,
                "under_no_vig_probability": under_implied / total if pd.notna(total) and total > 0 else math.nan,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns), metadata
    metadata.update({"available": True, "reason": None, "events_with_props": int(frame[["away_team", "home_team"]].drop_duplicates().shape[0])})
    return frame, metadata


def attach_game_pks(prop_odds: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    if prop_odds.empty:
        return prop_odds.copy()
    games = slate[["game_pk", "away_team", "home_team"]].copy()
    return prop_odds.merge(games, on=["away_team", "home_team"], how="inner")
