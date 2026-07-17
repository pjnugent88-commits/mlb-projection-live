from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd
import requests

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"


def american_to_implied_probability(odds: float | int) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    return 100.0 / (odds + 100.0) if odds > 0 else (-odds) / ((-odds) + 100.0)


def probability_to_american(probability: float) -> int:
    if not 0.0 < probability < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")
    if probability >= 0.5:
        return int(round(-100.0 * probability / (1.0 - probability)))
    return int(round(100.0 * (1.0 - probability) / probability))


def expected_value_per_unit(probability: float, american_odds: float | int) -> float:
    odds = float(american_odds)
    profit_if_win = odds / 100.0 if odds > 0 else 100.0 / (-odds)
    return probability * profit_if_win - (1.0 - probability)


def _normalize_team_name(name: str) -> str:
    replacements = {
        "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
        "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
        "New York Yankees": "NYY", "Athletics": "ATH", "Oakland Athletics": "ATH",
        "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
        "San Francisco Giants": "SF", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSH",
    }
    return replacements.get(name, name)


def fetch_moneyline_odds(api_key: str | None = None, regions: str = "us") -> pd.DataFrame:
    api_key = api_key or os.getenv("THE_ODDS_API_KEY")
    columns = ["away_team", "home_team", "best_away_odds", "best_home_odds", "away_book", "home_book"]
    if not api_key:
        return pd.DataFrame(columns=columns)

    response = requests.get(
        ODDS_API_URL,
        params={
            "apiKey": api_key, "regions": regions, "markets": "h2h",
            "oddsFormat": "american", "dateFormat": "iso",
        },
        timeout=30,
    )
    response.raise_for_status()
    events: list[dict[str, Any]] = response.json()
    rows: list[dict[str, Any]] = []
    for event in events:
        away_name, home_name = event["away_team"], event["home_team"]
        best = {away_name: (-10_000, None), home_name: (-10_000, None)}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    team_name, price = outcome.get("name"), outcome.get("price")
                    if team_name in best and price is not None and price > best[team_name][0]:
                        best[team_name] = (price, bookmaker.get("title"))
        rows.append({
            "away_team": _normalize_team_name(away_name),
            "home_team": _normalize_team_name(home_name),
            "best_away_odds": best[away_name][0] if best[away_name][1] is not None else math.nan,
            "best_home_odds": best[home_name][0] if best[home_name][1] is not None else math.nan,
            "away_book": best[away_name][1], "home_book": best[home_name][1],
        })
    return pd.DataFrame(rows, columns=columns)
