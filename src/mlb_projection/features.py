from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TeamState:
    elo: float = 1500.0


FEATURE_COLUMNS = [
    "elo_diff_home",
    "home_win_rate_rolling",
    "away_win_rate_rolling",
    "win_rate_diff",
    "home_run_diff_rolling",
    "away_run_diff_rolling",
    "run_diff_gap",
    "home_runs_scored_rolling",
    "away_runs_scored_rolling",
    "home_runs_allowed_rolling",
    "away_runs_allowed_rolling",
    "rest_diff",
    "venue_run_factor",
]


def _mean(values: deque, default: float) -> float:
    return float(np.mean(values)) if values else default


def build_pregame_features(
    completed_games: pd.DataFrame,
    future_games: pd.DataFrame | None = None,
    rolling_window: int = 18,
    venue_window: int = 80,
    elo_k: float = 20.0,
    home_advantage: float = 35.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = completed_games.copy()
    history["game_datetime"] = pd.to_datetime(history["game_datetime"], utc=True, errors="coerce")
    history["game_date"] = pd.to_datetime(history["game_date"]).dt.normalize()
    history = history.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)

    states = defaultdict(TeamState)
    recent_wins = defaultdict(lambda: deque(maxlen=rolling_window))
    recent_scored = defaultdict(lambda: deque(maxlen=rolling_window))
    recent_allowed = defaultdict(lambda: deque(maxlen=rolling_window))
    recent_diff = defaultdict(lambda: deque(maxlen=rolling_window))
    venue_totals = defaultdict(lambda: deque(maxlen=venue_window))
    league_totals = deque(maxlen=max(venue_window * 8, 400))
    last_played: dict[str, pd.Timestamp] = {}

    def make_row(row: pd.Series) -> dict:
        home, away = row["home_team"], row["away_team"]
        game_date = pd.Timestamp(row["game_date"])
        venue = str(row.get("venue") or f"{home}-home")
        home_rest = min((game_date - last_played[home]).days, 10) if home in last_played else 5
        away_rest = min((game_date - last_played[away]).days, 10) if away in last_played else 5
        home_wr = _mean(recent_wins[home], 0.5)
        away_wr = _mean(recent_wins[away], 0.5)
        home_rd = _mean(recent_diff[home], 0.0)
        away_rd = _mean(recent_diff[away], 0.0)
        league_run_mean = _mean(league_totals, 8.8)
        venue_run_mean = _mean(venue_totals[venue], league_run_mean)
        venue_factor = float(np.clip(venue_run_mean / max(league_run_mean, 0.1), 0.78, 1.24))
        return {
            "game_pk": int(row["game_pk"]),
            "game_date": game_date,
            "away_team": away,
            "home_team": home,
            "elo_diff_home": states[home].elo + home_advantage - states[away].elo,
            "home_win_rate_rolling": home_wr,
            "away_win_rate_rolling": away_wr,
            "win_rate_diff": home_wr - away_wr,
            "home_run_diff_rolling": home_rd,
            "away_run_diff_rolling": away_rd,
            "run_diff_gap": home_rd - away_rd,
            "home_runs_scored_rolling": _mean(recent_scored[home], 4.4),
            "away_runs_scored_rolling": _mean(recent_scored[away], 4.4),
            "home_runs_allowed_rolling": _mean(recent_allowed[home], 4.4),
            "away_runs_allowed_rolling": _mean(recent_allowed[away], 4.4),
            "rest_diff": home_rest - away_rest,
            "venue_run_factor": venue_factor,
        }

    training_rows: list[dict] = []
    for _, row in history.iterrows():
        features = make_row(row)
        hr, ar = int(row["home_runs"]), int(row["away_runs"])
        home_win = int(hr > ar)
        features.update(home_runs=hr, away_runs=ar, home_win=home_win)
        training_rows.append(features)

        home, away = row["home_team"], row["away_team"]
        expected = 1.0 / (1.0 + 10.0 ** ((states[away].elo - (states[home].elo + home_advantage)) / 400.0))
        change = elo_k * (home_win - expected)
        states[home].elo += change
        states[away].elo -= change
        recent_wins[home].append(home_win)
        recent_wins[away].append(1 - home_win)
        recent_scored[home].append(hr)
        recent_scored[away].append(ar)
        recent_allowed[home].append(ar)
        recent_allowed[away].append(hr)
        recent_diff[home].append(hr - ar)
        recent_diff[away].append(ar - hr)
        total = hr + ar
        venue_totals[str(row.get("venue") or f"{home}-home")].append(total)
        league_totals.append(total)
        last_played[home] = pd.Timestamp(row["game_date"])
        last_played[away] = pd.Timestamp(row["game_date"])

    future_rows: list[dict] = []
    if future_games is not None and not future_games.empty:
        future = future_games.copy()
        future["game_datetime"] = pd.to_datetime(future["game_datetime"], utc=True, errors="coerce")
        future["game_date"] = pd.to_datetime(future["game_date"]).dt.normalize()
        for _, row in future.sort_values(["game_datetime", "game_pk"]).iterrows():
            future_rows.append(make_row(row))
    return pd.DataFrame(training_rows), pd.DataFrame(future_rows)
