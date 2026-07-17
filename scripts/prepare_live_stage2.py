from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_projection.data_sources import fetch_historical_games
from mlb_projection.live_enrichment import build_daily_statcast_snapshot_history, fetch_historical_pregame_weather, fetch_recent_statcast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real, leakage-safe MLB training inputs.")
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def attach_actual_starters(history: pd.DataFrame, pitches: pd.DataFrame) -> pd.DataFrame:
    required = {"game_pk", "home_team", "away_team", "inning_topbot", "pitcher"}
    if pitches.empty or not required.issubset(pitches.columns):
        return history
    frame = pitches.copy()
    code_map = {"CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH", "OAK": "ATH"}
    frame["home_team"] = frame["home_team"].replace(code_map)
    frame["away_team"] = frame["away_team"].replace(code_map)
    frame["field_team"] = frame["home_team"].where(frame["inning_topbot"].eq("Top"), frame["away_team"])
    sort_cols = [c for c in ["game_pk", "field_team", "inning", "at_bat_number", "pitch_number"] if c in frame]
    starters = frame.sort_values(sort_cols).groupby(["game_pk", "field_team"], as_index=False).first()[["game_pk", "field_team", "pitcher"]].rename(columns={"pitcher": "actual_starter_id"})
    out = history.copy()
    home = starters.rename(columns={"field_team": "home_team", "actual_starter_id": "actual_home_starter_id"})
    away = starters.rename(columns={"field_team": "away_team", "actual_starter_id": "actual_away_starter_id"})
    out = out.merge(home, on=["game_pk", "home_team"], how="left").merge(away, on=["game_pk", "away_team"], how="left")
    out["home_probable_pitcher_id"] = pd.to_numeric(out["actual_home_starter_id"], errors="coerce").combine_first(pd.to_numeric(out["home_probable_pitcher_id"], errors="coerce"))
    out["away_probable_pitcher_id"] = pd.to_numeric(out["actual_away_starter_id"], errors="coerce").combine_first(pd.to_numeric(out["away_probable_pitcher_id"], errors="coerce"))
    return out.drop(columns=["actual_home_starter_id", "actual_away_starter_id"])


def main() -> None:
    args = parse_args()
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    live = config["live_data"]
    target = pd.Timestamp(args.date).normalize()
    history_start = target - pd.Timedelta(days=int(live["history_days"]))
    history_end = target - pd.Timedelta(days=1)
    cache_dir, processed_dir = ROOT / "data" / "cache", ROOT / "data" / "processed"
    cache_dir.mkdir(parents=True, exist_ok=True); processed_dir.mkdir(parents=True, exist_ok=True)

    history = fetch_historical_games(history_start.strftime("%Y-%m-%d"), history_end.strftime("%Y-%m-%d"), cache_path=cache_dir / "historical_games.csv")
    minimum = int(config["model"]["minimum_training_games"])
    if len(history) < minimum:
        raise RuntimeError(f"Only {len(history)} completed model games were available; at least {minimum} are required.")
    statcast_start = pd.Timestamp(history["game_date"].min()).normalize() - pd.Timedelta(days=int(live["statcast_lookback_days"]))
    pitches = fetch_recent_statcast(statcast_start.strftime("%Y-%m-%d"), history_end.strftime("%Y-%m-%d"), cache_path=cache_dir / "statcast_pitch_cache.pkl", chunk_days=int(live["statcast_chunk_days"]))
    if pitches.empty:
        raise RuntimeError("Statcast returned no pitch data for the training window.")
    history = attach_actual_starters(history, pitches)
    snapshot_dates = list(history["game_date"].drop_duplicates()) + [target]
    team, starter = build_daily_statcast_snapshot_history(pitches, snapshot_dates, int(live["statcast_lookback_days"]))
    if team.empty:
        raise RuntimeError("Team Statcast snapshots could not be built.")

    venues = pd.read_csv(ROOT / "data" / "reference" / "venues.csv")
    weather_cache = cache_dir / "historical_weather.csv"
    cached_weather = pd.read_csv(weather_cache) if weather_cache.exists() else pd.DataFrame()
    covered = set(pd.to_numeric(cached_weather["game_pk"], errors="coerce").dropna().astype(int)) if not cached_weather.empty else set()
    missing_games = history[~history["game_pk"].isin(covered)]
    new_weather = fetch_historical_pregame_weather(missing_games, venues) if not missing_games.empty else pd.DataFrame()
    weather = pd.concat([cached_weather, new_weather], ignore_index=True) if not cached_weather.empty else new_weather
    if not weather.empty:
        weather = weather.drop_duplicates("game_pk", keep="last")
        weather.to_csv(weather_cache, index=False)

    history.to_csv(processed_dir / "history_live.csv", index=False)
    team.to_csv(processed_dir / "team_snapshots_live.csv", index=False)
    starter.to_csv(processed_dir / "starter_snapshots_live.csv", index=False)
    weather.to_csv(processed_dir / "historical_weather_live.csv", index=False)
    print(f"Prepared {len(history)} games, {len(pitches):,} pitches, {len(team):,} team snapshots, {len(starter):,} starter snapshots, and {len(weather):,} weather rows.")


if __name__ == "__main__":
    main()
