from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlb_projection.data_sources import COMPLETED_STATUSES, MODEL_GAME_TYPES, fetch_schedule
from mlb_projection.lineups import fetch_game_lineups
from mlb_projection.live_enrichment import fetch_open_meteo_game_weather
from mlb_projection.player_props import (
    build_live_batter_features, build_live_pitcher_features, build_prop_training_frames,
    project_player_props, save_player_prop_bundle, train_player_prop_models,
)
from mlb_projection.player_props_dashboard import render_player_props_dashboard
from mlb_projection.prop_odds import attach_game_pks, fetch_player_prop_odds, normalize_player_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 MLB player-prop projections.")
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def _read_csv(path: Path, dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=dates) if path.exists() else pd.DataFrame()


def _historical_context(history: pd.DataFrame, weather: pd.DataFrame, team_snapshots: pd.DataFrame) -> pd.DataFrame:
    base = history[["game_pk", "game_datetime", "home_team", "venue"]].copy()
    base["game_datetime"] = pd.to_datetime(base["game_datetime"], utc=True, errors="coerce")
    if not team_snapshots.empty and {"team", "snapshot_time", "park_hr_factor"}.issubset(team_snapshots.columns):
        park = team_snapshots[["team", "snapshot_time", "park_hr_factor"]].copy()
        park["snapshot_time"] = pd.to_datetime(park["snapshot_time"], utc=True, errors="coerce")
        left = base.sort_values(["game_datetime", "home_team"])
        right = park.sort_values(["snapshot_time", "team"])
        base = pd.merge_asof(
            left,
            right,
            left_on="game_datetime",
            right_on="snapshot_time",
            left_by="home_team",
            right_by="team",
            direction="backward",
            allow_exact_matches=False,
        ).drop(columns=["team", "snapshot_time"], errors="ignore")
    if "park_hr_factor" not in base:
        base["park_hr_factor"] = 1.0
    if not weather.empty:
        columns = [c for c in ["game_pk", "temperature_f", "humidity_pct", "wind_speed_mph", "precipitation_in", "roof_control_factor"] if c in weather]
        base = base.merge(weather[columns].drop_duplicates("game_pk", keep="last"), on="game_pk", how="left")
    return base.drop_duplicates("game_pk", keep="last")


def _attach_live_park_factors(slate: pd.DataFrame, team_snapshots: pd.DataFrame) -> pd.DataFrame:
    if slate.empty or team_snapshots.empty or "park_hr_factor" not in team_snapshots:
        return slate
    latest = team_snapshots.sort_values("snapshot_time").groupby("team", as_index=False).last()[["team", "park_hr_factor"]]
    return slate.merge(latest.rename(columns={"team": "home_team"}), on="home_team", how="left")


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    prop_config = config.get("player_props", {})
    processed, cache, outputs = ROOT / "data" / "processed", ROOT / "data" / "cache", ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    pitch_path = cache / "statcast_pitch_cache.pkl"
    if not pitch_path.exists():
        raise RuntimeError("Statcast cache is missing; run prepare_live_stage2.py first.")
    pitches = pd.read_pickle(pitch_path)
    history = _read_csv(processed / "history_live.csv", ["game_date", "game_datetime"])
    historical_weather = _read_csv(processed / "historical_weather_live.csv", ["forecast_issued_at"])
    team_snapshots = _read_csv(processed / "team_snapshots_live.csv", ["snapshot_time"])
    venues = pd.read_csv(ROOT / "data" / "reference" / "venues.csv")
    context = _historical_context(history, historical_weather, team_snapshots)
    rolling_games = int(prop_config.get("rolling_games", 15))
    batter_train, pitcher_train, batter_history, pitcher_history, team_history = build_prop_training_frames(pitches, context, rolling_games)
    bundle = train_player_prop_models(
        batter_train, pitcher_train,
        test_fraction=float(prop_config.get("test_fraction", 0.20)),
        tree_weight=float(prop_config.get("ensemble_tree_weight", 0.55)),
        minimum_pitcher_rows=int(prop_config.get("minimum_pitcher_rows", 300)),
        minimum_batter_rows=int(prop_config.get("minimum_batter_rows", 1200)),
        random_state=int(config.get("model", {}).get("random_state", 42)),
    )
    save_player_prop_bundle(bundle, ROOT / "models")

    slate = fetch_schedule(args.date)
    if not slate.empty:
        slate = slate[slate["game_type"].isin(MODEL_GAME_TYPES) & ~slate["status"].isin(COMPLETED_STATUSES)].copy()
    slate = _attach_live_park_factors(slate, team_snapshots)
    lineups = fetch_game_lineups(slate)
    live_weather = fetch_open_meteo_game_weather(slate, venues) if not slate.empty else pd.DataFrame()
    pitcher_features = build_live_pitcher_features(slate, pitcher_history, team_history, live_weather, venues, rolling_games) if not slate.empty else pd.DataFrame()
    batter_features = build_live_batter_features(lineups, slate, batter_history, pitcher_history, live_weather, venues, rolling_games) if not slate.empty else pd.DataFrame()

    prop_odds, odds_metadata = fetch_player_prop_odds(
        api_key=os.getenv("THE_ODDS_API_KEY"), regions=os.getenv("ODDS_REGION", "us"), target_date=args.date,
    )
    if not prop_odds.empty:
        prop_odds = attach_game_pks(prop_odds, slate)
        prop_odds["player_key"] = prop_odds["player_name"].map(normalize_player_name)
        name_parts = []
        for frame in (pitcher_features, batter_features):
            if frame is not None and not frame.empty and {"game_pk", "player_name"}.issubset(frame.columns):
                name_parts.append(frame[["game_pk", "player_name"]])
        name_rows = pd.concat(name_parts, ignore_index=True) if name_parts else pd.DataFrame(columns=["game_pk", "player_name"])
        name_rows["player_key"] = name_rows["player_name"].map(normalize_player_name)
        prop_odds = prop_odds.drop(columns="player_name").merge(name_rows.drop_duplicates(["game_pk", "player_key"]), on=["game_pk", "player_key"], how="inner")

    props = project_player_props(
        bundle, pitcher_features, batter_features, prop_odds,
        minimum_edge=float(prop_config.get("minimum_edge", config.get("betting", {}).get("minimum_edge", 0.025))),
        minimum_ev=float(prop_config.get("minimum_ev", config.get("betting", {}).get("minimum_ev", 0.02))),
    )
    generated = pd.Timestamp.now(tz="UTC").isoformat()
    metrics = {
        **bundle.metrics, "projection_date": args.date, "generated_at_utc": generated,
        "market_odds_available": bool(odds_metadata.get("available", False)),
        "projected_props": int(len(props)), "pitcher_props": int(props["player_type"].eq("pitcher").sum()) if not props.empty else 0,
        "batter_props": int(props["player_type"].eq("batter").sum()) if not props.empty else 0,
        "confirmed_lineup_games": int(lineups[lineups["lineup_status"].eq("confirmed")]["game_pk"].nunique()) if not lineups.empty else 0,
    }
    metadata = {
        "production_mode": True, "synthetic_data_used": False, "projection_date": args.date,
        "generated_at_utc": generated,
        "data_sources": {
            "schedule_and_lineups": "MLB Stats API schedule and boxscore",
            "pitch_tracking": "Baseball Savant Statcast via pybaseball",
            "weather": "Open-Meteo",
            "market_odds": "The Odds API event player props" if odds_metadata.get("available") else "not configured or tier unavailable",
        },
        "odds_status": odds_metadata,
        "lineup_policy": "Batter props are published only for batting orders returned by MLB Stats API.",
    }
    props.to_csv(outputs / "player_props.csv", index=False)
    (outputs / "prop_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (outputs / "prop_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    render_player_props_dashboard(props.head(int(prop_config.get("top_props", 80))), metrics, outputs / "props.html", prop_config.get("title", "MLB Player Props"))
    print(f"Published {len(props)} player props: {metrics['pitcher_props']} pitcher and {metrics['batter_props']} batter rows.")


if __name__ == "__main__":
    main()
