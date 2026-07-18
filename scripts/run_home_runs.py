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

from mlb_projection.data_sources import MODEL_GAME_TYPES, fetch_schedule
from mlb_projection.home_run_bvp import attach_true_bvp_features, true_bvp_history
from mlb_projection.home_run_dashboard import render_home_run_dashboard
from mlb_projection.home_run_features import build_live, build_training
from mlb_projection.home_run_model import HR_FEATURES, save_home_run_model, score_home_runs, train_home_run_model
from mlb_projection.lineups import fetch_game_lineups
from mlb_projection.live_enrichment import fetch_open_meteo_game_weather
from mlb_projection.prop_odds import attach_game_pks, fetch_player_prop_odds, normalize_player_name

PREGAME_STATUSES = {"Scheduled", "Pre-Game", "Warmup", "Delayed Start"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dedicated MLB home-run probability model.")
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
        base = pd.merge_asof(
            base.sort_values(["game_datetime", "home_team"]),
            park.sort_values(["snapshot_time", "team"]),
            left_on="game_datetime", right_on="snapshot_time",
            left_by="home_team", right_by="team", direction="backward", allow_exact_matches=False,
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


def _match_hr_odds(odds: pd.DataFrame, slate: pd.DataFrame, hitters: pd.DataFrame) -> pd.DataFrame:
    if odds.empty or hitters.empty:
        return pd.DataFrame()
    odds = attach_game_pks(odds, slate)
    odds = odds[odds["market_key"].eq("batter_home_runs")].copy()
    if odds.empty:
        return odds
    odds["player_key"] = odds["player_name"].map(normalize_player_name)
    names = hitters[["game_pk", "player_name"]].copy()
    names["player_key"] = names["player_name"].map(normalize_player_name)
    return odds.drop(columns="player_name").merge(
        names.drop_duplicates(["game_pk", "player_key"]),
        on=["game_pk", "player_key"], how="inner",
    )


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    cfg = config.get("home_run_model", {})
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

    training, batter_history, pitcher_history, _ = build_training(
        pitches, context,
        long_window=int(cfg.get("long_window_games", 60)),
        recent_window=int(cfg.get("recent_window_games", 15)),
        pitcher_window=int(cfg.get("pitcher_window_starts", 15)),
    )
    pair_history = true_bvp_history(pitches)
    training = attach_true_bvp_features(training, pair_history)
    model = train_home_run_model(
        training,
        tree_weight=float(cfg.get("tree_weight", 0.35)),
        test_fraction=float(cfg.get("test_fraction", 0.15)),
        calibration_fraction=float(cfg.get("calibration_fraction", 0.15)),
        minimum_rows=int(cfg.get("minimum_rows", 4000)),
        random_state=int(config.get("model", {}).get("random_state", 42)),
    )
    save_home_run_model(model, ROOT / "models")

    slate = fetch_schedule(args.date)
    if not slate.empty:
        slate = slate[slate["game_type"].isin(MODEL_GAME_TYPES) & slate["status"].isin(PREGAME_STATUSES)].copy()
    slate = _attach_live_park_factors(slate, team_snapshots)
    lineups = fetch_game_lineups(slate)
    weather = fetch_open_meteo_game_weather(slate, venues) if not slate.empty else pd.DataFrame()
    live = build_live(
        lineups, slate, batter_history, pitcher_history, pair_history, weather, venues,
        long_window=int(cfg.get("long_window_games", 60)),
        recent_window=int(cfg.get("recent_window_games", 15)),
        pitcher_window=int(cfg.get("pitcher_window_starts", 15)),
    ) if not slate.empty else pd.DataFrame()
    for column in HR_FEATURES:
        if column not in live:
            live[column] = pd.NA

    prop_odds, odds_metadata = fetch_player_prop_odds(
        api_key=os.getenv("THE_ODDS_API_KEY"), regions=os.getenv("ODDS_REGION", "us"), target_date=args.date,
    )
    hr_odds = _match_hr_odds(prop_odds, slate, live)
    scored = score_home_runs(
        live, model, hr_odds,
        minimum_edge=float(cfg.get("minimum_edge", 0.04)),
        minimum_ev=float(cfg.get("minimum_ev", 0.04)),
        watch_probability=float(cfg.get("watch_probability", 0.16)),
    )
    top_n = int(cfg.get("dashboard_rows", 120))
    dashboard = scored.head(top_n).copy()
    generated = pd.Timestamp.now(tz="UTC").isoformat()
    metrics = {
        **model.metrics,
        "projection_date": args.date,
        "generated_at_utc": generated,
        "projected_hitters": int(len(scored)),
        "dashboard_hitters": int(len(dashboard)),
        "hr_values": int(scored["signal"].eq("HR VALUE").sum()) if not scored.empty else 0,
        "hr_watches": int(scored["signal"].eq("HR WATCH").sum()) if not scored.empty else 0,
        "confirmed_lineup_games": int(lineups[lineups["lineup_status"].eq("confirmed")]["game_pk"].nunique()) if not lineups.empty else 0,
        "market_odds_available": bool(odds_metadata.get("available", False) and not hr_odds.empty),
        "matched_hr_prices": int(len(hr_odds)),
    }
    metadata = {
        "production_mode": True,
        "synthetic_data_used": False,
        "projection_date": args.date,
        "generated_at_utc": generated,
        "model_scope": "Probability a confirmed starting batter hits at least one home run in the game.",
        "data_sources": {
            "schedule_and_lineups": "MLB Stats API schedule and boxscore",
            "pitch_tracking": "Baseball Savant Statcast via pybaseball",
            "weather": "Open-Meteo",
            "market_odds": "The Odds API batter_home_runs" if not hr_odds.empty else "unavailable",
        },
        "bvp_policy": "Only actual pitch-level plate appearances against the listed pitcher are included. PA, HR and barrels are shifted to pregame history and Bayesian-shrunk toward league priors with 40 prior PA.",
        "lineup_policy": "Only confirmed MLB batting orders with a listed probable opposing starter are published.",
        "odds_status": odds_metadata,
    }
    scored.to_csv(outputs / "home_run_probabilities.csv", index=False)
    (outputs / "home_run_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (outputs / "home_run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    render_home_run_dashboard(dashboard, metrics, outputs / "home_runs.html", cfg.get("title", "MLB Home Run Probability"))
    print(f"Published {len(scored)} standalone home-run probabilities with {metrics['matched_hr_prices']} matched sportsbook prices.")


if __name__ == "__main__":
    main()
