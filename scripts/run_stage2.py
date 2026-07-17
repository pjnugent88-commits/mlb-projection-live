from __future__ import annotations

import argparse
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
from mlb_projection.live_enrichment import fetch_open_meteo_game_weather
from mlb_projection.mobile_dashboard import render_mobile_dashboard
from mlb_projection.odds import fetch_moneyline_odds
from mlb_projection.pipeline_v2 import project_stage2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the real Stage 2 MLB projection pipeline.")
    parser.add_argument("--date", default=date.today().isoformat())
    return parser.parse_args()


def read_csv(path: Path, dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=dates) if path.exists() else pd.DataFrame()


def main() -> None:
    args = parse_args()
    target = pd.Timestamp(args.date).normalize()
    load_dotenv(ROOT / ".env")
    with (ROOT / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    processed, outputs = ROOT / "data" / "processed", ROOT / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    required = {
        "history": processed / "history_live.csv",
        "team": processed / "team_snapshots_live.csv",
        "starters": processed / "starter_snapshots_live.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Prepared inputs are missing: {', '.join(missing)}")
    history = read_csv(required["history"], ["game_date", "game_datetime"])
    team = read_csv(required["team"], ["snapshot_time"])
    starters = read_csv(required["starters"], ["snapshot_time"])
    historical_weather = read_csv(processed / "historical_weather_live.csv", ["forecast_issued_at"])
    venues = pd.read_csv(ROOT / "data" / "reference" / "venues.csv")

    slate = fetch_schedule(args.date)
    if not slate.empty:
        slate = slate[slate["game_type"].isin(MODEL_GAME_TYPES) & ~slate["status"].isin(COMPLETED_STATUSES)].copy()
    if slate.empty:
        label = target.strftime("%Y-%m-%d")
        metrics = {"projection_date": label, "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "status": "no-games", "market_odds_available": False, "projected_games": 0}
        render_mobile_dashboard(pd.DataFrame(), metrics, outputs / "index.html", config["mobile_dashboard"]["title"])
        pd.DataFrame(columns=["game_pk", "game_date", "away_team", "home_team"]).to_csv(outputs / "projections.csv", index=False)
        (outputs / "metrics.json").write_text(pd.Series(metrics).to_json(indent=2), encoding="utf-8")
        (outputs / "metadata.json").write_text(pd.Series({"production_mode": True, "synthetic_data_used": False, **metrics}).to_json(indent=2), encoding="utf-8")
        print(f"No uncompleted MLB games found for {label}; published a no-games board.")
        return

    live_weather = fetch_open_meteo_game_weather(slate, venues)
    weather = pd.concat([historical_weather, live_weather], ignore_index=True)
    odds = fetch_moneyline_odds(api_key=os.getenv("THE_ODDS_API_KEY"), regions=os.getenv("ODDS_REGION", "us"))
    projections, metrics = project_stage2(
        history, slate, odds, config, team, starters, venues, weather,
        ROOT / "models", outputs,
    )
    print("\nHoldout metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("\nCurrent slate")
    columns = ["away_team", "home_team", "away_win_probability", "home_win_probability", "expected_total_runs", "model_signal"]
    print(projections[columns].to_string(index=False))


if __name__ == "__main__":
    main()
