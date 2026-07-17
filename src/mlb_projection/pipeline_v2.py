from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .advanced_features import ADVANCED_FEATURE_COLUMNS, attach_advanced_features
from .features import FEATURE_COLUMNS, build_pregame_features
from .mobile_dashboard import render_mobile_dashboard
from .models_v2 import save_stage2_bundle, train_stage2_models
from .odds import american_to_implied_probability, expected_value_per_unit, probability_to_american


def _json_ready(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _latest_live_weather(weather: pd.DataFrame | None, game_ids: set[int]) -> pd.DataFrame:
    if weather is None or weather.empty:
        return pd.DataFrame()
    frame = weather[weather["game_pk"].isin(game_ids)].copy()
    if frame.empty:
        return frame
    frame["forecast_issued_at"] = pd.to_datetime(frame["forecast_issued_at"], utc=True)
    return frame.sort_values("forecast_issued_at").drop_duplicates("game_pk", keep="last")


def project_stage2(
    history: pd.DataFrame,
    slate: pd.DataFrame,
    odds: pd.DataFrame | None,
    config: dict,
    team_snapshots: pd.DataFrame | None,
    starter_snapshots: pd.DataFrame | None,
    venue_factors: pd.DataFrame | None,
    weather_snapshots: pd.DataFrame | None,
    model_dir: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    if history.empty or slate.empty:
        raise ValueError("Historical games and a current slate are required.")
    mc = config["model"]
    training, future = build_pregame_features(
        history, slate,
        rolling_window=int(mc["rolling_window_games"]), venue_window=int(mc["venue_window_games"]),
        elo_k=float(mc["elo_k"]), home_advantage=float(mc["elo_home_advantage"]),
    )
    metadata_columns = ["game_pk", "game_datetime", "venue", "home_probable_pitcher_id", "away_probable_pitcher_id"]
    training = training.merge(history[[c for c in metadata_columns if c in history]], on="game_pk", how="left")
    future = future.merge(slate[[c for c in metadata_columns if c in slate]], on="game_pk", how="left")
    training["game_datetime"] = pd.to_datetime(training["game_datetime"], utc=True, errors="coerce")
    future["game_datetime"] = pd.to_datetime(future["game_datetime"], utc=True, errors="coerce")
    training = training.merge(attach_advanced_features(training, team_snapshots, starter_snapshots, venue_factors, weather_snapshots), on="game_pk", how="left")
    future = future.merge(attach_advanced_features(future, team_snapshots, starter_snapshots, venue_factors, weather_snapshots), on="game_pk", how="left")
    feature_columns = FEATURE_COLUMNS + ADVANCED_FEATURE_COLUMNS
    bundle = train_stage2_models(training, feature_columns, float(mc["test_fraction"]), int(mc["minimum_training_games"]), float(mc["ensemble_tree_weight"]))
    save_stage2_bundle(bundle, model_dir)

    home_probability = np.clip(bundle.predict_home_win(future[feature_columns]), 0.02, 0.98)
    expected_home_runs, expected_away_runs = bundle.predict_runs(future[feature_columns])
    display_columns = ["game_pk", "game_date", "game_datetime", "away_team", "home_team", "venue", "away_probable_pitcher", "home_probable_pitcher", "away_probable_pitcher_id", "home_probable_pitcher_id", "status"]
    out = slate[[c for c in display_columns if c in slate]].copy()
    predictions = pd.DataFrame({
        "game_pk": future["game_pk"].astype(int), "home_win_probability": home_probability,
        "away_win_probability": 1.0 - home_probability, "expected_home_runs": expected_home_runs,
        "expected_away_runs": expected_away_runs,
    })
    predictions["expected_total_runs"] = predictions["expected_home_runs"] + predictions["expected_away_runs"]
    predictions["home_fair_odds"] = [probability_to_american(float(p)) for p in predictions["home_win_probability"]]
    predictions["away_fair_odds"] = [probability_to_american(float(p)) for p in predictions["away_win_probability"]]
    predictions["model_confidence"] = 2.0 * np.abs(predictions["home_win_probability"] - 0.5)
    out = out.merge(predictions, on="game_pk", how="left")

    live_weather = _latest_live_weather(weather_snapshots, set(out["game_pk"].astype(int)))
    weather_columns = ["game_pk", "forecast_issued_at", "temperature_f", "humidity_pct", "wind_speed_mph", "precipitation_in", "roof_control_factor", "weather_source"]
    if not live_weather.empty:
        out = out.merge(live_weather[[c for c in weather_columns if c in live_weather]], on="game_pk", how="left")
    for col in weather_columns[1:]:
        if col not in out:
            out[col] = np.nan
    home_ids = out["home_probable_pitcher_id"] if "home_probable_pitcher_id" in out else pd.Series(np.nan, index=out.index)
    away_ids = out["away_probable_pitcher_id"] if "away_probable_pitcher_id" in out else pd.Series(np.nan, index=out.index)
    out["starter_data_complete"] = home_ids.notna() & away_ids.notna()
    out["weather_data_complete"] = out["temperature_f"].notna()

    market_available = odds is not None and not odds.empty
    if market_available:
        out = out.merge(odds, on=["away_team", "home_team"], how="left")
        out["home_implied_probability"] = out["best_home_odds"].apply(lambda x: american_to_implied_probability(x) if pd.notna(x) else np.nan)
        out["away_implied_probability"] = out["best_away_odds"].apply(lambda x: american_to_implied_probability(x) if pd.notna(x) else np.nan)
        overround = out["home_implied_probability"] + out["away_implied_probability"]
        out["home_market_probability_novig"] = out["home_implied_probability"] / overround
        out["away_market_probability_novig"] = out["away_implied_probability"] / overround
        out["home_edge"] = out["home_win_probability"] - out["home_market_probability_novig"]
        out["away_edge"] = out["away_win_probability"] - out["away_market_probability_novig"]
        out["home_ev_per_unit"] = out.apply(lambda r: expected_value_per_unit(r["home_win_probability"], r["best_home_odds"]) if pd.notna(r["best_home_odds"]) else np.nan, axis=1)
        out["away_ev_per_unit"] = out.apply(lambda r: expected_value_per_unit(r["away_win_probability"], r["best_away_odds"]) if pd.notna(r["best_away_odds"]) else np.nan, axis=1)
        minimum_edge, minimum_ev = float(config["betting"]["minimum_edge"]), float(config["betting"]["minimum_ev"])
        def signal(row: pd.Series) -> str:
            choices = []
            if pd.notna(row["home_ev_per_unit"]) and row["home_edge"] >= minimum_edge and row["home_ev_per_unit"] >= minimum_ev:
                choices.append(("HOME VALUE", float(row["home_ev_per_unit"])))
            if pd.notna(row["away_ev_per_unit"]) and row["away_edge"] >= minimum_edge and row["away_ev_per_unit"] >= minimum_ev:
                choices.append(("AWAY VALUE", float(row["away_ev_per_unit"])))
            return max(choices, key=lambda item: item[1])[0] if choices else "PASS"
        out["model_signal"] = out.apply(signal, axis=1)
        out["best_ev_per_unit"] = out[["home_ev_per_unit", "away_ev_per_unit"]].max(axis=1, skipna=True)
        out = out.sort_values(["best_ev_per_unit", "model_confidence"], ascending=False, na_position="last")
    else:
        out["model_signal"] = np.select([out["home_win_probability"] >= 0.58, out["away_win_probability"] >= 0.58], ["HOME LEAN", "AWAY LEAN"], default="PASS")
        out["best_ev_per_unit"] = np.nan
        out = out.sort_values("model_confidence", ascending=False)

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    projection_date = pd.Timestamp(out["game_date"].iloc[0]).strftime("%Y-%m-%d")
    generated_at = datetime.now(timezone.utc).isoformat()
    metrics = dict(bundle.metrics)
    metrics.update({
        "projection_date": projection_date, "generated_at_utc": generated_at,
        "market_odds_available": bool(market_available), "projected_games": int(len(out)),
        "starter_complete_games": int(out["starter_data_complete"].sum()), "weather_complete_games": int(out["weather_data_complete"].sum()),
    })
    metadata = {
        "projection_date": projection_date, "generated_at_utc": generated_at,
        "data_sources": {"schedule_and_results": "MLB Stats API", "pitch_tracking": "Baseball Savant Statcast via pybaseball", "weather": "Open-Meteo forecast and 24-hour previous-run forecast", "market_odds": "The Odds API" if market_available else "not configured"},
        "production_mode": True, "synthetic_data_used": False,
        "metrics": {key: _json_ready(value) for key, value in metrics.items()},
    }
    out.to_csv(output_dir / "projections.csv", index=False)
    out.to_csv(output_dir / f"projections_{projection_date}.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_ready), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_ready), encoding="utf-8")
    render_mobile_dashboard(out, metrics, output_dir / "index.html", config["mobile_dashboard"]["title"])
    return out, metrics
