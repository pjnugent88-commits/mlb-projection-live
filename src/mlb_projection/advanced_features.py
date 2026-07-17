from __future__ import annotations

import numpy as np
import pandas as pd

ADVANCED_FEATURE_COLUMNS = [
    "offense_xwoba_diff",
    "offense_hard_hit_diff",
    "offense_barrel_diff",
    "offense_k_minus_bb_diff",
    "starter_xwoba_allowed_edge",
    "starter_k_minus_bb_diff",
    "starter_projected_innings_diff",
    "bullpen_xwoba_allowed_edge",
    "bullpen_fatigue_diff",
    "bullpen_pitchers_used_diff",
    "park_hr_factor",
    "temperature_f",
    "humidity_pct",
    "wind_speed_mph",
    "precipitation_in",
    "roof_control_factor",
]

DEFAULTS = {
    "offense_xwoba": 0.315,
    "offense_hard_hit_rate": 0.385,
    "offense_barrel_rate": 0.075,
    "offense_k_minus_bb": -0.140,
    "starter_xwoba_allowed": 0.315,
    "starter_k_minus_bb": 0.140,
    "starter_projected_innings": 5.2,
    "bullpen_xwoba_allowed": 0.315,
    "bullpen_pitches_last_3d": 135.0,
    "bullpen_pitchers_used_last_3d": 8.0,
    "park_hr_factor": 1.0,
    "temperature_f": 72.0,
    "humidity_pct": 55.0,
    "wind_speed_mph": 7.0,
    "precipitation_in": 0.0,
    "roof_control_factor": 0.0,
}


def _latest_strictly_before(
    events: pd.DataFrame,
    snapshots: pd.DataFrame,
    left_by: str,
    right_by: str,
) -> pd.DataFrame:
    if snapshots is None or snapshots.empty:
        return events.copy()
    left = events.copy()
    right = snapshots.copy()
    left["game_datetime"] = pd.to_datetime(left["game_datetime"], utc=True)
    right["snapshot_time"] = pd.to_datetime(right["snapshot_time"], utc=True)
    left = left.sort_values(["game_datetime", left_by])
    right = right.sort_values(["snapshot_time", right_by])
    return pd.merge_asof(
        left,
        right,
        left_on="game_datetime",
        right_on="snapshot_time",
        left_by=left_by,
        right_by=right_by,
        direction="backward",
        allow_exact_matches=False,
    )


def attach_advanced_features(
    games: pd.DataFrame,
    team_snapshots: pd.DataFrame | None = None,
    starter_snapshots: pd.DataFrame | None = None,
    venue_factors: pd.DataFrame | None = None,
    weather_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    del venue_factors
    frame = games.copy()
    frame["game_datetime"] = pd.to_datetime(frame["game_datetime"], utc=True)
    base_cols = ["game_pk", "game_datetime", "home_team", "away_team"]
    for col in ["home_probable_pitcher_id", "away_probable_pitcher_id"]:
        if col in frame:
            base_cols.append(col)
    base = frame[base_cols].copy()

    team_metrics = [
        "offense_xwoba", "offense_hard_hit_rate", "offense_barrel_rate",
        "offense_k_minus_bb", "bullpen_xwoba_allowed",
        "bullpen_pitches_last_3d", "bullpen_pitchers_used_last_3d", "park_hr_factor",
    ]
    if team_snapshots is not None and not team_snapshots.empty:
        for side in ("home", "away"):
            left = base[["game_pk", "game_datetime", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
            joined = _latest_strictly_before(left, team_snapshots, "team", "team")
            keep = ["game_pk"] + [c for c in team_metrics if c in joined]
            renamed = joined[keep].rename(columns={c: f"{side}_{c}" for c in keep if c != "game_pk"})
            base = base.merge(renamed, on="game_pk", how="left")

    starter_metrics = ["starter_xwoba_allowed", "starter_k_minus_bb", "starter_projected_innings"]
    if starter_snapshots is not None and not starter_snapshots.empty:
        ss = starter_snapshots.copy()
        ss["pitcher_id"] = pd.to_numeric(ss["pitcher_id"], errors="coerce")
        for side in ("home", "away"):
            id_col = f"{side}_probable_pitcher_id"
            if id_col not in base:
                continue
            left = base[["game_pk", "game_datetime", id_col]].rename(columns={id_col: "pitcher_id"})
            left["pitcher_id"] = pd.to_numeric(left["pitcher_id"], errors="coerce")
            valid = left[left["pitcher_id"].notna()]
            if valid.empty:
                continue
            joined = _latest_strictly_before(valid, ss, "pitcher_id", "pitcher_id")
            keep = ["game_pk"] + [c for c in starter_metrics if c in joined]
            renamed = joined[keep].rename(columns={c: f"{side}_{c}" for c in keep if c != "game_pk"})
            base = base.merge(renamed, on="game_pk", how="left")

    weather_metrics = ["temperature_f", "humidity_pct", "wind_speed_mph", "precipitation_in", "roof_control_factor"]
    if weather_snapshots is not None and not weather_snapshots.empty:
        ws = weather_snapshots.copy()
        ws["forecast_issued_at"] = pd.to_datetime(ws["forecast_issued_at"], utc=True)
        left = base[["game_pk", "game_datetime"]].sort_values(["game_datetime", "game_pk"])
        ws = ws.sort_values(["forecast_issued_at", "game_pk"])
        joined = pd.merge_asof(
            left,
            ws,
            left_on="game_datetime",
            right_on="forecast_issued_at",
            by="game_pk",
            direction="backward",
            allow_exact_matches=False,
        )
        base = base.merge(joined[["game_pk"] + [c for c in weather_metrics if c in joined]], on="game_pk", how="left")

    for side in ("home", "away"):
        for metric in team_metrics + starter_metrics:
            col = f"{side}_{metric}"
            if col not in base:
                base[col] = np.nan
            base[col] = pd.to_numeric(base[col], errors="coerce").fillna(DEFAULTS[metric])
    for metric in weather_metrics:
        if metric not in base:
            base[metric] = DEFAULTS[metric]
        base[metric] = pd.to_numeric(base[metric], errors="coerce").fillna(DEFAULTS[metric])

    out = pd.DataFrame({"game_pk": base["game_pk"]})
    out["offense_xwoba_diff"] = base["home_offense_xwoba"] - base["away_offense_xwoba"]
    out["offense_hard_hit_diff"] = base["home_offense_hard_hit_rate"] - base["away_offense_hard_hit_rate"]
    out["offense_barrel_diff"] = base["home_offense_barrel_rate"] - base["away_offense_barrel_rate"]
    out["offense_k_minus_bb_diff"] = base["home_offense_k_minus_bb"] - base["away_offense_k_minus_bb"]
    out["starter_xwoba_allowed_edge"] = base["away_starter_xwoba_allowed"] - base["home_starter_xwoba_allowed"]
    out["starter_k_minus_bb_diff"] = base["home_starter_k_minus_bb"] - base["away_starter_k_minus_bb"]
    out["starter_projected_innings_diff"] = base["home_starter_projected_innings"] - base["away_starter_projected_innings"]
    out["bullpen_xwoba_allowed_edge"] = base["away_bullpen_xwoba_allowed"] - base["home_bullpen_xwoba_allowed"]
    out["bullpen_fatigue_diff"] = base["home_bullpen_pitches_last_3d"] - base["away_bullpen_pitches_last_3d"]
    out["bullpen_pitchers_used_diff"] = base["home_bullpen_pitchers_used_last_3d"] - base["away_bullpen_pitchers_used_last_3d"]
    out["park_hr_factor"] = base["home_park_hr_factor"]
    for metric in weather_metrics:
        out[metric] = base[metric]
    return out
