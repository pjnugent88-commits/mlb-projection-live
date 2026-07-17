from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TEAM_CODE_MAP = {"CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH", "OAK": "ATH"}


def _get_json(url: str, params: dict, retries: int = 4) -> dict:
    last_error: Exception | None = None
    headers = {"User-Agent": "mlb-projection-live/2.1 (+GitHub Actions)"}
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def fetch_recent_statcast(start_date: str, end_date: str, cache_path: str | Path, chunk_days: int = 7) -> pd.DataFrame:
    try:
        from pybaseball import cache as pybaseball_cache
        from pybaseball import statcast
    except ImportError as exc:
        raise RuntimeError("Install pybaseball to pull Statcast data") from exc
    pybaseball_cache.enable()
    cache = Path(cache_path)
    cached = pd.read_pickle(cache) if cache.exists() else pd.DataFrame()
    start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    cursor = start
    if not cached.empty and "game_date" in cached:
        dates = pd.to_datetime(cached["game_date"], errors="coerce")
        covered = dates[(dates >= start) & (dates <= end)]
        if not covered.empty:
            cursor = covered.max().normalize() + pd.Timedelta(days=1)
    parts = [cached] if not cached.empty else []
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days - 1), end)
        frame = statcast(start_dt=cursor.strftime("%Y-%m-%d"), end_dt=chunk_end.strftime("%Y-%m-%d"), verbose=False, parallel=True)
        if frame is not None and not frame.empty:
            parts.append(frame)
        cursor = chunk_end + pd.Timedelta(days=1)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if combined.empty:
        return combined
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    combined = combined[combined["game_date"].between(start, end)].copy()
    keys = [c for c in ["game_pk", "at_bat_number", "pitch_number", "pitcher", "batter"] if c in combined]
    if keys:
        combined = combined.drop_duplicates(keys, keep="last")
    cache.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(cache)
    return combined


def _prepare_statcast(pitches: pd.DataFrame) -> pd.DataFrame:
    frame = pitches.copy()
    for col in ("home_team", "away_team"):
        frame[col] = frame[col].replace(TEAM_CODE_MAP)
    frame["game_date"] = pd.to_datetime(frame["game_date"], utc=True)
    if "game_type" in frame:
        frame = frame[frame["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    frame["bat_team"] = np.where(frame["inning_topbot"].eq("Top"), frame["away_team"], frame["home_team"])
    frame["field_team"] = np.where(frame["inning_topbot"].eq("Top"), frame["home_team"], frame["away_team"])
    frame["is_bbe"] = frame["launch_speed"].notna()
    frame["is_hard_hit"] = frame["launch_speed"].ge(95.0)
    frame["is_barrel"] = frame.get("launch_speed_angle", pd.Series(index=frame.index, dtype=float)).eq(6)
    frame["is_pa"] = frame["events"].notna()
    frame["is_k"] = frame["events"].isin(["strikeout", "strikeout_double_play"])
    frame["is_bb"] = frame["events"].isin(["walk", "intent_walk"])
    frame["is_hr"] = frame["events"].eq("home_run")
    expected = pd.to_numeric(frame["estimated_woba_using_speedangle"] if "estimated_woba_using_speedangle" in frame else pd.Series(np.nan, index=frame.index), errors="coerce")
    actual = pd.to_numeric(frame["woba_value"] if "woba_value" in frame else pd.Series(np.nan, index=frame.index), errors="coerce")
    frame["xwoba_value"] = expected.where(expected.notna(), actual)
    return frame


def build_daily_statcast_snapshot_history(pitches: pd.DataFrame, snapshot_dates: Iterable[pd.Timestamp], lookback_days: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pitches.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = _prepare_statcast(pitches)
    team_parts: list[pd.DataFrame] = []
    starter_parts: list[pd.DataFrame] = []
    for raw in sorted(pd.to_datetime(list(snapshot_dates), utc=True).normalize().unique()):
        snapshot = pd.Timestamp(raw)
        window = frame[(frame["game_date"] >= snapshot - pd.Timedelta(days=lookback_days)) & (frame["game_date"] < snapshot)].copy()
        if window.empty:
            continue
        batting = window.groupby("bat_team", dropna=True).agg(
            offense_xwoba=("xwoba_value", "mean"), bbe=("is_bbe", "sum"), hard_hits=("is_hard_hit", "sum"),
            barrels=("is_barrel", "sum"), pa=("is_pa", "sum"), strikeouts=("is_k", "sum"), walks=("is_bb", "sum"),
        ).reset_index().rename(columns={"bat_team": "team"})
        batting["offense_hard_hit_rate"] = batting["hard_hits"] / batting["bbe"].clip(lower=1)
        batting["offense_barrel_rate"] = batting["barrels"] / batting["bbe"].clip(lower=1)
        batting["offense_k_minus_bb"] = (batting["walks"] - batting["strikeouts"]) / batting["pa"].clip(lower=1)
        first_pitch = window.sort_values(["game_pk", "field_team", "at_bat_number", "pitch_number"]).groupby(["game_pk", "field_team"], as_index=False).first()[["game_pk", "field_team", "pitcher"]].rename(columns={"pitcher": "starter_id"})
        with_roles = window.merge(first_pitch, on=["game_pk", "field_team"], how="left")
        relievers = with_roles[with_roles["pitcher"] != with_roles["starter_id"]]
        fatigue = relievers[relievers["game_date"] >= snapshot - pd.Timedelta(days=3)].groupby("field_team").agg(
            bullpen_pitches_last_3d=("pitcher", "size"), bullpen_pitchers_used_last_3d=("pitcher", "nunique"),
        ).reset_index().rename(columns={"field_team": "team"})
        bullpen = relievers.groupby("field_team").agg(bullpen_xwoba_allowed=("xwoba_value", "mean")).reset_index().rename(columns={"field_team": "team"})
        game_hr = window.groupby(["game_pk", "home_team"], as_index=False).agg(home_runs_at_park=("is_hr", "sum"))
        league_hr = float(game_hr["home_runs_at_park"].mean()) if not game_hr.empty else 2.2
        park = game_hr.groupby("home_team", as_index=False)["home_runs_at_park"].mean().rename(columns={"home_team": "team"})
        park["park_hr_factor"] = (park["home_runs_at_park"] / max(league_hr, 0.1)).clip(0.65, 1.45)
        team = batting.merge(bullpen, on="team", how="outer").merge(fatigue, on="team", how="outer").merge(park[["team", "park_hr_factor"]], on="team", how="left")
        team["snapshot_time"] = snapshot
        team_parts.append(team)
        starts = with_roles[with_roles["pitcher"] == with_roles["starter_id"]]
        starter = starts.groupby("pitcher").agg(
            starter_xwoba_allowed=("xwoba_value", "mean"), batters_faced=("is_pa", "sum"),
            strikeouts=("is_k", "sum"), walks=("is_bb", "sum"), starts=("game_pk", "nunique"), pitches=("pitcher", "size"),
        ).reset_index().rename(columns={"pitcher": "pitcher_id"})
        starter["starter_k_minus_bb"] = (starter["strikeouts"] - starter["walks"]) / starter["batters_faced"].clip(lower=1)
        starter["starter_projected_innings"] = (starter["pitches"] / starter["starts"].clip(lower=1) / 15.5).clip(3.0, 7.0)
        starter["snapshot_time"] = snapshot
        starter_parts.append(starter)
    return (pd.concat(team_parts, ignore_index=True) if team_parts else pd.DataFrame(), pd.concat(starter_parts, ignore_index=True) if starter_parts else pd.DataFrame())


def _nearest_values(hourly: dict, game_time: pd.Timestamp, names: dict[str, str]) -> dict[str, float]:
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    if len(times) == 0:
        return {}
    idx = int(np.argmin(np.abs(times - game_time)))
    values: dict[str, float] = {}
    for output_name, source_name in names.items():
        series = hourly.get(source_name, [])
        if idx < len(series) and series[idx] is not None:
            values[output_name] = float(series[idx])
    return values


def fetch_historical_pregame_weather(games: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    merged = games.merge(venues, on="venue", how="left")
    variables = {"temperature_f": "temperature_2m_previous_day1", "humidity_pct": "relative_humidity_2m_previous_day1", "wind_speed_mph": "wind_speed_10m_previous_day1", "precipitation_in": "precipitation_previous_day1"}
    rows: list[dict] = []
    for _, group in merged.dropna(subset=["latitude", "longitude"]).groupby("venue"):
        payload = _get_json(PREVIOUS_RUNS_URL, {
            "latitude": float(group["latitude"].iloc[0]), "longitude": float(group["longitude"].iloc[0]),
            "start_date": pd.to_datetime(group["game_datetime"], utc=True).min().date().isoformat(),
            "end_date": pd.to_datetime(group["game_datetime"], utc=True).max().date().isoformat(),
            "hourly": ",".join(variables.values()), "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "precipitation_unit": "inch", "timezone": "UTC",
        })
        hourly = payload.get("hourly", {})
        for _, game in group.iterrows():
            game_time = pd.Timestamp(game["game_datetime"])
            values = _nearest_values(hourly, game_time, variables)
            roof = game.get("roof_control_factor", 0.0)
            values.update({"game_pk": int(game["game_pk"]), "forecast_issued_at": game_time - pd.Timedelta(hours=24), "roof_control_factor": float(roof) if pd.notna(roof) else 0.0, "weather_source": "open-meteo-previous-day1"})
            rows.append(values)
    return pd.DataFrame(rows)


def fetch_open_meteo_game_weather(games: pd.DataFrame, venues: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()
    merged = games.merge(venues, on="venue", how="left")
    issued = pd.Timestamp.now(tz="UTC")
    variables = {"temperature_f": "temperature_2m", "humidity_pct": "relative_humidity_2m", "wind_speed_mph": "wind_speed_10m", "precipitation_in": "precipitation"}
    rows: list[dict] = []
    for _, group in merged.dropna(subset=["latitude", "longitude"]).groupby("venue"):
        payload = _get_json(FORECAST_URL, {
            "latitude": float(group["latitude"].iloc[0]), "longitude": float(group["longitude"].iloc[0]),
            "hourly": ",".join(variables.values()), "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "precipitation_unit": "inch", "timezone": "UTC", "forecast_days": 16,
        })
        hourly = payload.get("hourly", {})
        for _, game in group.iterrows():
            values = _nearest_values(hourly, pd.Timestamp(game["game_datetime"]), variables)
            roof = game.get("roof_control_factor", 0.0)
            values.update({"game_pk": int(game["game_pk"]), "forecast_issued_at": issued, "roof_control_factor": float(roof) if pd.notna(roof) else 0.0, "weather_source": "open-meteo-live"})
            rows.append(values)
    return pd.DataFrame(rows)
