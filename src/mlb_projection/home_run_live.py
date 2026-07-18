from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .home_run_features import EXPECTED_PA, _live_bvp


def _safe_float(value) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame or frame.empty:
        return math.nan
    return _safe_float(pd.to_numeric(frame[column], errors="coerce").mean())


def _live_batter_safe(
    history: pd.DataFrame,
    player_id: int,
    long_window: int,
    recent_window: int,
) -> dict[str, float | str | None]:
    rows = history[
        pd.to_numeric(history["player_id"], errors="coerce").eq(player_id)
    ].sort_values(["game_date", "game_pk"])

    def summarize(window: int, suffix: str) -> dict[str, float]:
        sample = rows.tail(window)
        pa = max(_numeric_sum(sample, "pa"), 1.0)
        bbe = max(_numeric_sum(sample, "bbe"), 1.0)
        return {
            f"batter_hr_pa_{suffix}": _numeric_sum(sample, "home_runs") / pa,
            f"batter_barrel_pa_{suffix}": _numeric_sum(sample, "barrels") / pa,
            f"batter_barrel_bbe_{suffix}": _numeric_sum(sample, "barrels") / bbe,
            f"batter_hard_air_pa_{suffix}": _numeric_sum(sample, "hard_air") / pa,
            f"batter_sweet_pa_{suffix}": _numeric_sum(sample, "sweet") / pa,
            f"batter_k_rate_{suffix}": _numeric_sum(sample, "strikeouts") / pa,
            f"batter_bb_rate_{suffix}": _numeric_sum(sample, "walks") / pa,
            f"batter_xwoba_{suffix}": _numeric_mean(sample, "xwoba"),
            f"batter_ev90_{suffix}": _numeric_mean(sample, "ev90"),
        }

    output = summarize(long_window, "long")
    output.update(summarize(recent_window, "recent"))
    hands = rows.get("batter_hand", pd.Series(dtype=object)).dropna()
    output["batter_hand"] = str(hands.iloc[-1]) if not hands.empty else None
    return output


def _live_pitcher_safe(
    history: pd.DataFrame,
    player_id: int,
    window: int,
) -> dict[str, float | str | None]:
    rows = history[
        pd.to_numeric(history["player_id"], errors="coerce").eq(player_id)
    ].sort_values(["game_date", "game_pk"]).tail(window)
    if rows.empty:
        return {}
    bf = max(_numeric_sum(rows, "bf"), 1.0)
    bbe = max(_numeric_sum(rows, "bbe"), 1.0)
    hands = rows.get("pitcher_hand", pd.Series(dtype=object)).dropna()
    return {
        "pitcher_hr_bf": _numeric_sum(rows, "hr_allowed") / bf,
        "pitcher_barrel_bf": _numeric_sum(rows, "barrels") / bf,
        "pitcher_hard_air_bf": _numeric_sum(rows, "hard_air") / bf,
        "pitcher_fly_rate": _numeric_sum(rows, "fly") / bbe,
        "pitcher_k_rate": _numeric_sum(rows, "strikeouts") / bf,
        "pitcher_bb_rate": _numeric_sum(rows, "walks") / bf,
        "pitcher_xwoba": _numeric_mean(rows, "xwoba"),
        "pitcher_ev90": _numeric_mean(rows, "ev90"),
        "pitcher_outs": _numeric_mean(rows, "outs"),
        "opposing_pitcher_hand": str(hands.iloc[-1]) if not hands.empty else None,
    }


def build_live(
    lineups: pd.DataFrame,
    slate: pd.DataFrame,
    batter_history: pd.DataFrame,
    pitcher_history: pd.DataFrame,
    pair_history: pd.DataFrame,
    weather: pd.DataFrame | None,
    venues: pd.DataFrame | None,
    long_window: int = 60,
    recent_window: int = 15,
    pitcher_window: int = 15,
) -> pd.DataFrame:
    if lineups is None or lineups.empty or slate.empty:
        return pd.DataFrame()

    context = slate[["game_pk", "venue"]].copy()
    if weather is not None and not weather.empty:
        columns = [
            c for c in [
                "game_pk", "temperature_f", "humidity_pct", "wind_speed_mph",
                "precipitation_in", "roof_control_factor",
            ] if c in weather
        ]
        context = context.merge(
            weather[columns].drop_duplicates("game_pk", keep="last"),
            on="game_pk", how="left",
        )
    if "park_hr_factor" in slate:
        context = context.merge(
            slate[["game_pk", "park_hr_factor"]].drop_duplicates("game_pk"),
            on="game_pk", how="left",
        )
    defaults = {
        "park_hr_factor": 1.0,
        "temperature_f": 72.0,
        "humidity_pct": 55.0,
        "wind_speed_mph": 7.0,
        "precipitation_in": 0.0,
        "roof_control_factor": 0.0,
    }
    for column, default in defaults.items():
        if column not in context:
            context[column] = default
        context[column] = pd.to_numeric(
            context[column], errors="coerce"
        ).fillna(default)
    context = context.drop_duplicates("game_pk", keep="last").set_index("game_pk")
    games = slate.drop_duplicates("game_pk", keep="last").set_index("game_pk")

    rows: list[dict] = []
    for _, lineup_row in lineups.iterrows():
        game_pk = int(lineup_row["game_pk"])
        if game_pk not in games.index or game_pk not in context.index:
            continue
        game = games.loc[game_pk]
        side = str(lineup_row["side"])
        raw_pitcher = game.get(
            "home_probable_pitcher_id" if side == "away"
            else "away_probable_pitcher_id"
        )
        if pd.isna(raw_pitcher):
            continue
        batter_id = int(lineup_row["player_id"])
        pitcher_id = int(float(raw_pitcher))
        batting_order = int(float(lineup_row.get("batting_order", 9)))
        row = {
            "game_pk": game_pk,
            "game_datetime": game.get("game_datetime"),
            "away_team": game.get("away_team"),
            "home_team": game.get("home_team"),
            "team": lineup_row.get("team"),
            "opponent": game.get("home_team" if side == "away" else "away_team"),
            "is_home": int(side == "home"),
            "player_id": batter_id,
            "player_name": lineup_row.get("player_name") or str(batter_id),
            "lineup_status": lineup_row.get("lineup_status", "confirmed"),
            "batting_order": float(batting_order),
            "expected_pa": EXPECTED_PA.get(batting_order, 3.7),
            "opposing_starter_id": pitcher_id,
        }
        row.update(
            _live_batter_safe(
                batter_history, batter_id, long_window, recent_window
            )
        )
        row.update(_live_pitcher_safe(pitcher_history, pitcher_id, pitcher_window))
        row.update(_live_bvp(pair_history, batter_id, pitcher_id))
        row["same_hand"] = int(
            row.get("batter_hand") in {"L", "R"}
            and row.get("opposing_pitcher_hand") in {"L", "R"}
            and row.get("batter_hand") == row.get("opposing_pitcher_hand")
        )
        row.update(context.loc[game_pk].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)
