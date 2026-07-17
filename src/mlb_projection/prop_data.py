from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .prop_constants import BATTER_FEATURES, EVENT_OUTS, HIT_BASES, PITCHER_FEATURES, SWING_DESCRIPTIONS, TEAM_CODE_MAP, WHIFF_DESCRIPTIONS

def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)

def prepare_statcast_for_props(pitches: pd.DataFrame) -> pd.DataFrame:
    if pitches.empty:
        return pitches.copy()
    frame = pitches.copy()
    for col in ("home_team", "away_team"):
        if col in frame:
            frame[col] = frame[col].replace(TEAM_CODE_MAP)
    frame["game_date"] = pd.to_datetime(frame["game_date"], utc=True, errors="coerce")
    if "game_type" in frame:
        frame = frame[frame["game_type"].isin(["R", "F", "D", "L", "W"])].copy()
    frame["bat_team"] = np.where(frame["inning_topbot"].eq("Top"), frame["away_team"], frame["home_team"])
    frame["field_team"] = np.where(frame["inning_topbot"].eq("Top"), frame["home_team"], frame["away_team"])
    frame["event"] = frame.get("events", pd.Series(index=frame.index, dtype=object)).fillna("")
    frame["description_clean"] = frame.get("description", pd.Series(index=frame.index, dtype=object)).fillna("")
    frame["is_pa"] = frame["event"].ne("")
    frame["is_hit"] = frame["event"].isin(HIT_BASES)
    frame["total_bases"] = frame["event"].map(HIT_BASES).fillna(0).astype(float)
    frame["is_hr"] = frame["event"].eq("home_run")
    frame["is_k"] = frame["event"].isin(["strikeout", "strikeout_double_play"])
    frame["is_bb"] = frame["event"].isin(["walk", "intent_walk"])
    frame["event_outs"] = frame["event"].map(EVENT_OUTS).fillna(0).astype(float)
    frame["is_swing"] = frame["description_clean"].isin(SWING_DESCRIPTIONS)
    frame["is_whiff"] = frame["description_clean"].isin(WHIFF_DESCRIPTIONS)
    frame["is_called_strike"] = frame["description_clean"].eq("called_strike")
    frame["is_bbe"] = frame.get("launch_speed", pd.Series(index=frame.index, dtype=float)).notna()
    frame["is_hard_hit"] = _numeric(frame, "launch_speed", np.nan).ge(95.0)
    frame["is_barrel"] = _numeric(frame, "launch_speed_angle", np.nan).eq(6)
    expected = _numeric(frame, "estimated_woba_using_speedangle", np.nan)
    actual = _numeric(frame, "woba_value", np.nan)
    frame["xwoba_value"] = expected.where(expected.notna(), actual)
    sort_cols = [c for c in ["game_pk", "field_team", "inning", "at_bat_number", "pitch_number"] if c in frame]
    first = frame.sort_values(sort_cols).groupby(["game_pk", "field_team"], as_index=False).first()[["game_pk", "field_team", "pitcher"]]
    first = first.rename(columns={"pitcher": "starter_id"})
    return frame.merge(first, on=["game_pk", "field_team"], how="left")

def build_player_game_tables(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = prepare_statcast_for_props(pitches)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    order_source = frame[frame["is_pa"]].groupby(["game_pk", "bat_team", "batter"], as_index=False)["at_bat_number"].min()
    order_source["batting_order"] = order_source.groupby(["game_pk", "bat_team"])["at_bat_number"].rank(method="first")

    batter = frame.groupby(["game_pk", "game_date", "bat_team", "field_team", "home_team", "away_team", "batter"], as_index=False).agg(
        hits=("is_hit", "sum"), total_bases=("total_bases", "sum"), home_runs=("is_hr", "sum"),
        pa=("is_pa", "sum"), strikeouts=("is_k", "sum"), walks=("is_bb", "sum"),
        xwoba=("xwoba_value", "mean"), bbe=("is_bbe", "sum"), hard_hits=("is_hard_hit", "sum"),
        barrels=("is_barrel", "sum"), opposing_starter_id=("starter_id", "first"),
    ).rename(columns={"bat_team": "team", "field_team": "opponent", "batter": "player_id"})
    batter = batter.merge(order_source.rename(columns={"bat_team": "team", "batter": "player_id"})[["game_pk", "team", "player_id", "batting_order"]], on=["game_pk", "team", "player_id"], how="left")
    batter["hit_rate"] = batter["hits"] / batter["pa"].clip(lower=1)
    batter["tb_rate"] = batter["total_bases"] / batter["pa"].clip(lower=1)
    batter["hr_rate"] = batter["home_runs"] / batter["pa"].clip(lower=1)
    batter["k_rate"] = batter["strikeouts"] / batter["pa"].clip(lower=1)
    batter["bb_rate"] = batter["walks"] / batter["pa"].clip(lower=1)
    batter["hard_hit_rate"] = batter["hard_hits"] / batter["bbe"].clip(lower=1)
    batter["barrel_rate"] = batter["barrels"] / batter["bbe"].clip(lower=1)
    batter["is_home"] = batter["team"].eq(batter["home_team"]).astype(int)
    batter = batter[batter["batting_order"].between(1, 9, inclusive="both")].copy()

    starts = frame[frame["pitcher"].eq(frame["starter_id"])].copy()
    pitcher = starts.groupby(["game_pk", "game_date", "field_team", "bat_team", "home_team", "away_team", "pitcher"], as_index=False).agg(
        strikeouts=("is_k", "sum"), outs=("event_outs", "sum"), hits_allowed=("is_hit", "sum"),
        walks=("is_bb", "sum"), pitches=("pitcher", "size"), batters_faced=("is_pa", "sum"),
        swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"), called_strikes=("is_called_strike", "sum"),
        xwoba_allowed=("xwoba_value", "mean"), velocity=("release_speed", "mean"),
    ).rename(columns={"field_team": "team", "bat_team": "opponent", "pitcher": "player_id"})
    pitcher = pitcher[(pitcher["pitches"] >= 15) & (pitcher["outs"] >= 1)].copy()
    pitcher["k_rate"] = pitcher["strikeouts"] / pitcher["batters_faced"].clip(lower=1)
    pitcher["whiff_rate"] = pitcher["whiffs"] / pitcher["swings"].clip(lower=1)
    pitcher["called_strike_rate"] = pitcher["called_strikes"] / pitcher["pitches"].clip(lower=1)
    pitcher["hits_allowed_rate"] = pitcher["hits_allowed"] / pitcher["batters_faced"].clip(lower=1)
    pitcher["walk_rate"] = pitcher["walks"] / pitcher["batters_faced"].clip(lower=1)
    pitcher["is_home"] = pitcher["team"].eq(pitcher["home_team"]).astype(int)

    team = batter.groupby(["game_pk", "game_date", "team", "opponent"], as_index=False).agg(
        strikeouts=("strikeouts", "sum"), hits=("hits", "sum"), total_bases=("total_bases", "sum"), pa=("pa", "sum"),
    )
    team["k_rate"] = team["strikeouts"] / team["pa"].clip(lower=1)
    return batter, pitcher, team

def _add_shifted_rolling(frame: pd.DataFrame, group_col: str, sources: dict[str, str], window: int, min_periods: int = 3) -> pd.DataFrame:
    out = frame.sort_values([group_col, "game_date", "game_pk"]).copy()
    grouped = out.groupby(group_col, group_keys=False)
    for source, target in sources.items():
        out[target] = grouped[source].transform(lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())
    out["prior_games"] = grouped.cumcount()
    return out

def _context_columns(context: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["game_pk", "park_hr_factor", "temperature_f", "humidity_pct", "wind_speed_mph", "precipitation_in", "roof_control_factor"]
    if context is None or context.empty:
        return pd.DataFrame(columns=columns)
    out = context.copy()
    for col in columns:
        if col not in out:
            out[col] = np.nan
    return out[columns].drop_duplicates("game_pk", keep="last")

def build_prop_training_frames(pitches: pd.DataFrame, context: pd.DataFrame | None, rolling_games: int = 15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batter_raw, pitcher_raw, team_raw = build_player_game_tables(pitches)
    if batter_raw.empty or pitcher_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), batter_raw, pitcher_raw, team_raw

    pitcher_sources = {
        "strikeouts": "pitcher_k_per_start", "outs": "pitcher_outs_per_start", "hits_allowed": "pitcher_hits_allowed_per_start",
        "pitches": "pitcher_pitches_per_start", "batters_faced": "pitcher_bf_per_start", "k_rate": "pitcher_k_rate",
        "whiff_rate": "pitcher_whiff_rate", "called_strike_rate": "pitcher_called_strike_rate", "xwoba_allowed": "pitcher_xwoba_allowed",
        "hits_allowed_rate": "pitcher_hits_allowed_rate", "walk_rate": "pitcher_walk_rate", "velocity": "pitcher_velocity",
    }
    pitcher_features = _add_shifted_rolling(pitcher_raw, "player_id", pitcher_sources, rolling_games)

    team_sources = {"k_rate": "opponent_k_rate", "hits": "opponent_hits_per_game", "total_bases": "opponent_tb_per_game", "pa": "opponent_pa_per_game"}
    team_features = _add_shifted_rolling(team_raw, "team", team_sources, rolling_games)
    pitcher_features = pitcher_features.merge(
        team_features[["game_pk", "team"] + list(team_sources.values())].rename(columns={"team": "opponent"}),
        on=["game_pk", "opponent"], how="left",
    )

    batter_sources = {
        "hits": "batter_hits_per_game", "total_bases": "batter_tb_per_game", "home_runs": "batter_hr_per_game",
        "pa": "batter_pa_per_game", "hit_rate": "batter_hit_rate", "tb_rate": "batter_tb_rate", "hr_rate": "batter_hr_rate",
        "k_rate": "batter_k_rate", "bb_rate": "batter_bb_rate", "xwoba": "batter_xwoba",
        "hard_hit_rate": "batter_hard_hit_rate", "barrel_rate": "batter_barrel_rate",
    }
    batter_features = _add_shifted_rolling(batter_raw, "player_id", batter_sources, rolling_games)
    opponent_pitcher = pitcher_features[[
        "game_pk", "player_id", "pitcher_k_rate", "pitcher_xwoba_allowed", "pitcher_hits_allowed_rate", "pitcher_outs_per_start"
    ]].rename(columns={
        "player_id": "opposing_starter_id", "pitcher_k_rate": "opposing_pitcher_k_rate",
        "pitcher_xwoba_allowed": "opposing_pitcher_xwoba_allowed",
        "pitcher_hits_allowed_rate": "opposing_pitcher_hits_allowed_rate",
        "pitcher_outs_per_start": "opposing_pitcher_outs_per_start",
    })
    batter_features["opposing_starter_id"] = pd.to_numeric(batter_features["opposing_starter_id"], errors="coerce").astype("Int64")
    opponent_pitcher["opposing_starter_id"] = pd.to_numeric(opponent_pitcher["opposing_starter_id"], errors="coerce").astype("Int64")
    batter_features = batter_features.merge(opponent_pitcher, on=["game_pk", "opposing_starter_id"], how="left")

    game_context = _context_columns(context)
    pitcher_features = pitcher_features.merge(game_context, on="game_pk", how="left")
    batter_features = batter_features.merge(game_context, on="game_pk", how="left")
    pitcher_features = pitcher_features[pitcher_features["prior_games"] >= 3].copy()
    batter_features = batter_features[batter_features["prior_games"] >= 3].copy()
    return batter_features, pitcher_features, batter_raw, pitcher_raw, team_raw

def _recent_player_features(frame: pd.DataFrame, player_id: int, sources: dict[str, str], window: int) -> dict[str, float]:
    history = frame[pd.to_numeric(frame["player_id"], errors="coerce").eq(int(player_id))].sort_values(["game_date", "game_pk"]).tail(window)
    return {target: float(pd.to_numeric(history[source], errors="coerce").mean()) if not history.empty else math.nan for source, target in sources.items()}

def _recent_team_features(frame: pd.DataFrame, team: str, window: int) -> dict[str, float]:
    history = frame[frame["team"].eq(team)].sort_values(["game_date", "game_pk"]).tail(window)
    if history.empty:
        return {"opponent_k_rate": math.nan, "opponent_hits_per_game": math.nan, "opponent_tb_per_game": math.nan, "opponent_pa_per_game": math.nan}
    return {
        "opponent_k_rate": float(history["strikeouts"].sum() / max(history["pa"].sum(), 1)),
        "opponent_hits_per_game": float(history["hits"].mean()),
        "opponent_tb_per_game": float(history["total_bases"].mean()),
        "opponent_pa_per_game": float(history["pa"].mean()),
    }

def _game_context_lookup(slate: pd.DataFrame, weather: pd.DataFrame | None, venues: pd.DataFrame | None) -> pd.DataFrame:
    context = slate[["game_pk", "venue"]].copy()
    if weather is not None and not weather.empty:
        cols = [c for c in ["game_pk", "temperature_f", "humidity_pct", "wind_speed_mph", "precipitation_in", "roof_control_factor"] if c in weather]
        context = context.merge(weather[cols].drop_duplicates("game_pk", keep="last"), on="game_pk", how="left")
    if venues is not None and not venues.empty:
        available = [c for c in ["venue", "roof_control_factor"] if c in venues]
        if len(available) > 1:
            venue_context = venues[available].rename(columns={"roof_control_factor": "venue_roof_control_factor"})
            context = context.merge(venue_context, on="venue", how="left")
            if "roof_control_factor" not in context:
                context["roof_control_factor"] = context["venue_roof_control_factor"]
            else:
                context["roof_control_factor"] = context["roof_control_factor"].combine_first(context["venue_roof_control_factor"])
    if "park_hr_factor" not in context:
        context["park_hr_factor"] = 1.0
    for col, default in {
        "park_hr_factor": 1.0, "temperature_f": 72.0, "humidity_pct": 55.0,
        "wind_speed_mph": 7.0, "precipitation_in": 0.0, "roof_control_factor": 0.0,
    }.items():
        if col not in context:
            context[col] = default
        context[col] = pd.to_numeric(context[col], errors="coerce").fillna(default)
    return context.drop_duplicates("game_pk", keep="last")

def build_live_pitcher_features(
    slate: pd.DataFrame,
    pitcher_history: pd.DataFrame,
    team_history: pd.DataFrame,
    weather: pd.DataFrame | None,
    venues: pd.DataFrame | None,
    rolling_games: int,
) -> pd.DataFrame:
    pitcher_sources = {
        "strikeouts": "pitcher_k_per_start", "outs": "pitcher_outs_per_start", "hits_allowed": "pitcher_hits_allowed_per_start",
        "pitches": "pitcher_pitches_per_start", "batters_faced": "pitcher_bf_per_start", "k_rate": "pitcher_k_rate",
        "whiff_rate": "pitcher_whiff_rate", "called_strike_rate": "pitcher_called_strike_rate", "xwoba_allowed": "pitcher_xwoba_allowed",
        "hits_allowed_rate": "pitcher_hits_allowed_rate", "walk_rate": "pitcher_walk_rate", "velocity": "pitcher_velocity",
    }
    context = _game_context_lookup(slate, weather, venues).set_index("game_pk")
    rows: list[dict] = []
    for _, game in slate.iterrows():
        for side in ("away", "home"):
            raw_id = game.get(f"{side}_probable_pitcher_id")
            if pd.isna(raw_id):
                continue
            player_id = int(float(raw_id))
            opponent = str(game[f"home_team" if side == "away" else "away_team"])
            row = {
                "game_pk": int(game["game_pk"]), "game_datetime": game.get("game_datetime"),
                "away_team": game.get("away_team"), "home_team": game.get("home_team"),
                "team": game.get(f"{side}_team"), "opponent": opponent, "is_home": int(side == "home"),
                "player_id": player_id, "player_name": game.get(f"{side}_probable_pitcher") or str(player_id),
                "player_type": "pitcher", "lineup_status": "probable starter", "batting_order": math.nan,
            }
            row.update(_recent_player_features(pitcher_history, player_id, pitcher_sources, rolling_games))
            row.update(_recent_team_features(team_history, opponent, rolling_games))
            row.update(context.loc[int(game["game_pk"])].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)

def build_live_batter_features(
    lineups: pd.DataFrame,
    slate: pd.DataFrame,
    batter_history: pd.DataFrame,
    pitcher_history: pd.DataFrame,
    weather: pd.DataFrame | None,
    venues: pd.DataFrame | None,
    rolling_games: int,
) -> pd.DataFrame:
    if lineups is None or lineups.empty:
        return pd.DataFrame()
    batter_sources = {
        "hits": "batter_hits_per_game", "total_bases": "batter_tb_per_game", "home_runs": "batter_hr_per_game",
        "pa": "batter_pa_per_game", "hit_rate": "batter_hit_rate", "tb_rate": "batter_tb_rate", "hr_rate": "batter_hr_rate",
        "k_rate": "batter_k_rate", "bb_rate": "batter_bb_rate", "xwoba": "batter_xwoba",
        "hard_hit_rate": "batter_hard_hit_rate", "barrel_rate": "batter_barrel_rate",
    }
    pitcher_sources = {
        "k_rate": "opposing_pitcher_k_rate", "xwoba_allowed": "opposing_pitcher_xwoba_allowed",
        "hits_allowed_rate": "opposing_pitcher_hits_allowed_rate", "outs": "opposing_pitcher_outs_per_start",
    }
    context = _game_context_lookup(slate, weather, venues).set_index("game_pk")
    slate_by_game = slate.set_index("game_pk")
    rows: list[dict] = []
    for _, batter in lineups.iterrows():
        game_pk = int(batter["game_pk"])
        if game_pk not in slate_by_game.index:
            continue
        game = slate_by_game.loc[game_pk]
        side = str(batter["side"])
        opposing_id = game.get("home_probable_pitcher_id" if side == "away" else "away_probable_pitcher_id")
        row = {
            "game_pk": game_pk, "game_datetime": game.get("game_datetime"),
            "away_team": game.get("away_team"), "home_team": game.get("home_team"),
            "team": batter.get("team"), "opponent": game.get("home_team" if side == "away" else "away_team"),
            "is_home": int(side == "home"), "player_id": int(batter["player_id"]),
            "player_name": batter.get("player_name") or str(batter["player_id"]), "player_type": "batter",
            "lineup_status": batter.get("lineup_status", "confirmed"), "batting_order": float(batter.get("batting_order", math.nan)),
        }
        row.update(_recent_player_features(batter_history, int(batter["player_id"]), batter_sources, rolling_games))
        if pd.notna(opposing_id):
            row.update(_recent_player_features(pitcher_history, int(float(opposing_id)), pitcher_sources, rolling_games))
        else:
            row.update({target: math.nan for target in pitcher_sources.values()})
        row.update(context.loc[game_pk].to_dict())
        rows.append(row)
    return pd.DataFrame(rows)
