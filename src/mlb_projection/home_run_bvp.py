from __future__ import annotations

import numpy as np
import pandas as pd

from .home_run_features import (
    BVP_PRIOR_BARREL_RATE,
    BVP_PRIOR_HR_RATE,
    BVP_PRIOR_PA,
    prepare_hr_pitches,
)

BVP_COLUMNS = [
    "bvp_pa",
    "bvp_hr",
    "bvp_barrels",
    "bvp_hr_shrunk",
    "bvp_barrel_shrunk",
    "bvp_reliability",
]


def true_bvp_history(pitches: pd.DataFrame) -> pd.DataFrame:
    """Build actual pitch-level batter-versus-pitcher history."""
    frame = prepare_hr_pitches(pitches)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "game_date",
                "player_id",
                "opposing_starter_id",
                "pa",
                "home_runs",
                "barrels",
            ]
        )

    history = (
        frame.groupby(
            ["game_pk", "game_date", "batter", "pitcher"],
            as_index=False,
        )
        .agg(
            pa=("pa", "sum"),
            home_runs=("hr", "sum"),
            barrels=("barrel", "sum"),
        )
        .rename(
            columns={
                "batter": "player_id",
                "pitcher": "opposing_starter_id",
            }
        )
    )
    history["pa"] = pd.to_numeric(history["pa"], errors="coerce").fillna(0)
    history = history[history["pa"].gt(0)].copy()
    history["player_id"] = pd.to_numeric(history["player_id"], errors="coerce").astype("Int64")
    history["opposing_starter_id"] = pd.to_numeric(
        history["opposing_starter_id"], errors="coerce"
    ).astype("Int64")
    history["game_date"] = pd.to_datetime(history["game_date"], utc=True, errors="coerce")
    return history.dropna(subset=["player_id", "opposing_starter_id", "game_date"])


def _event_key(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["game_date"], utc=True, errors="coerce").dt.normalize()
    game_ids = pd.to_numeric(frame["game_pk"], errors="coerce").fillna(0).astype("int64")
    return dates + pd.to_timedelta(game_ids.mod(1_000_000), unit="us")


def attach_true_bvp_features(
    training: pd.DataFrame,
    pair_history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only information available before each target game.

    Each pair snapshot contains cumulative PA, HR and barrels through the
    previous matchup. An as-of join preserves older BvP history even when the
    batter did not face the listed starter in the target game.
    """
    output = training.drop(columns=[c for c in BVP_COLUMNS if c in training], errors="ignore").copy()
    if output.empty:
        return output

    output["player_id"] = pd.to_numeric(output["player_id"], errors="coerce").astype("Int64")
    output["opposing_starter_id"] = pd.to_numeric(
        output["opposing_starter_id"], errors="coerce"
    ).astype("Int64")
    output["_event_key"] = _event_key(output)
    output["_row_order"] = np.arange(len(output))

    pairs = pair_history.copy()
    if not pairs.empty:
        pairs["player_id"] = pd.to_numeric(pairs["player_id"], errors="coerce").astype("Int64")
        pairs["opposing_starter_id"] = pd.to_numeric(
            pairs["opposing_starter_id"], errors="coerce"
        ).astype("Int64")
        pairs["_event_key"] = _event_key(pairs)
        pairs = pairs.dropna(
            subset=["player_id", "opposing_starter_id", "_event_key"]
        ).sort_values(
            ["player_id", "opposing_starter_id", "_event_key", "game_pk"]
        )
        grouped = pairs.groupby(
            ["player_id", "opposing_starter_id"], group_keys=False
        )
        pairs["bvp_pa"] = grouped["pa"].cumsum() - pairs["pa"]
        pairs["bvp_hr"] = grouped["home_runs"].cumsum() - pairs["home_runs"]
        pairs["bvp_barrels"] = grouped["barrels"].cumsum() - pairs["barrels"]
        snapshots = pairs[
            [
                "player_id",
                "opposing_starter_id",
                "_event_key",
                "bvp_pa",
                "bvp_hr",
                "bvp_barrels",
            ]
        ].copy()
    else:
        snapshots = pd.DataFrame(
            columns=[
                "player_id",
                "opposing_starter_id",
                "_event_key",
                "bvp_pa",
                "bvp_hr",
                "bvp_barrels",
            ]
        )

    valid = output.dropna(
        subset=["player_id", "opposing_starter_id", "_event_key"]
    ).copy()
    invalid = output.loc[~output.index.isin(valid.index)].copy()

    if not valid.empty and not snapshots.empty:
        valid = valid.sort_values(
            ["player_id", "opposing_starter_id", "_event_key"]
        )
        snapshots = snapshots.sort_values(
            ["player_id", "opposing_starter_id", "_event_key"]
        )
        valid = pd.merge_asof(
            valid,
            snapshots,
            on="_event_key",
            by=["player_id", "opposing_starter_id"],
            direction="backward",
            allow_exact_matches=True,
        )
    else:
        for column in ["bvp_pa", "bvp_hr", "bvp_barrels"]:
            valid[column] = np.nan

    for column in ["bvp_pa", "bvp_hr", "bvp_barrels"]:
        invalid[column] = np.nan

    combined = pd.concat([valid, invalid], ignore_index=True, sort=False)
    for column in ["bvp_pa", "bvp_hr", "bvp_barrels"]:
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)

    combined["bvp_hr_shrunk"] = (
        combined["bvp_hr"] + BVP_PRIOR_PA * BVP_PRIOR_HR_RATE
    ) / (combined["bvp_pa"] + BVP_PRIOR_PA)
    combined["bvp_barrel_shrunk"] = (
        combined["bvp_barrels"] + BVP_PRIOR_PA * BVP_PRIOR_BARREL_RATE
    ) / (combined["bvp_pa"] + BVP_PRIOR_PA)
    combined["bvp_reliability"] = combined["bvp_pa"] / (
        combined["bvp_pa"] + BVP_PRIOR_PA
    )
    return combined.sort_values("_row_order").drop(
        columns=["_event_key", "_row_order"], errors="ignore"
    ).reset_index(drop=True)
