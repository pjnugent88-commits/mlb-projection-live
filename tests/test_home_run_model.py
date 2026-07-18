from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_projection.home_run_features import BVP_PRIOR_HR_RATE, _live_bvp
from mlb_projection.home_run_model import HR_FEATURES, score_home_runs, train_home_run_model


def synthetic_frame(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "game_date": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        "game_pk": np.arange(rows), "player_id": np.arange(rows) % 30,
    })
    for feature in HR_FEATURES:
        frame[feature] = rng.normal(size=rows)
    frame["expected_pa"] = rng.uniform(3.6, 4.8, rows)
    frame["batting_order"] = rng.integers(1, 10, rows)
    frame["is_home"] = rng.integers(0, 2, rows)
    frame["park_hr_factor"] = rng.uniform(.8, 1.25, rows)
    frame["bvp_pa"] = rng.integers(0, 30, rows)
    latent = -3.2 + .9 * frame["batter_barrel_pa_long"] + .4 * frame["pitcher_hr_bf"] + .25 * frame["park_hr_factor"]
    probability = 1 / (1 + np.exp(-latent))
    frame["target_hr"] = rng.binomial(1, np.clip(probability, .01, .45))
    frame.loc[frame.index[::17], "target_hr"] = 1
    frame.loc[frame.index[::19], "target_hr"] = 0
    return frame


def test_bvp_zero_sample_returns_prior():
    pair = pd.DataFrame(columns=["player_id", "opposing_starter_id", "pa", "home_runs", "barrels"])
    result = _live_bvp(pair, 1, 2)
    assert result["bvp_pa"] == 0
    assert result["bvp_reliability"] == 0
    assert abs(result["bvp_hr_shrunk"] - BVP_PRIOR_HR_RATE) < 1e-12


def test_home_run_model_outputs_probabilities_and_metrics():
    frame = synthetic_frame()
    model = train_home_run_model(frame, minimum_rows=200, random_state=4)
    probabilities = model.predict(frame.tail(10))
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert 0 <= model.metrics["brier_score"] <= 1
    assert model.metrics["model_version"] == "stage4-home-run-probability"


def test_scoring_uses_hr_value_thresholds():
    frame = synthetic_frame().tail(8).copy()
    frame["player_name"] = [f"Player {i}" for i in range(len(frame))]
    frame["away_team"] = "BOS"; frame["home_team"] = "NYY"; frame["game_datetime"] = pd.Timestamp("2026-07-18T23:00:00Z")
    model = train_home_run_model(synthetic_frame(), minimum_rows=200, random_state=5)
    odds = pd.DataFrame({
        "game_pk": frame["game_pk"], "player_name": frame["player_name"], "market_key": "batter_home_runs",
        "point": .5, "over_odds": 500, "under_odds": -700, "over_book": "TestBook", "under_book": "TestBook",
        "over_no_vig_probability": .14, "under_no_vig_probability": .86,
    })
    scored = score_home_runs(frame, model, odds, minimum_edge=-1, minimum_ev=-1)
    assert len(scored) == len(frame)
    assert scored["home_run_probability"].between(0, 1).all()
    assert scored["signal"].eq("HR VALUE").all()


def test_true_bvp_features_exclude_current_game_and_accumulate_prior_matchups():
    from mlb_projection.home_run_bvp import attach_true_bvp_features

    training = pd.DataFrame({
        "game_pk": [10, 20, 30],
        "game_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
        "player_id": [1, 1, 1],
        "opposing_starter_id": [2, 2, 2],
    })
    pair = pd.DataFrame({
        "game_pk": [10, 20, 30],
        "game_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True),
        "player_id": [1, 1, 1],
        "opposing_starter_id": [2, 2, 2],
        "pa": [3, 4, 2],
        "home_runs": [1, 0, 1],
        "barrels": [1, 1, 0],
    })
    result = attach_true_bvp_features(training, pair)
    assert result["bvp_pa"].tolist() == [0.0, 3.0, 7.0]
    assert result["bvp_hr"].tolist() == [0.0, 1.0, 1.0]
