from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Stage2Bundle:
    linear_win: Pipeline
    tree_win: Pipeline
    linear_home_runs: Pipeline
    tree_home_runs: Pipeline
    linear_away_runs: Pipeline
    tree_away_runs: Pipeline
    feature_columns: list[str]
    tree_weight: float
    metrics: dict

    def predict_home_win(self, X: pd.DataFrame) -> np.ndarray:
        p_linear = self.linear_win.predict_proba(X)[:, 1]
        p_tree = self.tree_win.predict_proba(X)[:, 1]
        return (1.0 - self.tree_weight) * p_linear + self.tree_weight * p_tree

    def predict_runs(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        home = (1.0 - self.tree_weight) * self.linear_home_runs.predict(X) + self.tree_weight * self.tree_home_runs.predict(X)
        away = (1.0 - self.tree_weight) * self.linear_away_runs.predict(X) + self.tree_weight * self.tree_away_runs.predict(X)
        return np.maximum(home, 0.05), np.maximum(away, 0.05)


def _linear_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
    ])


def _tree_classifier() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=140, max_depth=9, min_samples_leaf=18,
            max_features=0.75, n_jobs=-1, random_state=42,
        )),
    ])


def _linear_run_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0.35, max_iter=1000)),
    ])


def _tree_run_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=140, max_depth=10, min_samples_leaf=16,
            max_features=0.80, n_jobs=-1, random_state=42,
        )),
    ])


def train_stage2_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    test_fraction: float,
    minimum_training_games: int,
    tree_weight: float,
) -> Stage2Bundle:
    frame = frame.sort_values(["game_datetime", "game_pk"]).reset_index(drop=True)
    if len(frame) < minimum_training_games:
        raise ValueError(f"Need at least {minimum_training_games} games; received {len(frame)}")
    split = int(len(frame) * (1.0 - test_fraction))
    train, test = frame.iloc[:split], frame.iloc[split:]
    X_train, X_test = train[feature_columns], test[feature_columns]

    lw, tw = _linear_classifier(), _tree_classifier()
    lhr, thr = _linear_run_model(), _tree_run_model()
    lar, tar = _linear_run_model(), _tree_run_model()
    lw.fit(X_train, train["home_win"]); tw.fit(X_train, train["home_win"])
    lhr.fit(X_train, train["home_runs"]); thr.fit(X_train, train["home_runs"])
    lar.fit(X_train, train["away_runs"]); tar.fit(X_train, train["away_runs"])

    p = (1.0 - tree_weight) * lw.predict_proba(X_test)[:, 1] + tree_weight * tw.predict_proba(X_test)[:, 1]
    home = np.maximum((1.0 - tree_weight) * lhr.predict(X_test) + tree_weight * thr.predict(X_test), 0.05)
    away = np.maximum((1.0 - tree_weight) * lar.predict(X_test) + tree_weight * tar.predict(X_test), 0.05)
    metrics = {
        "train_games": int(len(train)),
        "test_games": int(len(test)),
        "test_start_date": str(pd.Timestamp(test["game_datetime"].min()).date()),
        "test_end_date": str(pd.Timestamp(test["game_datetime"].max()).date()),
        "home_win_log_loss": float(log_loss(test["home_win"], p)),
        "home_win_brier_score": float(brier_score_loss(test["home_win"], p)),
        "home_runs_mae": float(mean_absolute_error(test["home_runs"], home)),
        "away_runs_mae": float(mean_absolute_error(test["away_runs"], away)),
        "total_runs_mae": float(mean_absolute_error(test["home_runs"] + test["away_runs"], home + away)),
        "model_version": "stage2-ensemble",
        "advanced_feature_count": len(feature_columns),
    }

    X_all = frame[feature_columns]
    for model, target in [
        (lw, "home_win"), (tw, "home_win"), (lhr, "home_runs"),
        (thr, "home_runs"), (lar, "away_runs"), (tar, "away_runs")
    ]:
        model.fit(X_all, frame[target])

    return Stage2Bundle(lw, tw, lhr, thr, lar, tar, feature_columns, tree_weight, metrics)


def save_stage2_bundle(bundle: Stage2Bundle, model_dir: str | Path) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "stage2_bundle.joblib")
