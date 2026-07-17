from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .odds import expected_value_per_unit, probability_to_american
from .prop_constants import BATTER_FEATURES, MARKET_LABELS, PITCHER_FEATURES, TARGETS

def _linear_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=0.30, max_iter=1200)),
    ])

def _tree_model(random_state: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=60, max_depth=10, min_samples_leaf=14, max_features=0.80,
            n_jobs=-1, random_state=random_state,
        )),
    ])

@dataclass
class PropMarketModel:
    market_key: str
    linear: Pipeline
    tree: Pipeline
    tree_target_index: int
    feature_columns: list[str]
    tree_weight: float
    standardized_residuals: np.ndarray
    metrics: dict

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        linear = self.linear.predict(X[self.feature_columns])
        tree_prediction = self.tree.predict(X[self.feature_columns])
        tree = tree_prediction[:, self.tree_target_index] if getattr(tree_prediction, "ndim", 1) > 1 else tree_prediction
        return np.maximum((1.0 - self.tree_weight) * linear + self.tree_weight * tree, 0.01)

    def over_probability(self, means: Iterable[float], lines: Iterable[float]) -> np.ndarray:
        results: list[float] = []
        residuals = np.asarray(self.standardized_residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        for mean, line in zip(means, lines):
            mean = max(float(mean), 0.01)
            line = float(line)
            if self.market_key == "batter_home_runs" and line <= 0.5:
                results.append(float(np.clip(1.0 - math.exp(-mean), 0.01, 0.99)))
                continue
            threshold = (line - mean) / math.sqrt(max(mean, 1.0))
            if residuals.size >= 25:
                probability = (float(np.sum(residuals > threshold)) + 1.0) / (float(residuals.size) + 2.0)
            else:
                sigma = max(float(self.metrics.get("residual_std", 1.0)), 0.5)
                z = (line - mean) / sigma
                probability = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
            results.append(float(np.clip(probability, 0.01, 0.99)))
        return np.asarray(results)

@dataclass
class PlayerPropBundle:
    markets: dict[str, PropMarketModel]
    metrics: dict

def train_player_prop_models(
    batter_frame: pd.DataFrame,
    pitcher_frame: pd.DataFrame,
    test_fraction: float = 0.20,
    tree_weight: float = 0.55,
    minimum_pitcher_rows: int = 300,
    minimum_batter_rows: int = 1200,
    random_state: int = 42,
) -> PlayerPropBundle:
    markets: dict[str, PropMarketModel] = {}
    all_metrics: dict[str, object] = {"model_version": "stage3-player-props", "markets": {}}
    frame_specs = {
        "pitcher": (pitcher_frame, PITCHER_FEATURES, minimum_pitcher_rows, ["strikeouts", "outs", "hits_allowed"]),
        "batter": (batter_frame, BATTER_FEATURES, minimum_batter_rows, ["hits", "total_bases", "home_runs"]),
    }
    target_to_market = {target: market for market, (_, target) in TARGETS.items()}
    for frame_type, (source_frame, features, minimum, targets) in frame_specs.items():
        ordered = source_frame.sort_values(["game_date", "game_pk", "player_id"]).reset_index(drop=True)
        ordered = ordered.dropna(subset=targets, how="any").copy()
        if len(ordered) < minimum:
            raise ValueError(f"{frame_type} props require at least {minimum} rows; received {len(ordered)}")
        split = int(len(ordered) * (1.0 - test_fraction))
        split = min(max(split, 1), len(ordered) - 1)
        train, test = ordered.iloc[:split], ordered.iloc[split:]
        shared_tree = _tree_model(random_state)
        shared_tree.fit(train[features], train[targets])
        tree_test = np.asarray(shared_tree.predict(test[features]))
        if tree_test.ndim == 1:
            tree_test = tree_test.reshape(-1, 1)
        market_linears: dict[str, Pipeline] = {}
        for target_index, target in enumerate(targets):
            market_key = target_to_market[target]
            linear = _linear_model()
            linear.fit(train[features], train[target])
            prediction = np.maximum(
                (1.0 - tree_weight) * linear.predict(test[features]) + tree_weight * tree_test[:, target_index],
                0.01,
            )
            residuals = test[target].to_numpy(dtype=float) - prediction
            standardized = residuals / np.sqrt(np.maximum(prediction, 1.0))
            metrics = {
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "test_start_date": str(pd.Timestamp(test["game_date"].min()).date()),
                "test_end_date": str(pd.Timestamp(test["game_date"].max()).date()),
                "mae": float(mean_absolute_error(test[target], prediction)),
                "rmse": float(mean_squared_error(test[target], prediction) ** 0.5),
                "target_mean": float(test[target].mean()),
                "prediction_mean": float(prediction.mean()),
                "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0,
            }
            market_linears[market_key] = linear
            all_metrics["markets"][market_key] = metrics
            markets[market_key] = PropMarketModel(
                market_key,
                linear,
                shared_tree,
                target_index,
                features,
                tree_weight,
                standardized[-2000:],
                metrics,
            )
        shared_tree.fit(ordered[features], ordered[targets])
        for market_key, linear in market_linears.items():
            target = TARGETS[market_key][1]
            linear.fit(ordered[features], ordered[target])
            markets[market_key].linear = linear
            markets[market_key].tree = shared_tree
    all_metrics["pitcher_training_rows"] = int(len(pitcher_frame))
    all_metrics["batter_training_rows"] = int(len(batter_frame))
    return PlayerPropBundle(markets=markets, metrics=all_metrics)

def default_prop_line(market_key: str, mean: float) -> float:
    mean = max(float(mean), 0.0)
    if market_key in {"batter_hits", "batter_home_runs"}:
        return 0.5
    if market_key == "batter_total_bases":
        return 1.5
    return max(0.5, math.floor(mean) + 0.5)

def _safe_fair_odds(probability: float) -> int:
    return probability_to_american(float(np.clip(probability, 0.01, 0.99)))

def score_prop_market(
    features: pd.DataFrame,
    market_model: PropMarketModel,
    market_key: str,
    market_odds: pd.DataFrame | None = None,
    minimum_edge: float = 0.025,
    minimum_ev: float = 0.02,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    output = features.copy()
    output["market_key"] = market_key
    output["market_label"] = MARKET_LABELS[market_key]
    output["projection"] = market_model.predict_mean(output)
    odds = pd.DataFrame() if market_odds is None else market_odds.copy()
    if not odds.empty:
        odds = odds[odds["market_key"].eq(market_key)].copy()
        join_cols = ["market_key", "player_name", "game_pk"]
        output = output.merge(odds, on=join_cols, how="left", suffixes=("", "_market"))
    if "point" not in output:
        output["point"] = np.nan
    output["line"] = [float(point) if pd.notna(point) else default_prop_line(market_key, mean) for point, mean in zip(output["point"], output["projection"])]
    output["over_probability"] = market_model.over_probability(output["projection"], output["line"])
    output["under_probability"] = 1.0 - output["over_probability"]
    output["over_fair_odds"] = output["over_probability"].map(_safe_fair_odds)
    output["under_fair_odds"] = output["under_probability"].map(_safe_fair_odds)
    for col in ["over_odds", "under_odds", "over_no_vig_probability", "under_no_vig_probability", "over_book", "under_book"]:
        if col not in output:
            output[col] = np.nan if "book" not in col else None
    output["over_edge"] = output["over_probability"] - pd.to_numeric(output["over_no_vig_probability"], errors="coerce")
    output["under_edge"] = output["under_probability"] - pd.to_numeric(output["under_no_vig_probability"], errors="coerce")
    output["over_ev_per_unit"] = [expected_value_per_unit(p, o) if pd.notna(o) else math.nan for p, o in zip(output["over_probability"], output["over_odds"])]
    output["under_ev_per_unit"] = [expected_value_per_unit(p, o) if pd.notna(o) else math.nan for p, o in zip(output["under_probability"], output["under_odds"])]
    signals: list[str] = []
    for _, row in output.iterrows():
        if pd.notna(row["over_odds"]) or pd.notna(row["under_odds"]):
            over_value = pd.notna(row["over_edge"]) and row["over_edge"] >= minimum_edge and row["over_ev_per_unit"] >= minimum_ev
            under_value = pd.notna(row["under_edge"]) and row["under_edge"] >= minimum_edge and row["under_ev_per_unit"] >= minimum_ev
            if over_value and (not under_value or row["over_ev_per_unit"] >= row["under_ev_per_unit"]):
                signals.append("OVER VALUE")
            elif under_value:
                signals.append("UNDER VALUE")
            else:
                signals.append("PASS")
        elif market_key == "batter_home_runs":
            signals.append("HOME RUN WATCH" if row["over_probability"] >= 0.18 else "PASS")
        elif row["over_probability"] >= 0.60:
            signals.append("OVER LEAN")
        elif row["under_probability"] >= 0.60:
            signals.append("UNDER LEAN")
        else:
            signals.append("PASS")
    output["signal"] = signals
    output["confidence"] = (output["over_probability"] - 0.5).abs() * 2.0
    return output

def project_player_props(
    bundle: PlayerPropBundle,
    pitcher_features: pd.DataFrame,
    batter_features: pd.DataFrame,
    market_odds: pd.DataFrame | None,
    minimum_edge: float,
    minimum_ev: float,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for market_key, model in bundle.markets.items():
        feature_frame = pitcher_features if TARGETS[market_key][0] == "pitcher" else batter_features
        scored = score_prop_market(feature_frame, model, market_key, market_odds, minimum_edge, minimum_ev)
        if not scored.empty:
            outputs.append(scored)
    if not outputs:
        return pd.DataFrame()
    props = pd.concat(outputs, ignore_index=True)
    market_available = props[["over_odds", "under_odds"]].notna().any(axis=1)
    props["best_ev_per_unit"] = props[["over_ev_per_unit", "under_ev_per_unit"]].max(axis=1, skipna=True)
    props.loc[~market_available, "best_ev_per_unit"] = np.nan
    signal_rank = props["signal"].map(lambda value: 3 if "VALUE" in str(value) else (2 if ("LEAN" in str(value) or "WATCH" in str(value)) else 0))
    sort_value = props["best_ev_per_unit"].fillna(props["confidence"])
    return props.assign(_signal_rank=signal_rank, _sort=sort_value).sort_values(
        ["_signal_rank", "_sort", "game_datetime"], ascending=[False, False, True]
    ).drop(columns=["_signal_rank", "_sort"]).reset_index(drop=True)

def save_player_prop_bundle(bundle: PlayerPropBundle, model_dir: str | Path) -> Path:
    path = Path(model_dir) / "player_prop_bundle.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path
