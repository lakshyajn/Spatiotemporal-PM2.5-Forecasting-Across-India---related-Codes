"""Shared preprocessing, training, and reporting helpers for PM2.5 models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

N_SPLITS = 5
EVAL_LOG_PERIOD = 100
TOP_K_FEATURES = 10
CITY_NAME = "delhi"
CITY_MIN_ROWS = 5000

LGB_PARAMS_BASE: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_child_samples": 30,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "max_depth": -1,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

LGB_PARAMS_STRONG: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "min_child_samples": 50,
    "lambda_l1": 0.05,
    "lambda_l2": 0.1,
    "max_depth": 10,
    "min_gain_to_split": 0.01,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    candidates = (
        [Path(data_path)]
        if data_path
        else [
            Path("dataset_v4.parquet"),
            Path("/content/drive/MyDrive/dataset_v4.parquet"),
            Path("/content/drive/MyDrive/Colab Notebooks/dataset_v4.parquet"),
            Path("/content/dataset_v4.parquet"),
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No dataset found. Checked: {[str(path) for path in candidates]}"
    )


def load_base_data(data_path: str | Path | None = None) -> pd.DataFrame:
    path = resolve_data_path(data_path)
    df = pd.read_parquet(path)
    if "timestamp" not in df.columns:
        raise KeyError("The dataset must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = optimize_memory(df)
    print(f"Using dataset: {path}")
    print(f"Shape: {df.shape}")
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include=["float64"]).columns
    int_cols = out.select_dtypes(include=["int64"]).columns
    out[float_cols] = out[float_cols].astype("float32")
    out[int_cols] = out[int_cols].astype("int32")
    return out


def city_from_station(station: str) -> str:
    left = str(station).lower().split("_-_")[0]
    tokens = [token for token in re.split(r"[_\s]+", left) if token]
    tokens = [
        token for token in tokens
        if not token.isdigit() and re.search(r"[a-z]", token)
    ]
    return tokens[-1] if tokens else "unknown"


def preprocess_base(
    df: pd.DataFrame,
    with_extra_pollutants: bool = False,
    with_city_guess: bool = False,
) -> pd.DataFrame:
    out = df.sort_values("timestamp").reset_index(drop=True).copy()
    meteo_cols = [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "precipitation", "cloud_cover",
        "shortwave_radiation", "boundary_layer_height", "wind_speed",
        "wind_u", "wind_v",
    ]
    fire_cols = [
        "fire_count", "frp_sum", "frp_mean", "frp_max",
        "bright_ti4_mean", "bright_ti5_mean",
    ]

    existing_meteo = [col for col in meteo_cols if col in out.columns]
    existing_fire = [col for col in fire_cols if col in out.columns]
    if existing_meteo:
        out[existing_meteo] = out[existing_meteo].interpolate(
            method="linear", limit_direction="both"
        )
    if existing_fire:
        out[existing_fire] = out[existing_fire].fillna(0)
    for col in ["rush_hour", "pm25"]:
        if col in out.columns:
            out[col] = out[col].fillna(0) if col == "rush_hour" else out[col].interpolate(
                method="linear", limit_direction="both"
            )
    if with_extra_pollutants:
        for col in ["pm10", "no2", "so2", "o3"]:
            if col in out.columns:
                out[col] = out[col].interpolate(
                    method="linear", limit_direction="both"
                )
    if "station" not in out.columns:
        raise KeyError("The dataset must contain a 'station' column.")

    out["station_encoded"] = LabelEncoder().fit_transform(out["station"].astype(str))
    if {"wind_u", "wind_v"}.issubset(out.columns):
        out["wind_speed_computed"] = np.sqrt(out["wind_u"] ** 2 + out["wind_v"] ** 2)
    if with_city_guess:
        out["city_guess"] = out["station"].astype(str).map(city_from_station)
    return out


def add_engineered_features_nolag(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "wind_speed", "boundary_layer_height", "temperature_2m",
        "dew_point_2m", "surface_pressure", "relative_humidity_2m",
        "shortwave_radiation", "cloud_cover", "fire_count", "frp_mean",
        "precipitation", "timestamp", "is_winter",
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(f"Missing columns for engineered features: {missing}")
    out["ventilation_coeff"] = out["wind_speed"] * out["boundary_layer_height"]
    out["stagnation"] = (
        (out["wind_speed"] < 2.0) & (out["boundary_layer_height"] < 500)
    ).astype(np.float32)
    out["temp_dewpoint_spread"] = out["temperature_2m"] - out["dew_point_2m"]
    out["inversion_proxy"] = (
        out["temperature_2m"] * (1013.25 / out["surface_pressure"])
    ).astype(np.float32)
    out["humidity_temp_interaction"] = (
        out["relative_humidity_2m"] * out["temperature_2m"]
    )
    out["cloud_radiation_ratio"] = (
        out["shortwave_radiation"] / (out["cloud_cover"] + 1.0)
    )
    out["fire_intensity"] = out["fire_count"] * out["frp_mean"]
    out["wind_direction_deg"] = (
        np.degrees(np.arctan2(out["wind_v"], out["wind_u"])) % 360
    )
    out["precip_binary"] = (out["precipitation"] > 0).astype(np.float32)
    out["night_flag"] = out["timestamp"].dt.hour.isin(range(0, 6)).astype(np.float32)
    out["winter_night"] = (out["is_winter"] * out["night_flag"]).astype(np.float32)
    out["winter_stagnation"] = (
        out["is_winter"] * out["stagnation"]
    ).astype(np.float32)
    return out


def lag_columns() -> list[str]:
    return [
        "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12",
        "pm25_lag24", "pm25_lag48", "pm25_lag72", "pm25_rmean24",
        "pm25_rmax24", "pm25_rmin24", "pm25_rstd24", "pm25_trend6h",
    ]


def _exclude_target_columns(df: pd.DataFrame) -> set[str]:
    return {col for col in df.columns if col.startswith("target_")}


def get_feature_cols_pm25_lag(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "pm25", "station", "aqi_category"}
    return [
        col for col in df.columns
        if col not in excluded | _exclude_target_columns(df)
    ]


def get_feature_cols_pm25_nolag(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "pm25", "station", "aqi_category"} | set(lag_columns())
    return [
        col for col in df.columns
        if col not in excluded | _exclude_target_columns(df)
    ]


def get_feature_cols_lag_no_meteo(df: pd.DataFrame) -> list[str]:
    meteo_cols = {
        "temperature_2m", "relative_humidity_2m", "dew_point_2m",
        "surface_pressure", "precipitation", "cloud_cover",
        "shortwave_radiation", "boundary_layer_height", "wind_speed",
        "wind_u", "wind_v", "wind_speed_computed",
    }
    excluded = {"timestamp", "pm25", "station", "aqi_category"} | meteo_cols
    return [
        col for col in df.columns
        if col not in excluded | _exclude_target_columns(df)
    ]


def get_feature_cols_for_horizon(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "station", "aqi_category"}
    return [
        col for col in df.columns
        if col not in excluded | _exclude_target_columns(df)
    ]


def run_tscv_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    params: dict[str, Any],
    label: str,
    clip_nonnegative: bool = False,
    n_splits: int = N_SPLITS,
    eval_log_period: int = EVAL_LOG_PERIOD,
) -> dict[str, Any]:
    if not feature_cols:
        raise ValueError(f"{label} has no feature columns.")
    model_df = df[feature_cols + [target_col]].dropna().copy()
    X = model_df[feature_cols].to_numpy(dtype=np.float32)
    y = model_df[target_col].to_numpy(dtype=np.float32)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    predictions = np.full(len(y), np.nan)
    fold_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        print(
            f"[{label}] Fold {fold_idx}/{n_splits} started | "
            f"train_rows={len(train_idx)} val_rows={len(val_idx)}",
            flush=True,
        )
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], feature_name=feature_cols)
        dval = lgb.Dataset(
            X[val_idx], label=y[val_idx], feature_name=feature_cols, reference=dtrain
        )
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=eval_log_period),
            ],
        )
        y_pred = model.predict(X[val_idx], num_iteration=model.best_iteration)
        if clip_nonnegative:
            y_pred = np.maximum(0, y_pred)
        predictions[val_idx] = y_pred
        fold_rows.append({
            "fold": fold_idx,
            "rmse": float(np.sqrt(mean_squared_error(y[val_idx], y_pred))),
            "mae": float(mean_absolute_error(y[val_idx], y_pred)),
            "r2": float(r2_score(y[val_idx], y_pred)),
        })
        importance_frames.append(pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importance(importance_type="gain"),
            "fold": fold_idx,
        }))

    valid = ~np.isnan(predictions)
    importance = pd.concat(importance_frames).groupby("feature")["importance"].mean()
    return {
        "oof_rmse": float(np.sqrt(mean_squared_error(y[valid], predictions[valid]))),
        "oof_mae": float(mean_absolute_error(y[valid], predictions[valid])),
        "oof_r2": float(r2_score(y[valid], predictions[valid])),
        "scores_df": pd.DataFrame(fold_rows),
        "mean_importance": importance.sort_values(ascending=True),
    }


def print_tscv_results(title: str, result: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}\n{title}")
    print(f"OOF RMSE: {result['oof_rmse']:.4f}")
    print(f"OOF MAE:  {result['oof_mae']:.4f}")
    print(f"OOF R2:   {result['oof_r2']:.4f}")
    print("=" * 60)
    print(result["scores_df"].to_string(index=False))


def plot_top30_importance(
    mean_importance: pd.Series,
    title: str,
    output_file: str | Path,
    color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    mean_importance.tail(30).plot(kind="barh", ax=ax, color=color, edgecolor="none")
    ax.set_xlabel("Mean Gain Importance")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
