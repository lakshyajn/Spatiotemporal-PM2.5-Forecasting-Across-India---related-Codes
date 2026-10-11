"""Shared constants and helpers transcribed from the XGBoost notebook.

Core settings and evaluation loops follow PM2_5(1).ipynb (especially cells 21 and 23).
The only structural changes are a command-line dataset path and reusable functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# The notebook uses this exact parameter set for its walk-forward XGBoost models.
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 8,
    "tree_method": "hist",
    "device": "cuda",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Exact pruned feature list used in notebook cell 23 and the final forecasting cell.
PRUNED_FEATURES = [
    "pm25", "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12",
    "pm25_lag24", "pm25_lag48", "pm25_lag72", "pm25_rmean24",
    "pm25_rmax24", "pm25_rmin24", "pm25_rstd24", "pm25_trend6h", "pm10",
    "lat", "lon", "month_sin", "month_cos", "hour_sin", "hour_cos",
    "day_of_week_sin", "is_monsoon", "is_harvest_season", "is_holiday",
    "is_weekday", "surface_pressure", "boundary_layer_height",
    "shortwave_radiation", "precipitation", "wind_u", "wind_v", "wind_speed",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
    "co", "no2", "so2", "o3", "fire_count", "bright_ti5_mean",
    "traffic_index",
]

# The following groups preserve notebook cell 23's exact ablation lists.
WEATHER_COLS = [
    "surface_pressure", "boundary_layer_height", "shortwave_radiation",
    "precipitation", "wind_u", "wind_v", "wind_speed", "temperature_2m",
    "relative_humidity_2m", "dew_point_2m", "cloud_cover",
]

HISTORICAL_COLS = [
    "pm25", "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12",
    "pm25_lag24", "pm25_lag48", "pm25_lag72", "pm25_rmean24",
    "pm25_rmax24", "pm25_rmin24", "pm25_rstd24", "pm25_trend6h", "pm10",
]

TARGETS = ["target_1h", "target_24h"]


def default_dataset_path() -> Path:
    """Default to a dataset folder beside XGBoost/, with CLI override supported."""
    return Path(__file__).resolve().parent.parent / "data" / "dataset_v4_filled.parquet"


def load_dataset(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load and sort the prepared Parquet dataset, as in the source notebook."""
    path = Path(data_path).expanduser() if data_path else default_dataset_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPass --data /path/to/dataset_v4_filled.parquet."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    if "timestamp" not in df.columns:
        raise KeyError("The dataset must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return add_calendar_columns(df)


def get_season(month: int) -> str:
    """Season definition copied from notebook cells 12, 13 and 23."""
    if month in [11, 12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5, 6]:
        return "Summer"
    else:
        return "Monsoon"


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create the year-month, day-of-month and season columns used by the notebook."""
    out = df.copy()
    out["day_of_month"] = out["timestamp"].dt.day
    out["year_month"] = out["timestamp"].dt.strftime("%Y-%m")
    out["season"] = out["timestamp"].dt.month.apply(get_season)
    return out


def numeric_pruned_features(df: pd.DataFrame) -> list[str]:
    """Select the notebook's pruned 43-feature list, retaining only present numeric columns."""
    return [
        col for col in PRUNED_FEATURES
        if col in df.columns and df[col].dtype in ["float64", "float32", "int64", "int32"]
    ]


def all_numeric_forecast_features(df: pd.DataFrame) -> list[str]:
    """Feature selection from notebook cell 21 (primary overall forecast table)."""
    future_targets = [f"target_{i}h" for i in range(1, 25)]
    base_ignore = future_targets + [
        "timestamp", "station", "aqi_category", "day_of_month", "year_month"
    ]
    return [
        col for col in df.columns
        if col not in base_ignore and df[col].dtype in ["float64", "float32", "int64", "int32"]
    ]


def params_for_device(device: str | None = None) -> dict[str, Any]:
    """Return notebook XGBoost parameters, optionally selecting CPU for portability."""
    params = dict(XGB_PARAMS)
    if device:
        params["device"] = device
    return params


def run_walk_forward(
    data: pd.DataFrame,
    target_col: str,
    features: list[str],
    params: dict[str, Any] | None = None,
    cutoff_day: int = 20,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Notebook cell 23's mean-of-monthly-fold walk-forward engine.

    Per month: train on all previous months plus current-month days <= cutoff_day;
    test on current-month days > cutoff_day. Metrics are averaged across month folds,
    exactly as in cell 23 (rather than pooling every test prediction).
    """
    data = data.dropna(subset=[target_col]).copy()
    unique_months = sorted(data["year_month"].unique())
    r2_scores, mae_scores, rmse_scores = [], [], []
    fold_rows: list[dict[str, Any]] = []
    model_params = dict(params if params is not None else XGB_PARAMS)

    for month in unique_months:
        train_mask = (data["year_month"] < month) | (
            (data["year_month"] == month) & (data["day_of_month"] <= cutoff_day)
        )
        test_mask = (data["year_month"] == month) & (data["day_of_month"] > cutoff_day)

        if train_mask.sum() < 1000 or test_mask.sum() == 0:
            continue

        X_tr, X_te = data[features][train_mask], data[features][test_mask]
        y_tr, y_te = data[target_col][train_mask], data[target_col][test_mask]
        model = XGBRegressor(**model_params).fit(X_tr, y_tr)
        preds = np.maximum(0, model.predict(X_te))
        r2 = float(r2_score(y_te, preds))
        mae = float(mean_absolute_error(y_te, preds))
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        fold_rows.append({
            "target": target_col,
            "month": month,
            "cutoff_day": cutoff_day,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        })

    summary = {
        "r2": float(np.mean(r2_scores)) if r2_scores else float("nan"),
        "mae": float(np.mean(mae_scores)) if mae_scores else float("nan"),
        "rmse": float(np.mean(rmse_scores)) if rmse_scores else float("nan"),
        "folds": int(len(fold_rows)),
    }
    return summary, pd.DataFrame(fold_rows)
