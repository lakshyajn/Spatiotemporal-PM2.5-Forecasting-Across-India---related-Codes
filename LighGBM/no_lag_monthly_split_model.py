"""LightGBM no-lag model using days 1-22 for training and 23-31 for testing."""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common_lightgbm import (
    LGB_PARAMS_STRONG,
    add_engineered_features_nolag,
    get_feature_cols_pm25_nolag,
    load_base_data,
    preprocess_base,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to dataset_v4.parquet")
    parser.add_argument("--output-dir", default="model_outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_engineered_features_nolag(
        preprocess_base(load_base_data(args.data), with_extra_pollutants=True)
    )
    features = get_feature_cols_pm25_nolag(df)
    train_mask = df["timestamp"].dt.day <= 22
    test_mask = ~train_mask
    X_train = df.loc[train_mask, features].to_numpy(dtype=np.float32)
    y_train = df.loc[train_mask, "pm25"].to_numpy(dtype=np.float32)
    X_test = df.loc[test_mask, features].to_numpy(dtype=np.float32)
    y_test = df.loc[test_mask, "pm25"].to_numpy(dtype=np.float32)

    model = lgb.train(
        LGB_PARAMS_STRONG,
        lgb.Dataset(X_train, label=y_train, feature_name=features),
        num_boost_round=5000,
        valid_sets=[lgb.Dataset(X_test, label=y_test, feature_name=features)],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    prediction = np.maximum(
        0, model.predict(X_test, num_iteration=model.best_iteration)
    )
    print(f"Train rows: {len(X_train):,}\nTest rows: {len(X_test):,}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, prediction)):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test, prediction):.4f}")
    print(f"R2:   {r2_score(y_test, prediction):.4f}")

    test_df = df.loc[test_mask, ["timestamp", "pm25"]].copy()
    test_df["pred"] = prediction
    test_df["month"] = test_df["timestamp"].dt.to_period("M").astype(str)
    monthly = test_df.groupby("month").apply(
        lambda group: pd.Series({
            "rmse": np.sqrt(mean_squared_error(group["pm25"], group["pred"])),
            "mae": mean_absolute_error(group["pm25"], group["pred"]),
            "r2": r2_score(group["pm25"], group["pred"]),
            "rows": len(group),
        }),
        include_groups=False,
    )
    monthly.to_csv(output_dir / "no_lag_monthly_metrics.csv")
    print("\nPer-month results:")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
