"""LightGBM no-lag seasonal split model for Indian seasons."""

import argparse
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from common_lightgbm import (
    LGB_PARAMS_STRONG,
    EVAL_LOG_PERIOD,
    add_engineered_features_nolag,
    get_feature_cols_pm25_nolag,
    load_base_data,
    plot_top30_importance,
    preprocess_base,
)

INDIA_SEASONS = {
    "Winter": {"train": [11, 12], "test": [1]},
    "Spring": {"train": [2, 3], "test": [4]},
    "Summer": {"train": [5], "test": [6]},
    "Monsoon": {"train": [7, 8], "test": [9]},
    "Post-Monsoon": {"train": [10], "test": [11]},
}


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
    df["month"] = df["timestamp"].dt.month
    results = []
    actual: list[float] = []
    predicted: list[float] = []
    importance_frames = []

    for season, definition in INDIA_SEASONS.items():
        train_mask = df["month"].isin(definition["train"])
        test_mask = df["month"].isin(definition["test"])
        if not train_mask.any() or not test_mask.any():
            print(f"[SKIP] {season}: no training or testing rows")
            continue
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
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(max(EVAL_LOG_PERIOD, 200)),
            ],
        )
        prediction = np.maximum(
            0, model.predict(X_test, num_iteration=model.best_iteration)
        )
        results.append({
            "season": season,
            "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
            "mae": mean_absolute_error(y_test, prediction),
            "r2": r2_score(y_test, prediction),
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "best_iter": model.best_iteration,
        })
        actual.extend(y_test.tolist())
        predicted.extend(prediction.tolist())
        importance_frames.append(pd.DataFrame({
            "feature": features,
            "importance": model.feature_importance(importance_type="gain"),
        }))

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "seasonal_metrics.csv", index=False)
    if not results_df.empty:
        print(results_df.to_string(index=False))
        print(f"Overall RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.4f}")
        print(f"Overall MAE:  {mean_absolute_error(actual, predicted):.4f}")
        print(f"Overall R2:   {r2_score(actual, predicted):.4f}")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        results_df.plot(x="season", y=["rmse", "mae"], kind="bar", ax=axes[0])
        results_df.plot(x="season", y="r2", kind="bar", ax=axes[1], color="#4CAF50")
        plt.tight_layout()
        fig.savefig(output_dir / "seasonal_performance.png", dpi=150)
        plt.close(fig)
        importance = pd.concat(importance_frames).groupby("feature")["importance"].mean()
        plot_top30_importance(
            importance.sort_values(),
            "Top 30 Features - Seasonal Split",
            output_dir / "feature_importance_seasonal.png",
            "#AB47BC",
        )


if __name__ == "__main__":
    main()
