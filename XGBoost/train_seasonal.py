"""Run the notebook's primary t+1/t+24 forecast and seasonal analysis.

Primary overall metrics follow cell 21 (day 1-24 train / day 25+ test).
Seasonal metrics follow cell 23 (43 features; day 1-20 train / day 21+ test).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from common import (
    TARGETS, XGB_PARAMS, load_dataset, all_numeric_forecast_features,
    numeric_pruned_features, params_for_device, run_walk_forward,
)


def run_primary_forecast(df: pd.DataFrame, output_dir: Path, params: dict) -> None:
    """Exact model/split structure from notebook cell 21; scores match Table 4.1."""
    features = all_numeric_forecast_features(df)
    unique_months = sorted(df["year_month"].unique())
    fold_rows = []
    metrics = {"target_1h": {"r2": [], "mae": [], "rmse": []},
               "target_24h": {"r2": [], "mae": [], "rmse": []}}

    print("=" * 80)
    print("EVALUATING t+1h AND t+24h HORIZONS (Walk-Forward Folds)")
    print("=" * 80)
    for fold, month in enumerate(unique_months, 1):
        train_mask = (df["year_month"] < month) | (
            (df["year_month"] == month) & (df["day_of_month"] <= 24)
        )
        test_mask = (df["year_month"] == month) & (df["day_of_month"] > 24)
        if train_mask.sum() < 1000 or test_mask.sum() == 0:
            continue
        X_tr, X_te = df[features][train_mask], df[features][test_mask]

        for target_col in TARGETS:
            y_tr, y_te = df[target_col][train_mask], df[target_col][test_mask]
            valid_tr, valid_te = ~y_tr.isna(), ~y_te.isna()
            model = XGBRegressor(**params).fit(X_tr[valid_tr], y_tr[valid_tr])
            preds = np.maximum(0, model.predict(X_te[valid_te]))
            actual = y_te[valid_te]
            r2 = float(r2_score(actual, preds))
            mae = float(mean_absolute_error(actual, preds))
            rmse = float(np.sqrt(mean_squared_error(actual, preds)))
            metrics[target_col]["r2"].append(r2)
            metrics[target_col]["mae"].append(mae)
            metrics[target_col]["rmse"].append(rmse)
            fold_rows.append({
                "fold": fold, "month": month, "target": target_col,
                "train_rows": int(valid_tr.sum()), "test_rows": int(valid_te.sum()),
                "r2": r2, "mae": mae, "rmse": rmse,
            })
            if target_col == "target_1h":
                print(f"Fold {fold} ({month}) | t+1h  -> R²: {r2:5.3f} | MAE: {mae:5.2f} | RMSE: {rmse:5.2f}")
            else:
                print(f"               | t+24h -> R²: {r2:5.3f} | MAE: {mae:5.2f} | RMSE: {rmse:5.2f}")
        print("-" * 60)

    summary_rows = []
    print("OVERALL AVERAGES")
    for target_col in TARGETS:
        row = {"target": target_col}
        for metric in ("r2", "mae", "rmse"):
            row[metric] = float(np.mean(metrics[target_col][metric])) if metrics[target_col][metric] else float("nan")
        row["folds"] = len(metrics[target_col]["r2"])
        summary_rows.append(row)
        print(f"{target_col} -> R²: {row['r2']:.3f} | MAE: {row['mae']:.3f} | RMSE: {row['rmse']:.3f}")
    pd.DataFrame(fold_rows).to_csv(output_dir / "overall_forecast_fold_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "overall_forecast_metrics.csv", index=False)


def run_seasonal_analysis(df: pd.DataFrame, output_dir: Path, params: dict) -> None:
    """Season-specific walk-forward evaluations from notebook cell 23."""
    features = numeric_pruned_features(df)
    if not features:
        raise ValueError("None of the notebook's 43 pruned features were found in the dataset.")
    summary_rows, fold_frames = [], []
    print("\nSEASONAL MODELS (43 pruned features; monthly day-20 walk-forward)")
    for target_col in TARGETS:
        for season in ["Winter", "Summer", "Monsoon"]:
            season_df = df[df["season"] == season].copy()
            summary, folds = run_walk_forward(
                season_df, target_col, features, params=params, cutoff_day=20
            )
            summary_rows.append({"target": target_col, "season": season, **summary})
            if not folds.empty:
                folds.insert(1, "season", season)
                fold_frames.append(folds)
            print(f"{target_col} | {season:10} -> R²: {summary['r2']:.3f} | MAE: {summary['mae']:.3f} | RMSE: {summary['rmse']:.3f}")
    pd.DataFrame(summary_rows).to_csv(output_dir / "seasonal_metrics.csv", index=False)
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(output_dir / "seasonal_fold_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to dataset_v4_filled.parquet")
    parser.add_argument("--output-dir", default="model_outputs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--experiment", default="both", choices=["both", "overall", "seasonal"],
                        help="Run the overall report evaluation, seasonal evaluation, or both")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    print(f"Loaded {len(df):,} rows chronologically.")
    params = params_for_device(args.device)
    if args.experiment in ("both", "overall"):
        run_primary_forecast(df, out, params)
    if args.experiment in ("both", "seasonal"):
        run_seasonal_analysis(df, out, params)


if __name__ == "__main__":
    main()
