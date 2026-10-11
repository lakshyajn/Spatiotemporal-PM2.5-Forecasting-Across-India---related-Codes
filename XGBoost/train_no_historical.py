"""Historical PM2.5 feature ablation from notebook cell 23.

Compares the notebook's 43-feature full configuration with a second feature list
that removes the current PM2.5/lag/rolling/trend features and PM10 as defined in
that notebook. Runs both t+1h and t+24h targets.
"""
import argparse
from pathlib import Path
import pandas as pd
from common import (
    TARGETS, HISTORICAL_COLS, load_dataset, numeric_pruned_features,
    params_for_device, run_walk_forward,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to dataset_v4_filled.parquet")
    parser.add_argument("--output-dir", default="model_outputs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    all_features = numeric_pruned_features(df)
    no_history_features = [c for c in all_features if c not in HISTORICAL_COLS]
    params = params_for_device(args.device)
    rows, fold_frames = [], []

    for target in TARGETS:
        print("\n" + "=" * 80)
        print(f"HISTORICAL PM2.5 ABLATION: {target}")
        print("=" * 80)
        for experiment, feature_list in [
            ("full_features_with_history", all_features),
            ("without_historical_pm25_features", no_history_features),
        ]:
            summary, folds = run_walk_forward(
                df, target, feature_list, params=params, cutoff_day=20
            )
            rows.append({"target": target, "experiment": experiment, "feature_count": len(feature_list), **summary})
            if not folds.empty:
                folds.insert(1, "experiment", experiment)
                fold_frames.append(folds)
            print(f"{experiment}: R²={summary['r2']:.3f} | MAE={summary['mae']:.3f} | RMSE={summary['rmse']:.3f}")

    pd.DataFrame(rows).to_csv(out / "historical_ablation_metrics.csv", index=False)
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(out / "historical_ablation_fold_metrics.csv", index=False)
    print(f"\nSaved results to {out.resolve()}")


if __name__ == "__main__":
    main()
