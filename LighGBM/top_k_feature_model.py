"""Compare the full lag model with an importance-based Top-K feature model."""

import argparse
from pathlib import Path

from common_lightgbm import (
    LGB_PARAMS_BASE,
    TOP_K_FEATURES,
    get_feature_cols_pm25_lag,
    load_base_data,
    plot_top30_importance,
    preprocess_base,
    print_tscv_results,
    run_tscv_regression,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to dataset_v4.parquet")
    parser.add_argument("--output-dir", default="model_outputs")
    parser.add_argument("--top-k", type=int, default=TOP_K_FEATURES)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess_base(load_base_data(args.data))
    all_features = get_feature_cols_pm25_lag(df)
    baseline = run_tscv_regression(
        df, all_features, "pm25", LGB_PARAMS_BASE, "IMP_BASELINE"
    )
    top_k = min(max(1, args.top_k), len(all_features))
    selected = baseline["mean_importance"].tail(top_k).index.tolist()
    top_k_result = run_tscv_regression(
        df, selected, "pm25", LGB_PARAMS_BASE, "IMP_TOPK"
    )
    print_tscv_results("Top-K Model Results", top_k_result)
    print(f"Selected features ({top_k}): {selected}")
    print(f"RMSE delta: {top_k_result['oof_rmse'] - baseline['oof_rmse']:+.4f}")
    print(f"R2 delta:   {top_k_result['oof_r2'] - baseline['oof_r2']:+.4f}")
    plot_top30_importance(
        baseline["mean_importance"], "Baseline Lag Feature Importance",
        output_dir / "feature_importance_baseline.png", "#2196F3",
    )
    plot_top30_importance(
        top_k_result["mean_importance"], f"Top-{top_k} Feature Importance",
        output_dir / "feature_importance_top_k.png", "#43A047",
    )


if __name__ == "__main__":
    main()
