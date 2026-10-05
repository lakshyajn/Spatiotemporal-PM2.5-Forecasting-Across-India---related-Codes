"""LightGBM lag model with chronological cross-validation."""

import argparse
from pathlib import Path

from common_lightgbm import (
    LGB_PARAMS_BASE,
    load_base_data,
    get_feature_cols_pm25_lag,
    plot_top30_importance,
    preprocess_base,
    print_tscv_results,
    run_tscv_regression,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to dataset_v4.parquet")
    parser.add_argument("--output-dir", default="model_outputs")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess_base(load_base_data(args.data))
    features = get_feature_cols_pm25_lag(df)
    result = run_tscv_regression(
        df, features, "pm25", LGB_PARAMS_BASE, "LAG_BASE"
    )
    print_tscv_results("Lag Model OOF Results", result)
    plot_top30_importance(
        result["mean_importance"],
        "LightGBM Top 30 Feature Importance (Lag + TimeSeriesSplit)",
        output_dir / "feature_importance_lag_tscv.png",
        "#2196F3",
    )


if __name__ == "__main__":
    main()
