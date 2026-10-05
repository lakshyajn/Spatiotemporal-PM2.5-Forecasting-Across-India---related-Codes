"""LightGBM no-lag model with chronological cross-validation."""

import argparse
from pathlib import Path

from common_lightgbm import (
    LGB_PARAMS_BASE,
    get_feature_cols_pm25_nolag,
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
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess_base(load_base_data(args.data), with_extra_pollutants=True)
    features = get_feature_cols_pm25_nolag(df)
    result = run_tscv_regression(
        df, features, "pm25", LGB_PARAMS_BASE, "NO_LAG_TSCV",
        clip_nonnegative=True,
    )
    print_tscv_results("No-Lag TimeSeriesSplit OOF Results", result)
    plot_top30_importance(
        result["mean_importance"],
        "Top 30 Features - No Lag TimeSeriesSplit",
        output_dir / "feature_importance_no_lag_tscv.png",
        "#EF6C00",
    )


if __name__ == "__main__":
    main()
