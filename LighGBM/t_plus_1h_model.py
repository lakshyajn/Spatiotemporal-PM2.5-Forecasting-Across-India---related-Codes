"""LightGBM one-hour-ahead PM2.5 forecasting model."""

import argparse
from pathlib import Path

from common_lightgbm import (
    LGB_PARAMS_BASE,
    get_feature_cols_for_horizon,
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

    df = preprocess_base(load_base_data(args.data))
    target = "target_1h"
    if target not in df.columns:
        raise KeyError(f"Required target column missing: {target}")
    features = get_feature_cols_for_horizon(df)
    result = run_tscv_regression(
        df, features, target, LGB_PARAMS_BASE, "T_PLUS_1H",
        clip_nonnegative=True,
    )
    print_tscv_results("t+1 Forecast Results", result)
    plot_top30_importance(
        result["mean_importance"], "Top 30 Features - PM2.5 t+1",
        output_dir / "feature_importance_t_plus_1h.png", "#00897B",
    )


if __name__ == "__main__":
    main()
