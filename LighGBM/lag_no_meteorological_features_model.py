"""LightGBM lag model with meteorological columns removed."""

import argparse
from pathlib import Path

from common_lightgbm import (
    LGB_PARAMS_BASE,
    get_feature_cols_lag_no_meteo,
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
    features = get_feature_cols_lag_no_meteo(df)
    result = run_tscv_regression(
        df, features, "pm25", LGB_PARAMS_BASE, "LAG_NO_METEO",
        clip_nonnegative=True,
    )
    print_tscv_results("Lag (No Meteo) TimeSeriesSplit Results", result)
    plot_top30_importance(
        result["mean_importance"],
        "Top 30 Features - Lag Without Meteorological Features",
        output_dir / "feature_importance_lag_no_meteo.png",
        "#3949AB",
    )


if __name__ == "__main__":
    main()
