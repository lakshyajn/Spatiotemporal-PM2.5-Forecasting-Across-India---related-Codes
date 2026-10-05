"""Compare the lag model trained on all stations with one selected city."""

import argparse
from pathlib import Path

import pandas as pd

from common_lightgbm import (
    CITY_MIN_ROWS,
    CITY_NAME,
    LGB_PARAMS_BASE,
    get_feature_cols_pm25_lag,
    load_base_data,
    preprocess_base,
    run_tscv_regression,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to dataset_v4.parquet")
    parser.add_argument("--output-dir", default="model_outputs")
    parser.add_argument("--city", default=CITY_NAME)
    parser.add_argument("--city-min-rows", type=int, default=CITY_MIN_ROWS)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = preprocess_base(load_base_data(args.data), with_city_guess=True)
    features = get_feature_cols_pm25_lag(df)
    city_counts = df["city_guess"].value_counts()
    city = args.city.lower().strip()
    if city not in city_counts.index:
        raise ValueError(f"Invalid city '{args.city}'. Available: {list(city_counts.index[:20])}")
    city_df = df[df["city_guess"] == city].copy()
    if len(city_df) < args.city_min_rows:
        raise ValueError(f"{city} has {len(city_df)} rows, fewer than {args.city_min_rows}.")

    all_result = run_tscv_regression(
        df, features, "pm25", LGB_PARAMS_BASE, "CITY_ALL_DATA"
    )
    city_result = run_tscv_regression(
        city_df, features, "pm25", LGB_PARAMS_BASE, f"CITY_{city.upper()}"
    )
    comparison = pd.DataFrame([
        {"segment": "ALL_DATA", "rows": len(df), "rmse": all_result["oof_rmse"],
         "mae": all_result["oof_mae"], "r2": all_result["oof_r2"]},
        {"segment": city, "rows": len(city_df), "rmse": city_result["oof_rmse"],
         "mae": city_result["oof_mae"], "r2": city_result["oof_r2"]},
    ])
    print(comparison.to_string(index=False))
    comparison.to_csv(output_dir / "city_wise_lag_results.csv", index=False)


if __name__ == "__main__":
    main()
