"""LightGBM lag model with chronological cross-validation."""

import argparse
from pathlib import Path

from common_lightgbm import (
    LGB_PARAMS_BASE,
    fit_full_model_predictions,
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

    for horizon, target in ((1, "target_1h"), (24, "target_24h")):
        if target not in df.columns:
            raise KeyError(f"Required target column missing: {target}")
        predictions = fit_full_model_predictions(
            df, features, target, LGB_PARAMS_BASE, clip_nonnegative=True
        )
        export = df[["station", "timestamp", target]].copy()
        export = export.rename(columns={target: "actual_pm25"})
        export.insert(2, "pm25_at_prediction_time", df["pm25"])
        export["predicted_pm25"] = predictions
        output_file = output_dir / f"lightgbm_predictions_t_plus_{horizon}h.csv"
        export.to_csv(output_file, index=False)
        print(f"Saved {len(export)} rows to {output_file}")


if __name__ == "__main__":
    main()
