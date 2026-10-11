"""Train localized XGBoost models for Anand Vihar, following notebook cell 30.

Important: the original notebook fits on all labelled Anand Vihar rows and then
predicts the same full timeline to create a visualization CSV. This script keeps
that behavior for reproducing the figure; it is not a held-out evaluation.
"""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from common import default_dataset_path, params_for_device

warnings.filterwarnings("ignore")

LOCAL_FEATURES = [
    "pm25", "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12",
    "pm25_lag24", "pm25_lag48", "pm25_lag72", "pm25_rmean24",
    "pm25_rmax24", "pm25_rmin24", "pm25_rstd24", "pm25_trend6h", "pm10",
    "lat", "lon", "month_sin", "month_cos", "hour_sin", "hour_cos",
    "day_of_week_sin", "is_monsoon", "is_harvest_season", "is_holiday",
    "is_weekday", "surface_pressure", "boundary_layer_height",
    "shortwave_radiation", "precipitation", "wind_u", "wind_v", "wind_speed",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "cloud_cover",
    "co", "no2", "so2", "o3", "fire_count", "bright_ti5_mean", "traffic_index",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Prepared full Parquet dataset or a station-level CSV")
    parser.add_argument("--output", default="model_outputs/Anand_Vihar_Full_Timeline_XGBoost.csv")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    input_path = Path(args.data).expanduser() if args.data else default_dataset_path()
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    print(f"Loading data from: {input_path}")
    if input_path.suffix.lower() == ".parquet":
        df_raw = pd.read_parquet(input_path, engine="pyarrow")
    else:
        df_raw = pd.read_csv(input_path)

    if "timeline" in df_raw.columns and "timestamp" not in df_raw.columns:
        df_raw.rename(columns={"timeline": "timestamp"}, inplace=True)
    if "timestamp" not in df_raw.columns:
        raise KeyError("Input must contain a 'timestamp' or 'timeline' column.")
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], format="mixed", dayfirst=True)

    if "station" in df_raw.columns:
        station_normalized = df_raw["station"].astype(str).str.replace("_", " ", regex=False)
        match_mask = station_normalized.str.contains(r"anand\s*vihar", case=False, na=False, regex=True)
        if match_mask.any():
            df_anand = df_raw[match_mask].sort_values("timestamp").reset_index(drop=True)
            station_identified = df_anand["station"].iloc[0]
            print(f"Found station match: '{station_identified}'")
        else:
            print("Station name did not match Anand Vihar; treating the supplied file as station-level data.")
            df_anand = df_raw.sort_values("timestamp").reset_index(drop=True)
            df_anand["station"] = "Anand Vihar, New Delhi"
    else:
        df_anand = df_raw.sort_values("timestamp").reset_index(drop=True)
        df_anand["station"] = "Anand Vihar, New Delhi"

    print(f"Total records isolated: {len(df_anand):,}")
    target_1h_col = "target_1h" if "target_1h" in df_anand.columns else "actual_pm25_t1"
    target_24h_col = "target_24h" if "target_24h" in df_anand.columns else "actual_pm25_t24"
    for target in [target_1h_col, target_24h_col]:
        if target not in df_anand.columns:
            raise KeyError(f"Target column missing: {target}")
    valid_features = [col for col in LOCAL_FEATURES if col in df_anand.columns]
    if not valid_features:
        raise ValueError("None of the notebook's local features were found in the input data.")
    print(f"Using {len(valid_features)} of the notebook's localized features.")
    params = params_for_device(args.device)

    print("Training XGBoost model for t+1h...")
    train_1h = df_anand.dropna(subset=[target_1h_col])
    model_1h = XGBRegressor(**params).fit(train_1h[valid_features], train_1h[target_1h_col])
    df_anand["predicted_1h"] = np.maximum(0, model_1h.predict(df_anand[valid_features]))

    print("Training XGBoost model for t+24h...")
    train_24h = df_anand.dropna(subset=[target_24h_col])
    model_24h = XGBRegressor(**params).fit(train_24h[valid_features], train_24h[target_24h_col])
    df_anand["predicted_24h"] = np.maximum(0, model_24h.predict(df_anand[valid_features]))

    final_df = pd.DataFrame({
        "station": df_anand["station"],
        "timeline": df_anand["timestamp"],
        "actual_1h": df_anand[target_1h_col].round(2),
        "predicted_1h": df_anand["predicted_1h"].round(2),
        "actual_24h": df_anand[target_24h_col].round(2),
        "predicted_24h": df_anand["predicted_24h"].round(2),
    })
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Wrote predictions to: {output_path.resolve()}")
    print(final_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
