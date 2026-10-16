# %%
# ## Anand Vihar PM2.5 Forecast: t+1 & t+24 Horizon Plots (LightGBM)
# Train on the first 20 days of each month and test on the last 10 days of each month.

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ========== CONFIGURATION ==========
TARGET_STATION = "anand_vihar_new_delhi_-_dpcc_235"
TEST_DAYS = 10  # Last 10 days of each month (10 * 24 = 240 hours)

# Candidate paths for Kaggle, Colab, or local environments
DATA_CANDIDATES = [
    Path("/kaggle/input/datasets/lakshyajn/pm2-5-dataset-with-no-0s-and-nan/dataset_v4_filled.parquet"),
    Path("/kaggle/input/pm2-5-dataset-with-no-0s-and-nan/dataset_v4_filled.parquet"),
    Path("/kaggle/input/dataset_v4.parquet"),
    Path("LighGBM/data/dataset_v4.parquet"),
    Path("data/dataset_v4.parquet"),
    Path("dataset_v4.parquet"),
    Path("/content/drive/MyDrive/dataset_v4.parquet"),
    Path("/content/dataset_v4.parquet"),
]

# Output directory for saving plots and prediction CSV
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("LighGBM/model_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seasons definition matching the seasonal splits
SEASONS = {
    "spring": [2, 3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "winter": [10, 11, 12, 1],
}

# LightGBM Regressor Hyperparameters
LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "max_depth": -1,
    "n_jobs": -1,
    "verbose": -1,
    "seed": 42,
}

# %%
# ========== LOAD AND PREPARE DATA ==========
def resolve_data_path() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    # If not in candidates, check current directory for parquet files
    for p in Path(".").glob("**/*.parquet"):
        if "dataset" in p.name.lower():
            return p
    raise FileNotFoundError(f"Could not locate dataset. Checked: {[str(p) for p in DATA_CANDIDATES]}")

dataset_path = resolve_data_path()
print(f"Loading dataset from: {dataset_path} ...")
df = pd.read_parquet(dataset_path)

# Normalize timestamp column
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)

# Filter for Anand Vihar only
print(f"Filtering for station: {TARGET_STATION} ...")
sdf = df[df["station"] == TARGET_STATION].sort_values("timestamp").reset_index(drop=True)

if sdf.empty:
    raise ValueError(f"Station '{TARGET_STATION}' not found in dataset!")

print(f"Found {len(sdf):,} hourly records for Anand Vihar ({sdf['timestamp'].min()} to {sdf['timestamp'].max()})")

# Determine feature columns (exclude targets, timestamp, station, and non-numeric labels)
target_cols = [c for c in sdf.columns if c.startswith("target_")]
exclude_cols = {"timestamp", "station", "aqi_category"} | set(target_cols)
feat_cols = [c for c in sdf.columns if c not in exclude_cols]

# Impute missing values with forward/backward fill and fillna(0)
sdf[feat_cols] = sdf[feat_cols].ffill().bfill().fillna(0)

print(f"Total features used: {len(feat_cols)}")
print(f"Target horizons: target_1h (t+1) and target_24h (t+24)")

# %%
# ========== SEASONAL TRAIN / TEST SPLIT & MODEL TRAINING ==========
# Strategy: First ~20 days of each month -> TRAIN; Last 10 days of each month -> TEST
print("\n" + "=" * 65)
print("Training LightGBM Seasonal Models (Train: first ~20 days | Test: last 10 days)")
print("=" * 65)

results = []
season_metrics = []

for season, months in SEASONS.items():
    season_train_idx, season_test_idx = [], []
    
    # Split each month in the season
    for ym, g in sdf.groupby(sdf["timestamp"].dt.to_period("M")):
        if ym.month in months:
            g_indices = g.index.tolist()
            test_start = len(g_indices) - (TEST_DAYS * 24)
            if test_start > 0:
                season_train_idx.extend(g_indices[:test_start])
                season_test_idx.extend(g_indices[test_start:])
            else:
                # If month has <= 10 days of data, use for training
                season_train_idx.extend(g_indices)
                
    if not season_train_idx or not season_test_idx:
        print(f"Skipping {season}: insufficient data for train/test split.")
        continue
        
    print(f"\n--- Season: {season.upper()} (Months: {months}) ---")
    print(f"Train hours: {len(season_train_idx):,} (~{len(season_train_idx)/24:.1f} days) | Test hours: {len(season_test_idx):,} (~{len(season_test_idx)/24:.1f} days)")
    
    X_train = sdf.loc[season_train_idx, feat_cols].values.astype(np.float32)
    X_test = sdf.loc[season_test_idx, feat_cols].values.astype(np.float32)
    
    # Target 1h (t+1)
    y_train_t1 = sdf.loc[season_train_idx, "target_1h"].values.astype(np.float32)
    y_test_t1 = sdf.loc[season_test_idx, "target_1h"].values.astype(np.float32)
    
    # Target 24h (t+24)
    y_train_t24 = sdf.loc[season_train_idx, "target_24h"].values.astype(np.float32)
    y_test_t24 = sdf.loc[season_test_idx, "target_24h"].values.astype(np.float32)
    
    # --- Train t+1 Model ---
    dtrain_t1 = lgb.Dataset(X_train, label=y_train_t1, feature_name=feat_cols)
    dval_t1 = lgb.Dataset(X_test, label=y_test_t1, feature_name=feat_cols, reference=dtrain_t1)
    model_t1 = lgb.train(
        LGB_PARAMS,
        dtrain_t1,
        num_boost_round=1500,
        valid_sets=[dval_t1],
        valid_names=["test"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_t1 = np.maximum(0, model_t1.predict(X_test, num_iteration=model_t1.best_iteration))
    
    # --- Train t+24 Model ---
    dtrain_t24 = lgb.Dataset(X_train, label=y_train_t24, feature_name=feat_cols)
    dval_t24 = lgb.Dataset(X_test, label=y_test_t24, feature_name=feat_cols, reference=dtrain_t24)
    model_t24 = lgb.train(
        LGB_PARAMS,
        dtrain_t24,
        num_boost_round=1500,
        valid_sets=[dval_t24],
        valid_names=["test"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred_t24 = np.maximum(0, model_t24.predict(X_test, num_iteration=model_t24.best_iteration))
    
    # Metrics for this season
    s_rmse_t1 = np.sqrt(mean_squared_error(y_test_t1, pred_t1))
    s_r2_t1 = r2_score(y_test_t1, pred_t1)
    s_rmse_t24 = np.sqrt(mean_squared_error(y_test_t24, pred_t24))
    s_r2_t24 = r2_score(y_test_t24, pred_t24)
    print(f"  t+1  -> RMSE: {s_rmse_t1:.2f} | R²: {s_r2_t1:.3f} (best iteration: {model_t1.best_iteration})")
    print(f"  t+24 -> RMSE: {s_rmse_t24:.2f} | R²: {s_r2_t24:.3f} (best iteration: {model_t24.best_iteration})")
    
    season_metrics.append({
        "Season": season.capitalize(),
        "t+1 RMSE": s_rmse_t1,
        "t+1 R²": s_r2_t1,
        "t+24 RMSE": s_rmse_t24,
        "t+24 R²": s_r2_t24,
        "Test Rows": len(y_test_t1)
    })
    
    # Store predictions
    for i, pos in enumerate(season_test_idx):
        results.append({
            "timestamp": sdf.iloc[pos]["timestamp"],
            "actual_t1": y_test_t1[i],
            "predicted_t1": pred_t1[i],
            "actual_t24": y_test_t24[i],
            "predicted_t24": pred_t24[i],
        })

# Clean and sort predictions across the timeline
pred_df = pd.DataFrame(results).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
pred_df = pred_df.dropna().reset_index(drop=True)

# Save predictions to CSV
pred_csv_path = OUT_DIR / "lgbm_anand_vihar_predictions.csv"
pred_df.to_csv(pred_csv_path, index=False)
print(f"\nPredictions saved to: {pred_csv_path}")

# Print Seasonal Summary Table
print("\n" + "=" * 65)
print("Seasonal Performance Breakdown:")
print("=" * 65)
print(pd.DataFrame(season_metrics).to_string(index=False))

# %%
# ========== PLOTTING ==========
print(f"\nPlotting results ({len(pred_df):,} test timesteps)...")

# --- GRAPH 1: t+1 Hour Forecasting ---
rmse_t1 = np.sqrt(mean_squared_error(pred_df["actual_t1"], pred_df["predicted_t1"]))
mae_t1 = mean_absolute_error(pred_df["actual_t1"], pred_df["predicted_t1"])
r2_t1 = r2_score(pred_df["actual_t1"], pred_df["predicted_t1"])

plt.figure(figsize=(15, 6))
plt.plot(pred_df["timestamp"], pred_df["actual_t1"], label="Actual PM2.5", color="black", alpha=0.75, linewidth=1.2)
plt.plot(pred_df["timestamp"], pred_df["predicted_t1"], label="LightGBM Predicted (t+1h)", color="#1f77b4", alpha=0.85, linewidth=1.2)
plt.title(
    f"Anand Vihar, New Delhi — 1-Hour Ahead Forecasting (t+1)\n"
    f"RMSE: {rmse_t1:.2f} µg/m³ | MAE: {mae_t1:.2f} µg/m³ | R²: {r2_t1:.3f}",
    fontsize=13,
    pad=10
)
plt.xlabel("Timestamp", fontsize=11)
plt.ylabel("PM2.5 (µg/m³)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
fig1_path = OUT_DIR / "lgbm_anand_vihar_t1_forecast.png"
plt.savefig(fig1_path, dpi=150)
print(f"Graph 1 saved: {fig1_path}")
plt.show()

# --- GRAPH 2: t+24 Hour Forecasting ---
rmse_t24 = np.sqrt(mean_squared_error(pred_df["actual_t24"], pred_df["predicted_t24"]))
mae_t24 = mean_absolute_error(pred_df["actual_t24"], pred_df["predicted_t24"])
r2_t24 = r2_score(pred_df["actual_t24"], pred_df["predicted_t24"])

plt.figure(figsize=(15, 6))
plt.plot(pred_df["timestamp"], pred_df["actual_t24"], label="Actual PM2.5", color="black", alpha=0.75, linewidth=1.2)
plt.plot(pred_df["timestamp"], pred_df["predicted_t24"], label="LightGBM Predicted (t+24h)", color="#d62728", alpha=0.85, linewidth=1.2)
plt.title(
    f"Anand Vihar, New Delhi — 24-Hour Ahead Forecasting (t+24)\n"
    f"RMSE: {rmse_t24:.2f} µg/m³ | MAE: {mae_t24:.2f} µg/m³ | R²: {r2_t24:.3f}",
    fontsize=13,
    pad=10
)
plt.xlabel("Timestamp", fontsize=11)
plt.ylabel("PM2.5 (µg/m³)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
fig2_path = OUT_DIR / "lgbm_anand_vihar_t24_forecast.png"
plt.savefig(fig2_path, dpi=150)
print(f"Graph 2 saved: {fig2_path}")
plt.show()

print("\n" + "=" * 65)
print("Overall LightGBM Performance on Anand Vihar (Held-out Test Days):")
print(f"  t+1  Horizon: RMSE = {rmse_t1:.2f} µg/m³ | MAE = {mae_t1:.2f} µg/m³ | R² = {r2_t1:.3f}")
print(f"  t+24 Horizon: RMSE = {rmse_t24:.2f} µg/m³ | MAE = {mae_t24:.2f} µg/m³ | R² = {r2_t24:.3f}")
print("=" * 65)
