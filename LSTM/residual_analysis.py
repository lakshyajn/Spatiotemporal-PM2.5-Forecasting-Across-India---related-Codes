import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common import (
    PM25LSTM, load_data, slice_month, TARGET_COLS,
    SEQ_LEN, N_HORIZONS, TEST_DAYS,
)

SEASONS = {"spring": [2, 3, 4, 5], "monsoon": [6, 7, 8, 9], "winter": [10, 11, 12, 1]}


def collect_residuals(df, horizon_idx):
    residuals, actuals = [], []
    for season, months in SEASONS.items():
        ckpt = torch.load(f"lstm_{season}.pt", map_location="cpu", weights_only=False)
        feat_cols = ckpt["feature_cols"]
        model = PM25LSTM(ckpt["n_features"], ckpt["n_stations"])
        model.load_state_dict(ckpt["model"])
        model.eval()

        with open(f"scaler_X_{season}.pkl", "rb") as f:
            sx_dict = pickle.load(f)
        with open(f"scaler_y_{season}.pkl", "rb") as f:
            sy_dict = pickle.load(f)
        with open(f"scaler_lookup_{season}.json", "rb") as f:
            lookup = pickle.load(f) if False else None

        for station in df["station"].unique():
            sdf = df[df["station"] == station].sort_values("timestamp").reset_index(drop=True)
            sx = sx_dict.get(station, sx_dict.get("_global"))
            sy = sy_dict.get(station, sy_dict.get("_global"))
            sdf[feat_cols] = sdf[feat_cols].ffill().bfill().fillna(0)
            X = sx.transform(sdf[feat_cols].values.astype(np.float32))

            for pos in range(SEQ_LEN, len(sdf)):
                ts = sdf.iloc[pos]["timestamp"]
                if ts.month not in months:
                    continue
                window = X[pos - SEQ_LEN + 1:pos + 1]
                if len(window) < SEQ_LEN:
                    continue
                with torch.no_grad():
                    sids = torch.tensor([1], dtype=torch.long)
                    p = model(torch.tensor([window], dtype=torch.float32), sids).numpy()[0]
                pred = sy.inverse_transform(p.reshape(-1, 1)).reshape(-1)
                actual = sdf.iloc[pos][TARGET_COLS[horizon_idx]]
                residuals.append(actual - pred[horizon_idx])
                actuals.append(actual)
    return np.array(actuals), np.array(residuals)


def plot_residuals(actual, residual, horizon):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].hist(residual, bins=100, color="steelblue")
    axes[0].set_title(f"Residual distribution t+{horizon}h")
    axes[0].set_xlabel("Residual (Actual - Predicted)")
    axes[0].set_ylabel("Frequency")

    axes[1].scatter(actual, residual, alpha=0.15, s=3, color="steelblue")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_title(f"Heteroscedasticity check t+{horizon}h")
    axes[1].set_xlabel("Actual PM2.5 (µg/m³)")
    axes[1].set_ylabel("Residual error")
    plt.tight_layout()
    plt.savefig(f"lstm_residuals_t{horizon}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    df = load_data()
    for h, idx in [(1, 0), (24, 23)]:
        a, r = collect_residuals(df, idx)
        plot_residuals(a, r, h)