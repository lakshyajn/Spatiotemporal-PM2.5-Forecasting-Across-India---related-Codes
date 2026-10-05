import os, pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from common import (
    PM25LSTM, load_data, slice_month, TARGET_COLS,
    SEQ_LEN, N_HORIZONS,
)

STATION = "anand_vihar_new_delhi_-_dpcc_235"
SEASONS = {
    "spring": [2, 3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "winter": [10, 11, 12, 1],
}


def predict_station(df, model, feat_cols, sx, sy, months, horizon_idx):
    sdf = df[df["station"] == STATION].sort_values("timestamp").reset_index(drop=True)
    sdf[feat_cols] = sdf[feat_cols].ffill().bfill().fillna(0)
    X = sx.transform(sdf[feat_cols].values.astype(np.float32))

    rows = []
    for pos in range(SEQ_LEN, len(sdf)):
        ts = sdf.iloc[pos]["timestamp"]
        if ts.month not in months:
            continue
        window = X[pos - SEQ_LEN + 1:pos + 1]
        if len(window) < SEQ_LEN:
            continue
        with torch.no_grad():
            sids = torch.tensor([1], dtype=torch.long)
            pred_scaled = model(torch.tensor([window], dtype=torch.float32), sids).numpy()[0]
        pred_raw = sy.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
        actual = sdf.iloc[pos][TARGET_COLS[horizon_idx]]
        rows.append({"timestamp": ts, "actual": actual,
                     "predicted": pred_raw[horizon_idx]})
    return pd.DataFrame(rows)


def plot_horizon(df, horizon, tag):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["timestamp"], df["actual"], color="black",
            linewidth=0.8, label="Actual PM2.5")
    ax.plot(df["timestamp"], df["predicted"], color="#1976D2",
            linewidth=0.8, label=f"LSTM Predicted (t+{horizon}h)")
    ax.set_title(f"Anand Vihar – LSTM t+{horizon}h forecast")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(f"lstm_anand_vihar_t{horizon}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    df = load_data()
    all_dfs_t1, all_dfs_t24 = [], []
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
        sx = sx_dict.get(STATION, sx_dict.get("_global"))
        sy = sy_dict.get(STATION, sy_dict.get("_global"))

        all_dfs_t1.append(predict_station(df, model, feat_cols, sx, sy, months, 0))
        all_dfs_t24.append(predict_station(df, model, feat_cols, sx, sy, months, 23))

    df_t1 = pd.concat(all_dfs_t1).sort_values("timestamp").reset_index(drop=True)
    df_t24 = pd.concat(all_dfs_t24).sort_values("timestamp").reset_index(drop=True)
    plot_horizon(df_t1, 1, "t1")
    plot_horizon(df_t24, 24, "t24")