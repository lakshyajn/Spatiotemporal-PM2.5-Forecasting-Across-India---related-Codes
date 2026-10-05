import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from common import (
    PM25LSTM, WeightedHuberLoss, SequenceDataset,
    load_data, resolve_features, slice_month, compute_metrics,
    SEQ_LEN, N_HORIZONS, TARGET_COLS, TEST_DAYS,
)
from sklearn.preprocessing import RobustScaler

STATION = "anand_vihar_new_delhi_-_dpcc_235"
TOP_FEATURES = [
    "pm25", "aqi_category", "pm25_rmax24", "pm25_rmean24",
    "pm25_lag1", "lon", "pm25_lag6", "pm25_lag3",
    "shortwave_radiation", "pm25_rmin24", "pm25_lag12",
    "hour_sin", "surface_pressure", "pm25_trend6h",
    "pm25_rstd24", "dew_point_2m", "boundary_layer_height",
    "pm25_lag24", "pm25_lag72", "temperature_2m",
]
EPOCHS = 30
BATCH = 128
PEAK_LR = 3e-4
WARMUP = 3
STRIDE = 2
WEIGHT_DECAY = 5e-4


def build_station_split(df):
    sdf = df[df["station"] == STATION].sort_values("timestamp").reset_index(drop=True)
    tr_parts, te_parts = [], []
    for ym in sdf["timestamp"].dt.to_period("M").unique():
        tr, te = slice_month(sdf, ym.year, ym.month)
        if tr is not None:
            tr_parts.append(tr)
        if te is not None:
            te_parts.append(te)
    df_tr = pd_concat(tr_parts)
    df_te = pd_concat(te_parts)
    return df_tr, df_te


def pd_concat(parts):
    import pandas as pd
    if parts:
        return pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return pd.DataFrame()


def build_flat(df, feat_cols, sx, sy):
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[feat_cols] = df[feat_cols].ffill().bfill().fillna(0)
    X = np.nan_to_num(sx.transform(df[feat_cols].values.astype(np.float32)),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = sy.transform(df[TARGET_COLS].values.astype(np.float32).reshape(-1, 1)).reshape(-1, N_HORIZONS)
    return torch.from_numpy(X), torch.from_numpy(y)


def build_idx(n, stride):
    return np.arange(SEQ_LEN, n - N_HORIZONS + 1, stride, dtype=np.int32)


def train(df):
    feat_cols = resolve_features(df, TOP_FEATURES)
    df_tr, df_te = build_station_split(df)
    sx = RobustScaler().fit(df_tr[feat_cols].values.astype(np.float32))
    sy = RobustScaler().fit(df_tr[TARGET_COLS].values.reshape(-1, 1).astype(np.float32))
    ff_tr, ft_tr = build_flat(df_tr, feat_cols, sx, sy)
    ff_te, ft_te = build_flat(df_te, feat_cols, sx, sy)

    class SingleStationDS(torch.utils.data.Dataset):
        def __init__(self, ff, ft, idx):
            self.ff, self.ft, self.idx = ff, ft, idx
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            pos = int(self.idx[i])
            return self.ff[pos - SEQ_LEN:pos], self.ft[pos]

    idx_tr = build_idx(ff_tr.shape[0], STRIDE)
    idx_te = build_idx(ff_te.shape[0], 1)
    dl_tr = DataLoader(SingleStationDS(ff_tr, ft_tr, idx_tr), batch_size=BATCH,
                       shuffle=True, drop_last=True)
    dl_te = DataLoader(SingleStationDS(ff_te, ft_te, idx_te), batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PM25LSTM(len(feat_cols), n_stations=1).to(device)
    loss_fn = WeightedHuberLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    warm = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP)

    for epoch in range(EPOCHS):
        model.train()
        for X, y in tqdm(dl_tr, desc=f"local ep{epoch+1}", leave=False):
            X, y = X.to(device), y.to(device)
            sids = torch.ones(X.size(0), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X, sids), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch < WARMUP:
            warm.step()

    model.eval()
    yt_all, yp_all = [], []
    with torch.no_grad():
        for X, y in dl_te:
            X = X.to(device)
            sids = torch.ones(X.size(0), dtype=torch.long, device=device)
            yt_all.append(y.numpy())
            yp_all.append(model(X, sids).cpu().numpy())
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)

    yt_raw = sy.inverse_transform(yt.reshape(-1, 1)).reshape(yt.shape)
    yp_raw = sy.inverse_transform(yp.reshape(-1, 1)).reshape(yp.shape)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    for h in (1, 24):
        idx = h - 1
        mae = mean_absolute_error(yt_raw[:, idx], yp_raw[:, idx])
        rmse = mean_squared_error(yt_raw[:, idx], yp_raw[:, idx]) ** 0.5
        r2 = r2_score(yt_raw[:, idx], yp_raw[:, idx])
        print(f"local t+{h}: MAE {mae:.3f} RMSE {rmse:.3f} R2 {r2:.3f}")

    torch.save({"model": model.state_dict(),
                "n_features": len(feat_cols),
                "feature_cols": feat_cols},
               "lstm_localized_anand_vihar.pt")


if __name__ == "__main__":
    train(load_data())