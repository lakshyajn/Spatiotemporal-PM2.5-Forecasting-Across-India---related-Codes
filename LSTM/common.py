import os, json, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

DATASET_PATH = "path/to/dataset.parquet"

SEQ_LEN = 72
N_HORIZONS = 24
TEST_DAYS = 10
VAL_DAYS = 5

SEASONS = {
    "spring": [2, 3, 4, 5],
    "monsoon": [6, 7, 8, 9],
    "winter": [10, 11, 12, 1],
}

TARGET_COLS = [f"target_{h}h" for h in range(1, N_HORIZONS + 1)]

ALL_FEATURES = [
    "lat", "lon", "pm25",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12",
    "pm25_lag24", "pm25_lag48", "pm25_lag72",
    "pm25_rmean24", "pm25_rmax24", "pm25_rmin24", "pm25_rstd24",
    "pm25_trend6h", "aqi_category",
    "pm10", "no2", "so2", "o3", "co",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "precipitation",
    "wind_speed", "wind_u", "wind_v",
    "cloud_cover", "shortwave_radiation", "boundary_layer_height",
    "fire_count", "frp_sum", "frp_mean", "frp_max",
    "bright_ti4_mean", "bright_ti5_mean",
    "rush_hour", "is_weekday", "is_holiday", "traffic_index",
    "day_of_week_sin", "day_of_week_cos",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "is_monsoon", "is_winter", "is_harvest_season",
    "is_diwali_period", "is_holi_period",
    "news_dust_storm", "news_industrial_event",
    "news_crop_burning", "news_calamity",
    "news_fireworks", "news_smog_alert",
]

WEATHER_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "surface_pressure", "precipitation",
    "wind_speed", "wind_u", "wind_v",
    "cloud_cover", "shortwave_radiation", "boundary_layer_height",
]

HISTORICAL_FEATURES = [
    "pm25",
    "pm25_lag1", "pm25_lag3", "pm25_lag6", "pm25_lag12",
    "pm25_lag24", "pm25_lag48", "pm25_lag72",
    "pm25_rmean24", "pm25_rmax24", "pm25_rmin24", "pm25_rstd24",
    "pm25_trend6h", "aqi_category",
]


class PM25LSTM(nn.Module):
    def __init__(self, n_features, n_stations,
                 embed_dim=16, hidden1=128, hidden2=64,
                 dense1=128, dense2=64, heads=4, dropout=0.30):
        super().__init__()
        lstm1_out = hidden1 * 2
        self.embed = nn.Embedding(n_stations + 1, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(n_features + embed_dim, hidden1,
                              batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(lstm1_out)
        self.drop1 = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(lstm1_out, heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(lstm1_out)
        self.lstm2 = nn.LSTM(lstm1_out, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, dense1)
        self.fc1_norm = nn.LayerNorm(dense1)
        self.act1 = nn.GELU()
        self.drop3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense1, dense2)
        self.act2 = nn.GELU()
        self.out = nn.Linear(dense2, N_HORIZONS)

    def forward(self, x, sids):
        e = self.embed(sids).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, e], dim=-1)
        x, _ = self.bilstm(x)
        x = self.drop1(self.norm1(x))
        a, _ = self.attn(x, x, x)
        x = self.norm2(x + a)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])
        x = self.drop3(self.act1(self.fc1_norm(self.fc1(x))))
        x = self.act2(self.fc2(x))
        return self.out(x)


class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=15.0, decay=0.95):
        super().__init__()
        w = torch.tensor([decay ** (h - 1) for h in range(1, N_HORIZONS + 1)],
                         dtype=torch.float32)
        self.register_buffer("weights", w / w.sum())
        self.delta = delta

    def forward(self, pred, target):
        e = pred - target
        L = torch.where(e.abs() < self.delta,
                        0.5 * e ** 2,
                        self.delta * (e.abs() - 0.5 * self.delta))
        return (L * self.weights).mean()


class SequenceDataset(Dataset):
    def __init__(self, ff, ft, ao, ar, asid):
        self.ff, self.ft = ff, ft
        self.ao, self.ar, self.asid = ao, ar, asid

    def __len__(self):
        return len(self.ar)

    def __getitem__(self, idx):
        o = int(self.ao[idx])
        i = int(self.ar[idx])
        sid = int(self.asid[idx])
        X = self.ff[o + i - SEQ_LEN: o + i]
        y = self.ft[o + i]
        return X, torch.tensor(sid, dtype=torch.long), y


def load_data():
    df = pd.read_parquet(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
    return df


def resolve_features(df, feature_list):
    return [c for c in feature_list if c in df.columns]


def slice_month(sdf, year, month, seq_len=SEQ_LEN):
    mask = (sdf["timestamp"].dt.year == year) & (sdf["timestamp"].dt.month == month)
    m = sdf[mask].reset_index(drop=True)
    test_hours = TEST_DAYS * 24
    need = seq_len + N_HORIZONS + seq_len + test_hours
    if len(m) < need:
        return None, None
    test_start = len(m) - test_hours
    train_cut = test_start - seq_len
    tr = m.iloc[:train_cut]
    te = m.iloc[test_start:]
    if len(te) < seq_len + N_HORIZONS:
        return tr, None
    return tr, te


def slice_month_val(sdf, year, month, val_days, seq_len=SEQ_LEN):
    mask = (sdf["timestamp"].dt.year == year) & (sdf["timestamp"].dt.month == month)
    m = sdf[mask].reset_index(drop=True)
    test_hours = TEST_DAYS * 24
    val_hours = val_days * 24
    need = seq_len + N_HORIZONS + seq_len + test_hours + val_hours
    if len(m) < need:
        return None, None, None
    test_start = len(m) - test_hours
    val_start = test_start - val_hours
    train_cut = val_start - seq_len
    tr = m.iloc[:train_cut]
    va = m.iloc[val_start:test_start]
    te = m.iloc[test_start:]
    if len(te) < seq_len + N_HORIZONS:
        te = None
    if len(va) < seq_len + N_HORIZONS:
        va = None
    if len(tr) < seq_len + N_HORIZONS:
        tr = None
    return tr, va, te


def build_train_test(df, season_months):
    df["_ym"] = df["timestamp"].dt.to_period("M")
    ym_list = sorted([(ym.year, ym.month) for ym in df["_ym"].unique()
                      if ym.month in season_months])
    df.drop(columns=["_ym"], inplace=True)
    tr_parts, te_parts = [], []
    for station in df["station"].unique():
        sdf = df[df["station"] == station].sort_values("timestamp")
        for y, m in ym_list:
            tr, te = slice_month(sdf, y, m)
            if tr is not None:
                tr_parts.append(tr)
            if te is not None:
                te_parts.append(te)
    df_tr = pd.concat(tr_parts, ignore_index=True).sort_values(
        ["station", "timestamp"]).reset_index(drop=True) if tr_parts else pd.DataFrame()
    df_te = pd.concat(te_parts, ignore_index=True).sort_values(
        ["station", "timestamp"]).reset_index(drop=True) if te_parts else pd.DataFrame()
    return df_tr, df_te


def build_train_val_test(df, season_months, val_days=VAL_DAYS):
    df["_ym"] = df["timestamp"].dt.to_period("M")
    ym_list = sorted([(ym.year, ym.month) for ym in df["_ym"].unique()
                      if ym.month in season_months])
    df.drop(columns=["_ym"], inplace=True)
    tr_parts, va_parts, te_parts = [], [], []
    for station in df["station"].unique():
        sdf = df[df["station"] == station].sort_values("timestamp")
        for y, m in ym_list:
            tr, va, te = slice_month_val(sdf, y, m, val_days)
            if tr is not None:
                tr_parts.append(tr)
            if va is not None:
                va_parts.append(va)
            if te is not None:
                te_parts.append(te)
    cols = df.columns.tolist()
    df_tr = pd.concat(tr_parts, ignore_index=True).sort_values(
        ["station", "timestamp"]).reset_index(drop=True) if tr_parts else pd.DataFrame(columns=cols)
    df_va = pd.concat(va_parts, ignore_index=True).sort_values(
        ["station", "timestamp"]).reset_index(drop=True) if va_parts else pd.DataFrame(columns=cols)
    df_te = pd.concat(te_parts, ignore_index=True).sort_values(
        ["station", "timestamp"]).reset_index(drop=True) if te_parts else pd.DataFrame(columns=cols)
    return df_tr, df_va, df_te


def fit_scalers(df_train, feat_cols, min_station=500, min_region=2000, bin_deg=1.0):
    def region_key(lat, lon):
        return f"_region_{round(lat / bin_deg)}_{round(lon / bin_deg)}"

    scalers_X, scalers_y = {}, {}
    lookup = {}
    region_frames = {}
    for station, sdf in df_train.groupby("station"):
        clean = sdf[feat_cols + TARGET_COLS].dropna()
        if len(clean) >= min_station:
            scalers_X[station] = RobustScaler().fit(clean[feat_cols].values.astype(np.float32))
            scalers_y[station] = RobustScaler().fit(
                clean[TARGET_COLS].values.reshape(-1, 1).astype(np.float32))
            lookup[station] = station
        else:
            rkey = region_key(sdf["lat"].mean(), sdf["lon"].mean())
            region_frames.setdefault(rkey, []).append((station, clean))
    for rkey, entries in region_frames.items():
        region_df = pd.concat([e[1] for e in entries], ignore_index=True)
        if len(region_df) >= min_region:
            scalers_X[rkey] = RobustScaler().fit(region_df[feat_cols].values.astype(np.float32))
            scalers_y[rkey] = RobustScaler().fit(
                region_df[TARGET_COLS].values.reshape(-1, 1).astype(np.float32))
            for station, _ in entries:
                lookup[station] = rkey
        else:
            for station, _ in entries:
                lookup[station] = "_global"
    leftover = df_train[feat_cols + TARGET_COLS].dropna().sample(
        min(200_000, len(df_train)), random_state=42)
    scalers_X["_global"] = RobustScaler().fit(leftover[feat_cols].values.astype(np.float32))
    scalers_y["_global"] = RobustScaler().fit(
        leftover[TARGET_COLS].values.reshape(-1, 1).astype(np.float32))
    return scalers_X, scalers_y, lookup


def get_scaler(scalers_X, scalers_y, lookup, station):
    key = lookup.get(station, "_global")
    return scalers_X.get(key, scalers_X["_global"]), scalers_y.get(key, scalers_y["_global"])


def build_flat_tensors(df, feat_cols, scalers_X, scalers_y, lookup, station_map):
    f_chunks, t_chunks = [], []
    offsets, sids = {}, {}
    ptr = 0
    for station in sorted(df["station"].unique()):
        sdf = df[df["station"] == station].sort_values("timestamp").reset_index(drop=True)
        n = len(sdf)
        if n < SEQ_LEN + N_HORIZONS:
            continue
        sx, sy = get_scaler(scalers_X, scalers_y, lookup, station)
        sdf[feat_cols] = sdf[feat_cols].ffill().bfill().fillna(0)
        feat = np.nan_to_num(sx.transform(sdf[feat_cols].values.astype(np.float32)),
                             nan=0.0, posinf=0.0, neginf=0.0)
        tgt = sy.transform(
            sdf[TARGET_COLS].values.astype(np.float32).reshape(-1, 1)
        ).reshape(n, N_HORIZONS)
        f_chunks.append(feat)
        t_chunks.append(tgt)
        offsets[station] = (ptr, n)
        sids[station] = station_map.get(station, 0)
        ptr += n
    ff = torch.from_numpy(np.vstack(f_chunks)).share_memory_()
    ft = torch.from_numpy(np.vstack(t_chunks)).share_memory_()
    return ff, ft, offsets, sids


def build_index(offsets, sids, stride=1):
    ao, ar, asid = [], [], []
    for station in sorted(offsets):
        start, n = offsets[station]
        sid = sids[station]
        valid = np.arange(SEQ_LEN, n - N_HORIZONS + 1, stride, dtype=np.int32)
        if len(valid) == 0:
            continue
        ao.append(np.full(len(valid), start, dtype=np.int64))
        ar.append(valid)
        asid.append(np.full(len(valid), sid, dtype=np.int32))
    if not ao:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32))
    return np.concatenate(ao), np.concatenate(ar), np.concatenate(asid)


def inverse_transform(arr_sc, sids, inv_station_map, scalers_y, lookup):
    n_h = arr_sc.shape[1]
    out = np.empty_like(arr_sc)
    for sid in np.unique(sids):
        mask = sids == sid
        station = inv_station_map.get(int(sid), "_global")
        key = lookup.get(station, "_global")
        sy = scalers_y.get(key, scalers_y["_global"])
        out[mask] = sy.inverse_transform(arr_sc[mask].reshape(-1, 1)).reshape(-1, n_h)
    return out


def compute_metrics(y_true_sc, y_pred_sc, sids, inv_map, scalers_y, lookup,
                    horizons=(1, 24)):
    yt = inverse_transform(y_true_sc, sids, inv_map, scalers_y, lookup)
    yp = inverse_transform(y_pred_sc, sids, inv_map, scalers_y, lookup)
    metrics = {}
    for h in horizons:
        idx = h - 1
        metrics[f"t+{h}h"] = {
            "mae": float(mean_absolute_error(yt[:, idx], yp[:, idx])),
            "rmse": float(mean_squared_error(yt[:, idx], yp[:, idx]) ** 0.5),
            "r2": float(r2_score(yt[:, idx], yp[:, idx])),
        }
    return metrics, yt, yp