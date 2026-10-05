import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from common import (
    PM25LSTM, WeightedHuberLoss, SequenceDataset,
    load_data, resolve_features, build_train_test, fit_scalers,
    build_flat_tensors, build_index, compute_metrics,
    SEASONS, WEATHER_FEATURES,
)

TOP_WITHOUT_WEATHER = [
    "pm25", "aqi_category", "pm25_rmax24", "pm25_rmean24",
    "pm25_lag1", "lon", "pm25_lag6", "pm25_lag3",
    "pm25_rmin24", "pm25_lag12", "hour_sin",
    "pm25_trend6h", "pm25_rstd24", "pm25_lag24", "pm25_lag72",
]

ALL_SEASON_MONTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]
EPOCHS = 30
BATCH = 512
PEAK_LR = 3e-4
WARMUP = 3
STRIDE = 3
WEIGHT_DECAY = 5e-4


def train(df):
    feat_cols = resolve_features(df, TOP_WITHOUT_WEATHER)
    df_tr, df_te = build_train_test(df, ALL_SEASON_MONTHS)
    all_stations = sorted(df["station"].unique())
    station_map = {s: i + 1 for i, s in enumerate(all_stations)}
    inv_map = {v: k for k, v in station_map.items()}
    n_stations = len(station_map)

    scalers_X, scalers_y, lookup = fit_scalers(df_tr, feat_cols)
    ff_tr, ft_tr, off_tr, sid_tr = build_flat_tensors(
        df_tr, feat_cols, scalers_X, scalers_y, lookup, station_map)
    ao_tr, ar_tr, asid_tr = build_index(off_tr, sid_tr, stride=STRIDE)

    ff_te, ft_te, off_te, sid_te = build_flat_tensors(
        df_te, feat_cols, scalers_X, scalers_y, lookup, station_map)
    ao_te, ar_te, asid_te = build_index(off_te, sid_te, stride=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr = SequenceDataset(ff_tr, ft_tr, ao_tr, ar_tr, asid_tr)
    ds_te = SequenceDataset(ff_te, ft_te, ao_te, ar_te, asid_te)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=2)
    dl_te = DataLoader(ds_te, batch_size=512, shuffle=False, num_workers=2)

    model = PM25LSTM(len(feat_cols), n_stations).to(device)
    loss_fn = WeightedHuberLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    warm = LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP)

    for epoch in range(EPOCHS):
        model.train()
        for X, sids, y in tqdm(dl_tr, desc=f"no_met ep{epoch+1}", leave=False):
            X, sids, y = X.to(device), sids.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(X, sids)
            loss = loss_fn(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch < WARMUP:
            warm.step()

    model.eval()
    yt_all, yp_all, sid_all = [], [], []
    with torch.no_grad():
        for X, sids, y in dl_te:
            X, sids = X.to(device), sids.to(device)
            p = model(X, sids).cpu().numpy()
            yt_all.append(y.numpy())
            yp_all.append(p)
            sid_all.append(sids.cpu().numpy())
    yt = np.concatenate(yt_all)
    yp = np.concatenate(yp_all)
    sid = np.concatenate(sid_all)

    metrics, _, _ = compute_metrics(yt, yp, sid, inv_map, scalers_y, lookup)
    torch.save({"model": model.state_dict(),
                "n_features": len(feat_cols),
                "n_stations": n_stations,
                "feature_cols": feat_cols},
               "lstm_no_meteorology.pt")
    for h in (1, 24):
        m = metrics[f"t+{h}h"]
        print(f"no_met t+{h}: MAE {m['mae']:.3f} RMSE {m['rmse']:.3f} R2 {m['r2']:.3f}")


if __name__ == "__main__":
    train(load_data())