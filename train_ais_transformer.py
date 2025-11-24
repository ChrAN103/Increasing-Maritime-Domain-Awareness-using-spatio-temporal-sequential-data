from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import random
import glob
import matplotlib.pyplot as plt

from prepare_ais_data import (
    build_sequences_from_parquet,
    DYN_FEATURES,
    MultiSequenceAISDataset,
)

random.seed(151100)

TRAIN_PARQUET = "aisdk-2023-01-01.parquet"
VAL_PARQUET   = "aisdk-2023-01-02.parquet"
TEST_PARQUET  = "aisdk-2023-01-03.parquet"


def load_sequences_for_days(paths, lookback, horizon, min_extra=1):
    all_seqs = []

    for path in paths:
        print(f"Loading {path} ...")
        df_raw = pd.read_parquet(path)
        seqs = build_sequences_from_parquet(
            df_raw, min_len=lookback + horizon + min_extra
        )
        print(f"  {len(seqs)} sequences found")
        all_seqs.extend(seqs)

    return all_seqs



# -----------------------------
# Transformer model
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(0)
        return x + self.pe[:L].unsqueeze(1)


class TimeSeriesTransformerStatic(nn.Module):
    def __init__(
        self,
        d_dyn: int = 6,
        ship_type_emb_size: int = 8,
        mmsi_emb_size: int = 16,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        target_dim: int = 5,      # ΔLat, ΔLon, ΔSOG, ΔCOG_sin, ΔCOG_cos
        ship_type_vocab: int = 10,
        mmsi_vocab: int = 100,
    ):
        super().__init__()
        self.dyn_proj = nn.Linear(d_dyn, d_model)
        self.ship_emb = nn.Embedding(ship_type_vocab, ship_type_emb_size)
        self.mmsi_emb = nn.Embedding(mmsi_vocab, mmsi_emb_size)
        self.static_proj = nn.Linear(ship_type_emb_size + mmsi_emb_size + 2, d_model)

        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, target_dim))

    @staticmethod
    def _causal_mask(L, device):
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def forward(self, x_dyn, x_stat):
        """
        x_dyn: (B,L,6)
        x_stat: (B,4) = [ship_type_id, mmsi_id, width_norm, length_norm]
        """
        B, L, _ = x_dyn.shape
        device = x_dyn.device

        ship_id = x_stat[:, 0].long()
        mmsi_id = x_stat[:, 1].long()
        width_len = x_stat[:, 2:]

        ship_e = self.ship_emb(ship_id)
        mmsi_e = self.mmsi_emb(mmsi_id)
        stat = torch.cat([ship_e, mmsi_e, width_len], dim=-1)
        stat = self.static_proj(stat).unsqueeze(1).expand(B, L, -1)

        h = self.dyn_proj(x_dyn) + stat
        h = self.pos(h.transpose(0, 1))
        z = self.encoder(h, mask=self._causal_mask(L, device))
        return self.head(z[-1])


# -----------------------------
# Training config + loop
# -----------------------------
@dataclass
class TrainConfig:
    lookback: int = 64
    horizon: int = 1
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 50
    early_stopping_patience: int = 5   # stop if no val improvement for this many epochs
    early_stopping_min_delta: float = 5e-5  # minimum improvement in val MSE to count


PARQUET_FOLDER = "data/ais_parquets"   # <-- change to your folder

def split_parquet_files(train_n=300, val_n=30, test_n=35):
    # find all .parquet files
    files = sorted(glob.glob(f"{PARQUET_FOLDER}/*.parquet"))
    if len(files) < train_n + val_n + test_n:
        raise ValueError(f"Not enough parquet files. Found {len(files)}.")

    random.shuffle(files)

    train_files = files[:train_n]
    val_files   = files[train_n:train_n+val_n]
    test_files  = files[train_n+val_n:train_n+val_n+test_n]

    return train_files, val_files, test_files

# OVERRIDE: use fixed three days
train_files = ["aisdk-2023-01-01.parquet"]
val_files   = ["aisdk-2023-01-02.parquet"]
test_files  = ["aisdk-2023-01-03.parquet"]

print("\n*** OVERRIDE: USING FIXED 3 DAYS ***")
print("Train:", train_files)
print("Val:  ", val_files)
print("Test: ", test_files, "\n")


def train_with_train_val_test(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # pick random parquet files
    # OVERRIDE: use fixed three days
    train_files = ["aisdk-2023-01-01.parquet"]
    val_files   = ["aisdk-2023-01-02.parquet"]
    test_files  = ["aisdk-2023-01-03.parquet"]

    print("\n*** OVERRIDE: USING FIXED 3 DAYS ***")
    print("Train:", train_files)
    print("Val:  ", val_files)
    print("Test: ", test_files, "\n")


    # load sequences
    train_seqs = load_sequences_for_days(train_files, cfg.lookback, cfg.horizon)
    val_seqs   = load_sequences_for_days(val_files,   cfg.lookback, cfg.horizon)
    test_seqs  = load_sequences_for_days(test_files,  cfg.lookback, cfg.horizon)


    all_seqs = train_seqs + val_seqs + test_seqs
    all_mmsi = sorted({s["mmsi"] for s in all_seqs})
    mmsi_to_id = {m: i for i, m in enumerate(all_mmsi)}
    print(f"Total unique MMSI: {len(mmsi_to_id)}")

    train_ds = MultiSequenceAISDataset(
        train_seqs,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        normalize=True,
        fit_stats=None,
        mmsi_to_id=mmsi_to_id,
    )
    fit_stats = (train_ds.dyn_mean, train_ds.dyn_std, train_ds.wl_mean, train_ds.wl_std)

    def mk_ds(seqs):
        return MultiSequenceAISDataset(
            seqs,
            lookback=cfg.lookback,
            horizon=cfg.horizon,
            normalize=True,
            fit_stats=fit_stats,
            mmsi_to_id=mmsi_to_id,
        ) if seqs else None

    val_ds = mk_ds(val_seqs)
    test_ds = mk_ds(test_seqs)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False) if val_ds else None

    max_ship_type_id = max(int(s["df"]["Ship type_id"].max()) for s in train_seqs)
    ship_type_vocab = max_ship_type_id + 1
    mmsi_vocab = len(mmsi_to_id)

    model = TimeSeriesTransformerStatic(
        d_dyn=len(DYN_FEATURES),
        ship_type_vocab=ship_type_vocab,
        mmsi_vocab=mmsi_vocab,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state_dict = None
    no_improve_epochs = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.epochs + 1):
        # ----- TRAIN -----
        model.train()
        running_loss, n_batches = 0.0, 0
        for x_dyn, x_stat, y in train_dl:
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
            pred = model(x_dyn, x_stat)          # (B,5)

            # primary: Δlat/Δlon, aux: ΔSOG/ΔCOG
            loss_latlon = loss_fn(pred[:, :2], y[:, :2])
            loss_sogcog = loss_fn(pred[:, 2:], y[:, 2:])
            loss = loss_latlon + 0.2 * loss_sogcog

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += loss.item()
            n_batches += 1

        train_mse = running_loss / max(1, n_batches)

        # ----- VALIDATION -----
        if val_dl is not None:
            model.eval()
            val_se, val_n = 0.0, 0
            with torch.no_grad():
                for x_dyn, x_stat, y in val_dl:
                    x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
                    pred = model(x_dyn, x_stat)
                    val_se += (pred - y).pow(2).sum().item()
                    val_n += y.numel()
            val_mse = val_se / max(1, val_n)

            print(
                f"Epoch {epoch}/{cfg.epochs} - "
                f"train MSE: {train_mse:.6f} | val MSE: {val_mse:.6f}"
            )

            # ----- EARLY STOPPING -----
            if val_mse + cfg.early_stopping_min_delta < best_val_mse:
                best_val_mse = val_mse
                best_state_dict = model.state_dict()
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= cfg.early_stopping_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(best val MSE: {best_val_mse:.6f})"
                    )
                    break
        else:
            print(f"Epoch {epoch}/{cfg.epochs} - train MSE: {train_mse:.6f}")

        train_losses.append(train_mse)
        if val_dl is not None:
            val_losses.append(val_mse)
        else:
            val_losses.append(None)

    print("Training finished.")

    # Restore best model (if we used validation)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, train_ds, val_ds, test_ds, train_seqs, val_seqs, test_seqs, train_losses, val_losses



# -----------------------------
# Helpers for inference
# -----------------------------
def estimate_dt_seconds(df, default=60.0):
    ts = df["Timestamp"].to_numpy()
    if len(ts) < 2:
        return default
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    return float(np.median(diffs)) if len(diffs) else default


@torch.no_grad()
def _make_stat_tensor(ds, seq0, row_idx, device):
    df = seq0["df"]
    ship_type_id = df["Ship type_id"].iloc[row_idx]
    width = df["Width"].iloc[row_idx]
    length = df["Length"].iloc[row_idx]
    wl_norm = (np.array([width, length], np.float32) - ds.wl_mean) / ds.wl_std
    mmsi_id = ds.mmsi_to_id[seq0["mmsi"]]
    x_stat = np.array([ship_type_id, mmsi_id, wl_norm[0], wl_norm[1]], np.float32)
    return torch.tensor(x_stat, device=device).unsqueeze(0)  # (1,4)


@torch.no_grad()
def predict_trajectory_with_truth_direct(model, ds, seq0, horizon=32, history_extra=50):
    device = next(model.parameters()).device
    df = seq0["df"].reset_index(drop=True)
    lookback = ds.lookback
    dyn_mean, dyn_std = ds.dyn_mean, ds.dyn_std
    idx_ts  = DYN_FEATURES.index("Timestamp")
    idx_lat = DYN_FEATURES.index("Latitude")
    idx_lon = DYN_FEATURES.index("Longitude")

    N = len(df)
    if N < lookback + horizon + 5:
        raise ValueError("Sequence too short for lookback + horizon.")

    end_input = N - horizon - 1
    start_input = end_input - lookback + 1
    if start_input < 0:
        raise ValueError("Not enough history for lookback.")

    df_input = df.iloc[start_input:end_input + 1]
    df_future_true = df.iloc[end_input + 1:end_input + 1 + horizon]
    hist_start = max(0, start_input - history_extra)
    history_df = df.iloc[hist_start:end_input + 1]

    window_raw = df_input[DYN_FEATURES].to_numpy(np.float32)  # (L,6)
    x_stat = _make_stat_tensor(ds, seq0, end_input, device)
    ts_last = float(window_raw[-1, idx_ts])
    dt_seconds = estimate_dt_seconds(df, default=60.0)

    future_pred = []
    model.eval()
    for step in range(1, horizon + 1):
        dyn_norm = (window_raw - dyn_mean) / dyn_std
        x_dyn = torch.tensor(dyn_norm, dtype=torch.float32, device=device).unsqueeze(0)

        delta_norm = model(x_dyn, x_stat).cpu().numpy()[0]  # (5,)
        lat_norm_last = dyn_norm[-1, idx_lat]
        lon_norm_last = dyn_norm[-1, idx_lon]

        lat_norm_next = lat_norm_last + delta_norm[0]
        lon_norm_next = lon_norm_last + delta_norm[1]

        lat_next = float(lat_norm_next * dyn_std[idx_lat] + dyn_mean[idx_lat])
        lon_next = float(lon_norm_next * dyn_std[idx_lon] + dyn_mean[idx_lon])

        ts_last += dt_seconds
        future_pred.append(
            dict(
                step=step,
                t_rel_sec=step * dt_seconds,
                timestamp_sec=ts_last,
                lat=lat_next,
                lon=lon_next,
            )
        )

        last_row = window_raw[-1].copy()
        next_row = last_row
        next_row[idx_ts]  = ts_last
        next_row[idx_lat] = lat_next
        next_row[idx_lon] = lon_next
        window_raw = np.vstack([window_raw[1:], next_row])

    return history_df, df_future_true, future_pred


@torch.no_grad()
def predict_trajectory_10_steps_direct(model, ds, seq0):
    device = next(model.parameters()).device
    df = seq0["df"]
    lookback = ds.lookback
    dyn_mean, dyn_std = ds.dyn_mean, ds.dyn_std
    idx_ts  = DYN_FEATURES.index("Timestamp")
    idx_lat = DYN_FEATURES.index("Latitude")
    idx_lon = DYN_FEATURES.index("Longitude")

    window_raw = df[DYN_FEATURES].iloc[-lookback:].to_numpy(np.float32)
    x_stat = _make_stat_tensor(ds, seq0, len(df) - 1, device)
    ts_last = float(window_raw[-1, idx_ts])
    dt_seconds = estimate_dt_seconds(df, default=60.0)

    traj = []
    model.eval()
    for step in range(1, 33):
        dyn_norm = (window_raw - dyn_mean) / dyn_std
        x_dyn = torch.tensor(dyn_norm, dtype=torch.float32, device=device).unsqueeze(0)
        pred_norm = model(x_dyn, x_stat).cpu().numpy()[0]  # (5,)

        lat_pred = pred_norm[0] * dyn_std[idx_lat] + dyn_mean[idx_lat]
        lon_pred = pred_norm[1] * dyn_std[idx_lon] + dyn_mean[idx_lon]

        ts_last += dt_seconds
        traj.append(
            dict(
                step=step,
                t_rel_sec=step * dt_seconds,
                timestamp_sec=ts_last,
                lat=float(lat_pred),
                lon=float(lon_pred),
            )
        )

        last_row = window_raw[-1].copy()
        next_row = last_row
        next_row[idx_ts]  = ts_last
        next_row[idx_lat] = lat_pred
        next_row[idx_lon] = lon_pred
        window_raw = np.vstack([window_raw[1:], next_row])

    return traj


def evaluate_mse(model, dataset, batch_size=128, device=None):
    if dataset is None or len(dataset) == 0:
        return None
    if device is None:
        device = next(model.parameters()).device

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    se_total, n_total = 0.0, 0
    with torch.no_grad():
        for x_dyn, x_stat, y in dl:
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
            pred = model(x_dyn, x_stat)
            se_total += (pred - y).pow(2).sum().item()
            n_total += y.numel()
    return se_total / max(1, n_total)


def save_model(model, path="trained_ais_transformer.pt"):
    torch.save(model.state_dict(), path)
    print(f"\nModel saved to: {path}")

# -----------------------------
# Main
# -----------------------------
def main():
    cfg = TrainConfig(lookback=64, horizon=1, batch_size=64, lr=3e-4, epochs=5)
    (
    model,
    train_ds,
    val_ds,
    test_ds,
    train_seqs,
    val_seqs,
    test_seqs,
    train_losses,
    val_losses,
) = train_with_train_val_test(cfg)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_mse = evaluate_mse(model, test_ds, batch_size=cfg.batch_size, device=device)
    if test_mse is not None:
        print(f"\nFinal TEST MSE: {test_mse:.6f}")
    else:
        print("\nNo test data available, TEST MSE not computed.")
    
    # Save model
    save_model(model, "ais_transformer_model.pt")

    # Save normalization statistics
    np.savez(
        "ais_transformer_stats.npz",
        dyn_mean=train_ds.dyn_mean,
        dyn_std=train_ds.dyn_std,
        wl_mean=train_ds.wl_mean,
        wl_std=train_ds.wl_std,
        mmsi_to_id=train_ds.mmsi_to_id,
    )

    print("Saved normalization stats to ais_transformer_stats.npz")


    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training & Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
