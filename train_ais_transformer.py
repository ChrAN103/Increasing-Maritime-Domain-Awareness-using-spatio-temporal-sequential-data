import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Import your data utilities from the previous file
from prepare_ais_data import (
    PARQUET_PATH,
    build_sequences_from_parquet,
    prep_dataframe,
    DYN_FEATURES,
    MultiSequenceAISDataset, 
)
import pandas as pd

PREDICT_COLS = [3, 4, 5]  # SOG, COG_sin, COG_cos


# -----------------------------
# Transformer model (static + dynamic)
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
        d_dyn: int = 6,                 # Timestamp, Lat, Lon, SOG, COG_sin, COG_cos
        ship_type_emb_size: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        target_dim: int = 6,            # how many dyn features you predict
        ship_type_vocab: int = 10,      # number of unique ship types
    ):
        super().__init__()

        # dynamic projection
        self.dyn_proj = nn.Linear(d_dyn, d_model)

        # static: [ship_type_id, width_norm, length_norm]
        self.ship_emb = nn.Embedding(ship_type_vocab, ship_type_emb_size)
        self.static_proj = nn.Linear(ship_type_emb_size + 2, d_model)

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
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, target_dim),
        )

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def forward(self, x_dyn, x_stat):
        """
        x_dyn: (B, L, 6)
        x_stat: (B, 3) = [ship_type_id, width_norm, length_norm]
        """
        B, L, _ = x_dyn.shape
        device = x_dyn.device

        ship_id = x_stat[:, 0].long()   # (B,)
        width_len = x_stat[:, 1:]       # (B, 2)

        emb = self.ship_emb(ship_id)                      # (B, emb_dim)
        stat_full = torch.cat([emb, width_len], dim=-1)   # (B, emb_dim+2)
        stat_proj = self.static_proj(stat_full).unsqueeze(1)  # (B,1,D)
        stat_expand = stat_proj.repeat(1, L, 1)               # (B,L,D)

        dyn_proj = self.dyn_proj(x_dyn)   # (B,L,D)
        h = dyn_proj + stat_expand        # inject static context

        h = h.transpose(0, 1)  # (L,B,D)
        h = self.pos(h)
        mask = self._causal_mask(L, device)
        z = self.encoder(h, mask=mask)    # (L,B,D)
        z_last = z[-1]                    # (B,D)
        out = self.head(z_last)           # (B,target_dim)
        return out


# -----------------------------
# Training config
# -----------------------------
@dataclass
class TrainConfig:
    lookback: int = 64
    horizon: int = 1
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 10   # small to start; you can increase


# -----------------------------
# Training loop
# -----------------------------
def train_on_all_sequences(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) Load parquet and build sequences
    df_raw = pd.read_parquet(PARQUET_PATH)
    sequences = build_sequences_from_parquet(df_raw, min_len=cfg.lookback + cfg.horizon + 1)
    print(f"Found {len(sequences)} sequences with length >= {cfg.lookback + cfg.horizon + 1}")

    if not sequences:
        raise RuntimeError("No sequences available for training.")

    # 2) Dataset & DataLoader over ALL sequences
    ds = MultiSequenceAISDataset(
        sequences,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        target_cols=PREDICT_COLS,
        normalize=True,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # 3) Build transformer with target_dim = len(PREDICT_COLS)
    max_ship_type_id = max(int(seq["df"]["Ship type_id"].max()) for seq in sequences)
    ship_type_vocab = max_ship_type_id + 1

    model = TimeSeriesTransformerStatic(
        d_dyn=len(DYN_FEATURES),        # 6 dynamic inputs
        target_dim=len(PREDICT_COLS),   # 3 outputs: SOG, COG_sin, COG_cos
        ship_type_vocab=ship_type_vocab,
        d_model=128,
        nhead=4,
        num_layers=4,
        d_ff=512,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    # 4) Training loop
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for x_dyn, x_stat, y in dl:
            x_dyn = x_dyn.to(device)   # (B,L,6)
            x_stat = x_stat.to(device) # (B,3)
            y = y.to(device)           # (B,3)

            pred = model(x_dyn, x_stat)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        print(f"Epoch {epoch}/{cfg.epochs} - train MSE: {avg_loss:.6f}")

    print("Training finished.")
    return model, ds, sequences



# -----------------------------
# Simple inference: predict next step for last window
# -----------------------------
@torch.no_grad()
def predict_next_step_position(model, ds: MultiSequenceAISDataset, seq0):
    """
    Uses the last lookback window from seq0, predicts next-step SOG + COG,
    and computes the next latitude/longitude.
    """
    device = next(model.parameters()).device
    df = seq0["df"]

    lookback = ds.lookback
    dyn_mean, dyn_std = ds.dyn_mean, ds.dyn_std

    # Indices in DYN_FEATURES
    idx_SOG = DYN_FEATURES.index("SOG")         # 3
    idx_sin = DYN_FEATURES.index("COG_sin")     # 4
    idx_cos = DYN_FEATURES.index("COG_cos")     # 5

    # 1) Get last window of *normalized* dynamic features
    last_dyn = df[DYN_FEATURES].iloc[-lookback:].to_numpy(np.float32)  # (L,6)
    dyn_norm = (last_dyn - dyn_mean) / dyn_std

    # 2) Static features (ship_type_id, width, length)
    ship_type_id = df["Ship type_id"].iloc[-1]
    width = df["Width"].iloc[-1]
    length = df["Length"].iloc[-1]
    wl_norm = (np.array([width, length], dtype=np.float32) - ds.wl_mean) / ds.wl_std

    x_dyn = torch.tensor(dyn_norm, dtype=torch.float32, device=device).unsqueeze(0)   # (1,L,6)
    x_stat = torch.tensor(
        np.concatenate([[ship_type_id], wl_norm]),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)   # (1,3)

    model.eval()
    pred_norm = model(x_dyn, x_stat).cpu().numpy()[0]  # (3,) -> [SOG, COG_sin, COG_cos] normalized

    # 3) Unnormalize to original scale
    pred_dyn = np.zeros(len(DYN_FEATURES), dtype=np.float32)
    for i, col_idx in enumerate(PREDICT_COLS):
        pred_dyn[col_idx] = pred_norm[i] * dyn_std[col_idx] + dyn_mean[col_idx]

    sog = float(pred_dyn[idx_SOG])
    cog_sin = float(pred_dyn[idx_sin])
    cog_cos = float(pred_dyn[idx_cos])

    # 4) Reconstruct COG in degrees
    cog_rad = math.atan2(cog_sin, cog_cos)
    cog_deg = (math.degrees(cog_rad) + 360.0) % 360.0

    # 5) Compute dt (seconds) as difference between last two timestamps
    #    (You can clamp it or fall back to e.g. 60s if needed.)
    ts = df["Timestamp"].iloc[-lookback-1 : -lookback+1].to_numpy()
    if len(ts) >= 2:
        dt_seconds = float(ts[-1] - ts[-2])
    else:
        dt_seconds = 60.0  # fallback, if only one timestamp available

    # 6) Last known position
    last_lat = float(df["Latitude"].iloc[-1])
    last_lon = float(df["Longitude"].iloc[-1])

    # 7) Propagate position
    next_lat, next_lon = step_position(
        lat_deg=last_lat,
        lon_deg=last_lon,
        sog_knots=sog,
        cog_deg=cog_deg,
        dt_seconds=dt_seconds,
    )

    return {
        "last_lat": last_lat,
        "last_lon": last_lon,
        "next_lat": next_lat,
        "next_lon": next_lon,
        "dt_seconds": dt_seconds,
        "SOG_pred": sog,
        "COG_deg_pred": cog_deg,
    }


EARTH_RADIUS_M = 6371000.0       # mean Earth radius in meters
KNOT_TO_MPS = 0.514444           # 1 knot = 0.514444 m/s

def step_position(lat_deg, lon_deg, sog_knots, cog_deg, dt_seconds):
    """
    Move a ship from (lat, lon) with speed sog_knots and heading cog_deg
    for dt_seconds along a great-circle path.

    Returns new (lat_deg2, lon_deg2).
    """
    # Convert to radians
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    bearing = math.radians(cog_deg)

    # Distance travelled on Earth's surface
    speed_mps = sog_knots * KNOT_TO_MPS
    distance = speed_mps * dt_seconds   # meters
    delta = distance / EARTH_RADIUS_M   # angular distance (radians)

    # Great-circle formulas
    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)
    cos_bear = math.cos(bearing)
    sin_bear = math.sin(bearing)

    sin_lat2 = sin_lat1 * cos_delta + cos_lat1 * sin_delta * cos_bear
    lat2 = math.asin(sin_lat2)

    y = sin_bear * sin_delta * cos_lat1
    x = cos_delta - sin_lat1 * sin_lat2
    lon2 = lon1 + math.atan2(y, x)

    # Normalize lon to [-180, 180) or [0, 360) if you like
    lon2 = (lon2 + math.pi) % (2 * math.pi) - math.pi

    return math.degrees(lat2), math.degrees(lon2)


def estimate_dt_seconds(df, default=60.0):
    """
    Estimate a typical timestep in seconds as the median of positive timestamp differences.
    df['Timestamp'] must already be in seconds since start (float).
    """
    ts = df["Timestamp"].to_numpy()
    if len(ts) < 2:
        return default
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return default
    return float(np.median(diffs))

@torch.no_grad()
def predict_trajectory_10_steps(model, ds: MultiSequenceAISDataset, seq0):
    """
    Autoregressively predict 10 future steps of motion (SOG + COG)
    and integrate to get future lat/lon positions.

    Returns a list of dicts, each with:
        step, timestamp, lat, lon, SOG_pred, COG_deg_pred
    """
    device = next(model.parameters()).device
    df = seq0["df"]  # prepared df with Timestamp in seconds, etc.

    lookback = ds.lookback
    dyn_mean, dyn_std = ds.dyn_mean, ds.dyn_std
    wl_mean, wl_std = ds.wl_mean, ds.wl_std

    # Where in DYN_FEATURES are the things we care about?
    idx_ts  = DYN_FEATURES.index("Timestamp")
    idx_lat = DYN_FEATURES.index("Latitude")
    idx_lon = DYN_FEATURES.index("Longitude")
    idx_sog = DYN_FEATURES.index("SOG")
    idx_sin = DYN_FEATURES.index("COG_sin")
    idx_cos = DYN_FEATURES.index("COG_cos")

    # Initial window in *raw* scale (not normalized)
    window_raw = df[DYN_FEATURES].iloc[-lookback:].to_numpy(np.float32)  # (L,6)

    # Static features (same for all steps)
    ship_type_id = df["Ship type_id"].iloc[-1]
    width = df["Width"].iloc[-1]
    length = df["Length"].iloc[-1]
    wl_norm = (np.array([width, length], dtype=np.float32) - wl_mean) / wl_std

    # Starting point
    ts = float(window_raw[-1, idx_ts])
    lat = float(window_raw[-1, idx_lat])
    lon = float(window_raw[-1, idx_lon])

    dt_seconds = estimate_dt_seconds(df, default=60.0)

    trajectory = []

    model.eval()
    for step in range(1, 11):  # 10 steps
        # 1) Normalize current window
        dyn_norm = (window_raw - dyn_mean) / dyn_std  # (L,6)

        x_dyn = torch.tensor(dyn_norm, dtype=torch.float32, device=device).unsqueeze(0)  # (1,L,6)
        x_stat = torch.tensor(
            np.concatenate([[ship_type_id], wl_norm]),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)  # (1,3)

        # 2) Predict normalized [SOG, COG_sin, COG_cos]
        pred_norm = model(x_dyn, x_stat).cpu().numpy()[0]  # (3,)

        # 3) Unnormalize into full dynamic space (we only fill the predicted cols)
        pred_dyn = np.zeros(len(DYN_FEATURES), dtype=np.float32)
        for i, col_idx in enumerate(PREDICT_COLS):
            pred_dyn[col_idx] = pred_norm[i] * dyn_std[col_idx] + dyn_mean[col_idx]

        sog = float(pred_dyn[idx_sog])
        cog_sin = float(pred_dyn[idx_sin])
        cog_cos = float(pred_dyn[idx_cos])

        # 4) Heading back to degrees
        cog_rad = math.atan2(cog_sin, cog_cos)
        cog_deg = (math.degrees(cog_rad) + 360.0) % 360.0

        # 5) Advance time
        ts = ts + dt_seconds

        # 6) Integrate position
        lat, lon = step_position(
            lat_deg=lat,
            lon_deg=lon,
            sog_knots=sog,
            cog_deg=cog_deg,
            dt_seconds=dt_seconds,
        )

        # 7) Build next dynamic row in RAW scale
        next_row = np.zeros(len(DYN_FEATURES), dtype=np.float32)
        next_row[idx_ts]  = ts
        next_row[idx_lat] = lat
        next_row[idx_lon] = lon
        next_row[idx_sog] = sog
        next_row[idx_sin] = cog_sin
        next_row[idx_cos] = cog_cos

        # 8) Append to trajectory
        trajectory.append(
            dict(
                step=step,
                timestamp_sec=ts,
                lat=lat,
                lon=lon,
                SOG_pred=sog,
                COG_deg_pred=cog_deg,
            )
        )

        # 9) Roll window: drop oldest, append new
        window_raw = np.vstack([window_raw[1:], next_row])

    return trajectory

# -----------------------------
# Main
# -----------------------------
def main():
    cfg = TrainConfig(
        lookback=64,
        horizon=1,
        batch_size=128,
        lr=3e-4,
        epochs=5,
    )
    model, ds, sequences = train_on_all_sequences(cfg)

    # For trajectory visualization/prediction, we can still pick one sequence, e.g. the first:
    seq0 = sequences[0]
    traj = predict_trajectory_10_steps(model, ds, seq0)

    print("\n10-step predicted trajectory:")
    for row in traj:
        print(
            f"step {row['step']:2d}  t+{(row['step'] * traj[0]['timestamp_sec'] - traj[0]['timestamp_sec']):.1f}s  "
            f"lat={row['lat']:.5f}, lon={row['lon']:.5f},  "
            f"SOG={row['SOG_pred']:.2f} kn, COG={row['COG_deg_pred']:.1f}°"
        )


if __name__ == "__main__":
    main()
