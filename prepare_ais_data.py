# prepare_ais_data.py

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
PARQUET_PATH = "aisdk-2023-01-01.parquet"

# Dynamic features fed to the transformer at each timestep
DYN_FEATURES = ["Timestamp", "Latitude", "Longitude", "SOG", "COG_sin", "COG_cos"]

# Static features (do not change within a sequence)
STATIC_FEATURES = ["Ship type", "Width", "Length"]


# -----------------------------
# Preprocessing for ONE sequence
# -----------------------------
def prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare ONE continuous time series for the transformer.

    Expected columns in df (raw):
        Timestamp (datetime or string)
        Latitude  (float)
        Longitude (float)
        SOG       (float)
        COG       (float, degrees 0–360)
        Ship type (string or int)
        Width     (float)
        Length    (float)
    """
    df = df.copy()

    # Keep only needed columns
    required = ["Timestamp", "Latitude", "Longitude", "SOG", "COG",
                "Ship type", "Width", "Length"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}")
    df = df[required].dropna().reset_index(drop=True)

    if len(df) == 0:
        return df  # empty

    # Timestamp → float seconds since start
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    if len(df) == 0:
        return df

    t0 = df["Timestamp"].iloc[0]
    df["Timestamp"] = (df["Timestamp"] - t0).dt.total_seconds().astype("float32")

    # COG → sin/cos on unit circle
    cog_rad = np.deg2rad(df["COG"].astype("float32").to_numpy())
    df["COG_sin"] = np.sin(cog_rad).astype("float32")
    df["COG_cos"] = np.cos(cog_rad).astype("float32")
    df = df.drop(columns=["COG"])

    # Numeric fields to float32
    for c in ["Latitude", "Longitude", "SOG", "Width", "Length"]:
        df[c] = df[c].astype("float32")

    # --- Ship type → integer ID (for embedding later) ---
    col_str = df["Ship type"].astype("string")
    unique_types = sorted(col_str.dropna().unique())
    ship_map = {v: i for i, v in enumerate(unique_types)}
    df["Ship type_id"] = col_str.map(ship_map).astype("int64")

    return df


# -----------------------------
# Build sequences from the parquet
# -----------------------------
def build_sequences_from_parquet(df_raw: pd.DataFrame,
                                 min_len: int = 100) -> List[Dict[str, Any]]:
    """
    Group by (MMSI, Segment), prepare each group, and return
    a list of sequences:

    Each item:
        {
          "mmsi": ...,
          "segment": ...,
          "df": prepared_dataframe_for_this_sequence
        }
    """
    # Safety: drop rows missing key columns
    needed = [
        "Timestamp", "Latitude", "Longitude", "SOG", "COG",
        "Ship type", "Width", "Length", "MMSI", "Segment"
    ]
    for col in needed:
        if col not in df_raw.columns:
            raise ValueError(f"Missing expected column {col!r} in raw parquet data.")

    df_raw = df_raw.dropna(subset=needed)

    sequences: List[Dict[str, Any]] = []

    # Group by MMSI + Segment so each group is one track
    for (mmsi, seg), df_group in df_raw.groupby(["MMSI", "Segment"]):
        df_group = df_group.sort_values("Timestamp")
        df_prep = prep_dataframe(df_group)
        if len(df_prep) < min_len:
            continue

        sequences.append({
            "mmsi": mmsi,
            "segment": seg,
            "df": df_prep,
        })

    return sequences


# -----------------------------
# Dataset for one prepared sequence
# -----------------------------
class AISDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int = 64,
        horizon: int = 1,
        target_cols: Optional[Sequence[int]] = None,
        normalize: bool = True,
        fit_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        """
        Each sample:
            x_dyn:  (L, 6)   → [Timestamp, Lat, Lon, SOG, COG_sin, COG_cos] (normalized)
            x_stat: (3,)     → [ship_type_id, width_norm, length_norm]
            y:      (T,)     → target(s) from dynamic features at prediction time (normalized)

        df is the output of prep_dataframe().
        """
        df = df.reset_index(drop=True)

        # dynamic features
        dyn = df[DYN_FEATURES].to_numpy(np.float32)               # (N,6)
        # static features
        ship_type = df["Ship type_id"].to_numpy(np.int64)         # (N,)
        width_len = df[["Width", "Length"]].to_numpy(np.float32)  # (N,2)

        self.lookback = lookback
        self.horizon = horizon
        self.target_cols = np.array(
            target_cols if target_cols is not None else np.arange(len(DYN_FEATURES))
        )

        max_start = len(dyn) - (lookback + horizon)
        self.starts = np.arange(0, max_start + 1, dtype=np.int64)

        # Fit or use normalization stats
        if fit_stats is None:
            if normalize:
                dyn_mean = dyn.mean(axis=0)
                dyn_std = dyn.std(axis=0)
                wl_mean = width_len.mean(axis=0)
                wl_std = width_len.std(axis=0)
            else:
                dyn_mean = np.zeros(dyn.shape[1], dtype=np.float32)
                dyn_std = np.ones(dyn.shape[1], dtype=np.float32)
                wl_mean = np.zeros(2, dtype=np.float32)
                wl_std = np.ones(2, dtype=np.float32)
        else:
            dyn_mean, dyn_std, wl_mean, wl_std = fit_stats

        dyn_std = np.where(dyn_std < 1e-6, 1.0, dyn_std).astype("float32")
        wl_std = np.where(wl_std < 1e-6, 1.0, wl_std).astype("float32")

        self.dyn = (dyn - dyn_mean) / dyn_std
        self.ship_type = ship_type
        self.width_len = (width_len - wl_mean) / wl_std

        self.dyn_mean, self.dyn_std = dyn_mean, dyn_std
        self.wl_mean, self.wl_std = wl_mean, wl_std

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = int(self.starts[idx])
        e = s + self.lookback
        t = e + self.horizon - 1

        x_dyn = self.dyn[s:e]  # (L, 6)
        ship_id = self.ship_type[s]
        wl = self.width_len[s]  # (2,)

        y = self.dyn[t, self.target_cols]  # (T,)

        x_dyn = torch.from_numpy(x_dyn)  # (L,6)
        x_stat = torch.tensor(
            np.concatenate([[ship_id], wl.astype("float32")]),
            dtype=torch.float32,
        )  # (3,)
        y = torch.from_numpy(y.astype("float32"))

        return x_dyn, x_stat, y

# -----------------------------
# Multi-sequence dataset (all MMSI+segments)
# -----------------------------
class MultiSequenceAISDataset(Dataset):
    """
    Dataset that samples sliding windows from *multiple* sequences.

    sequences: list of dicts from build_sequences_from_parquet, each:
        { "mmsi": ..., "segment": ..., "df": prepared_dataframe }

    Normalization (mean/std) is computed across *all* sequences together.
    """

    def __init__(
        self,
        sequences,
        lookback: int = 64,
        horizon: int = 1,
        target_cols: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.sequences = sequences
        self.lookback = lookback
        self.horizon = horizon

        self.target_cols = np.array(
            target_cols if target_cols is not None else np.arange(len(DYN_FEATURES))
        )

        # 1) Collect all dynamic + width/length data to compute global mean/std
        dyn_list = []
        wl_list = []
        for seq in sequences:
            df = seq["df"].reset_index(drop=True)
            dyn_list.append(df[DYN_FEATURES].to_numpy(np.float32))            # (Ni,6)
            wl_list.append(df[["Width", "Length"]].to_numpy(np.float32))      # (Ni,2)

        all_dyn = np.concatenate(dyn_list, axis=0) if dyn_list else np.zeros((0, len(DYN_FEATURES)), dtype=np.float32)
        all_wl = np.concatenate(wl_list, axis=0) if wl_list else np.zeros((0, 2), dtype=np.float32)

        if normalize and len(all_dyn) > 0:
            dyn_mean = all_dyn.mean(axis=0)
            dyn_std = all_dyn.std(axis=0)
            wl_mean = all_wl.mean(axis=0)
            wl_std = all_wl.std(axis=0)
        else:
            dyn_mean = np.zeros(len(DYN_FEATURES), dtype=np.float32)
            dyn_std = np.ones(len(DYN_FEATURES), dtype=np.float32)
            wl_mean = np.zeros(2, dtype=np.float32)
            wl_std = np.ones(2, dtype=np.float32)

        dyn_std = np.where(dyn_std < 1e-6, 1.0, dyn_std).astype("float32")
        wl_std = np.where(wl_std < 1e-6, 1.0, wl_std).astype("float32")

        self.dyn_mean, self.dyn_std = dyn_mean, dyn_std
        self.wl_mean, self.wl_std = wl_mean, wl_std

        # 2) Normalize each sequence separately & build index of (seq_idx, start)
        self.seq_dyn = []        # list of (Ni,6)
        self.seq_ship_type = []  # list of (Ni,)
        self.seq_wl = []         # list of (Ni,2)
        self.index = []          # list of (seq_idx, start)

        for seq_idx, seq in enumerate(sequences):
            df = seq["df"].reset_index(drop=True)

            dyn = df[DYN_FEATURES].to_numpy(np.float32)
            ship_type = df["Ship type_id"].to_numpy(np.int64)
            wl = df[["Width", "Length"]].to_numpy(np.float32)

            dyn_norm = (dyn - dyn_mean) / dyn_std
            wl_norm = (wl - wl_mean) / wl_std

            self.seq_dyn.append(dyn_norm)
            self.seq_ship_type.append(ship_type)
            self.seq_wl.append(wl_norm)

            N = len(dyn_norm)
            max_start = N - (lookback + horizon)
            if max_start < 0:
                continue

            for start in range(max_start + 1):
                self.index.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        seq_idx, start = self.index[idx]
        dyn = self.seq_dyn[seq_idx]
        ship_type = self.seq_ship_type[seq_idx]
        wl = self.seq_wl[seq_idx]

        s = start
        e = s + self.lookback
        t = e + self.horizon - 1

        x_dyn = dyn[s:e]  # (L,6)
        ship_id = ship_type[s]
        wl_row = wl[s]    # (2,)

        y = dyn[t, self.target_cols]  # (T,)

        x_dyn = torch.from_numpy(x_dyn)  # (L,6)
        x_stat = torch.tensor(
            np.concatenate([[ship_id], wl_row.astype("float32")]),
            dtype=torch.float32,
        )  # (3,)
        y = torch.from_numpy(y.astype("float32"))

        return x_dyn, x_stat, y


# -----------------------------
# Main: load parquet & inspect
# -----------------------------
def main():
    path = Path(PARQUET_PATH)
    print(f"Loading parquet: {path}")
    df_raw = pd.read_parquet(path)
    print(f"Loaded {len(df_raw)} rows.")
    print("Columns:", df_raw.columns.tolist())
    print("\nHead:\n", df_raw.head())

    sequences = build_sequences_from_parquet(df_raw, min_len=100)
    print(f"\nFound {len(sequences)} sequences with length >= 100 (grouped by MMSI + Segment).")

    if not sequences:
        print("No sequences long enough – try lowering min_len in build_sequences_from_parquet.")
        return

    # Some basic stats
    lengths = [len(seq["df"]) for seq in sequences]
    print(f"Min sequence length: {min(lengths)}")
    print(f"Max sequence length: {max(lengths)}")
    print(f"Mean sequence length: {sum(lengths)/len(lengths):.1f}")

    # Look at the first sequence
    seq0 = sequences[0]
    print("\nFirst sequence info:")
    print("  MMSI:   ", seq0["mmsi"])
    print("  Segment:", seq0["segment"])
    print("  Length: ", len(seq0["df"]))
    print("\nFirst sequence head:\n", seq0["df"].head())
    print("\nDtypes:\n", seq0["df"].dtypes)

    # Build a dataset + dataloader for this first sequence
    lookback = 64
    horizon = 1
    ds = AISDataset(seq0["df"], lookback=lookback, horizon=horizon, normalize=True)
    dl = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)

    x_dyn, x_stat, y = next(iter(dl))
    print("\nSample batch shapes:")
    print("  x_dyn:  ", x_dyn.shape, "  # (batch, lookback, 6 dynamic features)")
    print("  x_stat: ", x_stat.shape, "  # (batch, 3) [ship_type_id, width_norm, length_norm]")
    print("  y:      ", y.shape, "  # (batch, target_dim = number of predicted dyn features)")


if __name__ == "__main__":
    main()
