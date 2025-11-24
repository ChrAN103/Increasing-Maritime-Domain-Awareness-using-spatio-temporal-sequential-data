from typing import Optional, Sequence, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DYN_FEATURES = ["Timestamp", "Latitude", "Longitude", "SOG", "COG_sin", "COG_cos"]
STATIC_FEATURES = ["Ship type", "Width", "Length"]


def prep_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # all relevant coloums, removes the rest
    cols = ["Timestamp", "Latitude", "Longitude", "SOG", "COG", "Ship type", "Width", "Length"]
    df = df[cols].dropna().reset_index(drop=True)

    # changes timestamp to datetime and sorts them accordenly
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # changes timestamp to time since first timestamp
    t0 = df["Timestamp"].iloc[0]
    df["Timestamp"] = (df["Timestamp"] - t0).dt.total_seconds().astype("float32")

    # converts COG degrees to COG sin and cos for more smooth transition
    cog_rad = np.deg2rad(df["COG"].astype("float32"))
    df["COG_sin"] = np.sin(cog_rad).astype("float32")
    df["COG_cos"] = np.cos(cog_rad).astype("float32")
    df = df.drop(columns=["COG"])

    # changes colounms to floats
    for c in ["Latitude", "Longitude", "SOG", "Width", "Length"]:
        df[c] = df[c].astype("float32")

    # gives every ship type an unique id
    col_str = df["Ship type"].astype("string")
    types = sorted(col_str.dropna().unique())
    ship_map = {v: i for i, v in enumerate(types)}
    df["Ship type_id"] = col_str.map(ship_map).astype("int64")

    return df


def build_sequences_from_parquet(
    # builds dataframe around segments which is at least 100 long
    df_raw: pd.DataFrame, min_len: int = 100
) -> List[Dict[str, Any]]:
    
    # cuts rest of data, which is not relevant
    needed = [
        "Timestamp", "Latitude", "Longitude", "SOG", "COG",
        "Ship type", "Width", "Length", "MMSI", "Segment"
    ]
    df_raw = df_raw.dropna(subset=needed)

    # makes new list
    sequences: List[Dict[str, Any]] = []

    # groups by mmsi and segments
    for (mmsi, seg), df_group in df_raw.groupby(["MMSI", "Segment"]):
        # sorts by timestamp
        df_prep = prep_dataframe(df_group.sort_values("Timestamp"))
        # only uses segments longer than min_len
        if len(df_prep) < min_len:
            continue
        # appends relevant data which was cut in prep_dataframe
        sequences.append({"mmsi": mmsi, "segment": seg, "df": df_prep})

    return sequences


class MultiSequenceAISDataset(Dataset):
    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        lookback: int = 64,
        horizon: int = 1,
        target_cols: Optional[Sequence[int]] = None,
        normalize: bool = True,
        fit_stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
        mmsi_to_id: Optional[dict] = None,
    ):
        # parameters
        super().__init__()
        self.sequences = sequences
        self.lookback = lookback
        self.horizon = horizon

        self.target_cols = np.array(
            target_cols if target_cols is not None else np.arange(len(DYN_FEATURES))
        )

        # getting dynamic and static features
        dyn_list = []
        wl_list = []
        for seq in sequences:
            df = seq["df"].reset_index(drop=True)
            dyn_list.append(df[DYN_FEATURES].to_numpy(np.float32))
            wl_list.append(df[["Width", "Length"]].to_numpy(np.float32))

        all_dyn = (
            np.concatenate(dyn_list, axis=0)
            if dyn_list
            else np.zeros((0, len(DYN_FEATURES)), dtype=np.float32)
        )
        all_wl = (
            np.concatenate(wl_list, axis=0)
            if wl_list
            else np.zeros((0, 2), dtype=np.float32)
        )

        # finding mean and standard deviation
        if fit_stats is None:
            if normalize and len(all_dyn) > 0:
                dyn_mean, dyn_std = all_dyn.mean(0), all_dyn.std(0)
                wl_mean, wl_std = all_wl.mean(0), all_wl.std(0)
            else:
                dyn_mean = np.zeros(len(DYN_FEATURES), np.float32)
                dyn_std = np.ones(len(DYN_FEATURES), np.float32)
                wl_mean = np.zeros(2, np.float32)
                wl_std = np.ones(2, np.float32)
        else:
            dyn_mean, dyn_std, wl_mean, wl_std = fit_stats

        # prevents devision with 0
        dyn_std = np.where(dyn_std < 1e-6, 1.0, dyn_std).astype("float32")
        wl_std = np.where(wl_std < 1e-6, 1.0, wl_std).astype("float32")

        self.dyn_mean, self.dyn_std = dyn_mean, dyn_std
        self.wl_mean, self.wl_std = wl_mean, wl_std

        # defining storage
        self.mmsi_to_id = mmsi_to_id or {}

        self.seq_dyn: List[np.ndarray] = []
        self.seq_ship_type: List[np.ndarray] = []
        self.seq_wl: List[np.ndarray] = []
        self.seq_mmsi_id: List[int] = []
        self.index: List[Tuple[int, int]] = []


        for seq in sequences:
            df = seq["df"].reset_index(drop=True)
            dyn = df[DYN_FEATURES].to_numpy(np.float32)
            ship_type = df["Ship type_id"].to_numpy(np.int64)
            wl = df[["Width", "Length"]].to_numpy(np.float32)

            # normalization
            dyn_norm = (dyn - dyn_mean) / dyn_std
            wl_norm = (wl - wl_mean) / wl_std

            N = len(dyn_norm)
            max_start = N - (lookback + horizon)
            if max_start < 0:
                continue

            self.seq_dyn.append(dyn_norm)
            self.seq_ship_type.append(ship_type)
            self.seq_wl.append(wl_norm)

            mmsi = seq["mmsi"]
            if mmsi not in self.mmsi_to_id:
                self.mmsi_to_id[mmsi] = len(self.mmsi_to_id)
            self.seq_mmsi_id.append(self.mmsi_to_id[mmsi])

            seq_idx = len(self.seq_dyn) - 1
            self.index.extend((seq_idx, s) for s in range(max_start + 1))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        seq_idx, start = self.index[idx]
        dyn = self.seq_dyn[seq_idx]
        ship_type = self.seq_ship_type[seq_idx]
        wl = self.seq_wl[seq_idx]
        mmsi_id = self.seq_mmsi_id[seq_idx]

        s = start
        e = s + self.lookback
        t = e + self.horizon - 1

        x_dyn = dyn[s:e]                
        ship_id = ship_type[s]
        wl_row = wl[s]                  

        # deltas for [Lat, Lon, SOG, COG_sin, COG_cos] (indices 1:6 in DYN_FEATURES)
        cur = dyn[t, 1:6]
        prev = dyn[t - 1, 1:6]
        y = (cur - prev).astype(np.float32)

        x_dyn = torch.from_numpy(x_dyn)
        x_stat = torch.tensor(
            [ship_id, mmsi_id, wl_row[0], wl_row[1]], dtype=torch.float32
        )
        y = torch.from_numpy(y)

        return x_dyn, x_stat, y
    
