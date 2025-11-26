import os
from Dataloader import *
import folium
import zipfile
import requests

import pandas as pd
import pyarrow
import pyarrow.parquet
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import geopandas as gpd


def download(url, filepath):
    """Simple download with progress tracking"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                
                # Progress bar
                if total_size > 0:
                    percent = (downloaded_size / total_size) * 100
                    mb_downloaded = downloaded_size / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r📥 Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
    
    print()
    print(f"✅ Download complete: {filepath}")


# We are encountering that a ships sailing over midnight gets split into two segments, even though it's the same journey.
# We create a function to overhaul the segment IDs accordingly.
# This is happening because in the data cleaning function each day is handled separately, causing segments to reset at midnight.
def overhaul_segments(df, time_threshold_minutes=7.5):
    """
    Overhauls segment IDs based on a time threshold between consecutive points.
    A new segment is created for a ship if the time gap between two of its
    data points is greater than the specified threshold.

    input:
        df: DataFrame with 'MMSI' and 'Timestamp' columns.
        time_threshold_minutes: The maximum time gap in minutes before a new segment is started.
    output:
        df: DataFrame with a new 'Segment_ID' column.
    """
    # Data is sorted correctly
    df = df.sort_values(by=['MMSI', 'Timestamp']).reset_index(drop=True)

    # Calculate the time difference between consecutive points for each ship
    time_diffs = df.groupby('MMSI')['Timestamp'].diff()

    # A new segment starts if the time difference is larger than our threshold
    # or if it's the first point for a new ship (where time_diff is NaT)
    new_segment_starts = (time_diffs > pd.Timedelta(minutes=time_threshold_minutes)) | (time_diffs.isna())

    # Use cumsum() to create a unique, incrementing ID for each segment
    df['Segment_ID'] = new_segment_starts.cumsum()

    # To make segment IDs restart from 0 for each ship, we can do:
    df['Segment_ID'] = df.groupby('MMSI')['Segment_ID'].transform(lambda x: x - x.min())
    
    return df

def plot_trajectory_on_map(df, percentage_of_vessels=0.5):
    """
    Plots the trajectory of a vessel on a map, respecting the new segment IDs.
    """

    # Visualize every vessel's trajectory on a map - RESPECTING SEGMENTS
    m = folium.Map(location=[55.6761, 12.5683], zoom_start=6)

    # Sample the vessels to plot
    vessel_ids = df["MMSI"].unique()
    sample_size = int(percentage_of_vessels * len(vessel_ids))
    vessels_to_plot = vessel_ids[:sample_size]

    # Plot each vessel's trajectory, with separate polylines for each segment
    for vessel_id in vessels_to_plot:
        vessel_data = df[df["MMSI"] == vessel_id]
        
        # Group only by the corrected 'Segment' column
        for segment_id, segment in vessel_data.groupby('Segment'):
            # Sort by timestamp within the segment
            segment = segment.sort_values('Timestamp')
            
            # Extract coordinates
            points = list(zip(segment['Latitude'], segment['Longitude']))
            
            # Only plot if we have at least 2 points to make a line
            if len(points) >= 2:
                folium.PolyLine(
                    locations=points,
                    color="blue",
                    weight=2,
                    opacity=0.6,
                    popup=f"Vessel {vessel_id}<br>Segment {segment_id}<br>{len(points)} points"
                ).add_to(m)
    return m

def add_destination_port(df, ports):
        # Convert AIS df to GeoDataFrame
        df_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
            crs="EPSG:4326"
        )

        # Initialize Port column as no_port
        df_gdf['Port'] = 'no_port'

        # Group by MMSI and Segment
        for (mmsi, segment), group in df_gdf.groupby(['MMSI', 'Segment']):
            # Take the last point of the segment
            last_point = group.iloc[-1].geometry

            # Check which port contains this point
            matching_port = ports[ports.contains(last_point)]

            if not matching_port.empty:
                port_locode = matching_port.iloc[0]['LOCODE']
                df_gdf.loc[group.index, 'Port'] = port_locode

        # Optional: drop geometry if you want plain DataFrame
        return pd.DataFrame(df_gdf.drop(columns='geometry'))

class MaritimeDataset(Dataset):
    def __init__(self, df, port_encoder=None, ship_type_encoder=None, feature_scaler=None, max_len=500):
        self.max_len = max_len
        
        # --- 1. ENCODING SETUP ---

        if port_encoder is None:

            self.port_encoder = LabelEncoder()

            self.port_encoder.fit(df['Port'].astype(str).unique())

        else:

            self.port_encoder = port_encoder


        # --- Verification ---
        # After this fit, self.port_encoder.transform(['no_port']) will always be 0
        if self.port_encoder.transform(['no_port'])[0] == 0:
            self.no_port_id = 0
        else:
            # Fallback (shouldn't happen with the code above)
            self.no_port_id = self.port_encoder.transform(['no_port'])[0]
            
        print(f"The 'no_port' ID is successfully set to: {self.no_port_id}")
            
        if ship_type_encoder is None:
            self.ship_type_encoder = LabelEncoder()
            # Handle NaNs before fitting
            unique_types = df['Ship type'].fillna('Undefined').astype(str).unique()
            self.ship_type_encoder.fit(unique_types)
        else:
            self.ship_type_encoder = ship_type_encoder

        # Work on copy
        df_clean = df.copy()

        # --- 2. VECTORIZED PRE-PROCESSING (Fast) ---
        
        # Filter unknown destinations efficiently BEFORE the loop
        # This removes the need for the 'if' check inside the loop
        df_clean = df_clean[df_clean['Port'].isin(self.port_encoder.classes_)]
        
        # Pre-calculate Target Indices
        df_clean['Target_Idx'] = self.port_encoder.transform(df_clean['Port'])

        # Cyclical Encoding
        cog_rad = np.deg2rad(df_clean['COG'].fillna(0))
        df_clean['sin_COG'] = np.sin(cog_rad)
        df_clean['cos_COG'] = np.cos(cog_rad)
        
        # Ship Type Encoding
        s_types = df_clean['Ship type'].fillna('Undefined').astype(str)
        known_types = set(self.ship_type_encoder.classes_)
        # Map unknown types to the first known class (safe fallback)
        fallback_class = list(known_types)[0]
        s_types = s_types.apply(lambda x: x if x in known_types else fallback_class)
        df_clean['Ship type'] = self.ship_type_encoder.transform(s_types)

        # Scaling
        self.cols_to_scale = ['Latitude', 'Longitude', 'SOG']
        self.cols_no_scale = ['sin_COG', 'cos_COG', 'Log_RelativeTime']
        self.feature_cols = self.cols_to_scale + self.cols_no_scale

        # Fill NaNs
        for col in self.feature_cols:
            if col not in df_clean.columns: df_clean[col] = 0
            df_clean[col] = df_clean[col].fillna(0)

        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(df_clean[self.cols_to_scale].values)
        else:
            self.feature_scaler = feature_scaler

        df_clean[self.cols_to_scale] = self.feature_scaler.transform(df_clean[self.cols_to_scale].values)

        # --- 3. SEQUENCE CREATION ---
        self.sequences = []
        self.targets = []
        
        print("Grouping data into sequences...")
        
        # GroupBy is the last remaining slow part, but necessary for structure
        grouped = df_clean.groupby(['MMSI', 'Segment'])
        
        for _, group in grouped:
            # Sort by timestamp
            group = group.sort_values('Timestamp')
            group['Log_RelativeTime'] = np.log((group['Timestamp'] - group['Timestamp'].min()).dt.total_seconds() + 1)
            
            feats = group[self.feature_cols].values
            
            # Fast check for bad data
            if np.isnan(feats).any() or np.isinf(feats).any():
                continue
                
            # Truncate
            if len(feats) > self.max_len:
                feats = feats[-self.max_len:]
                
            # Convert to Tensor IMMEDIATELY (Float32 for features)
            # clone().detach() is the safest way to avoid warnings
            seq_tensor = torch.tensor(feats, dtype=torch.float32)
            self.sequences.append(seq_tensor)
            
            # Append Target (Long for classification)
            # We already computed this in 'Target_Idx' column
            target_val = group['Target_Idx'].iloc[0]
            self.targets.append(target_val)
            
        # Convert targets list to a single tensor (faster than list of 0-d tensors)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        
        print(f"Created {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # Zero-cost lookup
        return self.sequences[idx], self.targets[idx]

def get_device():
    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        
    return device