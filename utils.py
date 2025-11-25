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

class MaritimeDataset(Dataset):
    def __init__(self, df, port_encoder=None, feature_scaler=None, max_len=500):
        """
        Args:
            df: DataFrame containing the ship trajectories
            port_encoder: Fitted LabelEncoder for destinations (optional)
            feature_scaler: Fitted StandardScaler for features (optional)
            max_len: Maximum sequence length to truncate to
        """
        self.max_len = max_len
        
        # 1. Encode Destinations (Targets)
        if port_encoder is None:
            self.port_encoder = LabelEncoder()
            # Fit on all unique destinations in the dataframe
            self.port_encoder.fit(df['Destination'].astype(str).unique())
        else:
            self.port_encoder = port_encoder
                   
        # We work on a copy to avoid modifying the original dataframe
        df_clean = df.copy()


        # --- CYCLICAL ENCODING FOR COG ---
        # Convert degrees to radians
        # Fill NaNs with 0 (North) before conversion to avoid issues (they should already have been removed earlier but just in case)
        cog_rad = np.deg2rad(df_clean['COG'].fillna(0))
        
        # Create new features
        df_clean['sin_COG'] = np.sin(cog_rad)
        df_clean['cos_COG'] = np.cos(cog_rad)
        
        # 2. Prepare Features
        # Select numerical features to use
        # We REMOVE 'COG' and ADD 'sin_COG', 'cos_COG' instead
        self.feature_cols = ['Latitude', 'Longitude', 'SOG', 'sin_COG', 'cos_COG', 'Length', 'Width']

        # Handle missing values for linear features by filling with 0 (just as a backup incase it previously slipped through)
        for col in self.feature_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0
            df_clean[col] = df_clean[col].fillna(0)

        # Normalize features
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            self.feature_scaler.fit(df_clean[self.feature_cols].values)
        else:
            self.feature_scaler = feature_scaler
            
        # 3. Group by Trajectory (MMSI + Segment)
        # We create a list of (features, target) tuples
        self.sequences = []
        self.targets = []
        
        print("Grouping data into sequences...")
        # Group by MMSI and Segment to get individual trips
        grouped = df_clean.groupby(['MMSI', 'Segment'])
        
        for _, group in grouped:
               
            # Sort by timestamp just in case
            group = group.sort_values('Timestamp')
            
            # Get features
            feats = group[self.feature_cols].values

            # Double check for any remaining NaNs or Infs that fucks with StandardScaler 
            if np.isnan(feats).any() or np.isinf(feats).any():
                continue
            
            feats = self.feature_scaler.transform(feats)
            
            # Truncate if too long (good for memory and our PC's will thank us)
            if len(feats) > self.max_len:
                feats = feats[-self.max_len:]
                
            # Get target (Destination should be the same for the whole segment)
            dest = str(group['Destination'].iloc[0])
            
            # Only add if destination is known/in encoder 
            if dest in self.port_encoder.classes_:
                target_idx = self.port_encoder.transform([dest])[0]
                
                self.sequences.append(torch.FloatTensor(feats))
                self.targets.append(target_idx)
                
        print(f"Created {len(self.sequences)} sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

