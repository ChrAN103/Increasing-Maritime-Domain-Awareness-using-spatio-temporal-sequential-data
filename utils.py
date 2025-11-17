import os
from Dataloader import *
import folium
import zipfile
import requests

import pandas as pd
import pyarrow
import pyarrow.parquet
import matplotlib.pyplot as plt

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