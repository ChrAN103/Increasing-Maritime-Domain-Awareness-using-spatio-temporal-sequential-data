
"""
Columns in *.csv file                   Format
----------------------------------------------------------------------------------------------------------------------------------------------------
1.    Timestamp                         Timestamp from the AIS basestation, format: 31/12/2015 23:59:59 

2.    Type of mobile                    Describes what type of target this message is received from (class A AIS Vessel, Class B AIS vessel, etc)

3.    MMSI                              MMSI number of vessel

4.    Latitude                          Latitude of message report (e.g. 57,8794)

5.    Longitude                         Longitude of message report (e.g. 17,9125)

6.    Navigational status               Navigational status from AIS message if available, e.g.: 'Engaged in fishing', 'Under way using engine', mv.

7.    ROT                               Rot of turn from AIS message if available

8.    SOG                               Speed over ground from AIS message if available

9.    COG                               Course over ground from AIS message if available

10.   Heading                           Heading from AIS message if available

11.   IMO                               IMO number of the vessel

12.   Callsign                          Callsign of the vessel 

13.   Name                              Name of the vessel

14.   Ship type                         Describes the AIS ship type of this vessel 

15.   Cargo type                        Type of cargo from the AIS message 

16.   Width                             Width of the vessel

17.   Length                            Lenght of the vessel 

18.   Type of position fixing device    Type of positional fixing device from the AIS message 

19.   Draught                           Draugth field from AIS message

20.   Destination                       Destination from AIS message

21.   ETA                               Estimated Time of Arrival, if available  

22.   Data source type                  Data source type, e.g. AIS

23.   Size A                            Length from GPS to the bow

24.   Size B                            Length from GPS to the stern

25.   Size C                            Length from GPS to starboard side

26.   Size D                            Length from GPS to port side


Chosen columns for processing:
1.    Timestamp
2.    Latitude
3.    Longitude
4.    MMSI
5.    SOG
6.    COG
7.    Ship type
8.    Length
9.    Width
10.   Type of mobile
11.   Destination
"""

import pandas as pd
import pyarrow
import pyarrow.parquet
from geopy.distance import geodesic
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
import zipfile
from shapely.geometry import MultiPoint
import numpy as np

class Dataloader:
    """A class for loading and preprocessing maritime spatio-temporal sequential data.
    It should get the data from a CSV file, clean it and convert it to a Parquet file for efficient storage and retrieval.
    Furthermore it should implement our scope of work in terms of data segmentation and feature engineering.
    """
    def __init__(self, file_path: str = None, out_path: str = None, zip_path: str = None, csv_internal_path: str = None):
        self.file_path = file_path
        self.out_path = out_path
        self.zip_path = zip_path
        self.csv_internal_path = csv_internal_path

    def pre_clean_data(self):
        dtypes = {
            "MMSI": "object",
            "SOG": float,
            "COG": float,
            "Longitude": float,
            "Latitude": float,
            "# Timestamp": "object",
            "Type of mobile": "object",
            "Ship type": "object",
            "Length": float,
            "Width": float,
        }
        usecols = list(dtypes.keys())

        # Read CSV: either from ZIP or from direct file path
        if self.zip_path and self.csv_internal_path:
            # Read from ZIP archive
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(self.csv_internal_path) as csv_file:
                    df = pd.read_csv(csv_file, usecols=usecols, dtype=dtypes)
        else:
            # Read from direct file path
            df = pd.read_csv(self.file_path, usecols=usecols, dtype=dtypes)
        
        # Remove errors
        bbox = [60, 0, 50, 20]
        north, west, south, east = bbox
        df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & 
                (df["Longitude"] >= west) & (df["Longitude"] <= east)]
        
        #keep Class A only (better data quality for commercial routes)
        df = df[df["Type of mobile"] == "Class A"]
        #keep tanker and cargo only
        df = df[(df["Ship type"].str.startswith("Tanker")) | (df["Ship type"].str.startswith("Cargo"))]
        df = df.rename(columns={"# Timestamp": "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")

        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        pyarrow.parquet.write_to_dataset(
            table,
            root_path=self.out_path,
            partition_cols=["MMSI"]
        )

    def clean_data(self):
        dtypes = {
            "MMSI": "object",
            "SOG": float,
            "COG": float,
            "Longitude": float,
            "Latitude": float,
            "# Timestamp": "object",
            "Type of mobile": "object",
            "Ship type": "object",
            "Length": float,
            "Width": float,
            "Destination": "object",
        }
        usecols = list(dtypes.keys())

        # Read CSV: either from ZIP or from direct file path
        if self.zip_path and self.csv_internal_path:
            # Read from ZIP archive
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                with zip_ref.open(self.csv_internal_path) as csv_file:
                    df = pd.read_csv(csv_file, usecols=usecols, dtype=dtypes)
        else:
            # Read from direct file path
            df = pd.read_csv(self.file_path, usecols=usecols, dtype=dtypes)

        # Remove errors
        bbox = [60, 0, 50, 20]
        north, west, south, east = bbox
        df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & 
                (df["Longitude"] >= west) & (df["Longitude"] <= east)]
    
        # Keep Class A only (better data quality for commercial routes)
        df = df[df["Type of mobile"] == "Class A"]
        # Remove the entire coloumn. All not class A ships are removed
        df = df.drop(columns=["Type of mobile"])

        df = df[df["MMSI"].str.len() == 9]
        df = df[df["MMSI"].str[:3].astype(int).between(200, 775)]
    
        df = df.rename(columns={"# Timestamp": "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    
        df = df.drop_duplicates(["Timestamp", "MMSI"], keep="first")
    
        def track_filter(g):
            len_filt = len(g) > 256
            sog_filt = 1 <= g["SOG"].max() <= 50
            time_filt = (g["Timestamp"].max() - g["Timestamp"].min()).total_seconds() >= 60 * 60
            return len_filt and sog_filt and time_filt
    
        df = df.groupby("MMSI").filter(track_filter)
        df = df.sort_values(['MMSI', 'Timestamp'])
    
        # Now we check the destinations if it exist in our port_locodes file:
        port_locodes = pd.read_csv("port_locodes.csv", sep=";")
        valid_destinations = set(port_locodes["LOCODE"].str.upper().str.strip())
        df = df[df["Destination"].isin(valid_destinations)]

        # # Segment route if time gap >= 7.5 minutes
        df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
            lambda x: (x.diff().dt.total_seconds().fillna(999999) >= 7.5 * 60).cumsum())
    
        df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
        df = df.reset_index(drop=True)
    
        knots_to_ms = 0.514444
        df["SOG"] = knots_to_ms * df["SOG"]
    
        # Double check: Clean destination field for non port labels
        if "Destination" in df.columns:
            df["Destination"] = df["Destination"].str.upper().str.strip()
            # Keep only non empty so we can actually use the route
            df = df[df["Destination"].notna() & (df["Destination"] != "") & (df["Destination"] != "UNKNOWN")]



        def detect_anomalies(df, max_speed_knots=50):
            """Flag physically impossible movements based on implied speed between consecutive points."""
            df = df.sort_values(['MMSI', 'Segment', 'Timestamp'])
            
            # Calculate distance between consecutive points
            def calc_distance_safe(group):
                coords = list(zip(group['Latitude'], group['Longitude']))
                distances = [0] + [geodesic(coords[i], coords[i+1]).km 
                                for i in range(len(coords)-1)]
                return pd.Series(distances, index=group.index)

            df['distance_km'] = df.groupby(['MMSI', 'Segment'], group_keys=False).apply(
                calc_distance_safe
            )
            # Calculate time difference
            df['time_diff_hours'] = df.groupby(['MMSI', 'Segment'])['Timestamp'].diff().dt.total_seconds() / 3600
            
            # Calculate implied speed
            df['implied_speed_knots'] = (df['distance_km'] / df['time_diff_hours']) / 1.852
            
            # Flag speed anomalies
            speed_anomaly = df['implied_speed_knots'] > max_speed_knots
            
            df['anomaly'] = speed_anomaly
            
            # Print statistics
            print(f"Found {speed_anomaly.sum()} speed anomalies (>{max_speed_knots} knots)")
            
            return df

        df = detect_anomalies(df, max_speed_knots=50)


        # Get unique (MMSI, Segment) combinations with anomalies
        anomalous_segments = df[df["anomaly"] == True][['MMSI', 'Segment']].drop_duplicates()

        print(f"Removing {len(anomalous_segments)} segments with anomalies...")

        # Remove ALL rows belonging to those segments
        df = df[~df.set_index(['MMSI', 'Segment']).index.isin(
            anomalous_segments.set_index(['MMSI', 'Segment']).index
        )]

        print(f"Remaining rows after anomaly removal: {len(df)}")

        # After removing anomalies, drop temporary columns
        df = df.drop(columns=['distance_km', 'time_diff_hours', 'implied_speed_knots', 'anomaly'])
        
        # To reduce size, we dont keep data for the same ship that are within 10 seconds of each other
        def downsample_group(g):
            # 1. Sort by time (ensure chronological order)
            g = g.sort_values("Timestamp")
            # 2. Calculate time difference between consecutive points
            time_diffs = g["Timestamp"].diff().dt.total_seconds().fillna(600)  # First point always kept
            # diff() computes time between row[i] and row[i-1]
            # fillna(600) sets first point's diff to 600 seconds (always kept)
            # 3. Keep only points that are >= 10 seconds apart
            mask = time_diffs >= 10 # 10 seconds in seconds
            # 4. Return filtered DataFrame
            return g[mask]

        df = df.groupby(["MMSI", "Segment"], group_keys=False).apply(downsample_group).reset_index(drop=True)

        df["Ship type"] = df["Ship type"].astype('category')  # ~10-20 unique ship types
        df["Destination"] = df["Destination"].astype('category')  # Limited set of ports
        df["MMSI"] = df["MMSI"].astype('category')  # Repeated vessel IDs

        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        pyarrow.parquet.write_to_dataset(
            table,
            root_path=self.out_path,
            partition_cols=["MMSI", "Segment"]
        )
        
    def load_data(self, date_folders: list = None):
        """
        Load processed Parquet data from one or multiple date folders
        
        Args:
            date_folders: List of folder names like ['aisdk-2023-01-15', 'aisdk-2023-03-15']
                         If None, loads all folders in out_path
        
        Returns:
            pd.DataFrame: Combined dataframe from all specified dates
        """
        import os
        
        if date_folders is None:
            # Load all folders in out_path
            date_folders = [f for f in os.listdir(self.out_path) 
                           if os.path.isdir(os.path.join(self.out_path, f))]
        
        dfs = []
        for folder in date_folders:
            folder_path = os.path.join(self.out_path, folder)
            if os.path.exists(folder_path):
                print(f"Loading {folder}...")
                dataset = pyarrow.parquet.ParquetDataset(folder_path)
                table = dataset.read()
                df = table.to_pandas()
                dfs.append(df)
            else:
                print(f"Warning: {folder_path} does not exist, skipping...")
        
        if not dfs:
            raise ValueError("No data loaded. Check your date_folders and out_path.")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df)} total records from {len(dfs)} date folder(s)")
        return combined_df
    
    def train_test_split(self, df: pd.DataFrame,validation_size: float = 0.15, test_size: float = 0.15):
        """
        Split data into train/test with last X hours removed for prediction task
        
        Args:
            df: DataFrame loaded from load_data()
            validation_size: Proportion of segments to use for validation (0.0 to 1.0)
            test_size: Proportion of segments to use for testing (0.0 to 1.0)
        
        Returns:
            train_df, validation_df, test_df: DataFrames ready for model training
        """
       
        train_segments = []
        validation_segments = []
        test_segments = []
        
        # Calculate threshold for test split (e.g., 20% -> hash % 5 == 0)
        test_cutoff = int(test_size * 100)
        val_cutoff = int((test_size + validation_size) * 100)
        
        total_segments = 0
        kept_segments = 0
        
        for (mmsi, segment), group in df.groupby(['MMSI', 'Segment']):
            total_segments += 1
            
            # Ensure Timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(group['Timestamp']):
                group['Timestamp'] = pd.to_datetime(group['Timestamp'])
            
            split_mod = kept_segments % 100
            test_cutoff = int(test_size * 100)
            val_cutoff = int((test_size + validation_size) * 100)
        
            is_test = split_mod < test_cutoff
            is_val = (split_mod >= test_cutoff) and (split_mod < val_cutoff)

            if is_test or is_val:
                horizons_minutes = [15, 30, 60, 120]
            else:
            # Train gets RANDOMIZED horizons for augmentation
                edges = np.linspace(15, 120, num=5+1)
                horizons_minutes = np.random.uniform(edges[:-1], edges[1:], size=5)
    
            max_time = group['Timestamp'].max()
            routes = []
                # CHANGED: Use counter-based split (Round Robin) instead of Hash
                # This ensures that even if we split an already-split dataset, we still get a valid test set.

            for idx, prediction_horizon_minutes in enumerate(horizons_minutes):
                short = False
                sample_id = f"{mmsi}_{segment}_{idx}"
                cutoff_time = max_time - pd.Timedelta(minutes=prediction_horizon_minutes)
                # Remove last X hours
                # If max_time is 14:00 and we want to remove 2 hours, cutoff is 12:00
                # We keep all timestamps < 12:00 (the early part of the route)
                partial_route = group[group['Timestamp'] < cutoff_time].copy()
                partial_route['Horizon_Min'] = prediction_horizon_minutes
                if len(partial_route) > 256:  # Ensure minimum track length after cutting
                    # Preserve destination label from original route
                    partial_route.loc[:, 'Port'] = group['Port'].iloc[0]
                    partial_route['SampleID'] = sample_id
                    routes.append(partial_route)
                else:
                    short = True
                    break
            if not short:
                kept_segments += 1
                if is_test:
                    test_segments.extend(routes)
                elif is_val:
                    validation_segments.extend(routes)
                else:
                    port_label = group['Port'].iloc[0] if 'Port' in group.columns else None
                    if port_label == "no_port":
                        # take a single of routes random
                        routes = [routes[np.random.randint(len(routes))]]
                    train_segments.extend(routes)
        
        print(f"Total segments: {total_segments}")
        print(f"Train segments: {len(train_segments)}")
        print(f"Test segments: {len(test_segments)}")
        
        if not train_segments or not test_segments:
            raise ValueError("Train or test set is empty! Adjust prediction_horizon_hours or check data.")

        train_df = pd.concat(train_segments, ignore_index=True)
        test_df = pd.concat(test_segments, ignore_index=True)
        validation_df = pd.concat(validation_segments, ignore_index=True)
        
        return train_df, validation_df, test_df