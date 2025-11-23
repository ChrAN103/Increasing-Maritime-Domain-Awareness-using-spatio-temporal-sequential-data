import pandas as pd
from shapely import wkt
import geopandas as gpd
import hashlib
import numpy as np
import pyarrow
import pyarrow.parquet

class DataProcessor:
    def __init__(self):
        self.df = pd.DataFrame()
        self.counter = 0

        locodes = pd.read_csv("port_locodes.csv", sep=";")

        # Convert coordinates to valid WKT POLYGON string
        def to_wkt_polygon(coord_string):
            # Wrap coordinates in POLYGON(( ... ))
            return f"POLYGON(({coord_string}))"

        locodes['WKT'] = locodes['POLYGON'].apply(to_wkt_polygon)

        # Now load as Shapely geometry
        locodes['geometry'] = locodes['WKT'].apply(wkt.loads)

        # Convert to GeoDataFrame
        ports_gdf = gpd.GeoDataFrame(locodes, geometry='geometry', crs="EPSG:4326")

        # Filter Scandinavian ports (Denmark, Norway, Sweden)
        scandi_ports = ports_gdf[ports_gdf['LOCODE'].str[:2].isin(['DK','NO','SE','DE', 'PL'])]

        self.scandi_ports = scandi_ports

    def _overhaul_segments(self, df, time_threshold_minutes=7.5):
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
        df.drop(columns=['Segment'], inplace=True)
        df.rename(columns={"Segment_ID": "Segment"}, inplace=True)

        return df
    
    def _add_destination_port(self, df):
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
            matching_port = self.scandi_ports[self.scandi_ports.contains(last_point)]

            if not matching_port.empty:
                port_locode = matching_port.iloc[0]['LOCODE']
                df_gdf.loc[group.index, 'Port'] = port_locode

        # Optional: drop geometry if you want plain DataFrame
        return pd.DataFrame(df_gdf.drop(columns='geometry'))
    
    def _relative_time(self, df):
        # Calculate relative time in seconds from the start of each segment
        df['RelativeTime'] = df.groupby(['MMSI', 'Segment'])['Timestamp'].transform(lambda x: (x - x.min()).dt.total_seconds())
        return df
    
    def _check_date(self, df, new_date_str):
        """
        Splits the DataFrame based on SEGMENT completion.
        
        Returns:
            processed_output: Segments that finished STRICTLY BEFORE new_date.
            (implicitly updates self.df to hold segments that touch new_date).
        """
        # Convert string date to a Timestamp at midnight
        midnight_threshold = pd.to_datetime(new_date_str)

        # 1. Determine the Max Timestamp for every (MMSI, Segment) pair
        # transform('max') broadcasts the max value to every row in that segment
        segment_max_times = df.groupby(['MMSI', 'Segment'])['Timestamp'].transform('max')

        # 2. Create a mask for segments that are "Finished"
        # A segment is finished if its very last point is BEFORE the new day started.
        is_finished_segment = segment_max_times < midnight_threshold

        # 3. Split the data
        # Output: Segments that are completely in the past
        processed_output = df[is_finished_segment].copy()

        # Buffer: Segments that touch the new day (or are fully within the new day)
        self.df = df[~is_finished_segment].reset_index(drop=True)

        return processed_output

    def add_df(self, df):
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.counter += 1
        if self.counter == 10:
            df_date = str(df.iloc[0]['Timestamp'].date())
            df = self._overhaul_segments(self.df)
            df = self._add_destination_port(df)
            df = self._relative_time(df)
            df = self._check_date(df, df_date)
            self.train_validation_split(df)
            self.counter = 0
    
    def flush_df(self):
        if not self.df.empty:
            df = self._overhaul_segments(self.df)
            df = self._add_destination_port(df)
            df = self._relative_time(df)
            self.train_validation_split(df)
            self.df = pd.DataFrame()
            self.counter = 0

    def train_validation_split(self, df, min_prediction_horizon_minutes: float = 15, max_prediction_horizon_minutes: float = 120.0, permutations: int = 5, validation_size: float = 0.2):
        """
        Split data into train/validation with last X hours removed for prediction task
        
        Args:
            df: DataFrame loaded from load_data()
            min_prediction_horizon_minutes: Minimum minutes to remove from end of routes
            max_prediction_horizon_minutes: Maximum minutes to remove from end of routes
            time_permutations: Number of different time cutoffs to create per segment
            validation_size: Proportion of segments to use for validationing (0.0 to 1.0)
        
        Returns:
            train_df, validation_df: DataFrames ready for model training
        """

        train_segments = []
        validation_segments = []
        
        # Calculate threshold for validation split (e.g., 20% -> hash % 5 == 0)
        validation_threshold = int(1 / validation_size)
        
        total_segments = 0
        kept_segments = 0
        
        for (mmsi, segment), group in df.groupby(['MMSI', 'Segment']):
            total_segments += 1
            
            # Ensure Timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(group['Timestamp']):
                group['Timestamp'] = pd.to_datetime(group['Timestamp'])
            
            time_permutations = 1 if group['Port'].iloc[0] == 'no_port' else permutations  # Only one permutation for no_port segments
            
            edges = np.linspace(min_prediction_horizon_minutes, max_prediction_horizon_minutes, num=time_permutations+1)
            random_minutes = np.random.uniform(edges[:-1], edges[1:], size=time_permutations)
            max_time = group['Timestamp'].max()
            routes = []
            for idx, prediction_horizon_minutes in enumerate(random_minutes):
                sample_id = f"{mmsi}_{segment}_{idx}"
                cutoff_time = max_time - pd.Timedelta(minutes=prediction_horizon_minutes)
                # Remove last X hours
                # If max_time is 14:00 and we want to remove 2 hours, cutoff is 12:00
                # We keep all timestamps < 12:00 (the early part of the route)
                input_route = group[group['Timestamp'] < cutoff_time].copy()
                input_route['is_target'] = False
                target_route = group[group['Timestamp'] >= cutoff_time].copy()
                target_route['is_target'] = True
                if len(input_route) > 256:  # Ensure minimum track length after cutting
                    # Preserve destination label from original route
                    input_route.loc[:, 'Port'] = group['Port'].iloc[0]
                    target_route.loc[:, 'Port'] = group['Port'].iloc[0]
                    partial_route = pd.concat([input_route, target_route], ignore_index=True)
                    partial_route['SampleID'] = sample_id
                    kept_segments += 1
                    routes.append(partial_route)
            # Deterministic hash-based split (same MMSI+Segment always goes to same set)
            hash_val = int(hashlib.md5(f"{mmsi}_{segment}".encode()).hexdigest(), 16)
            if hash_val % validation_threshold == 0:
                for partial_route in routes:
                    validation_segments.append(partial_route)
            else:
                for partial_route in routes:
                    train_segments.append(partial_route)
        
        print(f"Total segments: {total_segments}")
        # print(f"Segments after {prediction_horizon_minutes}h cutoff (len > 256): {kept_segments}")
        print(f"Train segments: {len(train_segments)}")
        print(f"validation segments: {len(validation_segments)}")
        
        if not train_segments or not validation_segments:
            return ValueError("Train or validation set is empty! Adjust prediction_horizon_hours or check data.")
        
        train_df = pd.concat(train_segments, ignore_index=True)
        validation_df = pd.concat(validation_segments, ignore_index=True)

        train_table = pyarrow.Table.from_pandas(train_df, preserve_index=False)
        validation_table = pyarrow.Table.from_pandas(validation_df, preserve_index=False)
        pyarrow.parquet.write_to_dataset(
            table=train_table,
            root_path="../data/train_set",
        )
        pyarrow.parquet.write_to_dataset(
            table=validation_table,
            root_path="../data/validation_set",
        )