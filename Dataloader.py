
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

class Dataloader:
    """A class for loading and preprocessing maritime spatio-temporal sequential data.
    It should get the data from a CSV file, clean it and convert it to a Parquet file for efficient storage and retrieval.
    Furthermore it should implement our scope of work in terms of data segmentation and feature engineering.
    """
    def __init__(self, file_path: str, out_path: str):
        self.file_path = file_path
        self.out_path = out_path

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
        df = pd.read_csv(self.file_path, usecols=usecols, dtype=dtypes)

        # Remove errors
        bbox = [60, 0, 50, 20]
        north, west, south, east = bbox
        df = df[(df["Latitude"] <= north) & (df["Latitude"] >= south) & 
                (df["Longitude"] >= west) & (df["Longitude"] <= east)]
    
        # CHANGED: Keep the Type of mobile column, just filter it
        # df = df[df["Type of mobile"].isin(["Class A", "Class B"])]
        # Don't drop it: .drop(columns=["Type of mobile"])

        # Keep Class A only (better data quality for commercial routes)
        df = df[df["Type of mobile"] == "Class A"]

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
    
        # Change: segment route if time gap >= 30 minutes
        df['Segment'] = df.groupby('MMSI')['Timestamp'].transform(
            lambda x: (x.diff().dt.total_seconds().fillna(0) >= 30 * 60).cumsum())
    
        df = df.groupby(["MMSI", "Segment"]).filter(track_filter)
        df = df.reset_index(drop=True)
    
        knots_to_ms = 0.514444
        df["SOG"] = knots_to_ms * df["SOG"]
    
        # ADD: Clean destination field for port labels
        if "Destination" in df.columns:
            df["Destination"] = df["Destination"].str.upper().str.strip()
            # Keep only non empty so we can actually use the route
            df = df[df["Destination"].notna() & (df["Destination"] != "") & (df["Destination"] != "UNKNOWN")]

        # Now we check the destinations if it exist in our port_locodes file:
        port_locodes = pd.read_csv("../port_locodes.csv", sep=";")
        valid_destinations = set(port_locodes["LOCODE"].str.upper().str.strip())
        df = df[df["Destination"].isin(valid_destinations)]

        table = pyarrow.Table.from_pandas(df, preserve_index=False)
        pyarrow.parquet.write_to_dataset(
            table,
            root_path=self.out_path,
            partition_cols=["MMSI", "Segment"]
        )