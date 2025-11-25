from google_drive_fetcher import GoogleDriveFetcher 
import torch
import pandas as pd
import numpy as np
TRAIN_URL = "https://drive.google.com/drive/folders/1Jr95Jgu4uqEKNrYAILOu2cLVHeEKsW7f?hl=da"

import time
class BatchStream: 
    def __init__(self, batch_size, port_encoder):
        self.train_url = TRAIN_URL
        self.gdf = GoogleDriveFetcher(URL = TRAIN_URL)
        self.items = self.gdf.list_directory(self.train_url)   
        self.batch_size = batch_size
        self.batch_X_list= []
        self.batch_Y_list = []
        self.port_encoder = port_encoder

    def stream_data(self): 
        count = 0
        count_total = len(self.items)
        while self.has_items_left(): 
            time.sleep(1)
            print("Streaming file ", count+1, " of ", count_total)
            count += 1
            df = self.stream_next_df() 
            self.save_baches(df=df)  
        self.create_batch()

    def create_batch(self):
        self.batch_Y_list = torch.tensor([torch.tensor(np.argmax(y)) for y in self.batch_Y_list])


    def stream_next(self) -> tuple[list[torch.Tensor], list[str]]:
        df = self.stream_next_df() 
        self.save_baches(df=df)
        if (self.should_stream_next()): 
            print("Streaming next file")
            self.stream_next() 
        else: 
            batch_X = self.batch_X_list[0: self.batch_size ]
            batch_Y = self.batch_Y_list[0: self.batch_size ] 
            self.batch_X_list = self.batch_X_list[self.batch_size : ]
            self.batch_Y_list = self.batch_Y_list[self.batch_size : ] 
            return batch_X, batch_Y
        

    def should_stream_next(self): 
        if (len(self.batch_X_list) < self.batch_size) or (len(self.batch_Y_list) < self.batch_size): 
            return True 
        return False

            


    def stream_next_df(self) -> pd.DataFrame:
        item = self.get_next_item()
        df = self.item_to_df(item)
        return df

    def item_to_df(self, item: dict) -> pd.DataFrame: 
        url = "https://drive.google.com/drive/folders/" + item['id'] + "?hl=da"
        df = self.gdf.fetch_parquet_to_dataframe(url)
        return df 

    
    def get_next_item(self) -> dict:
        return self.items.pop(0)
    
    def save_baches(self, df): 
        batch_X = self.extract_batch_list(series = df["input_segment"]) 
        batch_Y = self.extract_batch_list_port(series_port = df["Port"])
        self.batch_X_list.extend(batch_X)
        self.batch_Y_list.extend(batch_Y)

    def extract_batch_list_port(self, series_port: pd.Series): 
        port_list = [] 

        for port in series_port.values:
            port_arr = np.zeros(len(self.port_encoder))
            id = self.port_encoder[port]
            port_arr[id] = 1  
            port_list.append(port_arr) 

        return port_list

    
    def extract_batch_list(self, series: pd.Series):
        batch_list = [self.convert_array_to_tensor(np.array(arr)) for arr in series.values] 
        return batch_list

        


    def convert_array_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        t = torch.tensor(np.stack(array), dtype=torch.float32)
        return t

    

    def has_items_left(self) -> bool:
        return len(self.items) > 0 