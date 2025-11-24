from google_drive_fetcher import GoogleDriveFetcher 
import torch
import pandas as pd
import numpy as np
TRAIN_URL = "https://drive.google.com/drive/folders/1Jr95Jgu4uqEKNrYAILOu2cLVHeEKsW7f?hl=da"


class BatchStream: 
    def __init__(self):
        self.train_url = TRAIN_URL
        self.gdf = GoogleDriveFetcher(URL = TRAIN_URL)
        self.items = self.gdf.list_directory(self.train_url)   

    def stream_next(self) -> tuple[list[torch.Tensor], list[str]]:
        df = self.stream_next_df() 
        batch_X = [self.convert_array_to_tensor(np.array(arr)) for arr in df.iloc[:, 0].values]
        batch_Y = [port for port in df["Port"].values]
        return batch_X, batch_Y


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

    def convert_array_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        t = torch.tensor(np.stack(array), dtype=torch.float32)
        return t

    

    def has_items_left(self) -> bool:
        return len(self.items) > 0  