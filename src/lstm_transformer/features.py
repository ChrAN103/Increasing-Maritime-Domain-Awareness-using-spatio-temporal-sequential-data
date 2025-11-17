import torch 
import pandas as pd

class FeatureEngineering: 
    def __init__(self):
        pass

    def extract_features(self, data):
        # Use iloc for positional indexing on DataFrame
        feature = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32) 
        datetime = data.iloc[:, 0]
        
        # Extract datetime features
        datetime_features = torch.tensor(
            [[dt.hour, dt.day, dt.month, dt.year] for dt in datetime], 
            dtype=torch.float32
        )
        
        # Concatenate features with datetime components
        feature = torch.cat([feature, datetime_features], dim=1)

        return feature 