import unittest 
from src.lstm_transformer.features import FeatureEngineering
from torch import Tensor
import pandas as pd

class TestFeaturesEngineering(unittest.TestCase):

    def setUp(self):
        self.feature_engineering = FeatureEngineering()
    
    def test_initialize(self):
        self.assertIsInstance(self.feature_engineering, FeatureEngineering) 

    def test_output(self):
        data = self.generate_sample_data() 
        features = self.feature_engineering.extract_features(data)  
        self.assertIsInstance(features, Tensor) 

    def generate_sample_data(self):
        data = {
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='h'),
            'value': range(100)
        }
        df = pd.DataFrame(data)
        return df


    

if __name__ == '__main__':   
    unittest.main()