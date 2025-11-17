import unittest 
from src.lstm_transformer.features import FeatureEngineering
from torch import Tensor
import pandas as pd
from src.lstm_transformer.lstm_encoder import LSTMPositionalEncoding 
from utils import generate_sample_data

class TestLSTMPositionalEncoding(unittest.TestCase):

    def setUp(self):
        #Generate sample data
        start_date = '2023-01-01'
        days = 100 
        num_features = 10 
        self.sample_data = generate_sample_data(start_date, days, num_features) 

        #Extract features
        self.feature_engineering = FeatureEngineering()
        self.input = self.feature_engineering.extract_features(self.sample_data)

        self.input_size = self.input.shape[1]
        self.hidden_size = 20
        self.num_layers = 2
        

        self.lstm_pos_enc = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers, 
            batch_first=True
        )

    def test_initialization(self):
        self.assertEqual(self.lstm_pos_enc.input_size, self.input_size)
        self.assertEqual(self.lstm_pos_enc.hidden_size, self.hidden_size)
        self.assertEqual(self.lstm_pos_enc.num_layers, self.num_layers)

    def test_encode_method(self):
        sample_input = Tensor([[1.0] * self.input_size])
        encoded_output = self.lstm_pos_enc.encode(sample_input)
        self.assertTrue((encoded_output == sample_input).all())