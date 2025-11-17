from torch import nn 
from .lstm_encoder import LSTMPositionalEncoding
from .timeseries_transformer import TimeSeriesTransformer

class LSTMTransformerClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_size = kwargs.get('input_size')
        self.hidden_size = kwargs.get('hidden_size')
        self.num_layers = kwargs.get('num_layers')
        self.num_heads = kwargs.get('num_heads')
        self.num_transformer_layers = kwargs.get('num_transformer_layers')
        self.output_size = kwargs.get('output_size')
        self.batch_first = kwargs.get('batch_first', True)

        # LSTM Positional Encoding
        self.lstm_positional_encoding = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first
        )

        # Transformer 
        self.transformer_encoder = TimeSeriesTransformer(
            positional_encoding=self.lstm_positional_encoding,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers,
            output_size=self.output_size,
            batch_first=self.batch_first
        )

    def forward(self, x):
        x = self.transformer_encoder(x)
        return x