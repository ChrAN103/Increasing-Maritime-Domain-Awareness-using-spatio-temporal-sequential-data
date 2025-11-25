from torch import nn 
from .lstm_encoder import LSTMPositionalEncoding
from .timeseries_transformer import TimeSeriesTransformer 
from .classifier import Classifier

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
        self.dropout = kwargs.get('dropout', 0.1)


        # LSTM Positional Encoding & Embedder
        self.lstm_positional_encoding = LSTMPositionalEncoding(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first
        )

        # Transformer 
        self.transformer_encoder = TimeSeriesTransformer(
            input_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=self.num_transformer_layers,
            batch_first=self.batch_first
        )

        # Classifier
        self.fc_head = nn.Sequential(
            nn.Dropout(self.dropout),                  # Regularization on LSTM output
            nn.Linear(self.hidden_size, self.hidden_size),  # Intermediate projection
            nn.BatchNorm1d(self.hidden_size),          # Stabilize training
            nn.ReLU(),                            # Non-linearity
            nn.Dropout(self.dropout),                  # More regularization
            nn.Linear(self.hidden_size, self.output_size)   # Final prediction
        )

        self.dropout_layer = nn.Dropout(p=self.dropout) 

        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(self, x, mask=None, lengths=None):

        x = self.lstm_positional_encoding(x, lengths)

        x = self.norm(x)
        x = self.dropout_layer(x)
        x = self.transformer_encoder(x, mask=mask)

        # preform pooling 
        x = x.mean(dim=1)  # Mean pooling over the sequence length dimension

        x = self.fc_head(x)
        return x