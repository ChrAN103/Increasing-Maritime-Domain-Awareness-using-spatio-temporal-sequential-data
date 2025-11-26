from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class LSTMPositionalEncoding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_size = kwargs.get('input_size')
        self.hidden_size = kwargs.get('hidden_size')
        self.num_layers = kwargs.get('num_layers')  
        self.batch_first = kwargs.get('batch_first', True)   
        self.dropout = kwargs.get('dropout', 0.1)

        self.output_size = self.hidden_size 

        # Define LSTM layer
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout
        ) 



    def forward(self, x, lengths): 

        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        packed_out, (hn, _) = self.encoder_lstm(packed_x)

        out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)
   
        return out