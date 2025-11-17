from torch import nn

class LSTMPositionalEncoding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_size = kwargs.get('input_size')
        self.hidden_size = kwargs.get('hidden_size')
        self.num_layers = kwargs.get('num_layers')  
        self.batch_first = kwargs.get('batch_first', True)  

        # Define LSTM layer
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first 
        ) 
        #Define normalization 
        self.norm_input = nn.LayerNorm(self.input_size)
        self.norm_output = nn.LayerNorm(self.hidden_size) 



        self.seq = nn.Sequential(
            self.norm_input,
            self.encoder_lstm, 
            self.norm_output
        )

    def forward(self, x): 
        output, _ = self.encoder_lstm(x)
        return output