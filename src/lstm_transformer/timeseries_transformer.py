import torch 

class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()  

        self.output_size = kwargs.get('output_size')

        self.num_heads = kwargs.get('num_heads')
        self.num_layers = kwargs.get('num_layers') 
        self.positional_encoding = kwargs.get('positional_encoding', None)
        self.batch_first = kwargs.get('batch_first', True)

        if (not self.positional_encoding):
            raise ValueError("Positional encoding module must be provided")
        
        self.input_size = self.positional_encoding.output_size

        self.norm = torch.nn.LayerNorm(self.input_size)
        
        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.input_size,
            nhead=self.num_heads,
            batch_first=self.batch_first
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=self.num_layers
        )

        self.fc = torch.nn.Linear(self.input_size, self.output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # Apply positional encoding (LSTM encoder)
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)

        #Normalize x 
        
        
        # Apply fully connected layer and sigmoid
        x = self.norm(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


        

       

