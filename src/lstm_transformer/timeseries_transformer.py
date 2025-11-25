import torch 

class TimeSeriesTransformer(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()  
        
        self.num_heads = kwargs.get('num_heads')
        self.num_layers = kwargs.get('num_layers') 
        self.input_size = kwargs.get('input_size', None)
        self.batch_first = kwargs.get('batch_first', True)

        

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

    def forward(self, x, mask=None):
        # Apply transformer encoder with mask
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Normalize x
        x = self.norm(x)
    
        
        return x


        

       

