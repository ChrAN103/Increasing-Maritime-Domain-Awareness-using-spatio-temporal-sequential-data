import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from Dataloader import Dataloader
from utils import overhaul_segments, MaritimeDataset
from shapely import wkt
import geopandas as gpd

def filter_destination(df, ports, drop_destinations=False):
    #set destination outside of ports to 'transit'
    if drop_destinations:
        df = df[~df['Destination'].isin(ports['LOCODE'])]
    else:
        df['Destination'] = df['Destination'].where(df['Destination'].isin(ports['LOCODE']), 'transit')
    return df

def find_ports():
    # 1. Load CSV
    locodes = pd.read_csv("port_locodes.csv", sep=";")

    # 2. Clean the data: Drop rows where POLYGON is missing (NaN)
    locodes = locodes.dropna(subset=['POLYGON'])

    # 3. Convert to WKT
    # The function wraps the coordinate pairs in the standard WKT format
    def to_wkt_polygon(coord_string):
        return f"POLYGON(({coord_string}))"

    locodes['WKT'] = locodes['POLYGON'].apply(to_wkt_polygon)
    locodes['geometry'] = locodes['WKT'].apply(wkt.loads)

    # 4. Convert to GeoDataFrame
    ports_gdf = gpd.GeoDataFrame(locodes, geometry='geometry', crs="EPSG:4326")

    # 5. Calculate Lat/Lon for filtering
    ports_gdf["Latitude"] = ports_gdf.geometry.centroid.y
    ports_gdf["Longitude"] = ports_gdf.geometry.centroid.x

    # 6. Filter by Bbox
    bbox = [60, 0, 50, 20] # North, West, South, East
    north, west, south, east = bbox

    ports = ports_gdf[
        (ports_gdf["Latitude"] <= north) & 
        (ports_gdf["Latitude"] >= south) & 
        (ports_gdf["Longitude"] >= west) & 
        (ports_gdf["Longitude"] <= east)
    ]

    return ports

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Returns:
        padded_seqs: Tensor of shape (batch_size, max_len, input_size)
        lengths: Tensor of shape (batch_size) containing original lengths
        targets: Tensor of shape (batch_size)
    """
    sequences, targets = zip(*batch)
    
    # Get lengths of each sequence
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequences (batch_first=True makes it [Batch, Seq, Feat])
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=-1000) # Use a unlikely padding value according to standardization as we use StandardScaler
    
    targets = torch.tensor(targets, dtype=torch.long) 
    
    return padded_seqs, lengths, targets

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM is generally better than vanilla RNN for long sequences
        # batch_first=True expects input as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout if num_layers > 1 else 0)
        
        # Fully connected layer to map hidden state to output classes (ports)
        # self.fc = nn.Linear(hidden_size, output_size)
        # We add an intermediate layer with Batch Norm and ReLU for better learning
        self.fc_head = nn.Sequential(
            nn.Dropout(dropout),                  # Regularization on LSTM output
            nn.Linear(hidden_size, hidden_size),  # Intermediate projection
            nn.BatchNorm1d(hidden_size),          # Stabilize training
            nn.ReLU(),                            # Non-linearity
            nn.Dropout(dropout),                  # More regularization
            nn.Linear(hidden_size, output_size)   # Final prediction
        )
        
    def forward(self, x, lengths):
        # x shape: (batch_size, max_seq_len, input_size)
        
        # Pack the sequence to ignore padding
        # enforce_sorted=False allows sequences to be in any order
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Forward pass through LSTM
        # out is packed, _ contains (hidden_state, cell_state)
        packed_out, (hn, cn) = self.lstm(packed_x)
        
        # We typically use the final hidden state for classification
        # hn shape: (num_layers, batch_size, hidden_size)
        # We take the last layer's hidden state
        final_hidden = hn[-1]
        
        # Pass through fully connected layer
        # out = self.fc(final_hidden)
        out = self.fc_head(final_hidden)
        return out

def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for sequences, lengths, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(sequences, lengths)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients (common cause of NaN in RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Validation 
        if val_loader:
            evaluate_model(model, val_loader, device)
            
    return model

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, lengths, targets in val_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

def setup_and_train(train_df, val_df, test_df, model):
    # 1. Create Datasets
    # Initialize dataset with training data to fit scalers/encoders
    train_dataset = MaritimeDataset(train_df)
    
    # Use the fitted encoder/scaler for test data
    test_dataset = MaritimeDataset(test_df, 
                                 port_encoder=train_dataset.port_encoder,
                                 feature_scaler=train_dataset.feature_scaler)
    
    val_dataset = MaritimeDataset(val_df, 
                                 port_encoder=train_dataset.port_encoder,
                                 feature_scaler=train_dataset.feature_scaler)

    # 2. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # 3. Initialize Model
    input_size = len(train_dataset.feature_cols) # e.g., 4 (Lat, Lon, SOG, COG)
    hidden_size = 128
    output_size = len(train_dataset.port_encoder.classes_) # Number of unique ports
    if model == "LSTM":
        model = LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            num_layers=5,      # Deeper network (5 stacked LSTMs)
            dropout=0.3        # 30% Dropout to prevent overfitting
        )
    if model == "LSTM_Transformer_Classifier":
        pass  # Placeholder for future model implementation

    # 4. Train
    trained_model = train_model(model, train_loader, val_loader, num_epochs=25,learning_rate=0.0001)
    
    # 5. evalutate on test set
    print("Final evaluation on test set:")
    evaluate_model(trained_model, test_loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return trained_model

if __name__ == "__main__":
    path = "../data/processed_data"
    dataloader = Dataloader(out_path=path)
    df = dataloader.load_data()  # load all files in the processed_data folderS
    # Ensure ship and segment can be told apart by adding column for date
    df['Date'] = df['Timestamp'].dt.date
    df = overhaul_segments(df)
    df.drop(columns=['Segment'], inplace=True)
    df.rename(columns={"Segment_ID": "Segment"}, inplace=True)
    df = filter_destination(df, find_ports())
    print("Data loaded and segments overhauled.")
    # This prepares the data by removing the last X hours from test routes
    train_df, test_df = dataloader.train_test_split(df = df, prediction_horizon_hours=2.0)
    train_df, val_df = dataloader.train_test_split(df = train_df, prediction_horizon_hours=0, test_size=0.2) # should not remove further hours for validation as 2 hours are already removed
    print("Data split into train and test.")
    # 4. Train the RNN
    print("Starting training...")
    trained_model = setup_and_train(train_df=train_df, val_df=val_df, test_df=test_df, model="LSTM")
    print("Training completed.")