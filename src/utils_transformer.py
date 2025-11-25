from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
from torch.utils.data import Dataset

def create_masked_batch(tensor_list):

    # Get original lengths
    lengths = torch.tensor([len(seq) for seq in tensor_list])
    
    # Pad sequences (pads with 0 by default)
    padded_batch = pad_sequence(tensor_list, batch_first=True, padding_value=0)
    
    # Create mask: True for real data, False for padding
    batch_size, max_len = padded_batch.shape[0], padded_batch.shape[1]
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    return padded_batch, mask, lengths 


def list_to_tensor(batch_list):
    """Convert list of one-hot encoded arrays to class indices tensor."""
    # Convert list of numpy arrays to numpy array first (faster)
    batch_array = np.array(batch_list)
    
    # Convert one-hot to class indices: argmax along last dimension
    class_indices = np.argmax(batch_array, axis=1)
    
    # Convert to Long tensor (int64) for CrossEntropyLoss
    return torch.tensor(class_indices, dtype=torch.long)


def collate_fn(batch, max_seq_length=512):
    """
    Collate function with sequence length truncation.
    
    Args:
        batch: List of (sequence, label) tuples
        max_seq_length: Maximum sequence length (default: 512)
    """
    sequences, labels = zip(*batch)
    
    # Truncate sequences to max_seq_length
    truncated_sequences = []
    for seq in sequences:
        if len(seq) > max_seq_length:
            # Take the last max_seq_length points (most recent trajectory)
            seq = seq[-max_seq_length:]
        truncated_sequences.append(seq)
    
    lengths = torch.tensor([len(seq) for seq in truncated_sequences])
    padded_sequences = pad_sequence(truncated_sequences, batch_first=True, padding_value=0)
    
    batch_size, max_len = padded_sequences.shape[0], padded_sequences.shape[1]
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)
    
    labels = torch.stack(labels)
    
    return padded_sequences, mask, lengths, labels


class SimpleDataset(Dataset):
    def __init__(self, X_list, Y_list):
        self.X = X_list
        self.Y = Y_list
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]