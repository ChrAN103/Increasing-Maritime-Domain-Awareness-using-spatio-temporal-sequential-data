import torch

class Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.norm = torch.nn.LayerNorm(hidden_size)


    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x