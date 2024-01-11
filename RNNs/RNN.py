import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1 = nn.RNN(input_size=300, hidden_size=150, num_layers=2)
        self.rnn2 = nn.RNN(input_size=150, hidden_size=150, num_layers=2)
        self.fc1 = nn.Linear(150, 150)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(150, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x[-1]
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x