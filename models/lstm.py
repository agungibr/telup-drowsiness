import torch
import torch.nn as nn
from utils.config import Config

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(Config.INPUT_DIM, 128, num_layers=2, batch_first=True, dropout=Config.DROPOUT)
        self.fc = nn.Linear(128, 2)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])