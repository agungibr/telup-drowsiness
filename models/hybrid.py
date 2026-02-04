import torch
import torch.nn as nn
from utils.config import Config

class TemporalReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_in = nn.LayerNorm(Config.INPUT_DIM)
        self.project = nn.Sequential(
            nn.Linear(Config.INPUT_DIM, Config.HIDDEN_DIM), 
            nn.GELU(), 
            nn.Dropout(Config.DROPOUT)
        )
        self.pos_embed = nn.Parameter(torch.randn(1, Config.SEQ_LEN, Config.HIDDEN_DIM) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.HIDDEN_DIM, 
            nhead=Config.NUM_HEADS, 
            dim_feedforward=Config.HIDDEN_DIM*4, 
            dropout=Config.DROPOUT, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.NUM_LAYERS)
        
        self.lstm = nn.LSTM(
            Config.HIDDEN_DIM, 
            Config.HIDDEN_DIM // 2, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.attn_pool = nn.Sequential(nn.Linear(Config.HIDDEN_DIM, 1), nn.Tanh())
        self.fc = nn.Sequential(
            nn.Linear(Config.HIDDEN_DIM, 64), 
            nn.ReLU(), 
            nn.Dropout(Config.DROPOUT), 
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.norm_in(x)
        x = self.project(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)
        context = torch.sum(x * attn_weights, dim=1)
        return self.fc(context)