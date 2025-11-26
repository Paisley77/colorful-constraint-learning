import torch
import torch.nn as nn
import numpy as np

class MinimalTCN(nn.Module):
    """Lightweight TCN for temporal pattern detection in HSV trajectories"""
    
    def __init__(self, input_dim=3, hidden_dim=16, num_layers=2, kernel_size=3, output_dim=1):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=(kernel_size-1)//2),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_channels = hidden_dim
        
        self.temporal_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)  # -> [batch_size, input_dim, seq_len]
        features = self.temporal_layers(x)  # [batch_size, hidden_dim, seq_len]
        features = features.transpose(1, 2)  # -> [batch_size, seq_len, hidden_dim]
        output = self.output_layer(features)  # [batch_size, seq_len, output_dim]
        return output.mean(dim=1)  # Global temporal pooling

class TemporalPatternBank(nn.Module):
    """Bank of TCNs learning different temporal patterns"""
    
    def __init__(self, num_tcns=4, hsv_dim=3):
        super().__init__()
        self.tcns = nn.ModuleList([MinimalTCN(input_dim=hsv_dim) for _ in range(num_tcns)])
        self.weights = nn.Parameter(torch.ones(num_tcns) / num_tcns)  # Sparse combination
        
    def forward(self, hsv_trajectories):
        """Apply all TCNs and return weighted combination"""
        pattern_scores = []
        for tcn in self.tcns:
            scores = tcn(hsv_trajectories)  # [batch_size, 1]
            pattern_scores.append(scores)
        
        pattern_scores = torch.stack(pattern_scores, dim=-1)  # [batch_size, 1, num_tcns]
        pattern_scores = pattern_scores.squeeze(dim=1) # [batch_size, num_tcns]
        weighted_scores = (pattern_scores * torch.softmax(self.weights, dim=0)).sum(dim=-1)
        
        return weighted_scores, pattern_scores # [batch_size,], [batch_size, num_tcns] 