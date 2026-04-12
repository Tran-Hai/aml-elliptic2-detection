"""
Mamba Classifier Model for AML Detection
Architecture: Mamba for sequential modeling + MLP Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Simple Mamba-like block for sequential modeling"""
    
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.x_proj = nn.Linear(d_model, d_state * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        residual = x
        x = self.norm(x)
        
        # Gate mechanism
        gate = self.in_proj(x)
        gate = torch.tanh(gate[:, :, :self.d_model]) * torch.sigmoid(gate[:, :, self.d_model:])
        
        # Simple state projection
        state = self.x_proj(x)
        state = torch.tanh(state[:, :, :self.d_state])
        
        # Output
        out = self.out_proj(gate)
        out = out + residual
        
        return out


class MambaEncoder(nn.Module):
    """Mamba encoder for sequential data"""
    
    def __init__(self, in_features, hidden_dim, num_layers=2, dropout=0.3, pooling='last'):
        super().__init__()
        self.pooling = pooling
        
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # x: [batch, 2, seq_len, features] -> [batch, seq_len, features]
        x = x.mean(dim=1)  # Average in/out flows
        
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        if self.pooling == 'last':
            return x[:, -1, :]  # [batch, hidden_dim]
        elif self.pooling == 'mean':
            return x.mean(dim=1)
        else:
            return x[:, -1, :]


class MambaClassifier(nn.Module):
    """
    Mamba-based classifier for AML detection.
    Uses Mamba for sequential feature extraction + MLP classifier.
    """
    
    def __init__(
        self,
        feature_dim=96,
        mamba_hidden_dim=64,
        mamba_num_layers=2,
        classifier_hidden_dim=128,
        num_classes=2,
        dropout=0.3,
    ):
        super().__init__()
        
        # Mamba for sequential modeling
        self.mamba = MambaEncoder(
            in_features=feature_dim,
            hidden_dim=mamba_hidden_dim,
            num_layers=mamba_num_layers,
            dropout=dropout,
            pooling='last'
        )
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(mamba_hidden_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.LayerNorm(classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, sequences):
        """
        Args:
            sequences: [N, 2, K, F] (in/out flows)
        
        Returns:
            logits: [N, num_classes]
        """
        mamba_repr = self.mamba(sequences)
        
        if torch.isnan(mamba_repr).any():
            mamba_repr = torch.zeros_like(mamba_repr)
        
        logits = self.classifier(mamba_repr)
        
        return logits


def create_model(config):
    """
    Factory function to create Mamba classifier.
    
    Args:
        config: dict with model parameters
    
    Returns:
        MambaClassifier model
    """
    return MambaClassifier(
        feature_dim=config.get('feature_dim', 96),
        mamba_hidden_dim=config.get('mamba_hidden_dim', 64),
        mamba_num_layers=config.get('mamba_num_layers', 2),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.3),
    )