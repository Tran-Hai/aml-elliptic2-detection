"""
Mamba Layer - State Space Model for Sequence Encoding
Sequential modeling for transaction sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SSMBlock(nn.Module):
    """
    State Space Model Block
    
    Implements: h'(t) = A * h(t) + B * x(t)
                y(t) = C * h(t) + D * x(t)
    
    Using selective state space mechanism for efficiency.
    """
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Project input
        self.x_proj = nn.Linear(d_model, d_model * 2)
        self.x_proj_gate = nn.Linear(d_model, d_model)
        
        # State projection
        self.state_proj = nn.Linear(d_model, d_state)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # A matrix (state transition) - learnable
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        
        # D matrix (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.A_log, mean=0, std=0.02)
    
    def forward(self, x, state=None):
        """
        x: [batch, seq_len, d_model]
        state: [batch, d_state] (optional)
        
        Returns:
            output: [batch, seq_len, d_model]
            state: [batch, d_state]
        """
        batch, seq_len, d_model = x.shape
        
        # Gate mechanism
        x_gate = F.silu(self.x_proj_gate(x))
        
        # Project to state space
        s = self.state_proj(x_gate)
        
        # A matrix with softplus for stability
        A = -torch.exp(self.A_log.float())
        
        # Discretization (ZOH)
        A_exp = torch.exp(A.unsqueeze(0) * torch.arange(seq_len, device=x.device).float().unsqueeze(1).unsqueeze(2))
        
        # State computation (simplified)
        if state is None:
            state = torch.zeros(batch, self.d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            u = x_gate[:, t, :]
            
            # State update
            state = state * torch.exp(A.mean(dim=-1)) + s[:, t, :]
            
            # Output
            y = torch.matmul(state, self.state_proj.weight) + self.D * u
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        output = self.out_proj(output)
        
        return output, state


class MambaBlock(nn.Module):
    """
    Full Mamba block with normalization.
    """
    
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SSMBlock(d_model, d_state, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, state=None):
        residual = x
        x = self.norm(x)
        x, state = self.ssm(x, state)
        x = self.dropout(x)
        return x + residual, state


class MambaEncoder(nn.Module):
    """
    Mamba Encoder for sequence modeling.
    
    Processes transaction sequences using State Space Models.
    
    Input:  [N, K, F] - N nodes, K transactions, F features
    Output: [N, hidden_dim] - Encoded representation
    """
    
    def __init__(
        self,
        in_features=96,
        hidden_dim=64,
        num_layers=2,
        d_state=16,
        dropout=0.3,
        pooling='last'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Pooling
        if pooling == 'attention':
            self.attention_pool = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        x: [N, K, F] - Node sequence
        mask: [N, K] - Optional mask for padding
        
        Returns:
            output: [N, hidden_dim]
        """
        N, K, F = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [N, K, hidden]
        
        # Mamba layers
        state = None
        for layer in self.layers:
            x, state = layer(x, state)
        
        # Pooling
        if self.pooling == 'last':
            output = x[:, -1, :]  # [N, hidden]
        elif self.pooling == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                output = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                output = x.mean(dim=1)
        elif self.pooling == 'attention':
            weights = torch.softmax(self.attention_pool(x), dim=1)
            output = (weights * x).sum(dim=1)
        else:
            output = x.mean(dim=1)
        
        # Output projection
        output = self.output_proj(output)
        
        return output


class MambaDualEncoder(nn.Module):
    """
    Dual Mamba Encoder for both in-flow and out-flow sequences.
    
    Input:  [N, 2, K, F] - N nodes, 2 flows (in/out), K transactions, F features
    Output: [N, hidden_dim] - Combined representation
    """
    
    def __init__(
        self,
        in_features=96,
        hidden_dim=64,
        num_layers=2,
        d_state=16,
        dropout=0.3,
        pooling='last'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        
        # Separate encoders for in-flow and out-flow
        self.in_encoder = MambaEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            dropout=dropout,
            pooling=pooling
        )
        
        self.out_encoder = MambaEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            dropout=dropout,
            pooling=pooling
        )
        
        # Combiner
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x, mask=None):
        """
        x: [N, 2, K, F]
        mask: [N, 2, K]
        
        Returns:
            output: [N, hidden_dim]
        """
        in_flow = x[:, 0, :, :]  # [N, K, F]
        out_flow = x[:, 1, :, :]  # [N, K, F]
        
        if mask is not None:
            in_mask = mask[:, 0, :]
            out_mask = mask[:, 1, :]
        else:
            in_mask = None
            out_mask = None
        
        in_repr = self.in_encoder(in_flow, in_mask)
        out_repr = self.out_encoder(out_flow, out_mask)
        
        combined = torch.cat([in_repr, out_repr], dim=1)
        output = self.combiner(combined)
        
        return output


def create_mamba_layer(config):
    """Factory function to create Mamba layer from config."""
    
    use_dual = config.get('use_dual_encoder', True)
    
    if use_dual:
        return MambaDualEncoder(
            in_features=config.get('feature_dim', 96),
            hidden_dim=config.get('mamba_hidden_dim', 64),
            num_layers=config.get('mamba_num_layers', 2),
            d_state=config.get('mamba_d_state', 16),
            dropout=config.get('dropout', 0.3),
            pooling=config.get('mamba_pooling', 'last')
        )
    else:
        return MambaEncoder(
            in_features=config.get('feature_dim', 96),
            hidden_dim=config.get('mamba_hidden_dim', 64),
            num_layers=config.get('mamba_num_layers', 2),
            d_state=config.get('mamba_d_state', 16),
            dropout=config.get('dropout', 0.3),
            pooling=config.get('mamba_pooling', 'last')
        )