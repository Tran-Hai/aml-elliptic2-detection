"""
LAS Layer - Liquidity Analysis Layer
Analyze liquidity patterns from transaction sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LASPooling(nn.Module):
    """Pooling methods for sequence aggregation."""
    
    def __init__(self, pool_type='attention'):
        super().__init__()
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            self.attention = nn.Linear(96, 1)
    
    def forward(self, x):
        """
        x: [N, K, F] or [N, 2, K, F]
        """
        if x.dim() == 3:
            if self.pool_type == 'mean':
                return x.mean(dim=1)
            elif self.pool_type == 'max':
                return x.max(dim=1)[0]
            elif self.pool_type == 'attention':
                weights = torch.softmax(self.attention(x), dim=1)
                return (weights * x).sum(dim=1)
            elif self.pool_type == 'last':
                return x[:, -1, :]
            else:
                return x.mean(dim=1)
        else:
            return x


class LASLayer(nn.Module):
    """
    Liquidity Analysis Layer
    
    Analyzes transaction patterns from in-flow and out-flow sequences.
    
    Input:  [N, 2, K, F] - N nodes, 2 flows (in/out), K transactions, F features
    Output: [N, hidden_dim] - Liquidity representation for each node
    """
    
    def __init__(
        self,
        in_features=96,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        pool_type='attention'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Pooling for each flow
        self.in_pool = LASPooling(pool_type)
        self.out_pool = LASPooling(pool_type)
        
        # Process each flow separately
        self.flow_encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Process combined features
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """
        Args:
            x: Node features [N, 2, K, F] = [N, 2, 50, 96]
        
        Returns:
            Liquidity features [N, hidden_dim]
        """
        N = x.size(0)
        
        in_flow = x[:, 0, :, :]  # [N, K, F] - incoming transactions
        out_flow = x[:, 1, :, :]  # [N, K, F] - outgoing transactions
        
        in_repr = self.flow_encoder(in_flow)  # [N, K, hidden]
        in_repr = self.in_pool(in_repr)  # [N, hidden]
        
        out_repr = self.flow_encoder(out_flow)  # [N, K, hidden]
        out_repr = self.out_pool(out_repr)  # [N, hidden]
        
        combined = torch.cat([in_repr, out_repr], dim=1)  # [N, hidden*2]
        
        combined = self.combiner(combined)  # [N, hidden]
        
        output = self.output_proj(combined)  # [N, hidden]
        
        return output


class LASWithStatistics(nn.Module):
    """
    Enhanced LAS Layer with additional statistics.
    Includes aggregated statistics from sequences.
    """
    
    def __init__(
        self,
        in_features=96,
        hidden_dim=64,
        dropout=0.3,
        use_statistics=True
    ):
        super().__init__()
        
        self.use_statistics = use_statistics
        
        self.base_las = LASLayer(
            in_features=in_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        if use_statistics:
            self.stat_encoder = nn.Sequential(
                nn.Linear(8, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            
            hidden_dim = hidden_dim * 2
    
    def compute_statistics(self, x):
        """
        Compute statistical features from sequences.
        
        x: [N, 2, K, F]
        
        Returns: [N, 8] - 8 statistics per node
        """
        in_flow = x[:, 0, :, :]
        out_flow = x[:, 1, :, :]
        
        stats = []
        
        for flow in [in_flow, out_flow]:
            mean_val = flow.mean(dim=1)
            std_val = flow.std(dim=1)
            max_val = flow.max(dim=1)[0]
            min_val = flow.min(dim=1)[0]
            
            stats.extend([mean_val, std_val, max_val, min_val])
        
        stats = torch.stack(stats, dim=1)  # [N, 8]
        
        return stats
    
    def forward(self, x):
        las_output = self.base_las(x)
        
        if self.use_statistics:
            stats = self.compute_statistics(x)
            stats_repr = self.stat_encoder(stats)
            
            combined = torch.cat([las_output, stats_repr], dim=1)
            return combined
        
        return las_output


def create_las_layer(config):
    """Factory function to create LAS layer from config."""
    
    if config.get('use_statistics', False):
        return LASWithStatistics(
            in_features=config.get('feature_dim', 96),
            hidden_dim=config.get('las_hidden_dim', 64),
            dropout=config.get('dropout', 0.3),
            use_statistics=True
        )
    else:
        return LASLayer(
            in_features=config.get('feature_dim', 96),
            hidden_dim=config.get('las_hidden_dim', 64),
            num_layers=config.get('las_num_layers', 2),
            dropout=config.get('dropout', 0.3),
            pool_type=config.get('pool_type', 'attention')
        )