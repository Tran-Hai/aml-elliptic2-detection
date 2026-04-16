"""
Mamba-GNN Model
Complete architecture combining Mamba and Graph Neural Networks for AML detection.

    Architecture:
    1. Mamba Layer: Sequential modeling using State Space Models (SSM)
    2. GNN Layer: Graph structure encoding (GAT/GCN/GraphSAGE)
    3. Classifier: Final prediction based on Attention Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_layer import MambaDualEncoder
from .gnn_layer import GNNEncoder


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for multiple representations.
    Learns to weight the importance of each branch (Mamba, GNN).
    """
    def __init__(self, input_dims, output_dim, dropout=0.3):
        super().__init__()
        self.num_branches = len(input_dims)
        self.output_dim = output_dim
        
        # Project each branch to a common dimension if they differ
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            ) for dim in input_dims
        ])
        
        # Attention mechanism
        self.attn_linear = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, representations):
        """
        Args: representations (list of Tensors): [N, dim_i]
        """
        projected = [proj(repr).unsqueeze(1) for proj, repr in zip(self.projections, representations)]
        combined = torch.cat(projected, dim=1) # [N, num_branches, output_dim]
        
        # Simple attention
        scores = self.attn_linear(combined).squeeze(-1) # [N, num_branches]
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # [N, num_branches, 1]
        
        # Weighted sum
        fused = (combined * weights).sum(dim=1) # [N, output_dim]
        return fused, weights


class MambaGNN(nn.Module):
    """
    Mamba-GNN: Complete model for AML detection
    Enhanced with Attention Fusion and Neighbor Expansion support.
    """
    
    def __init__(
        self,
        feature_dim=96,
        mamba_hidden_dim=64,
        mamba_num_layers=2,
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        gnn_type='gat',
        num_heads=4,
        classifier_hidden_dim=128,
        num_classes=2,
        dropout=0.3,
        use_mamba=True,
        use_gnn=True,
        mamba_pooling='last'
    ):
        super().__init__()
        
        # input_adapter handles 576-dim aggregated features
        self.input_adapter = nn.Linear(576, feature_dim)
        self.use_mamba = use_mamba
        self.use_gnn = use_gnn
        
        branch_dims = []
        if use_mamba:
            self.mamba = MambaDualEncoder(
                in_features=feature_dim, 
                hidden_dim=mamba_hidden_dim, 
                num_layers=mamba_num_layers, 
                dropout=dropout, 
                pooling=mamba_pooling
            )
            branch_dims.append(mamba_hidden_dim)
        
        if use_gnn:
            self.gnn = GNNEncoder(
                in_features=feature_dim, 
                hidden_dim=gnn_hidden_dim, 
                out_features=gnn_hidden_dim, 
                num_layers=gnn_num_layers, 
                num_heads=num_heads, 
                dropout=dropout, 
                gnn_type=gnn_type
            )
            branch_dims.append(gnn_hidden_dim)
        
        if not branch_dims:
            raise ValueError("At least one branch (Mamba or GNN) must be enabled.")

        # Attention Fusion instead of simple concat
        self.attention_fusion = AttentionFusion(branch_dims, classifier_hidden_dim, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.LayerNorm(classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_features, sequences, edge_index):
        representations = []
        
        if self.use_mamba:
            mamba_repr = self.mamba(sequences)
            representations.append(mamba_repr)
        
        if self.use_gnn:
            # node_features already contains [mean, max, last] info
            node_features = self.input_adapter(node_features)
            gnn_repr = self.gnn(node_features, edge_index)
            representations.append(gnn_repr)
        
        # Intelligent Fusion
        fused, attn_weights = self.attention_fusion(representations)
        
        logits = self.classifier(fused)
        
        # Safety clamp
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.clamp(logits, min=-100, max=100)
        
        return logits
    
    def get_embeddings(self, node_features, sequences, edge_index):
        """Get intermediate embeddings for analysis."""
        representations = []
        if self.use_mamba:
            representations.append(self.mamba(sequences))
        if self.use_gnn:
            node_features = self.input_adapter(node_features)
            representations.append(self.gnn(node_features, edge_index))
        
        fused, _ = self.attention_fusion(representations)
        return fused


def create_mamba_gnn(config):
    """
    Factory function to create Mamba-GNN model from config.
    """
    return MambaGNN(
        feature_dim=config.get('feature_dim', 96),
        mamba_hidden_dim=config.get('mamba_hidden_dim', 64),
        mamba_num_layers=config.get('mamba_num_layers', 2),
        gnn_hidden_dim=config.get('gnn_hidden_dim', 128),
        gnn_num_layers=config.get('gnn_num_layers', 2),
        gnn_type=config.get('gnn_type', 'gat'),
        num_heads=config.get('num_heads', 4),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.3),
        use_mamba=config.get('use_mamba', True),
        use_gnn=config.get('use_gnn', True),
        mamba_pooling=config.get('mamba_pooling', 'last')
    )
