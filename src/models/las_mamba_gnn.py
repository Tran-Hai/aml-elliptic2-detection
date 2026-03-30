"""
LAS-Mamba-GNN Model
Complete architecture combining LAS, Mamba, and Graph Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .las_layer import LASLayer, LASWithStatistics, create_las_layer
from .mamba_layer import MambaDualEncoder, create_mamba_layer
from .gnn_layer import GNNEncoder, create_gnn_encoder


class LASMambaGNN(nn.Module):
    """
    LAS-Mamba-GNN: Complete model for AML detection
    
    Architecture:
    1. LAS Layer: Analyze liquidity patterns from transaction sequences
    2. Mamba Layer: Sequential modeling using State Space Models
    3. GNN Layer: Graph structure encoding
    4. Classifier: Final prediction
    
    Input:
        - node_features: Node feature vectors [N, feature_dim]
        - sequences: Transaction sequences [N, 2, K, F] (in/out flows)
        - edge_index: Graph connectivity [2, E]
    
    Output:
        - logits: [N, num_classes]
    """
    
    def __init__(
        self,
        feature_dim=96,
        las_hidden_dim=64,
        mamba_hidden_dim=64,
        mamba_num_layers=2,
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        gnn_type='gat',
        num_heads=4,
        classifier_hidden_dim=128,
        num_classes=2,
        dropout=0.3,
        use_las=True,
        use_mamba=True,
        use_gnn=True,
        use_statistics=False
    ):
        super().__init__()
        
        self.use_las = use_las
        self.use_mamba = use_mamba
        self.use_gnn = use_gnn
        
        feature_dims = []
        
        if use_las:
            self.las = LASLayer(
                in_features=feature_dim,
                hidden_dim=las_hidden_dim,
                dropout=dropout
            )
            if use_statistics:
                self.las = LASWithStatistics(
                    in_features=feature_dim,
                    hidden_dim=las_hidden_dim,
                    dropout=dropout,
                    use_statistics=True
                )
            feature_dims.append(las_hidden_dim)
        
        if use_mamba:
            self.mamba = MambaDualEncoder(
                in_features=feature_dim,
                hidden_dim=mamba_hidden_dim,
                num_layers=mamba_num_layers,
                dropout=dropout,
                pooling='last'
            )
            feature_dims.append(mamba_hidden_dim)
        
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
            feature_dims.append(gnn_hidden_dim)
        
        combined_dim = sum(feature_dims)
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.LayerNorm(classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(classifier_hidden_dim // 2, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_features, sequences, edge_index):
        """
        Args:
            node_features: [N, feature_dim]
            sequences: [N, 2, K, F]
            edge_index: [2, E]
        
        Returns:
            logits: [N, num_classes]
        """
        representations = []
        
        if self.use_las:
            las_repr = self.las(sequences)
            representations.append(las_repr)
        
        if self.use_mamba:
            mamba_repr = self.mamba(sequences)
            representations.append(mamba_repr)
        
        if self.use_gnn:
            gnn_repr = self.gnn(node_features, edge_index)
            representations.append(gnn_repr)
        
        combined = torch.cat(representations, dim=1)
        
        fused = self.fusion(combined)
        
        logits = self.classifier(fused)
        
        return logits
    
    def get_embeddings(self, node_features, sequences, edge_index):
        """
        Get intermediate embeddings for analysis.
        """
        representations = []
        
        if self.use_las:
            las_repr = self.las(sequences)
            representations.append(las_repr)
        
        if self.use_mamba:
            mamba_repr = self.mamba(sequences)
            representations.append(mamba_repr)
        
        if self.use_gnn:
            gnn_repr = self.gnn(node_features, edge_index)
            representations.append(gnn_repr)
        
        combined = torch.cat(representations, dim=1)
        
        fused = self.fusion(combined)
        
        return fused


class LASMambaGNNWithEntity(nn.Module):
    """
    Extended model that also uses entity type features.
    """
    
    def __init__(
        self,
        feature_dim=96,
        entity_dim=1,
        las_hidden_dim=64,
        mamba_hidden_dim=64,
        mamba_num_layers=2,
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        gnn_type='gat',
        num_heads=4,
        classifier_hidden_dim=128,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.entity_embed = nn.Embedding(3, 16)
        
        self.las = LASLayer(
            in_features=feature_dim,
            hidden_dim=las_hidden_dim,
            dropout=dropout
        )
        
        self.mamba = MambaDualEncoder(
            in_features=feature_dim,
            hidden_dim=mamba_hidden_dim,
            num_layers=mamba_num_layers,
            dropout=dropout
        )
        
        self.gnn = GNNEncoder(
            in_features=feature_dim,
            hidden_dim=gnn_hidden_dim,
            out_features=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            num_heads=num_heads,
            dropout=dropout,
            gnn_type=gnn_type
        )
        
        combined_dim = las_hidden_dim + mamba_hidden_dim + gnn_hidden_dim + 16
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Linear(classifier_hidden_dim // 2, num_classes)
    
    def forward(self, node_features, sequences, edge_index, entity_type):
        """
        Args:
            node_features: [N, feature_dim]
            sequences: [N, 2, K, F]
            edge_index: [2, E]
            entity_type: [N] - Entity type (0=unknown, 1=wallet, 2=exchange)
        
        Returns:
            logits: [N, num_classes]
        """
        las_repr = self.las(sequences)
        mamba_repr = self.mamba(sequences)
        gnn_repr = self.gnn(node_features, edge_index)
        entity_repr = self.entity_embed(entity_type)
        
        combined = torch.cat([las_repr, mamba_repr, gnn_repr, entity_repr], dim=1)
        
        fused = self.fusion(combined)
        
        logits = self.classifier(fused)
        
        return logits


def create_las_mamba_gnn(config):
    """
    Factory function to create LAS-Mamba-GNN model from config.
    """
    return LASMambaGNN(
        feature_dim=config.get('feature_dim', 96),
        las_hidden_dim=config.get('las_hidden_dim', 64),
        mamba_hidden_dim=config.get('mamba_hidden_dim', 64),
        mamba_num_layers=config.get('mamba_num_layers', 2),
        gnn_hidden_dim=config.get('gnn_hidden_dim', 128),
        gnn_num_layers=config.get('gnn_num_layers', 2),
        gnn_type=config.get('gnn_type', 'gat'),
        num_heads=config.get('num_heads', 4),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.3),
        use_las=config.get('use_las', True),
        use_mamba=config.get('use_mamba', True),
        use_gnn=config.get('use_gnn', True),
        use_statistics=config.get('use_statistics', False)
    )
