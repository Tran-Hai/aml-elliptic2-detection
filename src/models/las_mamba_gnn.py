"""
Mamba Classifier Model for AML Detection
Architecture: Mamba for sequential modeling + MLP Classifier

    Input: Transaction sequences [N, 2, K, F] (in/out flows)
    Output: logits [N, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_layer import MambaDualEncoder, create_mamba_layer


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
        self.mamba = MambaDualEncoder(
            in_features=feature_dim,
            hidden_dim=mamba_hidden_dim,
            num_layers=mamba_num_layers,
            dropout=dropout,
            pooling='last'
        )
        
        mamba_output_dim = mamba_hidden_dim
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(mamba_output_dim, classifier_hidden_dim),
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
        """Careful weight initialization to prevent NaN"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, node_features, sequences, edge_index=None):
        """
        Args:
            node_features: [N, feature_dim]
            sequences: [N, 2, K, F] (in/out flows)
            edge_index: Not used (kept for compatibility)
        
        Returns:
            logits: [N, num_classes]
        """
        # Extract sequential features using Mamba
        mamba_repr = self.mamba(sequences)
        
        if torch.isnan(mamba_repr).any():
            mamba_repr = torch.zeros_like(mamba_repr)
        
        # Classify
        logits = self.classifier(mamba_repr)
        
        return logits


def create_mamba_classifier(config):
    """Factory function to create Mamba classifier."""
    return MambaClassifier(
        feature_dim=config.get('feature_dim', 96),
        mamba_hidden_dim=config.get('mamba_hidden_dim', 64),
        mamba_num_layers=config.get('mamba_num_layers', 2),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.3),
    )


# Keep backward compatibility
class LASMambaGNN(nn.Module):
    """Legacy wrapper - now only uses Mamba + Classifier"""
    
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
        use_las=False,
        use_mamba=True,
        use_gnn=False,
        use_statistics=False
    ):
        super().__init__()
        
        # Now only uses Mamba
        self.mamba = MambaDualEncoder(
            in_features=feature_dim,
            hidden_dim=mamba_hidden_dim,
            num_layers=mamba_num_layers,
            dropout=dropout,
            pooling='last'
        )
        
        mamba_output_dim = mamba_hidden_dim
        
        # MLP Classifier
        self.classifier = nn.Sequential(
            nn.Linear(mamba_output_dim, classifier_hidden_dim),
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
    
    def forward(self, node_features, sequences, edge_index=None):
        mamba_repr = self.mamba(sequences)
        
        if torch.isnan(mamba_repr).any():
            mamba_repr = torch.zeros_like(mamba_repr)
        
        logits = self.classifier(mamba_repr)
        
        return logits


# Factory function - now creates Mamba-only model
def create_las_mamba_gnn(config):
    """Create Mamba classifier model."""
    return LASMambaGNN(
        feature_dim=config.get('feature_dim', 96),
        mamba_hidden_dim=config.get('mamba_hidden_dim', 64),
        mamba_num_layers=config.get('mamba_num_layers', 2),
        classifier_hidden_dim=config.get('classifier_hidden_dim', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.3),
        use_mamba=config.get('use_mamba', True),
    )