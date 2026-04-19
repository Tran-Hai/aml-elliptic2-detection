"""
Mamba-GNN Model
Kiến trúc kết hợp Mamba (Temporal) và GNN (Structural) cho bài toán AML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_layer import MambaDualEncoder
from .gnn_layer import GNNEncoder


class AttentionFusion(nn.Module):
    """
    Cơ chế Attention Fusion để tự động học trọng số giữa Mamba và GNN.
    """
    def __init__(self, input_dims, output_dim, dropout=0.3):
        super().__init__()
        self.num_branches = len(input_dims)
        
        # Project các nhánh về cùng một kích thước
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            ) for dim in input_dims
        ])
        
        self.attn_linear = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, representations):
        # representations: list các tensor [N, dim]
        projected = [proj(repr).unsqueeze(1) for proj, repr in zip(self.projections, representations)]
        combined = torch.cat(projected, dim=1) # [N, 2, output_dim]
        
        # Tính attention weights
        scores = self.attn_linear(combined).squeeze(-1) # [N, 2]
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # [N, 2, 1]
        
        # Tổng hợp có trọng số
        fused = (combined * weights).sum(dim=1) # [N, output_dim]
        return fused, weights


class MambaGNN(nn.Module):
    """
    Mô hình Mamba-GNN tinh gọn, loại bỏ LAS.
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
        mamba_pooling='mean'
    ):
        super().__init__()
        
        # Adapter để đưa node features về feature_dim (mặc định xử lý 576 chiều từ trainer)
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
            
        # Fusion
        self.fusion = AttentionFusion(branch_dims, classifier_hidden_dim, dropout)
        
        # Classifier
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
        
        # 1. Nhánh Mamba
        if self.use_mamba:
            mamba_repr = self.mamba(sequences)
            representations.append(mamba_repr)
            
        # 2. Nhánh GNN
        if self.use_gnn:
            # Đảm bảo feature đầu vào đúng dim
            node_feat_proj = self.input_adapter(node_features)
            gnn_repr = self.gnn(node_feat_proj, edge_index)
            representations.append(gnn_repr)
            
        # 3. Fusion & Prediction
        fused, attn_weights = self.fusion(representations)
        logits = self.classifier(fused)
        
        return logits


def create_mamba_gnn(config):
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
        mamba_pooling=config.get('mamba_pooling', 'mean')
    )
