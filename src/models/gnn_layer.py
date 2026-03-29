"""
GNN Layer - Graph Neural Network using PyTorch Geometric
Supports GAT, GCN, and SAGE architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    
    Implements multi-head attention for graph convolution.
    """
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.3):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.out_features = out_features
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        self.query = nn.Linear(in_features, out_features)
        self.key = nn.Linear(in_features, out_features)
        self.value = nn.Linear(in_features, out_features)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(out_features, out_features)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Graph connectivity [2, E]
        
        Returns:
            Updated node features [N, out_features]
        """
        N = x.size(0)
        
        Q = self.query(x).view(N, self.num_heads, self.head_dim)
        K = self.key(x).view(N, self.num_heads, self.head_dim)
        V = self.value(x).view(N, self.num_heads, self.head_dim)
        
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        attention_scores = self.leaky_relu(attention_scores)
        
        row, col = edge_index
        mask = torch.zeros(N, N, device=x.device, dtype=torch.bool)
        mask[row, col] = True
        
        attention_scores = attention_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        aggregated = torch.matmul(attention_weights, V)
        
        aggregated = aggregated.transpose(0, 1).contiguous()
        aggregated = aggregated.view(N, self.out_features)
        
        output = self.output_proj(aggregated)
        
        return output


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer (GCN)
    
    Standard graph convolution operation.
    """
    
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Graph connectivity [2, E]
        
        Returns:
            Updated node features [N, out_features]
        """
        N = x.size(0)
        
        row, col = edge_index
        
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        deg = deg.pow(-0.5)
        deg[deg == float('inf')] = 0
        
        norm = deg[row] * deg[col]
        
        aggregated = torch.zeros_like(x)
        
        src = x[col] * norm.unsqueeze(-1)
        aggregated.index_add_(0, row, src)
        
        output = self.linear(aggregated)
        output = self.dropout(output)
        
        return output


class SAGEConvLayer(nn.Module):
    """
    GraphSAGE Convolution Layer
    
    Implements mean/pool aggregation for graph representation learning.
    """
    
    def __init__(self, in_features, out_features, dropout=0.3, aggr='mean'):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.aggr = aggr
        
        self.linear = nn.Linear(in_features * 2, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Graph connectivity [2, E]
        
        Returns:
            Updated node features [N, out_features]
        """
        N = x.size(0)
        row, col = edge_index
        
        aggregated = torch.zeros(N, self.in_features, device=x.device)
        
        src_features = x[col]
        aggregated.index_add_(0, row, src_features)
        
        num_neighbors = torch.zeros(N, device=x.device)
        num_neighbors.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
        num_neighbors = num_neighbors.clamp(min=1).unsqueeze(-1)
        
        aggregated = aggregated / num_neighbors
        
        if self.aggr == 'mean':
            neighbor_repr = aggregated
        elif self.aggr == 'max':
            neighbor_repr = torch.zeros(N, self.in_features, device=x.device)
            for i in range(row.size(0)):
                dst_idx = row[i]
                neighbor_repr[dst_idx] = torch.maximum(neighbor_repr[dst_idx], src_features[i])
        else:
            neighbor_repr = aggregated
        
        combined = torch.cat([x, neighbor_repr], dim=1)
        
        output = self.linear(combined)
        output = self.dropout(output)
        
        return output


class GNNBlock(nn.Module):
    """
    Single GNN block with normalization and activation.
    """
    
    def __init__(self, gnn_layer, norm_type='batch'):
        super().__init__()
        
        self.gnn_layer = gnn_layer
        
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(gnn_layer.out_features)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm(gnn_layer.out_features)
        else:
            self.norm = None
        
        self.activation = nn.ReLU()
    
    def forward(self, x, edge_index):
        residual = x
        
        x = self.gnn_layer(x, edge_index)
        
        if self.norm is not None:
            x = self.norm(x)
        
        x = self.activation(x)
        
        x = x + residual
        
        return x


class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder
    
    Supports multiple GNN architectures: GAT, GCN, SAGE
    """
    
    def __init__(
        self,
        in_features,
        hidden_dim=128,
        out_features=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3,
        gnn_type='gat'
    ):
        super().__init__()
        
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == 'gat':
                layer = GraphAttentionLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            elif gnn_type == 'gcn':
                layer = GraphConvLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    dropout=dropout
                )
            elif gnn_type == 'sage':
                layer = SAGEConvLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    dropout=dropout,
                    aggr='mean'
                )
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            
            self.layers.append(GNNBlock(layer))
        
        self.output_proj = nn.Linear(hidden_dim, out_features)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [N, in_features]
            edge_index: Graph connectivity [2, E]
        
        Returns:
            Graph-encoded node features [N, out_features]
        """
        x = self.input_proj(x)
        
        for layer in self.layers:
            x = layer(x, edge_index)
        
        x = self.output_proj(x)
        
        return x


def create_gnn_encoder(config):
    """
    Factory function to create GNN encoder from config.
    """
    return GNNEncoder(
        in_features=config.get('gnn_input_dim', 96),
        hidden_dim=config.get('gnn_hidden_dim', 128),
        out_features=config.get('gnn_output_dim', 128),
        num_layers=config.get('gnn_num_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('gnn_dropout', 0.3),
        gnn_type=config.get('gnn_type', 'gat')
    )
