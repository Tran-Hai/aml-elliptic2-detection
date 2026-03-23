"""
Elliptic2 Dataset for AML Detection
PyTorch Geometric dataset implementation
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional, Callable


class EllipticInMemoryDataset:
    """In-memory dataset for AML detection."""
    
    def __init__(
        self,
        graph_dir: str,
        sequences_dir: str,
        transform: Optional[Callable] = None
    ):
        self.graph_dir = Path(graph_dir)
        self.sequences_dir = Path(sequences_dir)
        self.transform = transform
        
        self.data = self._load_data()
    
    def _load_data(self):
        print("Loading graph structure...")
        edge_index = np.load(self.graph_dir / 'edge_index.npy')
        edge_attr = np.load(self.graph_dir / 'edge_attr.npy')
        
        with open(self.graph_dir / 'train_val_test_split.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        with open(self.graph_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        num_nodes = metadata['num_nodes']
        num_edges = edge_index.shape[1]
        
        print(f"  Graph: {num_nodes} nodes, {num_edges} edges")
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[torch.tensor(splits['train'])] = True
        val_mask[torch.tensor(splits['val'])] = True
        test_mask[torch.tensor(splits['test'])] = True
        
        print("Loading node features...")
        node_features, labels = self._load_features_and_labels(num_nodes)
        
        data = Data(
            x=node_features,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
            num_edges=num_edges
        )
        
        return data
    
    def _load_features_and_labels(self, num_nodes):
        K, F = 50, 96
        features = np.zeros((num_nodes, 2, K, F), dtype=np.float32)
        labels = np.zeros(num_nodes, dtype=np.int64)
        
        for i in range(num_nodes):
            seq_file = self.sequences_dir / f'node_{i:06d}.npz'
            
            if seq_file.exists():
                data = np.load(seq_file)
                features[i, 0] = data['in_flow']
                features[i, 1] = data['out_flow']
                labels[i] = data['label']
            
            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i + 1:,} / {num_nodes:,} nodes")
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data


def load_elliptic_dataset(graph_dir, sequences_dir, in_memory=True):
    """
    Factory function to load Elliptic dataset.
    
    Args:
        graph_dir: Path to graph directory
        sequences_dir: Path to sequences directory
        in_memory: If True, load all into memory
    
    Returns:
        PyG Data object
    """
    dataset = EllipticInMemoryDataset(graph_dir, sequences_dir)
    return dataset.data


def get_data_info(data):
    """Print information about the loaded data."""
    print("DATASET INFORMATION")
    print(f"Number of nodes: {data.num_nodes:,}")
    print(f"Number of edges: {data.num_edges:,}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Number of classes: {data.y.max().item() + 1}")
    print(f"\nClass distribution:")
    
    train_nodes = data.train_mask.sum().item()
    val_nodes = data.val_mask.sum().item()
    test_nodes = data.test_mask.sum().item()
    
    train_licit = (data.y[data.train_mask] == 0).sum().item()
    train_suspicious = (data.y[data.train_mask] == 1).sum().item()
    
    val_licit = (data.y[data.val_mask] == 0).sum().item()
    val_suspicious = (data.y[data.val_mask] == 1).sum().item()
    
    test_licit = (data.y[data.test_mask] == 0).sum().item()
    test_suspicious = (data.y[data.test_mask] == 1).sum().item()
    
    print(f"  Train: {train_nodes:,} ({train_licit:,} licit, {train_suspicious:,} suspicious)")
    print(f"  Val:   {val_nodes:,} ({val_licit:,} licit, {val_suspicious:,} suspicious)")
    print(f"  Test:  {test_nodes:,} ({test_licit:,} licit, {test_suspicious:,} suspicious)")
