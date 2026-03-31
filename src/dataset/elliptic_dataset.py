"""
Elliptic2 Dataset for AML Detection
Memory-efficient implementation with batch loading
"""

import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import os


class LazyEllipticDataset:
    """
    Memory-efficient dataset that loads sequences on-the-fly.
    """
    
    def __init__(
        self,
        graph_dir: str,
        sequences_dir: str,
        transform: Optional[Callable] = None
    ):
        self.graph_dir = Path(graph_dir)
        self.sequences_dir = Path(sequences_dir)
        self.transform = transform
        
        self._load_graph_structure()
    
    def _load_graph_structure(self):
        """Load graph structure (small, fits in memory)."""
        print("Loading graph structure...")
        
        edge_index = np.load(self.graph_dir / 'edge_index.npy')
        edge_attr = np.load(self.graph_dir / 'edge_attr.npy')
        
        with open(self.graph_dir / 'train_val_test_split.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        with open(self.graph_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.num_nodes = metadata['num_nodes']
        self.num_edges = edge_index.shape[1]
        
        print(f"  Graph: {self.num_nodes} nodes, {self.num_edges} edges")
        
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        
        self.train_indices = np.array(splits['train'])
        self.val_indices = np.array(splits['val'])
        self.test_indices = np.array(splits['test'])
        
        self.labels = self._load_all_labels()
        
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        self.train_mask[torch.tensor(self.train_indices)] = True
        self.val_mask[torch.tensor(self.val_indices)] = True
        self.test_mask[torch.tensor(self.test_indices)] = True
    
    def _load_all_labels(self):
        """Load only labels (small, ~1.7MB)."""
        print("Loading labels...")
        labels = np.zeros(self.num_nodes, dtype=np.int64)
        
        count = 0
        for i in range(self.num_nodes):
            seq_file = self.sequences_dir / f'node_{i:06d}.npz'
            if seq_file.exists():
                data = np.load(seq_file)
                labels[i] = data['label']
                count += 1
        
        print(f"  Loaded {count} labels")
        return torch.tensor(labels, dtype=torch.long)
    
    def load_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a batch of sequences on-the-fly.
        
        Args:
            indices: List of node indices to load
        
        Returns:
            features: [batch_size, 2, K, F]
            labels: [batch_size]
        """
        K, F = 50, 96
        batch_size = len(indices)
        
        features = np.zeros((batch_size, 2, K, F), dtype=np.float32)
        
        for i, idx in enumerate(indices):
            seq_file = self.sequences_dir / f'node_{idx:06d}.npz'
            if seq_file.exists():
                data = np.load(seq_file)
                features[i, 0] = data['in_flow']
                features[i, 1] = data['out_flow']
            else:
                features[i] = np.random.randn(2, K, F).astype(np.float32) * 0.01
        
        labels = self.labels[indices]
        
        return torch.tensor(features, dtype=torch.float32), labels
    
    def get_all_data(self) -> Data:
        """Get full graph data (without full features in memory)."""
        return Data(
            x=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.labels,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges
        )
    
    @property
    def num_classes(self):
        return 2
    
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        return idx


class EllipticInMemoryDataset:
    """
    Original in-memory dataset (kept for backward compatibility).
    WARNING: Uses too much memory for full dataset.
    """
    
    def __init__(
        self,
        graph_dir: str,
        sequences_dir: str,
        transform: Optional[Callable] = None
    ):
        import warnings
        warnings.warn(
            "EllipticInMemoryDataset uses too much memory. "
            "Use LazyEllipticDataset instead.",
            DeprecationWarning
        )
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


def load_elliptic_dataset(graph_dir, sequences_dir, lazy=True):
    """
    Factory function to load Elliptic dataset.
    
    Args:
        graph_dir: Path to graph directory
        sequences_dir: Path to sequences directory
        lazy: If True, use LazyEllipticDataset (memory-efficient)
    
    Returns:
        Dataset object
    """
    if lazy:
        return LazyEllipticDataset(graph_dir, sequences_dir)
    else:
        return EllipticInMemoryDataset(graph_dir, sequences_dir)


def get_data_info(dataset):
    """Print information about the loaded data."""
    print("DATASET INFORMATION")
    print(f"Number of nodes: {dataset.num_nodes:,}")
    print(f"Number of edges: {dataset.num_edges:,}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"\nClass distribution:")
    
    train_nodes = dataset.train_mask.sum().item()
    val_nodes = dataset.val_mask.sum().item()
    test_nodes = dataset.test_mask.sum().item()
    
    train_licit = (dataset.labels[dataset.train_mask] == 0).sum().item()
    train_suspicious = (dataset.labels[dataset.train_mask] == 1).sum().item()
    
    val_licit = (dataset.labels[dataset.val_mask] == 0).sum().item()
    val_suspicious = (dataset.labels[dataset.val_mask] == 1).sum().item()
    
    test_licit = (dataset.labels[dataset.test_mask] == 0).sum().item()
    test_suspicious = (dataset.labels[dataset.test_mask] == 1).sum().item()
    
    print(f"  Train: {train_nodes:,} ({train_licit:,} licit, {train_suspicious:,} suspicious)")
    print(f"  Val:   {val_nodes:,} ({val_licit:,} licit, {val_suspicious:,} suspicious)")
    print(f"  Test:  {test_nodes:,} ({test_licit:,} licit, {test_suspicious:,} suspicious)")