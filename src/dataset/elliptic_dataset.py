"""
Elliptic2 Dataset for AML Detection
Optimized version with fast loading and DataLoader support
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Tuple
import os
from concurrent.futures import ThreadPoolExecutor
import threading


class FastEllipticDataset(Dataset):
    """
    Optimized dataset that:
    1. Loads labels from cache (node_labels.pkl) instead of reading 444k files
    2. Supports DataLoader with multiple workers for parallel loading
    3. Uses memory-mapped file access for speed
    """
    
    def __init__(
        self,
        graph_dir: str,
        sequences_dir: str,
        index_dir: Optional[str] = None,
        split: str = 'train',
        transform: Optional[callable] = None
    ):
        self.graph_dir = Path(graph_dir)
        self.sequences_dir = Path(sequences_dir)
        self.index_dir = Path(index_dir) if index_dir else self.graph_dir.parent / 'index'
        self.transform = transform
        
        self._load_metadata()
        self._load_labels_from_cache()
        self._create_split_indices(split)
        
        self._file_cache = {}
        self._cache_lock = threading.Lock()
    
    def _load_metadata(self):
        """Load graph structure efficiently."""
        print("Loading graph structure...")
        
        edge_index = np.load(self.graph_dir / 'edge_index.npy')
        
        with open(self.graph_dir / 'train_val_test_split.pkl', 'rb') as f:
            splits = pickle.load(f)
        
        with open(self.graph_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.num_nodes = metadata['num_nodes']
        self.num_edges = edge_index.shape[1]
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        self.train_indices = splits['train']
        self.val_indices = splits['val']
        self.test_indices = splits['test']
        
        print(f"  Graph: {self.num_nodes:,} nodes, {self.num_edges:,} edges")
    
    def _load_labels_from_cache(self):
        """Load labels from cached pickle file (FAST - <1 second)."""
        print("Loading labels from cache...")
        
        with open(self.index_dir / 'node_to_idx.pkl', 'rb') as f:
            node_to_idx = pickle.load(f)
        
        with open(self.index_dir / 'node_labels.pkl', 'rb') as f:
            node_labels = pickle.load(f)
        
        labels_array = np.zeros(self.num_nodes, dtype=np.int64)
        for node_id, label in node_labels.items():
            if node_id in node_to_idx:
                idx = node_to_idx[node_id]
                if idx < self.num_nodes:
                    labels_array[idx] = label
        
        self.labels = torch.tensor(labels_array, dtype=torch.long)
        print(f"  Loaded {self.labels.sum().item()} suspicious / {(self.labels == 0).sum().item()} licit")
    
    def _create_split_indices(self, split: str):
        """Create indices for train/val/test split."""
        if split == 'train':
            self.indices = self.train_indices
            self.mask = self._create_mask(self.train_indices)
        elif split == 'val':
            self.indices = self.val_indices
            self.mask = self._create_mask(self.val_indices)
        elif split == 'test':
            self.indices = self.test_indices
            self.mask = self._create_mask(self.test_indices)
        elif split == 'all':
            self.indices = np.arange(self.num_nodes)
            self.mask = torch.ones(self.num_nodes, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"  Split '{split}': {len(self.indices):,} samples")
    
    def _create_mask(self, indices):
        """Create boolean mask for indices."""
        mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        mask[torch.tensor(indices)] = True
        return mask
    
    def _load_sequence(self, idx: int) -> np.ndarray:
        """Load a single sequence with caching."""
        with self._cache_lock:
            if idx in self._file_cache:
                return self._file_cache[idx]
        
        seq_file = self.sequences_dir / f'node_{idx:06d}.npz'
        if seq_file.exists():
            data = np.load(seq_file)
            features = np.stack([data['in_flow'], data['out_flow']], axis=0)
        else:
            features = np.zeros((2, 50, 96), dtype=np.float32)
        
        with self._cache_lock:
            if len(self._file_cache) < 1000:
                self._file_cache[idx] = features
        
        return features
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, local_idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get item for DataLoader.
        
        Returns:
            features: [2, 50, 96] - in_flow and out_flow sequences
            label: scalar - 0 (licit) or 1 (suspicious)
            global_idx: global node index for edge mapping
        """
        global_idx = self.indices[local_idx]
        
        features = self._load_sequence(global_idx)
        label = self.labels[global_idx].item()
        
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label, global_idx
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        """Custom collate function with global indices."""
        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        global_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
        return features, labels, global_indices
    
    def get_batch(self, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of sequences (for custom batching).
        
        Returns:
            features: [batch_size, 2, 50, 96]
            labels: [batch_size]
        """
        batch_features = []
        batch_labels = []
        
        for idx in indices:
            features, label = self[idx]
            batch_features.append(features)
            batch_labels.append(label)
        
        return torch.stack(batch_features), torch.stack(batch_labels)


class EllipticDataLoader:
    """
    Custom DataLoader that handles graph structure properly.
    """
    
    def __init__(
        self,
        dataset: FastEllipticDataset,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        edge_index: Optional[torch.Tensor] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.edge_index = edge_index
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        """Custom collate function with global indices."""
        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        global_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
        return features, labels, global_indices
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)


def load_elliptic_dataset(graph_dir, sequences_dir, split='train', index_dir=None, num_workers=4):
    """
    Factory function to create optimized Elliptic dataset.
    
    Args:
        graph_dir: Path to graph directory
        sequences_dir: Path to sequences directory
        split: 'train', 'val', 'test', or 'all'
        index_dir: Path to index directory (optional)
        num_workers: Number of workers for parallel loading
    
    Returns:
        FastEllipticDataset
    """
    return FastEllipticDataset(
        graph_dir=graph_dir,
        sequences_dir=sequences_dir,
        index_dir=index_dir,
        split=split
    )


def get_data_info(dataset):
    """Print information about the loaded data."""
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    print(f"Number of nodes: {dataset.num_nodes:,}")
    print(f"Number of edges: {dataset.num_edges:,}")
    print(f"Dataset size: {len(dataset):,}")
    print(f"Number of classes: 2")
    print(f"\nClass distribution in dataset:")
    
    labels = dataset.labels[torch.tensor(dataset.indices)]
    licit = (labels == 0).sum().item()
    suspicious = (labels == 1).sum().item()
    
    print(f"  Licit: {licit:,} ({licit/len(labels)*100:.2f}%)")
    print(f"  Suspicious: {suspicious:,} ({suspicious/len(labels)*100:.2f}%)")
    print("=" * 50)