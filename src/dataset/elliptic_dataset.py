"""
Elliptic2 Dataset for AML Detection
PyTorch Geometric dataset implementation
"""

import os
import pickle
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset


class EllipticDataset(Dataset):
    """
    Elliptic2 dataset for AML detection using PyTorch Geometric.

    This dataset loads processed data from the 4-phase pipeline:
    - Graph structure: edge_index, edge_attr
    - Node features: sequences (in_flow, out_flow)
    - Labels: licit/suspicious
    - Train/Val/Test splits
    """

    def __init__(
        self,
        root: str,
        graph_dir: str,
        sequences_dir: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Args:
            root: Root directory
            graph_dir: Directory containing graph files (from Phase 4)
            sequences_dir: Directory containing sequence files (from Phase 3)
            transform: Transform applied to data
            pre_transform: Transform applied before saving
            pre_filter: Filter before saving
        """
        self.graph_dir = Path(graph_dir)
        self.sequences_dir = Path(sequences_dir)

        super().__init__(root, transform, pre_transform, pre_filter)

    def len(self):
        """Return the number of graphs (always 1 for this dataset)."""
        return 1

    def get(self, idx):
        """
        Get the single graph data object.

        Returns:
            PyG Data object with:
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 1]
                - x: Node features [num_nodes, 2, K, F]
                - y: Labels [num_nodes]
                - train_mask: Training mask [num_nodes]
                - val_mask: Validation mask [num_nodes]
                - test_mask: Test mask [num_nodes]
        """
        # Load graph structure
        edge_index = np.load(self.graph_dir / "edge_index.npy")
        edge_attr = np.load(self.graph_dir / "edge_attr.npy")

        # Load train/val/test splits
        with open(self.graph_dir / "train_val_test_split.pkl", "rb") as f:
            splits = pickle.load(f)

        # Load metadata
        with open(self.graph_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        num_nodes = metadata["num_nodes"]

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[torch.tensor(splits["train"])] = True
        val_mask[torch.tensor(splits["val"])] = True
        test_mask[torch.tensor(splits["test"])] = True

        # Load node features from sequences
        print("Loading node features from sequences...")
        node_features = self._load_sequences(num_nodes)

        # Create labels (for labeled nodes only)
        # Load labels from sequences (stored as 'label' in each npz file)
        labels = self._load_labels(num_nodes)

        # Create PyG Data object
        data = Data(
            x=node_features,  # [N, 2, K, F] = [N, 2, 50, 96]
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor(labels, dtype=torch.long),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
            num_edges=edge_index.shape[1],
        )

        return data

    def _load_sequences(self, num_nodes):
        """
        Load node features from sequence files.

        Args:
            num_nodes: Total number of nodes

        Returns:
            Node features tensor [num_nodes, 2, K, F]
        """
        K = 50  # Sequence length
        F = 96  # Feature dimension (95 features + 1 timestamp)

        # Initialize features array
        features = np.zeros((num_nodes, 2, K, F), dtype=np.float32)

        # Load sequence files
        for i in range(num_nodes):
            seq_file = self.sequences_dir / f"node_{i:06d}.npz"

            if seq_file.exists():
                data = np.load(seq_file)
                in_flow = data["in_flow"]  # [K, F]
                out_flow = data["out_flow"]  # [K, F]

                features[i, 0] = in_flow
                features[i, 1] = out_flow

            # Progress logging
            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i + 1:,} / {num_nodes:,} nodes")

        return torch.tensor(features, dtype=torch.float32)

    def _load_labels(self, num_nodes):
        """
        Load labels from sequence files.

        Args:
            num_nodes: Total number of nodes

        Returns:
            Labels tensor [num_nodes]
        """
        labels = np.zeros(num_nodes, dtype=np.int64)

        for i in range(num_nodes):
            seq_file = self.sequences_dir / f"node_{i:06d}.npz"

            if seq_file.exists():
                data = np.load(seq_file)
                labels[i] = data["label"]

        return labels


class EllipticInMemoryDataset(InMemoryDataset):
    """
    In-memory version of Elliptic dataset for faster loading.
    Use this when dataset fits in memory.
    """

    def __init__(
        self, graph_dir: str, sequences_dir: str, transform: Optional[Callable] = None
    ):
        """
        Args:
            graph_dir: Directory containing graph files (from Phase 4)
            sequences_dir: Directory containing sequence files (from Phase 3)
            transform: Transform applied to data
        """
        self.graph_dir = Path(graph_dir)
        self.sequences_dir = Path(sequences_dir)

        # Load all data
        self.data, self.slices = self._load_data()

        super().__init__(None, transform)

    def _load_data(self):
        """Load all data into memory."""
        # Load graph structure
        print("Loading graph structure...")
        edge_index = np.load(self.graph_dir / "edge_index.npy")
        edge_attr = np.load(self.graph_dir / "edge_attr.npy")

        # Load splits
        with open(self.graph_dir / "train_val_test_split.pkl", "rb") as f:
            splits = pickle.load(f)

        # Load metadata
        with open(self.graph_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        num_nodes = metadata["num_nodes"]
        num_edges = edge_index.shape[1]

        print(f"  Graph: {num_nodes} nodes, {num_edges} edges")

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[torch.tensor(splits["train"])] = True
        val_mask[torch.tensor(splits["val"])] = True
        test_mask[torch.tensor(splits["test"])] = True

        # Load features and labels
        print("Loading node features...")
        node_features, labels = self._load_features_and_labels(num_nodes)

        # Create Data object
        data = Data(
            x=node_features,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=num_nodes,
            num_edges=num_edges,
        )

        return data, None

    def _load_features_and_labels(self, num_nodes):
        """Load features and labels from sequence files."""
        K = 50
        F = 96

        features = np.zeros((num_nodes, 2, K, F), dtype=np.float32)
        labels = np.zeros(num_nodes, dtype=np.int64)

        for i in range(num_nodes):
            seq_file = self.sequences_dir / f"node_{i:06d}.npz"

            if seq_file.exists():
                data = np.load(seq_file)
                features[i, 0] = data["in_flow"]
                features[i, 1] = data["out_flow"]
                labels[i] = data["label"]

            if (i + 1) % 50000 == 0:
                print(f"  Loaded {i + 1:,} / {num_nodes:,} nodes")

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.long
        )


def load_elliptic_dataset(graph_dir, sequences_dir, in_memory=True):
    """
    Factory function to load Elliptic dataset.

    Args:
        graph_dir: Path to graph directory
        sequences_dir: Path to sequences directory
        in_memory: If True, load all into memory (faster but uses more RAM)

    Returns:
        PyG Data object
    """
    if in_memory:
        dataset = EllipticInMemoryDataset(graph_dir, sequences_dir)
    else:
        dataset = EllipticDataset(
            root=".", graph_dir=graph_dir, sequences_dir=sequences_dir
        )
        dataset = dataset[0]  # Get the single graph

    return dataset


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

    print(
        f"  Train: {train_nodes:,} ({train_licit:,} licit, {train_suspicious:,} suspicious)"
    )
    print(
        f"  Val:   {val_nodes:,} ({val_licit:,} licit, {val_suspicious:,} suspicious)"
    )
    print(
        f"  Test:  {test_nodes:,} ({test_licit:,} licit, {test_suspicious:,} suspicious)"
    )
