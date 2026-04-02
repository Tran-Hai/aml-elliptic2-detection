# AML Detection on Elliptic2 Dataset

A deep learning project for anti-money laundering (AML) detection in Bitcoin transactions using the Elliptic2 dataset with a hybrid LAS-Mamba-GNN architecture.

---

## Overview

This project implements a state-of-the-art solution to detect suspicious Bitcoin transactions potentially related to money laundering activities. The approach combines three powerful techniques:

- **LAS (Liquidity Analysis)**: Analyzes liquidity patterns from transaction flow sequences
- **Mamba (State Space Models)**: Captures long-range temporal dependencies in transaction history
- **GNN (Graph Neural Networks)**: Learns structural patterns in the transaction network

### Key Features

- ✅ Hybrid architecture combining sequence modeling (LAS + Mamba) and graph learning (GNN)
- ✅ Handles severe class imbalance (41:1 ratio) with weighted loss functions
- ✅ Supports multiple GNN variants: GAT, GCN, GraphSAGE
- ✅ Mini-batch training with local edge indexing for large-scale graphs
- ✅ Memory-efficient lazy loading with DataLoader support
- ✅ Comprehensive evaluation metrics (F1, AUC-ROC, AUC-PR)
- ✅ Mixed precision (AMP) training support for GPU

---

## Dataset

### Elliptic2 Dataset

The Elliptic2 dataset is a comprehensive Bitcoin transaction dataset for blockchain analytics and AML research.

| Component | Count |
|-----------|-------|
| Labeled Nodes | 444,521 |
| Edges | 734,274 |
| Connected Components | 121,810 |
| Features | 96 dimensions |

### Class Distribution

| Class | Count | Ratio |
|-------|-------|-------|
| Licit (legitimate) | 434,055 | 97.65% |
| Suspicious | 10,466 | 2.35% |

---

## Quick Start

### Training

```bash
# Training with full LAS-Mamba-GNN
python -m src.training.train \
    --device cuda \
    --batch-size 256 \
    --epochs 100

# Training without GNN
python -m src.training.train \
    --device cpu \
    --batch-size 64 \
    --epochs 100 \
    --no-gnn
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device to use: cuda/cpu/auto | auto |
| `--batch-size` | Batch size | 256 |
| `--epochs` | Number of epochs | 100 |
| `--lr` | Learning rate | 0.0001 |
| `--num-workers` | DataLoader workers | 4 |
| `--no-gnn` | Disable GNN layer | False |
| `--seed` | Random seed | 42 |

### Evaluation

```bash
python -m src.training.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --split test
```

### Inference

```bash
python -m src.training.infer \
    --checkpoint checkpoints/best_model.pt \
    --node-idx 12345
```

---

## Data Processing Pipeline

The project implements a 4-phase data processing pipeline to convert raw Elliptic2 data into training-ready format.

### Phase 1: Build Index

Creates lookup tables for efficient data access:

```bash
python -m src.data_processing.phase1_build_index
```

Output: `data/processed/index/`
- `node_to_idx.pkl` - Node ID to index mapping
- `idx_to_node.pkl` - Reverse mapping
- `node_labels.pkl` - Node labels
- `component_to_nodes.pkl` - Component membership

### Phase 2: Extract Features

Extracts 95 features from transaction data:

```bash
python -m src.data_processing.phase2_extract_features
```

### Phase 3: Build Sequences

Creates temporal sequences for each node (K=50 transactions):

```bash
python -m src.data_processing.phase3_build_sequences
```

### Phase 4: Build Graph

Prepares graph structure for GNN training:

```bash
python -m src.data_processing.phase4_build_graph
```

Output: `data/processed/graph/`
- `edge_index.npy` - Edge connectivity
- `edge_attr.npy` - Edge features
- `train_val_test_split.pkl` - Data splits

---

## Usage

### Training

```bash
# Training with full LAS-Mamba-GNN (recommended)
python -m src.training.train \
    --device cuda \
    --batch-size 256 \
    --epochs 100

# Training without GNN (faster on CPU, if needed)
python -m src.training.train \
    --device cpu \
    --batch-size 64 \
    --epochs 100 \
    --no-gnn
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--device` | Device to use: cuda/cpu/auto | auto |
| `--batch-size` | Batch size | 256 |
| `--epochs` | Number of epochs | 100 |
| `--lr` | Learning rate | 0.0001 |
| `--num-workers` | DataLoader workers | 4 |
| `--no-gnn` | Disable GNN layer | False |
| `--seed` | Random seed | 42 |

### Evaluation

```bash
python -m src.training.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --split test
```

### Inference

```bash
python -m src.training.infer \
    --checkpoint checkpoints/best_model.pt \
    --node-idx 12345
```

---

## Model Architecture

### LAS-Mamba-GNN

The model consists of three parallel branches that process different aspects of the data:

```
Input: [N, 2, K, F]  (N nodes, 2 flows, K timesteps, F features)
           |
    ┌──────┼──────┐
    |      |      |
   LAS   Mamba   GNN
    |      |      |
    └──────┼──────┘
           |
        Fusion
           |
       Classifier
           |
    Output: [N, 2]  (logits for licit/suspicious)
```

### Components

1. **LAS Layer**: Pooling-based aggregation of in/out flow sequences
2. **Mamba Layer**: State Space Model for sequential encoding
3. **GNN Layer**: Graph convolution (GAT/GCN/SAGE)
4. **Fusion Module**: Combines all representations
5. **Classifier**: Final prediction layer

### Loss Functions

The project uses Weighted Cross Entropy Loss for handling class imbalance:
- Class weights: [1.0, 41.47] (licit:suspicious ratio)

---

## Configuration

Model and training parameters are configured in `src/utils/config.py`:

```python
MODEL_CONFIG = {
    'las_hidden_dim': 64,
    'mamba_hidden_dim': 64,
    'gnn_hidden_dim': 128,
    'gnn_type': 'gat',
    'num_heads': 4,
    'dropout': 0.3,
}

TRAINING_CONFIG = {
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'num_epochs': 100,
    'batch_size': 256,
    'early_stopping_patience': 15,
    'use_amp': True,
}

LOSS_CONFIG = {
    'loss_type': 'weighted_ce',
    'class_weights': [1.0, 41.47],
}
```

---

## Project Structure

```
aml-elliptic2-detection/
├── data/
│   ├── raw/                   # Original Elliptic2 data
│   └── processed/             # Preprocessed data
│       ├── index/             # Phase 1: Lookup tables
│       ├── features/          # Phase 2: Extracted features
│       ├── sequences/         # Phase 3: Temporal sequences
│       └── graph/            # Phase 4: Graph structure
├── src/
│   ├── data_processing/       # Data pipeline scripts
│   │   ├── phase1_build_index.py
│   │   ├── phase2_extract_features.py
│   │   ├── phase3_build_sequences.py
│   │   └── phase4_build_graph.py
│   ├── dataset/              # Dataset loaders
│   │   └── elliptic_dataset.py
│   ├── models/               # Model architectures
│   │   ├── las_layer.py
│   │   ├── mamba_layer.py
│   │   ├── gnn_layer.py
│   │   ├── las_mamba_gnn.py
│   │   └── loss.py
│   ├── training/             # Training scripts
│   │   ├── trainer.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── infer.py
│   └── utils/                # Utilities
│       ├── config.py
│       └── metrics.py
├── checkpoints/             # Model checkpoints
├── logs/                   # Training logs
├── results/                # Evaluation results
└── requirements.txt        # Python dependencies
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)

### Python Dependencies

```
torch>=2.0.0
torch-geometric>=2.3.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
tqdm>=4.64.0
```

---


## References

- [Elliptic2 Dataset](https://www.elliptic.co/) - Blockchain transaction dataset for AML research
- [Mamba: State Space Models for Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

