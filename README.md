# AML Detection on Elliptic2 Dataset

A machine learning project for anti-money laundering (AML) detection in Bitcoin transactions using the Elliptic2 dataset with a hybrid LAS-Mamba-GNN architecture.

---

## Overview

This project implements a deep learning solution to detect suspicious Bitcoin transactions potentially related to money laundering activities. The approach combines:

- **Graph Neural Networks (GNN)** for learning structural patterns in transaction networks
- **Mamba (State Space Models)** for capturing temporal dependencies
- **LAS (Liquidity Analysis)** for analyzing liquidity patterns

## Dataset

### Elliptic2 Dataset

The Elliptic2 dataset is a comprehensive Bitcoin transaction dataset for blockchain analytics and AML research.

| Component | Count |
|-----------|-------|
| Labeled Nodes | 444,521 |
| Edges | 367,137 |
| Connected Components | 121,810 |
| Background Nodes | ~49M |
| Background Edges | ~196M |

### Class Distribution

| Class | Count | Ratio |
|-------|-------|-------|
| Licit (legitimate) | 434,055 | 97.65% |
| Suspicious | 10,466 | 2.35% |

**Note**: The dataset exhibits significant class imbalance (41:1 ratio), requiring special handling during model training.

## Project Structure

```
aml-elliptic2-detection/
├── data/
│   ├── raw/              # Original Elliptic2 data
│   │   ├── nodes.csv
│   │   ├── edges.csv
│   │   ├── connected_components.csv
│   │   ├── background_nodes.csv
│   │   └── background_edges.csv
│   └── processed/        # Preprocessed data
│       ├── index/        # Phase 1: Lookup tables
│       ├── features/     # Phase 2: Extracted features
│       ├── sequences/    # Phase 3: Temporal sequences
│       └── graph/        # Phase 4: Graph structure
├── src/
│   ├── data_processing/  # Data pipeline scripts
│   │   ├── phase1_build_index.py
│   │   ├── phase2_extract_features.py
│   │   ├── phase3_build_sequences.py
│   │   └── phase4_build_graph.py
│   └── models/          # Model architectures
├── notebooks/            # Jupyter notebooks (EDA)
├── docs/                # Documentation
├── tests/               # Unit tests
└── requirements.txt     # Python dependencies
```

## Data Processing Pipeline

The project implements a 4-phase data processing pipeline:

### Phase 1: Build Index
Creates lookup tables for efficient data access:
- `node_to_idx.pkl` - Node ID to index mapping
- `idx_to_node.pkl` - Reverse mapping
- `node_labels.pkl` - Node labels (licit/suspicious)
- `component_to_nodes.pkl` - Component membership
- `edges_index.pkl` - Adjacency list

### Phase 2: Extract Features
Extracts 95 features from the 78GB background edges file:
- Streaming processing with chunk size of 50,000 rows
- Generates per-node feature files for in-flow and out-flow transactions

### Phase 3: Build Sequences
Creates temporal sequences for each node:
- Sequence length: K=50 transactions
- Features: 96 dimensions (95 edge features + 1 timestamp proxy)
- Output format: NPZ files with shape (2, 50, 96)

### Phase 4: Build Graph
Prepares graph structure for GNN:
- `edge_index.npy` - Edge connectivity [2, num_edges]
- `edge_attr.npy` - Edge features [num_edges, 1]
- Train/Val/Test splits (70/15/15) with stratification

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the data processing pipeline sequentially:

```bash
# Phase 1: Build Index
python src/data_processing/phase1_build_index.py

# Phase 2: Extract Features (requires background_edges.csv)
python src/data_processing/phase2_extract_features.py

# Phase 3: Build Sequences
python src/data_processing/phase3_build_sequences.py

# Phase 4: Build Graph
python src/data_processing/phase4_build_graph.py
```

## Model Architecture

The LAS-Mamba-GNN model combines:

1. **Liquidity Analysis (LAS)**: Analyzes transaction flow patterns
2. **Mamba**: State Space Model for sequence modeling
3. **Graph Neural Network**: Captures graph structure and relational patterns



## Requirements

Key dependencies:
- Python 3.10+
- PyTorch
- PyTorch Geometric
- Pandas
- NumPy
- scikit-learn

See `requirements.txt` for full list.

## Documentation

Additional documentation is available in the `docs/` directory:
- Project reports
- Research papers (LAS-GNN, Mamba)


## References

- Elliptic2 Dataset - Blockchain transaction dataset for AML research
- Mamba: State Space Models for Sequence Modeling
- LAS-GNN: Liquidity Analysis with Graph Neural Networks
