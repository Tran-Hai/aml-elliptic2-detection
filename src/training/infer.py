"""
Inference script for LAS-Mamba-GNN model
Predict for a single node
"""

import argparse
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset.elliptic_dataset import load_elliptic_dataset
from src.models.las_mamba_gnn import create_las_mamba_gnn
from src.utils.config import MODEL_CONFIG, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for a single Elliptic2 node")
    parser.add_argument(
        "--graph-dir",
        type=str,
        default="data/processed/graph",
        help="Path to graph directory"
    )
    parser.add_argument(
        "--sequences-dir",
        type=str,
        default="data/processed/sequences",
        help="Path to sequences directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pt)"
    )
    parser.add_argument(
        "--node-idx",
        type=int,
        required=True,
        help="Node index to predict"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use"
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    """Resolve path relative to root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def get_split_name(data, idx: int) -> str:
    """Get which split the node belongs to."""
    if data.train_mask[idx]:
        return "train"
    if data.val_mask[idx]:
        return "val"
    if data.test_mask[idx]:
        return "test"
    return "unknown"


def main() -> None:
    args = parse_args()
    
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    graph_dir = resolve_path(ROOT_DIR, args.graph_dir)
    sequences_dir = resolve_path(ROOT_DIR, args.sequences_dir)
    checkpoint_path = resolve_path(ROOT_DIR, args.checkpoint)
    
    print(f"Loading dataset from:")
    print(f"  Graph: {graph_dir}")
    print(f"  Sequences: {sequences_dir}")
    
    data = load_elliptic_dataset(str(graph_dir), str(sequences_dir))
    data = data.to(device)
    
    num_nodes = data.y.shape[0]
    if args.node_idx < 0 or args.node_idx >= num_nodes:
        raise ValueError(f"node-idx must be in [0, {num_nodes - 1}]")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = create_las_mamba_gnn(MODEL_CONFIG).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Predicting for node {args.node_idx}...")
    
    with torch.no_grad():
        node_features = data.x
        sequences = data.x
        edge_index = data.edge_index
        
        logits = model(node_features, sequences, edge_index)
        probs = torch.softmax(logits, dim=1)
        
        pred = int(torch.argmax(probs[args.node_idx]).item())
        suspicious_prob = float(probs[args.node_idx, 1].item())
        licit_prob = float(probs[args.node_idx, 0].item())
        true_label = int(data.y[args.node_idx].item())
    
    pred_name = "suspicious" if pred == 1 else "licit"
    true_name = "suspicious" if true_label == 1 else "licit"
    split_name = get_split_name(data, args.node_idx)
    
    print(f"\nResults:")
    print(f"  Node index: {args.node_idx} ({split_name} split)")
    print(f"  Prediction: {pred} ({pred_name})")
    print(f"  Probability: licit={licit_prob:.6f}, suspicious={suspicious_prob:.6f}")
    print(f"  True label: {true_label} ({true_name})")
    
    if pred == true_label:
        print(f"  ✓ Correct prediction!")
    else:
        print(f"  ✗ Incorrect prediction")


if __name__ == "__main__":
    main()
