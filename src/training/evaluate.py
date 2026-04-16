"""
Evaluation script for LAS-Mamba-GNN model
"""

import argparse
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset.elliptic_dataset import load_elliptic_dataset
from src.models.mamba_gnn import create_mamba_gnn
from src.models.loss import get_loss_function
from src.utils.config import MODEL_CONFIG, LOSS_CONFIG, get_device
from src.utils.metrics import compute_metrics, print_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LAS-Mamba-GNN checkpoint")
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
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate"
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


def get_mask(data, split: str):
    """Get mask for the specified split."""
    if split == "train":
        return data.train_mask
    if split == "val":
        return data.val_mask
    return data.test_mask


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
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = create_mamba_gnn(MODEL_CONFIG).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Evaluating on {args.split} split...")
    
    mask = get_mask(data, args.split)
    
    with torch.no_grad():
        node_features = data.x
        sequences = data.x
        edge_index = data.edge_index
        
        logits = model(node_features, sequences, edge_index)
        
        mask_indices = torch.where(mask)[0]
        
        probs = torch.softmax(logits[mask_indices], dim=1)[:, 1]
        preds = logits[mask_indices].argmax(dim=1)
    
    metrics = compute_metrics(data.y[mask_indices], preds, probs)
    
    print(f"\nEvaluation Results ({args.split} split):")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics.get('auc_roc', 0.0):.4f}")
    print(f"  AUC-PR: {metrics.get('auc_pr', 0.0):.4f}")
    
    if 'val_metrics' in checkpoint:
        print(f"\nTraining info:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val metrics: {checkpoint['val_metrics']}")


if __name__ == "__main__":
    main()
