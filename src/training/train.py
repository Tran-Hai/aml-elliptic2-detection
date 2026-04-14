"""
Training script for LAS-Mamba-GNN model
Optimized for GPU training with efficient data loading
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset.elliptic_dataset import load_elliptic_dataset, FastEllipticDataset
from src.models.las_mamba_gnn import create_las_mamba_gnn
from src.models.loss import get_loss_function
from src.training.trainer import Trainer, create_optimizer, create_scheduler, OptimizedTrainer, get_local_edge_index
from src.utils.config import (
    MODEL_CONFIG,
    TRAINING_CONFIG,
    LOSS_CONFIG,
    DATA_CONFIG,
    get_device
)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LAS-Mamba-GNN on Elliptic2")
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
        "--index-dir",
        type=str,
        default="data/processed/index",
        help="Path to index directory"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no-gnn",
        action="store_true",
        help="Disable GNN (use only LAS + Mamba)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile optimization"
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    """Resolve path relative to root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def main() -> None:
    args = parse_args()
    
    set_seed(args.seed)
    
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    graph_dir = resolve_path(ROOT_DIR, args.graph_dir)
    sequences_dir = resolve_path(ROOT_DIR, args.sequences_dir)
    index_dir = resolve_path(ROOT_DIR, args.index_dir)
    
    print(f"Loading dataset from:")
    print(f"  Graph: {graph_dir}")
    print(f"  Sequences: {sequences_dir}")
    print(f"  Index: {index_dir}")
    
    train_dataset = FastEllipticDataset(
        graph_dir=str(graph_dir),
        sequences_dir=str(sequences_dir),
        index_dir=str(index_dir),
        split='train'
    )
    
    val_dataset = FastEllipticDataset(
        graph_dir=str(graph_dir),
        sequences_dir=str(sequences_dir),
        index_dir=str(index_dir),
        split='val'
    )
    
    test_dataset = FastEllipticDataset(
        graph_dir=str(graph_dir),
        sequences_dir=str(sequences_dir),
        index_dir=str(index_dir),
        split='test'
    )
    
    from torch.utils.data import DataLoader
    from src.training.trainer import OptimizedTrainer
    
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": FastEllipticDataset.collate_fn
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size or TRAINING_CONFIG['batch_size'],
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size or TRAINING_CONFIG['batch_size'],
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size or TRAINING_CONFIG['batch_size'],
        shuffle=False,
        **loader_kwargs
    )
    
    edge_index = train_dataset.edge_index.to(device)
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val: {len(val_dataset):,}")
    print(f"  Test: {len(test_dataset):,}")
    print(f"  Device: {device}")
    print(f"  Batch size: {args.batch_size or TRAINING_CONFIG['batch_size']}")
    
    num_epochs = args.epochs if args.epochs is not None else TRAINING_CONFIG['num_epochs']
    batch_size = args.batch_size if args.batch_size is not None else TRAINING_CONFIG['batch_size']
    learning_rate = args.lr if args.lr is not None else TRAINING_CONFIG['learning_rate']
    
    train_config = {
        **TRAINING_CONFIG,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'batch_size': batch_size
    }
    
    model_config = {
        **MODEL_CONFIG,
        'use_gnn': not args.no_gnn
    }
    
    print(f"\nCreating model...")
    model = create_las_mamba_gnn(model_config).to(device)
    if device.type == "cuda" and not args.no_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    print(f"  Using GNN: {not args.no_gnn}")
    
    class_weights = torch.tensor(LOSS_CONFIG['class_weights'], dtype=torch.float32, device=device)
    
    class_weights = torch.tensor(LOSS_CONFIG['class_weights'], dtype=torch.float32, device=device)
    criterion = get_loss_function(
        loss_type=LOSS_CONFIG['loss_type'],
        class_weights=class_weights,
        focal_gamma=LOSS_CONFIG.get('focal_gamma', 2.0)
    )
    
    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config)
    
    print(f"\nTraining configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Loss: {LOSS_CONFIG['loss_type']}")
    print(f"  Class weights: {LOSS_CONFIG['class_weights']}")
    print(f"  AMP enabled: {TRAINING_CONFIG.get('use_amp', False)}")
    print(f"  Workers: {args.num_workers}")
    
    trainer = OptimizedTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        grad_clip_norm=TRAINING_CONFIG.get('max_grad_norm', 1.0),
        use_amp=bool(TRAINING_CONFIG.get('use_amp', True) and device.type == 'cuda'),
        print_fn=print
    )
    
    print("Starting training...")
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    results = trainer.train_with_loaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        edge_index=edge_index,
        num_epochs=num_epochs,
        val_interval=1,
        early_stopping_patience=TRAINING_CONFIG.get('early_stopping_patience', 15),
        early_stopping_metric=TRAINING_CONFIG.get('early_stopping_metric', 'val_f1'),
        checkpoint_dir=ROOT_DIR / "checkpoints",
        best_model_name="best_model.pt",
        verbose=True
    )

    print("Training completed!")
    print(f"Best validation metric: {results['best_val_metric']:.4f}")

    
    print(f"\nEvaluating on test set...")
    checkpoint_dir = ROOT_DIR / "checkpoints"
    best_model_path = checkpoint_dir / "best_model.pt"
    
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = trainer.evaluate_loader(test_loader, edge_index)
        
        print(f"\nTest metrics:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC: {test_metrics.get('auc_roc', 0.0):.4f}")
        print(f"  AUC-PR: {test_metrics.get('auc_pr', 0.0):.4f}")
    else:
        print(f"No checkpoint found at {best_model_path}")


if __name__ == "__main__":
    main()
