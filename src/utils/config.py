"""
Configuration for training phase
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
SEQUENCES_DIR = PROCESSED_DIR / "sequences"

# Output paths
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"


# MODEL HYPERPARAMETERS


MODEL_CONFIG = {
    # LAS Layer
    "las_hidden_dim": 64,
    "las_output_dim": 64,
    # Mamba Layer
    "mamba_hidden_dim": 64,
    "mamba_output_dim": 64,
    "mamba_num_layers": 2,
    # GNN Layer
    "gnn_hidden_dim": 128,
    "gnn_num_layers": 2,
    "gnn_dropout": 0.3,
    "gnn_type": "gat",  # 'gcn', 'gat', 'sage'
    # Attention for GAT
    "num_heads": 4,
    # Final classifier
    "classifier_hidden_dim": 128,
    "dropout": 0.3,
    # Feature dimensions
    "num_edge_features": 1,  # timestamp_proxy
    "sequence_length": 50,
    "num_flows": 2,  # in_flow + out_flow
    "feature_dim": 96,  # 95 features + 1 timestamp
    # Number of classes
    "num_classes": 2,
}


# TRAINING HYPERPARAMETERS


TRAINING_CONFIG = {
    # Learning
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "optimizer": "adam",  # 'adam', 'adamw', 'sgd'
    # Training loop
    "num_epochs": 100,
    "batch_size": 256,
    # Early stopping
    "early_stopping_patience": 15,
    "early_stopping_metric": "val_f1",  # 'val_f1', 'val_auc_roc', 'val_loss'
    # Learning rate scheduler
    "use_scheduler": True,
    "scheduler_type": "reduce_on_plateau",  # 'reduce_on_plateau', 'cosine', 'step'
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "min_lr": 1e-6,
    # Gradient clipping
    "use_gradient_clipping": True,
    "max_grad_norm": 1.0,
    # Mixed precision training
    "use_amp": False,  # Automatic Mixed Precision
}


# LOSS CONFIGURATION

LOSS_CONFIG = {
    "loss_type": "weighted_ce",
    "class_weights": [1.0, 41.47],
}


# DATA SPLIT CONFIGURATION


DATA_CONFIG = {
    # These should match Phase 4 output
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    "stratified": True,
}


# DATASET CONFIGURATION


DATASET_CONFIG = {
    "num_nodes": 444521,
    "num_edges": 367137,
    "num_classes": 2,
    # Class distribution (for reference)
    "class_counts": {
        "train": {"licit": 304898, "suspicious": 6390},
        "val": {"licit": 65278, "suspicious": 1371},
        "test": {"licit": 64879, "suspicious": 1371},
    },
    # Sequence config
    "sequence_length": 50,
    "feature_dim": 96,
    # Graph config
    "edge_feature_dim": 1,  # timestamp_proxy
}


# LOGGING AND CHECKPOINTING


LOGGING_CONFIG = {
    "log_interval": 10,  # Log every N batches
    "eval_interval": 1,  # Evaluate every N epochs
    "save_checkpoint_interval": 5,  # Save checkpoint every N epochs
    # What to save
    "save_best_only": True,
    "save_optimizer": True,
    "save_scheduler": True,
    # Metrics to track
    "metrics_to_track": [
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc_roc",
        "val_auc_pr",
    ],
}


# DEVICE CONFIGURATION


DEVICE_CONFIG = {
    # 'cuda', 'cpu', 'mps' (for Apple Silicon)
    "device": "auto",  # Will auto-detect
    # Number of workers for dataloader
    "num_workers": 4,
    # Pin memory for faster data transfer
    "pin_memory": True,
}


# REPRODUCIBILITY


REPRODUCIBILITY_CONFIG = {
    "random_seed": 42,
    "deterministic": True,
    "benchmark": False,
}


# HELPER FUNCTIONS


def get_device():
    """Get the best available device."""
    if DEVICE_CONFIG["device"] != "auto":
        return DEVICE_CONFIG["device"]

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_class_weights():
    """Get class weights for loss function."""
    weights = LOSS_CONFIG["class_weights"]

    # Support both list and dict format
    if isinstance(weights, dict):
        return torch.tensor(
            [weights["licit"], weights["suspicious"]], dtype=torch.float32
        )
    else:
        return torch.tensor(weights, dtype=torch.float32)


def create_directories():
    """Create necessary directories."""
    dirs = [CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Import torch at the end to avoid circular import
import torch
