"""
Configuration for training phase
"""

from pathlib import Path
import torch


BASE_DIR = Path(__file__).parent.parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
SEQUENCES_DIR = PROCESSED_DIR / "sequences"

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"


MODEL_CONFIG = {
    'mamba_hidden_dim': 64,
    'mamba_output_dim': 64,
    'mamba_num_layers': 2,
    
    'gnn_hidden_dim': 128,
    'gnn_num_layers': 2,
    'gnn_dropout': 0.3,
    'gnn_type': 'gat',
    
    'num_heads': 4,
    'classifier_hidden_dim': 128,
    'dropout': 0.3,
    
    'num_edge_features': 1,
    'sequence_length': 50,
    'num_flows': 2,
    'feature_dim': 96,
    
    'num_classes': 2,
    
    'use_mamba': True,
    'use_gnn': True,
    'mamba_pooling': 'mean'
}


TRAINING_CONFIG = {
    'learning_rate': 0.0001,
    'weight_decay': 0.0001,
    'optimizer': 'adamw',
    
    'num_epochs': 80,
    'batch_size': 192,
    
    'early_stopping_patience': 20,
    'early_stopping_metric': 'f1',
    
    'use_scheduler': True,
    'scheduler_type': 'reduce_on_plateau',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'min_lr': 1e-7,
    
    'use_gradient_clipping': True,
    'max_grad_norm': 0.5,
    
    'use_amp': True,
    'num_workers': 4,
}


LOSS_CONFIG = {
    'loss_type': 'focal',
    'class_weights': [1.0, 50.0],
    'threshold': 0.2, # Khuyên dùng threshold thấp để tăng Recall
    'focal_gamma': 2.0,
    'focal_alpha': 0.5,
}


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def get_class_weights():
    weights = LOSS_CONFIG['class_weights']
    return torch.tensor(weights, dtype=torch.float32)


def create_directories():
    dirs = [CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
