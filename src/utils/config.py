"""
Configuration for training phase
"""

from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_DIR = PROCESSED_DIR / "graph"
SEQUENCES_DIR = PROCESSED_DIR / "sequences"

CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"


MODEL_CONFIG = {
    'feature_dim': 96,
    'mamba_hidden_dim': 64,
    'mamba_num_layers': 2,
    'classifier_hidden_dim': 128,
    'dropout': 0.3,
    'num_classes': 2,
}


TRAINING_CONFIG = {
    'learning_rate': 0.0005,
    'weight_decay': 0.0001,
    'optimizer': 'adam',
    'num_epochs': 100,
    'batch_size': 128,
    'early_stopping_patience': 100,
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
    'class_weights': [1.0, 5000.0],
    'threshold': 0.15,
    'focal_gamma': 1.0,
    'focal_alpha': 0.75,
}


DATA_CONFIG = {
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'random_seed': 42,
    'stratified': True,
}


DATASET_CONFIG = {
    'num_nodes': 444521,
    'num_edges': 367137,
    'num_classes': 2,
    'sequence_length': 50,
    'feature_dim': 96,
}


LOGGING_CONFIG = {
    'log_interval': 10,
    'eval_interval': 1,
    'save_checkpoint_interval': 5,
    'save_best_only': True,
}


DEVICE_CONFIG = {
    'device': 'auto',
    'num_workers': 4,
    'pin_memory': True,
}


REPRODUCIBILITY_CONFIG = {
    'random_seed': 42,
    'deterministic': True,
    'benchmark': False,
}


def get_device():
    import torch
    if DEVICE_CONFIG['device'] != 'auto':
        return DEVICE_CONFIG['device']
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_class_weights():
    import torch
    weights = LOSS_CONFIG['class_weights']
    return torch.tensor(weights, dtype=torch.float32)


def create_directories():
    dirs = [CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)