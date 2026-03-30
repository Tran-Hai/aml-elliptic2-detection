"""
Trainer class for LAS-Mamba-GNN model
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Optional, Any
import numpy as np


class Trainer:
    """
    Trainer for LAS-Mamba-GNN model with mini-batch support.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        grad_clip_norm: float = 1.0,
        use_amp: bool = False,
        print_fn=print
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp
        self.print = print_fn
        
        self.scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    def train_epoch(self, data: Any, batch_size: int = 256) -> Dict[str, float]:
        """
        Train for one epoch with mini-batches.
        
        Args:
            data: PyG Data object
            batch_size: Batch size for training
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        train_mask = data.train_mask
        num_train_nodes = train_mask.sum().item()
        
        train_indices = torch.where(train_mask)[0]
        num_batches = (num_train_nodes + batch_size - 1) // batch_size
        
        total_loss = 0.0
        num_batches_processed = 0
        
        indices = train_indices[torch.randperm(num_train_nodes)]
        
        for start in range(0, num_train_nodes, batch_size):
            end = min(start + batch_size, num_train_nodes)
            batch_indices = indices[start:end]
            
            self.optimizer.zero_grad()
            
            node_features = data.x[batch_indices]
            sequences = data.x[batch_indices]
            edge_index = data.edge_index
            labels = data.y[batch_indices]
            
            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(node_features, sequences, edge_index)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(node_features, sequences, edge_index)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches_processed += 1
        
        avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0.0
        
        return {'loss': avg_loss}
    
    def evaluate(self, data: Any, mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model on given mask.
        
        Args:
            data: PyG Data object
            mask: Boolean mask for evaluation
        
        Returns:
            Dictionary of evaluation metrics
        """
        from src.utils.metrics import compute_metrics
        
        self.model.eval()
        
        with torch.no_grad():
            node_features = data.x
            sequences = data.x
            edge_index = data.edge_index
            
            logits = self.model(node_features, sequences, edge_index)
            
            mask_indices = torch.where(mask)[0]
            
            loss = self.criterion(logits[mask_indices], data.y[mask_indices]).item()
            
            probs = torch.softmax(logits[mask_indices], dim=1)[:, 1]
            preds = logits[mask_indices].argmax(dim=1)
            
            metrics = compute_metrics(data.y[mask_indices], preds, probs)
            metrics['loss'] = loss
        
        return metrics
    
    def train(
        self,
        data: Any,
        num_epochs: int,
        batch_size: int,
        val_interval: int = 1,
        early_stopping_patience: int = 15,
        early_stopping_metric: str = 'f1',
        checkpoint_dir: Optional[Any] = None,
        best_model_name: str = 'best_model.pt',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            data: PyG Data object
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            val_interval: Evaluate on validation set every N epochs
            early_stopping_patience: Patience for early stopping
            early_stopping_metric: Metric to use for early stopping
            checkpoint_dir: Directory to save checkpoints
            best_model_name: Name for best model checkpoint
            verbose: Print training progress
        
        Returns:
            Training history and best metrics
        """
        best_val_metric = -float('inf') if early_stopping_metric != 'loss' else float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_auc_pr': []
        }
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(data, batch_size)
            
            if verbose:
                log_msg = f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f}"
            
            if epoch % val_interval == 0:
                val_metrics = self.evaluate(data, data.val_mask)
                
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_auc_pr'].append(val_metrics.get('auc_pr', 0.0))
                
                if verbose:
                    log_msg += f" | val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f} | val_auc_pr={val_metrics.get('auc_pr', 0.0):.4f}"
                
                current_metric = val_metrics.get(early_stopping_metric, val_metrics['f1'])
                
                if early_stopping_metric == 'loss':
                    is_better = current_metric < best_val_metric
                else:
                    is_better = current_metric > best_val_metric
                
                if is_better:
                    best_val_metric = current_metric
                    patience_counter = 0
                    
                    if checkpoint_dir is not None:
                        checkpoint_path = checkpoint_dir / best_model_name
                        torch.save({
                            'model_state_dict': self.model.state_dict(),
                            'epoch': epoch,
                            'val_metrics': val_metrics,
                        }, checkpoint_path)
                        if verbose:
                            self.print(f"  -> Saved best model to {checkpoint_path}")
                else:
                    patience_counter += 1
                
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    elif isinstance(self.scheduler, CosineAnnealingLR):
                        self.scheduler.step()
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        self.print(f"Early stopping at epoch {epoch}")
                    break
            
            history['train_loss'].append(train_metrics['loss'])
            
            if verbose:
                self.print(log_msg)
        
        return {
            'history': history,
            'best_val_metric': best_val_metric
        }


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    
    optimizer_type = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0001)
    
    if optimizer_type == 'adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[Any]:
    """Create learning rate scheduler from config."""
    
    if not config.get('use_scheduler', True):
        return None
    
    scheduler_type = config.get('scheduler_type', 'reduce_on_plateau')
    
    if scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5),
            min_lr=config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
    
    return None
