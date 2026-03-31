"""
Trainer class for LAS-Mamba-GNN model
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Optional, Any
import numpy as np


def get_local_edge_index(full_edge_index, batch_indices, device):
    """Create local edge index for a batch of nodes."""
    batch_indices_set = set(batch_indices)
    batch_node_to_local = {idx: i for i, idx in enumerate(batch_indices)}
    
    src = full_edge_index[0]
    dst = full_edge_index[1]
    
    local_edges = []
    for i in range(full_edge_index.shape[1]):
        s = src[i].item()
        d = dst[i].item()
        if s in batch_indices_set and d in batch_indices_set:
            local_edges.append([batch_node_to_local[s], batch_node_to_local[d]])
    
    if len(local_edges) > 0:
        edge_index = torch.tensor(local_edges, dtype=torch.long, device=device).t()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    return edge_index


class Trainer:
    """
    Trainer for LAS-Mamba-GNN model with lazy batch loading.
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
    
    def train_epoch(self, dataset: Any, batch_size: int = 256) -> Dict[str, float]:
        """Train for one epoch with lazy batch loading."""
        self.model.train()
        
        use_gnn = getattr(self.model, 'use_gnn', True)
        
        train_mask = dataset.train_mask
        num_train_nodes = train_mask.sum().item()
        
        train_indices = torch.where(train_mask)[0].numpy()
        
        total_loss = 0.0
        num_batches_processed = 0
        
        indices = train_indices[np.random.permutation(num_train_nodes)]
        
        if use_gnn:
            full_edge_index = dataset.edge_index.to(self.device)
        
        for start in range(0, num_train_nodes, batch_size):
            end = min(start + batch_size, num_train_nodes)
            batch_indices = indices[start:end].tolist()
            
            if use_gnn:
                edge_index = get_local_edge_index(full_edge_index, batch_indices, self.device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            
            self.optimizer.zero_grad()
            
            sequences, labels = dataset.load_batch(batch_indices)
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            node_features = sequences.mean(dim=(1, 2))
            
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
    
    def evaluate(self, dataset: Any, mask: torch.Tensor, eval_batch_size: int = 512) -> Dict[str, float]:
        """Evaluate model on given mask using batched evaluation."""
        from src.utils.metrics import compute_metrics
        
        self.model.eval()
        
        use_gnn = getattr(self.model, 'use_gnn', True)
        
        mask_indices = torch.where(mask)[0].numpy()
        num_samples = len(mask_indices)
        
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0.0
        
        if use_gnn:
            full_edge_index = dataset.edge_index.to(self.device)
        
        with torch.no_grad():
            for start in range(0, num_samples, eval_batch_size):
                end = min(start + eval_batch_size, num_samples)
                batch_indices = mask_indices[start:end].tolist()
                
                if use_gnn:
                    edge_index = get_local_edge_index(full_edge_index, batch_indices, self.device)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                
                sequences, labels = dataset.load_batch(batch_indices)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                node_features = sequences.mean(dim=(1, 2))
                
                logits = self.model(node_features, sequences, edge_index)
                
                loss = self.criterion(logits, labels).item()
                total_loss += loss
                
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)
                
                all_preds.append(preds.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        all_preds = torch.cat(all_preds)
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        avg_loss = total_loss / ((num_samples + eval_batch_size - 1) // eval_batch_size)
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def train(
        self,
        dataset: Any,
        num_epochs: int,
        batch_size: int,
        val_interval: int = 1,
        early_stopping_patience: int = 15,
        early_stopping_metric: str = 'f1',
        checkpoint_dir: Optional[Any] = None,
        best_model_name: str = 'best_model.pt',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Full training loop."""
        best_val_metric = -float('inf') if early_stopping_metric != 'loss' else float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_auc_pr': []
        }
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(dataset, batch_size)
            
            log_msg = f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f}"
            
            if epoch % val_interval == 0:
                val_metrics = self.evaluate(dataset, dataset.val_mask)
                
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_auc_pr'].append(val_metrics.get('auc_pr', 0.0))
                
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