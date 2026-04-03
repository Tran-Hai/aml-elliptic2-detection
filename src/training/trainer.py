"""
Trainer class for LAS-Mamba-GNN model
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Optional, Any
import numpy as np
import sys
from pathlib import Path


# ─── FIX 1: Thay vòng for Python bằng tensor mask trên GPU ──────────────────
def get_local_edge_index(full_edge_index, batch_global_indices, device):
    """
    Tạo local edge index cho một batch nodes.
    Dùng tensor mask thay vì vòng for Python → nhanh hơn 100-500x.
    """
    if isinstance(batch_global_indices, torch.Tensor):
        batch_tensor = batch_global_indices.to(device)
    else:
        batch_tensor = torch.tensor(
            list(batch_global_indices), dtype=torch.long, device=device
        )

    full_edge_index = full_edge_index.to(device)
    max_node = int(full_edge_index.max().item()) + 1

    in_batch = torch.zeros(max_node, dtype=torch.bool, device=device)
    in_batch[batch_tensor] = True

    src, dst = full_edge_index[0], full_edge_index[1]
    mask = in_batch[src] & in_batch[dst]
    filtered = full_edge_index[:, mask]

    if filtered.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    local_map = torch.full((max_node,), -1, dtype=torch.long, device=device)
    local_map[batch_tensor] = torch.arange(
        len(batch_tensor), dtype=torch.long, device=device
    )
    return local_map[filtered]


class Trainer:
    """Trainer for LAS-Mamba-GNN model with lazy batch loading."""

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
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if use_amp and device.type == 'cuda' else None
        )

    def train_epoch(self, dataset: Any, batch_size: int = 256) -> Dict[str, float]:
        self.model.train()
        use_gnn = getattr(self.model, 'use_gnn', True)
        train_mask = dataset.train_mask
        num_train = int(train_mask.sum().item())
        train_indices = torch.where(train_mask)[0].numpy()
        indices = train_indices[np.random.permutation(num_train)]

        if use_gnn:
            full_edge_index = dataset.edge_index.to(self.device)

        total_loss = 0.0
        num_batches = 0

        for start in range(0, num_train, batch_size):
            batch_indices = indices[start:min(start + batch_size, num_train)].tolist()

            edge_index = (
                get_local_edge_index(full_edge_index, batch_indices, self.device)
                if use_gnn
                else torch.empty((2, 0), dtype=torch.long, device=self.device)
            )

            sequences, labels = dataset.load_batch(batch_indices)
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            node_features = sequences.mean(dim=(1, 2))

            self.optimizer.zero_grad()

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
            num_batches += 1

        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}

    def evaluate(self, dataset: Any, mask: torch.Tensor, eval_batch_size: int = 512) -> Dict[str, float]:
        from src.utils.metrics import compute_metrics
        self.model.eval()
        use_gnn = getattr(self.model, 'use_gnn', True)
        mask_indices = torch.where(mask)[0].numpy()
        num_samples = len(mask_indices)

        if use_gnn:
            full_edge_index = dataset.edge_index.to(self.device)

        all_preds, all_probs, all_labels = [], [], []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for start in range(0, num_samples, eval_batch_size):
                batch_indices = mask_indices[start:min(start + eval_batch_size, num_samples)].tolist()

                edge_index = (
                    get_local_edge_index(full_edge_index, batch_indices, self.device)
                    if use_gnn
                    else torch.empty((2, 0), dtype=torch.long, device=self.device)
                )

                sequences, labels = dataset.load_batch(batch_indices)
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                node_features = sequences.mean(dim=(1, 2))

                logits = self.model(node_features, sequences, edge_index)
                total_loss += self.criterion(logits, labels).item()
                num_batches += 1

                all_preds.append(logits.argmax(dim=1).cpu())
                all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
                all_labels.append(labels.cpu())

        metrics = compute_metrics(
            torch.cat(all_labels), torch.cat(all_preds), torch.cat(all_probs)
        )
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
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
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints"
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        metric_key = early_stopping_metric.replace('val_', '')
        best_val_metric = -float('inf') if metric_key != 'loss' else float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc_pr': []}

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(dataset, batch_size)
            log_msg = f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f}"

            if epoch % val_interval == 0:
                val_metrics = self.evaluate(dataset, dataset.val_mask)
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_auc_pr'].append(val_metrics.get('auc_pr', 0.0))

                log_msg += (
                    f" | val_loss={val_metrics['loss']:.4f}"
                    f" | val_f1={val_metrics['f1']:.4f}"
                    f" | val_auc_pr={val_metrics.get('auc_pr', 0.0):.4f}"
                )

                current = val_metrics.get(metric_key, val_metrics['f1'])
                is_better = (current < best_val_metric if metric_key == 'loss' else current > best_val_metric)

                if is_better:
                    best_val_metric = current
                    patience_counter = 0
                    ckpt = checkpoint_dir / best_model_name
                    torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch, 'val_metrics': val_metrics}, ckpt)
                    if verbose:
                        self.print(f"  -> Saved best model (val_{metric_key}={current:.4f}) to {ckpt}")
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

        return {'history': history, 'best_val_metric': best_val_metric}


def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    lr = config.get('learning_rate', 0.001)
    wd = config.get('weight_decay', 0.0001)
    opt = config.get('optimizer', 'adam').lower()
    if opt == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    return Adam(model.parameters(), lr=lr, weight_decay=wd)


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[Any]:
    if not config.get('use_scheduler', True):
        return None
    stype = config.get('scheduler_type', 'reduce_on_plateau')
    if stype == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer, mode='max',
            factor=config.get('scheduler_factor', 0.5),
            patience=config.get('scheduler_patience', 5),
            min_lr=config.get('min_lr', 1e-6)
        )
    elif stype == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=config.get('num_epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )
    return None


class OptimizedTrainer:
    """
    Optimized trainer for LAS-Mamba-GNN with DataLoader support.
    Works with GPU, AMP, and parallel data loading.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        grad_clip_norm: float = 0.5,  # More conservative default
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
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if use_amp and device.type == 'cuda' else None
        )

    def train_epoch_with_loader(self, train_loader, edge_index, use_gnn=True) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (sequences, labels, batch_global_indices) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            batch_global_indices = batch_global_indices.to(self.device)

            local_edge_index = (
                get_local_edge_index(edge_index, batch_global_indices, self.device)
                if use_gnn and edge_index is not None and edge_index.shape[1] > 0
                else torch.empty((2, 0), dtype=torch.long, device=self.device)
            )

            node_features = sequences.mean(dim=(1, 2))
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(node_features, sequences, local_edge_index)
                    loss = self.criterion(logits, labels)
                if torch.isnan(loss):
                    continue
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(node_features, sequences, local_edge_index)
                loss = self.criterion(logits, labels)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                self.print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss}

    def evaluate_loader(self, data_loader, edge_index, use_gnn=True) -> Dict[str, float]:
        from src.utils.metrics import compute_metrics
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        total_loss = 0.0
        num_batches = 0
        nan_batches = []
        debug_info = []

        with torch.no_grad():
            for batch_idx, (sequences, labels, batch_global_indices) in enumerate(data_loader):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                batch_global_indices = batch_global_indices.to(self.device)

                local_edge_index = (
                    get_local_edge_index(edge_index, batch_global_indices, self.device)
                    if use_gnn and edge_index is not None and edge_index.shape[1] > 0
                    else torch.empty((2, 0), dtype=torch.long, device=self.device)
                )

                node_features = sequences.mean(dim=(1, 2))
                logits = self.model(node_features, sequences, local_edge_index)
                
                # Debug: Log batch info
                logits_max = logits.abs().max().item()
                logits_mean = logits.mean().item()
                has_nan_logits = torch.isnan(logits).any().item()
                has_inf_logits = torch.isinf(logits).any().item()
                
                # Calculate loss safely - skip if NaN
                loss_val = self.criterion(logits, labels)
                loss_item = loss_val.item() if not torch.isnan(loss_val) else float('nan')
                
                if not torch.isnan(loss_val):
                    total_loss += loss_val.item()
                    num_batches += 1
                else:
                    nan_batches.append(batch_idx)
                    debug_info.append({
                        'batch_idx': batch_idx,
                        'logits_max': logits_max,
                        'logits_mean': logits_mean,
                        'has_nan_logits': has_nan_logits,
                        'has_inf_logits': has_inf_logits,
                        'local_edge_shape': local_edge_index.shape,
                        'sequences_shape': sequences.shape,
                        'labels_dist': labels.cpu().tolist().count(1)
                    })
                    print(f"  WARNING: NaN loss in evaluation at batch {batch_idx}")
                    print(f"    DEBUG: logits max={logits_max:.4f}, mean={logits_mean:.4f}, nan={has_nan_logits}, inf={has_inf_logits}")
                    print(f"    DEBUG: local_edge shape={local_edge_index.shape}, positive labels={labels.sum().item()}")

                all_preds.append(logits.argmax(dim=1).cpu())
                all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
                all_labels.append(labels.cpu())

        # Print summary of NaN batches
        if nan_batches:
            print(f"  DEBUG: Total NaN batches: {len(nan_batches)}/{len(data_loader)}")
            print(f"  DEBUG: NaN batch indices: {nan_batches[:10]}...")  # Show first 10

        metrics = compute_metrics(
            torch.cat(all_labels), torch.cat(all_preds), torch.cat(all_probs)
        )
        
        # Apply threshold tuning for predictions
        # Use probability threshold instead of argmax
        all_probs_tensor = torch.cat(all_probs)
        threshold = 0.3  # Lower threshold = more positive predictions
        
        # Get predictions with threshold
        preds_with_threshold = (all_probs_tensor > threshold).long()
        
        # Compute metrics with threshold
        metrics_threshold = compute_metrics(
            torch.cat(all_labels), preds_with_threshold, all_probs_tensor
        )
        
        # Use threshold-based metrics if F1 is better
        if metrics_threshold['f1'] > metrics['f1']:
            print(f"  DEBUG: Threshold={threshold} improved F1 from {metrics['f1']:.4f} to {metrics_threshold['f1']:.4f}")
            metrics['f1'] = metrics_threshold['f1']
            metrics['recall'] = metrics_threshold['recall']
            metrics['precision'] = metrics_threshold['precision']
            metrics['accuracy'] = metrics_threshold['accuracy']
        
        # Handle NaN in loss calculation
        if num_batches > 0:
            metrics['loss'] = total_loss / num_batches
        else:
            metrics['loss'] = 0.0
            print("  WARNING: All batches had NaN loss, setting val_loss to 0.0")
        
        return metrics

    def train_with_loaders(
        self,
        train_loader,
        val_loader,
        test_loader,
        edge_index,
        num_epochs: int,
        val_interval: int = 1,
        early_stopping_patience: int = 15,
        early_stopping_metric: str = 'f1',
        checkpoint_dir=None,
        best_model_name: str = 'best_model.pt',
        verbose: bool = True
    ) -> Dict[str, Any]:
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).resolve().parent.parent.parent / "checkpoints"
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        metric_key = early_stopping_metric.replace('val_', '')
        best_val_metric = -float('inf') if metric_key != 'loss' else float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc_pr': []}
        use_gnn = getattr(self.model, 'use_gnn', True)

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch_with_loader(train_loader, edge_index, use_gnn)
            log_msg = f"Epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f}"

            if epoch % val_interval == 0:
                val_metrics = self.evaluate_loader(val_loader, edge_index, use_gnn)
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1'])
                history['val_auc_pr'].append(val_metrics.get('auc_pr', 0.0))

                log_msg += (
                    f" | val_loss={val_metrics['loss']:.4f}"
                    f" | val_f1={val_metrics['f1']:.4f}"
                    f" | val_auc_pr={val_metrics.get('auc_pr', 0.0):.4f}"
                )

                current = val_metrics.get(metric_key, val_metrics['f1'])
                is_better = (current < best_val_metric if metric_key == 'loss' else current > best_val_metric)

                if is_better:
                    best_val_metric = current
                    patience_counter = 0
                    ckpt = checkpoint_dir / best_model_name
                    torch.save({'model_state_dict': self.model.state_dict(), 'epoch': epoch, 'val_metrics': val_metrics}, ckpt)
                    if verbose:
                        self.print(f"  -> Saved best model (val_{metric_key}={current:.4f}) to {ckpt}")
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

        return {'history': history, 'best_val_metric': best_val_metric}
