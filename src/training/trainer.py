"""
Trainer class cho mô hình Mamba-GNN.
Tích hợp Neighbor Expansion cho hiệu năng GNN và Mamba cho chuỗi thời gian.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
from pathlib import Path
from torch_geometric.utils import k_hop_subgraph


def get_neighbor_subgraph(full_edge_index, batch_indices, num_hops=1):
    """Mở rộng batch indices để bao gồm các node lân cận."""
    if not isinstance(batch_indices, torch.Tensor):
        batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    
    subset, local_edge_index, mapping, edge_mask = k_hop_subgraph(
        batch_indices, 
        num_hops, 
        full_edge_index, 
        relabel_nodes=True
    )
    return subset, local_edge_index, mapping


class OptimizedTrainer:
    """Trainer cho Mamba-GNN với Neighbor Expansion."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        grad_clip_norm: float = 0.5,
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
            torch.amp.GradScaler('cuda')
            if use_amp and device.type == 'cuda' else None
        )

    def _prepare_node_features(self, sequences):
        """Tạo node features 576-dim [mean, max, last] từ chuỗi [N, 2, 50, 96]."""
        node_feat_list = []
        for flow_idx in [0, 1]:
            flow_data = sequences[:, flow_idx] # [N, 50, 96]
            node_feat_list.append(flow_data.mean(dim=1))
            node_feat_list.append(flow_data.max(dim=1)[0])
            node_feat_list.append(flow_data[:, -1, :])
        return torch.cat(node_feat_list, dim=1) # [N, 576]

    def train_epoch(self, train_loader, dataset, edge_index, num_hops=1):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (sequences, labels, batch_global_indices) in enumerate(train_loader):
            batch_global_indices = batch_global_indices.to(self.device)
            labels = labels.to(self.device)

            if self.model.use_gnn and edge_index is not None:
                # 1. Mở rộng đồ thị con
                subset, local_edge_index, mapping = get_neighbor_subgraph(
                    edge_index, batch_global_indices, num_hops=num_hops
                )
                local_edge_index = local_edge_index.to(self.device)
                
                # 2. Load data chuỗi cho subgraph
                all_sequences, _ = dataset.get_batch(subset.tolist())
                all_sequences = all_sequences.to(self.device)
                all_node_features = self._prepare_node_features(all_sequences)
                
                self.optimizer.zero_grad(set_to_none=True)
                
                if self.use_amp and self.scaler is not None:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        all_logits = self.model(all_node_features, all_sequences, local_edge_index)
                        logits = all_logits[mapping]
                        loss = self.criterion(logits, labels)
                    if not torch.isnan(loss):
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    all_logits = self.model(all_node_features, all_sequences, local_edge_index)
                    logits = all_logits[mapping]
                    loss = self.criterion(logits, labels)
                    if not torch.isnan(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.optimizer.step()
            else:
                # Trường hợp không dùng GNN
                sequences = sequences.to(self.device)
                node_features = self._prepare_node_features(sequences)
                self.optimizer.zero_grad()
                logits = self.model(node_features, sequences, torch.empty((2,0), dtype=torch.long, device=self.device))
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
            
            if batch_idx % 100 == 0:
                self.print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        return {'loss': total_loss / num_batches if num_batches > 0 else 0.0}

    def evaluate(self, loader, dataset, edge_index, num_hops=1):
        from src.utils.metrics import compute_metrics
        self.model.eval()
        all_probs, all_labels = [], []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for sequences, labels, batch_global_indices in loader:
                batch_global_indices = batch_global_indices.to(self.device)
                labels = labels.to(self.device)

                if self.model.use_gnn and edge_index is not None:
                    subset, local_edge_index, mapping = get_neighbor_subgraph(
                        edge_index, batch_global_indices, num_hops=num_hops
                    )
                    local_edge_index = local_edge_index.to(self.device)
                    all_sequences, _ = dataset.get_batch(subset.tolist())
                    all_sequences = all_sequences.to(self.device)
                    all_node_features = self._prepare_node_features(all_sequences)
                    
                    all_logits = self.model(all_node_features, all_sequences, local_edge_index)
                    logits = all_logits[mapping]
                else:
                    sequences = sequences.to(self.device)
                    node_features = self._prepare_node_features(sequences)
                    logits = self.model(node_features, sequences, torch.empty((2,0), dtype=torch.long, device=self.device))

                loss_val = self.criterion(logits, labels)
                if not torch.isnan(loss_val):
                    total_loss += loss_val.item()
                    num_batches += 1

                all_probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
                all_labels.append(labels.cpu())

        y_true = torch.cat(all_labels)
        y_prob = torch.cat(all_probs)
        
        # Tìm ngưỡng tối ưu cho F1
        best_f1, best_thr = -1.0, 0.2
        for thr in np.arange(0.05, 0.9, 0.05):
            y_pred = (y_prob > thr).long()
            metrics_t = compute_metrics(y_true, y_pred, y_prob)
            if metrics_t['f1'] > best_f1:
                best_f1 = metrics_t['f1']
                best_thr = thr
        
        final_preds = (y_prob > best_thr).long()
        metrics = compute_metrics(y_true, final_preds, y_prob)
        metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        metrics['threshold'] = best_thr
        
        return metrics

    def train(
        self,
        train_loader,
        val_loader,
        dataset,
        edge_index,
        num_epochs: int,
        num_hops: int = 1,
        val_interval: int = 1,
        early_stopping_patience: int = 15,
        checkpoint_dir=None
    ):
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        best_val_f1 = -1.0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

        for epoch in range(1, num_epochs + 1):
            train_results = self.train_epoch(train_loader, dataset, edge_index, num_hops)
            history['train_loss'].append(train_results['loss'])
            
            if epoch % val_interval == 0:
                val_metrics = self.evaluate(val_loader, dataset, edge_index, num_hops)
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1'].append(val_metrics['f1'])
                
                self.print(f"Epoch {epoch:03d} | Train Loss: {train_results['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} | Thr: {val_metrics['threshold']:.2f}")

                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'val_metrics': val_metrics,
                        'epoch': epoch,
                        'best_threshold': val_metrics['threshold']
                    }, checkpoint_dir / 'best_model.pt')
                else:
                    patience_counter += 1

                if self.scheduler is not None:
                    self.scheduler.step(val_metrics['f1'])

                if patience_counter >= early_stopping_patience:
                    self.print(f"Early stopping at epoch {epoch}")
                    break
        
        return history
