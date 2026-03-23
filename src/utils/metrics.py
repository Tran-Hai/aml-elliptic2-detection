"""
Metrics utilities for AML Detection
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Compute all metrics for binary classification.
    
    Args:
        y_true: Ground truth labels (numpy array or torch tensor)
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class
    
    Returns:
        Dictionary of metrics
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if y_pred_proba is not None and torch.is_tensor(y_pred_proba):
        y_pred_proba = y_pred_proba.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division='warn'),
        'recall': recall_score(y_true, y_pred, zero_division='warn'),
        'f1': f1_score(y_true, y_pred, zero_division='warn'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_pred_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
    
    return metrics


def print_metrics(metrics, prefix=''):
    """Print metrics in a formatted way."""
    if prefix:
        print(f"\n{prefix}")
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    if 'auc_roc' in metrics:
        print(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
        print(f"AUC-PR:   {metrics['auc_pr']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  TN: {cm[0][0]:5d}  FP: {cm[0][1]:5d}")
    print(f"  FN: {cm[1][0]:5d}  TP: {cm[1][1]:5d}")


class MetricsTracker:
    """Track metrics across epochs."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc_roc': [],
            'val_auc_pr': []
        }
    
    def update(self, epoch, train_loss, val_metrics):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_metrics.get('loss', 0))
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_f1'].append(val_metrics['f1'])
        
        if 'auc_roc' in val_metrics:
            self.history['val_auc_roc'].append(val_metrics['auc_roc'])
        if 'auc_pr' in val_metrics:
            self.history['val_auc_pr'].append(val_metrics['auc_pr'])
    
    def get_best_epoch(self, metric='val_f1'):
        if metric not in self.history or not self.history[metric]:
            return 0
        return np.argmax(self.history[metric]) + 1
    
    def get_best_value(self, metric='val_f1'):
        if metric not in self.history or not self.history[metric]:
            return 0.0
        return max(self.history[metric])
    
    def print_summary(self):
        print("TRAINING SUMMARY")
        
        best_f1 = self.get_best_value('val_f1')
        best_f1_epoch = self.get_best_epoch('val_f1')
        print(f"Best F1-Score: {best_f1:.4f} (Epoch {best_f1_epoch})")
        
        if self.history['val_auc_roc']:
            best_auc = self.get_best_value('val_auc_roc')
            best_auc_epoch = self.get_best_epoch('val_auc_roc')
            print(f"Best AUC-ROC:  {best_auc:.4f} (Epoch {best_auc_epoch})")
        
        print(f"Total epochs: {len(self.history['train_loss'])}")
