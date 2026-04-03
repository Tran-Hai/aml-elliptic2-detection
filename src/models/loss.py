"""
Loss functions for AML Detection
Weighted Cross Entropy Loss for class imbalance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance."""
    
    def __init__(self, weights=None, eps=1e-7):
        super().__init__()
        self.weights = weights
        self.eps = eps
    
    def forward(self, outputs, targets):
        # Clamp outputs to avoid extreme values
        outputs = torch.clamp(outputs, min=-100, max=100)
        
        # Check for NaN/Inf inputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print("  DEBUG Loss: NaN/Inf detected in outputs, returning zero loss")
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        return F.cross_entropy(outputs, targets, weight=self.weights)


def get_loss_function(loss_type='weighted_ce', class_weights=None):
    """
    Factory function to get loss function.
    
    Args:
        loss_type: Type of loss (only 'ce' or 'weighted_ce' supported)
        class_weights: Tensor of class weights [weight_licit, weight_suspicious]
    
    Returns:
        Loss function
    """
    if class_weights is not None:
        if isinstance(class_weights, list):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(weights=class_weights)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Only 'ce' and 'weighted_ce' are supported.")


def compute_class_weights(labels):
    """
    Compute class weights from labels.
    
    Args:
        labels: Tensor or array of labels
    
    Returns:
        Class weights [weight_neg, weight_pos]
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    weights = []
    for count in counts:
        weight = total / (len(unique) * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)
