"""
Loss functions for AML Detection
Weighted Cross Entropy Loss and Focal Loss for class imbalance
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


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t = probability of correct class
    - gamma = focusing parameter (default: 2.0)
    - alpha = class weighting
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            alpha_tensor = alpha if torch.is_tensor(alpha) else torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha_tensor.float())
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        # Clamp outputs
        outputs = torch.clamp(outputs, min=-100, max=100)
        
        # Check for NaN/Inf inputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # Get probabilities
        p_t = torch.exp(-ce_loss)
        
        # Apply focal weighting
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.to(outputs.device)[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='weighted_ce', class_weights=None, focal_gamma=2.0):
    """
    Factory function to get loss function.
    
    Args:
        loss_type: Type of loss ('ce', 'weighted_ce', 'focal')
        class_weights: Tensor of class weights [weight_licit, weight_suspicious]
        focal_gamma: Gamma parameter for Focal Loss
    
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
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. Supported: 'ce', 'weighted_ce', 'focal'")


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
