"""
Loss functions for AML Detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss for class imbalance."""
    
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
    
    def forward(self, outputs, targets):
        return F.cross_entropy(outputs, targets, weight=self.weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses on hard-to-classify samples by down-weighting easy samples.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
    
    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=self.weights)
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class WeightedFocalLoss(nn.Module):
    """
    Combined Weighted Cross Entropy and Focal Loss.
    Uses class weights AND focuses on hard samples.
    """
    
    def __init__(self, weights=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none', weight=self.weights)
        
        pt = torch.exp(-ce_loss)
        
        focal_weight = (1 - pt) ** self.gamma
        
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for extreme class imbalance.
    Different penalty for false positive vs false negative.
    """
    
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
    
    def forward(self, outputs, targets):
        probs = torch.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
        
        probs_pos = probs * targets_one_hot
        probs_neg = probs * (1 - targets_one_hot)
        
        if self.gamma_pos > 0:
            probs_pos = probs_pos ** self.gamma_pos
        
        if self.gamma_neg > 0:
            probs_neg = (probs_neg + self.clip) ** self.gamma_neg
        
        loss = -probs_pos.sum(dim=1) - probs_neg.sum(dim=1)
        
        return loss.mean()


def get_loss_function(loss_type='weighted_focal', class_weights=None):
    """
    Factory function to get loss function.
    
    Args:
        loss_type: Type of loss ('ce', 'weighted_ce', 'focal', 'weighted_focal', 'asymmetric')
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
    
    elif loss_type == 'focal':
        return FocalLoss()
    
    elif loss_type == 'weighted_focal':
        return WeightedFocalLoss(weights=class_weights)
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


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
