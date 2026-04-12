"""
Loss functions for AML Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights [weight_0, weight_1]
        gamma: Focusing parameter (default: 2.0)
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        outputs = torch.clamp(outputs, min=-100, max=100)
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                self.alpha = torch.tensor(self.alpha, device=outputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross Entropy Loss"""
    
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
    
    def forward(self, outputs, targets):
        return F.cross_entropy(outputs, targets, weight=self.weight)


def get_loss_function(loss_type='focal', class_weights=None, focal_gamma=2.0, focal_alpha=0.25):
    """
    Factory function to get loss function.
    
    Args:
        loss_type: 'focal' or 'weighted_ce'
        class_weights: [weight_0, weight_1]
        focal_gamma: Focal loss gamma
        focal_alpha: Focal loss alpha
    
    Returns:
        Loss function
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        weight_tensor = None
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        return WeightedCrossEntropyLoss(weight=weight_tensor)