# ------------------------------------------------------------------------------------------
# Copyright (c) 2026 Chen Zhang, Tao Li
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


class KLDivergenceBasedLoss(nn.Module):
    """
    KL divergence based loss that naturally handles imbalanced magnitudes
    by treating predictions and targets as probability-like distributions
    """
    
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Ensure positive values and add small epsilon for numerical stability
        pred_positive = torch.clamp(pred, min=self.epsilon)
        target_positive = torch.clamp(target, min=self.epsilon)
        
        # Normalize each sample to make them probability-like (sum to 1)
        pred_norm = pred_positive / (pred_positive.sum(dim=1, keepdim=True) + self.epsilon)
        target_norm = target_positive / (target_positive.sum(dim=1, keepdim=True) + self.epsilon)
        
        # KL divergence: KL(target || pred) = sum(target * log(target / pred))
        kl_div = F.kl_div(torch.log(pred_norm + self.epsilon), target_norm, reduction='batchmean')
        
        return kl_div


class LogSpaceLoss(nn.Module):
    """
    Loss function that works in log space to naturally handle different magnitude scales
    """
    
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target):
        # Convert to log space
        log_pred = torch.log(pred + self.epsilon)
        log_target = torch.log(target + self.epsilon)
        
        # MSE in log space
        return F.mse_loss(log_pred, log_target)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing to prevent overconfident predictions"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        n_classes = x.size(1)
        log_probs = F.log_softmax(x, dim=-1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(self.smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # Calculate loss
        loss = (-true_dist * log_probs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in binary classification"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probabilities for the correct class
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Compute focal loss
        focal_weight = (1 - pt) ** self.gamma
        loss = -self.alpha * focal_weight * torch.log(pt + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss