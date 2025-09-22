import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Focal Loss Implementation --
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, temp=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor for the rare class.
            gamma: Focusing parameter that reduces the relative loss for well-classified examples.
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.reduction = reduction
        self.ce_Loss = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes] raw, unnormalized scores.
            targets: [B] LongTensor with ground truth class indices.
        """
        # Compute the standard cross entropy loss (no reduction)
        # ce_loss = F.cross_entropy(logits, targets, reduction='none')
        ce_loss = self.ce_Loss(logits, targets)
        # Compute the probability for the true class
        pt = torch.exp(-ce_loss / self.temp)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
