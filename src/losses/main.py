import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_loss(loss_type: str, weight: torch.tensor, device: torch.device, ignore_index=0):
    if loss_type == 'CombinedLoss':
        return CombinedLoss(device, weight=weight, ignore_index=ignore_index)
    elif loss_type == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index).to(device)
    elif loss_type == 'FocalLoss':
        return smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index).to(device)
    elif loss_type == 'DiceLoss':
        return smp.losses.DiceLoss(mode='multiclass', ignore_index=ignore_index).to(device)
    elif loss_type == 'LovaszLoss':
        return smp.losses.LovaszLoss(mode='multiclass', ignore_index=ignore_index).to(device)
    else:
        raise ValueError(f'Unknown loss: {type}')


class CombinedLoss(nn.Module):
    def __init__(self, device, weight: torch.Tensor, ignore_index=0):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index).to(device)
        self.lovasz = smp.losses.LovaszLoss(mode='multiclass', ignore_index=ignore_index).to(device)

    def forward(self, logits, targets):
        return self.cross_entropy(logits, targets) + self.lovasz(logits, targets)
