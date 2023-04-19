import torch
import torch.nn as nn

from .lovasz import LovaszSoftmax


def get_loss(loss_type: str, weight: torch.tensor, device: torch.device, ignore_index=0):
    if loss_type == 'semantic':
        return SemanticLoss(device, weight=weight, ignore_index=ignore_index)
    else:
        raise ValueError(f'Unknown loss: {type}')


class SemanticLoss(nn.Module):
    def __init__(self, device, weight: torch.Tensor, ignore_index=0):
        super(SemanticLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index).to(device)
        self.lovasz = LovaszSoftmax(ignore=ignore_index).to(device)

    def forward(self, logits, targets):
        loss = self.cross_entropy(logits, targets)
        # loss = self.lovasz(logits, targets)
        return loss
