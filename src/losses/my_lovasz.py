import torch
import torch.nn as nn


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """

    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union

    if p == 1:
        return jaccard

    jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probs, labels, classes='present', per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss
      probs: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """

    if per_image:
        losses = []
        for prob, lab in zip(probs, labels):
            flat_probs, flat_labels = flatten_probs(prob.unsqueeze(0), lab.unsqueeze(0), ignore)
            losses.append(lovasz_softmax_flat(flat_probs, flat_labels, classes=classes))
        loss = torch.mean(torch.tensor(losses))
    else:
        flat_probs, flat_labels = flatten_probs(probs, labels, ignore)
        loss = lovasz_softmax_flat(flat_probs, flat_labels, classes=classes)
    return loss


def lovasz_softmax_flat(probs: torch.Tensor, labels: torch.Tensor, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probs: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """

    # Handle the situation where there are no valid pixels by killing the gradient
    if probs.numel() == 0:
        return probs * 0.

    losses = []
    num_classes = probs.size(1)
    classes_to_sum = list(range(num_classes)) if classes in ['all', 'present'] else classes
    for c in classes_to_sum:
        class_mask = (labels == c).float()
        if classes == 'present' and class_mask.sum() == 0:
            continue
        if num_classes == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = torch.abs(class_mask - class_pred)
        errors_sorted, indices = torch.sort(errors, 0, descending=True)
        class_mask_sorted = class_mask[indices]
        losses.append(torch.dot(errors_sorted, lovasz_grad(class_mask_sorted)))
    return torch.mean(torch.tensor(losses))


def flatten_probs(probs: torch.Tensor, labels: torch.Tensor, ignore: torch.Tensor = None):
    """Flattens predictions in the batch and removes labels equal to 'ignore', if specified.

    - probs: (B, C, H, W) -> (B * H * W, C)
    - labels: (B, H, W) -> (B * H * W)

    :param probs: (B, C, H, W) Variable, class probabilities at each prediction (between 0 and 1).
    :param labels: (B, H, W) Tensor, ground truth labels (between 0 and C - 1).
    :param ignore: void class labels.
    :return: (B * H * W, C) Tensor, (B * H * W) Tensor
    """

    # If the input is of shape (B, H, W), we reshape to (B, 1, H, W)
    if probs.dim() == 3:
        probs = probs.unsqueeze(1)

    # Flatten the predictions and labels in the batch
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)

    # Remove labels equal to 'ignore'
    if ignore is not None:
        valid = torch.ne(labels, ignore)
        probs = probs[valid, :]
        labels = labels[valid]

    return probs, labels


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probs, labels):
        return lovasz_softmax(probs, labels, self.classes, self.per_image, self.ignore)
