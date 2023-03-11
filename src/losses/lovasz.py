import torch
import torch.nn as nn
from itertools import filterfalse
from torch.autograd import Variable


def isnan(x):
    return x != x


def mean(loss, ignore_nan=False, empty=0):
    loss = iter(loss)
    if ignore_nan:
        loss = filterfalse(isnan, loss)
    try:
        n = 1
        acc = next(loss)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(loss, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probs, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probs: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probs(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probs, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probs(probs, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probs, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probs: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    print(f'function: {lovasz_softmax_flat.__name__}, probs shape: {probs.shape}')
    print(f'function: {lovasz_softmax_flat.__name__}, labels shape: {labels.shape}')
    print(f'function: {lovasz_softmax_flat.__name__}, classes: {classes}')

    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == 'present' and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probs(probs, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    print(f'function: {flatten_probs.__name__}, probs shape: {probs.shape}, MY: [1,20,64,1024]')
    print(f'function: {flatten_probs.__name__}, labels shape: {labels.shape}, MY: [1,64,1024]')
    print(f'function: {flatten_probs.__name__}, ignore: {ignore}, MY: 0')

    if probs.dim() == 3:
        print(f'function: {flatten_probs.__name__}, THIS SHOULD NOT HAPPEN')

        # assumes output of a sigmoid layer
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)

    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probs, labels
    valid = torch.ne(labels, ignore)
    print(f'function: {flatten_probs.__name__}, valid shape: {valid.shape}, MY: [65536]')
    print(
        f'function: {flatten_probs.__name__}, valid.nonzero().squeeze() shape: {valid.squeeze().shape}, MY: [65536]')
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probs, labels):
        return lovasz_softmax(probs, labels, self.classes, self.per_image, self.ignore)
