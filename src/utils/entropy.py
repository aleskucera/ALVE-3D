import torch
import matplotlib.pyplot as plt
import numpy as np


# visualize entropy
def _visualize_entropy(entropy, title):
    plt.imshow(entropy)
    plt.colorbar()
    plt.title(title)
    plt.show()


# test calculation of entropy
def _prob2entropy(p, axis=1, eps=1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    h = torch.sum(-p * torch.log10(p), dim=axis)
    return h

