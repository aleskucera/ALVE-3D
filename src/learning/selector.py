import logging

import torch
import numpy as np
from tqdm import tqdm

from .utils import parse_data

log = logging.getLogger(__name__)


class Selector:
    def __init__(self, model, loader, device):
        self.model = model
        self.device = device
        self.loader = loader

    def calculate_entropies(self) -> np.ndarray:
        entropies = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.loader):
                image_batch, _, idx_batch = parse_data(data, self.device)
                output = self.model(image_batch)['out']

                # Calculate entropy
                entropy_batch = create_entropy_batch(output, idx_batch)
                entropies.append(entropy_batch)

        entropies = np.concatenate(entropies, axis=0)
        entropies = sort_entropies(entropies)
        return entropies


def sort_entropies(entropies: np.ndarray) -> np.ndarray:
    order = np.argsort(entropies[:, 0])[::-1]
    return entropies[order]


def create_entropy_batch(output: torch.Tensor, idx_batch: torch.Tensor) -> np.ndarray:
    entropy_batch = calculate_entropy(output)
    entropy_batch = entropy_batch.cpu().numpy()[..., np.newaxis]

    idx_batch = idx_batch.cpu().numpy()[..., np.newaxis]

    data_batch = np.concatenate((entropy_batch, idx_batch), axis=1)
    return data_batch


def calculate_entropy(output, eps=1e-6) -> torch.Tensor:
    prob = torch.nn.functional.softmax(output, dim=1)
    prob = torch.clamp(prob, eps, 1.0 - eps)
    h = - torch.sum(prob * torch.log10(prob), dim=1)
    return h.mean(axis=(1, 2))
