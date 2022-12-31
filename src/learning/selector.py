import logging

import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from .utils import parse_data

log = logging.getLogger(__name__)


class Selector:
    def __init__(self, model, loader, device, num_classes=20):
        self.model = model
        self.device = device
        self.loader = loader
        acc = MulticlassAccuracy(num_classes=num_classes, ignore_index=0, validate_args=False).to(device)
        iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0, validate_args=False).to(device)
        self.metric_collection = MetricCollection([acc, iou])

    def calculate_entropies(self) -> tuple[torch.Tensor, torch.Tensor]:
        entropies = []
        indices = []
        ious = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.loader):
                image_batch, label_batch, idx_batch = parse_data(data, self.device)
                output = self.model(image_batch)

                entropy_batch = calculate_entropy(output)
                entropies.append(entropy_batch)

                metrics = self.metric_collection(output, label_batch)
                # iou = metrics["MulticlassJaccardIndex"]
                # ious.append(iou)

                indices.append(idx_batch)

        # Create tensor of entropies and indices
        entropies = torch.cat(entropies)
        indices = torch.cat(indices)
        # ious = torch.tensor(ious, device=self.device)
        order = torch.argsort(entropies, descending=True)
        return entropies[order], indices[order]


def sort_entropies(entropies: np.ndarray) -> np.ndarray:
    order = np.argsort(entropies[:, 0])[::-1]
    return entropies[order]


def create_entropy_batch(output: torch.Tensor, idx_batch: torch.Tensor) -> torch.Tensor:
    entropy_batch = calculate_entropy(output)

    # Concatenate entropy with indices
    data_batch = torch.vstack([idx_batch, entropy_batch]).T

    # entropy_batch = entropy_batch.cpu().numpy()[..., np.newaxis]
    #
    # idx_batch = idx_batch.cpu().numpy()[..., np.newaxis]
    #
    # data_batch = np.concatenate((entropy_batch, idx_batch), axis=1)
    return data_batch


def calculate_entropy(output, eps=1e-6) -> torch.Tensor:
    prob = torch.nn.functional.softmax(output, dim=1)
    prob = torch.clamp(prob, eps, 1.0 - eps)
    h = - torch.sum(prob * torch.log10(prob), dim=1)
    return h.mean(axis=(1, 2))
