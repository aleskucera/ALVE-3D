import logging

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from .utils import parse_data

log = logging.getLogger(__name__)


class Selector:
    def __init__(self, model, dataset, device, num_classes: int = 20, criterion: str = "entropy"):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.criterion = criterion

        acc = MulticlassAccuracy(num_classes=num_classes, ignore_index=0, validate_args=False).to(device)
        iou = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0, validate_args=False).to(device)
        self.metric_collection = MetricCollection([acc, iou])

        self.entropies, self.indices, self.jaccard_indices = self.calculate_entropies()

    def calculate_entropies(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entropies = []
        indices = []
        jaccard_indices = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.loader):
                image_batch, label_batch, idx_batch = parse_data(data, self.device)
                output = self.model(image_batch)

                entropy_batch = calculate_entropy(output)
                entropies.append(entropy_batch)

                metrics = self.metric_collection(output, label_batch)
                jaccard_index = metrics["MulticlassJaccardIndex"]
                jaccard_indices.append(jaccard_index)

                indices.append(idx_batch)

        # Create tensor of entropies and indices
        entropies = torch.cat(entropies)
        indices = torch.cat(indices)
        jaccard_indices = torch.tensor(jaccard_indices, device=self.device)
        order = torch.argsort(entropies, descending=True)
        return entropies[order], indices[order], jaccard_indices[order]

    def select_entropies(self, limit: int):
        if self.criterion == "entropy":
            # Select samples with the entropy higher than the limit
            selected_indices = self.indices[self.entropies > limit]
        elif self.criterion == "quantity":
            # Select first n samples
            selected_indices = self.indices[:limit]
        else:
            raise ValueError("Unknown criterion")
        return selected_indices


def calculate_entropy(output, eps=1e-6) -> torch.Tensor:
    prob = torch.nn.functional.softmax(output, dim=1)
    prob = torch.clamp(prob, eps, 1.0 - eps)
    h = - torch.sum(prob * torch.log10(prob), dim=1)
    return h.mean(axis=(1, 2))
