import os
from typing import Iterable

import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset


def calculate_entropy(output, eps=1e-6) -> torch.Tensor:
    prob = torch.nn.functional.softmax(output, dim=1)
    prob = torch.clamp(prob, eps, 1.0 - eps)
    h = - torch.sum(prob * torch.log10(prob), dim=1)
    return h.mean(axis=(1, 2))


class BaseSampleSelector:
    def __init__(self, dataset_path: str, sequences: Iterable[int], device: torch.device,
                 dataset_percentage: float = 10):
        self.device = device
        self.sequences = sequences
        self.dataset_path = dataset_path
        self.dataset_percentage = dataset_percentage

        self.num_samples = 0
        self.samples_labeled = 0

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the samples to be labeled """

        raise NotImplementedError

    def is_finished(self):
        return self.samples_labeled == self.num_samples


class RandomSampleSelector(BaseSampleSelector):
    def __init__(self, dataset_path: str, sequences: Iterable[int], device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, sequences, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the samples to be labeled """

        selection_mask = torch.from_numpy(dataset.selection_mask).to(self.device)
        selection_size = int(self.dataset_percentage * dataset.get_true_length() / 100)

        # Select random samples from the dataset that are not already labeled (selection_mask == 0)
        sample_indices = torch.randperm(dataset.get_true_length(), device=self.device)
        sample_indices = sample_indices[~selection_mask]

        # Select the first selection_size samples
        sample_indices = sample_indices[:selection_size]

        # Label the selected samples
        # TODO: Label the samples


class EntropySampleSelector(BaseSampleSelector):
    def __init__(self, dataset_path: str, sequences: Iterable[int], device: torch.device,
                 dataset_percentage: float = 10):
        super().__init__(dataset_path, sequences, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        selection_size = int(self.dataset_percentage * dataset.get_true_length() / 100)
        selection_mask = torch.from_numpy(dataset.selection_mask).to(self.device)
        sample_map = torch.arange(dataset.get_true_length(), device=self.device)
        entropy = torch.zeros(dataset.get_true_length(), device=self.device)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for i in tqdm(range(dataset.get_true_length())):
                proj_image, proj_label, _, _, _ = dataset.get_item(i)
                proj_image, proj_label = proj_image.to(self.device), proj_label.to(self.device)

                model_output = model(proj_image.unsqueeze(0))
                model_output = model_output.squeeze(0)

                # Calculate the entropy of the model output
                # TODO: Calculate the entropy
                entropy_value = 0

                entropy[i] = entropy_value

            # Select the entropies and indices of the samples that are not already labeled
            entropy = entropy[~selection_mask]
            sample_map = sample_map[~selection_mask]

            # Sort the entropies and indices
            indices = torch.argsort(entropy, descending=True)
            sample_map = sample_map[indices]

            # Select the first selection_size samples
            sample_map = sample_map[:selection_size]

            # Label the selected samples
            # TODO: Label the samples
