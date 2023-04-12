import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset


class BaseSampleSelector:
    """ Base class for sample selectors.

    :param dataset_path: Path to the dataset
    :param device: Device to use for calculations
    :param dataset_percentage: Percentage of the dataset to be labeled in each iteration (default: 10)
    """

    def __init__(self, dataset_path: str, device: torch.device, dataset_percentage: float = 10):
        self.device = device
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
    def __init__(self, dataset_path: str, device: torch.device, dataset_percentage: float = 10):
        super().__init__(dataset_path, device, dataset_percentage)

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the samples to be labeled. This is done by executing the following steps:

        1. Create a mask of the samples that are already labeled (selection_mask == 1)
        2. Calculate the number of samples to be labeled in this iteration
        3. Select random samples from the dataset that are not already labeled (selection_mask == 0)
        4. Label the selected samples by calling dataset.label_samples() function

        :param dataset: Dataset object
        :param model: Model object (not used in this selector)
        """

        selection_mask = torch.from_numpy(dataset.selection_mask).to(self.device)
        selection_size = int(self.dataset_percentage * dataset.get_length() / 100)

        # Select random samples from the dataset that are not already labeled (selection_mask == 0)
        sample_indices = torch.randperm(dataset.get_length(), device=self.device)
        sample_indices = sample_indices[~selection_mask]

        # Select the first selection_size samples
        sample_indices = sample_indices[:selection_size]

        # Label the selected samples
        dataset.label_samples(sample_indices.cpu().numpy())


class EntropySampleSelector(BaseSampleSelector):
    def __init__(self, dataset_path: str, device: torch.device, dataset_percentage: float = 10):
        super().__init__(dataset_path, device, dataset_percentage)
        self.eps = 1e-6

    def select(self, dataset: Dataset, model: nn.Module):
        """ Select the samples to be labeled. This is done by executing the following steps:

        1. Create a mask of the samples that are already labeled (selection_mask == 1)
        2. Calculate the number of samples to be labeled in this iteration
        3. Iterate over the dataset and calculate the entropy of the model output for each sample
        4. Select the samples with the highest entropy that are not already labeled (selection_mask == 0)
        5. Label the selected samples by calling dataset.label_samples() function

        :param dataset: Dataset object
        :param model: Model object used for selecting the samples
        """

        selection_mask = torch.from_numpy(dataset.selection_mask).to(self.device)
        selection_size = int(self.dataset_percentage * dataset.get_length() / 100)

        entropies = torch.zeros(dataset.get_length(), device=self.device)
        sample_map = torch.arange(dataset.get_length(), device=self.device)

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for i in tqdm(range(dataset.get_true_length())):
                proj_image, _, _, _ = dataset.get_item(i)
                proj_image = proj_image.to(self.device)

                # Forward pass
                model_output = model(proj_image.unsqueeze(0))
                model_output = model_output.squeeze(0).flatten(start_dim=1).permute(1, 0)

                # Calculate the entropy of the model output
                prob = torch.clamp(model_output, min=self.eps, max=1 - self.eps)
                entropy = torch.sum(- prob * torch.log(prob), dim=1).mean()
                entropies[i] = entropy

            # Select the entropies and indices of the samples that are not already labeled
            entropies = entropies[~selection_mask]
            sample_map = sample_map[~selection_mask]

            # Select the samples with the highest entropy
            indices = torch.argsort(entropies, descending=True)
            sample_map = sample_map[indices]
            sample_map = sample_map[:selection_size]

            dataset.label_samples(sample_map)
