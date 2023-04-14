from omegaconf import DictConfig
from torch.utils.data import Dataset


class PartitionDataset(Dataset):
    def __init__(self, dataset_path: str, project_name: str, cfg: DictConfig, split: str,
                 size: int = None, selection_mode: bool = False, al_experiment: bool = False,
                 sequences: iter = None, resume: bool = False):
        pass
