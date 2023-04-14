from omegaconf import DictConfig
from torch.utils.data import Dataset


class SemanticDataset(Dataset):
    def __init__(self, dataset_path: str, project_name: str, cfg: DictConfig, split: str, size: int = None,
                 active_mode: bool = False, sequences: iter = None, resume: bool = False):
        pass
