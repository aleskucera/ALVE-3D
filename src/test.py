import os

import torch.nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .dataset import SemanticDataset
from .learning import Tester


def test_model(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_ds = SemanticDataset(cfg.path.kitti, cfg.kitti, 'valid', cfg.test.dataset_size)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=os.cpu_count() // 2)

    model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
    model = torch.load(model_path)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    tester = Tester(model, test_loader, criterion, device)
    test_data = tester.test()

    print(test_data)
