import os
import logging

import torch.nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .dataset import SemanticDataset
from .learning import Tester

log = logging.getLogger(__name__)


def test_model(cfg: DictConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info(f'Using device {device}')

    test_ds = SemanticDataset(cfg.path.kitti, cfg.kitti, split='valid', size=cfg.test.dataset_size)
    log.info(f'Test dataset size: {len(test_ds)}')

    test_loader = DataLoader(test_ds, batch_size=4, shuffle=True, num_workers=os.cpu_count() // 2)

    model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
    model = torch.load(model_path).to(device)

    tester = Tester(model=model, test_loader=test_loader, device=device,
                    num_classes=33, output_path=cfg.path.output)
    entropies = tester.calculate_entropies()

    print(entropies.shape)

    entropies = entropies[:10]
    print(entropies)

    for _, i in entropies:
        print(test_ds.points[int(i)])
