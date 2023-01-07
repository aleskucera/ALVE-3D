#!/usr/bin/env python
import os
import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("WARNING: Can't import open3d.")

from src import SemanticDataset, set_paths, LaserScan, ScanVis, create_global_cloud, create_superpoints

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'paths':
        show_paths(cfg)
    elif cfg.action == 'dataset':
        show_dataset(cfg)
    elif cfg.action == 'global_cloud':
        show_global_cloud(cfg)
    elif cfg.action == 'superpoints':
        superpoints(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def show_paths(cfg: DictConfig) -> None:
    print('\nPaths dynamically generated to DictConfig object:')
    for name, path in cfg.path.items():
        print(f'\t{name}: {path}')
    print('')


def show_dataset(cfg: DictConfig) -> None:
    # dataset attributes
    size = None
    sequences = [0]
    indices = None

    # create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=sequences, cfg=cfg.ds,
                              split='train', indices=indices, size=size)

    # create semantic laser scan
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    # create scan visualizer
    vis = ScanVis(scan=scan, scans=dataset.points, labels=dataset.labels, raw_cloud=True, instances=False,
                  projection=False)
    vis.run()


def show_global_cloud(cfg: DictConfig) -> None:
    sequence = 7
    file_name = f'global_cloud.npz'
    path = os.path.join(cfg.ds.path, 'sequences', f'{sequence:02d}', file_name)
    create_global_cloud(cfg, sequence, path)

    data = np.load(path)
    cloud, color = data['cloud'], data['color']

    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)
    vis = ScanVis(scan=scan, scans=[cloud], scan_colors=[color], raw_cloud=True, instances=False, projection=False)
    vis.run()


def superpoints(cfg: DictConfig):
    sequence = 4
    number_of_superpoints = 20000

    path = os.path.join(cfg.ds.path, 'sequences', f'{sequence:02d}', 'superpoints')
    os.makedirs(path, exist_ok=True)

    create_superpoints(cfg=cfg, sequence=sequence, num_points=number_of_superpoints, directory=path)

    dataset = SemanticDataset(dataset_path=cfg.ds.path, sequences=[sequence], cfg=cfg.ds, split='train')

    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)
    vis = ScanVis(scan=scan, scans=dataset.points, superpoints=dataset.superpoints, raw_cloud=False)
    vis.run()


# def select_indices(cfg: DictConfig):
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
#     test_ds = SemanticDataset(cfg.ds.path, cfg.ds, split='valid')
#
#     test_loader = DataLoader(test_ds, batch_size=10, shuffle=False, num_workers=4)
#
#     model_path = os.path.join(cfg.path.models, 'pretrained', cfg.test.model_name)
#     model = SalsaNext(cfg.ds.num_classes)
#     model.load_state_dict(torch.load(model_path))
#     model = model.to(device)
#
#     selector = Selector(model=model, loader=test_loader, device=device)
#     entropies, indices = tester.calculate_entropies()
#
#     print(f'\nFirst 10 entropies: \n{entropies[:10]}')
#     print(f'\nFirst 10 indices: \n{indices[:10]}')
#     print(f'\nFirst 10 ious: \n{ious[:10]}')


if __name__ == '__main__':
    main()
