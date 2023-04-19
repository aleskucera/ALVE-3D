#!/usr/bin/env python
import os
import logging

import h5py
import torch
import wandb
import hydra
import omegaconf
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils import set_paths, visualize_global_cloud, visualize_cloud_values, visualize_cloud
from src.utils import plot, bar_chart, grouped_bar_chart, plot_confusion_matrix, map_colors, ScanInterface, \
    CloudInterface, Experiment
from src.datasets import SemanticDataset, PartitionDataset, get_parser
from src.kitti360 import KITTI360Converter, create_kitti360_config
from src.laserscan import LaserScan, ScanVis
from src.models import get_model

import matplotlib

matplotlib.use('Qt5Agg')

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    if cfg.action == 'config_object':
        show_hydra_config(cfg)
    elif cfg.action == 'experiment_object':
        show_experiment(cfg)
    elif cfg.action == 'test_model':
        test_model(cfg)
    elif cfg.action == 'visualize_dataset_scans':
        visualize_dataset_scans(cfg)
    elif cfg.action == 'visualize_dataset_clouds':
        visualize_dataset_clouds(cfg)
    elif cfg.action == 'visualize_superpoints':
        visualize_superpoints(cfg)
    elif cfg.action == 'visualize_experiment':
        visualize_experiment(cfg)
    elif cfg.action == 'log_sequence':
        log_sequence(cfg)
    elif cfg.action == 'create_kitti360_config':
        create_kitti360_config()
    elif cfg.action == 'visualize_kitti360_conversion':
        visualize_kitti360_conversion(cfg)
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


def show_hydra_config(cfg: DictConfig) -> None:
    """ Show how to use Hydra for configuration management.
    :param cfg: Configuration object.
    """
    print('\nThis project uses Hydra for configuration management.')
    print(f'Configuration object type is: {type(cfg)}')
    print(f'Configuration content is separated into groups:')
    for group in cfg.keys():
        print(f'\t{group}')

    print(cfg.train.batch_size)

    print('\nPaths dynamically generated to DictConfig object:')
    for name, path in cfg.path.items():
        print(f'\t{name}: {path}')
    print('')


def show_experiment(cfg: DictConfig) -> None:
    cfg.action = 'train_semantic_active'
    experiment = Experiment(cfg)
    print(experiment)


def test_model(cfg: DictConfig) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # artifact_path = cfg.artifact_path
    # model_name = artifact_path.split('/')[-1].split(':')[0]
    # dataset = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path,
    #                           project_name='demo', num_scans=None)
    wandb_cfg = wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(project='demo', config=wandb_cfg, job_type='test_model') as run:
        pass
        # artifact_dir = wandb.use_artifact(artifact_path).download()
        # model = torch.load(os.path.join(artifact_dir, f'{model_name}.pt'), map_location=device)
        # model_state_dict = model['model_state_dict']
        # model = get_model(cfg=cfg, device=device)
        # model.load_state_dict(model_state_dict)
        #
        # parser = get_parser('semantic', device)
        #
        # model.eval()
        # with torch.no_grad():
        #     for i, scan_file in enumerate(dataset.scans):
        #         scan, label, _, _, _ = dataset[i]
        #         scan = torch.from_numpy(scan).to(device).unsqueeze(0)
        #         pred = model(scan)
        #         pred = pred.argmax(dim=1)
        #         pred = pred.cpu().numpy().squeeze()
        #
        #         label = map_colors(label, cfg.ds.color_map_train)
        #         pred = map_colors(pred, cfg.ds.color_map_train)
        #
        #         # Plot labels and predictions
        #         fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
        #         ax[0].imshow(label)
        #         ax[0].set_title("Labels")
        #         ax[1].imshow(pred)
        #         ax[1].set_title("Predictions")
        #         plt.show()
        #         # time.sleep(1)


def visualize_dataset_scans(cfg: DictConfig):
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)

    # Create scan object
    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True,
                     H=cfg.ds.projection.H, W=cfg.ds.projection.W, fov_up=cfg.ds.projection.fov_up,
                     fov_down=cfg.ds.projection.fov_down)

    # Visualizer
    vis = ScanVis(scan=scan, scans=dataset.scan_files, labels=dataset.scan_files, raw_cloud=True, instances=False)
    vis.run()


def visualize_dataset_clouds(cfg: DictConfig):
    size = cfg.size if 'size' in cfg else None
    split = cfg.split if 'split' in cfg else 'train'
    sequences = [cfg.sequence] if 'sequence' in cfg else None

    # Create dataset
    dataset = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo',
                              cfg=cfg.ds, split=split, num_clouds=size, sequences=sequences)
    cloud_interface = CloudInterface(project_name='demo', label_map=cfg.ds.learning_map)

    for cloud_file in dataset.clouds:
        points = cloud_interface.read_points(cloud_file)
        labels = cloud_interface.read_labels(cloud_file)

        colors = map_colors(labels, cfg.ds.color_map_train)
        visualize_cloud(points, colors)


def visualize_superpoints(cfg: DictConfig) -> None:
    sequence = cfg.sequence if 'sequence' in cfg else 3

    cloud_dir = os.path.join(cfg.ds.path, 'sequences', f'{sequence:02d}', 'voxel_clouds')
    cloud_files = sorted([os.path.join(cloud_dir, f) for f in os.listdir(cloud_dir) if f.endswith('.h5')])

    for cloud in cloud_files:
        with h5py.File(cloud, 'r') as f:
            points = np.asarray(f['points'])
            superpoints = np.asarray(f['superpoints'])

            visualize_cloud_values(points, superpoints, random_colors=True)


def visualize_experiment(cfg: DictConfig) -> None:
    project_name = 'active_learning_average_entropy_voxels (test)'

    percentages = ['5%', '10%', '15%']
    history_versions = ['v0', 'v1', 'v2']
    dataset_statistics_versions = ['v0', 'v1', 'v2']

    ignore_index = cfg.ds.ignore_index
    label_names = [v for k, v in cfg.ds.labels_train.items() if k != ignore_index]

    data = dict()

    with wandb.init(project='demo'):
        for h_v, d_v, p in zip(history_versions, dataset_statistics_versions, percentages):
            history_artifact = wandb.use_artifact(f'aleskucera/{project_name}/history:{h_v}')
            dataset_statistics_artifact = wandb.use_artifact(f'aleskucera/{project_name}/dataset_statistics:{d_v}')

            history_artifact_dir = history_artifact.download()
            dataset_statistics_artifact_dir = dataset_statistics_artifact.download()

            history_path = os.path.join(history_artifact_dir, f'history.pt')
            dataset_statistics_path = os.path.join(dataset_statistics_artifact_dir, f'dataset_statistics.pt')

            history = torch.load(history_path)
            dataset_statistics = torch.load(dataset_statistics_path)

            for key, value in history.items():
                if key not in data:
                    data[key] = dict()
                data[key][p] = value

            for key, value in dataset_statistics.items():
                if key not in data:
                    data[key] = dict()
                data[key][p] = value

        # Visualize the full dataset class distribution
        dataset_distributions = data['class_distribution']
        distributions_matrix = np.vstack(list(dataset_distributions.values()))
        assert np.all(distributions_matrix == distributions_matrix[0]), 'Distributions are not equal.'
        bar_chart(distributions_matrix[0], label_names, 'Mass [%]')
        grouped_bar_chart(data['labeled_class_distribution'], label_names, 'Mass [%]')
        grouped_bar_chart(data['class_labeling_progress'], label_names, 'Percentage labeled [%]')

        max_mious = [np.max(m) for m in data['miou_val'].values()]
        plot({'max': max_mious}, 'percentages', 'Maximum mIoU [%]')

        confusion_matrix = data['confusion_matrix']['15%'][-1].numpy()
        plot_confusion_matrix(confusion_matrix, label_names)


def log_sequence(cfg: DictConfig) -> None:
    """ Log sample from dataset sequence to Weights & Biases.

    :param cfg: Configuration object.
    """

    sequence = cfg.sequence if 'sequence' in cfg else 3

    train_ds = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo', split='train',
                               sequences=[sequence], cfg=cfg.ds, active_mode=False)
    val_ds = SemanticDataset(dataset_path=cfg.ds.path, project_name='demo', split='val',
                             sequences=[sequence], cfg=cfg.ds, active_mode=False)

    scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True)

    if len(train_ds) > 0:
        with wandb.init(project='Sequence Sample Visualization', group=cfg.ds.name,
                        name=f'Sequence {sequence} - train'):
            _log_sequence(train_ds, scan)
    else:
        log.info(f'Train dataset for sequence {sequence} is empty.')

    if len(val_ds) > 0:
        with wandb.init(project='Sequence Sample Visualization', group=cfg.ds.name, name=f'Sequence {sequence} - val'):
            _log_sequence(val_ds, scan)
    else:
        log.info(f'Validation dataset for sequence {sequence} is empty.')


def _log_sequence(dataset, scan):
    i = np.random.randint(0, len(dataset))
    scan.open_scan(dataset.scan_files[i])
    scan.open_label(dataset.label_files[i])

    cloud = np.concatenate([scan.points, scan.color * 255], axis=1)
    label = np.concatenate([scan.points, scan.sem_label_color * 255], axis=1)

    wandb.log({'Point Cloud': wandb.Object3D(cloud),
               'Point Cloud Label': wandb.Object3D(label),
               'Projection': wandb.Image(scan.proj_color),
               'Projection Label': wandb.Image(scan.proj_sem_color)})

    log.info(f'Logged scan: {dataset.scan_files[i]}')
    log.info(f'Logged label: {dataset.label_files[i]}')


def visualize_kitti360_conversion(cfg: DictConfig):
    """ Visualize KITTI360 conversion.
    :param cfg: Configuration object.
    """

    converter = KITTI360Converter(cfg)
    converter.visualize()


if __name__ == '__main__':
    main()
