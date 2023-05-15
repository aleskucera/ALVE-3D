import logging

import torch
import wandb
import omegaconf
from omegaconf import DictConfig

from src.selection import get_selector
from src.learn.trainer import SemanticTrainer
from src.utils.wb import push_artifact, pull_artifact
from src.datasets.semantic_dataset import SemanticDataset
from src.utils.log import log_selection_metric_statistics, log_dataset_statistics

log = logging.getLogger(__name__)


def train_model_active(cfg: DictConfig, device: torch.device) -> None:
    # model_artifact = 'aleskucera/AL-Seed/SalsaNext_KITTI360:v1'
    # selection_artifact = 'aleskucera/AL-Seed/Seed_KITTI360:v1'
    # percentages = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    percentages = cfg.active.percentages
    model_artifact = cfg.active.model_artifact
    selection_artifact = cfg.active.selection_artifact

    info = f'{cfg.active.strategy}_{cfg.active.cloud_partitions}_{cfg.model.architecture}_{cfg.ds.name}'
    model_name = f'Model_{info}'
    selection_name = f'Selection_{info}'
    history_name = f'History_{info}'
    metric_stats = f'MetricStats_{info}'
    weighted_metric_stats = f'WeightedMetricStats_{info}'
    dataset_stats = f'DatasetStats_{info}'

    wandb_project = 'AL-KITTI360-2'
    wandb_group = f'{cfg.active.strategy}_{cfg.active.cloud_partitions}'

    # Create datasets
    train_ds = SemanticDataset(split='train',
                               cfg=cfg.ds,
                               dataset_path=cfg.ds.path,
                               project_name=info,
                               num_clouds=cfg.train.dataset_size,
                               al_experiment=True,
                               selection_mode=False,
                               filter_type=cfg.active.filter_type)

    val_ds = SemanticDataset(split='val',
                             cfg=cfg.ds,
                             dataset_path=cfg.ds.path,
                             project_name=info,
                             num_clouds=cfg.train.dataset_size,
                             al_experiment=True,
                             selection_mode=False)

    # Load Selector for selecting labeled voxels
    selector = get_selector(cfg=cfg,
                            project_name=info,
                            cloud_paths=train_ds.cloud_files,
                            device=device)

    # Create trainer
    trainer = SemanticTrainer(cfg=cfg,
                              train_ds=train_ds,
                              val_ds=val_ds,
                              device=device)

    selection = pull_artifact(selection_artifact, device=torch.device('cpu'))
    selector.load_voxel_selection(selection, train_ds)

    model_state_dict = pull_artifact(model_artifact, device=device)
    trainer.model.load_state_dict(model_state_dict)
    selector.model.load_state_dict(model_state_dict)

    log.info(f"Labeled percentage of seed selection {train_ds.statistics['labeled_ratio'] * 100:.2f}%")

    for p in percentages:
        # cfg.active.percentage = p
        with wandb.init(project=wandb_project,
                        group=wandb_group,
                        name=f'Iteration-{p}%',
                        config=omegaconf.OmegaConf.to_container(cfg, resolve=True)):

            # Select voxels
            if trainer.best_model['state_dict'] is not None:
                selector.model.load_state_dict(trainer.best_model['state_dict'])
            selection, normal_metric_statistics, weighted_metric_statistics = selector.select(train_ds, p)
            selector.load_voxel_selection(selection, train_ds)

            push_artifact(selection_name, selection, 'selection')
            if normal_metric_statistics is not None:
                log_selection_metric_statistics(cfg, normal_metric_statistics, metric_stats)

            if weighted_metric_statistics is not None:
                log_selection_metric_statistics(cfg, weighted_metric_statistics, weighted_metric_stats, weighted=True)
            log_dataset_statistics(cfg, train_ds, dataset_stats)

            # Train model on selected voxels
            print(f'Labeled percentage of selection {trainer.train_ds.statistics["labeled_ratio"] * 100:.2f}%')
            trainer.train()

            push_artifact(model_name, trainer.best_model['state_dict'], 'model')
            push_artifact(history_name, trainer.history, 'history')
            trainer.reset()


def train_iteration(cfg: DictConfig, device: torch.device) -> None:
    raise NotImplementedError


def create_seed(cfg: DictConfig, device: torch.device) -> None:
    selection_name = f'Seed_{cfg.ds.name}'
    model_name = f'{cfg.model.architecture}_{cfg.ds.name}'
    project_name = f'Seed_{cfg.active.strategy}'

    # Create datasets
    train_ds = SemanticDataset(split='train',
                               cfg=cfg.ds,
                               dataset_path=cfg.ds.path,
                               project_name=project_name,
                               num_clouds=cfg.train.dataset_size,
                               al_experiment=True,
                               selection_mode=False)

    val_ds = SemanticDataset(split='val',
                             cfg=cfg.ds,
                             dataset_path=cfg.ds.path,
                             project_name=project_name,
                             num_clouds=cfg.train.dataset_size,
                             al_experiment=True,
                             selection_mode=False)

    # Load Selector for selecting labeled voxels
    selector = get_selector(cfg=cfg,
                            project_name=project_name,
                            cloud_paths=train_ds.cloud_files,
                            device=device)

    # Create trainer
    trainer = SemanticTrainer(cfg=cfg,
                              train_ds=train_ds,
                              val_ds=val_ds,
                              device=device)

    with wandb.init(project='AL-Seed',
                    group=f'{cfg.active.strategy}_{cfg.active.cloud_partitions}',
                    config=omegaconf.OmegaConf.to_container(cfg, resolve=True)):
        selection, normal_metric_statistics, weighted_metric_statistics = selector.select(train_ds,
                                                                                          cfg.active.seed_percentage)

        selector.load_voxel_selection(selection, train_ds)

        push_artifact(selection_name, selection, 'selection')
        log_dataset_statistics(cfg, train_ds)

        # Train model on selected voxels
        print(f'Labeled percentage of selection {train_ds.statistics["labeled_ratio"] * 100:.2f}%')
        trainer.train()

        push_artifact(model_name, trainer.best_model['state_dict'], 'model')
