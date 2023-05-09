import logging

import torch
import wandb
import omegaconf
import numpy as np
from omegaconf import DictConfig

from src.utils.wb import push_artifact
from src.learn.trainer import SemanticTrainer
from src.datasets import SemanticDataset, SemanticKITTIDataset

log = logging.getLogger(__name__)


def train_model_passive(cfg: DictConfig, device: torch.device) -> None:
    info = f'{cfg.model.architecture}_{cfg.ds.name}_{cfg.train.loss}'
    model_name = f'Model_{info}'
    history_name = f'History_{info}'

    project_name = cfg.project_name if cfg.project_name is not None else 'Baseline'

    train_ds = SemanticDataset(split='train',
                               cfg=cfg.ds,
                               dataset_path=cfg.ds.path,
                               project_name=info,
                               num_clouds=cfg.train.dataset_size,
                               al_experiment=False)
    val_ds = SemanticDataset(split='val',
                             cfg=cfg.ds,
                             dataset_path=cfg.ds.path,
                             project_name=info,
                             num_clouds=cfg.train.dataset_size,
                             al_experiment=False)

    trainer = SemanticTrainer(cfg=cfg,
                              train_ds=train_ds,
                              val_ds=val_ds,
                              device=device)

    print(f'Labeled percentage of selection {train_ds.statistics["labeled_ratio"] * 100:.2f}%')

    with wandb.init(project=project_name,
                    group=cfg.ds.name, name=f'{cfg.model.architecture}-{cfg.train.loss}',
                    config=omegaconf.OmegaConf.to_container(cfg, resolve=True)):
        trainer.train()
        push_artifact(model_name, trainer.best_model['state_dict'], 'model')
        push_artifact(history_name, trainer.history, 'history')


def train_semantickitti_original(cfg: DictConfig, device: torch.device) -> None:
    info = f'{cfg.model.architecture}_{cfg.ds.name}_{cfg.train.loss}'
    model_name = f'Model_{info}'
    history_name = f'History_{info}'

    train_ds = SemanticKITTIDataset(cfg.ds, 'train')
    val_ds = SemanticKITTIDataset(cfg.ds, 'val')

    weights = calculate_weights(cfg.ds.content, cfg.ds.learning_map)

    trainer = SemanticTrainer(cfg=cfg,
                              train_ds=train_ds,
                              val_ds=val_ds,
                              device=device,
                              weights=weights)

    with wandb.init(project='Baseline',
                    group=cfg.ds.name, name=f'{cfg.model.architecture}-{cfg.train.loss}',
                    config=omegaconf.OmegaConf.to_container(cfg, resolve=True)):
        trainer.train()
        push_artifact(model_name, trainer.best_model['state_dict'], 'model')
        push_artifact(history_name, trainer.history, 'history')


def calculate_weights(content: dict, mapping: dict):
    train_labels = np.unique(np.array(list(mapping.values())))
    sums = np.zeros((len(train_labels)), dtype=np.float32)
    for key, value in content.items():
        sums[mapping[key]] += value
    weights = 1 / (sums + 1e-6)
    return weights

# def train_model_active(cfg: DictConfig, experiment: Experiment, device: torch.device) -> None:
#     """ Trains a semantic segmentation model using active learning. This function executes only one iteration of the
#     training. The training is executed in the following steps:
#         1. Load the model and the history from the artifacts.
#         2. Load the dataset and initialize all labels to unlabeled.
#         3. Create a Selector object.
#         4. Download the information about the labeled voxels from the artifact and using Selector,
#            label the voxels that has been previously selected.
#         5. Train the model.
#         6. Save the model during the training and the history of the training to the artifact.
#
#     :param cfg: The configuration object containing the dataset parameters.
#     :param experiment: The experiment object containing the names of the artifacts to be used.
#     :param device: The device to be used for the training.
#     """
#
#     criterion = cfg.active.strategy
#     selection_objects = cfg.active.cloud_partitions
#
#     model_version = cfg.active.model_version
#     selection_version = cfg.active.selection_version
#
#     load_model = cfg.load_model if 'load_model' in cfg else False
#
#     # Load Datasets
#     train_ds = SemanticDataset(split='train', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=experiment.info,
#                                num_clouds=cfg.train.dataset_size, al_experiment=True, selection_mode=False)
#     val_ds = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path, project_name=experiment.info,
#                              num_clouds=cfg.train.dataset_size, al_experiment=True, selection_mode=False)
#
#     # Load Selector for selecting labeled voxels
#     selector = get_selector(selection_objects=selection_objects, criterion=criterion,
#                             dataset_path=cfg.ds.path, project_name=experiment.info,
#                             cloud_paths=train_ds.cloud_files, device=device,
#                             cfg=cfg)
#
#     # Load selected voxels from W&B
#     selection_artifact = cfg.active.selection if cfg.active.selection is not None else \
#         f'{experiment.selection}:{selection_version}'
#     selection = load_artifact(selection_artifact)
#
#     # Label train dataset
#     selector.load_voxel_selection(selection, train_ds)
#
#     # Load model from W&B
#     model_artifact = cfg.active.model if cfg.active.model is not None else \
#         f'{experiment.model}:{model_version}'
#     model = load_artifact(model_artifact, device=device) if load_model else None
#
#     # Log dataset statistics and calculate the weights for the loss function from them
#     class_distribution = log_dataset_statistics(cfg=cfg, dataset=train_ds, artifact_name=experiment.dataset_stats)
#     _ = log_dataset_statistics(cfg=cfg, dataset=val_ds, val=True)
#     weights = 1 / (class_distribution + 1e-6)
#
#     # Train model
#     trainer = SemanticTrainer(cfg=cfg, train_ds=train_ds, val_ds=val_ds, device=device,
#                               model=model, model_name=experiment.model, history_name=experiment.history)
#     trainer.train()
