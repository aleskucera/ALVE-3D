import os
import logging
import numpy as np

from omegaconf import DictConfig
from src.datasets import SemanticDataset
from src.utils.visualize import bar_chart
from src.utils.wb import pull_artifact
from ruamel.yaml import YAML
from src.visualizations.experiment_visualizer import ActiveLearningVisualizer, PassiveLearningVisualizer

log = logging.getLogger(__name__)


def visualize_model_comparison(cfg: DictConfig) -> None:
    experiment_file = os.path.join(cfg.path.experiments, 'model_comparison', f'{cfg.project_name}.yaml')
    vis = PassiveLearningVisualizer(experiment_file)
    vis.plot_miou(ewm_span=20)
    vis.plot_accuracy(ewm_span=20)


def visualize_loss_comparison(cfg: DictConfig) -> None:
    experiment_file = os.path.join(cfg.path.experiments, 'loss_comparison', f'{cfg.project_name}.yaml')
    vis = PassiveLearningVisualizer(experiment_file)
    vis.plot_miou(ewm_span=20)
    vis.plot_accuracy(ewm_span=20)


def visualize_baseline(cfg: DictConfig) -> None:
    experiment_file = os.path.join(cfg.path.experiments, 'baseline', f'{cfg.project_name}.yaml')
    vis = PassiveLearningVisualizer(experiment_file)
    vis.plot_miou(ewm_span=20)
    vis.plot_accuracy(ewm_span=20)


def visualize_learning(cfg: DictConfig) -> None:
    experiment_file = os.path.join(cfg.path.experiments, 'strategy_comparison', f'{cfg.project_name}.yaml')
    vis = ActiveLearningVisualizer(experiment_file)
    vis.plot_miou()
    vis.plot_accuracy()


def visualize_class_distribution(cfg: DictConfig) -> None:
    experiment_file = os.path.join(cfg.path.experiments, 'class_distribution', f'{cfg.project_name}.yaml')
    yaml = YAML()
    with open(experiment_file, 'r') as f:
        data = yaml.load(f)

    statistics = pull_artifact(data['DatasetStats'])
    print(statistics.keys())

    label_names = [v for v in cfg.ds.labels_train.values() if v != 'void']
    class_distribution = statistics['labeled_class_distribution'][1:] * 100
    class_progress = statistics['class_progress'][1:] * 100

    # remove 'bicycle' key
    if 'bicycle' in label_names:
        idx = label_names.index('bicycle')
        label_names.pop(idx)
        class_distribution = np.delete(class_distribution, idx)
        class_progress = np.delete(class_progress, idx)

    bar_chart(values=class_distribution, labels=label_names, value_label='Class Distribution [%]')
    bar_chart(values=class_progress, labels=label_names, value_label='Class Progress [%]')
