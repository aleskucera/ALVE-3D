import os
import logging

from omegaconf import DictConfig
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
    vis.plot_miou()
    vis.plot_accuracy()


def visualize_learning(cfg: DictConfig) -> None:
    experiment_file = os.path.join(cfg.path.experiments, 'strategy_comparison', f'{cfg.project_name}.yaml')
    vis = ActiveLearningVisualizer(experiment_file)
    vis.plot_miou()
    vis.plot_accuracy()
