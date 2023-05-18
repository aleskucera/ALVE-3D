import logging
from omegaconf import DictConfig
from src.visualizations.experiment_visualizer import ActiveLearningVisualizer, PassiveLearningVisualizer

log = logging.getLogger(__name__)


def visualize_model_comparison(cfg: DictConfig) -> None:
    vis = PassiveLearningVisualizer('experiments/model_comparison/KITTI360.yaml')
    vis.plot_miou()
    vis.plot_accuracy()


def visualize_loss_comparison(cfg: DictConfig) -> None:
    vis = PassiveLearningVisualizer('experiments/loss_comparison/KITTI360.yaml')
    vis.plot_miou()
    vis.plot_accuracy()


def visualize_baseline(cfg: DictConfig) -> None:
    vis = PassiveLearningVisualizer('experiments/baseline/KITTI360.yaml')
    vis.plot_miou()
    vis.plot_accuracy()


def visualize_learning(cfg: DictConfig) -> None:
    vis = ActiveLearningVisualizer('experiments/strategy_comparison/KITTI360-3.yaml')
    vis.plot_miou()
    vis.plot_accuracy()
