import os
import logging

import torch
import wandb
import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig
from ruamel.yaml import YAML

from src.utils.wb import pull_artifact


class ExperimentVisualizer(object):
    def __init__(self, file: str):
        self.names = list()
        self.histories = dict()
        self.percentages = dict()

        self.max_accs = dict()
        self.max_mious = dict()

        self.load_data()

    def calculate_max_metrics(self):
        for strategy in self.strategies:
            histories = self.histories[strategy]
            for history in histories:
                self.max_mious[strategy].append(np.max(history['miou_val']))

    def plot_mious(self):
        fig, ax = plt.subplots()
        cmap = cmap_small

        for i, strategy in enumerate(self.strategies):
            print(self.percentages[strategy])
            print(self.max_mious[strategy])
            ax.plot(self.percentages[strategy], self.max_mious[strategy], label=strategy, linewidth=1.5, color=cmap(i))

        ax.set_title('Max mious over percentages')
        ax.set_xlabel('Percentage of training data')
        ax.set_ylabel('Max mious')
        ax.legend(loc='best')
        ax.grid()
        plt.show()


def visualize_model_comparison(cfg: DictConfig) -> None:
    yaml = YAML()
    experiment_file = 'experiments/model_comparison/KITTI360.yaml'
    histories = dict()
    losses = dict()
    mious = dict()
    accs = dict()
    with open(experiment_file, 'r') as f:
        data = yaml.load(f)

    for model, history_artifact in data.items():
        history = pull_artifact(history_artifact)
        histories[model] = history
        losses[model] = history['loss_val']
        mious[model] = history['miou_val']
        accs[model] = history['acc_val']

    print(losses)


def visualize_loss_comparison(cfg: DictConfig) -> None:
    raise NotImplementedError


def visualize_baseline(cfg: DictConfig) -> None:
    raise NotImplementedError


def visualize_learning(cfg: DictConfig) -> None:
    raise NotImplementedError
