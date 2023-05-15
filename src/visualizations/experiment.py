import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from ruamel.yaml import YAML

from src.utils.wb import pull_artifact

log = logging.getLogger(__name__)


class StrategyExperimentVisualizer(object):
    def __init__(self, file: str):

        self.data = None
        self.strategies = list()
        self.histories = dict()
        self.percentages = dict()
        self.baseline_history = None

        self.max_accs = dict()
        self.max_mious = dict()

        self.baseline_acc = None
        self.baseline_miou = None

        self.__init(file)

    def __init(self, file: str):
        yaml = YAML()

        with open(file, 'r') as f:
            self.data = yaml.load(f)

        self.baseline_history = pull_artifact(self.data['baseline'])
        self.baseline_acc = np.max(self.baseline_history['accuracy_val'])
        self.baseline_miou = np.max(self.baseline_history['miou_val'])

        print(f'Baseline accuracy: {self.baseline_acc}')
        print(f'Baseline miou: {self.baseline_miou}')

        for s in self.data['strategies']:
            strategy = s['name']
            history_artifact = s['history']['artifact']
            history_versions = s['history']['versions']

            self.strategies.append(strategy)
            self.percentages[strategy] = s['percentages']
            self.histories[strategy] = [pull_artifact(f'{history_artifact}:v{v}') for v in history_versions]

            self.max_mious[strategy] = self.calculate_max_metric(self.histories[strategy], 'miou_val')
            self.max_accs[strategy] = self.calculate_max_metric(self.histories[strategy], 'accuracy_val')

    @staticmethod
    def calculate_max_metric(history_list: list, metric: str):
        return [np.max(h[metric]) for h in history_list]

    def plot_miou(self):
        fig, ax = plt.subplots()

        for i, strategy in enumerate(self.strategies):
            ax.plot(self.percentages[strategy], self.max_mious[strategy], label=strategy, linewidth=1.5)

        ax.axhline(y=self.baseline_miou, linestyle='--', linewidth=2, label='100% Baseline')
        ax.axhline(y=0.9 * self.baseline_miou, linestyle='--', linewidth=2, label='90% Baseline')

        ax.set_title('Max mious over percentages')
        ax.set_xlabel('Percentage of training data')
        ax.set_ylabel('Max mious')
        ax.legend(loc='lower right')
        ax.grid()
        plt.show()

    def plot_accuracy(self):
        fig, ax = plt.subplots()

        for i, strategy in enumerate(self.strategies):
            ax.plot(self.percentages[strategy], self.max_accs[strategy], label=strategy, linewidth=1.5)

        ax.axhline(y=self.baseline_acc, linestyle='--', linewidth=2, label='100% Baseline')
        ax.axhline(y=0.9 * self.baseline_acc, linestyle='--', linewidth=2, label='90% Baseline')

        ax.set_title('Max accuracy over percentages')
        ax.set_xlabel('Percentage of training data')
        ax.set_ylabel('Max accuracy')
        ax.legend(loc='lower right')
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
    vis = StrategyExperimentVisualizer('experiments/strategy_comparison/KITTI360.yaml')
    vis.plot_miou()
    vis.plot_accuracy()
