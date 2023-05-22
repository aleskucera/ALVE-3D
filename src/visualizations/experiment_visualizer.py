import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.utils.wb import pull_artifact


class ActiveLearningVisualizer(object):
    def __init__(self, file: str):

        self.font_size = 18
        self.line_width = 3
        self.figure_size = (10, 8)

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
        fig, ax = plt.subplots(figsize=self.figure_size)

        for i, strategy in enumerate(self.strategies):
            miou = self.nondecreasing_series(np.array(self.max_mious[strategy])) * 100
            ax.plot(self.percentages[strategy], miou, label=strategy, linewidth=self.line_width)

        # Color black
        ax.axhline(y=self.baseline_miou * 100, linestyle='--', linewidth=self.line_width, color='black')
        ax.axhline(y=self.baseline_miou * 90, linestyle=':', linewidth=self.line_width, color='black')

        ax.set_xlabel('Labeled voxels [%]', fontsize=self.font_size)
        ax.set_ylabel('mIoU [%]', fontsize=self.font_size)
        ax.tick_params(axis='x', labelsize=self.font_size)
        ax.tick_params(axis='y', labelsize=self.font_size)
        ax.grid(color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))

        leg = ax.legend(loc='lower right', fontsize=self.font_size)
        for obj in leg.legendHandles:
            obj.set_linewidth(self.line_width)

        plt.show()

    def plot_accuracy(self):
        fig, ax = plt.subplots(figsize=self.figure_size)

        for i, strategy in enumerate(self.strategies):
            accuracy = self.nondecreasing_series(np.array(self.max_accs[strategy])) * 100
            ax.plot(self.percentages[strategy], accuracy, label=strategy, linewidth=self.line_width)

        # Color black
        ax.axhline(y=self.baseline_acc * 100, linestyle='--', linewidth=self.line_width, color='black')
        ax.axhline(y=self.baseline_acc * 90, linestyle=':', linewidth=self.line_width, color='black')

        ax.set_xlabel('Labeled voxels [%]', fontsize=self.font_size)
        ax.set_ylabel('Accuracy [%]', fontsize=self.font_size)
        ax.tick_params(axis='x', labelsize=self.font_size)
        ax.tick_params(axis='y', labelsize=self.font_size)
        ax.grid(color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))

        leg = ax.legend(loc='lower right', fontsize=self.font_size)
        for obj in leg.legendHandles:
            obj.set_linewidth(self.line_width)

        plt.show()

    @staticmethod
    def nondecreasing_series(series: np.ndarray):
        """ Create a non-decreasing series from a numpy array by taking the maximum of all previous values.
        Example:
            Input: [0.1, 0.2, 0.3, 0.4, 0.2, 0.6]
            Output: [0.1, 0.2, 0.3, 0.4, 0.4, 0.6]
        """
        max_series = np.zeros(len(series))
        max_series[0] = series[0]

        for i in range(1, len(series)):
            max_series[i] = max(series[i], max_series[i - 1])

        return max_series


class PassiveLearningVisualizer(object):
    def __init__(self, file: str):

        self.font_size = 22
        self.line_width = 4
        self.figure_size = (10, 8)

        self.data = None
        self.names = list()
        self.histories = dict()

        self.__init(file)

    def __init(self, file: str):
        yaml = YAML()

        with open(file, 'r') as f:
            self.data = yaml.load(f)

        for name, history in self.data.items():
            self.names.append(name)
            self.histories[name] = pull_artifact(history)

    def plot_miou(self, ewm_span: int = 10):
        fig, ax = plt.subplots(figsize=self.figure_size)

        for name, history in self.histories.items():
            miou = pd.Series(history['miou_val'])
            ewm_miou = miou.ewm(span=ewm_span).mean()
            epochs = np.arange(1, len(miou) + 1)
            ax.plot(epochs, ewm_miou, label=name, linewidth=self.line_width)

        ax.set_xlabel('Epoch', fontsize=self.font_size)
        ax.set_ylabel('mIoU', fontsize=self.font_size)
        ax.tick_params(axis='x', labelsize=self.font_size)
        ax.tick_params(axis='y', labelsize=self.font_size)
        ax.grid(color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))

        leg = ax.legend(loc='lower right', fontsize=self.font_size)
        for obj in leg.legendHandles:
            obj.set_linewidth(self.line_width)

        plt.show()

    def plot_accuracy(self, ewm_span: int = 10):

        fig, ax = plt.subplots(figsize=self.figure_size)

        for name, history in self.histories.items():
            acc = pd.Series(history['accuracy_val'])
            ewm_acc = acc.ewm(span=ewm_span).mean()
            epochs = np.arange(1, len(acc) + 1)
            ax.plot(epochs, ewm_acc, label=name, linewidth=self.line_width)

        ax.set_xlabel('Epoch', fontsize=self.font_size)
        ax.set_ylabel('Accuracy', fontsize=self.font_size)
        ax.tick_params(axis='x', labelsize=self.font_size)
        ax.tick_params(axis='y', labelsize=self.font_size)
        ax.grid(color='gray', linestyle='-', linewidth=2, alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.2f}"))

        leg = ax.legend(loc='lower right', fontsize=self.font_size)
        for obj in leg.legendHandles:
            obj.set_linewidth(self.line_width)

        plt.show()
