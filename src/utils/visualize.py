import os

import torch
import wandb
import numpy as np
import seaborn as sn
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from ruamel.yaml import YAML

cmap_small = cm.get_cmap('tab10')
cmap_large = cm.get_cmap('tab20')


def bar_chart(values: np.ndarray, labels: list, value_label: str, title: str = None, save_path: str = None):
    y = np.arange(len(labels))

    fig, ax = plt.subplots()
    rects = ax.barh(y, values, color=cmap_small(0.5))
    bar_labels = [f"{val:.1e}" if val >= 0.00001 else "" for val in values]
    ax.bar_label(rects, labels=bar_labels, padding=10)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(value_label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, np.max(values) + 0.15)
    fig.subplots_adjust(left=0.2)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def grouped_bar_chart(values: dict, labels: list, value_label: str, title: str = None, save_path: str = None):
    width = 0.2
    y = np.arange(len(labels))
    max_value = max([max(v) for v in values.values()])

    fig, ax = plt.subplots()

    # Create a bar plot for each key in labeled_distributions
    for i, (key, vals) in enumerate(values.items()):
        offset = i - (len(vals) - 1) / 2
        rects = ax.barh(y + offset * width, vals, width, label=key, color=cmap_small(i))
        # bar_labels = [f"{val:.1e}" if val >= 0.00001 else "" for val in vals]
        # ax.bar_label(rects, labels=bar_labels, padding=10)

    # Set the axis labels and title
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(value_label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    # ax.set_xlim(0, max_value + 0.15)
    fig.subplots_adjust(left=0.2)

    # Add a legend in the top-left corner
    ax.legend(loc='best')

    plt.show()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot(values: dict, x_label: str, y_label: str, title: str = None, save_path: str = None):
    fig, ax = plt.subplots()
    cmap = cmap_small if len(values) <= 10 else cmap_large

    for i, (key, values) in enumerate(values.items()):
        if isinstance(values, dict):
            for j, (key2, values2) in enumerate(values.items()):
                ax.plot(values2, label=f'{key} {key2}', linewidth=1.5,
                        linestyle=['-', '--', '-.'][j], color=cmap(i))
        else:
            ax.plot(values, label=key, linewidth=1.5, color=cmap(i))

    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc='best')
    ax.grid()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_confusion_matrix(confusion_matrix: np.ndarray, labels: list, ignore_index: int = 0, title: str = None,
                          save_path: str = None):
    # Remove the ignored class from the last two dimensions
    confusion_matrix = np.delete(confusion_matrix, ignore_index, axis=-1)
    confusion_matrix = np.delete(confusion_matrix, ignore_index, axis=-2)

    # Plot confusion matrix
    sn.set()
    plt.figure(figsize=(16, 16))
    sn.heatmap(confusion_matrix, annot=True, cmap='Blues',
               fmt='.2f', xticklabels=labels, yticklabels=labels)

    # Visualize confusion matrix
    if title is not None:
        plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


class ExperimentVisualizer(object):
    def __init__(self, file: str):
        self.file = file
        self.data = None

        self.strategies = list()
        self.percentages = dict()

        self.histories = dict()

        self.max_accs = dict()
        self.max_mious = dict()

        self.load_data()

    def load_data(self):
        yaml = YAML()
        api = wandb.Api()

        with open(self.file, 'r') as f:
            self.data = yaml.load(f)

        for s in self.data:
            strategy_name = s['name']
            self.strategies.append(strategy_name)
            self.histories[strategy_name] = []
            self.max_mious[strategy_name] = []
            self.max_accs[strategy_name] = []
            self.percentages[strategy_name] = s['percentages']

            for v in s['history']['versions']:
                history_artifact = f"{s['history']['url']}:v{v}"
                history_file = s['history']['file']

                artifact_dir = api.artifact(history_artifact).download()
                history_path = os.path.join(artifact_dir, f'{history_file}.pt')
                self.histories[strategy_name].append(torch.load(history_path))

        self.calculate_max_metrics()

    def calculate_max_metrics(self):
        for strategy in self.strategies:
            histories = self.histories[strategy]
            print(f'Calculating max metrics for {strategy}')
            print(f'Number of histories: {len(histories)}')
            for history in histories:
                max_miou = np.max(history['miou_val'])
                print(f'Max miou: {max_miou}')
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


def plot_result(file: str):
    import os
    import torch
    import wandb
    yaml = YAML()

    with open(file, 'r') as f:
        data = yaml.load(f)

    print(data['SemanticKITTI'][0]['name'])
    print(data['SemanticKITTI'][1]['name'])

    api = wandb.Api()
    history_artifact = f"{data['SemanticKITTI'][0]['history']['url']}:v{data['SemanticKITTI'][0]['history']['versions'][-1]}"
    history_file = data['SemanticKITTI'][0]['history']['file']
    history = api.artifact(history_artifact)
    history_path = os.path.join(history.download(), f'{history_file}.pt')
    history = torch.load(history_path)
    plot({'Loss Train': history['loss_train'], 'Loss Val': history['loss_val']}, 'Epoch', 'Value', 'Loss')
    plot({'Accuracy Train': history['accuracy_train'], 'Accuracy Val': history['accuracy_val']},
         'Epoch', 'Value', 'Accuracy')
    plot({'MIoU Train': history['miou_train'], 'IoU Val': history['miou_val']}, 'Epoch', 'Value', 'MIoU')


if __name__ == '__main__':
    experiment_visualizer = ExperimentVisualizer('test2.yaml')
    experiment_visualizer.plot_mious()
    # plot_result('test2.yaml')
    exit(0)

    # distributions = {'0.5%': np.array([0.1, 0.5, 0.4]),
    #                  '1%': np.array([0.1, 0.5, 0.4]),
    #                  '1.5%': np.array([0.1, 0.5, 0.4])}
    #
    # labeled_distributions = {'0.5%': np.array([0.1, 0.2, 0.7]),
    #                          '1%': np.array([0.1, 0.5, 0.4]),
    #                          '1.5%': np.array([0.6, 0.3, 0.1])}
    #
    # miou = {'0.5%': np.array([0.1, 0.1, 0.05, 0.2, 0.15]),
    #         '1%': np.array([0.2, 0.2, 0.3, 0.23, 0.4]),
    #         '1.5%': np.array([0.4, 0.35, 0.45, 0.6, 0.6])}
    #
    # label_names = ['Car', 'Pedestrian', 'Cyclist']
    #
    # # Assert that all distributions are equal
    # distributions_matrix = np.vstack(list(distributions.values()))
    # assert np.all(distributions_matrix == distributions_matrix[0]), 'Distributions are not equal.'
    #
    # bar_chart(distributions['0.5%'], label_names, 'Mass [%]')
    # grouped_bar_chart(labeled_distributions, label_names, 'Mass [%]')
    # plot(miou, 'Epoch [-]', 'mIoU [-]')
