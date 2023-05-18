import numpy as np
import seaborn as sn
import matplotlib.cm as cm
import matplotlib.pyplot as plt

cmap_small = cm.get_cmap('tab10')
cmap_large = cm.get_cmap('tab20')


def bar_chart(values: np.ndarray, labels: list, value_label: str, title: str = None, save_path: str = None):
    y = np.arange(len(labels))

    fig, ax = plt.subplots()
    rects = ax.barh(y, values, color=cmap_small(0))
    bar_labels = [f"{val:.1e}" if val >= 0.00001 else "" for val in values]
    ax.bar_label(rects, labels=bar_labels, padding=10)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(value_label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, np.max(values) + 0.25 * np.max(values))
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


if __name__ == '__main__':
    distributions = {'0.5%': np.array([0.1, 0.5, 0.4]),
                     '1%': np.array([0.1, 0.5, 0.4]),
                     '1.5%': np.array([0.1, 0.5, 0.4])}

    labeled_distributions = {'0.5%': np.array([0.1, 0.2, 0.7]),
                             '1%': np.array([0.1, 0.5, 0.4]),
                             '1.5%': np.array([0.6, 0.3, 0.1])}

    miou = {'0.5%': np.array([0.1, 0.1, 0.05, 0.2, 0.15]),
            '1%': np.array([0.2, 0.2, 0.3, 0.23, 0.4]),
            '1.5%': np.array([0.4, 0.35, 0.45, 0.6, 0.6])}

    label_names = ['Car', 'Pedestrian', 'Cyclist']

    # Assert that all distributions are equal
    distributions_matrix = np.vstack(list(distributions.values()))
    assert np.all(distributions_matrix == distributions_matrix[0]), 'Distributions are not equal.'

    bar_chart(distributions['0.5%'], label_names, 'Mass [%]')
    grouped_bar_chart(labeled_distributions, label_names, 'Mass [%]')
    plot(miou, 'Epoch [-]', 'mIoU [-]')
