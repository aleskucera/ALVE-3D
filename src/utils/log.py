import torch
import wandb
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import Dataset
from src.utils.wb import push_artifact

from src.laserscan import LaserScan


def log_class_iou(class_iou: torch.Tensor, labels: list[str],
                  step: int, ignore_index: int = 0) -> None:
    class_iou = class_iou.tolist()
    del class_iou[ignore_index]
    for label, iou in zip(labels, class_iou):
        wandb.log({f"IoU - {label}": iou}, step=step)


def log_class_accuracy(class_acc: torch.Tensor, labels: list[str],
                       step: int, ignore_index: int = 0) -> None:
    class_acc = class_acc.tolist()
    del class_acc[ignore_index]
    for label, acc in zip(labels, class_acc):
        wandb.log({f"Accuracy - {label}": acc}, step=step)


def log_class_bar_chart(name: str, values: iter, labels: list[str],
                        value_label: str, step: int = 0):
    data = [[label, value] for label, value in zip(labels, values)]
    table = wandb.Table(data=data, columns=["Class", value_label])
    wandb.log({name: wandb.plot.bar(table, "Class", value_label)}, step=step)


def log_confusion_matrix(confusion_matrix: torch.Tensor, labels: list[str],
                         step: int, ignore_index: int = 0) -> None:
    conf_matrix = confusion_matrix.numpy()

    # Remove the ignored class from the last two dimensions
    conf_matrix = np.delete(conf_matrix, ignore_index, axis=-1)
    conf_matrix = np.delete(conf_matrix, ignore_index, axis=-2)

    # Plot confusion matrix
    sn.set()
    plt.figure(figsize=(16, 16))
    sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f',
               xticklabels=labels, yticklabels=labels)

    # Visualize confusion matrix
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Multiclass Confusion Matrix')

    # Log confusion matrix to W&B
    wandb.log({"Confusion Matrix": wandb.Image(plt)}, step=step)

    plt.close()


def log_dataset_statistics(cfg: DictConfig, dataset: Dataset,
                           artifact_name: str = None, val: bool = False) -> None:
    ignore_index = cfg.ds.ignore_index
    label_names = [v for k, v in cfg.ds.labels_train.items() if k != ignore_index]

    stats = dataset.statistics
    p = stats['labeled_ratio']

    # if val:
    #     if abs(stats['labeled_ratio'] - 1.0) > 1e-5:
    #         raise ValueError(f"Validation dataset is not fully labeled. Labeled ratio: {stats['labeled_ratio']}")
    #     class_dist = np.delete(stats['class_distribution'], ignore_index)
    #     log_class_bar_chart(name=f"Val Class Distribution", values=class_dist, labels=label_names,
    #                         value_label="Distribution", step=0)
    #     return stats['class_distribution']

    # Log the dataset labeling progress
    wandb.log({f'Dataset Labeling Progress': stats['labeled_ratio']}, step=0)
    wandb.log({f'Labeled Voxels': stats['labeled_voxels']}, step=0)

    # # Filter and log the dataset class distribution
    # class_dist = np.delete(stats['class_distribution'], ignore_index)
    # log_class_bar_chart(name=f"Train Class Distribution", values=class_dist, labels=label_names,
    #                     value_label="Distribution", step=0)

    # Log the labeled class distribution
    labeled_class_dist = np.delete(stats['labeled_class_distribution'], ignore_index)
    log_class_bar_chart(name=f"Labeled Class Distribution - {p:.2f}%", values=labeled_class_dist, labels=label_names,
                        value_label="Distribution", step=0)

    # Filter and log the class labeling progress
    class_labeling_progress = np.delete(stats['class_progress'], ignore_index)
    log_class_bar_chart(name=f"Class Labeling Progress - {p:.2f}%", values=class_labeling_progress, labels=label_names,
                        value_label="Labeling Progress", step=0)

    if artifact_name is not None:
        push_artifact(artifact=artifact_name, data=stats,
                      artifact_type='statistics',
                      metadata={'labeled_ratio': stats['labeled_ratio']},
                      description='Dataset statistics')


def log_most_labeled_sample(dataset: Dataset, laser_scan: LaserScan) -> None:
    most_labeled_sample, sample_labeled_ratio, label_mask = dataset.most_labeled_sample

    # Open the scan and the label
    laser_scan.open_scan(dataset.scan_files[most_labeled_sample])
    laser_scan.open_label(dataset.scan_files[most_labeled_sample])

    # Create the point cloud and the projection with fully labeled points
    cloud = np.concatenate([laser_scan.points, laser_scan.color * 255], axis=1)
    cloud_label_full = np.concatenate([laser_scan.points, laser_scan.label_color * 255], axis=1)
    projection_label_full = laser_scan.proj_label_color

    # Open the label with the label mask
    laser_scan.open_scan(dataset.scan_files[most_labeled_sample])
    laser_scan.open_label(dataset.scan_files[most_labeled_sample], label_mask)

    # Create the point cloud and the projection with the most labeled points
    cloud_label = np.concatenate([laser_scan.points, laser_scan.label_color * 255], axis=1)
    projection_label = laser_scan.proj_label_color

    wandb.log({'Point Cloud': wandb.Object3D(cloud),
               'Point Cloud Label - Full': wandb.Object3D(cloud_label_full),
               f'Point Cloud Label ({sample_labeled_ratio:.2f})': wandb.Object3D(cloud_label),
               'Projection': wandb.Image(laser_scan.proj_color),
               'Projection Label - Full': wandb.Image(projection_label_full),
               f'Projection Label - ({sample_labeled_ratio:.2f})': wandb.Image(projection_label)}, step=0)


def log_model(model: torch.nn.Module, history: dict, epoch: int, model_name: str) -> None:
    model.eval()
    metadata = {'epoch': epoch}
    for key, value in history.items():
        metadata[key] = value[-1]
    push_artifact(artifact=model_name, data=model.state_dict(), artifact_type='model',
                  metadata=metadata, description='Model state dictionary')


def log_history(history: dict, history_name: str) -> None:
    push_artifact(artifact=history_name, data=history, artifact_type='history',
                  description='Metric history for each epoch')


def log_selection(selection: dict, selection_name: str) -> None:
    push_artifact(artifact=selection_name, data=selection, artifact_type='selection',
                  description='The selected voxels for the first active learning iteration.')


def log_selection_metric_statistics(cfg, metric_statistics: dict, metric_statistics_name: str = None,
                                    weighted: bool = False) -> None:
    if weighted:
        metric_title = f"Diversity Aware Selection Metric"
        labels_title = f"Diversity Aware Selected Labels"
    else:
        metric_title = f"Selection Metric"
        labels_title = f"Selected Labels"

    # Plot the metric values
    selected_values = metric_statistics['selected_values']
    left_values = metric_statistics['left_values']
    x = np.linspace(0, 1, len(selected_values) + len(left_values))

    plt.figure(figsize=(10, 8))
    plt.plot(x[:len(selected_values)], selected_values, label='Selected Values', linewidth=3)
    plt.plot(x[len(selected_values):], left_values, label='Left Values', linewidth=3)
    plt.axhline(y=min(selected_values), color='k', linestyle='--', label='Threshold')
    plt.title(f"{metric_title} Statistics")
    plt.xlabel('Sorted Values')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid()

    wandb.log({f"{metric_title} Statistics": wandb.Image(plt)}, step=0)
    plt.close()

    # Plot the selected labels
    label_names = []
    label_counts = metric_statistics['label_counts']
    selected_labels = metric_statistics['selected_labels']
    for l, c in zip(selected_labels, label_counts):
        if l != cfg.ds.ignore_index:
            label_names.append(cfg.ds.labels_train[l])

    log_class_bar_chart(name=f"{labels_title} Statistics", values=label_counts, labels=label_names,
                        value_label="Count", step=0)
    push_artifact(artifact=metric_statistics_name, data=metric_statistics, artifact_type='statistics',
                  description='Metric statistics for each epoch')


def log_gradient_flow(average_gradients: np.ndarray, maximum_gradients: np.ndarray, step: int) -> None:
    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')  # Set the figure face color to white
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('white')  # Set the axes face color to white

    plt.bar(np.arange(len(maximum_gradients)), maximum_gradients, alpha=0.7, lw=1)
    plt.bar(np.arange(len(average_gradients)), average_gradients, alpha=0.7, lw=1)
    plt.xlim(left=0, right=len(average_gradients))
    plt.ylim(bottom=0, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    leg = plt.legend(['max-gradient', 'mean-gradient'], loc='upper center')
    leg.get_frame().set_facecolor('white')  # Set the legend face color to white

    wandb.log({f"Gradient Flow": wandb.Image(plt)}, step=step)
    plt.close()
