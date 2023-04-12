import torch
import wandb
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from torch.utils.data import Dataset

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


def log_dataset_statistics(cfg: DictConfig, dataset: Dataset, save_artifact: bool) -> np.ndarray:
    ignore_index = cfg.ds.ignore_index
    statistics_artifact = cfg.wandb.statistics_artifact.name
    label_names = [v for k, v in cfg.ds.labels_train.items() if k != ignore_index]

    class_dist, labeled_class_distribution, class_labeling_progress, labeled_ratio = dataset.get_statistics()

    # Log the dataset labeling progress
    wandb.log({f'Dataset Labeling Progress': labeled_ratio}, step=0)

    # Filter and log the dataset class distribution
    class_dist = np.delete(class_dist, ignore_index)
    data = [[name, value] for name, value in zip(label_names, class_dist)]
    table = wandb.Table(data=data, columns=["Class", "Distribution"])
    wandb.log({f"Class Distribution - "
               f"{labeled_ratio:.2f}%": wandb.plot.bar(table, "Class", "Distribution")}, step=0)

    # Log the labeled class distribution
    labeled_class_dist = np.delete(labeled_class_distribution, ignore_index)
    data = [[name, value] for name, value in zip(label_names, labeled_class_dist)]
    table = wandb.Table(data=data, columns=["Class", "Distribution"])
    wandb.log({f"Labeled Class Distribution - "
               f"{labeled_ratio:.2f}%": wandb.plot.bar(table, "Class", "Distribution")}, step=0)

    # Filter and log the class labeling progress
    class_labeling_progress = np.delete(class_labeling_progress, ignore_index)
    data = [[name, value] for name, value in zip(label_names, class_labeling_progress)]
    table = wandb.Table(data=data, columns=["Class", "Labeling Progress"])
    wandb.log({f"Class Labeling Progress - "
               f"{labeled_ratio:.2f}%": wandb.plot.bar(table, "Class", "Labeling Progress")}, step=0)

    if save_artifact:
        metadata = {'labeled_ratio': labeled_ratio}
        dataset_statistics = {'class_distribution': class_dist,
                              'labeled_class_distribution': labeled_class_dist,
                              'class_labeling_progress': class_labeling_progress,
                              'labeled_ratio': labeled_ratio}

        torch.save(dataset_statistics, f'data/{statistics_artifact.name}.pt')
        artifact = wandb.Artifact(statistics_artifact.name,
                                  type='statistics',
                                  metadata=metadata,
                                  description='Dataset statistics')
        artifact.add_file(f'data/{statistics_artifact.name}.pt')
        wandb.run.log_artifact(artifact)

    return labeled_class_distribution


def log_most_labeled_sample(dataset: Dataset, laser_scan: LaserScan) -> None:
    most_labeled_sample, sample_labeled_ratio, label_mask = dataset.get_most_labeled_sample()

    # Open the scan and the label
    laser_scan.open_scan(dataset.scan_files[most_labeled_sample])
    laser_scan.open_label(dataset.label_files[most_labeled_sample])

    # Create the point cloud and the projection with fully labeled points
    cloud = np.concatenate([laser_scan.points, laser_scan.color * 255], axis=1)
    cloud_label_full = np.concatenate([laser_scan.points, laser_scan.sem_label_color * 255], axis=1)
    projection_label_full = laser_scan.proj_sem_color

    # Open the label with the label mask
    laser_scan.open_scan(dataset.scan_files[most_labeled_sample])
    laser_scan.open_label(dataset.label_files[most_labeled_sample], label_mask)

    # Create the point cloud and the projection with the most labeled points
    cloud_label = np.concatenate([laser_scan.points, laser_scan.sem_label_color * 255], axis=1)
    projection_label = laser_scan.proj_sem_color

    wandb.log({'Point Cloud': wandb.Object3D(cloud),
               'Point Cloud Label - Full': wandb.Object3D(cloud_label_full),
               f'Point Cloud Label ({sample_labeled_ratio:.2f})': wandb.Object3D(cloud_label),
               'Projection': wandb.Image(laser_scan.proj_color),
               'Projection Label - Full': wandb.Image(projection_label_full),
               f'Projection Label - ({sample_labeled_ratio:.2f})': wandb.Image(projection_label)}, step=0)


def log_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
              history: dict, epoch: int, model_name: str) -> None:
    model.eval()
    state_dict = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(state_dict, f'data/{model_name}.pt')

    # Create metadata for W&B artifact
    metadata = {'epoch': epoch}
    for key, value in history.items():
        metadata[key] = value[-1]

    # Create W&B artifact and log it
    artifact = wandb.Artifact(model_name, type='model', metadata=metadata,
                              description='Model state with optimizer state '
                                          'and metric history for each epoch.')
    artifact.add_file(f'data/{model_name}.pt')
    wandb.run.log_artifact(artifact)


def log_history(history_name: str, history: dict) -> None:
    torch.save(history, f'data/{history_name}.pt')
    artifact = wandb.Artifact(history_name, type='history',
                              description='Metric history for each epoch.')
    artifact.add_file(f'data/{history_name}.pt')
    wandb.run.log_artifact(artifact)
