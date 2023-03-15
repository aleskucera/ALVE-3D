import wandb
import torch
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy, \
    MulticlassJaccardIndex, MulticlassConfusionMatrix


def get_logger(logger_type: str, num_classes: int, labels: dict, device: torch.device, ignore_index: int):
    if logger_type == 'semantic':
        return SemanticLogger(num_classes, labels, device, ignore_index)
    elif logger_type == 'partition':
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown logger: {logger_type}')


class BaseLogger(object):
    def __init__(self):
        pass

    def load_history(self, history: dict):
        raise NotImplementedError

    def update(self, loss, outputs, targets):
        raise NotImplementedError

    def log_train(self, epoch):
        raise NotImplementedError

    def log_val(self, epoch):
        raise NotImplementedError

    def log_dataset_statistics(self, dataset, epoch):
        raise NotImplementedError


class SemanticLogger(object):
    def __init__(self, num_classes, labels, device, ignore_index=0):
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.labels = [k for k in labels.keys() if k != ignore_index]
        self.label_names = [v for k, v in labels.items() if k != ignore_index]

        metric_args = dict(num_classes=num_classes, ignore_index=ignore_index, validate_args=False)
        self.acc = MulticlassAccuracy(**metric_args).to(device)
        self.iou = MulticlassJaccardIndex(**metric_args).to(device)
        self.class_acc = MulticlassAccuracy(**metric_args, average='none').to(device)
        self.class_iou = MulticlassJaccardIndex(**metric_args, average='none').to(device)
        self.conf_matrix = MulticlassConfusionMatrix(**metric_args, normalize='true').to(device)

        self.batch_loss_history = []

        self.history = {'loss train': [], 'loss val': [],
                        'miou': [], 'class iou': [],
                        'accuracy': [], 'class accuracy': [],
                        'confusion matrix': [], 'dataset statistics': []}

    @property
    def miou_converged(self):
        """ Check if IoU has converged. If the IoU has
        not improved for 10 epochs, the training is stopped.
        """

        return len(self.history['miou']) - np.argmax(self.history['miou']) > 10

    @property
    def miou_improved(self):
        """ Check if IoU has improved. If the last IoU
        is the maximum, the model is saved.
        """

        return len(self.history['miou']) - np.argmax(self.history['miou']) == 1

    def load_history(self, history: dict):
        assert set(history.keys()) == set(self.history.keys()), "History keys don't match"
        self.history = history

    def update(self, loss: float, outputs: torch.Tensor, targets: torch.Tensor):
        """ Update loss and metrics

        :param loss: Loss value
        :param outputs: Model outputs
        :param targets: Targets
        """

        self.batch_loss_history.append(loss)

        self.acc.update(outputs, targets)
        self.iou.update(outputs, targets)

        self.class_acc.update(outputs, targets)
        self.class_iou.update(outputs, targets)

        self.conf_matrix.update(outputs, targets)

    def log_train(self, epoch: int) -> dict:
        """Log train metrics to W&B

        :param epoch: Current epoch
        """

        return self._log_epoch(epoch, 'train')

    def log_val(self, epoch: int) -> dict:
        """Log val metrics to W&B

        :param epoch: Current epoch
        """

        return self._log_epoch(epoch, 'val')

    def log_dataset_statistics(self, statistics: np.ndarray, epoch: int):
        self.history['dataset statistics'].append(statistics)
        statistics = np.delete(statistics, self.ignore_index)
        for name, stat in zip(self.label_names, statistics):
            wandb.log({f"Label ratio - {name}": stat}, step=epoch)

    def _log_epoch(self, epoch: int, phase: str) -> dict:
        """Log epoch metrics to W&B

        :param epoch: Current epoch
        :param phase: 'train' or 'val'
        """

        # Compute loss and metrics
        loss = sum(self.batch_loss_history) / len(self.batch_loss_history)
        acc = self.acc.compute().cpu().item()
        iou = self.iou.compute().cpu().item()
        class_acc = self.class_acc.compute().cpu()
        class_iou = self.class_iou.compute().cpu()
        conf_matrix = self.conf_matrix.compute().cpu()

        # Log loss
        self.history[phase].append(loss)
        wandb.log({f"Loss {phase}": loss}, step=epoch)

        # Log metrics
        self.history['accuracy'].append(acc)
        wandb.log({f"Accuracy {phase}": acc}, step=epoch)

        self.history['miou'].append(iou)
        wandb.log({f"MIoU {phase}": iou}, step=epoch)

        if phase == 'val':
            self.history['class accuracy'].append(class_acc)
            self._log_class_accuracy(class_acc, epoch)

            self.history['class iou'].append(class_iou)
            self._log_class_iou(class_iou, epoch)

            self.history['confusion matrix'].append(conf_matrix)
            self._log_confusion_matrix(conf_matrix, epoch)

        # Reset batch loss and metrics
        self.batch_loss_history = []
        self.iou.reset()
        self.acc.reset()
        self.class_acc.reset()
        self.class_iou.reset()
        self.conf_matrix.reset()

        return self.history

    def _log_class_accuracy(self, class_acc: torch.Tensor, epoch: int):
        class_acc = class_acc.tolist()
        del class_acc[self.ignore_index]
        for name, acc in zip(self.label_names, class_acc):
            wandb.log({f"Accuracy - {name}": acc}, step=epoch)

    def _log_class_iou(self, class_iou: torch.Tensor, epoch: int):
        class_iou = class_iou.tolist()
        del class_iou[self.ignore_index]
        for name, iou in zip(self.label_names, class_iou):
            wandb.log({f"IoU - {name}": iou}, step=epoch)

    def _log_confusion_matrix(self, confusion_matrix: torch.Tensor, epoch: int):
        conf_matrix = confusion_matrix.numpy()

        # Remove the ignored class from the last two dimensions
        conf_matrix = np.delete(conf_matrix, self.ignore_index, axis=-1)
        conf_matrix = np.delete(conf_matrix, self.ignore_index, axis=-2)

        # Plot confusion matrix
        sn.set()
        plt.figure(figsize=(16, 16))
        sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names)

        # Visualize confusion matrix
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Multiclass Confusion Matrix')

        # Log confusion matrix to W&B
        wandb.log({"Confusion Matrix": wandb.Image(plt)}, step=epoch)

        plt.close()


class PartitionLogger(object):
    def __init__(self):
        super().__init__()
