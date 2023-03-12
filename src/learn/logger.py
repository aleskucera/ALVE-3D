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

    def update(self, loss, outputs, targets):
        raise NotImplementedError

    def log_train(self, epoch):
        raise NotImplementedError

    def log_val(self, epoch):
        raise NotImplementedError


class SemanticLogger(object):
    def __init__(self, num_classes, labels, device, ignore_index=0):
        self.num_classes = num_classes
        self.labels = [k for k in labels.keys() if k != ignore_index]
        self.label_names = [v for k, v in labels.items() if k != ignore_index]
        self.device = device
        self.ignore_index = ignore_index

        self.loss_history = {'train': [], 'val': []}
        self.batch_loss_history = []

        metric_args = dict(num_classes=num_classes, ignore_index=ignore_index, validate_args=False)
        self.acc = MulticlassAccuracy(**metric_args).to(device)
        self.iou = MulticlassJaccardIndex(**metric_args).to(device)
        self.class_acc = MulticlassAccuracy(**metric_args, average='none').to(device)
        self.class_iou = MulticlassJaccardIndex(**metric_args, average='none').to(device)
        self.conf_matrix = MulticlassConfusionMatrix(**metric_args, normalize='true').to(device)

        self.metric_history = {'Accuracy': [], 'IoU': [], 'Confusion Matrix': [],
                               'Class Accuracy': [], 'Class IoU': []}

        wandb.define_metric("Loss train", summary="min")
        wandb.define_metric("Loss val", summary="min")

        wandb.define_metric("Accuracy train", summary="max")
        wandb.define_metric("Accuracy val", summary="max")

        wandb.define_metric("IoU train", summary="max")
        wandb.define_metric("IoU val", summary="max")

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
        self.loss_history[phase].append(loss)
        wandb.log({f"Loss {phase}": loss}, step=epoch)

        # Log metrics
        self.metric_history['Accuracy'].append(acc)
        wandb.log({f"Accuracy {phase}": acc}, step=epoch)

        self.metric_history['IoU'].append(iou)
        wandb.log({f"IoU {phase}": iou}, step=epoch)

        if phase == 'val':
            self.metric_history['Class Accuracy'].append(class_acc)
            self._log_class_accuracy(class_acc, epoch)

            self.metric_history['Class IoU'].append(class_iou)
            self._log_class_iou(class_iou, epoch)

            self.metric_history['Confusion Matrix'].append(conf_matrix)
            self._log_confusion_matrix(conf_matrix, epoch)

        # Reset batch loss and metrics
        self.batch_loss_history = []
        self.iou.reset()
        self.acc.reset()
        self.class_acc.reset()
        self.class_iou.reset()
        self.conf_matrix.reset()

        return {'loss': loss, 'acc': acc, 'iou': iou}

    def _log_class_accuracy(self, class_acc: torch.Tensor, epoch: int):
        for class_name, class_acc in zip(self.label_names, class_acc.tolist()):
            wandb.log({f"Accuracy - {class_name}": class_acc}, step=epoch)

    def _log_class_iou(self, class_iou: torch.Tensor, epoch: int):
        for class_name, class_iou in zip(self.label_names, class_iou.tolist()):
            wandb.log({f"IoU - {class_name}": class_iou}, step=epoch)

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

        # Print confusion matrix
        plt.show()

        # Log confusion matrix to W&B
        wandb.log({"Confusion Matrix": wandb.Image(plt)}, step=epoch)

        plt.close()


class PartitionLogger(object):
    def __init__(self):
        super().__init__()
