import logging

import wandb
import torch
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, \
    MulticlassJaccardIndex, MulticlassConfusionMatrix

from src.utils.log import log_class_iou, log_class_accuracy, log_confusion_matrix, log_gradient_flow

log = logging.getLogger(__name__)


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
        self.label_names = [v for k, v in labels.items() if k != ignore_index]

        metric_args = dict(num_classes=num_classes, ignore_index=ignore_index, validate_args=False)
        self.acc = MulticlassAccuracy(**metric_args).to(device)
        self.iou = MulticlassJaccardIndex(**metric_args).to(device)
        self.class_acc = MulticlassAccuracy(**metric_args, average='none').to(device)
        self.class_iou = MulticlassJaccardIndex(**metric_args, average='none').to(device)
        self.conf_matrix = MulticlassConfusionMatrix(**metric_args, normalize='true').to(device)

        self.batch_loss_history = []
        self.batch_avg_grad_history = []
        self.batch_max_grad_history = []

        self.history = {

            # Loss
            'loss_val': [],
            'loss_train': [],

            # IoU
            'miou_val': [],
            'miou_train': [],
            'class_iou': [],

            # Accuracy
            'accuracy_val': [],
            'accuracy_train': [],
            'class_accuracy': [],

            # Confusion matrix
            'confusion_matrix': [],

            # Gradients
            'maximum_gradients': [],
            'average_gradients': [],
        }

    def miou_converged(self, min_epochs: int = 30, patience: int = 10):
        """ Check if IoU has converged. If the IoU has
        not improved for 10 epochs, the training is stopped.
        """
        if len(self.history['miou_val']) < min_epochs:
            log.info(f'Skipping convergence check, not enough epochs: {len(self.history["miou_val"])}')
            return False

        if len(self.history['miou_val']) - np.argmax(self.history['miou_val']) > patience:
            log.info(f'MIoU converged, number of epochs without improvement: '
                     f'{len(self.history["miou_val"]) - np.argmax(self.history["miou_val"])}')
            return True
        else:
            log.info(f'MIoU not converged, number of epochs without improvement: '
                     f'{len(self.history["miou_val"]) - np.argmax(self.history["miou_val"])}')
            return False

    def miou_improved(self, min_epochs: int = 15):
        """ Check if IoU has improved. If the last IoU
        is the maximum, the model is saved.
        """
        if len(self.history['miou_val']) < min_epochs:
            log.info(f'Skipping improvement check, not enough epochs: {len(self.history["miou_val"])}')
            return False

        if len(self.history['miou_val']) - np.argmax(self.history['miou_val']) == 1:
            log.info(f'MIoU improved: {self.history["miou_val"][-1]}')
            return True
        else:
            log.info(f'MIoU not improved: {self.history["miou_val"][-1]}')
            return False

    def update(self, loss: float, outputs: torch.Tensor, targets: torch.Tensor, named_params: dict = None):
        """ Update loss and metrics

        :param loss: Loss value
        :param outputs: Model outputs
        :param targets: Targets
        :param named_params: Named parameters
        """

        self.batch_loss_history.append(loss)

        self.acc.update(outputs, targets)
        self.iou.update(outputs, targets)

        self.class_acc.update(outputs, targets)
        self.class_iou.update(outputs, targets)

        self.conf_matrix.update(outputs, targets)

        if named_params is not None:
            self.__update_gradients(named_params)

    def reset(self):
        self.acc.reset()
        self.iou.reset()
        self.class_acc.reset()
        self.class_iou.reset()
        self.conf_matrix.reset()

        self.history['loss_val'] = []
        self.history['loss_train'] = []

        self.history['miou_val'] = []
        self.history['miou_train'] = []
        self.history['class_iou'] = []

        self.history['accuracy_val'] = []
        self.history['accuracy_train'] = []
        self.history['class_accuracy'] = []

        self.history['confusion_matrix'] = []

        self.history['average_gradients'] = []
        self.history['maximum_gradients'] = []

    def log_train(self, epoch: int):
        """Log train metrics to W&B

        :param epoch: Current epoch
        """

        # Compute loss and metrics
        acc = self.acc.compute().cpu().item()
        iou = self.iou.compute().cpu().item()
        loss = sum(self.batch_loss_history) / len(self.batch_loss_history)

        # Compute gradients
        avg_batch_grads = np.array(self.batch_avg_grad_history)
        max_batch_grads = np.array(self.batch_max_grad_history)
        avg_grads = np.mean(avg_batch_grads, axis=0)
        max_grads = np.mean(max_batch_grads, axis=0)

        # Log loss
        self.history[f'loss_train'].append(loss)
        wandb.log({f"Loss Train": loss}, step=epoch)

        # Log metrics
        self.history[f'miou_train'].append(iou)
        wandb.log({f"MIoU Train": iou}, step=epoch)

        self.history[f'accuracy_train'].append(acc)
        wandb.log({f"Accuracy Train": acc}, step=epoch)

        # Log gradients
        self.history['average_gradients'].append(avg_grads)
        self.history['maximum_gradients'].append(max_grads)
        log_gradient_flow(average_gradients=avg_grads, maximum_gradients=max_grads, step=epoch)

        # Reset batch loss and metrics
        self.iou.reset()
        self.acc.reset()
        self.batch_loss_history = []
        self.batch_avg_grad_history = []
        self.batch_max_grad_history = []

    def log_val(self, epoch: int):
        """Log val metrics to W&B

        :param epoch: Current epoch
        """

        # Compute loss and metrics
        acc = self.acc.compute().cpu().item()
        iou = self.iou.compute().cpu().item()
        class_acc = self.class_acc.compute().cpu()
        class_iou = self.class_iou.compute().cpu()
        conf_matrix = self.conf_matrix.compute().cpu()
        loss = sum(self.batch_loss_history) / len(self.batch_loss_history)

        # Log loss
        self.history[f'loss_val'].append(loss)
        wandb.log({f"Loss Val": loss}, step=epoch)

        # Log metrics
        self.history['miou_val'].append(iou)
        wandb.log({f"MIoU Val": iou}, step=epoch)

        self.history['accuracy_val'].append(acc)
        wandb.log({f"Accuracy Val": acc}, step=epoch)

        self.history['class_accuracy'].append(class_acc)
        log_class_accuracy(class_acc=class_acc, labels=self.label_names,
                           ignore_index=self.ignore_index, step=epoch)

        self.history['class_iou'].append(class_iou)
        log_class_iou(class_iou=class_iou, labels=self.label_names,
                      ignore_index=self.ignore_index, step=epoch)

        self.history['confusion_matrix'].append(conf_matrix)
        log_confusion_matrix(confusion_matrix=conf_matrix, labels=self.label_names,
                             ignore_index=self.ignore_index, step=epoch)

        # Reset batch loss and metrics
        self.iou.reset()
        self.acc.reset()
        self.class_acc.reset()
        self.class_iou.reset()
        self.conf_matrix.reset()
        self.batch_loss_history = []

    def __update_gradients(self, named_params: dict):
        avg_grads = []
        max_grads = []
        for n, p in named_params:
            if p.requires_grad and ("bias" not in n):
                avg_grads.append(p.grad.abs().mean().cpu())
                max_grads.append(p.grad.abs().max().cpu())

        self.batch_avg_grad_history.append(np.array(avg_grads))
        self.batch_max_grad_history.append(np.array(max_grads))
