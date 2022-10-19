import numpy as np

from dataclasses import dataclass, field


@dataclass
class State:
    # Training phase
    train_loss: float = 0
    train_accuracy: float = 0
    train_iou_score: float = 0

    # Validation phase
    val_loss: float = 0
    val_accuracy: float = 0
    val_iou_score: float = 0

    # Testing phase
    test_loss: float = 0
    test_accuracy: float = 0
    test_iou_score: float = 0

    num_decreases: int = 0
    delta: float = float('inf')
    min_loss: float = float('inf')

    train_num_batches: int = 0
    val_num_batches: int = 0

    epoch: int = 0

    def reset_train_state(self):
        self.train_iou_score = 0
        self.train_accuracy = 0
        self.train_loss = 0

    def reset_val_state(self):
        self.val_loss = 0
        self.val_accuracy = 0
        self.val_iou_score = 0

    def reset_test_state(self):
        self.test_loss = 0
        self.test_accuracy = 0
        self.test_iou_score = 0

    def min_loss_exceeded(self):
        return self.min_loss > self.val_loss

    def update_min_loss(self):
        self.min_loss = self.val_loss

    def average_metrics(self):
        # Divide train data by number of batches
        self.train_loss /= self.train_num_batches
        self.train_accuracy /= self.train_num_batches
        self.train_iou_score /= self.train_num_batches

        # Divide val data by number of batches
        self.val_loss /= self.val_num_batches
        self.val_accuracy /= self.val_num_batches
        self.val_iou_score /= self.val_num_batches


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    train_iou: list[float] = field(default_factory=list)
    val_iou: list[float] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)

    def save_state(self, state: State) -> None:
        self.train_loss.append(state.train_loss)
        self.train_acc.append(state.train_accuracy)
        self.train_iou.append(state.train_iou_score)

        self.val_loss.append(state.val_loss)
        self.val_acc.append(state.val_accuracy)
        self.val_iou.append(state.val_iou_score)

    def epochs_stagnated(self) -> int:
        return len(self.val_acc) - self.val_acc.index(min(self.val_acc))

    def num_decreases(self) -> int:
        # Split the list by the minimum value
        min_val_loss_idx = np.argmin(self.val_loss)
        losses = self.val_loss[min_val_loss_idx:]

        # Count the number of decreases
        differences = np.diff(losses)
        return sum(differences < 0)


@dataclass
class TestData:
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    iou_score: list[float] = field(default_factory=list)

    def save_state(self, state: State) -> None:
        self.loss.append(state.test_loss)
        self.accuracy.append(state.test_accuracy)
        self.iou_score.append(state.test_iou_score)

