import os
import matplotlib.pyplot as plt
from .classes import History


def plot_results(history: History, save_path: str) -> None:
    plot_loss(history, save_path)
    plot_acc(history, save_path)


def plot_loss(history: History, save_path: str) -> None:
    plt.figure()
    plt.plot(history.val_loss, label='val', marker='o')
    plt.plot(history.train_loss, label='train', marker='o')
    plt.title('Loss per epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(save_path, 'loss.png'))


def plot_acc(history: History, save_path: str) -> None:
    plt.figure()
    plt.plot(history.train_acc, label='train_accuracy', marker='*')
    plt.plot(history.val_acc, label='val_accuracy', marker='*')
    plt.title('Accuracy per epoch')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.savefig(os.path.join(save_path, 'acc.png'))
