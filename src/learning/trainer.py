import torch
from tqdm import tqdm
from .classes import History, State


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.train_loader = train_loader

        self.history = History()
        self.state = State(train_num_batches=len(train_loader), val_num_batches=len(val_loader))

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            self.model.train()
            self.state.reset_train_state()
            for data in tqdm(self.train_loader):
                image_batch, label_batch = _parse_data(data, self.device)

                # Forward pass
                output = self.model(image_batch)['out']
                loss = self.criterion(output, label_batch)

                # evaluation metrics
                self.state.train_loss += loss.item()
                self.state.train_accuracy += _accuracy(output, label_batch)

                # compute gradient and make an optimization step
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"\nTrain loss: {self.state.train_loss}")

            self.model.eval()
            self.state.reset_val_state()
            with torch.no_grad():
                for data in tqdm(self.val_loader):
                    image_batch, label_batch = _parse_data(data, self.device)

                    # Forward pass
                    output = self.model(image_batch)['out']

                    # loss
                    loss = self.criterion(output, label_batch)
                    self.state.val_loss += loss.item()
                    self.state.val_accuracy += _accuracy(output, label_batch)

            print(f"\nVal loss: {self.state.val_loss}")

            # average loss and metrics by number of batches
            self.state.average_metrics()

            # save state to history
            self.history.save_state(self.state)

        return self.history

    def save(self, path):
        torch.save(self.model.state_dict(), path)


def _parse_data(data: tuple, device: torch.device):
    image_batch, label_batch = data
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)
    return image_batch, label_batch


def _accuracy(output, target):
    return torch.eq(output.argmax(1), target).float().mean()
