import torch
from tqdm import tqdm
from .classes import State, TestData


class Tester:
    def __init__(self, model, test_loader, criterion, device):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.test_loader = test_loader

        self.state = State()
        self.test_data = TestData()

    def test(self):
        self.model.eval()
        self.state.reset_test_state()
        for data in tqdm(self.test_loader):
            image_batch, label_batch = _parse_data(data, self.device)

            # Forward pass
            output = self.model(image_batch)['out']
            loss = self.criterion(output, label_batch)
            self.state.test_loss = loss.item()
            self.state.test_accuracy = _accuracy(output, label_batch)

            self.test_data.save_state(self.state)

        return self.test_data


def _parse_data(data: tuple, device: torch.device):
    image_batch, label_batch = data
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)
    return image_batch, label_batch


def _accuracy(output, target):
    return torch.eq(output.argmax(1), target).float().mean()
