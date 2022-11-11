import logging

import torch
import numpy as np
from tqdm import tqdm
from torchmetrics import Accuracy, JaccardIndex
from torch.utils.tensorboard import SummaryWriter

from .utils import parse_data, create_entropy_batch, sort_entropies

log = logging.getLogger(__name__)


class Tester:
    def __init__(self, model, test_loader, device, num_classes, output_path):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.writer = SummaryWriter(output_path)

        self.accuracy = Accuracy(mdmc_average='samplewise', top_k=1)
        self.iou = JaccardIndex(num_classes=num_classes).to(device)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.test_loader)):
                image_batch, label_batch, _ = parse_data(data, self.device)

                # Forward pass
                output = self.model(image_batch)['out']

                batch_acc = self.accuracy(output, label_batch)
                batch_iou = self.iou(output, label_batch)
                self.writer.add_scalar('Accuracy/test', batch_acc, global_step=i)
                self.writer.add_scalar('JaccardIndex/test', batch_iou, global_step=i)
                log.info(f'Batch {i} accuracy: {batch_acc}, JaccardIndex: {batch_iou}')

            total_acc = self.accuracy.compute()
            total_iou = self.iou.compute()

            log.info(f'Accuracy: {total_acc}, JaccardIndex: {total_iou}')

    def calculate_entropies(self) -> np.ndarray:
        entropies = []
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                image_batch, _, idx_batch = parse_data(data, self.device)
                output = self.model(image_batch)['out']

                # Calculate entropy
                entropy_batch = create_entropy_batch(output, idx_batch)
                entropies.append(entropy_batch)

        entropies = np.concatenate(entropies, axis=0)
        entropies = sort_entropies(entropies)
        return entropies
