import logging

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from torchmetrics import Accuracy, JaccardIndex
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

log = logging.getLogger(__name__)


class Tester:
    def __init__(self, model, dataset, test_loader, device, num_classes, output_path):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.test_loader = test_loader
        self.writer = SummaryWriter(output_path)

        self.accuracy = Accuracy(mdmc_average='samplewise', top_k=1)
        self.iou = JaccardIndex(num_classes=num_classes).to(device)

    def test(self):
        self.model.eval()
        for i, data in enumerate(tqdm(self.test_loader)):
            image_batch, label_batch, _ = _parse_data(data, self.device)

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

    def visualize_entropy(self):
        self.model.eval()
        for i, data in enumerate(tqdm(self.test_loader)):
            image_batch, label_batch, idx = _parse_data(data, self.device)
            vis_sample = self.dataset.get_sem_cloud(idx.item())
            output = self.model(image_batch)['out']
            prob = torch.nn.functional.softmax(output, dim=1)
            entropy = _prob2entropy(prob)
            self.visualize_sample(vis_sample, entropy)
            break

    @staticmethod
    def visualize_sample(sample, entropy):
        print(f"max entropy: {entropy.max()}, min entropy: {entropy.min()}")
        cmap = cm.plasma
        norm = mpl.colors.Normalize(vmin=entropy.min(), vmax=entropy.max())
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        entropy = entropy.flatten().detach().cpu().numpy()
        colors = np.array([m.to_rgba(e) for e in entropy])
        sample.colors = colors[:, :3]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(sample.points)
        cloud.colors = o3d.utility.Vector3dVector(sample.colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(cloud)
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        vis.run()
        vis.destroy_window()


def _parse_data(data: tuple, device: torch.device):
    image_batch, label_batch, idx = data
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)
    return image_batch, label_batch, idx


def _prob2entropy(p, axis=1, eps=1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    h = torch.sum(-p * torch.log10(p), dim=axis)
    return h
