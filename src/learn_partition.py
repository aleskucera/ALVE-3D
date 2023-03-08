import os
import sys

import torch
import wandb
import numpy as np
import open3d as o3d
from tqdm import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.kitti360 import KITTI360Dataset
from src.model import PointNet
from src.learning.crosspartition import compute_weight_loss, compute_loss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train_partition(cfg: DictConfig):
    train_ds = KITTI360Dataset(cfg, cfg.ds.path, split='train')
    val_ds = KITTI360Dataset(cfg, cfg.ds.path, split='val')

    train_loader = DataLoader(train_ds, batch_size=1,
                              shuffle=True, num_workers=2)
    # val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
    #                         shuffle=False, num_workers=2)

    model = PointNet(num_features=6, num_global_features=7, out_features=4)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), cfg.train.learning_rate)

    model.train()

    with wandb.init(project='KITTI-360 learning'):
        for epoch in range(cfg.train.epochs):
            for i, data in tqdm(enumerate(train_loader)):

                print(f'Epoch {epoch}, batch {i}')
                # Get data
                clouds, clouds_global, labels, edg_source, edg_target, is_transition, xyz = data
                clouds, clouds_global, labels = clouds.to(device), clouds_global.to(device), labels.to(device)
                is_transition = is_transition.squeeze(0).to(device)

                edg_source, edg_target, xyz = edg_source.squeeze(0).numpy(), edg_target.squeeze(0).numpy(), xyz.squeeze(
                    0).numpy()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                embeddings = model(clouds, clouds_global)

                # Compute loss
                diff = ((embeddings[edg_source, :] - embeddings[edg_target, :]) ** 2).sum(1)
                weights_loss, pred_comp, in_comp = compute_weight_loss(embeddings, edg_source, edg_target,
                                                                       is_transition, diff, xyz)
                loss1, loss2 = compute_loss(diff, is_transition, weights_loss)
                loss = (loss1 + loss2) / weights_loss[0] * 1000

                wandb.log({'loss': loss.item(), 'loss1': loss1.item(), 'loss2': loss2.item()})
                print(f'loss: {loss.item()}, loss1: {loss1.item()}, loss2: {loss2.item()}', end='\r')

                # Backward
                loss.backward()

                # Clip the gradient
                for p in model.parameters():
                    p.grad.data.clamp_(-1000, 1000)

                optimizer.step()

                if i % 5 == 0:
                    color_map = instances_color_map()
                    pred_components_color = color_map[in_comp]

                    cloud = np.concatenate([xyz, pred_components_color * 255], axis=1)

                    # Log statistics
                    wandb.log({'Point Cloud': wandb.Object3D(cloud)})

            # Save the model
            torch.save(model.state_dict(), os.path.join(cfg.path.models, f'PointNet_epoch_{epoch}.pth'))


def instances_color_map():
    # make instance colors
    max_inst_id = 100000
    color_map = np.random.uniform(low=0.0, high=1.0, size=(max_inst_id, 3))
    # force zero to a gray-ish color
    color_map[0] = np.full(3, 0.1)
    return color_map
