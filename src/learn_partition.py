import os
import sys

import torch
import wandb
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    model.train()

    with wandb.init(project='KITTI-360 learning'):
        for epoch in range(cfg.train.epochs):
            for data in tqdm(train_loader):
                # Get data
                clouds, clouds_global, labels, edg_source, edg_target, is_transition, xyz = data
                clouds, clouds_global, labels = clouds.to(device), clouds_global.to(device), labels.to(device)
                edg_source, edg_target = edg_source.to(device), edg_target.to(device)
                is_transition = is_transition.to(device)
                print(type(xyz))

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

                # Backward
                loss.backward()

                # Clip the gradient
                for p in model.parameters():
                    p.grad.data.clamp_(-1000, 1000)

                optimizer.step()

            # Save the model
            torch.save(model.state_dict(), os.path.join(cfg.path.models, f'PointNet_epoch_{epoch}.pth'))
