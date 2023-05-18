import os

import torch
import wandb
from omegaconf import DictConfig

from src.models import get_model
from src.datasets import SemanticDataset
from src.laserscan import LaserScan, ScanVis
from src.utils.io import ScanInterface
from src.utils.wb import pull_artifact


def visualize_model_predictions(cfg: DictConfig) -> None:
    size = cfg.size if 'size' in cfg else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_state_dict = pull_artifact(cfg.model_artifact, device=device)
    model = get_model(cfg=cfg, device=device)
    model.load_state_dict(model_state_dict)

    dataset = SemanticDataset(split='val', cfg=cfg.ds, dataset_path=cfg.ds.path,
                              project_name='demo', num_clouds=size)
    scan_interface = ScanInterface(dataset.project_name)

    laser_scan = LaserScan(label_map=cfg.ds.learning_map, color_map=cfg.ds.color_map_train, colorize=True,
                           H=cfg.ds.projection.H, W=cfg.ds.projection.W, fov_up=cfg.ds.projection.fov_up,
                           fov_down=cfg.ds.projection.fov_down)

    prediction_files = [path.replace('sequences', dataset.project_name) for path in dataset.scans]

    model.eval()
    with torch.no_grad():
        for i, scan_file in enumerate(dataset.scans):
            scan, label, _, _, _ = dataset[i]
            scan = torch.from_numpy(scan).to(device).unsqueeze(0)
            pred = model(scan)
            pred = pred.argmax(dim=1)
            pred = pred.cpu().numpy().squeeze()
            scan_interface.add_prediction(scan_file, pred)

    # Visualize the predictions
    vis = ScanVis(laser_scan=laser_scan, scans=dataset.scans, labels=dataset.scans,
                  predictions=prediction_files)
    vis.run()
