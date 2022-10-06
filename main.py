import logging

import hydra
import open3d as o3d
from omegaconf import DictConfig

from src.dataset import SemanticDataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    dataset = SemanticDataset(path=cfg.kitti.path, split='train', cfg=cfg.kitti)
    cloud = dataset.create_global_cloud(sequence=0, step=40)
    o3d.visualization.draw_geometries([cloud])


if __name__ == '__main__':
    main()
