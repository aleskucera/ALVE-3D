#!/usr/bin/env python
import os
import logging

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src import set_paths, ScanVis, LaserScan, SemLaserScan

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    if cfg.action == 'sequence':
        visualize_sequence(cfg)
    else:
        raise NotImplementedError


def visualize_sequence(cfg):
    # format sequence number
    seq = f"{cfg.vis.sequence:02d}"
    seq_path = os.path.join(cfg.path.kitti, seq)
    log.info(f"Visualizing sequence {seq} from {seq_path}")
    # read file names
    dir_struct = cfg.kitti.sequence_structure

    points_dir = os.path.join(seq_path, dir_struct.points_dir)
    labels_dir = os.path.join(seq_path, dir_struct.labels_dir)

    points_files = _read_file_names(points_dir, '.bin')
    if cfg.vis.semantics or cfg.vis.instances:
        label_files = _read_file_names(labels_dir, '.label')
        assert len(points_files) == len(label_files)
        scan = SemLaserScan(cfg.kitti.color_map, project=True, colorize=False)
        vis = ScanVis(scan=scan, scan_names=points_files, label_names=label_files,
                      semantics=cfg.vis.semantics, instances=cfg.vis.instances)
    else:
        scan = LaserScan(project=True, colorize=False)
        vis = ScanVis(scan=scan, scan_names=points_files)
    vis.run()


def _read_file_names(path: str, ext: str) -> list:
    """ Read all file names from a directory with a specific extension
    :param path: path to the directory
    :param ext: extension of the files
    :return: list of file names
    """
    files = sorted(os.listdir(path))
    return [os.path.join(path, f) for f in files if f.endswith(ext)]


if __name__ == '__main__':
    main()
