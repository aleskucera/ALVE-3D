#!/usr/bin/env python
import logging

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from src.utils.io import set_paths
from src.kitti360 import KITTI360Converter

from src.visualizations.filters import visualize_filters
from src.visualizations.dataset import visualize_scans, visualize_clouds, \
    visualize_statistics, visualize_augmentation, visualize_scan_mapping
from src.visualizations.superpoints import visualize_feature, visualize_superpoints
from src.visualizations.experiment import visualize_model_comparison, visualize_learning, \
    visualize_loss_comparison, visualize_baseline, visualize_class_distribution
from src.visualizations.model import visualize_model_predictions
from src.visualizations.selection import visualize_voxel_selection, \
    visualize_superpoint_selection, visualize_uncertainty_score_voxels, \
    visualize_uncertainty_score_superpoints

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cfg = set_paths(cfg, HydraConfig.get().runtime.output_dir)

    log.info(f'Starting demo: {cfg.action}')

    # ==================== DATASET VISUALIZATIONS ====================

    """ You can specify the dataset with the following options:
            - ds:{{ dataset_name }} (Options are KITTI360 or SemanticKITTI)
            - +split:{{ split_name }} (Options are train, val)
    
        You can specify the colorization with the following options for dataset_clouds:
            - +color:{{ color_arg }} (Options are rgb, labels) 
    """

    if cfg.option == 'dataset_scans':
        visualize_scans(cfg)
    elif cfg.option == 'dataset_clouds':
        visualize_clouds(cfg)
    elif cfg.option == 'dataset_statistics':
        visualize_statistics(cfg)
    elif cfg.option == 'augmentation':
        visualize_augmentation(cfg)
    elif cfg.option == 'scan_mapping':
        visualize_scan_mapping(cfg)

        # ==================== SUPERPOINT VISUALIZATIONS ====================

        """ You can specify the dataset with the following options:
            - ds:{{ dataset_name }} (Options are KITTI360 or SemanticKITTI)
            - +split:{{ split_name }} (Options are train, val)
        
        You can specify the feature with the following options for visualize_feature:
            - +feature:{{ feature_name }} (Options are planarity, linearity, scattering, verticality, height, density)
        """

    elif cfg.option == 'feature':
        visualize_feature(cfg)
    elif cfg.option == 'superpoints':
        visualize_superpoints(cfg)

        # ==================== EXPERIMENT VISUALIZATIONS ====================

        """ You can specify the Weights and Biases project with the following options:
            - project_name:{{ project_name }}
            
            Make sure that YAML configuration file for the experiment visualization is in the following path:
                
                experiments/{{ experiment_type }}/{{ project_name }}.yaml
        """

    elif cfg.option == 'model_comparison':
        visualize_model_comparison(cfg)
    elif cfg.option == 'loss_comparison':
        visualize_loss_comparison(cfg)
    elif cfg.option == 'baseline':
        visualize_baseline(cfg)
    elif cfg.option == 'strategy_comparison':
        visualize_learning(cfg)
    elif cfg.option == 'class_distribution':
        visualize_class_distribution(cfg)

        # ==================== SELECTION VISUALIZATIONS ====================

        """ You can specify the dataset with the following options:
            - ds:{{ dataset_name }} (Options are KITTI360 or SemanticKITTI)
            - +split:{{ split_name }} (Options are train, val)
        """

    elif cfg.option == 'voxel_selection':
        visualize_voxel_selection(cfg)
    elif cfg.option == 'superpoint_selection':
        visualize_superpoint_selection(cfg)
    elif cfg.option == 'scan_mapping':
        visualize_uncertainty_score_voxels(cfg)
    elif cfg.option == 'uncertainty_score_superpoints':
        visualize_uncertainty_score_superpoints(cfg)

    # ==================== FILTERING ====================

    elif cfg.option == 'filters':
        visualize_filters(cfg)

    # ==================== MODEL PREDICTIONS ====================

    elif cfg.option == 'model_predictions':
        visualize_model_predictions(cfg)

    # ==================== DATASET CONVERSION ====================

    elif cfg.option == 'kitti360_conversion':
        converter = KITTI360Converter(cfg)
        converter.visualize()
    elif cfg.option == 'semantic_kitti_conversion':
        raise NotImplementedError
    else:
        raise ValueError('Invalid demo type.')

    log.info('Demo completed.')


if __name__ == '__main__':
    main()
