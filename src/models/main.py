import torch
from omegaconf import DictConfig
import segmentation_models_pytorch as smp

from .pointnet import PointNet
from .salsanext import SalsaNext


def get_model(cfg: DictConfig, device: torch.device):
    num_outputs = cfg.ds.num_classes
    num_inputs = cfg.ds.num_channels

    if cfg.model.architecture == 'SalsaNext':
        model = SalsaNext(num_inputs, num_outputs)
    elif cfg.model.architecture == 'DeepLabV3':
        model = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=num_inputs,
            classes=num_outputs,
            activation='softmax2d',
        )

    # elif cfg.model.architecture == 'PointNet':
    #     model = PointNet(num_features=features['local'],
    #                      num_global_features=features['global'],
    #                      out_features=num_outputs)
    else:
        raise ValueError(f'Unknown architecture: {cfg.model.architecture}')

    model = model.to(device)
    return model
