import torch
from omegaconf import DictConfig
import torchvision.models.segmentation as tms

from .pointnet import PointNet
from .SalsaNext import SalsaNext


def get_model(cfg: DictConfig, device: torch.device):
    num_outputs = cfg.ds.num_classes
    num_inputs = cfg.ds.num_semantic_channels
    features = cfg.ds.num_partition_channels

    if cfg.model.architecture == 'SalsaNext':
        model = SalsaNext(num_inputs, num_outputs)
    elif cfg.model.architecture == 'DeepLabV3':
        model = tms.deeplabv3_resnet101(num_classes=num_outputs)
        model.backbone.conv1 = torch.nn.Conv2d(num_inputs, 64, kernel_size=(7, 7),
                                               stride=(2, 2), padding=(3, 3), bias=False)
    elif cfg.model.architecture == 'PointNet':
        model = PointNet(num_features=features['local'],
                         num_global_features=features['global'],
                         out_features=num_outputs)
    else:
        raise ValueError(f'Unknown architecture: {cfg.model.architecture}')

    model = model.to(device)
    return model
