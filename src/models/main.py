import torch
from omegaconf import DictConfig
import torchvision.models.segmentation as tms
import segmentation_models_pytorch as smp

from .pointnet import PointNet
from .salsanext import SalsaNext


def get_model(cfg: DictConfig, device: torch.device):
    num_outputs = cfg.ds.num_classes
    num_inputs = cfg.ds.num_semantic_channels
    features = cfg.ds.num_partition_channels

    if cfg.model.architecture == 'SalsaNext':
        model = SalsaNext(num_inputs, num_outputs)
    elif cfg.model.architecture == 'DeepLabV3':
        # model = tms.deeplabv3_resnet50(num_classes=num_outputs)
        # model.backbone.conv1 = torch.nn.Conv2d(num_inputs, 64, kernel_size=(7, 7),
        #                                        stride=(2, 2), padding=(3, 3), bias=False)

        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=num_inputs,
            classes=num_outputs,
            activation='softmax2d',
        )

    elif cfg.model.architecture == 'PointNet':
        model = PointNet(num_features=features['local'],
                         num_global_features=features['global'],
                         out_features=num_outputs)
    else:
        raise ValueError(f'Unknown architecture: {cfg.model.architecture}')

    model = model.to(device)
    return model
