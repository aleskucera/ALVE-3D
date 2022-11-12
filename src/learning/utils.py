import torch
from torch.nn import Conv2d
from omegaconf import DictConfig
import torchvision.models.segmentation as tms


def parse_data(data: tuple, device: torch.device):
    image_batch, label_batch, idx_batch = data
    image_batch = image_batch.to(device)
    label_batch = label_batch.to(device)
    return image_batch, label_batch, idx_batch


def create_model(cfg: DictConfig):
    model = _load_model(cfg.model.architecture, cfg.model.pretrained)
    n_inputs = cfg.train.n_inputs
    n_outputs = cfg.train.n_outputs

    if cfg.model.encoder == 'mobilenet_v3_large':
        model.backbone['0'][0] = Conv2d(n_inputs, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    else:
        model.backbone['conv1'] = Conv2d(n_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    if cfg.model.arch == 'lraspp':
        model.classifier.low_classifier = Conv2d(40, n_outputs, kernel_size=(1, 1), stride=(1, 1))
        model.classifier.high_classifier = Conv2d(128, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif cfg.model.arch == 'fcn':
        model.classifier[-1] = Conv2d(512, n_outputs, kernel_size=(1, 1), stride=(1, 1))
    elif cfg.model.arch == 'deeplabv3':
        model.classifier[-1] = Conv2d(256, n_outputs, kernel_size=(1, 1), stride=(1, 1))

    return model


def _load_model(architecture: str, pretrained: bool = True):
    model = eval(f'tms.{architecture}')(pretrained=pretrained)
    return model
