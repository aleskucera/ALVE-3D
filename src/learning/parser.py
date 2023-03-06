import torch


def get_parser(parser_type: str):
    if parser_type == 'semantic':
        return SemanticParser()
    elif parser_type == 'active':
        return ActiveParser()
    elif parser_type == 'partition':
        return PartitionParser()
    else:
        raise ValueError(f'Unknown parser: {parser_type}')


class BaseParser(object):
    def __init__(self):
        pass

    @staticmethod
    def parse_data(batch: tuple, device: torch.device) -> tuple:
        raise NotImplementedError


class SemanticParser(BaseParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_data(batch: tuple, device: torch.device) -> tuple:
        proj_images, proj_labels, indices = batch

        proj_images = proj_images.to(device)
        proj_labels = proj_labels.to(device)
        
        return proj_images, proj_labels


class ActiveParser(BaseParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_data(batch: tuple, device: torch.device) -> tuple:
        inputs, targets, indices = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets, indices


class PartitionParser(BaseParser):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse_data(batch: tuple, device: torch.device) -> tuple:
        clouds, clouds_global, labels, edg_source, edg_target, is_transition, xyz = batch

        clouds, clouds.to(device)
        clouds_global, clouds_global.to(device)
        labels = labels.to(device)

        is_transition = is_transition.squeeze(0).to(device)

        edg_source = edg_source.squeeze(0).numpy()
        edg_target = edg_target.squeeze(0).numpy()
        xyz = xyz.squeeze(0).numpy()

        return (clouds, clouds_global), (labels, edg_source, edg_target, is_transition, xyz)
