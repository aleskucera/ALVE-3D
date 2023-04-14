import torch


def get_parser(parser_type: str, device: torch.device):
    """ Get the parser for the dataset. Parser is used to parse the data from the dataset
    __getitem__ method to the format that the training function expects.

    :param parser_type: The type of parser to use (semantic, partition)
    :param device: The device used for training
    """

    if parser_type == 'semantic':
        return SemanticParser(device)
    elif parser_type == 'partition':
        return PartitionParser(device)
    else:
        raise ValueError(f'Unknown parser: {parser_type}')


class BaseParser(object):
    def __init__(self, device: torch.device):
        self.device = device

    def parse_batch(self, batch: tuple) -> tuple:
        raise NotImplementedError


class SemanticParser(BaseParser):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def parse_batch(self, batch: tuple) -> tuple:
        proj_images, proj_labels, _, _, _ = batch
        return proj_images.to(self.device), proj_labels.to(self.device)


class PartitionParser(BaseParser):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def parse_batch(self, batch: tuple) -> tuple:
        clouds, clouds_global, labels, edg_source, edg_target, is_transition, xyz = batch

        clouds, clouds.to(self.device)
        clouds_global, clouds_global.to(self.device)
        labels = labels.to(self.device)

        is_transition = is_transition.squeeze(0).to(self.device)

        edg_source = edg_source.squeeze(0).numpy()
        edg_target = edg_target.squeeze(0).numpy()
        xyz = xyz.squeeze(0).numpy()

        return (clouds, clouds_global), (labels, edg_source, edg_target, is_transition, xyz)
