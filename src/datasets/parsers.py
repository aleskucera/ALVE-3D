import torch


class Parser(object):
    def __init__(self, device: torch.device):
        self.device = device

    def parse_batch(self, batch: tuple) -> tuple:
        raise NotImplementedError


class SemanticParser(Parser):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def parse_batch(self, batch: tuple) -> tuple:
        proj_images, proj_labels, _, _, _ = batch
        return proj_images.to(self.device), proj_labels.to(self.device)


class PartitionParser(Parser):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def parse_batch(self, batch: tuple) -> tuple:
        clouds, clouds_global, objects, edge_sources, edge_targets, edge_transitions = batch

        clouds = clouds.to(self.device)
        objects = objects.to(self.device)
        clouds_global = clouds_global.to(self.device)

        edge_sources = edge_sources.numpy()
        edge_targets = edge_targets.numpy()

        return (clouds, clouds_global), (objects, edge_sources, edge_targets, edge_transitions)


class SemanticKITTIParser(Parser):
    def __init__(self, device: torch.device):
        super().__init__(device)

    def parse_batch(self, batch: tuple) -> tuple:
        proj_images, proj_labels = batch
        return proj_images.to(self.device), proj_labels.to(self.device)


def get_parser(parser_type: str, device: torch.device) -> Parser:
    """ Get the parser for the dataset. Parser is used to parse the data from the dataset
    __getitem__ method to the format that the training function expects.

    :param parser_type: The type of parser to use (semantic, partition)
    :param device: The device used for training
    """

    if parser_type == 'semantic':
        return SemanticParser(device)
    elif parser_type == 'partition':
        return PartitionParser(device)
    elif parser_type == 'semantic_kitti':
        return SemanticKITTIParser(device)
    else:
        raise ValueError(f'Unknown parser: {parser_type}')
