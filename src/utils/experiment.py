import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class Experiment(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        log.info(self.__repr__())

    @property
    def project(self):
        if self.cfg.action == 'train_semantic':
            return 'Train Semantic Model'
        elif self.cfg.action == 'train_partition':
            return 'Train Partition Model'
        elif self.cfg.action == 'train_semantic_active':
            return f'AL - {self.cfg.active.criterion}_{self.cfg.active.selection_objects}'
        elif self.cfg.action == 'train_partition_active':
            return f'AL - {self.cfg.active.criterion}_{self.cfg.active.selection_objects}'
        elif self.cfg.action == 'select_voxels':
            return f'AL - {self.cfg.active.criterion}_{self.cfg.active.selection_objects}'

    @property
    def group(self):
        if self.cfg.action == 'train_semantic':
            return self.cfg.ds.name
        elif self.cfg.action == 'train_partition':
            return self.cfg.ds.name
        elif self.cfg.action == 'train_semantic_active':
            return f'{self.cfg.ds.name}'
        elif self.cfg.action == 'train_partition_active':
            return f'{self.cfg.ds.name}'
        elif self.cfg.action == 'select_voxels':
            return f'{self.cfg.ds.name}'

    @property
    def job_type(self):
        if self.cfg.action == 'train_semantic':
            return None
        elif self.cfg.action == 'train_partition':
            return None
        elif self.cfg.action == 'train_semantic_active':
            return 'Training Semantic Model'
        elif self.cfg.action == 'train_partition_active':
            return 'Training Partition Model'
        elif self.cfg.action == 'select_voxels':
            return 'Selection'

    @property
    def name(self):
        if self.cfg.action == 'train_semantic':
            return self.cfg.model.architecture
        elif self.cfg.action == 'train_partition':
            return self.cfg.model.architecture
        elif self.cfg.action == 'train_semantic_active':
            return f'Training Semantic - {self.cfg.active.expected_percentage_labeled}%'
        elif self.cfg.action == 'train_partition_active':
            return f'Training Partition - {self.cfg.active.expected_percentage_labeled}%'
        elif self.cfg.action == 'select_voxels':
            return f'Selection - {self.cfg.active.expected_percentage_labeled + self.cfg.active.select_percentage}%'

    @property
    def info(self):
        if self.cfg.action == 'train_semantic':
            return f'{self.cfg.model.architecture}_{self.cfg.ds.name}'
        elif self.cfg.action == 'train_partition':
            return f'{self.cfg.model.architecture}_{self.cfg.ds.name}'
        elif self.cfg.action == 'train_semantic_active':
            return f'{self.cfg.active.criterion}_{self.cfg.active.selection_objects}' \
                   f'_{self.cfg.model.architecture}_{self.cfg.ds.name}'
        elif self.cfg.action == 'train_partition_active':
            return f'{self.cfg.active.criterion}_{self.cfg.active.selection_objects}' \
                   f'_{self.cfg.model.architecture}_{self.cfg.ds.name}'
        elif self.cfg.action == 'select_voxels':
            return f'{self.cfg.active.criterion}_{self.cfg.active.selection_objects}' \
                   f'_{self.cfg.model.architecture}_{self.cfg.ds.name}'

    @property
    def model(self):
        return f'Model_{self.info}'

    @property
    def history(self):
        return f'History_{self.info}'

    @property
    def selection(self):
        return f'Selection_{self.info}'

    @property
    def metric_stats(self):
        return f'MetricStats_{self.info}'

    @property
    def dataset_stats(self):
        return f'DatasetStats_{self.info}'

    def __str__(self):
        return f'\n\nExperiment: \n' \
               f'\t - Project: {self.project}\n' \
               f'\t - Group: {self.group}\n' \
               f'\t - Job Type: {self.job_type}\n' \
               f'\t - Name: {self.name}\n' \
               f'\t - Info: {self.info}\n' \
               f'\t - Model: {self.model}\n' \
               f'\t - History: {self.history}\n' \
               f'\t - Selection: {self.selection}\n' \
               f'\t - Metric Stats: {self.metric_stats}\n' \
               f'\t - Dataset Stats: {self.dataset_stats}\n'

    def __repr__(self):
        return self.__str__()
