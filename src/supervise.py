import os
import time
import logging

from fabric import Connection
from omegaconf import DictConfig
from .utils import pull_repository, get_directory

log = logging.getLogger(__name__)


def supervise_remote(cfg: DictConfig):
    repo_name = 'ALVE-3D'
    # c = Connection(host=cfg.connect.host, user=cfg.connect.user)
    c = Connection(host='login3.rci.cvut.cz', user='kuceral4')
    log.info(f'Connected to {cfg.connect.host} as {cfg.connect.user}')
    pull_repository(c)

    while True:
        log.info('Syncing the tmp directory...')
        get_directory(c, 'tmp', cfg.path.root)
        time.sleep(10)
