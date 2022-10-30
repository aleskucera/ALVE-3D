import os
import time
import logging
import subprocess

from fabric import Connection
from omegaconf import DictConfig
from .utils import remote_paths

log = logging.getLogger(__name__)


def supervise_remote(cfg: DictConfig):
    c = Connection(host=cfg.connect.host, user=cfg.connect.user)
    log.info(f'Connected to {cfg.connect.host} as {cfg.connect.user}')
    with c.cd(cfg.connect.repo):
        pwd = c.run('pwd')
        root_dir = pwd.stdout.strip()
        paths = remote_paths(cfg, root_dir)
        c.run('git pull')
        log.info(f'Local output directory: {cfg.path.output}')
        slave_out_dir = os.path.join(paths['output'], 'slave')
        log.info(f'Remote output directory: {slave_out_dir}')
        c.run(f'sh scripts/simulation.batch {slave_out_dir}', asynchronous=True)
    while True:
        log.info(f'Syncing the {slave_out_dir} directory')
        sync_dir(c, slave_out_dir, cfg.path.output)
        time.sleep(10)


def sync_dir(c: Connection, directory: str, dest: str):
    c.run(f'tar -czf package.tar.gz {directory}')
    c.get(f'package.tar.gz', f'{dest}/package.tar.gz')
    subprocess.run(['tar', '-xzf', f'{dest}/package.tar.gz', '-C', dest])
    subprocess.run(['rm', f'{dest}/package.tar.gz'])
    c.run(f'rm package.tar.gz')
