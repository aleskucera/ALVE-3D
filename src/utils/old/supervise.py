import os
import time
import logging
import subprocess

from fabric import Connection
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def supervise_remote(cfg: DictConfig):
    # Connect to the remote machine
    c = Connection(host=cfg.connect.host, user=cfg.connect.USER)
    log.info(f'Connected to {cfg.connect.host} as {cfg.connect.USER}')

    with c.cd(cfg.connect.repo):
        # Get absolute paths on the remote machine
        pwd = c.run('pwd')
        paths = remote_paths(cfg, pwd.stdout.strip())

        # Pull the latest changes from the repository
        c.run('git pull')

        # Get the log directory path and its root directory
        root_dir = os.path.dirname(paths.output)
        log_dir = paths.output.replace(f'{root_dir}/', '')

        # Run the script on the remote machine and wait for its start
        c.run(f'sh scripts/simulation.batch {os.path.join(paths.output, "slave")}', asynchronous=True)
        time.sleep(5)

    # Synchronize the log directory
    pkg = 'package.tar.gz'
    pkg_dir = os.path.join(root_dir, pkg)
    while True:
        log.info(f'Syncing the directory...')

        # Compress the log directory
        with c.cd(root_dir):
            c.run(f'tar -czf {pkg} {log_dir}')

        # Copy the compressed log directory to the local machine
        c.get(pkg_dir, pkg)

        # Extract the log directory and remove the compressed log directory
        subprocess.run(['tar', '-xzf', pkg, '-C', os.path.dirname(cfg.path.output)])
        subprocess.run(['rm', pkg])

        # Remove the compressed log directory from the remote machine
        c.run(f'rm {pkg_dir}')

        time.sleep(10)
