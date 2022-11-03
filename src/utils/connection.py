from fabric import Connection
import subprocess


def connect(host: str, user: str):
    return Connection(host=host, user=user)


def pull_repository(c: Connection, directory: str = 'ALVE-3D'):
    c.run(f'(cd {directory} && git pull)')
    c.run(f'ls {directory} -l')


def get_directory(c: Connection, directory: str, dest: str, root: str = 'ALVE-3D'):
    # Compress the directory
    c.run(f'(cd {root} && tar -czf package.tar.gz {directory})')

    # Copy the compressed directory to the local machine
    c.get(f'{root}/package.tar.gz', f'{dest}/package.tar.gz')

    # Uncompress the log directory
    subprocess.run(['tar', '-xzf', f'{dest}/package.tar.gz', '-C', dest])
    subprocess.run(['rm', f'{dest}/package.tar.gz'])

    # Remove the compressed log directory from the remote machine
    c.run(f'rm {root}/package.tar.gz')


def main():
    # connect to the remote machine to directory ALVE-3D
    c = Connection(host='login3.rci.cvut.cz', user='kuceral4')
    pull_repository(c)
    get_directory(c, 'tmp', '/home/ales/DeepLearning/ALVE-3D')


if __name__ == '__main__':
    main()
