import os
import subprocess


def start_tensorboard(output: str) -> subprocess.Popen:
    output_dir = os.path.dirname(output)
    child_process = subprocess.Popen(['tensorboard', f'--logdir={output_dir}', '--port=6006', '--load_fast=false'])
    return child_process


def terminate_tensorboard():
    subprocess.run(['pkill', 'tensorboard'])


if __name__ == '__main__':
    terminate_tensorboard()
