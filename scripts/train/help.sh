#!/bin/bash

show_help_job() {
  echo "Usage: sbatch [options] [script] [script options]"
  echo "Options:"
  echo "  --nodes              Number of nodes (default: 1)"
  echo "  --time               Time limit"
  echo "  --error              Standard error file"
  echo "  --output             Standard output file"
  echo "  --partition          Partition name"
  echo "  --gpus-per-node      Number of GPUs per node"
  echo "  --ntasks-per-node    Number of CPUs per node (default: 4)"
}

show_help_train () {
  echo "Usage: train_semantic.batch [options]"
  echo "Options:"
  echo "  -b, --batch-size      Batch size (default: 64)"
  echo "  -d, --dataset         Dataset name (default: kitti-360, options: kitti-360, semantic-kitti)"
  echo "  -e, --epochs          Number of epochs (default: 100)"
  echo "  -m, --model           Model name (default: salsanext, options: salsanext, deeplabv3)"
  echo "  -p, --patience        Patience (default: 20)"
  echo "  -h, --help            Display this help and exit"
}

show_help () {
  echo "Usage: help.sh [options]"
  echo "Options:"
  echo "  --job                 Display help for job"
  echo "  --train_semantic      Display help for training semantic model"
  echo "  -h, --help            Display this help and exit"
}


while [ "$#" -gt 0 ]; do
  case $1 in
    --train_semantic) show_help_train; exit 0 ;;
    --job) show_help_job; exit 0 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
  esac
  shift
done

show_help