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
  echo "Usage: train_model.batch [options]"
  echo "Options:"
  echo "  -b, --batch-size      Batch size (default: 32)"
  echo "  -e, --experiment      Experiment name (default: default)"
}

show_help_train_original () {
  echo "Usage: train_model_original.batch [options]"
  echo "Options:"
  echo "  -b, --batch-size      Batch size (default: 32)"
  echo "  -e, --experiment      Experiment name (default: default)"
}

show_help () {
  echo "Usage: help.sh [options]"
  echo "Options:"
  echo "  -j, --job                 Display help for job"
  echo "  -t, --train_semantic      Display help for training semantic model"
  echo "  -h, --help                Display this help and exit"
}


while [ "$#" -gt 0 ]; do
  case $1 in
    -t|--train_model) show_help_train; exit 0 ;;
    -j|--job) show_help_job; exit 0 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
  esac
  shift
done

show_help