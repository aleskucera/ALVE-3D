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

show_help_selection () {
  echo "Usage: select.batch [options]"
  echo "Options:"
  echo "  --first-selection     Select first voxels"
  echo "  -b, --batch-size     Batch size (default: 64)"
  echo "  -c, --criterion       Active learning criterion (default: random)"
  echo "  -e, --expected        Expected percentage of labeled data (default: 0)"
  echo "  -o, --objects         Active learning objects (default: voxels)"
  echo "  -p, --percentage      Percentage of dataset to select (default: 0.5)"
  echo "  -s, --size            Dataset size (default: null)"
  echo "  -h, --help            Display this help and exit"
}

show_help_train () {
  echo "Usage: train.batch [options]"
  echo "Options:"
  echo "  --from-scratch        Train from scratch, do not load model"
  echo "  -c, --criterion       Active learning criterion (default: random)"
  echo "  -e, --expected        Expected percentage of labeled data (default: 0)"
  echo "  -m, --model           Model type (default: semantic)"
  echo "  -o, --objects         Active learning objects (default: voxels)"
  echo "  -s, --size            Dataset size (default: null)"
  echo "  -h, --help            Display this help and exit"
}

show_help () {
  echo "Usage: help.sh [options]"
  echo "Options:"
  echo "  --job                 Display help for job"
  echo "  --selection           Display help for selection"
  echo "  --train               Display help for training"
  echo "  -h, --help            Display this help and exit"
}


while [ "$#" -gt 0 ]; do
  case $1 in
    --selection) show_help_selection; exit 0 ;;
    --train) show_help_train; exit 0 ;;
    --job) show_help_job; exit 0 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
  esac
  shift
done

show_help
