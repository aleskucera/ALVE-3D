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

show_help_conversion () {
  echo "Usage: convert.batch [options]"
  echo "Options:"
  echo "  -d, --dataset         Dataset name (default: KITTI360, options: KITTI360, SemanticKITTI)"
  echo "  -s, --sequence        Sequence to convert (default: 3)"
}

show_help_create_superpoints () {
  echo "Usage: create_superpoints.batch [options]"
  echo "Options:"
  echo "  -d, --dataset         Dataset name (default: KITTI360, options: KITTI360, SemanticKITTI)"
}

show_help () {
  echo "Usage: help.sh [options]"
  echo "Options:"
  echo "  --job                 Display help for job"
  echo "  --conversion          Display help for conversion"
  echo "  -h, --help            Display this help and exit"
}


while [ "$#" -gt 0 ]; do
  case $1 in
    --conversion) show_help_conversion; exit 0 ;;
    --job) show_help_job; exit 0 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown parameter passed: $1" >&2; exit 1 ;;
  esac
  shift
done

show_help