#!/bin/bash

echo "-----------------------------------"
echo "============= ALVE-3D ============="
echo "-----------------------------------"


echo "What do you want to do? Choose an option:"
echo "1) Train a model with a fully labeled dataset"
echo "2) Train a model in active learning experiment"
echo "3) Select voxels for labeling in active learning experiment"

read -r choice

case $choice in
  1)
    action="train_model"
    script="scripts/train/train_model.batch"
    ;;
  2)
    action="train_model_active"
    script="scripts/active/train.batch"
    ;;
  3)
    action="select_voxels"
    script="scripts/active/select.batch"
    ;;
  *)
    echo "Invalid action choice"
    exit 1
    ;;
esac

model=null
strategy=null

if [[ "$action" == "train_model" ]]; then
  echo "Select a model:"
  echo "1) DeepLabV3"
  echo "2) SalsaNext"

  read -r choice

  case $choice in
  1)
    model="DeepLabV3"
    ;;
  2)
    model="SalsaNext"
    ;;
  *)
    echo "Invalid model choice"
    exit 1
    ;;
  esac
elif [[ "$action" == "train_model_active" ]] || [[ "$action" == "select_voxels" ]]; then
  echo "Select an active learning strategy:"
  echo "1) Average Entropy"
  echo "2) Epistemic Uncertainty"
  echo "3) Random"
  echo "4) Viewpoint Entropy"
  echo "5) Viewpoint Variance"

  read -r choice

  case $choice in
  1)
    strategy="AverageEntropy"
    ;;
  2)
    strategy="EpistemicUncertainty"
    ;;
  3)
    strategy="Random"
    ;;
  4)
    strategy="ViewpointEntropy"
    ;;
  5)
    strategy="ViewpointVariance"
    ;;
  *)
    echo "Invalid strategy choice"
    exit 1
    ;;
  esac
else
  echo "Invalid choice"
  exit 1
fi

echo "Select a dataset:"
echo "1) KITTI360"
echo "2) SemanticKITTI"

read -r choice

case $choice in
  1)
    dataset="KITTI360"
    ;;
  2)
    dataset="SemanticKITTI"
    ;;
  *)
    echo "Invalid dataset choice"
    exit 1
    ;;
esac

# Print selected options
echo "-----------------------------------"
echo "Selected options:"
echo "Action: $action"
echo "Model: $model"
echo "Dataset: $dataset"
echo "Strategy: $strategy"
echo "-----------------------------------"