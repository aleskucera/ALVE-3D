<h1 align="center">ALVE-3D</h1>

<p align="center">
<img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white" alt="Tensorflow"/>
<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
</p>

## Introduction

This code is official implementation of the project **ALVE-3D (Active Learning with Viewpoint Entropy
for 3D Semantic Segmentation)**. We propose a novel active learning framework for 3D semantic segmentation based on the
viewpoint entropy.
The framework is will be evaluated on SemanticKITTI and SemanticUSL datasets.

## Overview

- [Requirements](#requirements)
- [Project Architecture](#project-architecture)
    - [Repository Structure](#repository-structure)
    - [Basic Principles](#basic-principles)
        - [Configuration](#configuration)
        - [Monitoring](#monitoring)
        - [Logging](#logging)
- [Usage](#usage)
    - [Available Launch Files](#available-launch-files)
    - [Available Demos](#available-demos)

## Requirements

- Python 3.9

for all other requirements, please see `environment.yaml`. You can recreate the environment with:

    conda env create -f environment.yaml

## Project Architecture

### Repository Structure

    .
    ├── conf
    ├── data
    │   ├── SemanticKITTI -> /path/to/SemanticKITTI
    │   └── SemanticUSL -> /path/to/SemanticUSL
    ├── models
    │   └── pretrained
    ├── outputs
    ├── scripts
    ├── singularity
    ├── src
    ├── demo.py
    ├── environment.yaml
    └── main.py

- `conf`: Folder containing the configuration files for the experiments. The configuration files are in YAML format.
  The project depends on [Hydra](https://hydra.cc/) for configuration management. More details about the configuration
  in the [Configuration](#configuration) section.
- `data`: Folder containing the symbolic links to the datasets.
- `models`: Folder containing the models that are used in the experiments.
  For evaluation of the pretrained models, please use the pretrained directory.
- `outputs`: Folder containing logs of the experiments. More details in the [Logging](#logging) section.
- `scripts`: Folder containing the scripts for training, evaluation and visualization of the models on RCI cluster.
- `src`: Folder containing the source code of the project.
- `demo.py`: Script is used for demo of the finished features of the project.
- `main.py`: The main script of the project. It is used for training and testing of the model. More details in
  the [Usage](#usage) section.

### Basic Principles

This section describes the basic principles of the project. For faster development, the [Hydra](https://hydra.cc/) is
used for configuration management and logging. Before running the project, you should set the configuration file. Let's
dive into the details.

#### Configuration

The main configuration file is `conf/config.yaml`. This file contains the following sections:

- `defaults`: These are default configuration files. The items have the following syntax `directory:filename`. The
  directory
  is the relative path to the `conf` directory. The filename is the name of the configuration file without the
  extension. The configuration files are loaded in the order they are specified in the list.
- **other**: These are the configuration parameters that are somewhat more important. The example is the `action`
  parameter. It specifies the action that should be performed and doesn't load any configuration file. The `action`
  parameter can be specified in the command line by specifying the `action={action_name}`. More details about the
  options of the `action` parameter can be found in the [Usage](#usage) section.

> **Note**: The `action` parameter is not the only parameter that can be specified in the command line. All the
> parameters can be specified in the command line. The syntax is `parameter=value`. For example, if you want to
> specify the dataset configuration file to SemanticKITTI, you can use `ds=kitti`.

#### Monitoring

The project uses [Tensorboard](https://www.tensorflow.org/tensorboard) for the monitoring. The Tensorboard is started
automatically when needed and the logs are saved in the `outputs` directory. The Tensorboard can be accessed on
url `http://localhost:6006`. After the action which uses the Tensorboard is finished, all the Tensorboard processes are
killed.

#### Logging

The project uses Hydra logging. It is configured in the `conf/hydra` files so that the logs will usually be saved
in `outputs/{date}/{action}-{time}-{info}` directory. The `date` and `time` are the date and the time when the project
was started. The `info` is the information about the configuration. The `action` is the name of the action that was
performed.

## Usage

The project can be used for training and testing of the model. Run the `main.py` script for training and testing of the
model by:

    python main.py launch={launch_file}

where `{launch_file}` is the name of the launch file without the extension. The launch files are located in
the `conf/lauch` directory.

#### Available Launch Files

- `train`: Train the model.
- `test`: Test the model.
- `train_dev`: Train the model on the development set.
- `test_dev`: Test the model on the development set.

You can also run the `demo.py` script for demo of the finished features of the project by:

    python demo.py action={action}

> **Note**: The difference between `action` and `launch` parameter is that the `launch` parameter is used for
> overriding multiple configuration parameters at once including the `action` parameter. You can see
> that in the launch files.

#### Available Demos

- `paths`: Demo of the absolute paths in the configuration files.
- `dataset`: Demo of the SemanticDataset and its visualization.
- `select_indices`: Demo of the selection of the indices of the samples based on the entropy.




