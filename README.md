# ALVE-3D

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
- [Usage](#usage)
    - [Training](#training)
    - [Testing](#testing)
    - [Visualization](#visualization)

## Requirements

- Python 3.9

for all other requirements, please see `environment.yaml`. You can recreate the environment with:

    conda env create -f environment.yaml

## Project Architecture

### Repository Structure

This is the main structure of the repository:

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

- `conf`: This folder contains the configuration files for the experiments. The configuration files are in YAML format.
  The project depends on [Hydra](https://hydra.cc/) for configuration management.
- `data`: This folder contains the symbolic links to the datasets.
  It is recommended to create symbolic links to the datasets in this folder, but you can also change the paths
  in the configuration files.
- `models`: This folder contains models that are used in the experiments.
  For evaluation of the pretrained models, please use the pretrained directory.
- `scripts`: This folder contains the scripts for training, evaluation and visualization of the models on RCI cluster.
- `src`: This folder contains the source code of the project.
- `demo.py`: This script is used for demo of the finished features of the project.
- `main.py`: This is the main script of the project. It is used for training and testing of the model.

### Basic Principles

#### Configuration

This section describes the basic principles of the project. For faster development, the [Hydra](https://hydra.cc/) is
used for configuration management. Before running the project, you should set the configuration file. Let's dive into
the
details.

The main configuration file is `conf/config.yaml`. This file contains the following sections:

- `defaults`: These are default configuration files. The items have the following syntax `directory:filename`. The
  directory
  is the relative path to the `conf` directory. The filename is the name of the configuration file without the
  extension.
  The configuration files are loaded in the order they are specified in the list.
- **other variables**: These are the variables that are defined directly in the `config.yaml` file or in the one of
  the `run`
  configuration files. These variables are used for determining, what should be run in the project.
    1. **action**: This variable is used for determining, what should be run in the project. The possible values are:
        - `train`: This value is used for training of the model.
        - `test`: This value is used for testing of the model.
        - *one of the demos*: This value is read when the `demo.py` is run.
    2. **node**: This variable is used for determining, what node is the project running on. The possible values
       are:
        - `master` (PC): This value is used for running the project on a local machine.
        - `slave` (RCI): This value is used for running the project on a cluster.

    3. **connection**: This variable is used for determining, what connection is used for running the project. The
       possible values are:
        - `local`: This value is used for running the project on a local machine.
        - `remote`: When this value is used, it indicates, that the communication between the master and the slave
          will
          be used.

  **Supported configurations**: The project supports the following configurations:
    - `train`/`test` on `master` with `local` connection - This configuration is used for development on a local
      machine. The development configuration files will be loaded. The monitoring of the progress will be used.
    - `train`/`test` on `master` with `remote` connection - This configuration is used for supervision of the
      training and the testing on the RCI cluster. The monitoring of the progress will be used.
    - `train`/`test` on `slave` with `remote` connection - This configuration will be activated on the RCI
      cluster
      when previous configuration is used. This configuration is used for training and testing of the model on
      the
      RCI cluster.
    - `train`/`test` on `slave` with `local` connection - This configuration is used for development on the RCI
      cluster. The development configuration files will be loaded. The monitoring software will not be used.

#### Monitoring

There was mentioned monitoring software. The project uses [Tensorboard](https://www.tensorflow.org/tensorboard) for the
monitoring. When running the project on the local machine (the `master` node is used), the Tensorboard will be started
automatically, even when the computing will be done on the RCI cluster (the `remote` connection is used). This is
possible by synchronization of the Tensorboard logs between the local machine and the RCI cluster. More information
logging can be found in the [Logging](#logging) section.

#### Logging

The project uses Hydra logging for logging. The logging is configured in the `conf/hydra` files. The logging is
configured that the logs will be saved in the `outputs/{date}/{time}` directory. The `date` and `time` are the date and
the time
when the project was started. Then there are the following subdirectories:

- `master`: This directory contains the logs of the `master` node when is used.
- `slave`: This directory contains the logs of the `slave` node when is used.
- *tensorboard file*: This file is used for the Tensorboard. It is created when the Tensorboard is used.

## Demo

There are 3 demos in the repository at the moment: `global_cloud`, `sample` and `formats`. You can run the demos with:

    python main.py demo=<demo_name>

or you can change the `demo` parameter in the configuration files.

## Dataset

The object SemanticDataset is Pytorch Dataset wrapper for SemanticKITTI and SemanticUSL datasets.
It is used for loading the data and creating the global map of the dataset.

Dataset uses two new [dataclasses](https://docs.python.org/3/library/dataclasses.html) for storing the data:

- `Sample`: This dataclass is used for storing the data of a single sample. It contains everything that is needed for
  training and evaluation of the model. For better performance, only the essential data are stored permanently in the
  dataset and the rest of are loaded on demand (e.g. point clouds, labels, etc.). More information about the dataclass
  can be found in the Sample section.
- `Sequence`: This dataclass is used for storing information about a structure of a single sequence. The structure of a
  sequence is defined by the `sequence_structure` parameter in the configuration file. The structure is used for
  creating
  the global map of the dataset.

## Sample class

The `Sample` class is used for storing the data of a single sample. It contains everything that is needed for training
and visualization of the dataset.
For better performance, only the essential data are stored permanently in the dataset and the rest of are loaded on
demand (e.g. point clouds, labels, etc.).
There are 3 main types of data that can be loaded in the `Sample` class:

- `learning_data`: This data are used for training and evaluation of the model. The data are loaded from the dataset
  and stored permanently in the `Sample` class by function `load_learning_data`.
- `semantic_cloud_data`: This data are used for visualization of the semantic point cloud. The data are loaded from the
  dataset
  and stored permanently in the `Sample` class by function `load_semantic_cloud`.
- `depth_image_data`: This data are used for visualization of the depth image. The data are loaded from the dataset
  and stored permanently in the `Sample` class by function `load_depth_image`.

## Sequence class

The `Sequence` class is used for storing information about a structure of a single sequence. The structure
of a sequence is defined by the `sequence_structure` parameter in the configuration file. It is used
loading the data and creating `Sample` objects by calling the `get_samples` function.

### TODO:

- [x] Create dataset wrapper for SemanticKITTI and SemanticUSL datasets
- [x] Create global map of the dataset
- [ ] Visualize the global map of the dataset
- [ ] Add singularity directory for creating singularity image from environment.yaml
- [ ] Check scripts for training, evaluation and visualization of the models




