<h1 align="center">MuVAL</h1>

<p align="center">
<img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/>
</p>

> **Warning**
> This repository is still under development and is not ready for use.

## Introduction

MuVAL (Multi-View Active Learning) is a framework for active learning on multi-view data for semantic segmentation of
LiDAR-based datasets like KITT-360 or SemanticKITTI. The idea is to use the multi-view nature of the data to improve
the reliability of the uncertainty estimates of the model.

## Overview

The documentation structure is as follows:

- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)

## Setup

First clone the repository:

```bash
git clone git@github.com:aleskucera/MuVAL.git
```

Then install the dependencies:

```bash
conda env create -f environment.yaml
```

Then download the datasets into the `data` folder. The directory structure should be as follows:

```
data
├── KITTI360
│   ├── data_3d_raw
│   ├── data_3d_semantics
│   ├── data_poses
├── SemanticKITTI
│   ├── sequences
```

To use the datasets, you need to convert them to a format that is compatible with the framework. To do so, run the
following commands:

```bash
python process.py option=convert_dataset ds="{{ dataset }}" +sequence="{{ sequence }}"
```

where `{{ dataset }}` is either `KITTI360` or `SemanticKITTI` and `{{ sequence }}` is the sequence number (e.g. `00`).

If the superpoints will be used, you can generate them by running:

```bash
python process.py option=create_superpoints ds="{{ dataset }}"
```

where `{{ dataset }}` is either `KITTI360` or `SemanticKITTI` and `{{ sequence }}` is the sequence number (e.g. `00`).

If the ReDAL features will be used, you can generate them by running:

```bash
python process.py option=compute_redal_features ds="{{ dataset }}"
```

where `{{ dataset }}` is either `KITTI360` or `SemanticKITTI` and `{{ sequence }}` is the sequence number (e.g. `00`).




