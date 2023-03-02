#!/bin/bash

CONDA_ENV=/opt/conda/envs/ALVE-3D
ALVE_3D=/home/kuceral4/ALVE-3D

#CONDA_ENV=/home/ales/anaconda3/envs/ALVE-3D
#ALVE_3D=/home/ales/Thesis/ALVE-3D

echo "---------- Compiling ply_c ----------"
cd $ALVE_3D/src/superpoints/ply_c || echo "Didn't find ply_c"
cmake . -DPYTHON_LIBRARY=$CONDA_ENV/lib/libpython3.10.so -DPYTHON_INCLUDE_DIR=$CONDA_ENV/include/python3.10 \
-DBOOST_INCLUDE_DIR=$CONDA_ENV/include -DEIGEN3_INCLUDE_DIR=$CONDA_ENV/include/eigen3
make

echo "---------- Compiling cut-pursuit ----------"
cd $ALVE_3D/src/superpoints/cut-pursuit || echo "Didn't find cut-pursuit"
mkdir build || echo "build folder exists"
cd build || echo "Didn't find build folder"
cmake .. -DPYTHON_LIBRARY=$CONDA_ENV/lib/libpython3.10.so -DPYTHON_INCLUDE_DIR=$CONDA_ENV/include/python3.10 \
-DBOOST_INCLUDE_DIR=$CONDA_ENV/include -DEIGEN3_INCLUDE_DIR=$CONDA_ENV/include/eigen3
make