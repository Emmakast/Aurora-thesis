#!/bin/bash
module load 2023
module load PROJ/9.2.0-GCCcore-12.3.0
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
/home/ekasteleyn/aurora_thesis/aurora_env/bin/python compute_pc_std.py
