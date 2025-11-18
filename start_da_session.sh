#!/bin/bash

module purge
module load gnu/4.8.5

# Initialize conda correctly
source /gpfs/research/software/python/anaconda311/etc/profile.d/conda.sh
conda activate speedy_da_env

export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

echo "Environment loaded on $(hostname) with conda env = $CONDA_DEFAULT_ENV"
