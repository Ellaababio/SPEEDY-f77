#!/bin/bash

#SBATCH --job-name=DA_MAIN
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --qos=normal
#SBATCH --output=DA_MAIN_%j.out
#SBATCH --error=DA_MAIN_%j.err


echo "--- Main DA Job Started $(date) ---"

eval "$(conda shell.bash hook)"
conda activate amlcs
export LD_LIBRARY_PATH=/gpfs/research/chipilskigroup/hch/AMLCS/speedy_libs/lib:$LD_LIBRARY_PATH

cd amlcs/

srun python amlcs_da.py amlcs_da_t21_LEnKF_s2r1.csv

echo "--- Main DA Job Finished $(date) ---"
