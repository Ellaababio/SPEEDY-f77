#!/bin/bash

#SBATCH --job-name=DA_PREP
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00          # 1 hour
#SBATCH --qos=normal
#SBATCH --output=DA_PREP_%j.out
#SBATCH --error=DA_PREP_%j.err

echo "--- Pre-processing Job Started ---"

# --- Environment Setup ---
eval "$(conda shell.bash hook)"
conda activate amlcs
export LD_LIBRARY_PATH=/gpfs/research/chipilskigroup/hch/AMLCS/speedy_libs/lib:$LD_LIBRARY_PATH

# --- Run Script ---
# Change to the directory containing the python scripts
cd amlcs/

# Run the pre-processing script with its config file
srun python amlcs_pre.py amlcs_pre_t21.csv

echo "--- Pre-processing Job Finished ---"
