#!/bin/bash
#SBATCH --job-name=ENSF_PRE_GAUSSIAN_CHECK
#SBATCH --output=ENSF_PRE_GAUSSIAN_CHECK_%j.out
#SBATCH --error=ENSF_PRE_GAUSSIAN_CHECK_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH --priority=normal
#SBATCH -t 12:00:00          # 12 hours
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G             # 12 GB memory

echo "--- Pre-processing Job Started ---"

# --- Environment Setup ---
module purge
module load gnu/4.8.5
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

# --- Run Script ---
# Change to the directory containing the python scripts
cd ../amlcs/

# Run the pre-processing script with its config file
srun python amlcs_pre.py t21_pre_gaussian_check.csv

echo "--- Pre-processing Job Finished ---"