#!/bin/bash
#SBATCH --job-name=ENSF_vs_ENKF_innov_inc
#SBATCH --output=ENSF_vs_ENKF_innov_inc_%j.out
#SBATCH --error=ENSF_vs_ENKF_innov_inc_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G

echo "--- Plotting Job Started ---"

module purge
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env

cd ../amlcs/

srun python innov_vs_inc.py

echo "--- Plotting Job Finished ---"