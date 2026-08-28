#!/bin/bash
#SBATCH --job-name=VARIABLE_LEVEL_DIAGNOSTIC
#SBATCH --output=VARIABLE_LEVEL_DIAGNOSTIC_%j.out
#SBATCH --error=VARIABLE_LEVEL_DIAGNOSTIC_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G

echo "--- Testing Job Started ---"

module purge
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env


srun python parse_reverse_sde_fallbacks.py ENSF_MAIN_GAUSSIAN_CHECK_11321070.out

echo "--- Testing Job Finished ---"