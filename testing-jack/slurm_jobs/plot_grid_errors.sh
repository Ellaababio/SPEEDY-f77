#!/bin/bash
#SBATCH --job-name=PLOT_GRID_ERRORS
#SBATCH --output=PLOT_GRID_ERRORS_%j.out
#SBATCH --error=PLOT_GRID_ERRORS_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G

echo "--- Plotting Job Started ---"

module purge
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env

cd ../amlcs/

srun python plot_grid_errors.py

echo "--- Plotting Job Finished ---"