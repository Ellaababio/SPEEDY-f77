#!/bin/bash
#SBATCH --job-name=error_obs_run_plots
#SBATCH --output=error_obs_run_plots_%j.out
#SBATCH --error=error_obs_run_plots_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G

echo "--- Plotting Job Started ---"

module purge
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env

cd amlcs/

srun python error_plots.py error_obs_run_plots.csv

echo "--- Plotting Job Finished ---"