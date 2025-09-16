#!/bin/bash

#SBATCH --job-name=DA_PLOT
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --qos=normal
#SBATCH --output=DA_PLOT_%j.out
#SBATCH --error=DA_PLOT_%j.err

echo "--- Plotting Job Started $(date) ---"

eval "$(conda shell.bash hook)"
conda activate amlcs

cd amlcs/

srun python error_plots.py error_plot_config.csv

echo "--- Plotting Job Finished $(date) ---"