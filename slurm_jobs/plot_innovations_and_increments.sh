#!/bin/bash
#SBATCH --job-name=plot_innovations_and_increments
#SBATCH --output=plot_innovations_and_increments_%j.out
#SBATCH --error=plot_innovations_and_increments_%j.err
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

srun python plot_innovations_and_increments.py

echo "--- Plotting Job Finished ---"