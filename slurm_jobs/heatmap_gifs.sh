#!/bin/bash
#SBATCH --job-name=heatmap_gifs
#SBATCH --output=heatmap_gifs_%j.out
#SBATCH --error=heatmap_gifs_%j.err
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

srun python heatmap_gifs.py

echo "--- Plotting Job Finished ---"