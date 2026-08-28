#!/bin/bash
#SBATCH --job-name=SPATIAL_ANALYZER
#SBATCH --output=SPATIAL_ANALYZER_%j.out
#SBATCH --error=SPATIAL_ANALYZER_%j.err
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

# Better to write to a new file with degrees included
srun python ../amlcs/augment_block_map_with_degrees.py \
  --in  ../runs/t21_50_0.05_5_ReverseSDE_1_5_100/block_map_detailed.json \
  --out ../runs/t21_50_0.05_5_ReverseSDE_1_5_100/block_map_detailed_deg.json

echo "--- Testing Job Finished ---"
