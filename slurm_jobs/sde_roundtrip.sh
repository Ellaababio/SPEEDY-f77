#!/bin/bash
#SBATCH --job-name=RT_SDE
#SBATCH --output=RT_SDE_%j.out
#SBATCH --error=RT_SDE_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 00:20:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=6G

echo "--- Reverse-SDE Round-trip Test Started ---"

module purge
module load gnu/4.8.5
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

# go where your scripts live; adjust if needed
cd ../amlcs/

# Example 1: use a specific saved block
srun python reverse_sde_roundtrip.py \
 --npy ../runs/t21_50_0.05_5_ReverseSDE_1_5_100/gauss_checks/XB_block_0.npy \
 --outdir ../runs/t21_50_0.05_5_ReverseSDE_1_5_100/roundtrip_block0 \
 --psteps 200 --eps_alpha 0.05 --deterministic 1

# Example 2: synthetic (no --npy)
srun python reverse_sde_roundtrip.py \
  --outdir ../runs/roundtrip_synth \
  --psteps 200 --eps_alpha 0.05 --deterministic 1

echo "--- Reverse-SDE Round-trip Test Finished ---"
