#!/bin/bash
#SBATCH --job-name=ENSF_GAUSS_CHECK
#SBATCH --output=ENSF_GAUSS_CHECK_%j.out
#SBATCH --error=ENSF_GAUSS_CHECK_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G

echo "--- Validation Job Started ---"

module purge
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env

cd ../amlcs
for f in ../runs/t21_10_0.05_5_ReverseSDE_1_5_100/gauss_checks/XB_block_*.npy; do
  bname=$(basename "$f" .npy)
  srun python gaussian_validation.py --npy "$f" --outdir "../runs/t21_10_0.05_5_ReverseSDE_1_5_100/gauss_checks/gauss_${bname}" --psteps 200 --eps_alpha 0.05
done


echo "--- Validation Job Finished ---"