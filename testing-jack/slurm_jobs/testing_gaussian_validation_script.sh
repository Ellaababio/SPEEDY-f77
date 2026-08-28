#!/bin/bash
#SBATCH --job-name=TESTING_GAUSSIAN_VALIDATION
#SBATCH --output=TESTING_GAUSSIAN_VALIDATION_%j.out
#SBATCH --error=TESTING_GAUSSIAN_VALIDATION_%j.err
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

cd ../amlcs/

srun python testing_gaussian_validation.py

echo "--- Testing Job Finished ---"