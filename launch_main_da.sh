#!/bin/bash
#SBATCH --job-name=DA_MAIN
#SBATCH --output=DA_MAIN_%j.out
#SBATCH --error=DA_MAIN_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G

echo "--- Main DA Job Started ---"

module purge
module load gnu/4.8.5
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

cd amlcs/

srun python amlcs_da.py amlcs_da_t21_LEnKF_s2r1.csv

echo "--- Main DA Job Finished ---"