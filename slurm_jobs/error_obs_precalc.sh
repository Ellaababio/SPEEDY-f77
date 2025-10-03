#!/bin/bash
#SBATCH --job-name=ERROR_OBS_PRECALC
#SBATCH --output=ERR_OBS_PRECALC_%j.out
#SBATCH --error=ERR_OBS_PRECALC_%j.err
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

cd ../amlcs/

srun python amlcs_da.py ensf_main_da_std4obs.csv

echo "--- Main DA Job Finished ---"