#!/bin/bash
#SBATCH --job-name=ENKF_MC_OBS
#SBATCH --output=ENKF_MC_OBS_%j.out
#SBATCH --error=ENKF_MC_OBS_%j.err
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

srun python amlcs_da.py amlcs_da_ENKF_MC_obs.csv

echo "--- Main DA Job Finished ---"