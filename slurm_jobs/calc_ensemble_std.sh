#!/bin/bash
#SBATCH --job-name=CALC_ENSEMBLE_STD
#SBATCH --output=CALC_ENSEMBLE_STD_%j.out
#SBATCH --error=CALC_ENSEMBLE_STD_%j.err
#SBATCH --account=chipilskigroup_q
#SBATCH --partition=chipilskigroup_q
#SBATCH -t 04:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G

echo "--- STD DEV calc starting ---"

module purge
module load gnu/4.8.5
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env
export LD_LIBRARY_PATH=$HOME/speedy_libs/lib:$LD_LIBRARY_PATH

cd ../amlcs/

srun python calc_ensemble_std.py

echo "--- STD DEV calc Finished ---"