#!/bin/bash
#SBATCH --job-name=ERROR_PLOTS_DUAL
#SBATCH --output=ERROR_PLOTS_DUAL_%j.out
#SBATCH --error=ERROR_PLOTS_DUAL_%j.err
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

srun python error_plots_dual.py \
  --ensf_exp  ../runs/t21_50_0.05_5_ReverseSDE_1_1_100 \
  --lenkf_exp  ../runs/t21_50_0.05_5_LEnKF_1_1_100 \
  --resolution t21 \
  --M 5 \
  --plot_dir_name ENSF_vs_LENKF_v3 \
  --anchor step1 \
  --vars TG1,UG1,VG1,TRG1,PSG1 \
  --scale both


echo "--- Plotting Job Finished ---"