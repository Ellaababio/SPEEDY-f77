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

echo "--- Validation+Analysis Job Started ---"

module purge
module load anaconda/3.11.5
eval "$(conda shell.bash hook)"
conda activate speedy_da_env

# >>> adjust RUN_ROOT to your experiment directory <<<
RUN_ROOT="../runs/t21_50_0.05_5_ReverseSDE_1_5_100"
GAUSS_ROOT="${RUN_ROOT}/gauss_checks"
BLOCK_MAP="${RUN_ROOT}/block_map.json"   # dumped during assimilation by perform_assimilation

cd ../amlcs

# 1) Validate each saved background block
for f in "${GAUSS_ROOT}"/XB_block_*.npy; do
  bname=$(basename "$f" .npy)
  outdir="${GAUSS_ROOT}/gauss_${bname}"
  echo "Validating $f -> ${outdir}"
  srun python gaussian_validation.py --npy "$f" --outdir "$outdir" --psteps 200 --eps_alpha 0.05
done

# 2) Aggregate results across blocks (attach var/level mapping if available)
OUT_COMBINED="${RUN_ROOT}/gauss_overview.csv"
MAP_ARG=""
if [ -f "${BLOCK_MAP}" ]; then
  MAP_ARG="--map ${BLOCK_MAP}"
  echo "Found block map: ${BLOCK_MAP}"
else
  echo "Block map not found at ${BLOCK_MAP}; proceeding without vars/levels."
fi

echo "Aggregating results to ${OUT_COMBINED}"
srun python analyze_gaussian_results.py --root "$GAUSS_ROOT" --out "$OUT_COMBINED" --ks95_warn 0.1 --print_top 15 ${MAP_ARG}

echo "--- Validation+Analysis Job Finished ---"
