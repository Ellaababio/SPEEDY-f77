#!/bin/bash
export LD_LIBRARY_PATH="/gpfs/home/ea25e/speedy_libs/lib:/gpfs/home/ea25e/.conda/envs/speedy_da_env/lib:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="/gpfs/home/ea25e/.conda/envs/speedy_da_env/lib:${LD_LIBRARY_PATH}"

# Absolute logs directory
LOG_DIR="/gpfs/home/ea25e/SPEEDY-f77/logs"

# First argument is the script name
script="$1"
shift  # Now $@ contains only the script's parameters

timestamp=$(date +%F_%H-%M-%S)

# Optional tag from first script argument (e.g. letkf_r4 from configs/letkf_r4.csv)
tag=""
if [[ $# -gt 0 ]]; then
    tag="_$(basename "${1%.*}")"
fi

# SLURM_JOB_ID is unique per job; fall back to PID for interactive runs
id="${SLURM_JOB_ID:-$$}"

log="$LOG_DIR/${script%.py}${tag}_${timestamp}_${id}.out"

# Print minimal info to console only
echo "=== Running: $script"
echo "=== Params: $@"
echo "=== Log file: $log"
echo "=========================================="

# ALSO write metadata to the log file
{
    echo "=== Running: $script"
    echo "=== Params: $@"
    echo "=== Log file: $log"
    echo "=== Timestamp: $timestamp"
    echo "=========================================="
} >> "$log"

# Python output goes ONLY to log file
PYTHONUNBUFFERED=1 /gpfs/home/ea25e/.conda/envs/speedy_da_env/bin/python "$script" "$@" &> "$log"
