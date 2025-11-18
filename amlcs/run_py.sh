#!/bin/bash

# Absolute logs directory
LOG_DIR="/gpfs/home/jjs21b/AMLCS/logs"
mkdir -p "$LOG_DIR"

# First argument is the script name
script="$1"
shift  # Now $@ contains only the script's parameters

timestamp=$(date +%F_%H-%M-%S)
log="$LOG_DIR/${script%.py}_${timestamp}.out"

# Print minimal info to console only
echo "=== Running: $script"
echo "=== Params: $@"
echo "=== Log file: $log"
echo "=========================================="

# Python output goes ONLY to log file
python "$script" "$@" &> "$log"
