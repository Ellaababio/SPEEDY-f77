#!/usr/bin/env python3
"""
Debug script to diagnose why error plots are blank.
"""

from pathlib import Path
import numpy as np
from netCDF4 import Dataset

# Configuration from error_plots_multi_nc.py
EXPERIMENTS = [
    "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/linear_no_norm_results",
    "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/linear_normalization_results",
    "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_no_norm_results",
    "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_normalization_results",
]

FREE_RUN_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_5/free_run"
TRUTH_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_5/snapshots"

# Test parameters
TEST_VAR = "TG1"
TEST_LEVEL = 0
TEST_CYCLE = 0

print("=" * 80)
print("DEBUGGING ERROR PLOT GENERATION")
print("=" * 80)

# 1. Check if directories exist
print("\n1. Checking directories...")
print(f"   FREE_RUN_DIR exists: {Path(FREE_RUN_DIR).exists()}")
print(f"   TRUTH_DIR exists: {Path(TRUTH_DIR).exists()}")
for i, exp in enumerate(EXPERIMENTS):
    print(f"   EXPERIMENTS[{i}] exists: {Path(exp).exists()}")

# 2. Check if test files exist
print(f"\n2. Checking test files for cycle {TEST_CYCLE}...")
truth_file = Path(TRUTH_DIR) / f"reference_solution_{TEST_CYCLE}.nc"
noda_file = Path(FREE_RUN_DIR) / f"free_run_{TEST_CYCLE}.nc"
print(f"   Truth file: {truth_file}")
print(f"   Exists: {truth_file.exists()}")
print(f"   NoDA file: {noda_file}")
print(f"   Exists: {noda_file.exists()}")

# 3. Try to read truth and NoDA fields
print(f"\n3. Reading truth and NoDA for {TEST_VAR} level {TEST_LEVEL}...")
try:
    with Dataset(truth_file, 'r') as nc:
        print(f"   Available variables in truth file: {list(nc.variables.keys())[:10]}...")
        field_name = f"truth_{TEST_VAR}_lev{TEST_LEVEL}"
        if field_name in nc.variables:
            truth_data = nc.variables[field_name][:]
            print(f"   {field_name}: shape={truth_data.shape}, min={truth_data.min():.3f}, max={truth_data.max():.3f}, mean={truth_data.mean():.3f}")
        else:
            print(f"   ERROR: {field_name} not found in truth file")
            # Try without prefix
            alt_name = f"{TEST_VAR}_lev{TEST_LEVEL}"
            if alt_name in nc.variables:
                truth_data = nc.variables[alt_name][:]
                print(f"   Found as {alt_name}: shape={truth_data.shape}, min={truth_data.min():.3f}, max={truth_data.max():.3f}")
            else:
                print(f"   Also tried {alt_name}, not found")
                truth_data = None
except Exception as e:
    print(f"   ERROR reading truth: {e}")
    truth_data = None

try:
    with Dataset(noda_file, 'r') as nc:
        print(f"   Available variables in NoDA file: {list(nc.variables.keys())[:10]}...")
        field_name = f"noda_{TEST_VAR}_lev{TEST_LEVEL}"
        if field_name in nc.variables:
            noda_data = nc.variables[field_name][:]
            print(f"   {field_name}: shape={noda_data.shape}, min={noda_data.min():.3f}, max={noda_data.max():.3f}, mean={noda_data.mean():.3f}")
        else:
            print(f"   ERROR: {field_name} not found in NoDA file")
            alt_name = f"{TEST_VAR}_lev{TEST_LEVEL}"
            if alt_name in nc.variables:
                noda_data = nc.variables[alt_name][:]
                print(f"   Found as {alt_name}: shape={noda_data.shape}, min={noda_data.min():.3f}, max={noda_data.max():.3f}")
            else:
                print(f"   Also tried {alt_name}, not found")
                noda_data = None
except Exception as e:
    print(f"   ERROR reading NoDA: {e}")
    noda_data = None

# 4. Compute NoDA error
if truth_data is not None and noda_data is not None:
    print("\n4. Computing NoDA error...")
    diff = noda_data - truth_data
    l2_error = np.sqrt(np.mean(diff**2))
    print(f"   L2 Error: {l2_error:.6f}")
else:
    print("\n4. Cannot compute NoDA error (missing data)")

# 5. Check experiment cycle files
print(f"\n5. Checking experiment cycle files for cycle {TEST_CYCLE}...")
for i, exp_path in enumerate(EXPERIMENTS):
    exp_path = Path(exp_path)
    print(f"\n   Experiment {i}: {exp_path.name}")
    
    # Try to find cycle file
    possible_names = [
        f"reverseSDE_cycle{TEST_CYCLE}.nc",
        f"enkf_cycle{TEST_CYCLE}.nc",
        f"cycle{TEST_CYCLE}.nc"
    ]
    
    cycle_file = None
    for name in possible_names:
        if (exp_path / name).exists():
            cycle_file = exp_path / name
            print(f"   Found cycle file: {name}")
            break
    
    if not cycle_file:
        print(f"   ERROR: No cycle file found")
        continue
    
    # Read analysis field
    try:
        with Dataset(cycle_file, 'r') as nc:
            field_name = f"xa_mean_{TEST_VAR}_lev{TEST_LEVEL}"
            if field_name in nc.variables:
                xa_data = nc.variables[field_name][:]
                print(f"   {field_name}: shape={xa_data.shape}, min={xa_data.min():.3f}, max={xa_data.max():.3f}, mean={xa_data.mean():.3f}")
                
                # Compute error if we have truth
                if truth_data is not None:
                    diff = xa_data - truth_data
                    l2_error = np.sqrt(np.mean(diff**2))
                    print(f"   L2 Error vs truth: {l2_error:.6f}")
            else:
                print(f"   ERROR: {field_name} not found in cycle file")
                print(f"   Available variables: {list(nc.variables.keys())[:10]}...")
    except Exception as e:
        print(f"   ERROR reading cycle file: {e}")

# 6. Test the actual functions from the script
print("\n" + "=" * 80)
print("6. Testing actual script functions...")
print("=" * 80)

def _read_nc_field(nc_path: Path, var: str, lev: int) -> np.ndarray:
    """
    Read a specific field from a NetCDF file.
    
    Handles three formats:
    1. Per-level variables: xa_mean_TG1_lev0, truth_TG1_lev0, etc. (cycle files)
    2. Multi-dimensional arrays: TG1[lev, lat, lon] (truth/NoDA files)
    3. Tracer variables: TRG1[tracer, lev, lat, lon] (truth/NoDA files for humidity)
    """
    with Dataset(nc_path, 'r') as nc:
        # First, try per-level variable names (cycle files)
        for prefix in ["xa_mean", "xb_mean", "truth", "noda"]:
            field_name = f"{prefix}_{var}_lev{lev}"
            if field_name in nc.variables:
                return nc.variables[field_name][:]
        
        # Second, try multi-dimensional arrays (truth/NoDA files)
        # Just the variable name without prefix or level suffix
        if var in nc.variables:
            var_data = nc.variables[var]
            ndims = len(var_data.shape)
            
            if ndims == 4:  # (tracer, level, lat, lon) - for TRG variables
                # Take first tracer (index 0) and the specified level
                return var_data[0, lev, :, :]
            elif ndims == 3:  # (level, lat, lon)
                return var_data[lev, :, :]
            elif ndims == 2:  # (lat, lon) - for single-level vars like PSG
                if lev == 0:
                    return var_data[:, :]
        
        raise KeyError(f"Field {var}_lev{lev} not found in {nc_path}")

def _compute_l2_error(field1: np.ndarray, field2: np.ndarray) -> float:
    """Compute L2 error between two fields."""
    diff = field1 - field2
    return np.sqrt(np.mean(diff**2))

# Test reading with actual function
print(f"\nTesting _read_nc_field on truth file...")
try:
    truth = _read_nc_field(truth_file, TEST_VAR, TEST_LEVEL)
    print(f"SUCCESS: Read truth, shape={truth.shape}, mean={truth.mean():.3f}")
except Exception as e:
    print(f"ERROR: {e}")

print(f"\nTesting _read_nc_field on NoDA file...")
try:
    noda = _read_nc_field(noda_file, TEST_VAR, TEST_LEVEL)
    print(f"SUCCESS: Read NoDA, shape={noda.shape}, mean={noda.mean():.3f}")
except Exception as e:
    print(f"ERROR: {e}")

print(f"\nTesting _read_nc_field on cycle file...")
cycle_file = Path(EXPERIMENTS[0]) / f"reverseSDE_cycle{TEST_CYCLE}.nc"
try:
    xa = _read_nc_field(cycle_file, TEST_VAR, TEST_LEVEL)
    print(f"SUCCESS: Read xa_mean, shape={xa.shape}, mean={xa.mean():.3f}")
except Exception as e:
    print(f"ERROR: {e}")

# 7. Test full error series computation for one exp
print("\n" + "=" * 80)
print("7. Investigating TRG (humidity) variable structure...")
print("=" * 80)

TRG_VAR = "TRG1"
print(f"\nChecking {TRG_VAR} structure in different files...")

# Check truth file
truth_file_0 = Path(TRUTH_DIR) / f"reference_solution_0.nc"
with Dataset(truth_file_0, 'r') as nc:
    if TRG_VAR in nc.variables:
        trg = nc.variables[TRG_VAR]
        print(f"   Truth file: {TRG_VAR} shape = {trg.shape}, dimensions = {trg.dimensions}")
    else:
        print(f"   Truth file: {TRG_VAR} not found")

# Check NoDA file
noda_file_0 = Path(FREE_RUN_DIR) / f"free_run_0.nc"
with Dataset(noda_file_0, 'r') as nc:
    if TRG_VAR in nc.variables:
        trg = nc.variables[TRG_VAR]
        print(f"   NoDA file: {TRG_VAR} shape = {trg.shape}, dimensions = {trg.dimensions}")
    else:
        print(f"   NoDA file: {TRG_VAR} not found")

# Check cycle file
cycle_file_0 = Path(EXPERIMENTS[0]) / f"reverseSDE_cycle0.nc"
with Dataset(cycle_file_0, 'r') as nc:
    # Check for xa_mean_TRG1_lev2 (TRG starts at level 2)
    test_var = "xa_mean_TRG1_lev2"
    if test_var in nc.variables:
        trg = nc.variables[test_var]
        print(f"   Cycle file: {test_var} shape = {trg.shape}, dimensions = {trg.dimensions}")
    else:
        print(f"   Cycle file: {test_var} not found")

print("\n" + "=" * 80)
print("8. Testing full error series computation...")
print("=" * 80)

CYCLES = list(range(5))

print(f"\nComputing NoDA series for {TEST_VAR} level {TEST_LEVEL}...")
errors_noda = []
for cycle_k in CYCLES:
    try:
        truth_file_k = Path(TRUTH_DIR) / f"reference_solution_{cycle_k}.nc"
        noda_file_k = Path(FREE_RUN_DIR) / f"free_run_{cycle_k}.nc"
        
        truth = _read_nc_field(truth_file_k, TEST_VAR, TEST_LEVEL)
        noda = _read_nc_field(noda_file_k, TEST_VAR, TEST_LEVEL)
        
        error = _compute_l2_error(noda, truth)
        errors_noda.append(error)
        print(f"   Cycle {cycle_k}: {error:.6f}")
    except Exception as e:
        print(f"   Cycle {cycle_k}: ERROR - {e}")
        errors_noda.append(np.nan)

print(f"\nNoDA error series: {errors_noda}")
print(f"Any NaNs? {np.any(np.isnan(errors_noda))}")

print(f"\nComputing analysis series for experiment 0...")
exp_path = Path(EXPERIMENTS[0])
errors_xa = []
for cycle_k in CYCLES:
    try:
        truth_file_k = Path(TRUTH_DIR) / f"reference_solution_{cycle_k}.nc"
        cycle_file_k = exp_path / f"reverseSDE_cycle{cycle_k}.nc"
        
        truth = _read_nc_field(truth_file_k, TEST_VAR, TEST_LEVEL)
        xa = _read_nc_field(cycle_file_k, TEST_VAR, TEST_LEVEL)
        
        error = _compute_l2_error(xa, truth)
        errors_xa.append(error)
        print(f"   Cycle {cycle_k}: {error:.6f}")
    except Exception as e:
        print(f"   Cycle {cycle_k}: ERROR - {e}")
        errors_xa.append(np.nan)

print(f"\nAnalysis error series: {errors_xa}")
print(f"Any NaNs? {np.any(np.isnan(errors_xa))}")

print("\n" + "=" * 80)
print("9. Testing ALL variables with level-averaged errors...")
print("=" * 80)

ALL_VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]

def _levels_for_var(var):
    if "PSG" in var:
        return [0]
    if var.startswith("TRG"):
        return list(range(2, 8))
    return list(range(8))

for var in ALL_VARS:
    print(f"\n{'='*60}")
    print(f"Variable: {var}")
    print(f"{'='*60}")
    
    lvls = _levels_for_var(var)
    print(f"Levels: {lvls}")
    
    # Compute errors for first experiment across all levels and cycles
    exp_path = Path(EXPERIMENTS[0])
    
    # Store errors: [level][cycle]
    errors_by_level = []
    
    for lev in lvls:
        errors_this_level = []
        for cycle_k in CYCLES:
            try:
                truth_file_k = Path(TRUTH_DIR) / f"reference_solution_{cycle_k}.nc"
                cycle_file_k = exp_path / f"reverseSDE_cycle{cycle_k}.nc"
                
                truth = _read_nc_field(truth_file_k, var, lev)
                xa = _read_nc_field(cycle_file_k, var, lev)
                
                error = _compute_l2_error(xa, truth)
                errors_this_level.append(error)
            except Exception as e:
                print(f"   ERROR at level {lev}, cycle {cycle_k}: {e}")
                errors_this_level.append(np.nan)
        
        errors_by_level.append(errors_this_level)
        print(f"   Level {lev}: {[f'{e:.3f}' if not np.isnan(e) else 'NaN' for e in errors_this_level]}")
    
    # Compute level-averaged errors
    print(f"\n   Level-Averaged Errors:")
    if len(errors_by_level) > 0:
        # Stack into array: [level, cycle]
        errors_array = np.array(errors_by_level)
        level_avg = np.nanmean(errors_array, axis=0)
        
        for cycle_k in range(len(CYCLES)):
            print(f"      Cycle {cycle_k}: {level_avg[cycle_k]:.6f}")
    else:
        print("      No data computed")

print("\n" + "=" * 80)
print("DEBUGGING COMPLETE")
print("=" * 80)
