#!/usr/bin/env python3
"""
Compare all 4 experiments to diagnose why nonlinear_no_norm performs poorly.
"""

from pathlib import Path
import numpy as np
from netCDF4 import Dataset

# Configuration
EXPERIMENTS = [
    ("Linear + No Norm", "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/linear_no_norm_results"),
    ("Linear + Norm", "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/linear_normalization_results"),
    ("Nonlinear + No Norm", "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_no_norm_results"),
    ("Nonlinear + Norm", "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_normalization_results"),
]

TRUTH_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_5/snapshots"
FREE_RUN_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_5/free_run"

TEST_VAR = "TG1"
TEST_LEVEL = 0
CYCLES = list(range(5))

def _read_nc_field(nc_path: Path, var: str, lev: int) -> np.ndarray:
    """Read a specific field from a NetCDF file."""
    with Dataset(nc_path, 'r') as nc:
        # First, try per-level variable names (cycle files)
        for prefix in ["xa_mean", "xb_mean", "truth", "noda"]:
            field_name = f"{prefix}_{var}_lev{lev}"
            if field_name in nc.variables:
                return nc.variables[field_name][:]
        
        # Second, try multi-dimensional arrays (truth/NoDA files)
        if var in nc.variables:
            var_data = nc.variables[var]
            ndims = len(var_data.shape)
            
            if ndims == 4:  # (tracer, level, lat, lon)
                return var_data[0, lev, :, :]
            elif ndims == 3:  # (level, lat, lon)
                return var_data[lev, :, :]
            elif ndims == 2:  # (lat, lon)
                if lev == 0:
                    return var_data[:, :]
        
        raise KeyError(f"Field {var}_lev{lev} not found in {nc_path}")

def _compute_l2_error(field1: np.ndarray, field2: np.ndarray) -> float:
    """Compute L2 error between two fields."""
    diff = field1 - field2
    return np.sqrt(np.mean(diff**2))

print("=" * 80)
print(f"COMPARING ALL EXPERIMENTS FOR {TEST_VAR} LEVEL {TEST_LEVEL}")
print("=" * 80)

# Read truth for cycle 0
truth_file = Path(TRUTH_DIR) / "reference_solution_0.nc"
truth = _read_nc_field(truth_file, TEST_VAR, TEST_LEVEL)
print(f"\nTruth field (cycle 0):")
print(f"  Shape: {truth.shape}")
print(f"  Min: {truth.min():.3f}, Max: {truth.max():.3f}, Mean: {truth.mean():.3f}, Std: {truth.std():.3f}")

# Compare all experiments at cycle 0
print("\n" + "=" * 80)
print("FIELD STATISTICS AT CYCLE 0:")
print("=" * 80)

for exp_name, exp_path in EXPERIMENTS:
    print(f"\n{exp_name}:")
    cycle_file = Path(exp_path) / "reverseSDE_cycle0.nc"
    
    try:
        xa = _read_nc_field(cycle_file, TEST_VAR, TEST_LEVEL)
        error = _compute_l2_error(xa, truth)
        
        print(f"  xa_mean field:")
        print(f"    Min: {xa.min():.3f}, Max: {xa.max():.3f}, Mean: {xa.mean():.3f}, Std: {xa.std():.3f}")
        print(f"  L2 Error vs Truth: {error:.6f}")
        
        # Check if field looks normalized
        if abs(xa.mean()) < 10:
            print(f"  ⚠️  WARNING: Field appears to be normalized (mean={xa.mean():.3f})")
        
        # Check a few sample values
        print(f"  Sample values (first 3x3):")
        print(f"    {xa[:3, :3]}")
        
    except Exception as e:
        print(f"  ERROR: {e}")

# Now compute errors across all cycles
print("\n" + "=" * 80)
print("L2 ERRORS ACROSS ALL CYCLES:")
print("=" * 80)

for exp_name, exp_path in EXPERIMENTS:
    print(f"\n{exp_name}:")
    errors = []
    
    for cycle_k in CYCLES:
        try:
            truth_file_k = Path(TRUTH_DIR) / f"reference_solution_{cycle_k}.nc"
            cycle_file_k = Path(exp_path) / f"reverseSDE_cycle{cycle_k}.nc"
            
            truth_k = _read_nc_field(truth_file_k, TEST_VAR, TEST_LEVEL)
            xa_k = _read_nc_field(cycle_file_k, TEST_VAR, TEST_LEVEL)
            
            error = _compute_l2_error(xa_k, truth_k)
            errors.append(error)
        except Exception as e:
            print(f"  Cycle {cycle_k}: ERROR - {e}")
            errors.append(np.nan)
    
    print(f"  Errors: {[f'{e:.3f}' if not np.isnan(e) else 'NaN' for e in errors]}")
    if len(errors) > 0:
        print(f"  Mean Error: {np.nanmean(errors):.3f}")

# Check if normalization metadata exists in the files
print("\n" + "=" * 80)
print("CHECKING FOR NORMALIZATION METADATA:")
print("=" * 80)

for exp_name, exp_path in EXPERIMENTS:
    print(f"\n{exp_name}:")
    cycle_file = Path(exp_path) / "reverseSDE_cycle0.nc"
    
    try:
        with Dataset(cycle_file, 'r') as nc:
            # Check global attributes
            if hasattr(nc, 'normalize') or hasattr(nc, 'normalization'):
                print(f"  Normalization attribute: {nc.normalize if hasattr(nc, 'normalize') else nc.normalization}")
            
            # Check variable attributes
            test_var_name = f"xa_mean_{TEST_VAR}_lev{TEST_LEVEL}"
            if test_var_name in nc.variables:
                var_obj = nc.variables[test_var_name]
                if hasattr(var_obj, 'scale_factor') or hasattr(var_obj, 'add_offset'):
                    print(f"  Variable has scale/offset attributes")
                    if hasattr(var_obj, 'scale_factor'):
                        print(f"    scale_factor: {var_obj.scale_factor}")
                    if hasattr(var_obj, 'add_offset'):
                        print(f"    add_offset: {var_obj.add_offset}")
            
            # List all global attributes
            print(f"  Global attributes: {list(nc.ncattrs())[:10]}")
            
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
