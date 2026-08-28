#!/usr/bin/env python3
"""
Compare NetCDF files from two ReverseSDE runs to verify PyTorch refactor
didn't change results.

Usage:
    1. Update OLD_DIR and NEW_DIR paths below
    2. Run: python verify_pytorch_results.py
"""

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
OLD_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100"
NEW_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_nc_files_numpy_01_obs"  # UPDATE THIS

# Comparison tolerances
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-8  # Absolute tolerance
# ============================================================================

import os
import sys
import numpy as np
from netCDF4 import Dataset
from pathlib import Path


def compare_netcdf_files(file1_path, file2_path, rtol=1e-5, atol=1e-8):
    """
    Compare two NetCDF files and return detailed comparison results.
    
    Args:
        file1_path: Path to first NetCDF file
        file2_path: Path to second NetCDF file
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        
    Returns:
        dict: Comparison results with 'match', 'diffs', and 'stats' keys
    """
    if not os.path.exists(file2_path):
        return {
            'match': False,
            'error': f"File missing in new run: {os.path.basename(file2_path)}"
        }
    
    results = {
        'match': True,
        'diffs': [],
        'stats': {}
    }
    
    try:
        with Dataset(file1_path, 'r') as nc1, Dataset(file2_path, 'r') as nc2:
            # Get all variables
            vars1 = set(nc1.variables.keys())
            vars2 = set(nc2.variables.keys())
            
            # Check for missing variables
            missing_in_new = vars1 - vars2
            extra_in_new = vars2 - vars1
            
            if missing_in_new:
                results['match'] = False
                results['diffs'].append(f"Variables missing in new file: {missing_in_new}")
            
            if extra_in_new:
                results['diffs'].append(f"Extra variables in new file: {extra_in_new}")
            
            # Compare common variables
            common_vars = vars1 & vars2
            
            for var_name in common_vars:
                var1 = nc1.variables[var_name]
                var2 = nc2.variables[var_name]
                
                # Get data arrays
                data1 = var1[:]
                data2 = var2[:]
                
                # Check shapes match
                if data1.shape != data2.shape:
                    results['match'] = False
                    results['diffs'].append(
                        f"{var_name}: Shape mismatch - old: {data1.shape}, new: {data2.shape}"
                    )
                    continue
                
                # Handle string/char variables
                if data1.dtype.kind in ('S', 'U', 'O') or data2.dtype.kind in ('S', 'U', 'O'):
                    if not np.array_equal(data1, data2):
                        results['match'] = False
                        results['diffs'].append(f"{var_name}: String/Object values differ")
                    continue

                # Handle NaN values for numeric types
                mask1 = np.isnan(data1)
                mask2 = np.isnan(data2)
                
                if not np.array_equal(mask1, mask2):
                    results['match'] = False
                    nan_diff = np.sum(mask1) - np.sum(mask2)
                    results['diffs'].append(
                        f"{var_name}: NaN pattern differs (old: {np.sum(mask1)}, new: {np.sum(mask2)}, diff: {nan_diff})"
                    )
                
                # Compare non-NaN values
                valid_mask = ~mask1 & ~mask2
                if np.any(valid_mask):
                    data1_valid = data1[valid_mask]
                    data2_valid = data2[valid_mask]
                    
                    # Check if values are close
                    close = np.allclose(data1_valid, data2_valid, rtol=rtol, atol=atol, equal_nan=True)
                    
                    if not close:
                        results['match'] = False
                        
                        # Compute statistics
                        abs_diff = np.abs(data1_valid - data2_valid)
                        rel_diff = abs_diff / (np.abs(data1_valid) + 1e-10)
                        
                        max_abs_diff = np.max(abs_diff)
                        max_rel_diff = np.max(rel_diff)
                        mean_abs_diff = np.mean(abs_diff)
                        
                        results['diffs'].append(
                            f"{var_name}: Values differ - "
                            f"max_abs={max_abs_diff:.3e}, max_rel={max_rel_diff:.3e}, "
                            f"mean_abs={mean_abs_diff:.3e}"
                        )
                        
                        results['stats'][var_name] = {
                            'max_abs_diff': float(max_abs_diff),
                            'max_rel_diff': float(max_rel_diff),
                            'mean_abs_diff': float(mean_abs_diff),
                            'num_values': int(np.sum(valid_mask))
                        }
    
    except Exception as e:
        results['match'] = False
        results['error'] = f"Error comparing files: {str(e)}"
    
    return results


def compare_directories(old_dir, new_dir, rtol=1e-5, atol=1e-8, verbose=True):
    """
    Compare all NetCDF files in two directories.
    
    Args:
        old_dir: Path to directory with old (NumPy) results
        new_dir: Path to directory with new (PyTorch) results
        rtol: Relative tolerance for numerical comparison
        atol: Absolute tolerance for numerical comparison
        verbose: Print detailed comparison results
        
    Returns:
        dict: Summary of comparison results
    """
    old_path = Path(old_dir)
    new_path = Path(new_dir)
    
    if not old_path.exists():
        print(f"❌ Old directory does not exist: {old_dir}")
        return None
    
    if not new_path.exists():
        print(f"❌ New directory does not exist: {new_dir}")
        return None
    
    # Find all NetCDF files in old directory
    old_files = sorted(old_path.glob("*.nc"))
    
    if not old_files:
        print(f"❌ No NetCDF files found in old directory: {old_dir}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Comparing NetCDF files:")
    print(f"  Old (NumPy):  {old_dir}")
    print(f"  New (PyTorch): {new_dir}")
    print(f"  Tolerance: rtol={rtol}, atol={atol}")
    print(f"{'='*80}\n")
    
    summary = {
        'total_files': len(old_files),
        'matched': 0,
        'mismatched': 0,
        'missing': 0,
        'errors': 0,
        'file_results': {}
    }
    
    for old_file in old_files:
        filename = old_file.name
        new_file = new_path / filename
        
        if verbose:
            print(f"Comparing: {filename}")
        
        results = compare_netcdf_files(str(old_file), str(new_file), rtol=rtol, atol=atol)
        summary['file_results'][filename] = results
        
        if 'error' in results:
            summary['errors'] += 1
            if verbose:
                print(f"  ❌ ERROR: {results['error']}")
        elif not new_file.exists():
            summary['missing'] += 1
            if verbose:
                print(f"  ⚠️  MISSING in new directory")
        elif results['match']:
            summary['matched'] += 1
            if verbose:
                print(f"  ✓ MATCH")
        else:
            summary['mismatched'] += 1
            if verbose:
                print(f"  ✗ MISMATCH:")
                for diff in results['diffs']:
                    print(f"      {diff}")
        
        if verbose:
            print()
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"{'='*80}")
    print(f"  Total files:     {summary['total_files']}")
    print(f"  ✓ Matched:       {summary['matched']}")
    print(f"  ✗ Mismatched:    {summary['mismatched']}")
    print(f"  ⚠️  Missing:       {summary['missing']}")
    print(f"  ❌ Errors:        {summary['errors']}")
    print(f"{'='*80}\n")
    
    if summary['matched'] == summary['total_files']:
        print("✅ SUCCESS: All files match! PyTorch refactor is verified.")
    else:
        print("⚠️  WARNING: Some files differ. Review the differences above.")
    
    return summary


if __name__ == "__main__":
    summary = compare_directories(OLD_DIR, NEW_DIR, rtol=RTOL, atol=ATOL, verbose=True)
    
    # Exit with error code if there were mismatches
    if summary and summary['mismatched'] > 0:
        sys.exit(1)
