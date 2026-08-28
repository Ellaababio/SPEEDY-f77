#!/usr/bin/env python3
"""
NetCDF Comparison Script
========================
Compares two NetCDF files and reports differences in:
- Global Attributes
- Dimensions
- Variables (Shapes, Attributes, Values)

Usage:
    Edit FILE1 and FILE2 variables below and run:
    python3 compare_netcdf.py
"""

import sys
import numpy as np
from netCDF4 import Dataset
import os

# ==============================================================================
# CONFIGURATION: Set the files to compare here
# ==============================================================================

FILE1 = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/unified_cycle1.nc"
FILE2 = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data_ps0001/reverseSDE_cycle1.nc"

# ==============================================================================

def compare_attributes(nc1, nc2, obj_name="Global"):
    """Compare attributes of two NetCDF objects (dataset or variable)."""
    attrs1 = set(nc1.ncattrs())
    attrs2 = set(nc2.ncattrs())
    
    unique1 = attrs1 - attrs2
    unique2 = attrs2 - attrs1
    common = attrs1 & attrs2
    
    diffs = []
    
    if unique1:
        diffs.append(f"{obj_name} attributes only in FILE1: {sorted(list(unique1))}")
    if unique2:
        diffs.append(f"{obj_name} attributes only in FILE2: {sorted(list(unique2))}")
        
    for attr in common:
        val1 = nc1.getncattr(attr)
        val2 = nc2.getncattr(attr)
        
        # Handle numpy arrays in attributes
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                 diffs.append(f"{obj_name} attribute '{attr}' differs:\n  FILE1: {val1}\n  FILE2: {val2}")
        elif val1 != val2:
             diffs.append(f"{obj_name} attribute '{attr}' differs: '{val1}' vs '{val2}'")
             
    return diffs

def compare_files(path1, path2):
    print("=" * 80)
    print(f"Comparing:")
    print(f"  FILE1: {path1}")
    print(f"  FILE2: {path2}")
    print("=" * 80)
    
    if not os.path.exists(path1):
        print(f"ERROR: FILE1 not found: {path1}")
        return
    if not os.path.exists(path2):
        print(f"ERROR: FILE2 not found: {path2}")
        return

    try:
        with Dataset(path1, "r") as nc1, Dataset(path2, "r") as nc2:
            
            # 1. Global Attributes
            attr_diffs = compare_attributes(nc1, nc2, "Global")
            for diff in attr_diffs:
                print(f"[!] {diff}")
            
            # 2. Dimensions
            dims1 = set(nc1.dimensions.keys())
            dims2 = set(nc2.dimensions.keys())
            
            unique_dims1 = dims1 - dims2
            unique_dims2 = dims2 - dims1
            common_dims = dims1 & dims2
            
            if unique_dims1: print(f"[-] Dimensions only in FILE1: {sorted(list(unique_dims1))}")
            if unique_dims2: print(f"[-] Dimensions only in FILE2: {sorted(list(unique_dims2))}")
                
            for dim in common_dims:
                len1 = len(nc1.dimensions[dim])
                len2 = len(nc2.dimensions[dim])
                if len1 != len2:
                    print(f"[!] Dimension '{dim}' size mismatch: {len1} vs {len2}")

            # 3. Variables
            vars1 = set(nc1.variables.keys())
            vars2 = set(nc2.variables.keys())
            
            unique_vars1 = vars1 - vars2
            unique_vars2 = vars2 - vars1
            common_vars = sorted(list(vars1 & vars2))
            
            if unique_vars1: 
                print(f"[-] Unique variables in FILE1 ({len(unique_vars1)}): {sorted(list(unique_vars1))}")
            if unique_vars2: 
                print(f"[-] Unique variables in FILE2 ({len(unique_vars2)}): {sorted(list(unique_vars2))}")

            for var in common_vars:
                v1 = nc1.variables[var]
                v2 = nc2.variables[var]
                
                # Compare shape
                if v1.shape != v2.shape:
                    print(f"[!] Variable '{var}' shape mismatch: {v1.shape} vs {v2.shape}")
                    continue # Cannot compare values if shapes differ
                
                # Compare attributes
                attr_diffs = compare_attributes(v1, v2, f"Variable '{var}'")
                for diff in attr_diffs:
                    print(f"[!] {diff}")
                
                # Compare values
                try:
                    data1 = v1[:]
                    data2 = v2[:]
                    
                    if np.issubdtype(data1.dtype, np.number) and np.issubdtype(data2.dtype, np.number):
                        # Numeric comparison
                        
                        # Check for NaNs
                        nan_mask1 = np.isnan(data1)
                        nan_mask2 = np.isnan(data2)
                        
                        if not np.array_equal(nan_mask1, nan_mask2):
                            count1 = np.sum(nan_mask1)
                            count2 = np.sum(nan_mask2)
                            print(f"[!] Variable '{var}' NaN mismatch: {count1} vs {count2} NaNs")
                        else:
                            # Compare non-NaN values
                            valid_mask = ~nan_mask1
                            if np.any(valid_mask):
                                d1_valid = data1[valid_mask]
                                d2_valid = data2[valid_mask]
                                
                                if not np.allclose(d1_valid, d2_valid, rtol=1e-5, atol=1e-8):
                                    diff = np.abs(d1_valid - d2_valid)
                                    # Count differences above threshold to be specific
                                    threshold = 1e-8 + 1e-5 * np.abs(d2_valid)
                                    diff_mask = diff > threshold
                                    num_diffs = np.sum(diff_mask)
                                    
                                    max_diff = np.max(diff)
                                    avg_diff = np.mean(diff) # Average of all differences (or just differing ones?) usually overall error is interesting
                                    avg_diff_mismatch = np.mean(diff[diff_mask]) if num_diffs > 0 else 0
                                    
                                    print(f"[!] Variable '{var}' differs:")
                                    print(f"    - Count: {num_diffs} values differ")
                                    print(f"    - Max Diff: {max_diff:.3e}")
                                    print(f"    - Avg Diff (all): {avg_diff:.3e}")
                                    print(f"    - Avg Diff (mismatch): {avg_diff_mismatch:.3e}")
                                else:
                                    print(f"[OK] Variable '{var}' values match.")
                            else:
                                print(f"[OK] Variable '{var}' (all NaNs) match.")
                    else:
                        # Non-numeric (e.g. char attributes), use simple equality
                        # For masked arrays, fill with a value to compare
                        if np.ma.is_masked(data1): data1 = data1.filled(fill_value=0)
                        if np.ma.is_masked(data2): data2 = data2.filled(fill_value=0)

                        if not np.array_equal(data1, data2):
                            print(f"[!] Variable '{var}' content differs (non-numeric)")
                        else:
                            print(f"[OK] Variable '{var}' matches.")
                             
                except Exception as e:
                    print(f"[Error] comparing values for '{var}': {e}")


            print("\nComparison Complete.")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_files(FILE1, FILE2)
