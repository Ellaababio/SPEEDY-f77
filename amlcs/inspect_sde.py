#!/usr/bin/env python3
import os
from pathlib import Path
from netCDF4 import Dataset

SDE_FILE = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data/sde_tracking.nc"

def main():
    if not os.path.exists(SDE_FILE):
        print(f"File not found: {SDE_FILE}")
        return

    print(f"Inspecting File: {SDE_FILE}")
    with Dataset(SDE_FILE, 'r') as nc:
        print("\n--- Dimensions ---")
        for dim_name, dim in nc.dimensions.items():
            print(f"  {dim_name}: size = {dim.size}")
            
        print("\n--- Variables ---")
        for var_name, var in nc.variables.items():
            print(f"  {var_name}: shape={var.shape}, dtype={var.dtype}")
            if "units" in var.ncattrs():
                print(f"      units: {var.units}")

if __name__ == "__main__":
    main()
