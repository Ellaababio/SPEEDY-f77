from netCDF4 import Dataset
import os

files = [
    "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/free_run/free_run_1.nc",
    "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/snapshots/reference_solution_1.nc"
]

for f in files:
    print(f"\nScanning: {f}")
    if not os.path.exists(f):
        print("  File not found!")
        continue
    try:
        with Dataset(f, 'r') as nc:
            print("  Dimensions:", list(nc.dimensions.keys()))
            print("  Variables:")
            for v in nc.variables:
                print(f"    {v}: {nc.variables[v].shape}")
    except Exception as e:
        print(f"  Error: {e}")
