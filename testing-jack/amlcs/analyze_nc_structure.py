#!/usr/bin/env python3
import netCDF4 as nc

def analyze_structure(file_path):
    print(f"Analyzing file: {file_path}")
    try:
        dataset = nc.Dataset(file_path, 'r')
        
        print("\n--- Dimensions ---")
        for dim_name, dim in dataset.dimensions.items():
            print(f"{dim_name}: size = {dim.size}")
            
        print("\n--- Variables ---")
        for var_name, var in dataset.variables.items():
            print(f"{var_name}: shape = {var.shape}, dimensions = {var.dimensions}, dtype = {var.dtype}")
            
        dataset.close()
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    file_path = '/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_LETKF_4_1_100/all_arctan/5x_obs_err/data/unified_cycle0.nc'
    analyze_structure(file_path)
