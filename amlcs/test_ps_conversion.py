import os
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

NC_FILE = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/free_run/free_run_1.nc"

def main():
    if not os.path.exists(NC_FILE):
        print(f"File not found: {NC_FILE}")
        return

    print(f"Reading file: {NC_FILE}")
    with Dataset(NC_FILE, 'r') as nc:
        # Check if PSG1 is in variables
        if "PSG1" not in nc.variables:
            print("PSG1 not found in variables. Available variables:")
            print(list(nc.variables.keys()))
            return
        
        ps_log = nc.variables["PSG1"][:]
        # Some dimensions might be (time, lat, lon) or (lat, lon)
        print(f"PSG1 shape: {ps_log.shape}")
        
    # Calculate statistics on log(ps/p0)
    print(f"\n--- Raw PSG1 (log(ps/p0)) ---")
    print(f"Min: {np.min(ps_log):.5f}")
    print(f"Max: {np.max(ps_log):.5f}")
    print(f"Mean: {np.mean(ps_log):.5f}")
    
    # Convert to Pa and hPa
    # Assuming standard p0 is 100,000 Pa (1,000 hPa)
    p0_pa = 100000.0
    p0_hpa = 1000.0
    
    ps_pa = p0_pa * np.exp(ps_log)
    ps_hpa = p0_hpa * np.exp(ps_log)
    
    print(f"\n--- Converted Pressure (Pa) ---")
    print(f"Min: {np.min(ps_pa):.2f} Pa")
    print(f"Max: {np.max(ps_pa):.2f} Pa")
    print(f"Mean: {np.mean(ps_pa):.2f} Pa")
    
    print(f"\n--- Converted Pressure (hPa) ---")
    print(f"Min: {np.min(ps_hpa):.2f} hPa")
    print(f"Max: {np.max(ps_hpa):.2f} hPa")
    print(f"Mean: {np.mean(ps_hpa):.2f} hPa")

    # Check if this makes sense (Earth surface pressure is typically 980-1050 hPa)
    is_valid = (np.min(ps_hpa) > 400.0) and (np.max(ps_hpa) < 1100.0)
    if is_valid:
        print("\n=> SUCCESS: Converted values are within reasonable earthly bounds (400-1100 hPa).")
    else:
        print("\n=> WARNING: Values look suspicious. Maybe p0 is not 1e5 Pa or the formula needs adjustment.")

if __name__ == "__main__":
    main()
