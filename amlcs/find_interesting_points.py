#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from netCDF4 import Dataset

ENSF_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data"
REF_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20"
P0_HPA = 1000.0

def main():
    ensf_file = Path(ENSF_DIR) / "reverseSDE_cycle0.nc"
    truth_file = Path(REF_DIR) / "snapshots/reference_solution_0.nc"
    
    if not (ensf_file.exists() and truth_file.exists()):
        print("Required files not found.")
        return

    # Extract PSG1
    with Dataset(truth_file, 'r') as nc:
        t_ps_log = nc.variables["PSG1"][:]
    with Dataset(ensf_file, 'r') as nc:
        xa_ps_log = nc.variables["xa_mean_PSG1_lev0"][:]
        xb_ps_log = nc.variables["xb_mean_PSG1_lev0"][:]
        lat = nc.variables["lat"][:] if "lat" in nc.variables else np.linspace(90, -90, xb_ps_log.shape[0])
        lon = nc.variables["lon"][:] if "lon" in nc.variables else np.linspace(0, 360, xb_ps_log.shape[1], endpoint=False)
        
    t_ps = P0_HPA * np.exp(t_ps_log)
    xa_ps = P0_HPA * np.exp(xa_ps_log)
    xb_ps = P0_HPA * np.exp(xb_ps_log)
    
    # Calculate metrics
    initial_error = np.abs(xb_ps - t_ps)  # Background error
    increment = np.abs(xa_ps - xb_ps)     # Analysis increment (correction)
    final_error = np.abs(xa_ps - t_ps)    # Analysis error

    # Find point with largest initial error
    flat_idx_err = np.nanargmax(initial_error)
    lat_idx_err, lon_idx_err = np.unravel_index(flat_idx_err, initial_error.shape)
    
    # Find point with largest increment (where EnSF made the biggest jump)
    flat_idx_inc = np.nanargmax(increment)
    lat_idx_inc, lon_idx_inc = np.unravel_index(flat_idx_inc, increment.shape)

    print("=== Interesting Point 1: Largest Initial Background Error ===")
    print(f"Indices: lat={lat_idx_err}, lon={lon_idx_err}")
    print(f"Coords:  lat={lat[lat_idx_err]:.1f}, lon={lon[lon_idx_err]:.1f}")
    print(f"  Truth:      {t_ps[lat_idx_err, lon_idx_err]:.1f} hPa")
    print(f"  Background: {xb_ps[lat_idx_err, lon_idx_err]:.1f} hPa (Err: {initial_error[lat_idx_err, lon_idx_err]:.1f})")
    print(f"  Analysis:   {xa_ps[lat_idx_err, lon_idx_err]:.1f} hPa (Err: {final_error[lat_idx_err, lon_idx_err]:.1f})")
    print()
    
    print("=== Interesting Point 2: Largest Analysis Increment ===")
    print(f"Indices: lat={lat_idx_inc}, lon={lon_idx_inc}")
    print(f"Coords:  lat={lat[lat_idx_inc]:.1f}, lon={lon[lon_idx_inc]:.1f}")
    print(f"  Truth:      {t_ps[lat_idx_inc, lon_idx_inc]:.1f} hPa")
    print(f"  Background: {xb_ps[lat_idx_inc, lon_idx_inc]:.1f} hPa (Err: {initial_error[lat_idx_inc, lon_idx_inc]:.1f})")
    print(f"  Analysis:   {xa_ps[lat_idx_inc, lon_idx_inc]:.1f} hPa (Err: {final_error[lat_idx_inc, lon_idx_inc]:.1f})")
    print(f"  Increment:  {increment[lat_idx_inc, lon_idx_inc]:.1f} hPa")

if __name__ == "__main__":
    main()
