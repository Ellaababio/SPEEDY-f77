#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

ENSF_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data"
REF_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20"
P0_HPA = 1000.0

def main():
    ensf_file = Path(ENSF_DIR) / "reverseSDE_cycle0.nc"
    truth_file = Path(REF_DIR) / "snapshots/reference_solution_0.nc"
    noda_file = Path(REF_DIR) / "free_run/free_run_0.nc"
    
    with Dataset(truth_file, 'r') as nc:
        t_ps_log = nc.variables["PSG1"][:]
    with Dataset(noda_file, 'r') as nc:
        n_ps_log = nc.variables["PSG1"][:]
    with Dataset(ensf_file, 'r') as nc:
        xa_ps_log = nc.variables["xa_mean_PSG1_lev0"][:]
        xb_ps_log = nc.variables["xb_mean_PSG1_lev0"][:]
        
    t_ps = P0_HPA * np.exp(t_ps_log)
    n_ps = P0_HPA * np.exp(n_ps_log)
    xa_ps = P0_HPA * np.exp(xa_ps_log)
    xb_ps = P0_HPA * np.exp(xb_ps_log)
    
    print(f"Cycle 0 PSG1 (hPa) RMSE:")
    print(f"  NoDA vs Truth: {np.sqrt(np.mean((n_ps - t_ps)**2)):.3f}")
    print(f"  EnSF Background (xb) vs Truth: {np.sqrt(np.mean((xb_ps - t_ps)**2)):.3f}")
    print(f"  EnSF Analysis (xa) vs Truth:   {np.sqrt(np.mean((xa_ps - t_ps)**2)):.3f}")

if __name__ == "__main__":
    main()
