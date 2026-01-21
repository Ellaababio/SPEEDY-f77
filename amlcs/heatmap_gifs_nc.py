#!/usr/bin/env python3
"""
Heatmaps + Time-Series (NetCDF Version)
Generates animated GIFs and spatial mean time-series plots from NetCDF cycle files.
"""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset

# ======================= USER SETTINGS =======================
# Experiment Directory (where cycle files are)
EXP_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_EnKF_MC_obs_1_1_100/linear_results"

# Reference Directory (where 'snapshots' and 'free_run' are)
REFERENCE_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20"

# Output Directory (relative to EXP_DIR by default)
OUT_DIR_NAME = "heatmaps_nc"

# Variables to process
VARS = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]

# Level index to plot (for 3D vars) - PSG1 is 2D so this is ignored for it
LEVEL_IDX = 7  # Surface level for T21 (0 is top, 7 is bottom)

# T21 grid dimensions
NLAT, NLON = 32, 64

# Units for time-series plots
UNITS = {"UG1": "m/s", "VG1": "m/s", "TG1": "K", "TRG1": "g/kg", "PSG1": "log(ps/p0)"}

# =============================================================

def _find_cycle_files(exp_path: Path):
    """Find available cycle files."""
    # Try different patterns
    for pattern in ["reverseSDE_cycle*.nc", "unified_cycle*.nc", "enkf_cycle*.nc", "cycle*.nc"]:
        files = sorted(exp_path.glob(pattern))
        if files:
            return files
    return []

def _extract_cycle_num(filename):
    """Extract cycle number from filename."""
    m = re.search(r'cycle(\d+)', filename.name)
    return int(m.group(1)) if m else -1

def _read_nc_field(nc_path: Path, var: str, lev: int, prefix: str = None) -> np.ndarray:
    """
    Read field from NetCDF.
    If prefix is provided (e.g. 'xa_mean'), only looks for that specific field.
    If prefix is None, tries raw variable name (3D/4D).
    """
    if not nc_path.exists():
        return np.full((NLAT, NLON), np.nan)

    with Dataset(nc_path, 'r') as nc:
        # 1. Specific prefix (for experiment files)
        if prefix:
            target_lev = 0 if "PSG" in var else lev
            field_name = f"{prefix}_{var}_lev{target_lev}"
            if field_name in nc.variables:
                return nc.variables[field_name][:]
            
            # Fallback: maybe it's not split? (unlikely for current schema but possible)
            return np.full((NLAT, NLON), np.nan)
        
        # 2. Raw variable name (for reference/truth files)
        if var in nc.variables:
            data = nc.variables[var]
            if data.ndim == 3:  # (nlev, lat, lon)
                return data[lev if "PSG" not in var else 0, :, :]
            elif data.ndim == 2:  # (lat, lon)
                return data[:]
            elif data.ndim == 4:  # (time, nlev, lat, lon)
                return data[0, lev if "PSG" not in var else 0, :, :]

    return np.full((NLAT, NLON), np.nan)

def main():
    exp_path = Path(EXP_DIR).resolve()
    ref_path = Path(REFERENCE_DIR).resolve()
    out_dir = exp_path / OUT_DIR_NAME
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Processing Experiment: {exp_path}")
    print(f"Reference Directory: {ref_path}")
    print(f"Output Directory: {out_dir}")

    # Discover cycles
    files = _find_cycle_files(exp_path)
    if not files:
        print("No cycle files found!")
        return
        
    # Sort files by cycle number
    files.sort(key=_extract_cycle_num)
    cycles = [_extract_cycle_num(f) for f in files]
    print(f"Found {len(cycles)} cycles: {cycles}")
    
    # Pre-define human labels
    diff_labels = {
        "innovation": "Innovation (Obs - Background)",
        "increment":  "Increment (Analysis - Background)",
        "ana_truth":  "Analysis Error |Analysis - Truth|",
        "background": "Background State",
        "analysis": "Analysis State"
    }

    # Loop over variables
    for var in VARS:
        print(f"\n=== Processing {var} ===")
        
        # Store spatial means for time series
        ts_data = {"innovation": [], "increment": [], "ana_truth": []}
        
        # Prepare 3D arrays for GIF min/max calculation [cycles, lat, lon]
        # We'll compute these on the fly to save memory, but we need ranges first?
        # Actually, let's collect all frames first to normalize colormap
        
        frames = {"innovation": [], "increment": [], "ana_truth": []}
        
        for i, cycle_k in enumerate(cycles):
            cycle_file = files[i]
            truth_file = ref_path / "snapshots" / f"reference_solution_{cycle_k}.nc"
            # We don't necessarily need free_run here unless we want NoDA error
            
            # Read fields
            xb = _read_nc_field(cycle_file, var, LEVEL_IDX, prefix="xb_mean")  # Background
            xa = _read_nc_field(cycle_file, var, LEVEL_IDX, prefix="xa_mean")  # Analysis
            xt = _read_nc_field(truth_file, var, LEVEL_IDX, prefix=None)       # Truth (raw var name)
            
            # Custom read for Innovation (Obs)
            # Try to read 'obs' prefix using helper, or custom fallback if naming is weird
            obs_val = _read_nc_field(cycle_file, var, LEVEL_IDX, prefix="obs")
            
            # If helper returned NaNs (because variable might be named differently or missing),
            # we check manually only if needed. But _read_nc_field with prefix="obs" does exactly what we want:
            # looks for obs_{var}_lev{k}. 
            
            # If it's pure NaN, it means obs were missing/not assimilated this cycle/loc.
            # That's fine, difference will be NaN.
            
            # --- COMPUTE DIFFERENCES ---
            
            # Innovation: Obs - Background (skip if obs is missing)
            innov = obs_val - xb
            
            # Increment: Analysis - Background
            incr = xa - xb
            
            # Error: |Analysis - Truth|
            err = np.abs(xa - xt)
            
            # Store frames
            frames["innovation"].append(innov)
            frames["increment"].append(incr)
            frames["ana_truth"].append(err)
            
            # Store means
            ts_data["innovation"].append(np.nanmean(innov))
            ts_data["increment"].append(np.nanmean(incr))
            ts_data["ana_truth"].append(np.nanmean(err))

        # --- GENERATE PLOTS AND GIFS ---
        
        disp_cycles = np.array(cycles) + 1 # 1-based indexing for display
        
        # 1. Time Series Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(disp_cycles, ts_data["innovation"], label="Innovation (obs-bkg)", marker='.', alpha=0.7)
        ax.plot(disp_cycles, ts_data["increment"], label="Increment (ana-bkg)", marker='.', alpha=0.7)
        ax.plot(disp_cycles, ts_data["ana_truth"], label="|Ana - Truth|", marker='.', color='red')
        
        ax.set_title(f"{var} - Spatially Averaged Diagnostics")
        ax.set_xlabel("Cycle")
        ax.set_ylabel(f"Mean {UNITS.get(var, '')}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ts_path = out_dir / f"{var}_spatial_means.png"
        plt.savefig(ts_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Time Series: {ts_path}")
        
        # 2. Generate GIFs
        for mode in ["innovation", "increment", "ana_truth"]:
            data_stack = np.array(frames[mode])
            
            # Determine Color Limits
            if mode == "ana_truth":
                # Sequential (0 to max)
                vmin = 0
                vmax = np.nanmax(data_stack) if np.any(np.isfinite(data_stack)) else 1.0
                cmap = "viridis"
            else:
                # Diverging (-max to max)
                abs_max = np.nanmax(np.abs(data_stack)) if np.any(np.isfinite(data_stack)) else 1.0
                vmin = -abs_max
                vmax = abs_max
                cmap = "coolwarm"
            
            # SKIP if all data is NaN (common for innovation if obs not saved)
            if np.all(np.isnan(data_stack)):
                print(f"  Skipping {mode} GIF for {var} (All NaNs)")
                continue

            # Setup Figure
            fig = plt.figure(figsize=(7, 4))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines(linewidth=0.7, color='black')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
            
            # Plot first frame
            im = ax.imshow(data_stack[0], origin='lower', extent=[0, 360, -90, 90],
                          cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest',
                          transform=ccrs.PlateCarree())
            
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
            title_text = ax.set_title(f"{var} {diff_labels[mode]} - Cycle {cycles[0]}")
            
            # Animate
            def update(frame_idx):
                im.set_data(data_stack[frame_idx])
                title_text.set_text(f"{var} {diff_labels[mode]} - Cycle {cycles[frame_idx]}")
                return [im, title_text]
            
            gif_path = out_dir / f"{var}_{mode}.gif"
            writer = PillowWriter(fps=2)
            
            try:
                with writer.saving(fig, gif_path, dpi=100):
                    for i in range(len(cycles)):
                        update(i)
                        writer.grab_frame()
                print(f"  Saved GIF: {gif_path}")
            except Exception as e:
                print(f"  Error saving GIF {gif_path}: {e}")
            
            plt.close()

if __name__ == "__main__":
    main()
