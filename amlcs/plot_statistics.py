#!/usr/bin/env python3
"""
Statistics Plotter for NetCDF Output
Plots Spatial Mean of:
1. O-B (Observation - Background)
2. A-B (Analysis - Background)
3. A-T (Analysis - Truth)

Strictly for 'G1' (Method 2) variables.
Configuration is set directly in the script below.

Usage:
    python amlcs/plot_statistics.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from netCDF4 import Dataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Experiment path relative to 'runs' directory
EXP_REL_PATH = "t21_50_0.05_20_ReverseSDE_1_1_100"

# Number of cycles to process (0 to M-1)
M = 20

# Variables to plot (Strictly G1 / Method 2)
# If None, defaults to all G1 model variables
VARIABLES = ["PSG1", "TG1", "TRG1", "UG1", "VG1"]

# Levels to plot (indices, e.g., 0 for surface/PS, 0-7 for others)
# If None, defaults to range(8)
LEVELS = [0, 1, 2, 3, 4, 5, 6, 7]

# Output directory name (relative to experiment folder)
PLOT_DIR_NAME = "plots/stat_plots"

# ==============================================================================
# END CONFIGURATION
# ==============================================================================

# Matplotlib styling
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})
try:
    import seaborn as sns
    sns.set_style("darkgrid")
except ImportError:
    pass

# Variable mappings for pretty titles
VAR_CODES = {
    "TG1": "T_1", "UG1": "u_1", "VG1": "v_1", "TRG1": "Hq_1", "PSG1": "PS_1",
}
PSLVL = [30, 100, 200, 300, 500, 700, 850, 925]

def spatial_mean(data):
    """Compute spatial mean, ignoring NaNs."""
    if data is None:
        return np.nan
    return np.nanmean(data)

def process_file_get_stats(nc_file, var, lev):
    """
    Reads a NetCDF file and returns spatial means for O-B, A-B, A-T.
    Returns tuple: (mean_OB, mean_AB, mean_AT)
    """
    if not nc_file.exists():
        return np.nan, np.nan, np.nan

    try:
        with Dataset(nc_file, 'r') as nc:
            lev_tag = f"lev{lev}"
            
            # Helper to read variable
            def read_var(prefix):
                # Construct possible variable names
                vnames = [f"{prefix}_{var}_{lev_tag}", f"{prefix}_{var}"]
                for vname in vnames:
                    if vname in nc.variables:
                        return nc.variables[vname][:]
                return None

            xa_mean = read_var("xa_mean")
            xb_mean = read_var("xb_mean")
            if xb_mean is None: xb_mean = read_var("xb") # Fallback
            truth = read_var("truth")
            obs = read_var("obs")
            
            # --- O-B (Obs - Background) ---
            ob_val = np.nan
            if obs is not None and xb_mean is not None:
                ob_diff = obs - xb_mean
                ob_val = spatial_mean(ob_diff)

            # --- A-B (Analysis - Background) ---
            ab_val = np.nan
            if xa_mean is not None and xb_mean is not None:
                ab_diff = xa_mean - xb_mean
                ab_val = spatial_mean(ab_diff)

            # --- A-T (Analysis - Truth) ---
            at_val = np.nan
            if xa_mean is not None and truth is not None:
                at_diff = xa_mean - truth
                at_val = spatial_mean(at_diff)
                
            return ob_val, ab_val, at_val

    except Exception as e:
        print(f"Warning: Error reading {nc_file} for {var} lev {lev}: {e}")
        return np.nan, np.nan, np.nan

def process_experiment(exp_path, variables, levels, M, plot_dir_name):
    """Process a single experiment and generate plots."""
    
    print(f"Processing experiment: {exp_path}")
    
    plots_path = exp_path / plot_dir_name
    plots_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plots_path}")

    # Identify files
    cycle_files = []
    valid_cycles = []
    
    for k in range(0, M):
        # List of candidate paths in order of preference
        candidates = [
            exp_path / f"unified_cycle{k}.nc",
            exp_path / "linear_results" / f"unified_cycle{k}.nc",
            exp_path / f"reverseSDE_cycle{k}.nc",
            exp_path / "linear_results" / f"reverseSDE_cycle{k}.nc",
        ]
        
        found_file = None
        for cand in candidates:
            if cand.exists():
                found_file = cand
                break
        
        if found_file:
            cycle_files.append((k, found_file))
            valid_cycles.append(k)

    if not valid_cycles:
        print(f"No valid cycle files found for {exp_path}")
        return

    # Process each variable and level
    for var in variables:
        if not var.endswith("G1"):
             # Strict filtering for G1
             continue

        for lev in levels:
            # Skip invalid combinations
            if "PSG" in var and lev > 0: continue
            if "TRG" in var and lev < 2: continue

            # Collect time series
            ob_series = []
            ab_series = []
            at_series = []
            times = []

            for k, nc_file in cycle_files:
                ob, ab, at = process_file_get_stats(nc_file, var, lev)
                ob_series.append(ob)
                ab_series.append(ab)
                at_series.append(at)
                times.append(k)

            # Convert to numpy for easier stats
            ob_series = np.array(ob_series)
            ab_series = np.array(ab_series)
            at_series = np.array(at_series)
            times = np.array(times)

            # Compute Time Averages (ignoring NaNs)
            ob_avg = np.nanmean(ob_series)
            ab_avg = np.nanmean(ab_series)
            at_avg = np.nanmean(at_series)

            # --- Plotting ---
            plt.figure(figsize=(10, 5))
            
            # Title
            lvl_str = f" at {PSLVL[lev]} mb" if "PSG" not in var else ""
            var_label = VAR_CODES.get(var, var)
            plt.title(rf"Statistics for $\mathrm{{{var_label}}}{lvl_str}$")

            # Plot curves
            # O-B
            if not np.all(np.isnan(ob_series)):
                label_ob = f"O-B (Avg: {ob_avg:.2e})"
                plt.plot(times, ob_series, label=label_ob, marker='.', linestyle='-')
            
            # A-B
            if not np.all(np.isnan(ab_series)):
                label_ab = f"A-B (Avg: {ab_avg:.2e})"
                plt.plot(times, ab_series, label=label_ab, marker='.', linestyle='-')

            # A-T
            if not np.all(np.isnan(at_series)):
                label_at = f"A-T (Avg: {at_avg:.2e})"
                # Using dashed for A-T to distinguish error from increments/innovations visually
                plt.plot(times, at_series, label=label_at, marker='.', linestyle='--')

            plt.xlabel("Cycle")
            plt.ylabel("Spatial Mean Difference")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save
            out_file = plots_path / f"stats_{var}_lev{lev}.png"
            plt.savefig(out_file, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_file}")

def main():
    root_path = Path.cwd()
    # Resolve 'runs' directory relative to script or CWD
    if (root_path / "runs").exists():
        runs_dir = root_path / "runs"
    elif (root_path.parent / "runs").exists():
        runs_dir = root_path.parent / "runs"
    else:
        runs_dir = root_path
        
    exp_path = runs_dir / EXP_REL_PATH
    
    if not exp_path.exists():
        print(f"Error: Experiment path not found: {exp_path}")
        return

    process_experiment(exp_path, VARIABLES, LEVELS, M, PLOT_DIR_NAME)

if __name__ == "__main__":
    main()
