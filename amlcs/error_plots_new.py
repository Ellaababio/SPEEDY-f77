#!/usr/bin/env python3
"""
Error Plotter for NetCDF Output (ReverseSDE / EnKF)
Plots absolute L2 error for Analysis, Background, and NODA.

Usage:
    python error_plots_nc_version.py <csv_config_file>

Config File Format (CSV):
    exp_path,variable,level,M,resolution,plot_dir_name
    runs/my_exp,PSG1,0,100,t21,my_plots
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from netCDF4 import Dataset

# Matplotlib styling
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})
try:
    import seaborn as sns
    sns.set_style("darkgrid")
except ImportError:
    pass

# Variable mappings
VAR_CODES = {
    "TG0": "T_0", "UG0": "u_0", "VG0": "v_0", "TRG0": "Hq_0", "PSG0": "PS_0",
    "TG1": "T_1", "UG1": "u_1", "VG1": "v_1", "TRG1": "Hq_1", "PSG1": "PS_1",
    "WDG1": r"\theta_1",
}
PSLVL = [30, 100, 200, 300, 500, 700, 850, 925]
MODEL_VARS = ["PSG0", "PSG1", "TG0", "TG1", "TRG0", "TRG1", "UG0", "UG1", "VG0", "VG1", "WDG1"]


def compute_l2_error(state, truth):
    """Compute L2 error between state and truth."""
    diff = state - truth
    return np.sqrt(np.mean(diff**2))


def process_experiment(exp_path, variables, levels, M, plot_dir_name, output_dir=None):
    """Process a single experiment and generate plots."""
    
    print(f"Processing experiment: {exp_path}")
    
    if output_dir and not pd.isna(output_dir):
        base_path = Path(output_dir)
    else:
        base_path = exp_path / "plots" / "errors_nc"

    if plot_dir_name and not pd.isna(plot_dir_name):
        plots_path = base_path / plot_dir_name
    else:
        plots_path = base_path
            
    plots_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plots_path}")

    # Data storage
    # Structure: data[var][lev] = {'ana': [], 'bkg': [], 'noda': []}
    data = {var: {lev: {'ana': [], 'bkg': [], 'noda': []} for lev in levels} for var in variables}
    
    # Iterate over cycles 1 to M
    # We treat cycle 0 as the anchor (if it exists) or synthesize it
    
    valid_cycles = []
    
    # First, try to get the anchor value from Cycle 0
    anchor_error = None
    
    # Check for cycle 0
    fn1 = exp_path / "reverseSDE_cycle0.nc"
    fn2 = exp_path / "linear_normalization_results" / "reverseSDE_cycle0.nc"
    fn3 = exp_path / "linear_results" / "unified_cycle0.nc"
    fn4 = exp_path / "unified_cycle0.nc"
    
    nc_0 = None
    if fn1.exists(): nc_0 = fn1
    elif fn2.exists(): nc_0 = fn2
    elif fn3.exists(): nc_0 = fn3
    elif fn4.exists(): nc_0 = fn4
    
    # Store cycle 0 NODA errors for anchor
    # Structure: anchor_data[var][lev] = error_value
    anchor_data = {var: {lev: None for lev in levels} for var in variables}
    
    if nc_0:
        try:
            with Dataset(nc_0, 'r') as nc:
                for var in variables:
                    for lev in levels:
                        if "PSG" in var and lev > 0: continue
                        if "TRG" in var and lev < 2: continue
                        
                        lev_tag = f"lev{lev}"
                        def read_var(prefix):
                            vname = f"{prefix}_{var}_{lev_tag}"
                            return nc.variables[vname][:] if vname in nc.variables else None

                        tr = read_var("truth")
                        nd = read_var("noda")
                        
                        if tr is not None and nd is not None:
                            anchor_data[var][lev] = compute_l2_error(nd, tr)
        except Exception as e:
            print(f"Warning: Failed to read anchor from {nc_0}: {e}")

    # Now read cycles 0 to M-1 (assuming 0-based indexing for cycles)
    for k in range(0, M):
        fn1 = exp_path / f"reverseSDE_cycle{k}.nc"
        fn2 = exp_path / "linear_normalization_results" / f"reverseSDE_cycle{k}.nc"
        fn3 = exp_path / "linear_results" / f"unified_cycle{k}.nc"
        fn4 = exp_path / f"unified_cycle{k}.nc"
        
        nc_file = None
        if fn1.exists(): nc_file = fn1
        elif fn2.exists(): nc_file = fn2
        elif fn3.exists(): nc_file = fn3
        elif fn4.exists(): nc_file = fn4
        
        if nc_file is None:
            continue
            
        valid_cycles.append(k)
        
        try:
            with Dataset(nc_file, 'r') as nc:
                for var in variables:
                    for lev in levels:
                        if "PSG" in var and lev > 0: continue
                        if "TRG" in var and lev < 2: 
                             # print(f"Skipping {var} at level {lev} (Stratosphere)")
                             continue
                        
                        
                        lev_tag = f"lev{lev}"
                        def read_var(prefix):
                            vname = f"{prefix}_{var}_{lev_tag}"
                            return nc.variables[vname][:] if vname in nc.variables else None

                        xa = read_var("xa_mean")
                        xb = read_var("xb_mean")
                        if xb is None: xb = read_var("xb")
                        tr = read_var("truth")
                        nd = read_var("noda")
                        
                        if xa is not None and tr is not None:
                            data[var][lev]['ana'].append(compute_l2_error(xa, tr))
                            
                            if xb is not None:
                                data[var][lev]['bkg'].append(compute_l2_error(xb, tr))
                            else:
                                data[var][lev]['bkg'].append(np.nan)
                                
                            if nd is not None:
                                err_noda = compute_l2_error(nd, tr)
                                data[var][lev]['noda'].append(err_noda)
                                # If we didn't find cycle 0, use cycle 1 NODA as anchor
                                if anchor_data[var][lev] is None:
                                    anchor_data[var][lev] = err_noda
                            else:
                                data[var][lev]['noda'].append(np.nan)
                        else:
                            # Missing data for this cycle, append NaNs to maintain alignment
                            data[var][lev]['ana'].append(np.nan)
                            data[var][lev]['bkg'].append(np.nan)
                            data[var][lev]['noda'].append(np.nan)
        except Exception as e:
            print(f"Error reading {nc_file}: {e}")
            continue

    if not valid_cycles:
        print(f"No valid cycle files found for {exp_path}")
        return

    # Plotting
    # Prepend 0 to cycles, and shift cycle indices by +1 so 0 is anchor, 1 is cycle0, etc.
    cycles = np.array([0] + [c + 1 for c in valid_cycles])
    
    for var in variables:
        for lev in levels:
            if "PSG" in var and lev > 0: continue
            if "TRG" in var and lev < 2: continue
            
            series = data[var][lev]
            if not series['ana']: continue
            
            # Get anchor
            anc = anchor_data[var][lev]
            if anc is None: anc = np.nan
            
            # Prepend anchor to all series
            ana = np.array([anc] + series['ana'])
            bkg = np.array([anc] + series['bkg'])
            noda = np.array([anc] + series['noda'])
            
            plt.figure(figsize=(9, 4))
            
            lvl_str = f" at {PSLVL[lev]} mb" if "PSG" not in var else ""
            plt.title(rf"$\mathrm{{{VAR_CODES.get(var, var)}}}{lvl_str}$")
            
            plt.plot(cycles, ana, color="r", label="Analysis")
            
            if not np.all(np.isnan(bkg)):
                plt.plot(cycles, bkg, color="b", label="Background")
                
            if not np.all(np.isnan(noda)):
                plt.plot(cycles, noda, color="k", label="NODA")
                
            plt.ylabel(r"RMSE")
            plt.xlabel(r"$\mathrm{Assimilation\ Step}$")
            plt.legend(loc="best", prop={"size": 14})
            plt.tight_layout()
            
            out_file = plots_path / f"error_nc_{var}_{lev}.png"
            plt.savefig(out_file, bbox_inches="tight")
            plt.close()
            print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Creates error plots from NetCDF files (Absolute L2 Error).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "file", help="The name of the CSV file containing the configuration"
    )
    args = parser.parse_args()

    input_file = args.file
    print(f"* Reading input file {input_file}")
    
    try:
        df_params = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
        
    root_path = Path.cwd()
    # Assuming script is run from project root or amlcs/
    # If run from amlcs/, parent is root. If run from root, cwd is root.
    # Adjust logic to find 'runs' folder
    if (root_path / "runs").exists():
        runs_dir = root_path / "runs"
    elif (root_path.parent / "runs").exists():
        runs_dir = root_path.parent / "runs"
    else:
        # Fallback: assume exp_path in CSV is relative to CWD or absolute
        runs_dir = root_path

    for _, row in df_params.iterrows():
        exp_rel_path = row["exp_path"]
        exp_path = runs_dir / exp_rel_path
        
        # Parse variables
        if pd.isna(row["variable"]):
            variables = MODEL_VARS
        else:
            variables = [v.strip() for v in row["variable"].split(",")]
            
        # Parse levels
        if pd.isna(row["level"]):
            levels = range(8)
        else:
            levels = [int(v) for v in str(row["level"]).split(",")]
            
        M = int(row["M"])
        plot_dir_name = row["plot_dir_name"]
        
        # Read optional output_dir
        output_dir = row.get("output_dir", None)
        
        process_experiment(exp_path, variables, levels, M, plot_dir_name, output_dir)


if __name__ == "__main__":
    main()
