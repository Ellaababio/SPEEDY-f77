#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC-safe
import matplotlib.pyplot as plt
from netCDF4 import Dataset

###############################################################################
# ========================= CONFIGURATION ====================================
###############################################################################

# Path to the sde_tracking.nc file
# Update this to point to your new experiment's output
NC_PATH = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/sde_tracking.nc"

# Output directory for plots
# Output directory base
OUT_DIR_BASE = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results/ps_0.003/sde_plots"

# Toggle splitting
SPLIT_PLOTS = False

# Variables to plot (None for all)
TARGET_VARS = None

# Construct final output directory
OUT_DIR = OUT_DIR_BASE + ("_split" if SPLIT_PLOTS else "")

# Units dictionary
UNITS = {
    "UG1":  "m/s",
    "VG1":  "m/s",
    "TG1":  "K",
    "TRG1": "g/kg",
    "PSG1": "log(ps/p0)",
}

###############################################################################
# ========================= END CONFIGURATION ================================
###############################################################################

# --------------------------------------------------------
# Robust var-name decoder
# --------------------------------------------------------
def decode_var_names(raw):
    decoded = []
    for v in raw:
        if isinstance(v, str):
            decoded.append(v.strip())
        elif hasattr(v, "tobytes"):
            decoded.append(v.tobytes().decode("utf-8").strip())
        elif isinstance(v, (bytes, bytearray)):
            decoded.append(v.decode("utf-8").strip())
        else:
            decoded.append(str(v).strip())
    return decoded


# --------------------------------------------------------
# Plotting Function
# --------------------------------------------------------
def generate_plots(xt_data, var_names, output_base_dir, suffix="", filename_suffix="", time_slice=None, target_vars=None):
    """
    Generates spaghetti plots for the given data.
    
    Args:
        xt_data: NetCDF variable or array-like (ncycle, block, psteps, var, ens)
        var_names: List of variable names
        output_base_dir: Base directory for output
        suffix: Suffix for plot titles (e.g., " (Mean)" or " (Gridpoint)")
        filename_suffix: Suffix for creating unique filenames
        time_slice: Slice object for time dimension
        target_vars: List of variables to plot (None for all)
    """
    ncycles = xt_data.shape[0]
    nblocks = xt_data.shape[1]
    psteps  = xt_data.shape[2]
    nvars   = xt_data.shape[3]
    nens    = xt_data.shape[4]
    
    FILL = getattr(xt_data, "_FillValue", None)

    print(f"\nProcessing data for suffix '{suffix}'...")
    print(f"Cycles={ncycles} Blocks={nblocks} psteps={psteps} vars={nvars} ens={nens}")

    for visual_cycle in range(1, ncycles + 1):
        internal_k = visual_cycle - 1
        print(f"  Cycle {visual_cycle}...")

        # Load slice
        xk = xt_data[internal_k][:]

        # masked → NaN
        if isinstance(xk, np.ma.MaskedArray):
            xk = xk.filled(np.nan)

        cycle_data = xk.astype(np.float64)

        # Time slicing
        if time_slice is not None:
            # cycle_data shape is (blocks, psteps, vars, ens)
            cycle_data = cycle_data[:, time_slice, :, :]

        # remove fill values
        if FILL is not None:
            bad = (cycle_data == FILL) | (np.abs(cycle_data) > 1e30)
            cycle_data[bad] = np.nan

        # Block filtering
        finite_counts = np.sum(np.isfinite(cycle_data), axis=(1, 2, 3))
        min_valid = 1 
        valid_blocks = np.where(finite_counts >= min_valid)[0]

        if len(valid_blocks) == 0:
            print(f"    No valid blocks for Cycle {visual_cycle} → skipping")
            continue

        # Average across blocks (collapses block dimension)
        M = np.nanmean(cycle_data[valid_blocks, :, :, :], axis=0)

        # Plotting
        target_indices = range(len(var_names))
        if target_vars is not None:
             target_indices = [i for i, v in enumerate(var_names) if v in target_vars]

        for j in target_indices:
            var = var_names[j]
            unit = UNITS.get(var, "")
            unit_tag = f" ({unit})" if unit else ""
            ylabel = f"{var} [{unit}]" if unit else var

            # Create subfolder
            var_out_dir = os.path.join(output_base_dir, var)
            os.makedirs(var_out_dir, exist_ok=True)

            # Setup figure
            fig = plt.figure(figsize=(12, 5))
            gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
            ax_main = fig.add_subplot(gs[0])
            ax_pdf = fig.add_subplot(gs[1], sharey=ax_main)

            # --- Main Spaghetti Plot ---
            for e in range(nens):
                ax_main.plot(M[:, j, e], alpha=0.4, color='tab:blue', linewidth=0.8)

            # robust y-limits
            clean = M[:, j, :].reshape(-1)
            clean = clean[np.isfinite(clean)]
            if len(clean) > 20:
                lo, hi = np.nanpercentile(clean, [1, 99])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    ax_main.set_ylim(lo, hi)

            ax_main.set_title(f"Cycle {visual_cycle} -- {var}{unit_tag}{suffix}")
            ax_main.set_xlabel("Pseudo-time")
            ax_main.set_ylabel(ylabel)
            ax_main.grid(True, alpha=0.3)

            # --- PDF Curves ---
            start_data = M[0, j, :]
            start_data = start_data[np.isfinite(start_data)]
            
            valid_steps = np.where(np.any(np.isfinite(M[:, j, :]), axis=1))[0]
            if len(valid_steps) > 0:
                last_idx = min(valid_steps[-1], M.shape[0] - 1)
                end_data = M[last_idx, j, :]
                end_data = end_data[np.isfinite(end_data)]
                ax_main.axvline(x=last_idx, color='k', linestyle='--', alpha=0.5, label='End')
            else:
                end_data = np.array([])

            from scipy.stats import gaussian_kde, shapiro
            
            def get_stats_label(data, name):
                if len(data) < 3:
                     return f"{name} (N/A)"
                try:
                    mu = np.mean(data)
                    sigma = np.std(data)
                    # Shapiro-Wilk test (returns statistic, p-value)
                    _, p_val = shapiro(data)
                    return f"{name}\nμ={mu:.2f}, σ={sigma:.2f}\nS-W p={p_val:.3f}"
                except Exception:
                     return f"{name} (Error)"

            def plot_pdf(ax, data, color, base_label):
                if len(data) < 2: return
                
                label = get_stats_label(data, base_label)
                
                try:
                    density = gaussian_kde(data)
                    ymin, ymax = ax_main.get_ylim()
                    y_grid = np.linspace(ymin, ymax, 100)
                    x_density = density(y_grid)
                    ax.plot(x_density, y_grid, color=color, label=label)
                    ax.fill_betweenx(y_grid, 0, x_density, color=color, alpha=0.2)
                except Exception as e:
                    pass

            plot_pdf(ax_pdf, start_data, 'tab:red', 'Start')
            plot_pdf(ax_pdf, end_data, 'tab:green', 'End')
            
            ax_pdf.set_xlabel("Density")
            plt.setp(ax_pdf.get_yticklabels(), visible=False)
            ax_pdf.grid(True, alpha=0.3)
            ax_pdf.legend(loc='upper right', fontsize='small')
            
            plt.tight_layout()
            plt.tight_layout()
            outpath = os.path.join(var_out_dir, f"cycle{visual_cycle}_{var}{filename_suffix}.png")
            plt.savefig(outpath, dpi=150)
            plt.close()

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    print("Opening:", NC_PATH)
    try:
        nc = Dataset(NC_PATH, "r")
    except FileNotFoundError:
        print(f"Error: File not found at {NC_PATH}")
        return

    raw_varnames = nc["var_names"][:]
    var_names = decode_var_names(raw_varnames)

    # Define time slices and target
    # Define time slices and target
    if SPLIT_PLOTS:
        slices = [
            (slice(0, 150), "_0-150"),
            (slice(150, None), "_150-plus")
        ]
    else:
        slices = [
            (slice(None), "")
        ]

    target_vars = TARGET_VARS

    # 1. Process Spatial Mean
    if "xt_state_mean" in nc.variables:
        mean_out_dir = os.path.join(OUT_DIR, "spatial_mean")
        for t_slice, f_suff in slices:
            generate_plots(nc["xt_state_mean"], var_names, mean_out_dir, 
                           suffix=" (Mean)", filename_suffix=f_suff, 
                           time_slice=t_slice, target_vars=target_vars)

    elif "xt_state" in nc.variables:
        # Fallback for old files
        print("Warning: 'xt_state_mean' not found, using 'xt_state' as mean.")
        mean_out_dir = os.path.join(OUT_DIR, "spatial_mean")
        for t_slice, f_suff in slices:
            generate_plots(nc["xt_state"], var_names, mean_out_dir, 
                           suffix=" (Mean)", filename_suffix=f_suff, 
                           time_slice=t_slice, target_vars=target_vars)
    else:
        print("Error: No mean state variable found.")

    # 2. Process Gridpoint
    if "xt_state_gridpoint" in nc.variables:
        grid_out_dir = os.path.join(OUT_DIR, "gridpoint")
        for t_slice, f_suff in slices:
            generate_plots(nc["xt_state_gridpoint"], var_names, grid_out_dir, 
                           suffix=" (Gridpoint)", filename_suffix=f_suff, 
                           time_slice=t_slice, target_vars=target_vars)
    else:
        print("Warning: 'xt_state_gridpoint' not found in NetCDF.")

    # 3. Process Normalized Mean
    if "xt_norm_mean" in nc.variables:
        norm_mean_out_dir = os.path.join(OUT_DIR, "normalized_mean")
        for t_slice, f_suff in slices:
            generate_plots(nc["xt_norm_mean"], var_names, norm_mean_out_dir, 
                           suffix=" (Norm Mean)", filename_suffix=f_suff, 
                           time_slice=t_slice, target_vars=target_vars)
    else:
        print("Warning: 'xt_norm_mean' not found in NetCDF.")

    # 4. Process Normalized Gridpoint
    if "xt_norm_gridpoint" in nc.variables:
        norm_grid_out_dir = os.path.join(OUT_DIR, "normalized_gridpoint")
        for t_slice, f_suff in slices:
            generate_plots(nc["xt_norm_gridpoint"], var_names, norm_grid_out_dir, 
                           suffix=" (Norm Gridpoint)", filename_suffix=f_suff, 
                           time_slice=t_slice, target_vars=target_vars)
    else:
        print("Warning: 'xt_norm_gridpoint' not found in NetCDF.")

    nc.close()
    print("\nAll plots generated.")

if __name__ == "__main__":
    main()
