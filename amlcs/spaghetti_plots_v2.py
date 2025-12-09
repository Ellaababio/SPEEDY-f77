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
NC_PATH = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_ps_only/sde_tracking.nc"

# Output directory for plots
OUT_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_ps_only/sde_plots"

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
def generate_plots(xt_data, var_names, output_base_dir, suffix=""):
    """
    Generates spaghetti plots for the given data.
    
    Args:
        xt_data: NetCDF variable or array-like (ncycle, block, psteps, var, ens)
        var_names: List of variable names
        output_base_dir: Base directory for output
        suffix: Suffix for plot titles (e.g., " (Mean)" or " (Gridpoint)")
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
        for j, var in enumerate(var_names):
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
                last_idx = min(valid_steps[-1], 180)
                end_data = M[last_idx, j, :]
                end_data = end_data[np.isfinite(end_data)]
                ax_main.axvline(x=last_idx, color='k', linestyle='--', alpha=0.5, label='End')
            else:
                end_data = np.array([])

            from scipy.stats import gaussian_kde
            def plot_pdf(ax, data, color, label):
                if len(data) < 2: return
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
            outpath = os.path.join(var_out_dir, f"cycle{visual_cycle}_{var}.png")
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

    # 1. Process Spatial Mean
    if "xt_state_mean" in nc.variables:
        mean_out_dir = os.path.join(OUT_DIR, "spatial_mean")
        generate_plots(nc["xt_state_mean"], var_names, mean_out_dir, suffix=" (Mean)")
    elif "xt_state" in nc.variables:
        # Fallback for old files
        print("Warning: 'xt_state_mean' not found, using 'xt_state' as mean.")
        mean_out_dir = os.path.join(OUT_DIR, "spatial_mean")
        generate_plots(nc["xt_state"], var_names, mean_out_dir, suffix=" (Mean)")
    else:
        print("Error: No mean state variable found.")

    # 2. Process Gridpoint
    if "xt_state_gridpoint" in nc.variables:
        grid_out_dir = os.path.join(OUT_DIR, "gridpoint")
        generate_plots(nc["xt_state_gridpoint"], var_names, grid_out_dir, suffix=" (Gridpoint)")
    else:
        print("Warning: 'xt_state_gridpoint' not found in NetCDF.")

    # 3. Process Normalized Mean
    if "xt_norm_mean" in nc.variables:
        norm_mean_out_dir = os.path.join(OUT_DIR, "normalized_mean")
        generate_plots(nc["xt_norm_mean"], var_names, norm_mean_out_dir, suffix=" (Norm Mean)")
    else:
        print("Warning: 'xt_norm_mean' not found in NetCDF.")

    # 4. Process Normalized Gridpoint
    if "xt_norm_gridpoint" in nc.variables:
        norm_grid_out_dir = os.path.join(OUT_DIR, "normalized_gridpoint")
        generate_plots(nc["xt_norm_gridpoint"], var_names, norm_grid_out_dir, suffix=" (Norm Gridpoint)")
    else:
        print("Warning: 'xt_norm_gridpoint' not found in NetCDF.")

    nc.close()
    print("\nAll plots generated.")

if __name__ == "__main__":
    main()
