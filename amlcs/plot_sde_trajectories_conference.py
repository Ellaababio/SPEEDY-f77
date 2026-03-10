#!/usr/bin/env python3
import os
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde, shapiro
from netCDF4 import Dataset

# Professional Formatting for Conference Plots
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})

ENSF_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data_ps0001"
SDE_FILE = Path(ENSF_DIR) / "sde_tracking.nc"
OUT_DIR = Path("/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/ps_only_conference_plots")

def get_stats_label(data, name):
    if len(data) < 3:
         return f"{name} (N/A)"
    try:
        mu = np.mean(data)
        sigma = np.std(data)
        _, p_val = shapiro(data)
        return f"{name}\nμ={mu:.2f}, σ={sigma:.2f}\nS-W p={p_val:.3f}"
    except Exception:
         return f"{name} (Error)"

def plot_pdf(ax, ax_main, data, color, base_label):
    if len(data) < 2: return
    label = get_stats_label(data, base_label)
    try:
        density = gaussian_kde(data)
        ymin, ymax = ax_main.get_ylim()
        y_grid = np.linspace(ymin, ymax, 100)
        x_density = density(y_grid)
        ax.plot(x_density, y_grid, color=color, label=label)
        ax.fill_betweenx(y_grid, 0, x_density, color=color, alpha=0.2)
    except Exception:
        pass

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not SDE_FILE.exists():
        print(f"Tracking file not found: {SDE_FILE}")
        return

    # Points targeted in sequential_methods.py
    POINTS = [
        {"desc": "Largest Initial Background Error", "lat_idx": 8, "lon_idx": 31, "pt_idx": 0},
        {"desc": "Largest Analysis Increment", "lat_idx": 24, "lon_idx": 36, "pt_idx": 1}
    ]

    CYCLE = 0

    print("Reading SDE tracking data...")
    with Dataset(SDE_FILE, 'r') as nc:
        # User wants the normalized unitless SDE space, like what spaghetti_plots_v2 did
        var_name = "xt_norm_gridpoint"
        if var_name not in nc.variables:
            print(f"Variable {var_name} not in SDE tracking file.")
            return

        raw = nc.variables[var_name][:]
        # Convert MaskedArray → plain float64 (fills masked cells with NaN)
        if isinstance(raw, np.ma.MaskedArray):
            xt_state = raw.filled(np.nan).astype(np.float64)
        else:
            xt_state = np.array(raw, dtype=np.float64)
        var_names = nc.variables["var_names"][:]
        
    print(f"{var_name} shape: {xt_state.shape}")
    
    # Identify which var index is PSG1
    var_list = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]
    try:
        psg_idx = var_list.index("PSG1")
    except ValueError:
        print("PSG1 not found in tracking vars!")
        return

    # Cycle Data Shape: (block, psteps, var, pts, ens)
    c0_state = xt_state[CYCLE]   # (block, psteps, var, pts, ens)
    num_steps = c0_state.shape[1]
    nens = c0_state.shape[4]
    
    # x-axis from 0 to psteps (Pseudo-time representation like spaghetti_plots_v2)
    x_axis = np.arange(num_steps)
    
    for pt in POINTS:
        pt_i = pt["pt_idx"]
        
        # Grab gridpoint states across all blocks: (block, psteps, ens)
        pt_blocks = c0_state[:, :, psg_idx, pt_i, :]
        
        # Find a block where enough values along psteps are finite
        valid_blocks = np.where(
            np.sum(np.isfinite(pt_blocks), axis=(1, 2)) > 0
        )[0]
            
        if len(valid_blocks) == 0:
            print(f"No valid block data found for point {pt['desc']} (idx {pt_i})")
            continue
            
        block_idx = valid_blocks[0]
        ens_norm = pt_blocks[block_idx] # (psteps, ens)

        # Setup 2-panel figure
        fig = plt.figure(figsize=(12, 6))
        gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_pdf = fig.add_subplot(gs[1], sharey=ax_main)
        
        # --- Main Spaghetti Plot ---
        for e in range(nens):
            ax_main.plot(x_axis, ens_norm[:, e], alpha=0.4, color='tab:blue', linewidth=1.0)

        # Robust y-limits
        clean = ens_norm.reshape(-1)
        clean = clean[np.isfinite(clean)]
        if len(clean) > 20:
            lo, hi = np.nanpercentile(clean, [1, 99])
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                # Add 10% vertical padding
                pad = (hi - lo) * 0.1
                ax_main.set_ylim(lo - pad, hi + pad)

        ax_main.set_title(f"ReverseSDE Trajectory - Cycle {CYCLE + 1} -- Surface Pressure \n{pt['desc']}", fontsize=12, fontweight='bold')
        ax_main.set_xlabel("Pseudo-time (Reverse Diffusion Step)")
        ax_main.set_ylabel("Normalized State Vector")
        ax_main.grid(True, linestyle=":", alpha=0.6)
        ax_main.set_xlim(0, 180)
        
        # --- PDF Curves ---
        start_data = ens_norm[0, :]
        start_data = start_data[np.isfinite(start_data)]
        
        valid_steps = np.where(np.any(np.isfinite(ens_norm), axis=1))[0]
        if len(valid_steps) > 0:
            last_idx = min(valid_steps[-1], num_steps - 1)
            end_data = ens_norm[last_idx, :]
            end_data = end_data[np.isfinite(end_data)]
        else:
            end_data = np.array([])
            
        plot_pdf(ax_pdf, ax_main, start_data, 'tab:red', 'Start')
        plot_pdf(ax_pdf, ax_main, end_data, 'tab:green', 'End')
        
        ax_pdf.set_xlabel("Density")
        plt.setp(ax_pdf.get_yticklabels(), visible=False)
        ax_pdf.grid(True, linestyle=":", alpha=0.6)
        ax_pdf.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),
                      fontsize='small', frameon=True)

        plt.tight_layout()
        if "Initial Background" in pt['desc']:
            fname = "Fig3_SDE_Trajectory_LargestError_Normalized.png"
        else:
            fname = "Fig4_SDE_Trajectory_LargestIncrement_Normalized.png"
            
        plot_file = OUT_DIR / fname
        plt.savefig(plot_file, dpi=300)
        print(f"Saved {plot_file}")
        plt.close(fig)

if __name__ == "__main__":
    main()
