#!/usr/bin/env python3
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from netCDF4 import Dataset

# Professional Formatting
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})

# --- User Settings ---
# EnSF Experiment directory (Assimilating Surface Pressure ONLY)
ENSF_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data_ps0001_old"

# Optional comparison experiments for Figure 0 (up to 2 total)
FIG0_EXPERIMENTS = [
    {
        "label": "EnSF (PS Obs Only)",
        "path": ENSF_DIR,
        "file_template": "reverseSDE_cycle{cycle}.nc",
        "analysis_prefix": "xa_mean",
        "background_prefix": "xb_mean",
        "style": "b-",
    },
    {
        "label": "LETKF",
        "path": "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_LETKF_4_1_100",
        "file_template": "unified_cycle{cycle}.nc",
        "analysis_prefix": "xa_mean",
        "background_prefix": "xb_mean",
        "style": "g-",
    },
]

# Reference Directory (Truth + NoDA Free Run)
REF_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20"

# Output Directory for Plots
OUT_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/ps_only_conference_plots"

# Setup Constants
P0_HPA = 1000.0
CYCLES = list(range(21))  # Assuming 21 cycles (0-20) based on previous logs

def _read_nc_field(nc_path: Path, var: str, lev: int, prefix: str = None) -> np.ndarray:
    if not nc_path.exists():
        return None

    with Dataset(nc_path, 'r') as nc:
        # EnSF analysis fields
        if prefix:
            target_lev = 0 if "PSG" in var else lev
            field_name = f"{prefix}_{var}_lev{target_lev}"
            if field_name in nc.variables:
                return nc.variables[field_name][:]
            return None
        
        # Raw truth or free run fields
        if var in nc.variables:
            data = nc.variables[var]
            if data.ndim == 3:  # (nlev, lat, lon)
                return data[lev if "PSG" not in var else 0, :, :]
            elif data.ndim == 2:  # (lat, lon)
                return data[:]
            elif data.ndim == 4:  # (time, nlev, lat, lon)
                return data[0, lev if "PSG" not in var else 0, :, :]

    return None


def _compute_ps_rmse(truth_file: Path, exp_file: Path, cycle: int, analysis_prefix: str, background_prefix: str) -> float | None:
    t_ps_log = _read_nc_field(truth_file, "PSG1", 0)
    e_prefix = background_prefix if cycle == 0 else analysis_prefix
    e_ps_log = _read_nc_field(exp_file, "PSG1", 0, prefix=e_prefix)

    if t_ps_log is None or e_ps_log is None:
        return None

    t_ps = P0_HPA * np.exp(t_ps_log)
    e_ps = P0_HPA * np.exp(e_ps_log)
    return np.sqrt(np.mean((e_ps - t_ps)**2))

def main():
    ensf_path = Path(ENSF_DIR)
    ref_path = Path(REF_DIR)
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Gathering data...")

    # Data structures over time
    # Metrics: mean (spatial mean), rmse (vs Truth)
    res = {
        "PSG1": {"noda_mean": [], "noda_rmse": [], "ensf_mean": [], "ensf_rmse": []},
        "UG1":  {"noda_mean": [], "noda_rmse": [], "ensf_mean": [], "ensf_rmse": []}
    }
    
    valid_cycles = []
    fig0_res = {
        exp["label"]: []
        for exp in FIG0_EXPERIMENTS[:2]
    }
    
    for c in CYCLES:
        truth_file = ref_path / "snapshots" / f"reference_solution_{c}.nc"
        noda_file = ref_path / "free_run" / f"free_run_{c}.nc"
        ensf_file = ensf_path / f"reverseSDE_cycle{c}.nc"

        exp_files = {
            exp["label"]: Path(exp["path"]) / exp["file_template"].format(cycle=c)
            for exp in FIG0_EXPERIMENTS[:2]
        }

        if not (truth_file.exists() and noda_file.exists() and ensf_file.exists()):
            continue
            
        valid_cycles.append(c)
            
        # --- 1. Surface Pressure (PSG1) ---
        t_ps_log = _read_nc_field(truth_file, "PSG1", 0)
        n_ps_log = _read_nc_field(noda_file, "PSG1", 0)
        
        # Fair initial comparison: use background (xb) at cycle 0 since xa already has obs assimilated
        prefix = "xb_mean" if c == 0 else "xa_mean"
        e_ps_log = _read_nc_field(ensf_file, "PSG1", 0, prefix=prefix)
        
        if t_ps_log is not None and n_ps_log is not None and e_ps_log is not None:
            # Convert to hPa
            t_ps = P0_HPA * np.exp(t_ps_log)
            n_ps = P0_HPA * np.exp(n_ps_log)
            e_ps = P0_HPA * np.exp(e_ps_log)
            
            res["PSG1"]["noda_mean"].append(np.mean(n_ps))
            res["PSG1"]["noda_rmse"].append(np.sqrt(np.mean((n_ps - t_ps)**2)))
            res["PSG1"]["ensf_mean"].append(np.mean(e_ps))
            res["PSG1"]["ensf_rmse"].append(np.sqrt(np.mean((e_ps - t_ps)**2)))
        else:
            for k in res["PSG1"]: res["PSG1"][k].append(np.nan)

        # --- 2. Surface U-Wind (UG1, level 7) --- 
        # Typically the surface level in T21 is index 7
        LEV = 7 
        t_u = _read_nc_field(truth_file, "UG1", LEV)
        n_u = _read_nc_field(noda_file, "UG1", LEV)
        
        prefix = "xb_mean" if c == 0 else "xa_mean"
        e_u = _read_nc_field(ensf_file, "UG1", LEV, prefix=prefix)
        
        if t_u is not None and n_u is not None and e_u is not None:
            res["UG1"]["noda_mean"].append(np.mean(n_u))
            res["UG1"]["noda_rmse"].append(np.sqrt(np.mean((n_u - t_u)**2)))
            res["UG1"]["ensf_mean"].append(np.mean(e_u))
            res["UG1"]["ensf_rmse"].append(np.sqrt(np.mean((e_u - t_u)**2)))
        else:
            for k in res["UG1"]: res["UG1"][k].append(np.nan)

        for exp in FIG0_EXPERIMENTS[:2]:
            exp_file = exp_files[exp["label"]]
            rmse = None
            if truth_file.exists() and exp_file.exists():
                rmse = _compute_ps_rmse(
                    truth_file,
                    exp_file,
                    c,
                    exp["analysis_prefix"],
                    exp["background_prefix"],
                )
            fig0_res[exp["label"]].append(np.nan if rmse is None else rmse)

    cycles = np.array(valid_cycles)
    cycle_ticks = np.arange(0, cycles[-1] + 1, 5) if len(cycles) else np.array([])
    
    # ---------------------------------------------------------
    # PLOT 0: RMSE Line plot for Surface Pressure
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(cycles, res["PSG1"]["noda_rmse"], 'k--', linewidth=2, label="NoDA")
    for exp in FIG0_EXPERIMENTS[:2]:
        plt.plot(
            cycles,
            fig0_res[exp["label"]],
            exp["style"],
            linewidth=2,
            label=exp["label"],
        )
    plt.title("Surface Pressure RMSE Evolution", fontweight="bold")
    plt.xlabel("Assimilation Cycle")
    plt.ylabel("RMSE (hPa)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    if len(cycle_ticks):
        plt.xticks(cycle_ticks)
    plt.tight_layout()
    plot_file = out_path / "Fig0_RMSE_PSG1.png"
    plt.savefig(plot_file, dpi=300)
    print(f"Saved {plot_file}")
    plt.close()

    # ---------------------------------------------------------
    # Helper to create a 4-panel figure
    # ---------------------------------------------------------
    def make_4panel(var_key, var_name, title, ylabel):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)
        
        # Row 1: NoDA Mean | NoDA Error
        axs[0,0].plot(cycles, res[var_key]["noda_mean"], 'k.-')
        axs[0,0].set_title(f"NoDA {var_name} Mean ({ylabel})")
        axs[0,0].set_ylabel(ylabel)
        
        axs[0,1].plot(cycles, res[var_key]["noda_rmse"], 'r.-')
        axs[0,1].set_title(f"NoDA {var_name} RMSE ({ylabel})")
        axs[0,1].set_ylabel(ylabel)
        
        # Row 2: EnSF Mean | EnSF Error
        axs[1,0].plot(cycles, res[var_key]["ensf_mean"], 'b.-')
        axs[1,0].set_title(f"EnSF {var_name} Mean ({ylabel})")
        axs[1,0].set_xlabel("Assimilation Cycle")
        axs[1,0].set_ylabel(ylabel)
        
        axs[1,1].plot(cycles, res[var_key]["ensf_rmse"], 'g.-')
        axs[1,1].set_title(f"EnSF {var_name} RMSE ({ylabel})")
        axs[1,1].set_xlabel("Assimilation Cycle")
        axs[1,1].set_ylabel(ylabel)
        
        for ax in axs.flat:
            ax.grid(True, linestyle=':', alpha=0.6)
            if len(cycle_ticks):
                ax.set_xticks(cycle_ticks)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        return fig

    # ---------------------------------------------------------
    # PLOT 1: 4-Panel Surface Pressure
    # ---------------------------------------------------------
    fig = make_4panel("PSG1", "Surface Pressure", "Experiment: Assimilating Surface Pressure Only\nVariable: Surface Pressure", "hPa")
    plot_file = out_path / "Fig1_4panel_PSG1.png"
    fig.savefig(plot_file, dpi=300)
    print(f"Saved {plot_file}")
    plt.close(fig)

    # ---------------------------------------------------------
    # PLOT 2: 4-Panel U-Wind
    # ---------------------------------------------------------
    fig = make_4panel("UG1", "U-Wind", "Experiment: Assimilating Surface Pressure Only\nVariable: Surface U-Wind", "Wind Speed (m/s)")
    plot_file = out_path / "Fig2_4panel_UG1.png"
    fig.savefig(plot_file, dpi=300)
    print(f"Saved {plot_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
