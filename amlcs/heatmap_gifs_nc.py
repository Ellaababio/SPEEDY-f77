#!/usr/bin/env python3
"""
Heatmaps + Time-Series (NetCDF Version)
Generates animated GIFs and spatial mean time-series plots from NetCDF cycle files.

Supports two modes:
  DUAL_MODE = False  →  Single experiment (original behaviour)
  DUAL_MODE = True   →  Side-by-side comparison of two experiments
                         • Heatmap GIFs use a shared colorbar
                         • Time-series are stacked (one panel per method)
"""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset

# ======================= USER SETTINGS =======================
# Experiment Directories
EXP_DIR   = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_1percent_obs_err/data"
EXP_DIR_2 = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_LETKF_4_1_100/linear_results_1percent_obs_err/data"  # Second experiment (only used when DUAL_MODE = True)

# Human-readable labels
EXP_LABEL_1 = "ReverseSDE"
EXP_LABEL_2 = "LETKF"

# Dual mode toggle
DUAL_MODE = True

# Reference Directory (where 'snapshots' and 'free_run' are)
REFERENCE_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20"

# Output Directory (relative to EXP_DIR by default)
OUT_DIR_NAME = "../heatmaps_nc_dual"

# Variables to process
VARS = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]

# Level index to plot (for 3D vars) - PSG1 is 2D so this is ignored for it
LEVEL_IDX = 7  # Surface level for T21 (0 is top, 7 is bottom)

# Nonlinear standard-observation settings used by the experiment.
# When enabled, innovations for standard observed variables are computed in
# observation space as y - atan(sf * xb) instead of mixing obs-space y with
# physical-space xb.
NONLINEAR_OBS = True
SCALEFACT = 0.5

# T21 grid dimensions
NLAT, NLON = 32, 64

# Units for time-series plots
UNITS = {"UG1": "m/s", "VG1": "m/s", "TG1": "K", "TRG1": "g/kg", "PSG1": "log(ps/p0)", "WSG1": "m/s", "WDG1": "rad"}

# =============================================================

def _circ_diff(a, b):
    """Circular difference between angles: (a - b) mapped to [-pi, pi]."""
    return (a - b + np.pi) % (2*np.pi) - np.pi


def _standard_obs_innovation(obs_val: np.ndarray, xb: np.ndarray, var: str) -> np.ndarray:
    """Compute innovations in the correct space for standard observations."""
    if var == "WDG1":
        return _circ_diff(obs_val, xb)
    if NONLINEAR_OBS and var not in {"WSG1", "WDG1"}:
        # The true assimilation operators use Z-score normalized grids BEFORE taking the arctan.
        # Since the heatmap script doesn't have the explicit mu/std vectors dumped in the NC files,
        # we will approximate it using the block's current spatial statistics so the arctan doesn't saturate to pi/2.
        xb_valid = xb[np.isfinite(xb)]
        if len(xb_valid) > 0:
            mu = np.mean(xb_valid)
            std = np.std(xb_valid) + 1e-6
            xb_norm = (xb - mu) / std
        else:
            xb_norm = xb
        return obs_val - np.arctan(SCALEFACT * xb_norm)
    return obs_val - xb

def _find_cycle_files(exp_path: Path):
    """Find available cycle files."""
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

        if var == "WSG1":
            if prefix:
                u = _read_nc_field(nc_path, "UG1", lev, prefix)
                v = _read_nc_field(nc_path, "VG1", lev, prefix)
            else:
                u = _read_nc_field(nc_path, "UG1", lev)
                v = _read_nc_field(nc_path, "VG1", lev)
            
            if not np.all(np.isnan(u)) and not np.all(np.isnan(v)):
                 return np.sqrt(u**2 + v**2)

        if var == "WDG1":
            if prefix:
                u = _read_nc_field(nc_path, "UG1", lev, prefix)
                v = _read_nc_field(nc_path, "VG1", lev, prefix)
            else:
                u = _read_nc_field(nc_path, "UG1", lev)
                v = _read_nc_field(nc_path, "VG1", lev)
            
            if not np.all(np.isnan(u)) and not np.all(np.isnan(v)):
                 return np.arctan2(v, u)

    return np.full((NLAT, NLON), np.nan)


# ─────────────────────────────────────────────────────────────
#  Helpers shared by both modes
# ─────────────────────────────────────────────────────────────
def _collect_experiment(exp_path, ref_path, var, cycles, files):
    """Return frames dict and ts_data dict for one experiment."""
    # Anchor (NoDA error at cycle 0)
    free_run_0 = ref_path / "free_run" / "free_run_0.nc"
    truth_0    = ref_path / "snapshots" / "reference_solution_0.nc"

    anchor_val = np.nan
    if free_run_0.exists() and truth_0.exists():
        fr = _read_nc_field(free_run_0, var, LEVEL_IDX, prefix=None)
        tr = _read_nc_field(truth_0,    var, LEVEL_IDX, prefix=None)
        if var == "WDG1":
            anchor_val = np.nanmean(np.abs(_circ_diff(fr, tr)))
        else:
            anchor_val = np.nanmean(np.abs(fr - tr))

    innov_anchor = np.nan
    incr_anchor = 0.0
    if free_run_0.exists() and len(files) > 0:
        c1_obs = _read_nc_field(files[0], var, LEVEL_IDX, prefix="obs")
        fr = _read_nc_field(free_run_0, var, LEVEL_IDX, prefix=None)
        if not np.all(np.isnan(c1_obs)) and not np.all(np.isnan(fr)):
            innov_0 = _standard_obs_innovation(c1_obs, fr, var)
            innov_anchor = np.nanmean(innov_0)

    ts_data = {"innovation": [innov_anchor], "increment": [incr_anchor], "ana_truth": [anchor_val]}
    frames  = {"innovation": [], "increment": [], "ana_truth": []}

    for i, cycle_k in enumerate(cycles):
        cycle_file = files[i]
        truth_file = ref_path / "snapshots" / f"reference_solution_{cycle_k}.nc"

        xb = _read_nc_field(cycle_file, var, LEVEL_IDX, prefix="xb_mean")
        xa = _read_nc_field(cycle_file, var, LEVEL_IDX, prefix="xa_mean")
        xt = _read_nc_field(truth_file, var, LEVEL_IDX, prefix=None)
        obs_val = _read_nc_field(cycle_file, var, LEVEL_IDX, prefix="obs")

        if var == "WDG1":
            innov = _standard_obs_innovation(obs_val, xb, var)
            incr  = _circ_diff(xa, xb)
            err   = np.abs(_circ_diff(xa, xt))
        else:
            innov = _standard_obs_innovation(obs_val, xb, var)
            incr  = xa - xb
            err   = np.abs(xa - xt)

        frames["innovation"].append(innov)
        frames["increment"].append(incr)
        frames["ana_truth"].append(err)

        ts_data["innovation"].append(np.nanmean(innov))
        ts_data["increment"].append(np.nanmean(incr))
        ts_data["ana_truth"].append(np.nanmean(err))

    return frames, ts_data


# ─────────────────────────────────────────────────────────────
#  SINGLE-EXPERIMENT MODE
# ─────────────────────────────────────────────────────────────
def _run_single(exp_path, ref_path, out_dir):
    files = _find_cycle_files(exp_path)
    if not files:
        print("No cycle files found!"); return
    files.sort(key=_extract_cycle_num)
    cycles = [_extract_cycle_num(f) for f in files]
    print(f"Found {len(cycles)} cycles: {cycles}")

    diff_labels = {
        "innovation": "Innovation (Obs - Background)",
        "increment":  "Increment (Analysis - Background)",
        "ana_truth":  "Analysis Error |Analysis - Truth|",
    }

    for var in VARS:
        print(f"\n=== Processing {var} ===")
        frames, ts_data = _collect_experiment(exp_path, ref_path, var, cycles, files)

        disp_cycles_all  = np.array([0] + [c + 1 for c in cycles])
        disp_cycles_post = np.array([c + 1 for c in cycles])

        # ── Time Series ──
        fig, ax = plt.subplots(figsize=(8, 4))
        # Optional: uncomment the line below to show estimated innovation. 
        # ax.plot(disp_cycles_all, ts_data["innovation"], label="Innovation (obs-bkg)", marker='.', linestyle=':', alpha=0.7)
        ax.plot(disp_cycles_all, ts_data["increment"],  label="Increment (ana-bkg)",  marker='.', linestyle='--', alpha=0.7)
        ax.plot(disp_cycles_all,  ts_data["ana_truth"],  label="|Ana - Truth|",        marker='.', linestyle='-', color='red')
        ax.set_title(f"{var} - Spatially Averaged Diagnostics")
        ax.set_xlabel("Cycle"); ax.set_ylabel(f"Mean {UNITS.get(var, '')}")
        ax.grid(True, alpha=0.3); ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(out_dir / f"{var}_spatial_means.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved Time Series")

        # ── Heatmap GIFs ──
        # 'innovation' is omitted because its mathematical approximation is misleading without explicit Normalizers
        for mode in ["increment", "ana_truth"]:
            _make_single_gif(frames[mode], cycles, var, mode, diff_labels[mode], out_dir)


def _make_single_gif(frame_list, cycles, var, mode, title_label, out_dir):
    data_stack = np.array(frame_list)
    if np.all(np.isnan(data_stack)):
        print(f"  Skipping {mode} GIF for {var} (All NaNs)"); return

    norm, cmap = _build_norm_cmap(data_stack, mode)

    fig = plt.figure(figsize=(7.2, 4.1))
    ax = plt.axes(projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.05, top=0.91)
    ax.coastlines(linewidth=0.7, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')
    im = ax.imshow(data_stack[0], origin='lower', extent=[0, 360, -90, 90],
                   cmap=cmap, norm=norm, interpolation='nearest',
                   transform=ccrs.PlateCarree())
    plt.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    title_text = ax.set_title(f"{var} {title_label} - Cycle {cycles[0]}", pad=8)

    gif_path = out_dir / f"{var}_{mode}.gif"
    writer = PillowWriter(fps=2)
    try:
        with writer.saving(fig, gif_path, dpi=100):
            for i in range(len(cycles)):
                im.set_data(data_stack[i])
                title_text.set_text(f"{var} {title_label} - Cycle {cycles[i]}")
                writer.grab_frame()
        print(f"  Saved GIF: {gif_path}")
    except Exception as e:
        print(f"  Error saving GIF {gif_path}: {e}")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  DUAL-EXPERIMENT MODE
# ─────────────────────────────────────────────────────────────
def _run_dual(exp_path_1, exp_path_2, ref_path, out_dir, label_1, label_2):
    files_1 = _find_cycle_files(exp_path_1)
    files_2 = _find_cycle_files(exp_path_2)
    if not files_1 or not files_2:
        print("Cycle files missing in one or both experiments!"); return

    files_1.sort(key=_extract_cycle_num);  files_2.sort(key=_extract_cycle_num)
    cycles_1 = [_extract_cycle_num(f) for f in files_1]
    cycles_2 = [_extract_cycle_num(f) for f in files_2]

    # Use intersection of cycles so frames are aligned
    common = sorted(set(cycles_1) & set(cycles_2))
    if not common:
        print("No common cycles between the two experiments!"); return
    print(f"Common cycles ({len(common)}): {common}")

    # Filter files/cycles to common set
    idx_1 = [cycles_1.index(c) for c in common]
    idx_2 = [cycles_2.index(c) for c in common]
    files_1 = [files_1[i] for i in idx_1]
    files_2 = [files_2[i] for i in idx_2]
    cycles  = common

    diff_labels = {
        "innovation": "Innovation (Obs - Bkg)",
        "increment":  "Increment (Ana - Bkg)",
        "ana_truth":  "Analysis Error |Ana - Truth|",
    }

    for var in VARS:
        print(f"\n=== Processing {var} ===")
        frames_1, ts_1 = _collect_experiment(exp_path_1, ref_path, var, cycles, files_1)
        frames_2, ts_2 = _collect_experiment(exp_path_2, ref_path, var, cycles, files_2)

        disp_all  = np.array([0] + [c + 1 for c in cycles])
        disp_post = np.array([c + 1 for c in cycles])

        # ── Stacked Time-Series ──
        _make_dual_timeseries(ts_1, ts_2, disp_all, disp_post, var, label_1, label_2, out_dir)

        # ── Side-by-side Heatmap GIFs ──
        # 'innovation' is omitted because its mathematical approximation is misleading without explicit Normalizers
        for mode in ["increment", "ana_truth"]:
            _make_dual_gif(frames_1[mode], frames_2[mode], cycles, var, mode,
                           diff_labels[mode], label_1, label_2, out_dir)


def _make_dual_timeseries(ts_1, ts_2, disp_all, disp_post, var, label_1, label_2, out_dir):
    """Two-panel stacked time-series (shared x-axis)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                    gridspec_kw={"hspace": 0.12})

    palette = {"innov": "#3b82f6", "incr": "#22c55e", "err": "#ef4444"}

    for ax, ts, lab in [(ax1, ts_1, label_1), (ax2, ts_2, label_2)]:
        # Optional: uncomment the line below to show estimated innovation
        # ax.plot(disp_all, ts["innovation"], label="Innovation", marker='.', linestyle=':', alpha=0.8, color=palette["innov"], linewidth=1.4)
        ax.plot(disp_all, ts["increment"],  label="Increment",  marker='.', linestyle='--', alpha=0.8, color=palette["incr"],  linewidth=1.4)
        ax.plot(disp_all,  ts["ana_truth"],  label="|Ana−Truth|", marker='.', linestyle='-', color=palette["err"], linewidth=1.6)
        ax.set_ylabel(f"Mean {UNITS.get(var, '')}", fontsize=10)
        ax.set_title(lab, fontsize=11, fontweight="bold", loc="left")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.set_xlabel("Cycle", fontsize=10)
    fig.suptitle(f"{var} — Spatially Averaged Diagnostics", fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(out_dir / f"{var}_spatial_means_dual.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved Dual Time Series")


def _make_dual_gif(frames_1, frames_2, cycles, var, mode, title_label,
                   label_1, label_2, out_dir):
    """Vertically stacked GIF with a shared colorbar and tight margins."""
    stack_1 = np.array(frames_1)
    stack_2 = np.array(frames_2)
    combined = np.concatenate([stack_1, stack_2], axis=0)

    if np.all(np.isnan(combined)):
        print(f"  Skipping {mode} dual GIF for {var} (All NaNs)"); return

    norm, cmap = _build_norm_cmap(combined, mode)

    proj = ccrs.PlateCarree()
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.4, 7.6), subplot_kw={"projection": proj},
        gridspec_kw={"hspace": 0.14}
    )
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.04, top=0.92)

    for ax in (ax1, ax2):
        ax.coastlines(linewidth=0.6, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor='gray')

    im1 = ax1.imshow(stack_1[0], origin='lower', extent=[0, 360, -90, 90],
                     cmap=cmap, norm=norm, interpolation='nearest', transform=proj)
    im2 = ax2.imshow(stack_2[0], origin='lower', extent=[0, 360, -90, 90],
                     cmap=cmap, norm=norm, interpolation='nearest', transform=proj)

    t1 = ax1.set_title(f"{label_1}", fontsize=13, fontweight="bold", loc="center", pad=8)
    t2 = ax2.set_title(f"{label_2}", fontsize=13, fontweight="bold", loc="center", pad=8)

    # Shared colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="vertical",
                        fraction=0.028, pad=0.015)
    cbar.set_label(UNITS.get(var, ''), fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    unit_str = f" [{UNITS.get(var, '')}]" if var in UNITS else ""
    suptitle = fig.suptitle(f"{var}  {title_label}{unit_str} — Cycle {cycles[0]}",
                            fontsize=14, fontweight="bold", y=0.98)

    gif_path = out_dir / f"{var}_{mode}_dual.gif"
    writer = PillowWriter(fps=2)
    try:
        with writer.saving(fig, gif_path, dpi=100):
            for i in range(len(cycles)):
                im1.set_data(stack_1[i])
                im2.set_data(stack_2[i])
                suptitle.set_text(f"{var}  {title_label}{unit_str} — Cycle {cycles[i]}")
                writer.grab_frame()
        print(f"  Saved Dual GIF: {gif_path}")
    except Exception as e:
        print(f"  Error saving dual GIF {gif_path}: {e}")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  Shared utilities
# ─────────────────────────────────────────────────────────────
def _build_norm_cmap(data_stack, mode):
    """Return (norm, cmap) for a given data stack and mode."""
    if mode == "ana_truth":
        vmax = np.nanmax(data_stack) if np.any(np.isfinite(data_stack)) else 1.0
        vmin = 0
        linthresh = vmax * 0.01 if vmax > 0 else 0.1
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        cmap = "viridis"
    else:
        abs_max = np.nanmax(np.abs(data_stack)) if np.any(np.isfinite(data_stack)) else 1.0
        vmin, vmax = -abs_max, abs_max
        linthresh = abs_max * 0.01 if abs_max > 0 else 0.1
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        cmap = "coolwarm"
    return norm, cmap


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
def main():
    exp_path = Path(EXP_DIR).resolve()
    ref_path = Path(REFERENCE_DIR).resolve()
    out_dir  = exp_path / OUT_DIR_NAME
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"Experiment 1 : {exp_path}")
    print(f"Reference    : {ref_path}")
    print(f"Output       : {out_dir}")

    if DUAL_MODE:
        if not EXP_DIR_2:
            print("ERROR: DUAL_MODE is True but EXP_DIR_2 is empty.")
            return
        exp_path_2 = Path(EXP_DIR_2).resolve()
        print(f"Experiment 2 : {exp_path_2}")
        _run_dual(exp_path, exp_path_2, ref_path, out_dir, EXP_LABEL_1, EXP_LABEL_2)
    else:
        _run_single(exp_path, ref_path, out_dir)


if __name__ == "__main__":
    main()
