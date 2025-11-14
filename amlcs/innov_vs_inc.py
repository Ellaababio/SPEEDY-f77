#!/usr/bin/env python3
"""
Compare 1D spatially averaged diagnostics between ReverseSDE and EnKF_MC_obs.

For each <VAR>1 variable (UG1, VG1, TG1, TRG1, PSG1), this script:
  - Reads unified CSVs from both runs
  - Computes spatial mean per cycle for:
      * Innovation  = obs - xb_mean
      * Increment   = xa_mean - xb_mean
      * Ana error   = |xa_mean - truth|
  - Produces 1 figure per variable with 2 subplots (ReverseSDE vs EnKF_MC_obs)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

# Directory containing unified CSVs for each run
REV_DIR  = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100"    # TODO: set this
ENKF_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_EnKF_MC_obs_1_1_100"   # TODO: set this

OUT_DIR = "./comparison_1d_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Variables and levels (surface-like level for each *1 variable)
VAR_LIST = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]
LEVEL_TAG_FOR = {
    "UG1":  "lev7",
    "VG1":  "lev7",
    "TG1":  "lev7",
    "TRG1": "lev7",
    "PSG1": "lev0",   # surface pressure
}

# Optional units for labels (just for y-axis, not strictly required)
UNITS = {
    "UG1":  "m/s",
    "VG1":  "m/s",
    "TG1":  "K",
    "TRG1": "g/kg",
    "PSG1": "log(ps/p0)",
}

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def discover_cycles(dir_path):
    """Return sorted list of unique cycle indices found in CSV filenames."""
    pat = re.compile(r"cycle(\d+)")
    cycles = set()
    for f in os.listdir(dir_path):
        if not f.endswith(".csv"):
            continue
        m = pat.search(f)
        if m:
            cycles.add(int(m.group(1)))
    return sorted(cycles)


def load_field(dir_path, var, comp_key, cycle):
    """
    Load one component from a unified CSV:
      comp_key in {"bkg", "ana", "truth", "obs"}.
    Returns a 1D numpy array of length n_grid.
    """
    lev_tag = LEVEL_TAG_FOR[var]
    fn = os.path.join(dir_path, f"{var}_{lev_tag}_cycle{cycle}.csv")
    df = pd.read_csv(fn)

    col_map = {
        "bkg":   "xb_mean",
        "ana":   "xa_mean",
        "truth": "truth",
        "obs":   "obs",
    }
    col = col_map[comp_key]
    arr = df[col].to_numpy().astype(float)
    return arr  # flattened already


def compute_diff(dir_path, var, diff_type, cycle):
    """
    diff_type in {"innovation", "increment", "ana_truth"}.
    """
    if diff_type == "innovation":
        obs = load_field(dir_path, var, "obs",   cycle)
        bkg = load_field(dir_path, var, "bkg",   cycle)
        return obs - bkg            # NaNs where no obs -> excluded in nanmean
    elif diff_type == "increment":
        ana = load_field(dir_path, var, "ana",   cycle)
        bkg = load_field(dir_path, var, "bkg",   cycle)
        return ana - bkg
    elif diff_type == "ana_truth":
        ana   = load_field(dir_path, var, "ana",   cycle)
        truth = load_field(dir_path, var, "truth", cycle)
        return np.abs(ana - truth)  # magnitude only
    else:
        raise ValueError(f"Unknown diff_type: {diff_type}")


def spatial_mean_series(dir_path, var, diff_type, cycles):
    """Spatial mean (nanmean) over grid for each cycle."""
    vals = []
    for c in cycles:
        D = compute_diff(dir_path, var, diff_type, c)
        vals.append(float(np.nanmean(D)))
    return np.array(vals)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

# Cycles common to both runs
cycles_rev  = set(discover_cycles(REV_DIR))
cycles_enkf = set(discover_cycles(ENKF_DIR))
cycles = sorted(cycles_rev & cycles_enkf)
if not cycles:
    raise RuntimeError("No overlapping cycles found between runs.")

# Display cycles as 1..N instead of 0..N-1
disp_x = np.arange(1, len(cycles) + 1)

for var in VAR_LIST:
    # ReverseSDE
    inn_rev = spatial_mean_series(REV_DIR,  var, "innovation", cycles)
    inc_rev = spatial_mean_series(REV_DIR,  var, "increment",  cycles)
    err_rev = spatial_mean_series(REV_DIR,  var, "ana_truth",  cycles)

    # EnKF_MC_obs
    inn_enk = spatial_mean_series(ENKF_DIR, var, "innovation", cycles)
    inc_enk = spatial_mean_series(ENKF_DIR, var, "increment",  cycles)
    err_enk = spatial_mean_series(ENKF_DIR, var, "ana_truth",  cycles)

    unit = UNITS.get(var, "")

    # Figure: 2 subplots side by side, shared y-axis
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10, 4), sharey=True, constrained_layout=True
    )

    # ReverseSDE subplot
    ax1.plot(disp_x, inn_rev, label="Innovation (obs - bkg)")
    ax1.plot(disp_x, inc_rev, label="Increment (ana - bkg)")
    ax1.plot(disp_x, err_rev, label="Analysis error |ana - truth|")
    ax1.set_title(f"{var} -- ReverseSDE")
    ax1.set_xlabel("Cycle")
    if unit:
        ax1.set_ylabel(f"Spatial mean [{unit}]")
    else:
        ax1.set_ylabel("Spatial mean")
    ax1.grid(alpha=0.3)
    ax1.set_xticks(disp_x)

    # EnKF_MC_obs subplot
    ax2.plot(disp_x, inn_enk, label="Innovation (obs - bkg)")
    ax2.plot(disp_x, inc_enk, label="Increment (ana - bkg)")
    ax2.plot(disp_x, err_enk, label="Analysis error |ana - truth|")
    ax2.set_title(f"{var} -- EnKF_MC_obs")
    ax2.set_xlabel("Cycle")
    ax2.grid(alpha=0.3)
    ax2.set_xticks(disp_x)

    # Put a legend only on the right subplot to avoid duplication
    ax2.legend(loc="best", fontsize=8)

    out_fn = os.path.join(OUT_DIR, f"{var}_ReverseSDE_vs_EnKF_MC_obs.png")
    plt.savefig(out_fn, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_fn}")

print("Done.")
