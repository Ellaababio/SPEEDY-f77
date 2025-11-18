#!/usr/bin/env python3
"""
Compare 1D spatially averaged diagnostics between ReverseSDE and EnKF_MC_obs,
but using the new unified NetCDF output files instead of CSVs.

For each <VAR>1 variable (UG1, VG1, TG1, TRG1, PSG1), this script:
  - Reads reverseSDE_cycle<k>.nc  (Reverse SDE)
  - Reads unified_cycle<k>.nc     (EnKF_MC_obs)
  - Computes spatial mean per cycle for:
      * Innovation  = obs - xb_mean
      * Increment   = xa_mean - xb_mean
      * Ana error   = |xa_mean - truth|
  - Produces 1 figure per variable with 2 subplots (ReverseSDE vs EnKF_MC_obs)
"""

import os
import re
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# USER SETTINGS — UPDATE THESE PATHS
# ---------------------------------------------------------------------

REV_DIR  = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100"    # reverseSDE_cycle<k>.nc
ENKF_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_EnKF_MC_obs_1_1_100"   # unified_cycle<k>.nc

OUT_DIR = os.path.join(ENKF_DIR, "comparison_1d_plots_nc")
os.makedirs(OUT_DIR, exist_ok=True)

VAR_LIST = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]
LEVEL_TAG = {
    "UG1":  "lev7",
    "VG1":  "lev7",
    "TG1":  "lev7",
    "TRG1": "lev7",
    "PSG1": "lev0",
}

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

def discover_cycles_nc(directory, prefix):
    """
    Detect cycle numbers from files matching:
        <directory>/<prefix>_cycle<k>.nc
    """
    cycles = []
    pat = re.compile(r"cycle(\d+)\.nc$")
    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname.endswith(".nc"):
            m = pat.search(fname)
            if m:
                cycles.append(int(m.group(1)))
    return sorted(cycles)


def load_field_nc(nc_path, var, lev_tag, key):
    """
    Load a 2D (lat,lon) field from a unified NetCDF.
    key ∈ {"xb_mean", "xa_mean", "truth", "obs"}.
    """
    vname = f"{key}_{var}_{lev_tag}"
    with Dataset(nc_path, "r") as ds:
        if vname not in ds.variables:
            raise RuntimeError(f"Variable {vname} not found in {nc_path}")
        arr = ds[vname][:]  # shape (lat, lon)
        return np.asarray(arr, float).ravel()


def compute_diff_nc(directory, prefix, var, lev_tag, diff_type, cycle):
    """
    diff_type ∈ {"innovation", "increment", "ana_truth"}.
    """
    path = os.path.join(directory, f"{prefix}_cycle{cycle}.nc")

    if diff_type == "innovation":
        obs = load_field_nc(path, var, lev_tag, "obs")
        bkg = load_field_nc(path, var, lev_tag, "xb_mean")
        return obs - bkg

    elif diff_type == "increment":
        ana = load_field_nc(path, var, lev_tag, "xa_mean")
        bkg = load_field_nc(path, var, lev_tag, "xb_mean")
        return ana - bkg

    elif diff_type == "ana_truth":
        ana   = load_field_nc(path, var, lev_tag, "xa_mean")
        truth = load_field_nc(path, var, lev_tag, "truth")
        return np.abs(ana - truth)

    else:
        raise ValueError(diff_type)


def spatial_mean_series_nc(directory, prefix, var, lev_tag, diff_type, cycles):
    vals = []
    for c in cycles:
        D = compute_diff_nc(directory, prefix, var, lev_tag, diff_type, c)
        vals.append(float(np.nanmean(D)))
    return np.array(vals)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

cycles_rev  = set(discover_cycles_nc(REV_DIR,  "reverseSDE"))
cycles_enkf = set(discover_cycles_nc(ENKF_DIR, "unified"))
cycles = sorted(cycles_rev & cycles_enkf)

if not cycles:
    raise RuntimeError("No overlapping cycles found.")

disp_x = np.arange(1, len(cycles) + 1)

print("Using cycles:", cycles)
print("Displayed as Cycle 1..{}.\n".format(len(cycles)))

for var in VAR_LIST:

    lev_tag = LEVEL_TAG[var]

    # ReverseSDE diagnostics --------------------------------------------
    inn_rev = spatial_mean_series_nc(REV_DIR,  "reverseSDE", var, lev_tag, "innovation", cycles)
    inc_rev = spatial_mean_series_nc(REV_DIR,  "reverseSDE", var, lev_tag, "increment",  cycles)
    err_rev = spatial_mean_series_nc(REV_DIR,  "reverseSDE", var, lev_tag, "ana_truth",  cycles)

    # EnKF_MC_obs diagnostics ------------------------------------------
    inn_enk = spatial_mean_series_nc(ENKF_DIR, "unified", var, lev_tag, "innovation", cycles)
    inc_enk = spatial_mean_series_nc(ENKF_DIR, "unified", var, lev_tag, "increment",  cycles)
    err_enk = spatial_mean_series_nc(ENKF_DIR, "unified", var, lev_tag, "ana_truth",  cycles)

    # Debug prints for humidity
    if var == "TRG1":
        print("TRG1 diagnostics (spatial mean, level 7):")
        for i, c in enumerate(cycles):
            print(
                f"  Cycle {i+1}: "
                f"inn_rev={inn_rev[i]: .6e}, inn_enk={inn_enk[i]: .6e}, "
                f"inc_rev={inc_rev[i]: .6e}, inc_enk={inc_enk[i]: .6e}"
            )
        print()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    unit = UNITS.get(var, "")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)

    ax1.plot(disp_x, inn_rev, label="Innovation (obs - bkg)")
    ax1.plot(disp_x, inc_rev, label="Increment (ana - bkg)")
    ax1.plot(disp_x, err_rev, label="Analysis error |ana - truth|")
    ax1.set_title(f"{var} -- ReverseSDE")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel(f"Spatial mean [{unit}]" if unit else "Spatial mean")
    ax1.grid(alpha=0.3)
    ax1.set_xticks(disp_x)

    ax2.plot(disp_x, inn_enk, label="Innovation (obs - bkg)")
    ax2.plot(disp_x, inc_enk, label="Increment (ana - bkg)")
    ax2.plot(disp_x, err_enk, label="Analysis error |ana - truth|")
    ax2.set_title(f"{var} -- EnKF_MC_obs")
    ax2.set_xlabel("Cycle")
    ax2.grid(alpha=0.3)
    ax2.set_xticks(disp_x)

    ax2.legend(loc="best", fontsize=8)

    outfile = os.path.join(OUT_DIR, f"{var}_ReverseSDE_vs_EnKF_MC_obs.png")
    plt.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f"Saved: {outfile}")

print("Done.")
