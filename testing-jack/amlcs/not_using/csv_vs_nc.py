#!/usr/bin/env python3
"""
Verify that ReverseSDE unified CSVs match their new NetCDF equivalents
for level 7 (UG1, VG1, TG1, TRG1) and level 0 (PSG1).

Directory:
    /gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100

Checks fields:
    xb_mean, xa_mean, truth, obs, sigma, is_obs

Outputs:
    OK     → arrays match to numerical tolerance
    DIFF   → arrays differ → NetCDF writing OR indexing error in sequential_methods
"""

import os
import re
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# -----------------------------
# DIRECTORY (NO CHANGES NEEDED)
# -----------------------------
BASE = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100"
CSV_DIR = BASE        # CSVs live here
NC_DIR  = BASE        # NetCDFs live here

# -----------------------------
# VARIABLE SETTINGS
# -----------------------------
VARS = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]
LEV_TAG = {
    "UG1":  "lev7",
    "VG1":  "lev7",
    "TG1":  "lev7",
    "TRG1": "lev7",
    "PSG1": "lev0",
}

FIELDS = ["xb_mean", "xa_mean", "truth", "obs", "sigma", "is_obs"]

# -----------------------------
# HELPERS
# -----------------------------
def discover_cycles_csv(directory):
    pat = re.compile(r"cycle(\d+)\.csv$")
    cycles = []
    for fn in os.listdir(directory):
        if fn.endswith(".csv"):
            m = pat.search(fn)
            if m:
                cycles.append(int(m.group(1)))
    return sorted(set(cycles))


def discover_cycles_nc(directory):
    pat = re.compile(r"reverseSDE_cycle(\d+)\.nc$")
    cycles = []
    for fn in os.listdir(directory):
        m = pat.search(fn)
        if m:
            cycles.append(int(m.group(1)))
    return sorted(set(cycles))


def load_nc_field(nc_path, var, lev_tag, field):
    key = f"{field}_{var}_{lev_tag}"
    with Dataset(nc_path, "r") as ds:
        if key not in ds.variables:
            print(f"    [!] Missing {key} in {nc_path}")
            return None
        return np.asarray(ds[key][:], float).ravel()


def compare_arrays(a, b):
    if a.shape != b.shape:
        return None, None
    diff = np.abs(a - b)
    max_diff = np.nanmax(diff)
    pct_diff = 100 * np.sum(diff > 1e-12) / diff.size
    return max_diff, pct_diff


# -----------------------------
# MAIN VERIFICATION
# -----------------------------
csv_cycles = discover_cycles_csv(CSV_DIR)
nc_cycles  = discover_cycles_nc(NC_DIR)
cycles = sorted(set(csv_cycles) & set(nc_cycles))

if not cycles:
    raise RuntimeError("No overlapping cycles found between CSV and NC.")

print("Checking cycles:", cycles)

for c in cycles:
    print("\n====================================")
    print(f"Cycle {c}")
    print("====================================")

    nc_path = os.path.join(NC_DIR, f"reverseSDE_cycle{c}.nc")

    for var in VARS:
        lev = LEV_TAG[var]
        csv_path = os.path.join(CSV_DIR, f"{var}_{lev}_cycle{c}.csv")

        print(f"\n  -- {var} ({lev}) --")

        if not os.path.exists(csv_path):
            print(f"    [!] Missing CSV: {csv_path}")
            continue

        if not os.path.exists(nc_path):
            print(f"    [!] Missing NC: {nc_path}")
            continue

        df = pd.read_csv(csv_path)

        for fld in FIELDS:
            if fld not in df.columns:
                print(f"    [!] Missing '{fld}' in CSV")
                continue

            a_csv = df[fld].to_numpy(float)
            a_nc  = load_nc_field(nc_path, var, lev, fld)

            if a_nc is None:
                continue

            maxd, pct = compare_arrays(a_csv, a_nc)

            if maxd is None:
                print(f"    {fld}: shape mismatch CSV={a_csv.shape}, NC={a_nc.shape}")
                continue

            status = "OK" if maxd < 1e-12 else "DIFF"
            print(f"    {fld:7s}: {status:4s} | max diff={maxd:.3e} | %diff={pct:.2f}%")
