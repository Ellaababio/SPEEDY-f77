#!/usr/bin/env python3

import numpy as np
from netCDF4 import Dataset

# ------------------------------------------------------------------
# CHANGE THIS IF NEEDED
# ------------------------------------------------------------------
nc_path = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/sde_tracking.nc"

print(f"Opening NetCDF: {nc_path}")
nc = Dataset(nc_path, "r")

xt = nc["xt_state"]  # (cycle, block, psteps, var, ens)
ncycles, nblocks, psteps, nvars, nens = xt.shape

print(f"xt_state shape = {xt.shape}")
print(f"psteps = {psteps}")

FILL = getattr(xt, "_FillValue", None)

# last 20 timesteps
start = psteps - 20
end = psteps
print(f"Checking last 20 timesteps: indices {start} .. {end-1}")
print("Each line: global_step_idx: finite_values / total_values (over var×ens)\n")

problem_found = False

for c in range(ncycles):
    print(f"=== Cycle {c} ===")

    # ---- load cycle like spaghetti_plots does ----
    xk = xt[c][:]  # (block, psteps, var, ens)

    # masked -> NaN
    if isinstance(xk, np.ma.MaskedArray):
        xk = xk.filled(np.nan)
    cycle_data = xk.astype(np.float64)

    # remove fill values / crazy large sentinels
    if FILL is not None:
        bad = (cycle_data == FILL) | (np.abs(cycle_data) > 1e30)
        cycle_data[bad] = np.nan

    # ---- same block filtering as spaghetti_plots ----
    finite_counts_per_block = np.sum(np.isfinite(cycle_data), axis=(1, 2, 3))
    min_valid = max(5, int(0.05 * psteps * nvars * nens))
    valid_blocks = np.where(finite_counts_per_block >= min_valid)[0]

    print(f"  valid_blocks: {valid_blocks.tolist()}")

    if len(valid_blocks) == 0:
        print("  !! no valid blocks for this cycle, skipping")
        problem_found = True
        continue

    # average across valid blocks → (psteps, var, ens)
    M = np.nanmean(cycle_data[valid_blocks, :, :, :], axis=0)

    # ---- inspect last 20 psteps of M ----
    total_per_step = nvars * nens
    for t in range(start, end):
        slice_t = M[t, :, :]  # (var, ens)
        finite_here = np.sum(np.isfinite(slice_t))
        if finite_here < total_per_step:
            problem_found = True
            status = "BAD" if finite_here == 0 else "PARTIAL"
        else:
            status = "OK"
        print(f"  step {t:3d}: {finite_here:4d} / {total_per_step:4d} finite   [{status}]")

nc.close()

print("\n==============================")
if problem_found:
    print("FINAL RESULT: At least one cycle has missing/NaN values in the last 20 steps.")
else:
    print("FINAL RESULT: All cycles have full, finite data in the last 20 steps.")
print("==============================")
