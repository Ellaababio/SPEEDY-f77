#!/usr/bin/env python3
import numpy as np
from netCDF4 import Dataset
import os

nc_path = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/sde_tracking.nc"

nc = Dataset(nc_path, "r")
xt = nc["xt_state"]   # (cycle, block, psteps, var, ens)
ncycles, nblocks, psteps, nvars, nens = xt.shape

print("Loaded xt_state shape =", xt.shape)

# ===========================
# TEST 1 — print first few values from file directly
# ===========================
print("\n=== TEST 1: Direct slice from file ===")
arr = xt[0, 58, :, 0, :]   # cycle0, block58, psteps, UG1
print("shape =", arr.shape)
print("finite count =", np.isfinite(arr).sum())
print("first row sample =", arr[0, :10])

# ===========================
# TEST 2 — apply plotting logic EXACTLY
# ===========================
print("\n=== TEST 2: After averaging ===")
cycle_data = np.array(xt[0])       # shape (block, psteps, var, ens)
block_data = cycle_data[:, :, 0, :]  # UG1 variable
block_valid = np.isfinite(block_data).any(axis=(1,2))
valid_blocks = np.where(block_valid)[0]

print("valid_blocks =", valid_blocks)

M = np.nanmean(block_data[valid_blocks], axis=0)   # expected (psteps, ens)
print("M.shape =", M.shape)
print("M finite count =", np.isfinite(M).sum())
print("M first row sample =", M[0, :10])

# ===========================
# TEST 3 — check for nonsense values
# ===========================
print("\n=== TEST 3: Range of M ===")
if np.isfinite(M).any():
    print("min =", np.nanmin(M))
    print("max =", np.nanmax(M))
    print("mean =", np.nanmean(M))
else:
    print("M contains NO finite values!!")

finite_counts = np.sum(np.isfinite(block_data), axis=(1,2))
print("finite_counts:", finite_counts)
block_valid = finite_counts > 10
print("valid_blocks:", np.where(block_valid)[0])
