#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # HPC-safe
import matplotlib.pyplot as plt
from netCDF4 import Dataset


# --------------------------------------------------------
# Units dictionary (YOUR PROVIDED UNITS)
# --------------------------------------------------------
UNITS = {
    "UG1":  "m/s",
    "VG1":  "m/s",
    "TG1":  "K",
    "TRG1": "g/kg",
    "PSG1": "log(ps/p0)",
}


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
# MAIN
# --------------------------------------------------------
def main():

    # --------------------------------------------------------
    # Hard-coded paths
    # --------------------------------------------------------
    nc_path = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/sde_tracking.nc"

    outdir = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/sde_spaghetti_plots_nonlinear_0.01_obs_error"
    os.makedirs(outdir, exist_ok=True)

    print("Opening:", nc_path)
    nc = Dataset(nc_path, "r")

    xt = nc["xt_state"]        # (ncycle, block, psteps, var, ens)
    raw_varnames = nc["var_names"][:]
    var_names = decode_var_names(raw_varnames)

    ncycles = xt.shape[0]      # likely 5 → indices 0..4
    nblocks = xt.shape[1]
    psteps  = xt.shape[2]
    # print(psteps)
    nvars   = xt.shape[3]
    nens    = xt.shape[4]

    print(f"Cycles={ncycles} Blocks={nblocks} psteps={psteps} vars={nvars} ens={nens}")

    FILL = getattr(xt, "_FillValue", None)

    # --------------------------------------------------------
    # VISUAL cycles 1–5 → internal cycles 0–4
    # --------------------------------------------------------
    for visual_cycle in range(1, 6):   # 1–5
        internal_k = visual_cycle - 1  # 0–4

        print(f"\n=== Cycle {visual_cycle} (internal index {internal_k}) ===")

        if internal_k >= ncycles:
            print(f"Internal cycle {internal_k} does not exist → skipping")
            continue

        # Load slice
        xk = xt[internal_k][:]

        # masked → NaN
        if isinstance(xk, np.ma.MaskedArray):
            xk = xk.filled(np.nan)

        cycle_data = xk.astype(np.float64)

        # remove fill values
        if FILL is not None:
            bad = (cycle_data == FILL) | (np.abs(cycle_data) > 1e30)
            cycle_data[bad] = np.nan

        # --------------------------------------------------------
        # Block filtering
        # --------------------------------------------------------
        finite_counts = np.sum(np.isfinite(cycle_data), axis=(1, 2, 3))
        min_valid = max(5, int(0.05 * psteps * nvars * nens))
        valid_blocks = np.where(finite_counts >= min_valid)[0]

        print("Valid blocks:", valid_blocks.tolist())

        if len(valid_blocks) == 0:
            print("No valid blocks → skipping cycle")
            continue

        # average across blocks
        M = np.nanmean(cycle_data[valid_blocks, :, :, :], axis=0)

        # --------------------------------------------------------
        # Plotting
        # --------------------------------------------------------
        for j, var in enumerate(var_names):

            unit = UNITS.get(var, "")       # default: blank
            unit_tag = f" ({unit})" if unit else ""
            ylabel = f"{var} [{unit}]" if unit else var

            plt.figure(figsize=(10, 4))

            # spaghetti = ensemble members
            for e in range(nens):
                plt.plot(M[:, j, e], alpha=0.4)

            # robust y-limits
            clean = M[:, j, :].reshape(-1)
            clean = clean[np.isfinite(clean)]
            if len(clean) > 20:
                lo, hi = np.nanpercentile(clean, [1, 99])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    plt.ylim(lo, hi)

            plt.title(f"Cycle {visual_cycle} -- {var}{unit_tag}")
            plt.xlabel("Pseudo-time")
            plt.ylabel(ylabel)
            plt.tight_layout()

            # filename uses visual cycle
            outpath = os.path.join(outdir, f"cycle{visual_cycle}_{var}.png")
            plt.savefig(outpath, dpi=150)
            plt.close()

            print("Saved:", outpath)

    nc.close()
    print("\nAll spaghetti plots generated with units.")


if __name__ == "__main__":
    main()
