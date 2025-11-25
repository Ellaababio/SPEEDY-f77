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
NC_PATH = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/sde_tracking.nc"

# Output directory for plots
OUT_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/sde_spaghetti_plots_gridpoint"

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
# MAIN
# --------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Opening:", NC_PATH)
    try:
        nc = Dataset(NC_PATH, "r")
    except FileNotFoundError:
        print(f"Error: File not found at {NC_PATH}")
        print("Please update the NC_PATH in the CONFIGURATION section.")
        return

    xt = nc["xt_state"]        # (ncycle, block, psteps, var, ens)
    raw_varnames = nc["var_names"][:]
    var_names = decode_var_names(raw_varnames)

    ncycles = xt.shape[0]
    nblocks = xt.shape[1]
    psteps  = xt.shape[2]
    nvars   = xt.shape[3]
    nens    = xt.shape[4]

    print(f"Cycles={ncycles} Blocks={nblocks} psteps={psteps} vars={nvars} ens={nens}")

    FILL = getattr(xt, "_FillValue", None)

    # --------------------------------------------------------
    # VISUAL cycles 1–5 → internal cycles 0–4
    # --------------------------------------------------------
    for visual_cycle in range(1, ncycles + 1):
        internal_k = visual_cycle - 1

        print(f"\n=== Cycle {visual_cycle} (internal index {internal_k}) ===")

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
        # Count finite values per block to find which ones have data
        # shape: (blocks, psteps, vars, ens) -> sum over (1,2,3) gives (blocks,)
        finite_counts = np.sum(np.isfinite(cycle_data), axis=(1, 2, 3))
        
        # Threshold: at least some data. 
        # For gridpoint tracking, only specific blocks will have data.
        # We can be lenient with the threshold to ensure we catch it.
        min_valid = 1 
        valid_blocks = np.where(finite_counts >= min_valid)[0]

        print("Valid blocks:", valid_blocks.tolist())

        if len(valid_blocks) == 0:
            print("No valid blocks → skipping cycle")
            continue

        # Average across blocks
        # If tracking a single gridpoint, valid_blocks should ideally be length 1 (or small)
        # Taking the mean collapses the block dimension.
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
            outpath = os.path.join(OUT_DIR, f"cycle{visual_cycle}_{var}.png")
            plt.savefig(outpath, dpi=150)
            plt.close()

            print("Saved:", outpath)

    nc.close()
    print("\nAll spaghetti plots generated.")

if __name__ == "__main__":
    main()
