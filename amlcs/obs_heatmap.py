#!/usr/bin/env python3
"""
obs_heatmap.py
---------------
Usage:
    python obs_heatmap.py /path/to/TG1_lev7_cycle0.csv [lat] [lon]

Reads the unified CSV produced by your AMLCS ReverseSDE run and plots a heatmap
showing *where* observations exist on the grid. Designed for TG1@level7, but it
works for any (var,level) CSV in the same format with columns like:

  idx, xb_mean, xa_mean, truth, noda, obs, sigma, is_obs

Assumptions:
- The "idx" column is 0..(lat*lon-1) with row-major order (C-order).
- If lat/lon are not provided, the script will guess common SPEEDY T21 dims (32x64)
  if N == 2048, or fall back to a square-ish factorization.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def infer_lat_lon(n: int):
    # Common SPEEDY grids
    common = {
        32*64: (32, 64),     # T21
        48*96: (48, 96),     # T30-ish variants
        64*128: (64, 128),   # T42-ish
        96*192: (96, 192),   # T63-ish
    }
    if n in common:
        return common[n]

    # Otherwise try to find a factorization close to 2:1 (lat:lon ~ 1:2)
    best = None
    for lat in range(2, int(np.sqrt(n)) + 2):
        if n % lat == 0:
            lon = n // lat
            ratio = lon / lat
            # Prefer lon >= lat, ratio near 2, but just pick a reasonable factor
            score = abs(ratio - 2.0)
            if best is None or score < best[0]:
                best = (score, lat, lon)
    if best is not None:
        _, lat, lon = best
        return lat, lon

    # Fallback (1 x n)
    return 1, n

def make_heatmap(csv_path: str, lat: int = None, lon: int = None):
    df = pd.read_csv(csv_path)
    if "is_obs" not in df.columns:
        raise ValueError("CSV is missing 'is_obs' column.")
    n = len(df["is_obs"])
    if lat is None or lon is None:
        lat_i, lon_i = infer_lat_lon(n)
    else:
        lat_i, lon_i = int(lat), int(lon)

    if lat_i * lon_i != n:
        raise ValueError(f"lat*lon != number of rows in CSV: {lat_i}*{lon_i} != {n}")

    # 0/1 mask of observed locations
    obs_mask = df["is_obs"].to_numpy(dtype=float).reshape(lat_i, lon_i)
    '''
    # After drawing obs_mask:
    obs_values = df["obs"].to_numpy().reshape(lat_i, lon_i)
    plt.imshow(obs_values, cmap="coolwarm")
    plt.title("Observation values where available")
    plt.colorbar(label="obs (TG1 units)")
    '''
    # Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(obs_mask, origin="upper", interpolation="nearest")
    ax.set_title(f"Observed grid cells (1=obs, 0=none)\n{os.path.basename(csv_path)}  |  {lat_i}x{lon_i}")
    ax.set_xlabel("longitude index (0..{})".format(lon_i-1))
    ax.set_ylabel("latitude index (0..{})".format(lat_i-1))
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("is_obs")

    # Summary text
    count = int(np.count_nonzero(obs_mask))
    pct = 100.0 * count / n if n > 0 else 0.0
    ax.text(0.01, -0.12, f"Observed points: {count}/{n} ({pct:.2f}%)",
            transform=ax.transAxes, ha="left", va="top")

    out_png = os.path.splitext(csv_path)[0] + "_obs_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[OK] Wrote heatmap -> {out_png}")
    print(f"[INFO] Observed points: {count}/{n} ({pct:.2f}%)")
    return out_png

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    csv_path = sys.argv[1]
    lat = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    lon = int(sys.argv[3]) if len(sys.argv) >= 4 else None
    make_heatmap(csv_path, lat, lon)

if __name__ == "__main__":
    main()
