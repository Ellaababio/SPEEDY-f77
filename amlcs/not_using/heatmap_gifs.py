#!/usr/bin/env python3
# ============================================================
# Heatmaps + Time-Series (AMLCS unified CSV schema)
# - Variables: UG1, VG1, TG1, TRG1, PSG1
# - Files: {var}_{level_tag}_cycle{N}.csv
# - Columns: idx, xb_mean, xa_mean, truth, noda, obs, sigma, is_obs
# - PSG1 kept in log-space; TRG1 converted to g/kg
# - GIFs: innovations (obs-bkg), increments (ana-bkg), |ana-truth|; 1 fps, coastlines, colorbar (no label)
# - Time series: spatial mean; units shown on y-axis; x-axis integer cycles 1..N
# ============================================================

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -------------------- USER SETTINGS --------------------
csv_dir = "../runs/t21_50_0.05_5_ReverseSDE_1_1_100/"  # <-- adjust if needed (no "results" subfolder)
save_dir = os.path.join(csv_dir, "heatmaps")
os.makedirs(save_dir, exist_ok=True)

variables = ["UG1", "VG1", "TG1", "TRG1", "PSG1"]
level_tag_for = {"PSG1": "lev0", "UG1": "lev7", "VG1": "lev7", "TG1": "lev7", "TRG1": "lev7"}

# T21 grid
nlat, nlon = 32, 64

# Exact CSV column names
COL = {
    "bkg":   "xb_mean",
    "ana":   "xa_mean",
    "truth": "truth",
    "noda":  "noda",
    "obs":   "obs",
    "sigma": "sigma",
}

# Human labels & colormaps for differences
diff_labels = {
    "innovation": "Innovation (obs - bkg)",
    "increment":  "Increment (ana - bkg)",
    "ana_truth":  "Analysis Error |ana - truth|",
}
diff_cmaps = {
    "innovation": "coolwarm",
    "increment":  "coolwarm",
    "ana_truth":  "viridis",
}

# Units to show ONLY on time-series plots
UNITS = {"UG1": "m/s", "VG1": "m/s", "TG1": "K", "TRG1": "g/kg", "PSG1": "log(ps/p0)"}


# -------------------- CYCLE DISCOVERY --------------------
def discover_cycles():
    """
    Find available numeric cycle indices from filenames; we will DISPLAY cycles as 1..N (index-based),
    regardless of zero-based file numbering.
    """
    pat = re.compile(r"cycle(\d+)")
    found = set()
    for f in os.listdir(csv_dir):
        if not f.endswith(".csv"):
            continue
        # only consider files for our variables and their level tags
        ok = False
        for v in variables:
            if f.startswith(v + "_") and level_tag_for[v] in f:
                ok = True
                break
        if not ok:
            continue
        m = pat.search(f)
        if m:
            found.add(int(m.group(1)))
    out = sorted(found)
    if not out:
        raise RuntimeError(f"No CSV cycles found in {csv_dir}.")
    return out

cycles = discover_cycles()             # e.g., [0,1,2,3,4]
disp_x = np.arange(1, len(cycles)+1)   # display as 1..N (integers only)


# -------------------- HELPERS --------------------
def filename_for(var: str, cycle: int) -> str:
    lvl = level_tag_for[var]
    fn = f"{var}_{lvl}_cycle{cycle}.csv"
    path = os.path.join(csv_dir, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return path

def load_field(var: str, comp_key: str, cycle: int) -> np.ndarray:
    """
    Load field (bkg/ana/truth/obs/etc.) and reshape to (nlat, nlon).
    - PSG1 stays in log-space (no conversion)
    """
    fn = filename_for(var, cycle)
    df = pd.read_csv(fn)
    if "idx" in df.columns:
        df = df.sort_values("idx", kind="mergesort")
    col = COL[comp_key]
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not in {fn}. Got columns: {list(df.columns)}")
    arr = df[col].to_numpy().astype(float)
    # PSG1 remains in log-space by design

    if arr.size != nlat * nlon:
        raise ValueError(f"Grid size mismatch in {fn}: {arr.size} != {nlat*nlon}")
    return arr.reshape((nlat, nlon))

def compute_diff(var: str, cycle: int, diff_type: str) -> np.ndarray:
    """innovation = obs - bkg; increment = ana - bkg; ana_truth = |ana - truth|"""
    if diff_type == "innovation":
        return load_field(var, "obs", cycle) - load_field(var, "bkg", cycle)
    elif diff_type == "increment":
        return load_field(var, "ana", cycle) - load_field(var, "bkg", cycle)
    elif diff_type == "ana_truth":
        return np.abs(load_field(var, "ana", cycle) - load_field(var, "truth", cycle))
    else:
        raise ValueError(f"Unknown diff_type: {diff_type}")

def spatial_mean_series(var: str, diff_type: str) -> np.ndarray:
    vals = []
    for c in cycles:
        D = compute_diff(var, c, diff_type)
        vals.append(float(np.nanmean(D)))
    return np.array(vals)


# -------------------- GIF MAKER (1 fps, single colorbar, no units/legend) --------------------
def make_gif(var: str, diff_type: str):
    # Determine fixed color scale across frames
    vmax_ref = 0.0
    for c in cycles:
        D = compute_diff(var, c, diff_type)
        if np.isfinite(D).any():
            vmax_ref = max(vmax_ref, float(np.nanmax(np.abs(D))))

    abs_metric = (diff_type == "ana_truth")
    cmap = diff_cmaps[diff_type]
    if abs_metric:
        norm = mcolors.Normalize(vmin=0.0, vmax=(vmax_ref or 1e-12))
    else:
        vmax = (vmax_ref or 1e-12)
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    # One figure/axes/colorbar
    fig = plt.figure(figsize=(7.6, 4.0))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_global()

    # First frame
    D0 = compute_diff(var, cycles[0], diff_type)
    im = ax.imshow(
        D0, origin="lower",
        extent=[0, 360, -90, 90],
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, interpolation="nearest",
    )

    # Single persistent colorbar (no label/units on GIFs)
    fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, fraction=0.05)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{var}_{diff_type}.gif")
    writer = PillowWriter(fps=1)  # exactly 1 frame/second

    # Save frames with strict 1 fps; DISPLAY cycle as 1..N (index-based)
    with writer.saving(fig, out_path, dpi=100):
        for i, c in enumerate(cycles, start=1):
            D = compute_diff(var, c, diff_type)
            im.set_data(D)
            ax.set_title(f"{var} • {diff_labels[diff_type]} • cycle {i}")  # show 1..N
            writer.grab_frame()

    plt.close(fig)
    print(f"GIF saved: {out_path}")


# -------------------- MAIN --------------------
print(f"Saving outputs to: {save_dir}")

# 15 GIFs
for var in variables:
    print(f"\n=== {var} ===")
    for diff in ("innovation", "increment", "ana_truth"):
        make_gif(var, diff)

# 5 time-series (spatial mean): with units on y-axis, integer cycle ticks 1..N
for var in variables:
    inn = spatial_mean_series(var, "innovation")   # signed mean
    inc = spatial_mean_series(var, "increment")    # signed mean
    err = spatial_mean_series(var, "ana_truth")    # mean absolute

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(disp_x, inn, label="Innovation (obs - bkg)")
    ax.plot(disp_x, inc, label="Increment (ana - bkg)")
    ax.plot(disp_x, err, label="Analysis error |ana - truth|")
    ax.set_xlabel("Cycle")
    ax.set_ylabel(f"Spatial Mean [{UNITS[var]}]")   # units ONLY on 1-D plots
    ax.set_title(f"{var} • Spatially Averaged Diagnostics (Surface Level)")
    ax.grid(alpha=0.3)
    ax.legend()
    # integer-only ticks 1..N (no fractional ticks)
    ax.set_xlim(1, len(disp_x))
    ax.set_xticks(disp_x)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    out_png = os.path.join(save_dir, f"{var}_spatial_means.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Plot saved: {out_png}")

print("Done.")
