#!/usr/bin/env python3
"""
plot_grid_errors_from_values_cartopy.py
---------------------------------------
Create animated heatmaps of per-gridpoint errors vs truth from per-cycle CSVs:
  <exp_dir>/<VAR>_lev<LEV>_values_cycle<K>.csv

Outputs up to 3 GIFs in <exp_dir>/error_gifs:
  bkg_vs_truth_<VAR>_lev<LEV>.gif
  ana_vs_truth_<VAR>_lev<LEV>.gif
  noda_vs_truth_<VAR>_lev<LEV>.gif (if NoDA column exists)

Assumes Cartopy is available. Falls back to synthetic lat/lon if diag NetCDF doesn’t exist.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from netCDF4 import Dataset as NC
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============ CONFIG ============
exp_dir  = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_5_100"  # where the CSVs live
var_name = "TG1"
level    = 7
nlat, nlon = 32, 64   # SPEEDY T21 grid (lat x lon)
fps = 3               # GIF speed
cmap = "magma"
smooth_sigma = 0.0    # Gaussian smoothing (grid cells). 0 = off (keeps it quicker)
dpi = 110             # figure DPI for rasterization
out_dir = os.path.join(exp_dir, "error_gifs")
os.makedirs(out_dir, exist_ok=True)
# ================================

try:
    from scipy.ndimage import gaussian_filter
    _HAS_GAUSS = True
except Exception:
    _HAS_GAUSS = False

def find_cycles(exp_dir, var_name, level):
    pat = os.path.join(exp_dir, f"{var_name}_lev{level}_values_cycle*.csv")
    files = sorted(glob.glob(pat), key=lambda p: int(re.search(r"cycle(\d+)", p).group(1)))
    return files

def load_errors_from_csv(csv_path, var_name, level, nlat, nlon):
    df = pd.read_csv(csv_path)
    col_b = f"{var_name}_bkg_lev{level}"
    col_a = f"{var_name}_ana_lev{level}"
    col_t = f"{var_name}_truth_lev{level}"
    col_n = f"{var_name}_noda_lev{level}"

    if col_t not in df.columns:
        raise ValueError(f"Truth column missing in {csv_path}")

    truth = df[col_t].to_numpy().reshape(nlat, nlon)

    def err(col):
        return np.abs(df[col].to_numpy().reshape(nlat, nlon) - truth)

    err_b = err(col_b) if col_b in df.columns else None
    err_a = err(col_a) if col_a in df.columns else None
    err_n = err(col_n) if col_n in df.columns else None

    return err_b, err_a, err_n

def try_load_latlon_from_diag(exp_dir, var_name, level):
    candidates = [
        os.path.join(exp_dir, f"{var_name}_lev{level}_diag_cycle0.nc"),
        os.path.join(exp_dir, f"{var_name}_lev{level}_diag_cycle1.nc"),
        os.path.join(exp_dir, f"{var_name}_lev{level}_diag.nc"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with NC(p, "r") as ds:
                    lat = np.array(ds["lat"][:])
                    lon = np.array(ds["lon"][:])
                    if lat.ndim == 1 and lon.ndim == 1:
                        return lat, lon
            except Exception:
                pass
    return None, None

def get_latlon(exp_dir, var_name, level, nlat, nlon):
    lat, lon = try_load_latlon_from_diag(exp_dir, var_name, level)
    if lat is not None and lon is not None:
        return lat, lon
    # fallback: evenly spaced centers (ok for visualization)
    lat = np.linspace(-90 + 90/nlat, 90 - 90/nlat, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    return lat, lon

def to_edges(x1d):
    """Compute cell edges from 1D centers for pcolormesh."""
    dx = np.diff(x1d).mean()
    return np.concatenate(([x1d[0] - dx/2], (x1d[:-1] + x1d[1:]) / 2, [x1d[-1] + dx/2]))

def apply_smoothing(arr):
    if smooth_sigma and _HAS_GAUSS:
        return gaussian_filter(arr, sigma=smooth_sigma, mode="nearest")
    return arr

def compute_vmin_vmax(csvs, var_name, level, nlat, nlon):
    # robust shared color scale (99th percentile across available errors)
    vals = []
    for csv in csvs:
        eb, ea, en = load_errors_from_csv(csv, var_name, level, nlat, nlon)
        for arr in (eb, ea, en):
            if arr is not None:
                vals.append(arr.ravel())
    if not vals:
        return 0.0, 1.0
    allv = np.abs(np.concatenate(vals))
    vmax = np.nanpercentile(allv, 99.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(allv)) if np.isfinite(np.nanmax(allv)) else 1.0
    return 0.0, float(vmax)

def render_gif(frames_arrays, titles, lat, lon, vmin, vmax, out_path):
    """Render a GIF from a list of 2D arrays using a single Cartopy axes."""
    if not frames_arrays:
        print(f"[warn] No frames for {os.path.basename(out_path)}, skipping.")
        return

    # Slower playback (fewer frames per second = longer per cycle)
    slow_fps = max(1, int(fps / 2))  # half speed (adjust if you want slower)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8.5, 4), dpi=dpi)
    ax = plt.axes(projection=proj)
    ax.set_global()

    try:
        ax.coastlines(resolution="110m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
    except Exception as e:
        print(f"[warn] coastlines disabled: {e}")

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5, linestyle="--")
    gl.right_labels = False
    gl.top_labels = False

    # --- Proper edges ---
    # lat/lon are centers; derive edges for full coverage
    lat_e = np.linspace(-90, 90, len(lat) + 1)
    lon_e = np.linspace(0, 360, len(lon) + 1)
    Lon_e, Lat_e = np.meshgrid(lon_e, lat_e)

    # initial frame
    arr0 = apply_smoothing(frames_arrays[0])
    quad = ax.pcolormesh(Lon_e, Lat_e, arr0, transform=proj, cmap=cmap,
                         vmin=vmin, vmax=vmax, shading="auto")
    ttl = ax.set_title(titles[0])

    # colorbar on a side axis
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(quad, cax=cax)
    cbar.set_label("|error|")

    frames_rgba = []

    def _capture():
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        frames_rgba.append(img.copy())

    # capture first frame
    _capture()

    # update for remaining frames
    for arr, title in zip(frames_arrays[1:], titles[1:]):
        quad.set_array(apply_smoothing(arr).ravel())
        ttl.set_text(title)
        _capture()

    imageio.mimsave(out_path, frames_rgba, fps=slow_fps, loop=0)
    plt.close(fig)
    print(f"Saved {out_path}  (fps={slow_fps})")


def main():
    csvs = find_cycles(exp_dir, var_name, level)
    if not csvs:
        print("No per-cycle CSVs found. Expected files like "
              f"{var_name}_lev{level}_values_cycle1.csv")
        return

    lat, lon = get_latlon(exp_dir, var_name, level, nlat, nlon)
    vmin, vmax = compute_vmin_vmax(csvs, var_name, level, nlat, nlon)

    # Build error stacks for each series in one pass
    bkg_stack, ana_stack, noda_stack = [], [], []
    bkg_titles, ana_titles, noda_titles = [], [], []

    # Title numbering to match your cycle filenames (usually 1..N)
    for csv in csvs:
        # extract cycle number for nice titles
        m = re.search(r"cycle(\d+)", csv)
        cyc = int(m.group(1)) if m else None
        label = f"cycle {cyc}" if cyc is not None else os.path.basename(csv)

        eb, ea, en = load_errors_from_csv(csv, var_name, level, nlat, nlon)
        if eb is not None:
            bkg_stack.append(eb)
            bkg_titles.append(f"Background | {label}")
        if ea is not None:
            ana_stack.append(ea)
            ana_titles.append(f"Analysis | {label}")
        if en is not None:
            noda_stack.append(en)
            noda_titles.append(f"NoDA | {label}")

    # Render GIFs (in-memory frames)
    if bkg_stack:
        render_gif(bkg_stack, bkg_titles, lat, lon, vmin, vmax,
                   os.path.join(out_dir, f"bkg_vs_truth_{var_name}_lev{level}.gif"))
    else:
        print("[warn] No background frames found.")

    if ana_stack:
        render_gif(ana_stack, ana_titles, lat, lon, vmin, vmax,
                   os.path.join(out_dir, f"ana_vs_truth_{var_name}_lev{level}.gif"))
    else:
        print("[warn] No analysis frames found.")

    if noda_stack:
        render_gif(noda_stack, noda_titles, lat, lon, vmin, vmax,
                   os.path.join(out_dir, f"noda_vs_truth_{var_name}_lev{level}.gif"))
    else:
        print("[info] No NoDA column found in CSVs; skipping NoDA GIF.")

if __name__ == "__main__":
    main()
