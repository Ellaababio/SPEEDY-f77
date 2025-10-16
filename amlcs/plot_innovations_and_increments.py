#!/usr/bin/env python3
"""
plot_innovations_and_increments.py
----------------------------------
Compute and visualize:
  1) Innovations (truth - bkg)  ~ approximated as (truth - background) per grid cell
  2) Increments (bkg - ana)   per grid cell

Inputs: per-cycle CSVs like
  <exp_dir>/<VAR>_lev<LEV>_values_cycle<K>.csv
with columns:
  <VAR>_bkg_lev<LEV>, <VAR>_ana_lev<LEV>, <VAR>_truth_lev<LEV>, [<VAR>_noda_lev<LEV>]

Outputs:
  - GIFs: innovations_vs_bkg_<VAR>_lev<LEV>.gif, increments_bkg_minus_ana_<VAR>_lev<LEV>.gif
  - PNG:  time_series_innovations_increments_<VAR>_lev<LEV>.png
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Optional Cartopy
_USE_CARTOPY = True
try:
    import cartopy, os as _os
    # if you installed via conda-forge, this points Cartopy at bundled offline data
    cartopy.config['data_dir'] = _os.path.join(_os.environ.get('CONDA_PREFIX', ''), 'share', 'cartopy')
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    _USE_CARTOPY = False

# ---------- CONFIG ----------
exp_dir  = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_5_100"
var_name = "TG1"
level    = 7
nlat, nlon = 32, 64
fps = 2               # base fps; script slows it by 2x below
dpi = 110
cmap = "magma"
out_dir = os.path.join(exp_dir, "innov_inc_gfx")
os.makedirs(out_dir, exist_ok=True)
# ---------------------------

def find_cycles(exp_dir, var_name, level):
    pat = os.path.join(exp_dir, f"{var_name}_lev{level}_values_cycle*.csv")
    files = sorted(glob.glob(pat), key=lambda p: int(re.search(r"cycle(\d+)", p).group(1)))
    return files

def load_grids_from_csv(csv_path, var_name, level, nlat, nlon):
    df = pd.read_csv(csv_path)
    c_b = f"{var_name}_bkg_lev{level}"
    c_a = f"{var_name}_ana_lev{level}"
    c_t = f"{var_name}_truth_lev{level}"
    if c_b not in df or c_a not in df or c_t not in df:
        raise ValueError(f"Missing required columns in {csv_path}")
    B = df[c_b].to_numpy().reshape(nlat, nlon)
    A = df[c_a].to_numpy().reshape(nlat, nlon)
    T = df[c_t].to_numpy().reshape(nlat, nlon)
    return B, A, T

def area_weights(nlat):
    # simple cos(lat) weights (lat centers)
    lat = np.linspace(-90 + 90/nlat, 90 - 90/nlat, nlat)
    w = np.cos(np.deg2rad(lat))
    w /= w.mean()  # normalize so mean weight ~ 1
    return w  # shape (nlat,)

def to_edges(x1d, start, stop):
    # produce edges spanning [start, stop] with len(x1d)+1 points
    return np.linspace(start, stop, len(x1d)+1)

def plot_frame_plain(arr, title, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8.5, 4), dpi=dpi)
    im = ax.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap,
                   interpolation="bilinear", extent=(0, 360, -90, 90), aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("value")
    fig.tight_layout()
    return fig

def render_gif(frames, titles, lat, lon, vmin, vmax, out_path, colorlabel):
    if not frames:
        print(f"[warn] No frames for {os.path.basename(out_path)}; skipping.")
        return

    slow_fps = max(1, fps // 2)  # slower playback

    if _USE_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(8.8, 4.2), dpi=dpi)
        ax = plt.axes(projection=proj)
        ax.set_global()
        try:
            ax.coastlines(resolution="110m", linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray")
        except Exception as e:
            print(f"[warn] coastlines disabled: {e}")
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5, linestyle="--")
        gl.right_labels = False; gl.top_labels = False

        lat_e = to_edges(lat, -90, 90)
        lon_e = to_edges(lon, 0, 360)
        Lon_e, Lat_e = np.meshgrid(lon_e, lat_e)

        quad = ax.pcolormesh(Lon_e, Lat_e, frames[0], transform=proj, cmap=cmap,
                             vmin=vmin, vmax=vmax, shading="auto")
        ttl = ax.set_title(titles[0])
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = plt.colorbar(quad, cax=cax); cbar.set_label(colorlabel)

        imgs = []
        def _cap():
            fig.canvas.draw()
            imgs.append(np.asarray(fig.canvas.buffer_rgba()).copy())

        _cap()
        for arr, t in zip(frames[1:], titles[1:]):
            quad.set_array(arr.ravel())
            ttl.set_text(t)
            _cap()

        imageio.mimsave(out_path, imgs, fps=slow_fps, loop=0)
        plt.close(fig)
    else:
        imgs = []
        for arr, t in zip(frames, titles):
            fig = plot_frame_plain(arr, t, vmin, vmax)
            fig.canvas.draw()
            imgs.append(np.asarray(fig.canvas.buffer_rgba()).copy())
            plt.close(fig)
        imageio.mimsave(out_path, imgs, fps=slow_fps, loop=0)

    print(f"Saved {out_path} (fps={slow_fps}, loop=∞)")

def main():
    csvs = find_cycles(exp_dir, var_name, level)
    if not csvs:
        print("No per-cycle CSVs found.")
        return

    # grid centers (for edges)
    lat = np.linspace(-90 + 90/nlat, 90 - 90/nlat, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    wlat = area_weights(nlat)  # (nlat,)

    innov_frames, incr_frames = [], []
    innov_titles, incr_titles = [], []
    innov_series, incr_series = [], []

    # first pass: load and compute fields
    for csv in csvs:
        m = re.search(r"cycle(\d+)", csv)
        cyc = int(m.group(1)) if m else None
        label = f"cycle {cyc}" if cyc is not None else os.path.basename(csv)

        B, A, T = load_grids_from_csv(csv, var_name, level, nlat, nlon)

        # Innovations (truth - bkg) approximated by (truth - background)
        INNOV = T - B
        # Increments (bkg - ana)
        INCR  = B - A

        innov_frames.append(INNOV)
        incr_frames.append(INCR)
        innov_titles.append(f"Innovations (truth-bkg) | {label}")
        incr_titles.append(f"Increments (bkg-ana) | {label}")

        # area-weighted spatial mean per cycle (signed)
        innov_series.append( (INNOV * wlat[:,None]).mean() )
        incr_series.append(  (INCR  * wlat[:,None]).mean() )

    # set shared color scales using robust (99th percentile of abs)
    def robust_limits(stack):
        if not stack: return (0.0, 1.0)
        vals = np.abs(np.concatenate([a.ravel() for a in stack]))
        vmax = np.nanpercentile(vals, 99.0)
        if not np.isfinite(vmax) or vmax <= 0: vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
        return (-vmax, vmax)  # signed fields
    vmin_i, vmax_i = robust_limits(innov_frames)
    vmin_k, vmax_k = robust_limits(incr_frames)

    # GIFs
    os.makedirs(out_dir, exist_ok=True)
    render_gif(innov_frames, innov_titles, lat, lon, vmin_i, vmax_i,
               os.path.join(out_dir, f"innovations_vs_bkg_{var_name}_lev{level}.gif"),
               colorlabel="truth - bkg")
    render_gif(incr_frames, incr_titles, lat, lon, vmin_k, vmax_k,
               os.path.join(out_dir, f"increments_bkg_minus_ana_{var_name}_lev{level}.gif"),
               colorlabel="bkg - ana")

    # 1D time series (spatially averaged per cycle)
    cycles = [int(re.search(r"cycle(\d+)", c).group(1)) for c in csvs]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=dpi)
    ax.plot(cycles, innov_series, marker='o', label='mean(truth - bkg)')
    ax.plot(cycles, incr_series,  marker='s', label='mean(bkg - ana)')
    ax.set_xlabel("cycle")
    ax.set_ylabel("area-weighted mean")
    ax.set_title(f"Spatially averaged (signed) innovations & increments — {var_name}@lev{level}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ts_path = os.path.join(out_dir, f"time_series_innovations_increments_{var_name}_lev{level}.png")
    fig.savefig(ts_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {ts_path}")

if __name__ == "__main__":
    main()
