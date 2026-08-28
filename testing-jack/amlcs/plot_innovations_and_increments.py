#!/usr/bin/env python3
"""
plot_innovations_and_increments.py  (UNIFIED CSV version)
---------------------------------------------------------
Computes and visualizes per-cycle fields from the unified flattened CSVs:
  * Innovations = obs - xb_mean
  * Increments  = xa_mean - xb_mean

Inputs: per-cycle unified CSVs like
  <exp_dir>/<VAR>_lev<LEV>_cycle<K>.csv
with columns:
  idx, xb_mean, xa_mean, truth, noda, obs, sigma, is_obs

Outputs:
  - GIF: innovations_obs_minus_bkg_<VAR>_lev<LEV>.gif
  - GIF: increments_ana_minus_bkg_<VAR>_lev<LEV>.gif
  - GIF: paired_innovations_and_increments_<VAR>_lev<LEV>.gif  (side-by-side)
  - PNG: time_series_innovations_increments_<VAR>_lev<LEV>.png

Notes:
  - 'sigma' in the unified CSV is the observation std (sqrt(diag(R))) in model units.
  - Innovations are NaN where obs are missing; time series uses NaN-safe means.
"""

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Optional Cartopy for the single-field GIFs (kept as before)
_USE_CARTOPY = True
try:
    import cartopy, os as _os
    cartopy.config['data_dir'] = _os.path.join(_os.environ.get('CONDA_PREFIX', ''), 'share', 'cartopy')
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:
    _USE_CARTOPY = False

# ---------- CONFIG ----------
exp_dir  = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100"
var_name = "TG0"
level    = 7
nlat, nlon = 32, 64
fps = 2               # base fps; script slows it a bit in GIFs
dpi = 110
cmap = "magma"
out_dir = os.path.join(exp_dir, "innov_inc_gfx")
os.makedirs(out_dir, exist_ok=True)

# Nonlinear standard-observation settings used by the experiment.
NONLINEAR_OBS = True
SCALEFACT = 0.5
# ---------------------------

def find_cycles(exp_dir, var_name, level):
    # unified CSVs are named like: <VAR>_lev<LEV>_cycle<K>.csv
    pat = os.path.join(exp_dir, f"{var_name}_lev{level}_cycle*.csv")
    files = sorted(glob.glob(pat), key=lambda p: int(re.search(r"cycle(\d+)", p).group(1)))
    return files

def load_unified_grids_from_csv(csv_path, nlat, nlon):
    """
    Expect columns: idx, xb_mean, xa_mean, truth, noda, obs, sigma, is_obs
    Returns: XB, XA, TRUTH, NODA, OBS, SIGMA as (nlat, nlon) arrays
    """
    df = pd.read_csv(csv_path)
    required = ["xb_mean", "xa_mean", "truth", "noda", "obs", "sigma", "is_obs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    def reshape(col):
        return df[col].to_numpy().reshape(nlat, nlon)

    XB    = reshape("xb_mean")
    XA    = reshape("xa_mean")
    TRUTH = reshape("truth")
    NODA  = reshape("noda")
    OBS   = reshape("obs")
    SIGMA = reshape("sigma")  # observation std (sqrt(diag(R)))
    # is_obs not used directly in fields, but could be helpful to mask OBS
    return XB, XA, TRUTH, NODA, OBS, SIGMA

def area_weights(nlat):
    lat = np.linspace(-90 + 90/nlat, 90 - 90/nlat, nlat)
    w = np.cos(np.deg2rad(lat))
    w /= np.nanmean(w)  # normalize
    return w  # (nlat,)

def to_edges(x1d, start, stop):
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
    slow_fps = max(1, fps // 2)

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

def render_pair_gif(frames_left, frames_right, titles, lat, lon,
                    vmin_l, vmax_l, vmin_r, vmax_r, out_path,
                    label_left="obs - bkg", label_right="ana - bkg"):
    """
    Side-by-side GIF: left = innovations, right = increments.
    Uses plain matplotlib (no Cartopy) for robustness.
    """
    if not frames_left or not frames_right or len(frames_left) != len(frames_right):
        print(f"[warn] Cannot make side-by-side GIF (mismatched frames).")
        return
    slow_fps = max(1, fps // 2)
    imgs = []
    for arrL, arrR, t in zip(frames_left, frames_right, titles):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi, constrained_layout=True)
        imL = axL.imshow(arrL, origin="lower", vmin=vmin_l, vmax=vmax_l, cmap=cmap,
                         interpolation="bilinear", extent=(0, 360, -90, 90), aspect='auto')
        axL.set_title(f"Innovations (obs - bkg) | {t}")
        axL.set_xlabel("lon"); axL.set_ylabel("lat")
        cbL = plt.colorbar(imL, ax=axL, fraction=0.046, pad=0.04); cbL.set_label(label_left)

        imR = axR.imshow(arrR, origin="lower", vmin=vmin_r, vmax=vmax_r, cmap=cmap,
                         interpolation="bilinear", extent=(0, 360, -90, 90), aspect='auto')
        axR.set_title(f"Increments (ana - bkg) | {t}")
        axR.set_xlabel("lon"); axR.set_ylabel("lat")
        cbR = plt.colorbar(imR, ax=axR, fraction=0.046, pad=0.04); cbR.set_label(label_right)

        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        plt.close(fig)

    imageio.mimsave(out_path, imgs, fps=slow_fps, loop=0)
    print(f"Saved {out_path} (side-by-side; fps={slow_fps}, loop=∞)")

def robust_limits(stack):
    if not stack: return (0.0, 1.0)
    vals = np.abs(np.concatenate([np.ravel(a) for a in stack]))
    vmax = np.nanpercentile(vals, 99.0)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
    return (-vmax, vmax)


def compute_innovation(obs: np.ndarray, xb: np.ndarray, var: str) -> np.ndarray:
    if var == "WDG1":
        return (obs - xb + np.pi) % (2 * np.pi) - np.pi
    if NONLINEAR_OBS and var not in {"WSG1", "WDG1"}:
        return obs - np.arctan(SCALEFACT * xb)
    return obs - xb

def main():
    csvs = find_cycles(exp_dir, var_name, level)
    if not csvs:
        print("No unified per-cycle CSVs found.")
        return

    # grid centers (for edges)
    lat = np.linspace(-90 + 90/nlat, 90 - 90/nlat, nlat)
    lon = np.linspace(0, 360, nlon, endpoint=False)
    wlat = area_weights(nlat)  # (nlat,)

    innov_frames, incr_frames = [], []
    titles = []
    innov_series, incr_series = [], []

    # Load & compute per cycle
    for csv in csvs:
        m = re.search(r"cycle(\d+)", csv)
        cyc = int(m.group(1)) if m else None
        label = f"cycle {cyc}" if cyc is not None else os.path.basename(csv)

        XB, XA, TRUTH, NODA, OBS, SIGMA = load_unified_grids_from_csv(csv, nlat, nlon)

        # Compute innovations in observation space for nonlinear standard obs.
        INNOV = compute_innovation(OBS, XB, var_name)

        # Increments: ana - bkg
        INCR = XA - XB

        innov_frames.append(INNOV)
        incr_frames.append(INCR)
        titles.append(label)

        # area-weighted spatial means (NaN-safe)
        innov_series.append(np.nanmean(INNOV * wlat[:, None]))
        incr_series.append(np.nanmean(INCR  * wlat[:, None]))

    # color limits (robust)
    vmin_i, vmax_i = robust_limits(innov_frames)
    vmin_k, vmax_k = robust_limits(incr_frames)

    # GIFs (single-field)
    os.makedirs(out_dir, exist_ok=True)
    render_gif(
        innov_frames, [f"Innovations (obs - bkg) | {t}" for t in titles],
        lat, lon, vmin_i, vmax_i,
        os.path.join(out_dir, f"innovations_obs_minus_bkg_{var_name}_lev{level}.gif"),
        colorlabel="obs - bkg"
    )
    render_gif(
        incr_frames, [f"Increments (ana - bkg) | {t}" for t in titles],
        lat, lon, vmin_k, vmax_k,
        os.path.join(out_dir, f"increments_ana_minus_bkg_{var_name}_lev{level}.gif"),
        colorlabel="ana - bkg"
    )

    # Side-by-side GIF
    render_pair_gif(
        innov_frames, incr_frames, titles, lat, lon,
        vmin_i, vmax_i, vmin_k, vmax_k,
        os.path.join(out_dir, f"paired_innovations_and_increments_{var_name}_lev{level}.gif"),
        label_left="obs - bkg", label_right="ana - bkg"
    )

    # Time series figure
    cycles = [int(re.search(r"cycle(\d+)", c).group(1)) for c in csvs]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=dpi)
    ax.plot(cycles, innov_series, marker='o', label='mean(obs - bkg)')
    ax.plot(cycles, incr_series,  marker='s', label='mean(ana - bkg)')
    ax.set_xlabel("cycle")
    ax.set_ylabel("area-weighted mean")
    ax.set_title(f"Spatially averaged innovations & increments — {var_name}@lev{level}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ts_path = os.path.join(out_dir, f"time_series_innovations_increments_{var_name}_lev{level}.png")
    fig.savefig(ts_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {ts_path}")

if __name__ == "__main__":
    main()
