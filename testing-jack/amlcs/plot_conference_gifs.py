#!/usr/bin/env python3
import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from matplotlib.animation import PillowWriter
from datetime import datetime, timedelta

# --- User Settings ---
TRUTH_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/snapshots"
FREE_RUN_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/free_run"
ENSF_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/data_ps0001"
OUTPUT_DIR = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_ReverseSDE_1_1_100/linear_results_ps_only/new_conference_plots"

OUTPUT_ENSF_PRESSURE = os.path.join(OUTPUT_DIR, "ensf_pressure_evolution.gif")
OUTPUT_NODA_WIND = os.path.join(OUTPUT_DIR, "noda_wind_u_evolution.gif")
OUTPUT_ENSF_WIND = os.path.join(OUTPUT_DIR, "ensf_wind_u_evolution.gif")

OUTPUT_PRESSURE_ERR = os.path.join(OUTPUT_DIR, "error_pressure_evolution_pair.gif")
OUTPUT_WIND_ERR = os.path.join(OUTPUT_DIR, "error_wind_u_evolution_pair.gif")

P0_HPA = 1000.0  # Reference pressure
WIND_LEVEL = 7   # Plot lowest level wind (0-7 indexing, 7 is typically 925 hPa)
START_DATE = datetime(1979, 1, 1, 0, 0, 0)
CYCLE_HOURS = 48  # 48 hours (2 days) per cycle

def _extract_cycle_num(filename, prefix="free_run_"):
    m = re.search(rf'{prefix}(\d+)\.nc', filename.name)
    return int(m.group(1)) if m else -1

def get_cycle_date(cycle_num):
    return START_DATE + timedelta(hours=(cycle_num * CYCLE_HOURS))

def create_gif(data_stack, times, output_filename, title_prefix, cmap, vmin, vmax, cbar_label):
    if len(data_stack) == 0:
        print(f"No data to create {output_filename}. Skipping.")
        return

    print(f"Generating GIF {output_filename} from {len(data_stack)} frames...")
    
    fig = plt.figure(figsize=(10, 5), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.coastlines(linewidth=0.8, color='black', zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', edgecolor='gray', zorder=10)

    im = ax.imshow(data_stack[0], origin='lower', extent=[0, 360, -90, 90],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bicubic',
                   transform=ccrs.PlateCarree())

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=40, extend='both')
    cbar.set_label(cbar_label, fontsize=12, fontweight='medium')

    initial_date_str = get_cycle_date(times[0]).strftime("%Y-%m-%d %H:00")
    title_text = ax.set_title(f"{title_prefix}\n{initial_date_str} (Cycle {times[0]})", 
                              fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    def update(frame_idx):
        im.set_data(data_stack[frame_idx])
        current_date_str = get_cycle_date(times[frame_idx]).strftime("%Y-%m-%d %H:00")
        title_text.set_text(f"{title_prefix}\n{current_date_str} (Cycle {times[frame_idx]})")
        return [im, title_text]

    writer = PillowWriter(fps=2)
    with writer.saving(fig, output_filename, dpi=120):
        for i in range(len(data_stack)):
            update(i)
            writer.grab_frame()
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(data_stack)} frames...")

    print(f"SUCCESS! Animated GIF saved to {output_filename}\n")
    plt.close()

def create_pair_gif(data_stack_top, data_stack_bot, times, output_filename, title_top, title_bot, cmap, vmin, vmax, cbar_label):
    n_frames = min(len(data_stack_top), len(data_stack_bot))
    if n_frames == 0:
        print(f"No data to create {output_filename}. Skipping.")
        return

    print(f"Generating paired GIF {output_filename} from {n_frames} frames...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, subplot_kw={'projection': ccrs.PlateCarree()})
    
    for ax in (ax1, ax2):
        ax.coastlines(linewidth=0.8, color='black', zorder=10)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', edgecolor='gray', zorder=10)

    im1 = ax1.imshow(data_stack_top[0], origin='lower', extent=[0, 360, -90, 90],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bicubic',
                   transform=ccrs.PlateCarree())
                   
    im2 = ax2.imshow(data_stack_bot[0], origin='lower', extent=[0, 360, -90, 90],
                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bicubic',
                   transform=ccrs.PlateCarree())                   

    # Shared colorbar
    plt.subplots_adjust(bottom=0.15, hspace=0.3)
    cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label(cbar_label, fontsize=12, fontweight='medium')

    initial_date_str = get_cycle_date(times[0]).strftime("%Y-%m-%d %H:00")
    title_text1 = ax1.set_title(f"{title_top}\n{initial_date_str} (Cycle {times[0]})", 
                              fontsize=12, fontweight='bold', pad=10)
    title_text2 = ax2.set_title(f"{title_bot}", 
                              fontsize=12, fontweight='bold', pad=10)

    def update(frame_idx):
        im1.set_data(data_stack_top[frame_idx])
        im2.set_data(data_stack_bot[frame_idx])
        current_date_str = get_cycle_date(times[frame_idx]).strftime("%Y-%m-%d %H:00")
        title_text1.set_text(f"{title_top}\n{current_date_str} (Cycle {times[frame_idx]})")
        return [im1, im2, title_text1]

    writer = PillowWriter(fps=2)
    with writer.saving(fig, output_filename, dpi=120):
        for i in range(n_frames):
            update(i)
            writer.grab_frame()
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{n_frames} frames...")

    print(f"SUCCESS! Animated Paired GIF saved to {output_filename}\n")
    plt.close()

def main():
    truth_path = Path(TRUTH_DIR)
    free_run_path = Path(FREE_RUN_DIR)
    ensf_path = Path(ENSF_DIR)
    
    # 1. Gather all files
    tr_files = list(truth_path.glob("reference_solution_*.nc"))
    tr_files.sort(key=lambda f: _extract_cycle_num(f, "reference_solution_"))

    fr_files = list(free_run_path.glob("free_run_*.nc"))
    fr_files.sort(key=lambda f: _extract_cycle_num(f, "free_run_"))
    
    ens_files = list(ensf_path.glob("unified_cycle*.nc"))
    # Fallback to older reverseSDE pattern if unified not found
    if not ens_files:
        ens_files = list(ensf_path.glob("reverseSDE_cycle*.nc"))
        ens_prefix = "reverseSDE_cycle"
    else:
        ens_prefix = "unified_cycle"
    ens_files.sort(key=lambda f: _extract_cycle_num(f, ens_prefix))
    
    print(f"Found {len(tr_files)} Truth (Reference Solution) files.")
    print(f"Found {len(fr_files)} No-DA (Free Run) files.")
    print(f"Found {len(ens_files)} ENSF files (prefix: {ens_prefix}).")
    
    # Storage
    truth_pres_data, truth_wind_data = {}, {}
    ensf_pres_data, ensf_pres_times, ensf_pres_err_data = [], [], []
    noda_pres_data, noda_pres_times, noda_pres_err_data = [], [], []
    noda_wind_data, noda_wind_times, noda_wind_err_data = [], [], []
    ensf_wind_data, ensf_wind_times, ensf_wind_err_data = [], [], []

    # --- READ TRUTH ---
    for f in tr_files:
        cycle = _extract_cycle_num(f, "reference_solution_")
        try:
            with Dataset(f, 'r') as nc:
                if "PSG1" in nc.variables:
                    ps_log = nc.variables["PSG1"][:]
                    ps_hpa = P0_HPA * np.exp(ps_log)
                    truth_pres_data[cycle] = ps_hpa
                if "UG1" in nc.variables:
                    u_all = nc.variables["UG1"][:]
                    if u_all.ndim == 3:
                        u = u_all[WIND_LEVEL]
                    else:
                         u = u_all
                    truth_wind_data[cycle] = u
        except Exception as e:
             print(f"Error reading Truth {f.name}: {e}")

    # --- READ ENSF PRESSURE & WIND ---
    for f in ens_files:
        cycle = _extract_cycle_num(f, ens_prefix)
        try:
            with Dataset(f, 'r') as nc:
                # 1. Pressure
                if "xa_mean_PSG1_lev0" in nc.variables:
                    ps_log = nc.variables["xa_mean_PSG1_lev0"][:]
                    ps_hpa = P0_HPA * np.exp(ps_log)
                    ensf_pres_data.append(ps_hpa)
                    ensf_pres_times.append(cycle)
                    if cycle in truth_pres_data:
                        ensf_pres_err_data.append(np.abs(ps_hpa - truth_pres_data[cycle]))
                    
                # 2. Wind (U-component)
                u_key = f"xa_mean_UG1_lev{WIND_LEVEL}"
                if u_key in nc.variables:
                    u = nc.variables[u_key][:]
                    ensf_wind_data.append(u)
                    ensf_wind_times.append(cycle)
                    if cycle in truth_wind_data:
                        ensf_wind_err_data.append(np.abs(u - truth_wind_data[cycle]))
        except Exception as e:
             print(f"Error reading ENSF {f.name}: {e}")

    # --- READ NODA WIND & PRESSURE ---
    for f in fr_files:
        cycle = _extract_cycle_num(f, "free_run_")
        try:
            with Dataset(f, 'r') as nc:
                if "PSG1" in nc.variables:
                    ps_log = nc.variables["PSG1"][:]
                    ps_hpa = P0_HPA * np.exp(ps_log)
                    noda_pres_data.append(ps_hpa)
                    noda_pres_times.append(cycle)
                    if cycle in truth_pres_data:
                        noda_pres_err_data.append(np.abs(ps_hpa - truth_pres_data[cycle]))

                if "UG1" in nc.variables:
                    # In NoDA, UG1 is usually 3D [lev, lat, lon]. WIND_LEVEL goes up to 7.
                    u_all = nc.variables["UG1"][:]
                    if u_all.ndim == 3:
                        u = u_all[WIND_LEVEL]
                    else:
                         u = u_all
                    noda_wind_data.append(u)
                    noda_wind_times.append(cycle)
                    if cycle in truth_wind_data:
                        noda_wind_err_data.append(np.abs(u - truth_wind_data[cycle]))
        except Exception as e:
             print(f"Error reading Free Run {f.name}: {e}")

    # --- GENERATE GIFS ---
    # 1. ENSF Pressure
    create_gif(
        data_stack=ensf_pres_data,
        times=ensf_pres_times,
        output_filename=OUTPUT_ENSF_PRESSURE,
        title_prefix="Global Surface Pressure - ENSF Analysis",
        cmap="Spectral_r",
        vmin=960, vmax=1040,
        cbar_label="Surface Pressure (hPa)"
    )

    # 2. No-DA Wind (U-Component)
    create_gif(
        data_stack=noda_wind_data,
        times=noda_wind_times,
        output_filename=OUTPUT_NODA_WIND,
        title_prefix=f"Global U-Wind (Surface) - No DA (Free Run)",
        cmap="seismic_r",
        vmin=-20, vmax=20, # U-wind centered at 0
        cbar_label="U-Wind Velocity (m/s)"
    )

    # 3. ENSF Wind (U-Component)
    create_gif(
        data_stack=ensf_wind_data,
        times=ensf_wind_times,
        output_filename=OUTPUT_ENSF_WIND,
        title_prefix=f"Global U-Wind (Surface) - ENSF Analysis",
        cmap="seismic_r",
        vmin=-20, vmax=20,
        cbar_label="U-Wind Velocity (m/s)"
    )

    # --- ERROR GIFS ---
    create_pair_gif(
        data_stack_top=noda_pres_err_data,
        data_stack_bot=ensf_pres_err_data,
        times=ensf_pres_times[:min(len(noda_pres_err_data), len(ensf_pres_err_data))],
        output_filename=OUTPUT_PRESSURE_ERR,
        title_top="Global Surface Pressure Error - No DA",
        title_bot="Global Surface Pressure Error - ENSF Analysis",
        cmap="Reds",
        vmin=0, vmax=30,
        cbar_label="Absolute Pressure Error (hPa)"
    )

    create_pair_gif(
        data_stack_top=noda_wind_err_data,
        data_stack_bot=ensf_wind_err_data,
        times=ensf_wind_times[:min(len(noda_wind_err_data), len(ensf_wind_err_data))],
        output_filename=OUTPUT_WIND_ERR,
        title_top="Global U-Wind Error (Surface) - No DA",
        title_bot="Global U-Wind Error (Surface) - ENSF Analysis",
        cmap="Reds",
        vmin=0, vmax=15, 
        cbar_label="Absolute U-Wind Error (m/s)"
    )

if __name__ == "__main__":
    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
