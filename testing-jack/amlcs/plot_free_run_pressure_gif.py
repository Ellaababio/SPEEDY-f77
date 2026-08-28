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

# --- User Settings ---
FREE_RUN_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/free_run"
OUTPUT_GIF = "free_run_pressure_evolution.gif"
P0_HPA = 1000.0  # Reference pressure

def _extract_cycle_num(filename):
    m = re.search(r'free_run_(\d+)\.nc', filename.name)
    return int(m.group(1)) if m else -1

def main():
    dir_path = Path(FREE_RUN_DIR)
    
    # 1. Gather all free_run files and sort them chronologically
    files = list(dir_path.glob("free_run_*.nc"))
    if not files:
        print(f"Error: No free_run_*.nc files found in {FREE_RUN_DIR}")
        return
        
    files.sort(key=_extract_cycle_num)
    
    print(f"Found {len(files)} files. Reading data...")

    # Lists to store fields and times
    ps_data = []
    times = []
    lat = None
    lon = None

    # 2. Read all the data
    from netCDF4 import num2date
    for f in files:
        cycle = _extract_cycle_num(f)
        try:
            with Dataset(f, 'r') as nc:
                if "PSG1" not in nc.variables:
                    print(f"Warning: PSG1 not found in {f.name}, skipping.")
                    continue
                
                # Model outputs log(ps/p0)
                ps_log = nc.variables["PSG1"][:]
                
                # Convert immediately to hPa
                ps_hpa = P0_HPA * np.exp(ps_log)
                
                # Store coordinates if we haven't already
                if lat is None:
                    lat = nc.variables["lat"][:] if "lat" in nc.variables else np.linspace(90, -90, ps_hpa.shape[0])
                    lon = nc.variables["lon"][:] if "lon" in nc.variables else np.linspace(0, 360, ps_hpa.shape[1], endpoint=False)
                    # For projection wrap
                    if lat.ndim == 1 and lon.ndim == 1:
                        lon, lat = np.meshgrid(lon, lat)

                ps_data.append(ps_hpa)
                times.append(cycle)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not ps_data:
        print("Error: No valid PSG1 data read. Exiting.")
        return

    # Convert to a 3D numpy array [time, lat, lon]
    data_stack = np.array(ps_data)
    
    # Calculate a global fixed colorbar range using standard earthly limits
    vmin = 960  # Typical low roughly
    vmax = 1040 # Typical high roughly

    print(f"Generating GIF from {len(data_stack)} frames...")
    
    # 3. Setup the initial Figure
    fig = plt.figure(figsize=(10, 5), dpi=150) # Set reasonable DPI for a GIF to keep size OK
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.coastlines(linewidth=0.8, color='black', zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', edgecolor='gray', zorder=10)
    #ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')

    # Initial plot using imshow just like heatmap_gifs_nc.py
    # extent=[0, 360, -90, 90] tells cartopy exactly where this array lives globally
    im = ax.imshow(data_stack[0], origin='lower', extent=[0, 360, -90, 90],
                   cmap='Spectral_r', vmin=vmin, vmax=vmax, interpolation='bicubic',
                   transform=ccrs.PlateCarree())

    # Styling the colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, aspect=40, extend='both')
    cbar.set_label('Surface Pressure (hPa)', fontsize=12, fontweight='medium')

    # Add a title text object we can update
    from datetime import datetime, timedelta
    
    # SPEEDY configuration from cls_instep.h:
    # IYEAR0=1979, IMONT0=1
    START_DATE = datetime(1979, 1, 1, 0, 0, 0)
    
    # Set to 48 hours to match the obs_steps=2 configuration (2 days per cycle)
    CYCLE_HOURS = 48

    def get_cycle_date(cycle_num):
        return START_DATE + timedelta(hours=(cycle_num * CYCLE_HOURS))

    initial_date_str = get_cycle_date(times[0]).strftime("%Y-%m-%d %H:00")
    title_text = ax.set_title(f"Global Surface Pressure - Free Run\n{initial_date_str} (Cycle {times[0]})", 
                              fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    # 4. Animate it
    def update(frame_idx):
        im.set_data(data_stack[frame_idx])
        current_date_str = get_cycle_date(times[frame_idx]).strftime("%Y-%m-%d %H:00")
        title_text.set_text(f"Global Surface Pressure - Free Run\n{current_date_str} (Cycle {times[frame_idx]})")
        return [im, title_text]

    # Save logic
    writer = PillowWriter(fps=2) # 4 frames per second
    with writer.saving(fig, OUTPUT_GIF, dpi=120):
        for i in range(len(data_stack)):
            update(i)
            writer.grab_frame()
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(data_stack)} frames...")

    print(f"\nSUCCESS! Animated GIF saved to {OUTPUT_GIF}")
    plt.close()

if __name__ == "__main__":
    main()
