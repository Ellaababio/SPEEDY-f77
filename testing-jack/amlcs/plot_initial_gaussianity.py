#!/usr/bin/env python3
"""
Plot the Shapiro-Wilks Gaussianity (p-value) of the initial ensemble
for each variable at each level.

Usage:
  python plot_initial_gaussianity.py --ens_dir <path_to_ensemble_0> --out_dir <output_dir>
"""

import os
import argparse
import glob
import numpy as np
from netCDF4 import Dataset
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def get_var_data(nc, var_name):
    """Safely extract data for a variable, handling the 'ntr' dimension for TRG."""
    data = nc.variables[var_name][:]
    if 'TRG' in var_name:
        return data[0, :, :, :]  # Drop ntr dimension
    return data

def plot_2d_map(data_2d, lat, lon, var_name, level_name, out_path):
    """Plot a 2D map of the Shapiro-Wilk p-value."""
    fig = plt.figure(figsize=(7.2, 4.1), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    fig.subplots_adjust(left=0.03, right=0.90, bottom=0.05, top=0.91)
    
    ax.coastlines(linewidth=0.7, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='gray')

    # Plot p-value gradient. p > 0.05 means we fail to reject Gaussianity.
    im = ax.imshow(data_2d, origin='lower', extent=[0, 360, -90, 90],
                   cmap='RdYlGn', vmin=0, vmax=1.0, interpolation='nearest',
                   transform=ccrs.PlateCarree())

    # Clearly mark significant gridpoints (p < 0.05) by overlaying solid magenta pixels
    # This keeps the "2D map style" (blocky pixels) instead of dots or lines.
    import matplotlib.colors as mcolors
    sig_mask = data_2d < 0.05
    if np.any(sig_mask):
        masked_data = np.ma.masked_where(~sig_mask, data_2d)
        ax.imshow(masked_data, origin='lower', extent=[0, 360, -90, 90],
                  cmap=mcolors.ListedColormap(['magenta']), vmin=0, vmax=1, 
                  interpolation='nearest', transform=ccrs.PlateCarree())
                  
        # Add a legend entry so the user knows what magenta means
        ax.scatter([], [], color='magenta', label='p < 0.05 (Non-Gaussian)', marker='s')
        ax.legend(loc='lower right', framealpha=0.9, fontsize=8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    cbar.set_label('p-value', fontsize=10)

    median_p = np.median(data_2d)
    title_text = f"Initial Gaussianity (Shapiro-Wilk) | {var_name}  |  {level_name}\nMedian p-value: {median_p:.4f}"
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=8)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

def main():
    ens_dir = '/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/ensemble_0'
    out_dir = '/gpfs/home/jjs21b/AMLCS/runs/initial_gaussianity_maps_level_averaged'

    # Find all ensemble members
    file_pattern = os.path.join(ens_dir, "ensemble_member_*.nc")
    ens_files = glob.glob(file_pattern)
    Nens = len(ens_files)
    
    if Nens < 3:
        print(f"Error: Found {Nens} files in {ens_dir}. Shapiro-Wilk requires at least 3 samples.")
        return

    print(f"Found {Nens} ensemble members in {ens_dir}")

    # Read coordinates from the first file
    with Dataset(ens_files[0], 'r') as nc:
        # Some Speedy output uses lat/lon directly or inferred from dimensions
        if 'latitude' in nc.variables:
            lat = nc.variables['latitude'][:]
        elif 'lat' in nc.variables:
            lat = nc.variables['lat'][:]
        else:
            nlat = nc.dimensions['latitude'].size if 'latitude' in nc.dimensions else nc.dimensions['lat'].size
            lat = np.linspace(90, -90, nlat) # Default Speedy
            
        if 'longitude' in nc.variables:
            lon = nc.variables['longitude'][:]
        elif 'lon' in nc.variables:
            lon = nc.variables['lon'][:]
        else:
            nlon = nc.dimensions['longitude'].size if 'longitude' in nc.dimensions else nc.dimensions['lon'].size
            lon = np.linspace(0, 360, nlon, endpoint=False) # Default Speedy

        var_names = ['UG1', 'VG1', 'TG1', 'TRG1', 'PSG1']
        
        # Verify variables exist
        for v in var_names:
            if v not in nc.variables:
                print(f"Warning: {v} not found in {ens_files[0]}")

    # Read all data into memory
    # Structure: ensemble_data[var_name] = np array of shape (Nens, levels, lat, lon) or (Nens, lat, lon)
    ensemble_data = {v: [] for v in var_names}

    print("Loading ensemble data...")
    for f in ens_files:
        with Dataset(f, 'r') as nc:
            for v in var_names:
                if v in nc.variables:
                    ensemble_data[v].append(get_var_data(nc, v))
                    
    for v in var_names:
        if len(ensemble_data[v]) > 0:
            ensemble_data[v] = np.array(ensemble_data[v])

    # Calculate Shapiro-Wilk and plot
    print("Calculating Shapiro-Wilk p-values and plotting...")
    for v in var_names:
        data = ensemble_data[v]
        if data.size == 0:
            continue
            
        print(f"Processing {v} (shape: {data.shape})")
        
        # Determine if 2D or 3D
        if data.ndim == 3:
            # Shape is (Nens, lat, lon) -> e.g., PSG1
            # Ensure Level directory exists
            level_dir = os.path.join(out_dir, "Surface")
            os.makedirs(level_dir, exist_ok=True)
            
            p_values = np.zeros((data.shape[1], data.shape[2]))
            for i in range(data.shape[1]):
                for j in range(data.shape[2]):
                    samples = data[:, i, j]
                    # Shapiro test returns (statistic, p-value)
                    stat, p = shapiro(samples)
                    p_values[i, j] = p
                    
            out_path = os.path.join(level_dir, f"{v}_shapiro_pval.png")
            plot_2d_map(p_values, lat, lon, v, "Surface", out_path)
            
        elif data.ndim == 4:
            # Shape is (Nens, levels, lat, lon)
            n_levels = data.shape[1]
            all_levels_p_values = []
            for lev in range(n_levels):
                if v == 'TRG1' and lev in [0, 1]:
                    print(f"  Skipping level {lev} for {v}")
                    continue
                    
                level_dir = os.path.join(out_dir, f"Level_{lev}")
                os.makedirs(level_dir, exist_ok=True)
                
                p_values = np.zeros((data.shape[2], data.shape[3]))
                for i in range(data.shape[2]):
                    for j in range(data.shape[3]):
                        samples = data[:, lev, i, j]
                        
                        # Sometimes values at boundary or poles are constant (variance=0)
                        if np.std(samples) < 1e-12:
                            p = 1.0 # Constant data is effectively perfectly normal for numerical purposes
                        else:
                            stat, p = shapiro(samples)
                        p_values[i, j] = p
                
                all_levels_p_values.append(p_values)
                out_path = os.path.join(level_dir, f"{v}_shapiro_pval.png")
                plot_2d_map(p_values, lat, lon, v, f"Level {lev}", out_path)
            
            # Calculate and plot the level-averaged p-values
            all_levels_p_values = np.array(all_levels_p_values)
            level_avg_p_values = np.mean(all_levels_p_values, axis=0)
            
            level_avg_dir = os.path.join(out_dir, "Level_Averaged")
            os.makedirs(level_avg_dir, exist_ok=True)
            out_path_avg = os.path.join(level_avg_dir, f"{v}_shapiro_pval_avg.png")
            
            global_median_p = np.median(all_levels_p_values)
            print(f"  -> Global median p-value for {v} (all levels pooled): {global_median_p:.4f}")
            print(f"  -> Median of level-averaged p-values for {v}: {np.median(level_avg_p_values):.4f}")
            
            plot_2d_map(level_avg_p_values, lat, lon, v, "Level-Averaged", out_path_avg)

    print(f"Done! All plots saved to {out_dir}")

if __name__ == "__main__":
    main()
