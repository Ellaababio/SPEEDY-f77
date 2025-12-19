#!/usr/bin/env python3
"""
General-purpose NetCDF file inspector.
Displays metadata, variable information, and basic statistics.

Configure the file path in the CONFIGURATION section below and run:
    python inspect_netcdf.py
"""

import numpy as np
from netCDF4 import Dataset

###############################################################################
# ========================= CONFIGURATION ====================================
###############################################################################

# Path to the NetCDF file to inspect
NETCDF_FILE = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_20_EnKF_MC_obs_1_1_100/linear_results_ps_only/unified_cycle0.nc"

# Show sample values for small arrays?
SHOW_SAMPLE = False

###############################################################################
# ========================= END CONFIGURATION ================================
###############################################################################


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_global_metadata(nc):
    """Print global attributes (metadata)."""
    print_section("GLOBAL METADATA")
    
    attrs = nc.ncattrs()
    if not attrs:
        print("  No global attributes found.")
        return
    
    for attr_name in attrs:
        attr_value = getattr(nc, attr_name)
        print(f"  {attr_name}: {attr_value}")


def print_dimensions(nc):
    """Print dimension information."""
    print_section("DIMENSIONS")
    
    if not nc.dimensions:
        print("  No dimensions found.")
        return
    
    for dim_name, dim in nc.dimensions.items():
        size_str = f"{len(dim)}" if not dim.isunlimited() else f"{len(dim)} (UNLIMITED)"
        print(f"  {dim_name}: {size_str}")


def get_stats(data):
    """Calculate statistics for numerical data."""
    try:
        # Flatten the data
        flat_data = np.asarray(data).flatten()
        
        # Count total elements
        total = flat_data.size
        
        # Count NaNs
        nan_count = np.isnan(flat_data).sum()
        nan_pct = (nan_count / total * 100) if total > 0 else 0
        
        # Calculate statistics on non-NaN values
        valid_data = flat_data[~np.isnan(flat_data)]
        
        if valid_data.size > 0:
            stats = {
                'total': total,
                'nan_count': nan_count,
                'nan_pct': nan_pct,
                'valid_count': valid_data.size,
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'mean': np.mean(valid_data),
                'std': np.std(valid_data),
                'median': np.median(valid_data)
            }
        else:
            stats = {
                'total': total,
                'nan_count': nan_count,
                'nan_pct': nan_pct,
                'valid_count': 0,
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'median': np.nan
            }
        
        return stats
    except Exception as e:
        return {'error': str(e)}


def print_variables(nc, show_sample=False):
    """Print variable information and statistics."""
    print_section("VARIABLES")
    
    if not nc.variables:
        print("  No variables found.")
        return
    
    for var_name, var in nc.variables.items():
        print(f"\n  Variable: {var_name}")
        print(f"    Shape: {var.shape}")
        print(f"    Dimensions: {var.dimensions}")
        print(f"    Data Type: {var.dtype}")
        
        # Print variable attributes
        var_attrs = var.ncattrs()
        if var_attrs:
            print("    Attributes:")
            for attr_name in var_attrs:
                attr_value = getattr(var, attr_name)
                print(f"      {attr_name}: {attr_value}")
        
        # Calculate statistics for numerical data
        # Check if dtype has 'kind' attribute (numpy dtypes do, string types don't)
        if hasattr(var.dtype, 'kind') and var.dtype.kind in ['f', 'i', 'u']:  # float, int, unsigned int
            try:
                # For large arrays, sample or use chunking
                if var.size > 1e7:  # If more than 10M elements
                    print("    Statistics: (Large array - showing sample stats)")
                    # Sample the data
                    if len(var.shape) == 1:
                        sample = var[::max(1, var.size // 10000)]
                    elif len(var.shape) == 2:
                        step = max(1, var.shape[0] // 100)
                        sample = var[::step, ::step]
                    elif len(var.shape) == 3:
                        step = max(1, var.shape[0] // 10)
                        sample = var[::step, ::step, ::step]
                    else:
                        # Just take first slice for higher dimensions
                        sample = var[0]
                    
                    stats = get_stats(sample)
                else:
                    data = var[:]
                    stats = get_stats(data)
                
                if 'error' in stats:
                    print(f"    Statistics: Error - {stats['error']}")
                else:
                    print(f"    Statistics:")
                    print(f"      Total elements: {stats['total']}")
                    print(f"      Valid elements: {stats['valid_count']}")
                    print(f"      NaN count: {stats['nan_count']} ({stats['nan_pct']:.2f}%)")
                    if stats['valid_count'] > 0:
                        print(f"      Min: {stats['min']:.6e}")
                        print(f"      Max: {stats['max']:.6e}")
                        print(f"      Mean: {stats['mean']:.6e}")
                        print(f"      Std: {stats['std']:.6e}")
                        print(f"      Median: {stats['median']:.6e}")
                
                # Optionally show a sample of the data
                if show_sample and var.size <= 1000:
                    print(f"    Sample values: {var[:5] if var.size > 5 else var[:]}")
                    
            except Exception as e:
                print(f"    Statistics: Could not compute - {e}")
        else:
            print(f"    Statistics: Non-numerical data type")


def inspect_netcdf(filepath, show_sample=False):
    """Main inspection function."""
    try:
        print(f"\nInspecting NetCDF file: {filepath}")
        print(f"File size: {get_file_size(filepath)}")
        
        with Dataset(filepath, 'r') as nc:
            # Print global metadata
            print_global_metadata(nc)
            
            # Print dimensions
            print_dimensions(nc)
            
            # Print variables with statistics
            print_variables(nc, show_sample=show_sample)
            
        print("\n" + "=" * 80)
        print("  Inspection complete!")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\nError inspecting file: {e}")
        import traceback
        traceback.print_exc()
        return


def get_file_size(filepath):
    """Get human-readable file size."""
    import os
    size = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} PB"


def main():
    inspect_netcdf(NETCDF_FILE, show_sample=SHOW_SAMPLE)


if __name__ == "__main__":
    main()
