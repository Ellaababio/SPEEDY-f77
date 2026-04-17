import numpy as np
import xarray as xr
import sys

def main():
    filepath = '/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_20/free_run/free_run_0.nc'
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        
    ds = xr.open_dataset(filepath)
    
    print(f"Calculating 1% of level-averaged dynamical range for variables in {filepath}\n")
    
    for varname, var in ds.data_vars.items():
        if np.issubdtype(var.dtype, np.number):
            dims = var.dims
            
            # identify vertical level dimension
            lev_dim = None
            for d in dims:
                if d.lower() in ['lev', 'level', 'z', 'sigma']:
                    lev_dim = d
                    break
            
            if lev_dim is not None:
                reduce_dims = [d for d in dims if d != lev_dim]
                if reduce_dims:
                    dyn_range_per_level = var.max(dim=reduce_dims) - var.min(dim=reduce_dims)
                else:
                    dyn_range_per_level = var.max() - var.min()
                avg_dyn_range = float(dyn_range_per_level.mean().values)
            else:
                # No level dimension
                avg_dyn_range = float((var.max() - var.min()).values)
                
            err_obs = 0.01 * avg_dyn_range
            print(f"Variable: {varname:<10} | err_obs: {err_obs:<12.6g} (dims: {dims})")

if __name__ == '__main__':
    main()
