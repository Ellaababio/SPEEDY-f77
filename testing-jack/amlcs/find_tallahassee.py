import numpy as np
from netCDF4 import Dataset

def get_gaussian_latitudes(nlat):
    # Approximate Gaussian latitudes using numpy's legendre polynomial roots
    # nlat is 32 for T21
    # roots are sin(lat)
    x, w = np.polynomial.legendre.leggauss(nlat)
    lats = np.arcsin(x) * 180 / np.pi
    return lats

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def main():
    nc_path = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/linear_no_norm_results_v2/reverseSDE_cycle0.nc"
    
    try:
        ds = Dataset(nc_path, 'r')
        print(f"Opened {nc_path}")
        print("Variables:", ds.variables.keys())
        
        if 'lat' in ds.variables and 'lon' in ds.variables:
            lats = ds.variables['lat'][:]
            lons = ds.variables['lon'][:]
            print("Read lat/lon from file.")
        else:
            print("lat/lon not found in file. Generating T21 grid.")
            # T21: 32 lat, 64 lon
            nlat = 32
            nlon = 64
            lats = get_gaussian_latitudes(nlat)
            lons = np.linspace(0, 360, nlon, endpoint=False)
            
        ds.close()
        
    except Exception as e:
        print(f"Error opening file: {e}")
        print("Generating T21 grid fallback.")
        nlat = 32
        nlon = 64
        lats = get_gaussian_latitudes(nlat)
        lons = np.linspace(0, 360, nlon, endpoint=False)

    # Tallahassee coordinates
    target_lat = 30.44
    target_lon = 360 - 84.28 # 275.72
    
    lat_idx, lat_val = find_nearest(lats, target_lat)
    lon_idx, lon_val = find_nearest(lons, target_lon)
    
    print(f"Target: Lat {target_lat}, Lon {target_lon}")
    print(f"Nearest: Lat {lat_val} (idx {lat_idx}), Lon {lon_val} (idx {lon_idx})")
    
    # Calculate 1D index if flattened (lat, lon)
    # Assuming row-major (C-style) flattening: index = lat_idx * nlon + lon_idx
    nlon = len(lons)
    flat_idx = lat_idx * nlon + lon_idx
    print(f"Flattened index (assuming lat*nlon + lon): {flat_idx}")

if __name__ == "__main__":
    main()
