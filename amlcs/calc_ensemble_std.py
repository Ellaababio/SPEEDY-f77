import os, glob
import numpy as np
import xarray as xr

# ---------------- CONFIG ----------------
ensemble_dir   = r"/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_5/ensemble_0"
target_sigma_n = 0.4
scalefact      = 1.0
var_names      = ['UG0','VG0','TG0','TRG0','PSG0','UG1','VG1','TG1','TRG1','PSG1']
# ----------------------------------------

def load_ensemble(ensemble_dir):
    files = sorted(glob.glob(os.path.join(ensemble_dir, "ensemble_member_*.nc")))
    if not files:
        raise FileNotFoundError(f"No ensemble_member_*.nc in {ensemble_dir}")
    ds_list = []
    for i, f in enumerate(files):
        ds = xr.open_dataset(f)
        if "member" in ds.dims:
            ds = ds.assign_coords(member=[i])
        else:
            ds = ds.expand_dims({"member": [i]})
        ds_list.append(ds)
    print(f"Loaded {len(ds_list)} ensemble members from {ensemble_dir}")
    return xr.concat(ds_list, dim="member", combine_attrs="override")

def robust_std_scalar(da):
    if "tracer" in da.dims and da.sizes.get("tracer", 1) == 1:
        da = da.isel(tracer=0, drop=True)
    std_field = da.std(dim="member", skipna=True)
    return float(np.nanmedian(std_field.values))

def format_num(x):
    if np.isnan(x) or np.isinf(x):
        return "nan"
    if abs(x) == 0:
        return "0"
    if 1e-3 <= abs(x) < 1e3:
        s = f"{x:.6g}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return f"{x:.1e}"

if __name__ == "__main__":
    ds = load_ensemble(ensemble_dir)

    units_map = {
        'UG*': 'm s^-1',
        'VG*': 'm s^-1',
        'TG*': 'K',
        'TRG*': 'g kg^-1 (if DA uses kg/kg, divide by 1000)',
        'PSG*': 'dimensionless ln(ps/1e5 Pa)',
    }

    print("\n=== Derived spreads and err_obs ===")
    print(f"{'var':<6} {'std_ens':>12}  {'err_obs':>12}  units")

    err_map = {}
    for v in var_names:
        if v not in ds:
            err_map[v] = np.nan
            continue
        s = robust_std_scalar(ds[v])
        err_map[v] = (target_sigma_n / scalefact) * s

        if v.startswith('UG'): u = units_map['UG*']
        elif v.startswith('VG'): u = units_map['VG*']
        elif v.startswith('TG'): u = units_map['TG*']
        elif v.startswith('TRG'): u = units_map['TRG*']
        elif v.startswith('PSG'): u = units_map['PSG*']
        else: u = ''
        print(f"{v:<6} {format_num(s):>12}  {format_num(err_map[v]):>12}  {u}")

    # Copy/paste line
    line_vals = [format_num(err_map[v]) for v in var_names]
    paste_line = ", ".join(line_vals)
    print("\n=== COPY/PASTE (err_obs for DA, canonical order) ===")
    print(paste_line)
    print("=== /COPY/PASTE ===")
