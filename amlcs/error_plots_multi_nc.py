#!/usr/bin/env python3
"""
Multi-run error plotting: Compare N methods vs NoDA.
Reads NetCDF cycle files directly.

NO COMMAND-LINE ARGUMENTS. Everything is configured in the USER SETTINGS
section below.

Generates:
  (A) Per-level plots (each level per figure) -> saved in /levels subdirectory
  (B) Level-averaged plots (for 3D variables) -> saved in top level
  (C) Surface Pressure (PSG) plots -> saved in top level
"""

###############################################################################
# ======================= USER SETTINGS (EDIT THESE) ==========================
###############################################################################

# LIST of FULL PATHS to the experiment directories you want to compare:
EXPERIMENTS = [
    "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/linear_no_norm_results_v2",
    "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/linear_normalization_results",
    "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/nonlinear_normalization_results",
]

# SPEEDY resolution:
RESOLUTION = "t21"

# Number of assimilation cycles to read (0-based indexing)
CYCLES = list(range(5))  # [0, 1, 2, 3, 4]

# Variables to compare:
VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]

# Anchor mode: "step0" or "step1"
ANCHOR = "step1"

# Scale mode: "log", "linear", or "both"
SCALE_MODE = "linear"

# Generate log plots? (Set False for absolute-only plots)
GENERATE_LOG_PLOTS = True

# Output directory name (optional)
# If None -> "multi_method_comparison"
PLOT_DIR_NAME = '/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100/norm_linear_multicomparison'

# Free run directory (NoDA baseline)
FREE_RUN_DIR = "/gpfs/home/jjs21b/AMLCS/ENSF_gaussian_check/t21_50_0.05_5/free_run"

###############################################################################
# ======================= END USER SETTINGS ==================================
###############################################################################

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib

# Formatting
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})

# Pretty variable names
VAR_CODES = {
    "TG0": "T_0", "UG0": "u_0", "VG0": "v_0", "TRG0": "Hq_0", "PSG0": "PS_0",
    "TG1": "T_1", "UG1": "u_1", "VG1": "v_1", "TRG1": "Hq_1", "PSG1": "PS_1",
}
PS_LEVELS_MB = [30, 100, 200, 300, 500, 700, 850, 925]

# Auto-assigned color palette (NoDA is always black)
# Extended palette for more methods
COLOR_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
]
COLOR_NAMES = [
    "blue", "orange", "green", "red", "purple", 
    "brown", "pink", "gray", "olive", "cyan"
]

###############################################################################
# --------------------- Utility Functions ------------------------------------
###############################################################################

def _parse_experiment_path(exp_path: Path) -> dict:
    """
    Parse experiment path to extract method and metadata.
    Handles various naming conventions to ensure unique labels.
    """
    full_path_str = str(exp_path)
    
    # Look for common metadata patterns in the path
    metadata_hints = []
    
    # Check for subdirectory metadata (e.g. if pointing to a results subdir)
    if exp_path.parent != exp_path:
        subdir = exp_path.name
        if "linear" in subdir.lower() and "nonlinear" not in subdir.lower():
            metadata_hints.append("linear obs")
        elif "nonlinear" in subdir.lower():
            metadata_hints.append("nonlinear obs")
        
        if "normalization" in subdir.lower() or "normalize" in subdir.lower():
            metadata_hints.append("normalization")
        elif "no_norm" in subdir.lower() or "unnorm" in subdir.lower():
            metadata_hints.append("no normalization")

    # Extract method name from base directory name
    # If exp_path is a subdir like '.../runs/EXP_NAME/results', go up to EXP_NAME
    if "t21" in exp_path.name:
        base_dir = exp_path
    elif "t21" in exp_path.parent.name:
        base_dir = exp_path.parent
    else:
        # Fallback: assume the provided path is the experiment root
        base_dir = exp_path

    name = base_dir.name.rstrip("/")
    parts = name.split("_")
    
    method = None
    # Heuristic: Try to find the method name in the standard position
    # Standard format: t21_50_0.05_5_METHOD_...
    if len(parts) > 5:
        # Try to identify where the method name starts
        # Usually after the 4th underscore (t21, 50, 0.05, 5)
        # But sometimes parameters vary.
        
        # Let's look for known method names or non-numeric strings
        known_methods = ["ReverseSDE", "EnKF", "LETKF", "LEnKF", "Climatology", "3DVar"]
        
        found_known = False
        for m in known_methods:
            if m in name:
                # Extract the chunk that contains the method
                # This is a bit tricky if we want exact parsing, but let's try to be smart
                pass

        # Fallback: take the middle part
        if len(parts) > 7:
             mid = parts[4:-3]
             if mid:
                 method = "_".join(mid)
    
    # Fallback method extraction if the above failed or produced garbage
    if not method:
        # Try to find the first non-numeric part after the resolution/params
        for i, token in enumerate(parts):
            if i < 3: continue # skip t21, 50, 0.05
            try:
                float(token)
            except ValueError:
                # This might be the start of the method
                # Collect until we hit numbers again at the end?
                # Simpler: just use the token
                method = token
                # If there are more non-numeric tokens following, append them?
                j = i + 1
                while j < len(parts):
                    try:
                        float(parts[j])
                        break # hit a number
                    except ValueError:
                        method += "_" + parts[j]
                        j += 1
                break

    if not method:
        method = name # Give up and use the full name

    # Refine metadata based on the name itself if not found in subdir
    if "linear" in name.lower() and "nonlinear" not in name.lower() and "linear obs" not in metadata_hints:
        metadata_hints.append("linear")
    if "nonlinear" in name.lower() and "nonlinear obs" not in metadata_hints:
        metadata_hints.append("nonlinear")
    
    metadata = ", ".join(metadata_hints) if metadata_hints else ""
    
    return {
        "method": method,
        "metadata": metadata,
        "full_label": f"{method} ({metadata})" if metadata else method,
        "path": exp_path
    }

def _compute_rmse(field1: np.ndarray, field2: np.ndarray) -> float:
    """Compute Root Mean Square Error (RMSE) between two fields."""
    diff = field1 - field2
    return np.sqrt(np.mean(diff**2))

def _read_nc_field(nc_path: Path, var: str, lev: int) -> np.ndarray:
    """
    Read a specific field from a NetCDF file.
    
    Handles three formats:
    1. Per-level variables: xa_mean_TG1_lev0, truth_TG1_lev0, etc. (cycle files)
    2. Multi-dimensional arrays: TG1[lev, lat, lon] (truth/NoDA files)
    3. Tracer variables: TRG1[tracer, lev, lat, lon] (truth/NoDA files for humidity)
    """
    with Dataset(nc_path, 'r') as nc:
        # First, try per-level variable names (cycle files)
        for prefix in ["xa_mean", "xb_mean", "truth", "noda"]:
            field_name = f"{prefix}_{var}_lev{lev}"
            if field_name in nc.variables:
                return nc.variables[field_name][:]
        
        # Second, try multi-dimensional arrays (truth/NoDA files)
        # Just the variable name without prefix or level suffix
        if var in nc.variables:
            var_data = nc.variables[var]
            ndims = len(var_data.shape)
            
            if ndims == 4:  # (tracer, level, lat, lon) - for TRG variables
                # Take first tracer (index 0) and the specified level
                return var_data[0, lev, :, :]
            elif ndims == 3:  # (level, lat, lon)
                return var_data[lev, :, :]
            elif ndims == 2:  # (lat, lon) - for single-level vars like PSG
                if lev == 0:
                    return var_data[:, :]
        
        raise KeyError(f"Field {var}_lev{lev} not found in {nc_path}")

def _compute_noda_series(exp_path: Path, var: str, lev: int, cycles: list) -> np.ndarray:
    """Compute NoDA error series from free_run snapshots."""
    errors = []
    
    free_run_dir = Path(FREE_RUN_DIR)
    if not free_run_dir.exists():
        print(f"Warning: Free run directory not found: {free_run_dir}")
        return np.full(len(cycles), np.nan)
    
    # Truth directory is assumed to be parallel to free_run
    truth_dir = free_run_dir.parent / "snapshots"
    if not truth_dir.exists():
         print(f"Warning: Truth directory derived from FREE_RUN_DIR not found: {truth_dir}")
    
    for cycle_k in cycles:
        try:
            truth_file = truth_dir / f"reference_solution_{cycle_k}.nc"
            noda_file = free_run_dir / f"free_run_{cycle_k}.nc"
            
            truth = _read_nc_field(truth_file, var, lev)
            noda = _read_nc_field(noda_file, var, lev)
            
            error = _compute_rmse(noda, truth)
            errors.append(error)
        except (FileNotFoundError, KeyError) as e:
            # print(f"Warning: Could not compute NoDA for cycle {cycle_k}: {e}")
            errors.append(np.nan)
    
    return np.array(errors)

def _compute_error_series(exp_path: Path, method: str, var: str, lev: int, cycles: list) -> np.ndarray:
    """Compute error series for analysis (xa_mean)."""
    errors = []
    
    # Use the global FREE_RUN_DIR to find the truth directory
    # This ensures consistency with the NoDA calculation
    if 'FREE_RUN_DIR' in globals() and FREE_RUN_DIR:
        truth_dir = Path(FREE_RUN_DIR).parent / "snapshots"
    else:
        # Fallback (should not happen with current config)
        truth_dir = exp_path.parent.parent / "ENSF_gaussian_check" / exp_path.name / "snapshots"

    if not truth_dir.exists():
        print(f"Warning: Truth directory not found: {truth_dir}")
            
    for cycle_k in cycles:
        try:
            # Find the cycle file
            # Try various naming conventions
            possible_names = [
                f"reverseSDE_cycle{cycle_k}.nc",
                f"enkf_cycle{cycle_k}.nc",
                f"{method.lower()}_cycle{cycle_k}.nc",
                f"cycle{cycle_k}.nc"
            ]
            
            cycle_file = None
            for name in possible_names:
                if (exp_path / name).exists():
                    cycle_file = exp_path / name
                    break
            
            if not cycle_file:
                print(f"Warning: Cycle file for {method} step {cycle_k} not found in {exp_path}")
                errors.append(np.nan)
                continue
            
            if not truth_dir or not truth_dir.exists():
                 # If we can't find truth, we can't compute error
                 # print(f"Warning: Truth dir not found for cycle {cycle_k}")
                 errors.append(np.nan)
                 continue

            truth_file = truth_dir / f"reference_solution_{cycle_k}.nc"
            if not truth_file.exists():
                print(f"Warning: Truth file not found: {truth_file}")
                errors.append(np.nan)
                continue

            truth = _read_nc_field(truth_file, var, lev)
            
            # Read analysis field
            with Dataset(cycle_file, 'r') as nc:
                # Try xa_mean first
                field_name = f"xa_mean_{var}_lev{lev}"
                if field_name not in nc.variables:
                    # Fallback? Maybe just 'var' if it's a simple dump?
                    # But usually it's xa_mean for analysis
                    raise KeyError(f"{field_name} not found")
                field = nc.variables[field_name][:]
            
            error = _compute_rmse(field, truth)
            errors.append(error)
            
        except (FileNotFoundError, KeyError) as e:
            # print(f"Warning: Error computing analysis error for cycle {cycle_k}: {e}")
            errors.append(np.nan)
    
    return np.array(errors)

def _make_anchor(series, anchor_val):
    if anchor_val is None:
        return series
    if len(series) == 0:
        return series
    return np.concatenate([np.array([anchor_val]), series])

def _multi_curves(method_series_map, noda_series, anchor_mode, scale):
    """
    Prepare curves for plotting.
    method_series_map: dict of {label: series}
    noda_series: numpy array
    """
    eps = 1e-12
    
    # Find minimum length
    lens = [len(s) for s in method_series_map.values()]
    if noda_series is not None:
        lens.append(len(noda_series))
    
    if not lens:
        return None, {}
        
    L = min(lens)
    if L == 0:
        return None, {}

    # Truncate all to L
    truncated_map = {k: v[:L] for k, v in method_series_map.items()}
    if noda_series is not None:
        noda_series = noda_series[:L]
    
    # Determine anchor
    anchor_val = None
    if anchor_mode == "step1" and noda_series is not None:
        anchor_val = noda_series[0]
    
    # Apply anchor
    anchored_map = {k: _make_anchor(v, anchor_val) for k, v in truncated_map.items()}
    if noda_series is not None:
        anchored_noda = _make_anchor(noda_series, anchor_val)
    else:
        anchored_noda = None

    # X axis
    xs = np.arange(len(list(anchored_map.values())[0]))

    curves = {}
    if anchored_noda is not None:
        if scale == "log":
            curves["NoDA"] = np.log(anchored_noda + eps)
        else:
            curves["NoDA"] = anchored_noda
            
    for label, series in anchored_map.items():
        if scale == "log":
            curves[label] = np.log(series + eps)
        else:
            curves[label] = series
            
    return xs, curves

def _plot_multi_curves(xs, curves, title, out_path, scale, exp_infos):
    """
    Plot N methods + NoDA.
    exp_infos: list of dicts, used to map labels to colors.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot NoDA first (black)
    if "NoDA" in curves:
        plt.plot(xs, curves["NoDA"], label="NoDA", color="k", linestyle="-", linewidth=2, zorder=10)
    
    # Plot methods
    # We want to maintain a consistent color for each experiment index
    # exp_infos should be in the same order as EXPERIMENTS
    
    for i, info in enumerate(exp_infos):
        label = info['full_label']
        if label in curves:
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            plt.plot(xs, curves[label], label=label, color=color, linestyle="-", linewidth=1.5)
            
    plt.title(title, fontsize=14)
    plt.xlabel("Assimilation Step")
    plt.ylabel("log(RMSE)" if scale == "log" else "RMSE")
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    # Legend outside if too many items? For now, best location
    plt.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def _levels_for_var(var):
    if "PSG" in var:
        return [0]
    if var.startswith("TRG"):
        return list(range(2, 8))
    return list(range(8))

###############################################################################
# --------------------------- Main Work ---------------------------------------
###############################################################################

def run_multi_plots():
    # 1. Parse all experiments
    exp_infos = []
    for p in EXPERIMENTS:
        path = Path(p).resolve()
        info = _parse_experiment_path(path)
        exp_infos.append(info)
        print(f"Loaded: {info['full_label']} -> {path}")

    # 2. Setup output directory
    out_name = PLOT_DIR_NAME
    if not out_name:
        out_name = "multi_method_comparison"
    
    # We will save in the first experiment's plot directory? 
    # Or a neutral place? The user prompt implied creating a file "similar to this",
    # usually these scripts output to one of the run directories or a common place.
    # Let's assume we output to the FIRST experiment's plots folder for now, 
    # or just a local folder if that's safer. 
    # The original script did: root1 = exp1 / "plots" / "errors"
    # Let's do the same for the first experiment.
    
    base_out_root = exp_infos[0]['path'] / "plots" / "errors" / out_name
    print(f"Output directory: {base_out_root}")

    def _ensure(scale, subdir=None):
        if scale == "log":
            d = base_out_root / "log"
        else:
            d = base_out_root / "linear"
        
        if subdir:
            d = d / subdir
            
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Determine scales
    if SCALE_MODE == "both":
        scales = ["log", "linear"] if GENERATE_LOG_PLOTS else ["linear"]
    elif SCALE_MODE == "log" and not GENERATE_LOG_PLOTS:
        scales = ["linear"]
    else:
        scales = [SCALE_MODE]

    # 3. Loop over variables
    for var in VARS:
        lvls = _levels_for_var(var)
        is_psg = ("PSG" in var)
        
        # Store level-averaged data (method -> list of arrays)
        method_level_errors = {info['full_label']: [] for info in exp_infos}
        noda_level_errors = []

        # --- Per-Level Plots ---
        for lvl in lvls:
            print(f"Processing {var} level {lvl}...")
            
            # Compute NoDA (only need once, use first exp path to find it)
            # Assuming all experiments share the same truth/NoDA
            s_noda = _compute_noda_series(exp_infos[0]['path'], var, lvl, CYCLES)
            noda_level_errors.append(s_noda)
            
            # Compute methods
            current_level_map = {}
            for info in exp_infos:
                s_meth = _compute_error_series(info['path'], info['method'], var, lvl, CYCLES)
                current_level_map[info['full_label']] = s_meth
                method_level_errors[info['full_label']].append(s_meth)
            
            # Plot this level
            for scale in scales:
                out_dir = _ensure(scale, subdir="levels")
                xs, curves = _multi_curves(
                    current_level_map, s_noda, ANCHOR, scale
                )
                if xs is None:
                    continue
                
                mb = PS_LEVELS_MB[lvl] if lvl < len(PS_LEVELS_MB) else lvl
                title = f"{VAR_CODES.get(var,var)} (lev {lvl}, {mb} mb)"
                out_file = out_dir / f"multi_{var}_lev{lvl}.png"
                _plot_multi_curves(xs, curves, title, out_file, scale, exp_infos)

        # --- Aggregated Plots ---
        if is_psg:
            # PSG is single level, so the "level average" is just the level 0 plot.
            # But we want it in the top directory as "Pressure Plot" (or just PSG plot)
            # We can just copy the lev0 plot or re-plot it to the top dir.
            # Let's re-plot it to the top dir for convenience.
            print(f"Processing {var} (Surface Pressure)...")
            
            # Retrieve the data we just computed (it's at index 0)
            s_noda = noda_level_errors[0]
            current_level_map = {label: method_level_errors[label][0] for label in method_level_errors}
            
            for scale in scales:
                out_dir = _ensure(scale) # Top level
                xs, curves = _multi_curves(
                    current_level_map, s_noda, ANCHOR, scale
                )
                if xs is None: continue
                
                title = f"{VAR_CODES.get(var,var)} (Surface Pressure)"
                out_file = out_dir / f"multi_{var}_surface.png"
                _plot_multi_curves(xs, curves, title, out_file, scale, exp_infos)
                
        else:
            # Level Average
            print(f"Processing {var} (Level Average)...")
            
            # Average Noda
            # Stack: (n_levels, n_cycles)
            # Need to handle different lengths if any errors occurred (NaNs are fine, but length mismatch is annoying)
            # _multi_curves truncates to min length, let's do that here too
            
            def safe_avg(list_of_arrays):
                if not list_of_arrays: return None
                min_len = min(len(x) for x in list_of_arrays)
                if min_len == 0: return None
                stacked = np.vstack([x[:min_len] for x in list_of_arrays])
                return np.nanmean(stacked, axis=0)

            s_noda_avg = safe_avg(noda_level_errors)
            
            method_avgs = {}
            for label, arrays in method_level_errors.items():
                avg = safe_avg(arrays)
                if avg is not None:
                    method_avgs[label] = avg
            
            for scale in scales:
                out_dir = _ensure(scale) # Top level
                xs, curves = _multi_curves(
                    method_avgs, s_noda_avg, ANCHOR, scale
                )
                if xs is None: continue
                
                title = f"{VAR_CODES.get(var,var)} (Level Average)"
                out_file = out_dir / f"multi_{var}_levelavg.png"
                _plot_multi_curves(xs, curves, title, out_file, scale, exp_infos)

if __name__ == "__main__":
    run_multi_plots()
