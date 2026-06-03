#!/usr/bin/env python3
"""
LETKF localization-radius (r) tuning sweep.

Two subcommands:

  submit   Generate one runner CSV per r value from a fixed observation
           template (only `r` and `exp_settings` are overridden), submit one
           sbatch job per r running amlcs_da.py (LETKF) followed by an organize
           step, record a manifest, and submit a dependent collection job that
           runs once all sweeps finish.

  organize Move one run's unified_cycle NetCDF files into
           <run_folder>/<name>/data/. Runs automatically at the end of each r's
           sbatch job (can also be run standalone).

  collect  Read the manifest, move unified_cycle NetCDF files into
           <run_folder>/<name>/data/ (postprocessing), compute the
           level-averaged analysis RMSE per variable for each run (the same way
           error_plots_dual_nc.py does it), reduce each per-cycle series to its
           average (mean over cycles) and lowest (min over cycles) value, and
           write a summary CSV plus print the best r.

The idea: hold the observation configuration fixed (e.g. everything observed
with arctan) and sweep r to find the value that minimizes analysis RMSE.

Examples
--------
    python letkf_r_tuning.py submit \
        --template letkf_runner_nonlinear_sq.csv \
        --r-values 2,4,6,8,10 \
        --exp-settings ../LETKF_tuning/t21_80_0.05_30/ \
        --name arctan

    python letkf_r_tuning.py collect letkf_tuning_runs/arctan/manifest.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
# netCDF4 is imported lazily inside the collect helpers so that `submit` works
# in environments where netCDF4 is not installed.

# Directory containing this script (amlcs/). All relative paths used by
# amlcs_da.py (e.g. ../runs/, ../LETKF_tuning/) are resolved against it.
SCRIPT_DIR = Path(__file__).resolve().parent

# Variables compared for RMSE (model variables), matching error_plots_dual_nc.py
VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]

# Default sbatch options, mirroring sbatch_runner.sh
DEFAULT_SBATCH = {
    "account": "chipilskigroup_q",
    "partition": "chipilskigroup_q",
    "time": "12:00:00",
    "mem": "12G",
}


###############################################################################
# ----------------------------- shared helpers -------------------------------
###############################################################################

def _levels_for_var(var):
    """Levels analysed for a given variable (matches error_plots_dual_nc.py)."""
    if "PSG" in var:
        return [0]
    if var.startswith("TRG"):
        return list(range(2, 8))
    return list(range(8))


def _compute_l2_error(field1, field2):
    """L2 (RMSE) error between two fields."""
    diff = np.asarray(field1) - np.asarray(field2)
    return float(np.sqrt(np.mean(diff ** 2)))


def _read_truth_field(nc_path, var, lev):
    """
    Read a truth field from a reference_solution NetCDF file.

    Reference files store raw variable names (TG1 as 3D, PSG1 as 2D, etc.).
    Mirrors the raw-variable branch of error_plots_dual_nc.py::_read_nc_field.
    """
    from netCDF4 import Dataset
    with Dataset(nc_path, "r") as nc:
        if var in nc.variables:
            data = nc.variables[var]
            if data.ndim == 3:      # (nlev, lat, lon)
                return data[lev, :, :]
            if data.ndim == 2:      # (lat, lon) e.g. PSG
                return data[:]
            if data.ndim == 4:      # (ntr, lev, lat, lon)
                return data[0, lev, :, :]
        raise KeyError(f"Truth field {var} (lev {lev}) not found in {nc_path}")


def _read_analysis_field(nc_path, var, lev):
    """Read the analysis-mean field (xa_mean_<var>_lev<lev>) from a cycle file."""
    from netCDF4 import Dataset
    with Dataset(nc_path, "r") as nc:
        field_name = f"xa_mean_{var}_lev{lev}"
        if field_name not in nc.variables:
            raise KeyError(f"{field_name} not found in {nc_path}")
        return nc.variables[field_name][:]


def _resolve(path_str):
    """Resolve a path relative to the script directory (amlcs/)."""
    p = Path(path_str)
    if not p.is_absolute():
        p = (SCRIPT_DIR / p)
    return p.resolve()


def _cycle_data_dir(run_folder, campaign_name):
    """Target directory for unified_cycle files: <run_folder>/<name>/data/."""
    return Path(run_folder) / campaign_name / "data"


def _organize_run_cycle_files(run_folder, campaign_name):
    """
    Move unified_cycle*.nc from the run directory root into <name>/data/.

    amlcs_da writes cycles directly under the run folder; this step mirrors the
    layout used elsewhere (e.g. wdg_only/data/unified_cycle0.nc).
    """
    run_folder = Path(run_folder)
    dest = _cycle_data_dir(run_folder, campaign_name)
    dest.mkdir(parents=True, exist_ok=True)

    moved = 0
    for src in sorted(run_folder.glob("unified_cycle*.nc")):
        if src.parent != run_folder:
            continue
        target = dest / src.name
        if target.exists():
            continue
        shutil.move(str(src), str(target))
        moved += 1
    return dest, moved


def organize(args):
    """Move one run's unified_cycle files into <run_folder>/<name>/data/.

    Intended to run inside each r's sbatch job right after amlcs_da.py finishes,
    so subfolders are created per r as soon as that assimilation completes.
    """
    run_folder = Path(args.run_folder)
    dest, moved = _organize_run_cycle_files(run_folder, args.name)
    print(f"organize: moved {moved} unified_cycle file(s) -> {dest}")


###############################################################################
# -------------------------------- submit ------------------------------------
###############################################################################

def _read_config_value(exp_settings_dir, key):
    cfg = pd.read_csv(Path(exp_settings_dir) / "config.csv")
    return cfg[key].iloc[0]


def _run_folder_name(template_df, code_path):
    """Reproduce amlcs_da.py's method_path naming (minus the r token)."""
    s = int(template_df["s"].iloc[0])
    infla = float(template_df["infla"].iloc[0])
    method = str(template_df["method"].iloc[0]).strip()

    def _make(r):
        return f"{code_path}_{method}_{int(r)}_{s}_{int(100 * infla)}"

    return _make


def _parse_job_id(sbatch_stdout):
    """Extract the numeric job id from `sbatch` output."""
    # Typical output: "Submitted batch job 1234567"
    for token in sbatch_stdout.split():
        if token.isdigit():
            return token
    return None


def submit(args):
    template_path = _resolve(args.template)
    if not template_path.exists():
        sys.exit(f"Template runner CSV not found: {template_path}")

    exp_settings_resolved = _resolve(args.exp_settings)
    if not (exp_settings_resolved / "config.csv").exists():
        sys.exit(f"exp_settings/config.csv not found under: {exp_settings_resolved}")

    r_values = [int(v.strip()) for v in args.r_values.split(",") if v.strip() != ""]
    if not r_values:
        sys.exit("No r values provided (use --r-values 2,4,6,...).")

    code_path = str(_read_config_value(exp_settings_resolved, "code_path"))
    M = int(_read_config_value(exp_settings_resolved, "M"))

    template_df = pd.read_csv(template_path)
    folder_for = _run_folder_name(template_df, code_path)

    # Campaign output layout: amlcs/letkf_tuning_runs/<name>/
    campaign_dir = SCRIPT_DIR / "letkf_tuning_runs" / args.name
    configs_dir = campaign_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    runs_root = (SCRIPT_DIR / ".." / "runs").resolve()

    have_sbatch = shutil.which("sbatch") is not None
    if not have_sbatch:
        print("WARNING: `sbatch` not found on PATH. Configs and manifest will be "
              "generated, but no jobs will be submitted.")

    runs = []
    run_job_ids = []
    for r in r_values:
        df = template_df.copy()
        df.loc[:, "r"] = r
        # exp_settings is stored as-is (relative to amlcs) so amlcs_da.py
        # resolves it the same way it resolves ../runs/.
        df.loc[:, "exp_settings"] = args.exp_settings

        cfg_path = configs_dir / f"letkf_r{r}.csv"
        df.to_csv(cfg_path, index=False)

        run_folder = runs_root / folder_for(r)

        # Path passed to run_py.sh / amlcs_da.py, relative to amlcs/.
        cfg_rel = os.path.relpath(cfg_path, SCRIPT_DIR)

        job_id = None
        if have_sbatch:
            # After the DA run finishes, organize this run's cycle files into
            # <run_folder>/<name>/data/ within the same job.
            organize_cmd = (f'./run_py.sh letkf_r_tuning.py organize '
                            f'"{run_folder}" --name {args.name}')
            wrap = f'./run_py.sh amlcs_da.py {cfg_rel} && {organize_cmd}'
            cmd = [
                "sbatch",
                f"--account={args.account}",
                f"--partition={args.partition}",
                f"--time={args.time}",
                f"--mem={args.mem}",
                "--nodes=1",
                "--ntasks-per-node=1",
                f"--job-name=letkf_r{r}",
                f"--wrap={wrap}",
            ]
            result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
            print(result.stdout.strip())
            if result.returncode != 0:
                print(result.stderr.strip(), file=sys.stderr)
                sys.exit(f"sbatch submission failed for r={r}")
            job_id = _parse_job_id(result.stdout)
            if job_id:
                run_job_ids.append(job_id)

        data_dir = run_folder / args.name / "data"
        runs.append({
            "r": r,
            "config": str(cfg_path),
            "config_rel": cfg_rel,
            "run_folder": str(run_folder),
            "data_dir": str(data_dir),
            "job_id": job_id,
        })
        print(f"  r={r}: config={cfg_rel} -> run_folder={run_folder} job_id={job_id}")

    manifest = {
        "name": args.name,
        "template": str(template_path),
        "exp_settings": args.exp_settings,
        "exp_settings_resolved": str(exp_settings_resolved),
        "snapshots_dir": str(exp_settings_resolved / "snapshots"),
        "code_path": code_path,
        "M": M,
        "vars": VARS,
        "runs": runs,
    }
    manifest_path = campaign_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")

    # Dependent collection job: runs once every sweep job completes OK.
    if have_sbatch and run_job_ids:
        manifest_rel = os.path.relpath(manifest_path, SCRIPT_DIR)
        dep = ":".join(run_job_ids)
        wrap = f'./run_py.sh letkf_r_tuning.py collect {manifest_rel}'
        cmd = [
            "sbatch",
            f"--account={args.account}",
            f"--partition={args.partition}",
            "--time=01:00:00",
            f"--mem={args.mem}",
            "--nodes=1",
            "--ntasks-per-node=1",
            f"--job-name=letkf_r_collect_{args.name}",
            f"--dependency=afterok:{dep}",
            f"--wrap={wrap}",
        ]
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
        print(result.stdout.strip())
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            print("WARNING: failed to submit dependent collection job. Run "
                  f"`python letkf_r_tuning.py collect {manifest_rel}` manually "
                  "after the sweeps finish.", file=sys.stderr)
        else:
            print(f"Collection job submitted (afterok:{dep}).")
    else:
        manifest_rel = os.path.relpath(manifest_path, SCRIPT_DIR)
        print("\nNo collection job submitted. After the runs finish, run:")
        print(f"  python letkf_r_tuning.py collect {manifest_rel}")


###############################################################################
# -------------------------------- collect -----------------------------------
###############################################################################

def _available_cycles(data_dir, M):
    """Cycle indices for which a unified_cycle file exists (0..M-1 superset)."""
    data_dir = Path(data_dir)
    cycles = []
    for k in range(M):
        if (data_dir / f"unified_cycle{k}.nc").exists():
            cycles.append(k)
    return cycles


def _level_averaged_analysis_series(data_dir, snapshots_dir, var, cycles):
    """
    Per-cycle level-averaged analysis RMSE series for one variable.

    For each level: L2(xa_mean, truth); average across levels per cycle.
    Mirrors the level-averaged path of error_plots_dual_nc.py.
    """
    levels = _levels_for_var(var)
    data_dir = Path(data_dir)
    snapshots_dir = Path(snapshots_dir)

    per_level = []
    for lev in levels:
        errs = []
        for k in cycles:
            cycle_file = data_dir / f"unified_cycle{k}.nc"
            truth_file = snapshots_dir / f"reference_solution_{k}.nc"
            try:
                ana = _read_analysis_field(cycle_file, var, lev)
                truth = _read_truth_field(truth_file, var, lev)
                errs.append(_compute_l2_error(ana, truth))
            except (FileNotFoundError, KeyError, OSError):
                errs.append(np.nan)
        per_level.append(np.array(errs, dtype=float))

    if not per_level:
        return np.array([])

    return np.nanmean(np.vstack(per_level), axis=0)


def collect(args):
    manifest_path = _resolve(args.manifest)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    snapshots_dir = manifest["snapshots_dir"]
    M = int(manifest["M"])
    vars_list = manifest.get("vars", VARS)
    campaign_name = manifest["name"]
    campaign_dir = manifest_path.parent

    rows = []
    for run in sorted(manifest["runs"], key=lambda d: d["r"]):
        r = run["r"]
        run_folder = run["run_folder"]
        data_dir, moved = _organize_run_cycle_files(run_folder, campaign_name)
        if moved:
            print(f"r={r}: moved {moved} unified_cycle file(s) -> {data_dir}")
        cycles = _available_cycles(data_dir, M)

        row = {"r": r, "run_folder": run_folder, "data_dir": str(data_dir), "n_cycles": len(cycles)}
        if not cycles:
            print(f"r={r}: no unified_cycle*.nc files found in {data_dir} (skipping).")
            for var in vars_list:
                row[f"{var}_avg"] = np.nan
                row[f"{var}_min"] = np.nan
            row["overall_avg"] = np.nan
            rows.append(row)
            continue

        var_avgs = []
        for var in vars_list:
            series = _level_averaged_analysis_series(data_dir, snapshots_dir, var, cycles)
            if series.size == 0 or np.all(np.isnan(series)):
                avg = np.nan
                lowest = np.nan
            else:
                avg = float(np.nanmean(series))
                lowest = float(np.nanmin(series))
            row[f"{var}_avg"] = avg
            row[f"{var}_min"] = lowest
            if not np.isnan(avg):
                var_avgs.append(avg)

        row["overall_avg"] = float(np.mean(var_avgs)) if var_avgs else np.nan
        rows.append(row)
        print(f"r={r}: cycles={len(cycles)} overall_avg={row['overall_avg']:.6g}")

    df = pd.DataFrame(rows)
    summary_path = campaign_dir / "rmse_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSummary written: {summary_path}")

    # Best r by per-variable wins (scale-invariant): the r that achieves the
    # lowest average RMSE for the greatest number of variables. Ties are broken
    # by mean rank across variables, then by smaller r.
    wins = {int(r): 0 for r in df["r"]}
    rank_sums = {int(r): 0.0 for r in df["r"]}
    rank_counts = {int(r): 0 for r in df["r"]}
    n_scored = 0
    for var in vars_list:
        col = f"{var}_avg"
        if col not in df.columns:
            continue
        sub = df[["r", col]].dropna(subset=[col])
        if sub.empty:
            continue
        n_scored += 1
        ranks = sub[col].rank(method="min", ascending=True)
        for r_val, rk in zip(sub["r"], ranks):
            rank_sums[int(r_val)] += float(rk)
            rank_counts[int(r_val)] += 1
        wins[int(sub.loc[sub[col].idxmin(), "r"])] += 1

    if n_scored > 0 and any(wins.values()):
        mean_rank = {r: (rank_sums[r] / rank_counts[r]) if rank_counts[r] else np.inf
                     for r in wins}
        best_r = sorted(wins, key=lambda r: (-wins[r], mean_rank[r], r))[0]
        print(f"Best r (most per-variable wins): r={best_r} "
              f"(wins {wins[best_r]}/{n_scored} variables, mean rank {mean_rank[best_r]:.3g})")
        print("  Run `python letkf_best_r.py "
              f"{os.path.relpath(summary_path, SCRIPT_DIR)}` for full diagnostics.")
    else:
        print("No valid runs to rank (no readable cycle files).")


###############################################################################
# --------------------------------- CLI --------------------------------------
###############################################################################

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sub = sub.add_parser("submit", help="Generate configs and submit the r sweep.")
    p_sub.add_argument("--template", required=True,
                       help="Path to a template LETKF runner CSV (fixed obs config).")
    p_sub.add_argument("--r-values", required=True,
                       help="Comma-separated list of localization radii, e.g. 2,4,6,8.")
    p_sub.add_argument("--exp-settings", default="../LETKF_tuning/t21_80_0.05_30/",
                       help="exp_settings folder (truth/free_run source), relative to amlcs/.")
    p_sub.add_argument("--name", default="arctan",
                       help="Campaign name; outputs go to letkf_tuning_runs/<name>/.")
    p_sub.add_argument("--account", default=DEFAULT_SBATCH["account"])
    p_sub.add_argument("--partition", default=DEFAULT_SBATCH["partition"])
    p_sub.add_argument("--time", default=DEFAULT_SBATCH["time"])
    p_sub.add_argument("--mem", default=DEFAULT_SBATCH["mem"])
    p_sub.set_defaults(func=submit)

    p_org = sub.add_parser("organize",
                           help="Move a single run's unified_cycle NetCDFs into <run_folder>/<name>/data/.")
    p_org.add_argument("run_folder", help="Run directory containing unified_cycle*.nc files.")
    p_org.add_argument("--name", required=True, help="Campaign name (subfolder under the run directory).")
    p_org.set_defaults(func=organize)

    p_col = sub.add_parser("collect",
                           help="Organize cycle NetCDFs into <run>/<name>/data and compute RMSE summary.")
    p_col.add_argument("manifest", help="Path to manifest.json produced by submit.")
    p_col.set_defaults(func=collect)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
