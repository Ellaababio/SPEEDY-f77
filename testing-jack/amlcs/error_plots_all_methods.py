#!/usr/bin/env python3
"""
Compare absolute analysis errors across multiple DA methods, now including:
  - ReverseSDE (Nonlinear)
  - ReverseSDE (Linear)
  - LEnKF
  - LETKF
  - ENKF_MC_obs
  - NoDA baseline

All curves are anchored so that cycle 0 starts at the first NoDA value.

Outputs:
  <ensf_path>/plots/errors/<plot_dir_name>/
    abs_ana_<VAR>_levelavg.png
    abs_ana_<VAR>_lev<L>.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from grid_resolution import grid_resolution
from postpro_tools import postpro_tools

# =====================================================================
# --- USER PARAMETERS -------------------------------------------------
# =====================================================================
ENSF_PATH      = Path("../runs/t21_50_0.05_5_ReverseSDE_1_1_100")      # nonlinear
LINEAR_PATH    = ENSF_PATH / "linear_results"                          # linear version (CSVs directly here)
LENKF_PATH     = Path("../runs/t21_50_0.05_5_LEnKF_1_1_100")
LETKF_PATH     = Path("../runs/t21_50_0.05_5_LETKF_1_1_100")
ENKF_MC_PATH   = Path("../runs/t21_50_0.05_5_EnKF_MC_obs_1_1_100")

RESOLUTION     = "t21"     # grid name
M_CYCLES       = 5         # number of assimilation cycles
PLOT_DIR_NAME  = "comparison_abs_full_v3"
VARS_LIST      = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]

# =====================================================================
VAR_CODES = {
    "TG1": "T₁", "UG1": "u₁", "VG1": "v₁", "TRG1": "Hq₁", "PSG1": "PS₁",
}
PS_LEVELS_MB = [30, 100, 200, 300, 500, 700, 850, 925]

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _read_series(df: pd.DataFrame, lvl: int) -> np.ndarray:
    col = str(lvl)
    return df[col].to_numpy() if col in df.columns else np.array([])

def _levels_for_var(var: str) -> list[int]:
    if "PSG" in var:
        return [0]
    if var.startswith("TRG"):
        return list(range(2, 8))
    return list(range(8))

def _avg_over_levels(df: pd.DataFrame, lvls: list[int]) -> np.ndarray:
    vals = []
    for L in lvls:
        s = _read_series(df, L)
        if len(s):
            vals.append(s)
    if not vals:
        return np.array([])
    Lmin = min(len(v) for v in vals)
    return np.vstack([v[:Lmin] for v in vals]).mean(axis=0)

def _make_anchor(series: np.ndarray, anchor_value: float | None):
    if anchor_value is None:
        return series
    return np.concatenate([np.array([anchor_value], dtype=float), series])

def _plot_curves(xs, data, title, out_path):
    plt.figure(figsize=(9, 4))
    plt.title(title)
    colors = {
        "NoDA": "k",
        "ReverseSDE (Nonlinear)": "tab:blue",
        "ReverseSDE (Linear)": "tab:cyan",
        "LEnKF": "tab:orange",
        "LETKF": "tab:green",
        "ENKF_MC_obs": "tab:red",
    }
    for label, series in data.items():
        plt.plot(xs, series, label=label, color=colors.get(label, "gray"), linewidth=2)
    plt.ylabel(r"$\mathcal{l}_2$")
    plt.xlabel("Assimilation Step")
    plt.legend(loc="best", ncol=2, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------
# MAIN WORKFLOW
# ---------------------------------------------------------------------
def main():
    paths = {
        "ReverseSDE (Nonlinear)": ENSF_PATH,
        "ReverseSDE (Linear)": LINEAR_PATH,
        "LEnKF": LENKF_PATH,
        "LETKF": LETKF_PATH,
        "ENKF_MC_obs": ENKF_MC_PATH,
    }

    # Compute NoDA baseline from nonlinear (ENSF_PATH)
    gs = grid_resolution(RESOLUTION)
    ppt = postpro_tools(RESOLUTION, gs, ENSF_PATH, M_CYCLES)
    ppt.compute_NODA()

    out_dir = ENSF_PATH / "plots" / "errors" / PLOT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    for var in VARS_LIST:
        lvls = _levels_for_var(var)

        # -------------------------------
        # Level-averaged plot
        # -------------------------------
        data_avg = {}
        for method, path in paths.items():
            # ---- FIX: linear_results case has CSVs directly ----
            if "Linear" in method:
                df_path = path / f"{var}_ana.csv"
            else:
                df_path = path / "results" / f"{var}_ana.csv"

            if not df_path.exists():
                print(f"[skip] missing {df_path}")
                continue

            df = pd.read_csv(df_path)
            s = _avg_over_levels(df, lvls)
            data_avg[method] = s

        # Add NoDA averaged baseline
        noda_levels = [np.asarray(ppt.noda[var][L, :], dtype=float) for L in lvls]
        if noda_levels:
            Lmin_noda = min(len(x) for x in noda_levels)
            noda_mean = np.vstack([x[:Lmin_noda] for x in noda_levels]).mean(axis=0)
            data_avg["NoDA"] = noda_mean
        else:
            continue

        # Anchor all series to start at first NoDA value
        anchor_val = noda_mean[0]
        for k in list(data_avg.keys()):
            data_avg[k] = _make_anchor(data_avg[k], anchor_val)
        xs = np.arange(len(next(iter(data_avg.values()))))

        _plot_curves(xs, data_avg, f"{VAR_CODES.get(var,var)} — Level Average",
                     out_dir / f"{var}_ana_levelavg.png")

        # -------------------------------
        # Per-level plots
        # -------------------------------
        for lvl in lvls:
            data_lvl = {}
            for method, path in paths.items():
                if "Linear" in method:
                    df_path = path / f"{var}_ana.csv"
                else:
                    df_path = path / "results" / f"{var}_ana.csv"

                if not df_path.exists():
                    continue

                df = pd.read_csv(df_path)
                s = _read_series(df, lvl)
                if len(s):
                    data_lvl[method] = s

            # Add NoDA for this level
            noda_series = np.asarray(ppt.noda[var][lvl, :], dtype=float)
            data_lvl["NoDA"] = noda_series

            if not any(len(v) for v in data_lvl.values()):
                print(f"[skip] no data for {var} level {lvl}")
                continue

            anchor_val = noda_series[0]
            for k in list(data_lvl.keys()):
                data_lvl[k] = _make_anchor(data_lvl[k], anchor_val)
            xs = np.arange(len(next(iter(data_lvl.values()))))
            mb_txt = PS_LEVELS_MB[lvl] if lvl < len(PS_LEVELS_MB) else lvl
            title = f"{VAR_CODES.get(var,var)} — lev{lvl} ({mb_txt} mb)"
            _plot_curves(xs, data_lvl, title, out_dir / f"{var}_ana_lev{lvl}.png")

if __name__ == "__main__":
    main()
