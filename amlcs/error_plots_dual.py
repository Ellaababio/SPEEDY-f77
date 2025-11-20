#!/usr/bin/env python3
"""
Dual-run error plotting: generic method1 vs method2 (five curves per figure).
NO COMMAND-LINE ARGUMENTS. Everything is configured in the USER SETTINGS
section below.

Generates two families of plots for each variable:
  (A) Per-level plots (each level per figure)
  (B) Level-averaged plots
"""

###############################################################################
# ======================= USER SETTINGS (EDIT THESE) ==========================
###############################################################################

# FULL PATHS to the two experiment directories you want to compare:
EXP1 = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_ReverseSDE_1_1_100"
EXP2 = "/gpfs/home/jjs21b/AMLCS/runs/t21_50_0.05_5_EnKF_MC_obs_1_1_100"

# SPEEDY resolution:
RESOLUTION = "t21"

# Number of assimilation steps M (needed to compute NODA)
M = 5

# Variables to compare:
VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]

# Anchor mode: "step0" or "step1"
ANCHOR = "step1"

# Scale mode: "log", "linear", or "both"
SCALE_MODE = "both"

# Output directory name (optional)
# If None → "<method1>_vs_<method2>"
PLOT_DIR_NAME = 'ENKF_MC_obs_vs_ReverseSDE_nonlinear_old_drift'  

###############################################################################
# ======================= END USER SETTINGS ==================================
###############################################################################

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# imports from AMLCS
from grid_resolution import grid_resolution
from postpro_tools import postpro_tools

# Formatting
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})

# Pretty variable names
VAR_CODES = {
    "TG0": "T_0", "UG0": "u_0", "VG0": "v_0", "TRG0": "Hq_0", "PSG0": "PS_0",
    "TG1": "T_1", "UG1": "u_1", "VG1": "v_1", "TRG1": "Hq_1", "PSG1": "PS_1",
}
PS_LEVELS_MB = [30, 100, 200, 300, 500, 700, 850, 925]

# Fixed colors — ReverseSDE is blue
METHOD_COLORS = {
    "ReverseSDE": "tab:blue",
    "EnKF_MC_obs": "tab:orange",
    "LETKF": "tab:green",
    "LEnKF": "tab:red",
    "EnKF": "tab:purple",
}

###############################################################################
# --------------------- Utility Functions ------------------------------------
###############################################################################

def _extract_method_name(exp_path: Path) -> str:
    """
    Option B: infer method name from directory basename.
    Pattern example:
        t21_50_0.05_5_EnKF_MC_obs_1_1_100
          ^   ^     ^  ^  [method]    ^  ^
          0   1     2  3     4:-3     -3:-1

    Returns METHOD = parts[4:-3] joined with "_".
    """
    name = exp_path.name.rstrip("/")
    parts = name.split("_")

    if len(parts) > 7:
        mid = parts[4:-3]
        if mid:
            return "_".join(mid)

    # fallback
    for token in reversed(parts):
        try:
            float(token)
        except ValueError:
            return token
    return name


def _read_series(df, lvl):
    col = str(lvl)
    if col not in df.columns:
        return np.array([])
    return df[col].to_numpy()


def _make_anchor(series, anchor_val):
    if anchor_val is None:
        return series
    return np.concatenate([np.array([anchor_val]), series])


def _five_curves(ana1, bkg1, ana2, bkg2, noda, anchor_mode, scale, m1, m2):
    eps = 1e-12
    L = min(len(ana1), len(bkg1), len(ana2), len(bkg2), len(noda))
    if L == 0:
        return None, {}

    ana1, bkg1, ana2, bkg2, noda = (
        ana1[:L], bkg1[:L], ana2[:L], bkg2[:L], noda[:L]
    )
    anchor_val = noda[0] if anchor_mode == "step1" else None

    ana1 = _make_anchor(ana1, anchor_val)
    bkg1 = _make_anchor(bkg1, anchor_val)
    ana2 = _make_anchor(ana2, anchor_val)
    bkg2 = _make_anchor(bkg2, anchor_val)
    noda = _make_anchor(noda, anchor_val)

    xs = np.arange(len(noda))

    if scale == "log":
        curves = {
            "NoDA": np.log(noda + eps),
            f"{m1} Analysis": np.log(ana1 + eps),
            f"{m1} Background": np.log(bkg1 + eps),
            f"{m2} Analysis": np.log(ana2 + eps),
            f"{m2} Background": np.log(bkg2 + eps),
        }
    else:
        curves = {
            "NoDA": noda,
            f"{m1} Analysis": ana1,
            f"{m1} Background": bkg1,
            f"{m2} Analysis": ana2,
            f"{m2} Background": bkg2,
        }

    return xs, curves


def _plot_curves(xs, curves, title, out_path, scale, m1, m2):
    plt.figure(figsize=(9, 4))
    style = {"Analysis": "-", "Background": "--"}

    order = [
        "NoDA",
        f"{m1} Analysis", f"{m1} Background",
        f"{m2} Analysis", f"{m2} Background",
    ]

    for label in order:
        if label not in curves:
            continue
        y = curves[label]
        if label == "NoDA":
            plt.plot(xs, y, label=label, color="k", linestyle="-")
        else:
            meth, kind = label.split(maxsplit=1)
            color = METHOD_COLORS.get(meth, None)
            ls = style[kind]
            plt.plot(xs, y, label=label, linestyle=ls, color=color)

    plt.title(title)
    plt.xlabel("Assimilation Step")
    plt.ylabel("log(l2)" if scale == "log" else "l2")
    plt.legend()
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

def run_dual_plots():
    exp1 = Path(EXP1).resolve()
    exp2 = Path(EXP2).resolve()

    method1 = _extract_method_name(exp1)
    method2 = _extract_method_name(exp2)

    out_name = PLOT_DIR_NAME or f"{method1}_vs_{method2}"

    print(f"Comparing:")
    print(f"  EXP1={exp1}  (method={method1})")
    print(f"  EXP2={exp2}  (method={method2})")
    print(f"Output directory tag: {out_name}")

    # Compute NoDA baseline using exp1
    gs = grid_resolution(RESOLUTION)
    ppt = postpro_tools(RESOLUTION, gs, exp1, M)
    ppt.compute_NODA()

    root1 = exp1 / "plots" / "errors"
    root2 = exp2 / "plots" / "errors"

    def _ensure(scale):
        if scale == "log":
            d1 = root1 / out_name
            d2 = root2 / out_name
        else:
            d1 = root1 / f"{out_name}_abs"
            d2 = root2 / f"{out_name}_abs"
        d1.mkdir(parents=True, exist_ok=True)
        d2.mkdir(parents=True, exist_ok=True)
        return d1  # only need one for saving

    scales = ["log", "linear"] if SCALE_MODE == "both" else [SCALE_MODE]

    # ------------------- LEVEL BY LEVEL -------------------
    for var in VARS:
        ana1 = pd.read_csv(exp1 / "results" / f"{var}_ana.csv")
        bkg1 = pd.read_csv(exp1 / "results" / f"{var}_bck.csv")
        ana2 = pd.read_csv(exp2 / "results" / f"{var}_ana.csv")
        bkg2 = pd.read_csv(exp2 / "results" / f"{var}_bck.csv")

        lvls = _levels_for_var(var)

        for lvl in lvls:
            s_ana1 = _read_series(ana1, lvl)
            s_bkg1 = _read_series(bkg1, lvl)
            s_ana2 = _read_series(ana2, lvl)
            s_bkg2 = _read_series(bkg2, lvl)
            s_noda = ppt.noda[var][lvl, :]

            for scale in scales:
                out_dir = _ensure(scale)
                xs, curves = _five_curves(
                    s_ana1, s_bkg1, s_ana2, s_bkg2, s_noda,
                    ANCHOR, scale, method1, method2
                )
                if xs is None:
                    continue

                mb = PS_LEVELS_MB[lvl] if lvl < len(PS_LEVELS_MB) else lvl
                title = f"{VAR_CODES.get(var,var)} (lev {lvl}, {mb} mb)"
                out_file = out_dir / f"dual_single_{var}_lev{lvl}.png"
                _plot_curves(xs, curves, title, out_file, scale, method1, method2)

    # ------------------- LEVEL-AVERAGED -------------------
    for var in VARS:
        ana1 = pd.read_csv(exp1 / "results" / f"{var}_ana.csv")
        bkg1 = pd.read_csv(exp1 / "results" / f"{var}_bck.csv")
        ana2 = pd.read_csv(exp2 / "results" / f"{var}_ana.csv")
        bkg2 = pd.read_csv(exp2 / "results" / f"{var}_bck.csv")

        lvls = _levels_for_var(var)

        def avg(df):
            arrs = [ _read_series(df, L) for L in lvls ]
            arrs = [a for a in arrs if len(a) > 0]
            if not arrs:
                return np.array([])
            Lmin = min(len(a) for a in arrs)
            return np.vstack([a[:Lmin] for a in arrs]).mean(axis=0)

        s_ana1 = avg(ana1)
        s_bkg1 = avg(bkg1)
        s_ana2 = avg(ana2)
        s_bkg2 = avg(bkg2)

        noda_levels = [
            ppt.noda[var][L, :] for L in lvls
        ]
        Lmin = min(len(x) for x in noda_levels)
        s_noda = np.vstack([x[:Lmin] for x in noda_levels]).mean(axis=0)

        for scale in scales:
            out_dir = _ensure(scale)
            xs, curves = _five_curves(
                s_ana1, s_bkg1, s_ana2, s_bkg2, s_noda,
                ANCHOR, scale, method1, method2
            )
            if xs is None:
                continue

            title = f"{VAR_CODES.get(var,var)} (Level Average)"
            out_file = out_dir / f"dual_levelavg_{var}.png"
            _plot_curves(xs, curves, title, out_file, scale, method1, method2)


###############################################################################
# ------------------------------ ENTRY POINT ----------------------------------
###############################################################################

if __name__ == "__main__":
    run_dual_plots()
