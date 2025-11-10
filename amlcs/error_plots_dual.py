#!/usr/bin/env python3
"""
Dual-run error plotting: ENSF vs LEnKF (five curves per figure).

Generates two families of plots:
  (A) Per-level plots for each requested variable:
      - For 3D variables (TG*, UG*, VG*, TRG*): one figure PER LEVEL.
        Note: TRG has no data at levels 0 or 1; those levels are skipped.
      - For PSG*: only level 0 exists.
      Each figure includes 5 curves:
        NoDA (black solid),
        ENSF Analysis (blue solid),
        ENSF Background (blue dashed),
        LEnKF Analysis (orange solid),
        LEnKF Background (orange dashed).
  (B) Level-averaged plots for each variable (averaging over available levels).

ANCHOR EXPLANATION (plotting only; does NOT alter your data):
- --anchor step1  → Prepends a single synthetic point (t=0) to ALL curves equal to the FIRST NoDA value.
                    This makes every curve start from the same visible baseline before cycle 1.
- --anchor step0  → No synthetic point; curves begin at the first cycle in the CSVs.

SCALING / OUTPUT DIRS:
- --scale log     → log-scaled errors; saved under <plot_dir_name>/
- --scale linear  → absolute errors; saved under <plot_dir_name>_abs/
- --scale both    → write BOTH sets to the two sibling directories above.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Resolved in the user's environment
from grid_resolution import grid_resolution
from postpro_tools import postpro_tools

# Consistent math font and sizing
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 14})

VAR_CODES = {
    "TG0": "T_0", "UG0": "u_0", "VG0": "v_0", "TRG0": "Hq_0", "PSG0": "PS_0",
    "TG1": "T_1", "UG1": "u_1", "VG1": "v_1", "TRG1": "Hq_1", "PSG1": "PS_1",
}
PS_LEVELS_MB = [30, 100, 200, 300, 500, 700, 850, 925]

def _read_series(df: pd.DataFrame, lvl: int) -> np.ndarray:
    col = str(lvl)
    if col not in df.columns:
        return np.array([], dtype=float)
    return df[col].to_numpy(dtype=float)

def _make_anchor(series: np.ndarray, anchor_value: float | None):
    if anchor_value is None:
        return series
    return np.concatenate([np.array([anchor_value], dtype=float), series])

def _five_curves(ana1, bkg1, ana2, bkg2, noda, anchor_mode: str, scale: str):
    eps = 1e-12
    L = min(len(ana1), len(bkg1), len(ana2), len(bkg2), len(noda))
    if L == 0:
        return None, {}
    ana1, bkg1, ana2, bkg2, noda = ana1[:L], bkg1[:L], ana2[:L], bkg2[:L], noda[:L]
    anchor_val = noda[0] if anchor_mode.lower() == "step1" else None
    ana1 = _make_anchor(ana1, anchor_val)
    bkg1 = _make_anchor(bkg1, anchor_val)
    ana2 = _make_anchor(ana2, anchor_val)
    bkg2 = _make_anchor(bkg2, anchor_val)
    noda = _make_anchor(noda, anchor_val)
    xs = np.arange(0, len(noda))
    if scale == "log":
        curves = {
            "NoDA": np.log(noda + eps),
            "ENSF Analysis": np.log(ana1 + eps),
            "ENSF Background": np.log(bkg1 + eps),
            "LEnKF Analysis": np.log(ana2 + eps),
            "LEnKF Background": np.log(bkg2 + eps),
        }
    else:
        curves = {
            "NoDA": noda,
            "ENSF Analysis": ana1,
            "ENSF Background": bkg1,
            "LEnKF Analysis": ana2,
            "LEnKF Background": bkg2,
        }
    return xs, curves

def _plot_curves(xs, curves, title, out_path, scale: str):
    plt.figure(figsize=(9, 4))
    plt.title(title)
    method_colors = {"ENSF": "tab:blue", "LEnKF": "tab:orange"}
    style = {"Analysis": "-", "Background": "--"}
    order = ["NoDA", "ENSF Analysis", "ENSF Background", "LEnKF Analysis", "LEnKF Background"]
    for label in order:
        if label not in curves:
            continue
        y = curves[label]
        if label == "NoDA":
            plt.plot(xs, y, label=label, color="k", linestyle="-")
        else:
            meth, kind = label.split()  # e.g., "ENSF", "Analysis"
            plt.plot(xs, y, label=label, color=method_colors[meth], linestyle=style[kind])
    plt.ylabel(r"$\log(\mathcal{l}_2)$" if scale == "log" else r"$\mathcal{l}_2$")
    plt.xlabel(r"$\mathrm{Assimilation\ Step}$")
    plt.legend(loc="best", prop={"size": 12}, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def _levels_for_var(var: str) -> list[int]:
    if "PSG" in var:
        return [0]
    if var.startswith("TRG"):
        # Humidity has no data at levels 0 or 1 in this setup
        return list(range(2, 8))
    # TG*, UG*, VG* have 8 levels (0..7)
    return list(range(8))

def dual_error_plots(ensf_path: Path, lenkf_path: Path, vars_list, grid_res: str, M: int, anchor: str, plot_dir_name: str, scale_mode: str):
    ensf_path = ensf_path.resolve()
    lenkf_path = lenkf_path.resolve()

    # Compute NoDA baseline once (method-independent)
    gs = grid_resolution(grid_res)
    ppt_ensf = postpro_tools(grid_res, gs, ensf_path, M)
    ppt_ensf.compute_NODA()

    # Decide output dirs
    out_log = ensf_path / "plots" / "errors" / plot_dir_name
    out_abs = ensf_path / "plots" / "errors" / f"{plot_dir_name}_abs"
    out_log_mirror = lenkf_path / "plots" / "errors" / plot_dir_name
    out_abs_mirror = lenkf_path / "plots" / "errors" / f"{plot_dir_name}_abs"

    def _ensure_dirs(scale: str):
        if scale == "log":
            out_log.mkdir(parents=True, exist_ok=True)
            out_log_mirror.mkdir(parents=True, exist_ok=True)
            return out_log
        elif scale == "linear":
            out_abs.mkdir(parents=True, exist_ok=True)
            out_abs_mirror.mkdir(parents=True, exist_ok=True)
            return out_abs
        else:
            raise ValueError("scale must be 'log' or 'linear' at this point.")

    scales = ["log", "linear"] if scale_mode == "both" else [scale_mode]

    # --- (A) Per-level plots ---
    for var in vars_list:
        # Input CSVs for each method
        ana1 = pd.read_csv(ensf_path / "results" / f"{var}_ana.csv")
        bkg1 = pd.read_csv(ensf_path / "results" / f"{var}_bck.csv")
        ana2 = pd.read_csv(lenkf_path / "results" / f"{var}_ana.csv")
        bkg2 = pd.read_csv(lenkf_path / "results" / f"{var}_bck.csv")

        lvls = _levels_for_var(var)
        for lvl in lvls:
            s_ana1 = _read_series(ana1, lvl)
            s_bkg1 = _read_series(bkg1, lvl)
            s_ana2 = _read_series(ana2, lvl)
            s_bkg2 = _read_series(bkg2, lvl)
            s_noda = np.asarray(ppt_ensf.noda[var][lvl, :], dtype=float)

            for scale in scales:
                out_dir = _ensure_dirs(scale)
                xs, curves = _five_curves(s_ana1, s_bkg1, s_ana2, s_bkg2, s_noda, anchor, scale)
                if xs is None:
                    print(f"[skip] no data for {var} level {lvl} (scale={scale})")
                    continue
                mb_txt = PS_LEVELS_MB[lvl] if lvl < len(PS_LEVELS_MB) else lvl
                title = rf"$\mathrm{{{VAR_CODES.get(var, var)}}} \ \mathrm{{(lev\ {lvl},\ {mb_txt}\ mb)}}$"
                out = out_dir / f"dual_single_{var}_lev{lvl}.png"
                _plot_curves(xs, curves, title, out, scale)

    # --- (B) Level-averaged plots ---
    for var in vars_list:
        ana1 = pd.read_csv(ensf_path / "results" / f"{var}_ana.csv")
        bkg1 = pd.read_csv(ensf_path / "results" / f"{var}_bck.csv")
        ana2 = pd.read_csv(lenkf_path / "results" / f"{var}_ana.csv")
        bkg2 = pd.read_csv(lenkf_path / "results" / f"{var}_bck.csv")

        lvls = _levels_for_var(var)

        def _avg_over_levels(df: pd.DataFrame, lvls: list[int]) -> np.ndarray:
            series_list = []
            for L in lvls:
                s = _read_series(df, L)
                if len(s) > 0:
                    series_list.append(s)
            if not series_list:
                return np.array([], dtype=float)
            Lmin = min(len(s) for s in series_list)
            stack = np.vstack([s[:Lmin] for s in series_list])
            return stack.mean(axis=0)

        s_ana1 = _avg_over_levels(ana1, lvls)
        s_bkg1 = _avg_over_levels(bkg1, lvls)
        s_ana2 = _avg_over_levels(ana2, lvls)
        s_bkg2 = _avg_over_levels(bkg2, lvls)

        noda_levels = [np.asarray(ppt_ensf.noda[var][L, :], dtype=float) for L in lvls]
        Lmin_noda = min(len(x) for x in noda_levels) if lvls else 0
        s_noda = np.vstack([x[:Lmin_noda] for x in noda_levels]).mean(axis=0) if Lmin_noda > 0 else np.array([], dtype=float)

        for scale in scales:
            out_dir = _ensure_dirs(scale)
            xs, curves = _five_curves(s_ana1, s_bkg1, s_ana2, s_bkg2, s_noda, anchor, scale)
            if xs is None:
                print(f"[skip] no averaged data for {var} (scale={scale})")
                continue
            title = rf"$\mathrm{{{VAR_CODES.get(var, var)}}} \ \mathrm{{(Level\ Average)}}$"
            out = out_dir / f"dual_levelavg_{var}.png"
            _plot_curves(xs, curves, title, out, scale)

def main():
    p = argparse.ArgumentParser(description="Plot ENSF vs LEnKF error curves (five lines) per variable.")
    p.add_argument("--ensf_exp", required=True, help="Path to ENSF experiment directory (e.g., ../runs/t21_..._ENSF_...)")
    p.add_argument("--lenkf_exp", required=True, help="Path to LEnKF experiment directory (e.g., ../runs/t21_..._LEnKF_...)")
    p.add_argument("--resolution", required=True, help="Grid resolution name (e.g., t21, t30)")
    p.add_argument("--M", type=int, required=True, help="Number of assimilation steps (used to compute NoDA series)")
    p.add_argument("--plot_dir_name", default="ENSF_vs_LENKF", help="Subdirectory under plots/errors/ to write figures")
    p.add_argument("--anchor", choices=["step0", "step1"], default="step1",
                   help="If step1, prepend a common anchor using the first NoDA value")
    p.add_argument("--vars", default="TG1,UG1,VG1,TRG1,PSG1",
                   help="Comma-separated variable list; choose only one of each pair (e.g., TG1 instead of TG0).")
    p.add_argument("--scale", choices=["log", "linear", "both"], default="both",
                   help="Which scaling to plot: log (original), linear (absolute errors), or both (creates sibling _abs dir).")
    args = p.parse_args()

    vars_list = [v.strip() for v in args.vars.split(",") if v.strip()]
    dual_error_plots(
        ensf_path=Path(args.ensf_exp),
        lenkf_path=Path(args.lenkf_exp),
        vars_list=vars_list,
        grid_res=args.resolution,
        M=args.M,
        anchor=args.anchor,
        plot_dir_name=args.plot_dir_name,
        scale_mode=args.scale
    )

if __name__ == "__main__":
    main()
