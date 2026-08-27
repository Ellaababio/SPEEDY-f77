#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

VARS = ["TG1", "UG1", "VG1", "TRG1", "PSG1"]
CASES = ["case1_linear", "case2_arctan", "case3_arctan_sq", "case4_only_wind", "case5_wind_vars"]

def compare_case(case_name, base_dir=Path('.')):
    letkf_csv = base_dir / "letkf_tuning_runs" / case_name / "rmse_summary.csv"
    reversesde_csv = base_dir / "reversesde_tuning_runs" / case_name / "rmse_summary.csv"

    if not letkf_csv.exists():
        print(f"[{case_name}] LETKF summary missing: {letkf_csv}")
        return
    if not reversesde_csv.exists():
        print(f"[{case_name}] ReverseSDE summary missing: {reversesde_csv}")
        return

    df_letkf = pd.read_csv(letkf_csv)
    df_revsde = pd.read_csv(reversesde_csv)

    print(f"\n==========================================")
    print(f" Head-to-Head Comparison for Case: {case_name}")
    print(f"==========================================")

    letkf_wins, revsde_wins, ties = 0, 0, 0
    print(f"{'Variable':<10} | {'LETKF Best (r, infla)':<22} | {'LETKF RMSE':<12} | {'RevSDE Best (r, infla)':<23} | {'RevSDE RMSE':<12} | {'Winner':<10}")
    print("-" * 100)

    for var in VARS:
        col = f"{var}_avg"
        if col not in df_letkf.columns or col not in df_revsde.columns:
            continue
        
        sub_l = df_letkf.dropna(subset=[col])
        if sub_l.empty:
            l_best_rmse, l_best_param = np.nan, "N/A"
        else:
            idx_l = sub_l[col].idxmin()
            l_best_rmse = sub_l.loc[idx_l, col]
            l_best_param = f"r={int(sub_l.loc[idx_l, 'r'])}, infla={sub_l.loc[idx_l, 'infla']}"

        sub_r = df_revsde.dropna(subset=[col])
        if sub_r.empty:
            r_best_rmse, r_best_param = np.nan, "N/A"
        else:
            idx_r = sub_r[col].idxmin()
            r_best_rmse = sub_r.loc[idx_r, col]
            r_best_param = f"r={int(sub_r.loc[idx_r, 'r'])}, infla={sub_r.loc[idx_r, 'infla']}"

        if np.isnan(l_best_rmse) or np.isnan(r_best_rmse):
            winner = "N/A"
        elif l_best_rmse < r_best_rmse:
            winner = "LETKF"
            letkf_wins += 1
        elif r_best_rmse < l_best_rmse:
            winner = "ReverseSDE"
            revsde_wins += 1
        else:
            winner = "Tie"
            ties += 1

        print(f"{var:<10} | {l_best_param:<22} | {l_best_rmse:<12.6f} | {r_best_param:<23} | {r_best_rmse:<12.6f} | {winner:<10}")

    print("-" * 100)
    print(f"Final Tally: LETKF won {letkf_wins}/5 | ReverseSDE won {revsde_wins}/5 | Ties: {ties}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        for c in CASES: compare_case(c)
    elif args.case:
        compare_case(args.case)
