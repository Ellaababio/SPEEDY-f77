#!/usr/bin/env python3
"""
Report the best LETKF localization radius (r) from an rmse_summary.csv.

The best r is the one that achieves the minimum RMSE for the greatest number of
variables (a scale-invariant "wins" count), NOT the lowest overall average,
because the variables live on very different scales.

Reads the summary written by `letkf_r_tuning.py collect` and prints:
  - the best r by number of per-variable wins (ties broken by mean rank)
  - a per-variable table of which r wins each variable
  - per-r win counts and overall_avg (shown for reference only)

Usage
-----
    python letkf_best_r.py letkf_tuning_runs/all_arctan/rmse_summary.csv
    python letkf_best_r.py letkf_tuning_runs/all_arctan/rmse_summary.csv --metric min
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Non-variable columns produced by the collect step.
META_COLS = {"r", "run_folder", "data_dir", "n_cycles", "overall_avg"}


def _detect_vars(df):
    """Derive variable names from `<var>_avg` columns present in the summary."""
    vars_found = []
    for col in df.columns:
        if col.endswith("_avg") and col != "overall_avg":
            vars_found.append(col[: -len("_avg")])
    return vars_found


def _fmt(x):
    return "n/a" if pd.isna(x) else f"{x:.6g}"


def rank_and_score(df, vars_found, metric):
    """
    For each variable, find the winning r (minimum RMSE) and per-r ranks.

    Returns:
        wins:        dict r -> number of variables won
        winner_of:   dict var -> winning r (or None)
        mean_rank:   dict r -> mean rank across variables (lower is better)
    """
    rs = [int(r) for r in df["r"]]
    wins = {r: 0 for r in rs}
    rank_sums = {r: 0.0 for r in rs}
    rank_counts = {r: 0 for r in rs}
    winner_of = {}

    for var in vars_found:
        col = f"{var}_{metric}"
        if col not in df.columns:
            winner_of[var] = None
            continue
        sub = df[["r", col]].dropna(subset=[col])
        if sub.empty:
            winner_of[var] = None
            continue

        # Rank ascending: lowest RMSE -> rank 1
        ranks = sub[col].rank(method="min", ascending=True)
        for r_val, rk in zip(sub["r"], ranks):
            r_i = int(r_val)
            rank_sums[r_i] += float(rk)
            rank_counts[r_i] += 1

        best_idx = sub[col].idxmin()
        win_r = int(sub.loc[best_idx, "r"])
        winner_of[var] = win_r
        wins[win_r] += 1

    mean_rank = {}
    for r in rs:
        mean_rank[r] = (rank_sums[r] / rank_counts[r]) if rank_counts[r] else np.inf

    return wins, winner_of, mean_rank


def pick_best_r(wins, mean_rank):
    """Best r: most wins; ties broken by lower mean rank, then smaller r."""
    if not wins:
        return None
    return sorted(wins.keys(),
                  key=lambda r: (-wins[r], mean_rank.get(r, np.inf), r))[0]


def row_for_r(df, r):
    """Return the row (as a dict) for a given r, or empty dict if absent."""
    sub = df[df["r"] == r]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", help="Path to rmse_summary.csv from letkf_r_tuning.py collect.")
    parser.add_argument("--metric", choices=["avg", "min"], default="avg",
                        help="Per-variable metric used to decide each variable's winner: "
                             "'avg' (mean over cycles) or 'min' (lowest over cycles).")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        sys.exit(f"Summary CSV not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if "r" not in df.columns:
        sys.exit("CSV does not look like an rmse_summary.csv (missing 'r').")

    df = df.sort_values("r").reset_index(drop=True)
    vars_found = _detect_vars(df)
    if not vars_found:
        sys.exit("No per-variable columns (<var>_avg) found in the summary.")

    wins, winner_of, mean_rank = rank_and_score(df, vars_found, args.metric)
    best_r = pick_best_r(wins, mean_rank)
    if best_r is None:
        sys.exit("No rankable data (all per-variable values are NaN).")

    n_scored = sum(1 for v in winner_of.values() if v is not None)

    print(f"Summary: {summary_path}")
    print(f"Variables: {', '.join(vars_found)}")
    print(f"r values:  {', '.join(str(int(r)) for r in df['r'])}")
    print(f"Ranking metric: per-variable '{args.metric}', best = most variable wins")
    print()
    print(f"==> BEST r = {best_r}  (wins {wins[best_r]}/{n_scored} variables, "
          f"mean rank {mean_rank[best_r]:.3g})")
    print()

    # Per-r win counts
    print("Per-r win counts (variables for which this r has the lowest RMSE):")
    for _, row in df.iterrows():
        r = int(row["r"])
        marker = "  <-- best" if r == best_r else ""
        overall = row.get("overall_avg", np.nan)
        print(f"  r={r:>3}: wins={wins[r]:>2}/{n_scored}   "
              f"mean_rank={mean_rank[r]:.3g}   overall_avg={_fmt(overall)} (ref){marker}")
    print()

    # Per-variable winners
    print(f"Per-variable winning r (by '{args.metric}'):")
    suffix = f"_{args.metric}"
    for var in vars_found:
        win_r = winner_of.get(var)
        if win_r is None:
            print(f"  {var:<6}: n/a")
            continue
        col = f"{var}{suffix}"
        win_val = row_for_r(df, win_r).get(col, np.nan)
        at_best = row_for_r(df, best_r).get(col, np.nan)
        flag = "  *" if win_r == best_r else ""
        print(f"  {var:<6}: best r={win_r:>3} ({col}={_fmt(win_val)})"
              f"  |  at best r={best_r}: {_fmt(at_best)}{flag}")
    print()

    # Full per-variable breakdown at the chosen best r
    print(f"Per-variable RMSE at best r={best_r}:")
    best_row = row_for_r(df, best_r)
    for var in vars_found:
        avg = best_row.get(f"{var}_avg", np.nan)
        low = best_row.get(f"{var}_min", np.nan)
        print(f"  {var:<6}: avg = {_fmt(avg)}   min = {_fmt(low)}")


if __name__ == "__main__":
    main()
