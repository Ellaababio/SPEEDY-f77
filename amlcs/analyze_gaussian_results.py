#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def summarize_gaussian_block(summary_csv: Path):
    """Return a dict of compact diagnostics for one block (reading its summary.csv)."""
    df = pd.read_csv(summary_csv)
    return {
        "block_dir": summary_csv.parent.name,
        "n_dims": int(len(df)),
        "mean_of_means": float(df["mean"].mean()),
        "mean_of_stds": float(df["std"].mean()),
        "median_abs_skew": float(df["skew"].abs().median()),
        "median_abs_kurt_excess": float(df["kurtosis_excess"].abs().median()),
        "ks_stat_p50": float(df["ks_stat"].median()),
        "ks_stat_p95": float(df["ks_stat"].quantile(0.95)),
        "ks_stat_max": float(df["ks_stat"].max()),
    }

def find_block_summaries(root: Path):
    """Yield paths to all summary.csv files under root (gauss_XB_block_*/summary.csv)."""
    for p in root.rglob("summary.csv"):
        yield p

def main():
    ap = argparse.ArgumentParser(description="Aggregate Gaussianity validator results across blocks.")
    ap.add_argument("--root", type=str, default="gauss_checks",
                    help="Root folder containing per-block results (default: gauss_checks)")
    ap.add_argument("--out", type=str, default="gauss_overview.csv",
                    help="Output CSV path for aggregated diagnostics")
    ap.add_argument("--ks95_warn", type=float, default=0.1,
                    help="Warn threshold for 95th percentile KS statistic (default 0.1)")
    ap.add_argument("--print_top", type=int, default=10,
                    help="How many worst blocks to print (by ks_stat_p95)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[analyze] Root folder not found: {root}")
        return

    rows = []
    for s in find_block_summaries(root):
        try:
            rows.append(summarize_gaussian_block(s))
        except Exception as e:
            print(f"[analyze] Skipping {s} due to error: {e}")

    if not rows:
        print(f"[analyze] No summary.csv files found under {root}")
        return

    df = pd.DataFrame(rows)
    df["flag_warn"] = df["ks_stat_p95"] > args.ks95_warn

    out_path = Path(args.out)
    df.sort_values(["flag_warn", "ks_stat_p95", "ks_stat_max"],
                   ascending=[False, False, False], inplace=True)
    df.to_csv(out_path, index=False)

    print(f"[analyze] Wrote {len(df)} block summaries to {out_path}")

    print("\n[analyze] Overall (medians across blocks):")
    for col in ["mean_of_means","mean_of_stds","median_abs_skew",
                "median_abs_kurt_excess","ks_stat_p50","ks_stat_p95"]:
        print(f"  {col}: {df[col].median():.4f}")

    print("\n[analyze] Worst blocks by ks_stat_p95:")
    print(df[["block_dir","n_dims","ks_stat_p95","ks_stat_max",
              "median_abs_kurt_excess","median_abs_skew"]]
          .head(args.print_top)
          .to_string(index=False))

if __name__ == "__main__":
    main()
