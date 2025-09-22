#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import pandas as pd
import re

BLOCK_RE = re.compile(r".*?_XB_block_(\d+)$")  # matches "gauss_XB_block_24" -> 24

def parse_block_index(block_dir_name: str):
    m = BLOCK_RE.match(block_dir_name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None

def summarize_gaussian_block(summary_csv: Path):
    """Return compact diagnostics for one block (reading its summary.csv)."""
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
    yield from root.rglob("summary.csv")

def try_autoload_map(root: Path, explicit: str):
    # Priority: explicit path, else ../block_map.json, else block_map.json under root
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    cand1 = (root / ".." / "block_map.json").resolve()
    cand2 = (root / "block_map.json").resolve()
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None

def load_block_map(map_path: Path):
    data = json.loads(Path(map_path).read_text())
    by_idx = {}
    for entry in data.get("blocks", []):
        idx = int(entry["block_idx"])
        by_idx[idx] = {"vars": entry.get("vars", []), "levels": entry.get("levels", [])}
    return by_idx

def main():
    ap = argparse.ArgumentParser(description="Aggregate Gaussianity validator results across blocks, with optional var/level mapping.")
    ap.add_argument("--root", type=str, default="gauss_checks",
                    help="Root folder containing per-block results (default: gauss_checks)")
    ap.add_argument("--out", type=str, default="gauss_overview.csv",
                    help="Output CSV path for aggregated diagnostics")
    ap.add_argument("--ks95_warn", type=float, default=0.1,
                    help="Warn threshold for 95th percentile KS statistic (default 0.1)")
    ap.add_argument("--print_top", type=int, default=10,
                    help="How many worst blocks to print (by ks_stat_p95)")
    ap.add_argument("--map", type=str, default="",
                    help="Optional path to block_map.json (if not provided, will try to auto-discover)")
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
    df["block_idx"] = df["block_dir"].apply(parse_block_index)

    # Try to attach vars/levels via mapping
    map_path = try_autoload_map(root, args.map)
    if map_path is not None:
        try:
            block_map = load_block_map(map_path)
            df["vars"] = df["block_idx"].apply(lambda i: ",".join(block_map.get(i, {}).get("vars", [])) if pd.notnull(i) else "")
            df["levels"] = df["block_idx"].apply(lambda i: ",".join(map(str, block_map.get(i, {}).get("levels", []))) if pd.notnull(i) else "")
            print(f"[analyze] Using block map: {map_path}")
        except Exception as e:
            print(f"[analyze] WARNING: failed to read block map {map_path}: {e}")
            df["vars"] = ""
            df["levels"] = ""
    else:
        print("[analyze] No block map provided or found; vars/levels will be empty.")
        df["vars"] = ""
        df["levels"] = ""

    # Flag and sort
    df["flag_warn"] = df["ks_stat_p95"] > args.ks95_warn
    df.sort_values(["flag_warn", "ks_stat_p95", "ks_stat_max"],
                   ascending=[False, False, False], inplace=True)

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    print(f"[analyze] Wrote {len(df)} block summaries to {out_path}")

    print("\n[analyze] Overall (medians across blocks):")
    for col in ["mean_of_means","mean_of_stds","median_abs_skew",
                "median_abs_kurt_excess","ks_stat_p50","ks_stat_p95"]:
        print(f"  {col}: {df[col].median():.4f}")

    print("\n[analyze] Worst blocks by ks_stat_p95:")
    cols_to_show = ["block_dir","block_idx","vars","levels","n_dims","ks_stat_p95","ks_stat_max","median_abs_kurt_excess","median_abs_skew"]
    print(df[cols_to_show].head(args.print_top).to_string(index=False))

if __name__ == "__main__":
    main()
