#!/usr/bin/env python3
"""
Aggregate Gaussianity results across blocks, supporting both PRE- and POST-SDE summaries.

Usage:
  python analyze_gaussian_results.py --root <ROOT> \
      --out <combined.csv> \
      --ks95_warn 0.10 \
      --print_top 15 \
      --map <block_map.json>

Notes:
- Robust block_id extraction: grabs the LAST integer from folder name (e.g. gauss_XB_block_24 -> 24).
- Always includes 'map_label' column (may be empty string if not found).
- Avoids scientific notation for typical values; uses scientific only for very small/large.
"""

import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

# -------- formatting helpers --------
def fmt_num(x):
    """Fixed-point unless |x| < 1e-4 or >= 1e6; then scientific with 3 sig figs."""
    try:
        x = float(x)
    except Exception:
        return str(x)
    ax = abs(x)
    if (ax != 0.0) and (ax < 1e-4 or ax >= 1e6):
        return f"{x:.3e}"
    return f"{x:.6f}"

def fmt_series(s):
    return s.apply(fmt_num)

# -------- I/O helpers --------
def load_summary(csv_path: Path):
    df = pd.read_csv(csv_path)
    need = {"mean", "std", "skew", "kurtosis_excess", "ks_stat"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    return df

def median_stats(df: pd.DataFrame):
    # use numpy ops for speed; cast to float explicitly
    m   = df["mean"].to_numpy(float)
    st  = df["std"].to_numpy(float)
    sk  = df["skew"].to_numpy(float)
    ku  = df["kurtosis_excess"].to_numpy(float)
    ks  = df["ks_stat"].to_numpy(float)
    return {
        "mean_abs_mean": float(np.median(np.abs(m))),
        "mean_std":      float(np.median(st)),
        "abs_skew":      float(np.median(np.abs(sk))),
        "abs_kurt":      float(np.median(np.abs(ku))),
        "ks_med":        float(np.median(ks)),
        "ks_p95":        float(np.quantile(ks, 0.95)),
        "n_dims":        int(len(df)),
    }

def read_block_map(path: Path):
    if not path or not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)

    # Case 1: list of dicts under "blocks"
    if isinstance(data, dict) and "blocks" in data:
        return {str(entry["block_idx"]): entry for entry in data["blocks"]}

    # Case 2: plain list
    if isinstance(data, list):
        return {str(i): data[i] for i in range(len(data))}

    # Case 3: dict of dicts keyed by block id
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}

    return {}

def label_from_map(block_id: int, mp: dict):
    """Build a short human-readable label from the mapping, if present."""
    k = str(block_id)
    if k not in mp:
        return ""
    entry = mp[k] if isinstance(mp[k], dict) else {}
    vars_ = entry.get("vars") or entry.get("variables") or entry.get("var_names") or []
    levs_ = entry.get("levels") or entry.get("levs") or entry.get("level") or []
    # Normalize a couple variants
    if isinstance(vars_, dict) and "names" in vars_:
        vars_ = vars_["names"]
    def _list_to_str(v, maxn=5):
        if isinstance(v, (list, tuple)):
            head = ",".join(map(str, v[:maxn]))
            return head + ("..." if len(v) > maxn else "")
        return str(v)
    vs = _list_to_str(vars_)
    ls = _list_to_str(levs_)
    parts = []
    if vs and vs != "[]": parts.append(f"vars={vs}")
    if ls and ls != "[]": parts.append(f"levs={ls}")
    return " ".join(parts)

def extract_block_id(dirname: str):
    """
    Return LAST integer from directory name, or None if no digits found.
    Examples:
      'gauss_XB_block_24' -> 24
      'XB_block_005'      -> 5
      'block42_extra'     -> 42
    """
    nums = re.findall(r'(\d+)', dirname)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory containing gauss_XB_block_*/ subfolders")
    ap.add_argument("--out", default=None, help="Write combined CSV with per-block medians/deltas")
    ap.add_argument("--ks95_warn", type=float, default=0.12, help="Warn if post KS p95 exceeds this")
    ap.add_argument("--print_top", type=int, default=15, help="How many worst blocks to print")
    ap.add_argument("--map", default=None, help="Optional block map JSON to annotate blocks")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[analyze] ROOT does not exist: {root}", file=sys.stderr)
        sys.exit(2)

    mp = read_block_map(Path(args.map)) if args.map else {}

    # discover block directories
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    # keep ones that actually contain summary.csv
    dirs = [p for p in dirs if (p / "summary.csv").exists()]

    if not dirs:
        print(f"[analyze] No block subfolders with summary.csv found under {root}", file=sys.stderr)
        sys.exit(2)

    rows = []
    for d in dirs:
        post_csv = d / "summary.csv"
        try:
            post = load_summary(post_csv)
        except Exception as e:
            print(f"[analyze] Skipping {d.name}: {e}", file=sys.stderr)
            continue

        rec = {"block_dir": d.name}

        # robust ID + label
        bid = extract_block_id(d.name)
        rec["block_id"] = bid if bid is not None else -1
        rec["map_label"] = label_from_map(bid, mp) if bid is not None else ""

        # POST stats
        post_med = median_stats(post)
        for k, v in post_med.items():
            rec[f"post_{k}"] = v

        # PRE stats (optional)
        pre_csv = d / "summary_pre.csv"
        if pre_csv.exists():
            try:
                pre = load_summary(pre_csv)
                pre_med = median_stats(pre)
                for k, v in pre_med.items():
                    rec[f"pre_{k}"] = v
                rec["d_ks_med"]   = pre_med["ks_med"]  - post_med["ks_med"]
                rec["d_ks_p95"]   = pre_med["ks_p95"]  - post_med["ks_p95"]
                rec["d_abs_skew"] = pre_med["abs_skew"] - post_med["abs_skew"]
                rec["d_abs_kurt"] = pre_med["abs_kurt"] - post_med["abs_kurt"]
            except Exception as e:
                print(f"[analyze] Warning: failed reading pre for {d.name}: {e}", file=sys.stderr)

        rows.append(rec)

    if not rows:
        print(f"[analyze] No valid blocks found under {root}", file=sys.stderr)
        sys.exit(2)

    T = pd.DataFrame(rows)

    # ----- Overall medians (pretty print without sci notation) -----
    print("[analyze] Overall medians (POST across blocks):")
    cols_post = ["post_mean_abs_mean","post_mean_std","post_abs_skew","post_abs_kurt","post_ks_med","post_ks_p95"]
    if all(c in T.columns for c in cols_post):
        post_med = T[cols_post].median()
        for k, v in post_med.items():
            print(f"{k:20s} {fmt_num(v)}")

    if "pre_ks_med" in T.columns:
        print("\n[analyze] Overall medians (PRE across blocks):")
        cols_pre = ["pre_mean_abs_mean","pre_mean_std","pre_abs_skew","pre_abs_kurt","pre_ks_med","pre_ks_p95"]
        pre_med = T[cols_pre].median()
        for k, v in pre_med.items():
            print(f"{k:20s} {fmt_num(v)}")

        print("\n[analyze] Overall median improvements (PRE -> POST):")
        cols_imp = ["d_ks_med","d_ks_p95","d_abs_skew","d_abs_kurt"]
        imp_med = T[cols_imp].median()
        for k, v in imp_med.items():
            print(f"{k:20s} {fmt_num(v)}")

    # ----- Warnings -----
    if "post_ks_p95" in T.columns:
        warn_mask = T["post_ks_p95"] > float(args.ks95_warn)
        n_warn = int(warn_mask.sum())
        if n_warn > 0:
            print(f"\n[analyze] WARNING: {n_warn} blocks exceed KS95 warn threshold ({fmt_num(args.ks95_warn)}).")

    # ----- Worst by post KS p95 -----
    topk = int(args.print_top)
    print(f"\n[analyze] Worst blocks by post KS (p95), top={topk}:")
    show_cols = [
        "block_dir","block_id","map_label",
        "post_n_dims","post_ks_p95","post_ks_med","post_abs_skew","post_abs_kurt"
    ]
    for c in show_cols:
        if c not in T.columns:
            T[c] = ""  # ensure column exists (map_label may be empty)
    worst = T.sort_values("post_ks_p95", ascending=False).head(topk)
    # Format numerics for display
    worst_disp = worst.copy()
    for c in ["post_n_dims","post_ks_p95","post_ks_med","post_abs_skew","post_abs_kurt"]:
        if c in worst_disp.columns:
            worst_disp[c] = worst_disp[c].apply(fmt_num)
    print(worst_disp[show_cols].to_string(index=False))

    # ----- Least improvement (if pre available) -----
    if "d_ks_med" in T.columns:
        print(f"\n[analyze] Least improvement in KS (pre->post), top={topk}:")
        show_cols2 = [
            "block_dir","block_id","map_label",
            "pre_ks_med","post_ks_med","d_ks_med","d_ks_p95"
        ]
        for c in show_cols2:
            if c not in T.columns:
                T[c] = ""
        least = T.sort_values("d_ks_med", ascending=True).head(topk)
        least_disp = least.copy()
        for c in ["pre_ks_med","post_ks_med","d_ks_med","d_ks_p95"]:
            if c in least_disp.columns:
                least_disp[c] = least_disp[c].apply(fmt_num)
        print(least_disp[show_cols2].to_string(index=False))

    # ----- Optional CSV output -----
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        T.to_csv(out_path, index=False)
        print(f"\n[analyze] Wrote combined CSV to: {out_path}")

if __name__ == "__main__":
    main()
