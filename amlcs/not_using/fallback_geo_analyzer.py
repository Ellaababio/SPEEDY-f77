#!/usr/bin/env python3
"""
Join ReverseSDE fallback events with block -> grid mapping to find geospatial hotspots.

Inputs:
  - block_map_detailed.json  (from ReverseSDE.dump_block_map(include_grid=True))
  - one or more .out logs that contain lines like:
      [ReverseSDE][block=29 [lev=3] vars=TRG1] Non-finite score at step 12. Using fallback for this block.

Outputs (CSV in --outdir):
  - geo_fallback_cells.csv   : counts per (var, lev, ilat, jlon)
  - geo_fallback_lat.csv     : counts per (var, lev, ilat)
  - geo_fallback_lon.csv     : counts per (var, lev, jlon)
  - geo_fallback_level.csv   : counts per (var, lev)

Console: prints top hotspots with fixed-width tables.

Usage:
  python fallback_geo_analyzer.py \
      --map /path/to/block_map_detailed.json \
      --logs run1.out run2.out \
      --outdir /path/to/geo_diag \
      --vars TRG0 TRG1 \
      --top 20
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

# --- Parse fallback lines ---
FALLBACK_RE = re.compile(
    r'\[ReverseSDE\]\[block=(?P<block>\d+)\s+\[lev=(?P<lev>\d+)\]\s+vars=(?P<vars>[^\]]+)\]\s+'
    r'(?P<reason>.+?)\s+at\s+step\s+(?P<step>\d+)\.\s+Using fallback for this block\.',
    flags=re.IGNORECASE
)
GENERIC_FALLBACK_RE = re.compile(
    r'\[ReverseSDE\]\[block=(?P<block>\d+)\s+\[lev=(?P<lev>\d+)\]\s+vars=(?P<vars>[^\]]+)\]\s+'
    r'.*?Using fallback for this block\.',
    flags=re.IGNORECASE
)

def parse_fallbacks(log_paths):
    """Return list of events: dict(block:int, level:int, vars:str)."""
    events = []
    for p in log_paths:
        with open(p, "r", errors="ignore") as f:
            for line in f:
                m = FALLBACK_RE.search(line)
                if m:
                    events.append({
                        "block": int(m.group("block")),
                        "level": int(m.group("lev")),
                        "vars":  m.group("vars").strip(),
                    })
                    continue
                mg = GENERIC_FALLBACK_RE.search(line)
                if mg:
                    events.append({
                        "block": int(mg.group("block")),
                        "level": int(mg.group("lev")),
                        "vars":  mg.group("vars").strip(),
                    })
    return events

def load_block_map(block_map_path):
    """
    Load block_map_detailed.json and return:
      blocks: dict[block_idx] -> list of entries {var, lev, ilat, jlon}
    """
    with open(block_map_path, "r") as f:
        data = json.load(f)
    # Expected shape: {"var_names": [...], "blocks": [{block_idx:int, entries:[{var,lev,ilat,jlon}, ...], ...}, ...]}
    blocks = {}
    for b in data.get("blocks", []):
        idx = int(b["block_idx"])
        entries = b.get("entries", [])
        # Ensure fields exist
        clean = []
        for e in entries:
            try:
                clean.append({
                    "var":  str(e["var"]),
                    "lev":  int(e["lev"]),
                    "ilat": int(e["ilat"]),
                    "jlon": int(e["jlon"]),
                })
            except Exception:
                # Skip malformed entry
                continue
        blocks[idx] = clean
    return blocks

def fixed_table(title, rows, cols, widths, top=20):
    print(f"\n{title}")
    print("-" * len(title))
    # header
    line = "  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(line)
    print("  " + "  ".join("-" * w for w in widths))
    # rows
    for r in rows[:top]:
        out = []
        for c, w in zip(cols, widths):
            v = r.get(c, "")
            if isinstance(v, int):
                out.append(str(v).rjust(w))
            else:
                s = str(v)
                out.append(s[:w].ljust(w))
        print("  " + "  ".join(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="Path to block_map_detailed.json")
    ap.add_argument("--logs", nargs="+", required=True, help="One or more .out files to parse")
    ap.add_argument("--outdir", required=True, help="Directory to write CSV summaries")
    ap.add_argument("--vars", nargs="*", default=["TRG0", "TRG1"], help="Only include these variable names (default: TRG0 TRG1)")
    ap.add_argument("--top", type=int, default=20, help="How many hotspots to print per table")
    args = ap.parse_args()

    blocks = load_block_map(args.map)
    events = parse_fallbacks(args.logs)

    # Filter events by variable name(s) if provided
    var_filter = set(args.vars) if args.vars else None
    if var_filter:
        events = [e for e in events if e["vars"] in var_filter]

    # Count fallbacks per grid cell by expanding each block to its entries
    cell_counts = defaultdict(int)    # (var, lev, ilat, jlon) -> count
    lat_counts  = defaultdict(int)    # (var, lev, ilat) -> count
    lon_counts  = defaultdict(int)    # (var, lev, jlon) -> count
    lev_counts  = defaultdict(int)    # (var, lev) -> count
    missing_blocks = 0

    for ev in events:
        b = ev["block"]
        entries = blocks.get(b)
        if not entries:
            missing_blocks += 1
            continue
        # Only count entries that match the variable in the event (defensive)
        for en in entries:
            if var_filter and en["var"] not in var_filter:
                continue
            # Note: levels should already match; keep the filter tight anyway
            if "level" in ev and en["lev"] != ev["level"]:
                # Some builds may store block level set; if mismatch, you can relax this
                pass
            key_cell = (en["var"], en["lev"], en["ilat"], en["jlon"])
            cell_counts[key_cell] += 1
            lat_counts[(en["var"], en["lev"], en["ilat"])] += 1
            lon_counts[(en["var"], en["lev"], en["jlon"])] += 1
            lev_counts[(en["var"], en["lev"])] += 1

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write CSVs
    def write_csv(path, header, rows):
        with open(path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(r[h]) for h in header) + "\n")

    # cells
    rows_cells = [
        {"var": v, "lev": L, "ilat": i, "jlon": j, "count": c}
        for (v, L, i, j), c in cell_counts.items()
    ]
    rows_cells.sort(key=lambda r: (-r["count"], r["var"], r["lev"], r["ilat"], r["jlon"]))
    write_csv(outdir/"geo_fallback_cells.csv",
              ["var","lev","ilat","jlon","count"], rows_cells)

    # by lat
    rows_lat = [
        {"var": v, "lev": L, "ilat": i, "count": c}
        for (v, L, i), c in lat_counts.items()
    ]
    rows_lat.sort(key=lambda r: (-r["count"], r["var"], r["lev"], r["ilat"]))
    write_csv(outdir/"geo_fallback_lat.csv",
              ["var","lev","ilat","count"], rows_lat)

    # by lon
    rows_lon = [
        {"var": v, "lev": L, "jlon": j, "count": c}
        for (v, L, j), c in lon_counts.items()
    ]
    rows_lon.sort(key=lambda r: (-r["count"], r["var"], r["lev"], r["jlon"]))
    write_csv(outdir/"geo_fallback_lon.csv",
              ["var","lev","jlon","count"], rows_lon)

    # by level
    rows_lev = [
        {"var": v, "lev": L, "count": c}
        for (v, L), c in lev_counts.items()
    ]
    rows_lev.sort(key=lambda r: (-r["count"], r["var"], r["lev"]))
    write_csv(outdir/"geo_fallback_level.csv",
              ["var","lev","count"], rows_lev)

    # Console top tables
    if rows_cells:
        fixed_table("Top grid-cell hotspots (var, lev, ilat, jlon)",
                    rows_cells,
                    cols=["var","lev","ilat","jlon","count"],
                    widths=[6,5,6,6,7],
                    top=args.top)

    if rows_lat:
        fixed_table("Top latitude-index hotspots (var, lev, ilat)",
                    rows_lat,
                    cols=["var","lev","ilat","count"],
                    widths=[6,5,6,7],
                    top=args.top)

    if rows_lon:
        fixed_table("Top longitude-index hotspots (var, lev, jlon)",
                    rows_lon,
                    cols=["var","lev","jlon","count"],
                    widths=[6,5,6,7],
                    top=args.top)

    if rows_lev:
        fixed_table("Top level hotspots (var, lev)",
                    rows_lev,
                    cols=["var","lev","count"],
                    widths=[6,5,7],
                    top=args.top)

    if missing_blocks:
        print(f"\n[info] {missing_blocks} fallback events referenced blocks not present in the map (skipped).")

if __name__ == "__main__":
    main()
