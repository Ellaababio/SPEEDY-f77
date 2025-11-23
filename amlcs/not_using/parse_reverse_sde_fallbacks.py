#!/usr/bin/env python3
"""
Minimal parser for ReverseSDE fallback messages in AMLCS DA .out logs.

Parses lines like:
  [ReverseSDE][block=29 [lev=3] vars=TRG1] Non-finite score at step 12. Using fallback for this block.
  [ReverseSDE][block=7  [lev=0] vars=UG0 ] State became non-finite at step 87. Using fallback for this block.

Outputs (fixed-width tables, integers only):
- Total fallback events and how many had an explicit step.
- Most troublesome variables / levels / blocks with:
    count, min_step, median_step, max_step

Usage:
  python parse_reverse_sde_fallbacks_min.py path/to/run.out [more.out ...]
  # Optional: limit rows shown per table (default 15)
  python parse_reverse_sde_fallbacks_min.py run.out --top 20
"""

import argparse
import math
import re
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Optional

# Canonical pattern (with explicit "at step N.")
FALLBACK_RE = re.compile(
    r'\[ReverseSDE\]\[block=(?P<block>\d+)\s+\[lev=(?P<lev>\d+)\]\s+vars=(?P<vars>[^\]]+)\]\s+'
    r'(?P<reason>.+?)\s+at\s+step\s+(?P<step>\d+)\.\s+Using fallback for this block\.',
    flags=re.IGNORECASE
)

# Fallback pattern (no explicit step, still count it)
GENERIC_FALLBACK_RE = re.compile(
    r'\[ReverseSDE\]\[block=(?P<block>\d+)\s+\[lev=(?P<lev>\d+)\]\s+vars=(?P<vars>[^\]]+)\]\s+'
    r'.*?Using fallback for this block\.',
    flags=re.IGNORECASE
)

def median_int(xs: List[int]) -> Optional[int]:
    if not xs:
        return None
    # statistics.median returns float for even counts; convert to nearest int
    m = stats.median(xs)
    # round half away from zero for stability
    return int(math.floor(m + 0.5)) if m >= 0 else int(math.ceil(m - 0.5))

def parse_file(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open("r", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            m = FALLBACK_RE.search(line)
            if m:
                events.append({
                    "block": int(m.group("block")),
                    "level": int(m.group("lev")),
                    "vars":  m.group("vars").strip(),
                    "step":  int(m.group("step")),
                })
                continue
            mg = GENERIC_FALLBACK_RE.search(line)
            if mg:
                events.append({
                    "block": int(mg.group("block")),
                    "level": int(mg.group("lev")),
                    "vars":  mg.group("vars").strip(),
                    "step":  None,
                })
    return events

def group_stats(events: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[Any, Dict[str, Any]] = {}
    for ev in events:
        k = ev[key]
        b = buckets.setdefault(k, {"count": 0, "steps": []})
        b["count"] += 1
        if ev["step"] is not None:
            b["steps"].append(int(ev["step"]))
    rows = []
    for k, b in buckets.items():
        steps = b["steps"]
        if steps:
            mn = min(steps)
            md = median_int(steps)
            mx = max(steps)
        else:
            mn = md = mx = None
        rows.append({
            key: k,
            "count": int(b["count"]),
            "min_step": (int(mn) if mn is not None else None),
            "median_step": (int(md) if md is not None else None),
            "max_step": (int(mx) if mx is not None else None),
            "n_steps": len(steps),
        })
    # sort by count desc, then key
    rows.sort(key=lambda r: (-r["count"], str(r[key])))
    return rows

def print_table(title: str, rows: List[Dict[str, Any]], key: str, top: int):
    # Fixed column widths for clean alignment
    COLS = [
        (key,          20),
        ("count",      7),
        ("n_steps",    8),
        ("min_step",   9),
        ("median_step",12),
        ("max_step",   8),
    ]
    print()
    print(title)
    print("-" * len(title))
    # header
    header = "  " + "  ".join(name.ljust(w) for name, w in COLS)
    print(header)
    print("  " + "  ".join("-" * w for _, w in COLS))
    # rows
    for r in rows[:top]:
        def fmt(v, w):
            if v is None or v == "":
                return "-".rjust(w)
            if isinstance(v, int):
                return str(v).rjust(w)
            # keys (vars) are strings: left-justify and trim if too long
            s = str(v)
            return (s[:w]).ljust(w)
        line = "  " + "  ".join(
            fmt(r[name], w) if name != key else fmt(r[name], w)
            for name, w in COLS
        )
        print(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help=".out files to parse")
    ap.add_argument("--top", type=int, default=15, help="rows to show per table")
    args = ap.parse_args()

    all_events: List[Dict[str, Any]] = []
    for p in args.logs:
        evs = parse_file(Path(p))
        all_events.extend(evs)

    total = len(all_events)
    with_steps = sum(1 for e in all_events if e["step"] is not None)

    print("\n=== ReverseSDE Fallbacks (minimal summary) ===")
    print(f"Total events : {total}")
    print(f"With steps   : {with_steps}")

    # Per-group summaries
    by_vars   = group_stats(all_events, "vars")
    by_levels = group_stats(all_events, "level")
    by_blocks = group_stats(all_events, "block")

    print_table("Most troublesome VARIABLES", by_vars,   "vars",  args.top)
    print_table("Most troublesome LEVELS",    by_levels, "level", args.top)
    print_table("Most troublesome BLOCKS",    by_blocks, "block", args.top)

if __name__ == "__main__":
    main()
