#!/usr/bin/env python3
"""
augment_block_map_with_degrees.py

Add lat/lon degrees to an existing block_map_detailed.json by probing nm.gs arrays.

Usage:
  python augment_block_map_with_degrees.py --in path/block_map_detailed.json --out path/block_map_detailed_deg.json
"""
import argparse, json, sys

def probe_lat_lon():
    """
    Best-effort probe for lat/lon arrays in the current environment.
    Assumes you can import the same nm/gs the DA uses.
    Modify this function if your grid access is different.
    """
    try:
        # Example: if you can import a module exposing nm or gs
        # from amlcs.numerical_model import global_nm
        # gs = global_nm.gs
        gs = None  # <-- replace with your access point if needed
        if gs is None:
            return None, None

        lats = None
        lons = None
        for nm in ["lats", "lat", "latitudes", "phi", "phi_deg"]:
            if hasattr(gs, nm):
                lats = getattr(gs, nm)
                break
        for nm in ["lons", "lon", "longitudes", "lambda_", "lambda_deg"]:
            if hasattr(gs, nm):
                lons = getattr(gs, nm)
                break
        if lats is not None and hasattr(lats, "tolist"):
            lats = lats.tolist()
        if lons is not None and hasattr(lons, "tolist"):
            lons = lons.tolist()
        return lats, lons
    except Exception:
        return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    with open(args.inp, "r") as f:
        data = json.load(f)

    lats, lons = probe_lat_lon()
    if lats is None or lons is None:
        print("Could not find latitude/longitude arrays; no degrees added.", file=sys.stderr)
    else:
        n_added = 0
        for b in data.get("blocks", []):
            for e in b.get("entries", []):
                try:
                    e["lat_deg"] = float(lats[int(e["ilat"])])
                    e["lon_deg"] = float(lons[int(e["jlon"])])
                    n_added += 1
                except Exception:
                    pass
        print(f"Added degrees to {n_added} entries.")

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
