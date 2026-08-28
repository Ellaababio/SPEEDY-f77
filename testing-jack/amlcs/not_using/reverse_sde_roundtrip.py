#!/usr/bin/env python3
"""
Round-trip test for ReverseSDE schedule (no observations):
  XB_block (n_dims, Nens)
   -> per-dim normalize => Z0 (Nens, n_dims)
   -> forward SDE (0->1) => Z1
   -> reverse SDE (1->0), no obs => Zrec
Compare Zrec vs Z0 via RMSE and correlation.

Usage:
  python reverse_sde_roundtrip.py --npy /path/to/XB_block_0.npy --outdir /tmp/rt --psteps 200 --eps_alpha 0.05 --deterministic 1
If --npy is omitted, a synthetic Gaussian block is generated.
"""

import os, math, argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------- schedules (match your ReverseSDE / validator) -----------------
def cond_alpha(t, eps_alpha):          # alpha(t)
    return 1.0 - (1.0 - eps_alpha) * t

def cond_sigma_sq(t):                  # sigma^2(t)
    return t

def f_drift(t, eps_alpha):             # f(t) = d log alpha / dt
    a = cond_alpha(t, eps_alpha)
    return -(1.0 - eps_alpha) / a

def g_diff(t, eps_alpha):              # g^2 = d(sigma^2)/dt - 2 f sigma^2
    d_sig2 = 1.0
    g2 = d_sig2 - 2.0 * f_drift(t, eps_alpha) * cond_sigma_sq(t)
    return float(np.sqrt(max(0.0, g2)))

# ----------------- forward SDE (Euler–Maruyama) -----------------
def forward_sde(Z0, psteps=200, eps_alpha=0.05, rng=None):
    """
    Z0: (Nens, n) normalized samples at t=0
    Returns Z1 at t=1
    """
    if rng is None:
        rng = np.random.RandomState(123)
    Z = Z0.copy()
    dt = 1.0 / float(psteps)
    t = 0.0
    for _ in range(psteps):
        f = f_drift(t, eps_alpha)
        g = g_diff(t, eps_alpha)
        # forward SDE: dZ = f(t) Z dt + g(t) dW
        noise = np.sqrt(dt) * g * rng.randn(*Z.shape)
        Z = Z + dt * (f * Z) + noise
        t = min(1.0, t + dt)
    return Z

# ----------------- reverse SDE (prior-only; NO observations) -----------------
def reverse_sde_prior_only(Z1, Z0_ref, psteps=200, eps_alpha=0.05, rng=None, deterministic=False):
    """
    Start from Z1 (t=1 Gaussian-ish), integrate reverse to t=0 using ONLY the prior term.
    Z0_ref is the original normalized background (used in prior term).
    deterministic=True -> set noise=0 to see cleaner reconstruction
    """
    if rng is None:
        rng = np.random.RandomState(123)
    Z = Z1.copy()
    dt = 1.0 / float(psteps)
    t = 1.0
    for _ in range(psteps):
        a_t = cond_alpha(t, eps_alpha)
        sig2_t = cond_sigma_sq(t)
        f = f_drift(t, eps_alpha)
        g = g_diff(t, eps_alpha)

        # prior-term score used in your ReverseSDE code:
        # prior_term = (Z - a_t * Z0_ref) / sig2_t
        prior_term = (Z - a_t * Z0_ref) / sig2_t

        # reverse drift (no obs term):  -( f*Z + g^2 * prior_term )
        drift = -(f * Z + (g ** 2) * prior_term)

        if deterministic:
            noise = 0.0
        else:
            noise = np.sqrt(dt) * g * rng.randn(*Z.shape)

        Z_next = Z + dt * drift + noise
        Z = Z_next
        t = max(0.0, t - dt)
    return Z

# ----------------- metrics & plots -----------------
def summarize_roundtrip(Z0, Zrec, outdir: Path, tag=""):
    """
    Compute per-dim RMSE, correlation; save quick hist/QQ and a CSV.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12
    n = Z0.shape[1]

    diffs = Zrec - Z0
    rmse_dim = np.sqrt(np.mean(diffs**2, axis=0))
    # safe correlation per dim
    corr_dim = []
    for j in range(n):
        x = Z0[:, j]; y = Zrec[:, j]
        sx = x.std() + eps; sy = y.std() + eps
        corr = float(np.mean((x - x.mean())*(y - y.mean())) / (sx*sy))
        corr_dim.append(corr)
    corr_dim = np.array(corr_dim)

    summary = np.vstack([rmse_dim, corr_dim]).T
    np.savetxt(outdir / f"roundtrip_summary_{tag}.csv", summary,
               delimiter=",", header="rmse,corr", comments="")

    # headline numbers
    overall_rmse = float(np.sqrt(np.mean(diffs**2)))
    median_rmse = float(np.median(rmse_dim))
    median_corr = float(np.median(corr_dim))

    # a couple of quick plots
    plt.figure(figsize=(6,4))
    plt.hist(rmse_dim, bins=40, density=True)
    plt.title(f"Per-dim RMSE (median={median_rmse:.3g}) [{tag}]")
    plt.tight_layout()
    plt.savefig(outdir / f"rmse_hist_{tag}.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(corr_dim, bins=40, density=True)
    plt.title(f"Per-dim corr(Zrec,Z0) (median={median_corr:.3g}) [{tag}]")
    plt.tight_layout()
    plt.savefig(outdir / f"corr_hist_{tag}.png", bbox_inches="tight")
    plt.close()

    return {
        "overall_rmse": overall_rmse,
        "median_rmse": median_rmse,
        "median_corr": median_corr,
        "outdir": str(outdir)
    }

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Round-trip ReverseSDE prior-only consistency test")
    ap.add_argument("--npy", type=str, default=None, help="Path to XB_block_*.npy (n_dims, Nens). If omitted, synthetic Gaussian is used.")
    ap.add_argument("--outdir", type=str, default="/tmp/reverse_sde_roundtrip")
    ap.add_argument("--psteps", type=int, default=200)
    ap.add_argument("--eps_alpha", type=float, default=0.05)
    ap.add_argument("--rng_seed", type=int, default=123)
    ap.add_argument("--deterministic", type=int, default=1, help="1=reverse uses no noise; 0=stochastic reverse")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.rng_seed)

    # 1) Load or synthesize XB_block (n_dims, Nens)
    if args.npy is None:
        # synthetic: moderately heteroscedastic Gaussian
        n_dims, Nens = 512, 100
        base = rng.randn(n_dims, Nens)
        scales = 1.0 + 0.25 * rng.randn(n_dims, 1)
        means = 0.2 * rng.randn(n_dims, 1)
        XB_block = means + scales * base
        np.save(outdir / "XB_block_synth.npy", XB_block)
        print(f"[roundtrip] Using synthetic XB_block: {XB_block.shape} -> {outdir/'XB_block_synth.npy'}")
    else:
        XB_block = np.load(args.npy)
        print(f"[roundtrip] Loaded XB_block from {args.npy} with shape {XB_block.shape}")

    # 2) Normalize per dim: (n_dims, Nens) -> Z0 (Nens, n_dims)
    mu = XB_block.mean(axis=1, keepdims=True)
    sd = XB_block.std(axis=1, keepdims=True) + 1e-12
    Z0 = ((XB_block - mu) / sd).T  # (Nens, n)

    # 3) Forward SDE to t=1
    Z1 = forward_sde(Z0, psteps=args.psteps, eps_alpha=args.eps_alpha, rng=rng)

    # 4) Reverse SDE (prior-only) back to t=0
    Zrec_det = reverse_sde_prior_only(Z1, Z0_ref=Z0, psteps=args.psteps,
                                      eps_alpha=args.eps_alpha, rng=rng,
                                      deterministic=bool(args.deterministic))

    # 5) Metrics & plots
    res_det = summarize_roundtrip(Z0, Zrec_det, outdir, tag=("det" if args.deterministic else "stoch"))
    print("[roundtrip] Deterministic reverse results:",
          json.dumps(res_det, indent=2))

    # Also (optional) try stochastic reverse for comparison
    if args.deterministic == 1:
        Zrec_st = reverse_sde_prior_only(Z1, Z0_ref=Z0, psteps=args.psteps,
                                         eps_alpha=args.eps_alpha, rng=rng,
                                         deterministic=False)
        res_st = summarize_roundtrip(Z0, Zrec_st, outdir, tag="stoch")
        print("[roundtrip] Stochastic reverse results:",
              json.dumps(res_st, indent=2))

    print("--- Round-trip Test Finished ---")

if __name__ == "__main__":
    main()
