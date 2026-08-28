#!/usr/bin/env python3
"""
Gaussianity validator for background blocks.

Input:
  --npy <path/to/XB_block_*.npy>  (n_dims, Nens), one block per run
If --npy is omitted, a synthetic block is generated (self-test).

What it does:
  1) Normalize per dimension -> Z0 (Nens, n).
  2) (NEW) Compute PRE-SDE stats on Z0 and save to summary_pre.csv
  3) Push through forward SDE (0 -> 1) -> Z1
  4) Compute POST-SDE stats on Z1 and save to summary.csv
  5) Save optional hist/QQ plots (post; pre optional via --pre-plots)

Outputs in --outdir:
  summary_pre.csv, summary.csv, (plots), metadata.json

Usage example (per your slurm loop):
  python gaussian_validation.py --npy /path/.../XB_block_24.npy --outdir /path/.../gauss_XB_block_24
"""

import os, math, json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ----------------- schedule pieces (match ReverseSDE) -----------------
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
    return float(np.sqrt(max(0.0, g2)))  # guard

# ----------------- helpers: stats & QQ support -----------------
def ks_statistic_standard_normal(samples):
    """One-sample Kolmogorov-Smirnov statistic vs N(0,1), using probit."""
    # Build empirical CDF grid via sorted samples
    s = np.sort(samples)
    n = s.size
    if n == 0:
        return np.nan
    # Theoretical CDF of N(0,1)
    # Convert s -> Phi(s) using error function
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    cdf_theo = 0.5 * (1.0 + erf_stable(s / np.sqrt(2.0)))
    # Empirical CDF points at k-th order statistic are k/n
    u = (np.arange(1, n + 1)) / n
    return float(np.max(np.abs(u - cdf_theo)))

def erf_stable(x):
    # numpy has erf in scipy.special, but to avoid SciPy hard dep, use numpy.poly approx or math.erf if available.
    # Many Python builds have math.erf; fall back to a polynomial if needed.
    try:
        import math as _m
        # math.erf works elementwise on numpy arrays via np.vectorize
        vf = np.vectorize(_m.erf)
        return vf(x)
    except Exception:
        # Abramowitz-Stegun approximation for erf
        # sign-preserving approximation
        sign = np.sign(x)
        ax = np.abs(x)
        t = 1.0 / (1.0 + 0.47047 * ax)
        # Coeffs
        a1, a2, a3 = 0.3480242, -0.0958798, 0.7478556
        y = 1.0 - (a1*t + a2*t*t + a3*t*t*t) * np.exp(-ax*ax)
        return sign * y

def skewness(samples):
    s = np.asarray(samples)
    m = s.mean()
    sd = s.std() + 1e-12
    z = (s - m) / sd
    return float(np.mean(z**3))

def kurtosis_excess(samples):
    s = np.asarray(samples)
    m = s.mean()
    sd = s.std() + 1e-12
    z = (s - m) / sd
    return float(np.mean(z**4) - 3.0)

def _ndtri(p):
    # Robust inverse standard normal CDF (Acklam approximation)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p <= 0.0: return float("-inf")
    if p >= 1.0: return float("inf")
    if p < plow:
        q = math.sqrt(-2.0*math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    elif p > phigh:
        q = math.sqrt(-2.0*math.log(1.0-p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    else:
        q = p-0.5
        r = q*q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    return x

# ----------------- forward SDE (Euler–Maruyama) -----------------
def forward_sde_gaussianize(Z0, psteps=200, eps_alpha=0.05, rng_seed=123):
    rng = np.random.RandomState(int(rng_seed))
    Z = Z0.copy()
    dt = 1.0 / float(psteps)
    t = 0.0
    for _ in range(psteps):
        f = f_drift(t, eps_alpha)
        g = g_diff(t, eps_alpha)
        noise = np.sqrt(dt) * g * rng.randn(*Z.shape)
        Z = Z + dt * (f * Z) + noise
        t = min(1.0, t + dt)
    return Z

# ----------------- main validation for a single block -----------------
def validate_block(XB_block, psteps=200, eps_alpha=0.05, outdir=Path("gauss_check"),
                   max_plots=6, rng_seed=123, report_pre=True, pre_plots=False):
    """
    XB_block: (n_dims, Nens)
    Outputs: summary_pre.csv (if report_pre), summary.csv, plots, metadata.json
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_dims, Nens = XB_block.shape

    # Normalize -> Z0 (Nens, n_dims)
    mu = XB_block.mean(axis=1, keepdims=True)
    sd = XB_block.std(axis=1, keepdims=True) + 1e-12
    Z0 = ((XB_block - mu) / sd).T

    # ---- PRE-SDE stats ----
    if report_pre:
        means0 = Z0.mean(axis=0)
        stds0  = Z0.std(axis=0)
        ks0    = np.array([ks_statistic_standard_normal(Z0[:, i]) for i in range(Z0.shape[1])])
        skw0   = np.array([skewness(Z0[:, i]) for i in range(Z0.shape[1])])
        kurt0  = np.array([kurtosis_excess(Z0[:, i]) for i in range(Z0.shape[1])])
        summary0 = np.vstack([means0, stds0, skw0, kurt0, ks0]).T
        header = ["mean", "std", "skew", "kurtosis_excess", "ks_stat"]
        np.savetxt(outdir / "summary_pre.csv", summary0, delimiter=",", header=",".join(header), comments="")
        if pre_plots:
            _save_some_plots(Z0, outdir, tag="pre", max_plots=max_plots)

    # ---- FORWARD SDE ----
    Z1 = forward_sde_gaussianize(Z0, psteps=psteps, eps_alpha=eps_alpha, rng_seed=rng_seed)

    # ---- POST-SDE stats ----
    means = Z1.mean(axis=0)
    stds  = Z1.std(axis=0)
    ks    = np.array([ks_statistic_standard_normal(Z1[:, i]) for i in range(Z1.shape[1])])
    skw   = np.array([skewness(Z1[:, i]) for i in range(Z1.shape[1])])
    kurt  = np.array([kurtosis_excess(Z1[:, i]) for i in range(Z1.shape[1])])
    summary = np.vstack([means, stds, skw, kurt, ks]).T
    header = ["mean", "std", "skew", "kurtosis_excess", "ks_stat"]
    np.savetxt(outdir / "summary.csv", summary, delimiter=",", header=",".join(header), comments="")

    _save_some_plots(Z1, outdir, tag="post", max_plots=max_plots)

    # metadata
    meta = {
        "n_dims": int(n_dims),
        "Nens": int(Nens),
        "psteps": int(psteps),
        "eps_alpha": float(eps_alpha),
        "rng_seed": int(rng_seed),
        "report_pre": bool(report_pre),
        "pre_plots": bool(pre_plots),
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "means": means, "stds": stds, "skew": skw, "kurtosis_excess": kurt,
        "ks_stat": ks, "outdir": str(outdir),
        "means_pre": (means0 if report_pre else None),
        "stds_pre":  (stds0  if report_pre else None),
        "skew_pre":  (skw0   if report_pre else None),
        "kurtosis_excess_pre": (kurt0 if report_pre else None),
        "ks_stat_pre": (ks0 if report_pre else None),
    }

def _save_some_plots(Z, outdir: Path, tag="post", max_plots=6):
    # Select dims to plot
    n_dims = Z.shape[1]
    if n_dims <= max_plots:
        idxs = np.arange(n_dims)
    else:
        idxs = np.linspace(0, n_dims - 1, max_plots).astype(int)

    for i in idxs:
        s = Z[:, i]
        # Histogram
        plt.figure(figsize=(6,4))
        plt.hist(s, bins=30, density=True)
        plt.title(f"Dim {i} - Histogram ({tag}-SDE)")
        plt.tight_layout()
        plt.savefig(outdir / f"hist_dim_{i}_{tag}.png", bbox_inches="tight")
        plt.close()

        # QQ plot vs N(0,1)
        n = s.size
        probs = (np.arange(1, n+1) - 0.5) / n
        theo_q = np.array([_ndtri(pv) for pv in probs])
        samp_q = np.sort(s)
        plt.figure(figsize=(6,4))
        plt.plot(theo_q, samp_q, ".", ms=3)
        # reference line via quartiles
        q25, q75 = np.quantile(samp_q, [0.25, 0.75])
        t25, t75 = np.quantile(theo_q, [0.25, 0.75])
        slope = (q75 - q25) / (t75 - t25 + 1e-12)
        intercept = q25 - slope * t25
        xx = np.linspace(theo_q.min(), theo_q.max(), 100)
        yy = slope * xx + intercept
        plt.plot(xx, yy, lw=2)
        plt.title(f"Dim {i} - QQ vs N(0,1) ({tag}-SDE)")
        plt.tight_layout()
        plt.savefig(outdir / f"qq_dim_{i}_{tag}.png", bbox_inches="tight")
        plt.close()

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", type=str, default=None, help="Path to XB_block_*.npy (n_dims, Nens). If omitted, use synthetic self-test.")
    ap.add_argument("--outdir", type=str, default="gauss_check")
    ap.add_argument("--psteps", type=int, default=200)
    ap.add_argument("--eps_alpha", type=float, default=0.05)
    ap.add_argument("--rng_seed", type=int, default=123)
    ap.add_argument("--max_plots", type=int, default=6)
    ap.add_argument("--no-pre", action="store_true", help="Disable writing pre-SDE stats")
    ap.add_argument("--pre-plots", action="store_true", help="Also write pre-SDE plots")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load or synthesize XB_block
    if args.npy:
        XB_block = np.load(args.npy)
    else:
        # synthetic self-test: mildly heteroscedastic Gaussian
        rng = np.random.RandomState(0)
        n_dims, Nens = 200, 50
        base = rng.randn(n_dims, Nens)
        scales = 1.0 + 0.15 * rng.randn(n_dims, 1)
        means = 0.1 * rng.randn(n_dims, 1)
        XB_block = means + scales * base

    res = validate_block(
        XB_block,
        psteps=args.psteps,
        eps_alpha=args.eps_alpha,
        outdir=outdir,
        max_plots=args.max_plots,
        rng_seed=args.rng_seed,
        report_pre=(not args.no_pre),
        pre_plots=args.pre_plots
    )
    print(f"[gauss] Wrote pre/post summaries to: {outdir}")

if __name__ == "__main__":
    main()
