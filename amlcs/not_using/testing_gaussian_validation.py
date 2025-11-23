#!/usr/bin/env python3
"""
Run three sanity checks against gaussian_validation.validate_block:

A) Standard Normal (should look Gaussian)
B) Non-Gaussian (mixture + lognormal) -- should be flagged
C) Correlated Gaussian (still should look Gaussian after per-dim normalization)
"""

import numpy as np
from pathlib import Path
from gaussian_validation import validate_block  # uses same forward SDE + diagnostics

def pretty_stats(tag, res):
    ks_med = float(np.median(res["ks_stat"]))
    ks_p95 = float(np.quantile(res["ks_stat"], 0.95))
    s_med  = float(np.median(res["stds"]))
    m_med  = float(np.median(res["means"]))
    skew_m = float(np.median(np.abs(res["skew"])))
    kurt_m = float(np.median(np.abs(res["kurtosis_excess"])))
    print(f"\n[{tag}]")
    print(f"  median |mean|            : {abs(m_med):.4f}")
    print(f"  median std               : {s_med:.4f}")
    print(f"  median |skew|            : {skew_m:.4f}")
    print(f"  median |kurtosis_excess| : {kurt_m:.4f}")
    print(f"  median KS                : {ks_med:.4f}")
    print(f"  95th pct KS              : {ks_p95:.4f}")

def case_A_standard_normal(out_root=Path("/tmp/gauss_ok"), n_dims=200, Nens=100, seed=0):
    np.random.seed(seed)
    XB_block = np.random.randn(n_dims, Nens)  # already Gaussian
    return validate_block(
        XB_block, psteps=200, eps_alpha=0.05,
        outdir=out_root, max_plots=6, rng_seed=123
    )

def case_B_non_gaussian(out_root=Path("/tmp/gauss_bad"), n_dims=200, Nens=100, seed=1):
    np.random.seed(seed)
    # Bimodal mixture (centered at ±2) in half the dims
    mix = (np.random.rand(n_dims, Nens) < 0.5).astype(float)
    XB_mix = mix * (np.random.randn(n_dims, Nens) - 2.0) + (1.0 - mix) * (np.random.randn(n_dims, Nens) + 2.0)
    # Lognormal (skewed/heavy-tailed) in the other half
    XB_logn = np.exp(0.8 * np.random.randn(n_dims, Nens)) - np.e
    # Stack to form one big block (2*n_dims × Nens)
    XB_block = np.vstack([XB_mix, XB_logn])
    return validate_block(
        XB_block, psteps=200, eps_alpha=0.05,
        outdir=out_root, max_plots=6, rng_seed=123
    )

def case_C_correlated_gaussian(out_root=Path("/tmp/gauss_corr"), n_dims=300, Nens=100, seed=2):
    np.random.seed(seed)
    # Build a random SPD covariance and draw a correlated Gaussian
    A = np.random.randn(n_dims, n_dims)
    C = A @ A.T + 0.1 * np.eye(n_dims)
    X = np.random.multivariate_normal(mean=np.zeros(n_dims), cov=C, size=Nens).T  # (n_dims, Nens)
    return validate_block(
        X, psteps=200, eps_alpha=0.05,
        outdir=out_root, max_plots=6, rng_seed=123
    )

def main():
    resA = case_A_standard_normal()
    pretty_stats("A: Standard Normal", resA)

    resB = case_B_non_gaussian()
    pretty_stats("B: Mixture + Lognormal (non-Gaussian)", resB)

    resC = case_C_correlated_gaussian()
    pretty_stats("C: Correlated Gaussian", resC)

    print("\nOutputs:")
    print(f"  A → {resA['outdir']}")
    print(f"  B → {resB['outdir']}")
    print(f"  C → {resC['outdir']}")

if __name__ == "__main__":
    main()
