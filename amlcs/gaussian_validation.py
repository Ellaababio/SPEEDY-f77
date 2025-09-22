import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

def normal_cdf(x):
    # CDF of standard normal using erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def ks_statistic_standard_normal(samples):
    """Compute one-sample KS statistic of 'samples' (1-D array) vs N(0,1)."""
    x = np.sort(np.asarray(samples).ravel())
    n = x.size
    if n == 0:
        return np.nan
    ecdf = (np.arange(1, n + 1)) / n
    tcdf = np.array([normal_cdf(v) for v in x])
    return float(np.max(np.abs(ecdf - tcdf)))

def skewness(samples):
    x = np.asarray(samples).ravel()
    n = x.size
    if n < 3:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s == 0:
        return 0.0
    m3 = np.mean((x - m)**3)
    return float(m3 / (s**3))

def kurtosis_excess(samples):
    x = np.asarray(samples).ravel()
    n = x.size
    if n < 4:
        return np.nan
    m = np.mean(x)
    s = np.std(x, ddof=0)
    if s == 0:
        return -3.0
    m4 = np.mean((x - m)**4)
    return float(m4 / (s**4) - 3.0)

def forward_sde_gaussianize(Z0, psteps=200, eps_alpha=0.05, rng_seed=123):
    """Simulate the forward SDE: dZ = f(t)Z dt + g(t)dW, from t=0 to 1."""
    rng = np.random.RandomState(int(rng_seed))
    dt = 1.0 / float(psteps)
    t = 0.0
    Z = np.asarray(Z0, dtype=float).copy()  # (Nens, n)

    def alpha(tt):
        return 1.0 - (1.0 - eps_alpha) * tt
    def f(tt):
        a = alpha(tt)
        return - (1.0 - eps_alpha) / a
    def g(tt):
        g2 = 1.0 - 2.0 * f(tt) * tt
        if g2 < 0: g2 = 0.0
        return math.sqrt(g2)

    for _ in range(psteps):
        ft = f(t)
        gt = g(t)
        dW = rng.randn(*Z.shape) * math.sqrt(dt)
        Z = Z + dt * (ft * Z) + gt * dW
        t += dt
        if t > 1.0: t = 1.0
    return Z

def validate_block(XB_block, psteps=200, eps_alpha=0.05, outdir=Path("gauss_check"),
                   max_plots=6, rng_seed=123):
    """
    XB_block: numpy array (n_dims, Nens)
    1) Normalize each dim.
    2) Forward SDE to t=1.
    3) Test Gaussianity (mean, std, skew, kurt, KS).
    4) Save summary + plots.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_dims, Nens = XB_block.shape
    mu = XB_block.mean(axis=1, keepdims=True)
    sd = XB_block.std(axis=1, keepdims=True) + 1e-12
    Z0 = ((XB_block - mu) / sd).T  # (Nens, n)

    Z1 = forward_sde_gaussianize(Z0, psteps=psteps, eps_alpha=eps_alpha, rng_seed=rng_seed)

    means = Z1.mean(axis=0)
    stds  = Z1.std(axis=0)
    ks    = np.array([ks_statistic_standard_normal(Z1[:, i]) for i in range(Z1.shape[1])])
    skw   = np.array([skewness(Z1[:, i]) for i in range(Z1.shape[1])])
    kurt  = np.array([kurtosis_excess(Z1[:, i]) for i in range(Z1.shape[1])])

    summary = np.vstack([means, stds, skw, kurt, ks]).T
    header = ["mean", "std", "skew", "kurtosis_excess", "ks_stat"]
    np.savetxt(outdir / "summary.csv", summary, delimiter=",", header=",".join(header), comments="")

    if Z1.shape[1] <= max_plots:
        idxs = np.arange(Z1.shape[1])
    else:
        idxs = np.linspace(0, Z1.shape[1]-1, max_plots).astype(int)

    # robust inverse normal CDF
    def _ndtri(p):
        # Acklam's approximation
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
            x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        elif p > phigh:
            q = math.sqrt(-2.0*math.log(1.0-p))
            x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        else:
            q = p-0.5
            r = q*q
            x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        return x

    for i in idxs:
        samples = Z1[:, i]
        # histogram
        plt.figure(figsize=(6,4))
        plt.hist(samples, bins=30, density=True)
        plt.title(f"Dim {i} - Histogram")
        plt.tight_layout()
        plt.savefig(outdir / f"hist_dim_{i}.png", bbox_inches="tight")
        plt.close()
        # QQ plot
        plt.figure(figsize=(6,4))
        n = samples.size
        probs = (np.arange(1, n+1) - 0.5) / n
        theo_q = np.array([_ndtri(pv) for pv in probs])
        samp_q = np.sort(samples)
        plt.plot(theo_q, samp_q, marker=".", linestyle="None")
        q25, q75 = np.quantile(samples, [0.25, 0.75])
        t25, t75 = np.quantile(theo_q, [0.25, 0.75])
        slope = (q75 - q25) / (t75 - t25 + 1e-12)
        intercept = q25 - slope * t25
        xline = np.linspace(theo_q.min(), theo_q.max(), 100)
        yline = slope * xline + intercept
        plt.plot(xline, yline)
        plt.title(f"Dim {i} - QQ plot vs N(0,1)")
        plt.tight_layout()
        plt.savefig(outdir / f"qq_dim_{i}.png", bbox_inches="tight")
        plt.close()

    return {
        "means": means, "stds": stds, "skew": skw,
        "kurtosis_excess": kurt, "ks_stat": ks,
        "indices_plotted": idxs, "outdir": str(outdir)
    }

def main():
    parser = argparse.ArgumentParser(description="Gaussianity validation for forward SDE on BACKGROUND block")
    parser.add_argument("--npy", type=str, default="", help="Path to npy file containing XB_block (n_dims, Nens)")
    parser.add_argument("--outdir", type=str, default="gauss_check", help="Output directory")
    parser.add_argument("--psteps", type=int, default=200, help="Pseudo-time steps")
    parser.add_argument("--eps_alpha", type=float, default=0.05, help="Epsilon for alpha schedule")
    parser.add_argument("--rng_seed", type=int, default=123, help="Random seed")
    parser.add_argument("--max_plots", type=int, default=6, help="Number of dimensions to plot")
    args = parser.parse_args()

    if args.npy:
        XB_block = np.load(args.npy)
    else:
        n_dims, Nens = 200, 50
        rng = np.random.RandomState(0)
        XB_block = rng.randn(n_dims, Nens) * (1+0.1*rng.randn(n_dims,1)) + 2.0*rng.randn(n_dims,1)

    res = validate_block(XB_block,
        psteps=args.psteps, eps_alpha=args.eps_alpha,
        outdir=Path(args.outdir), max_plots=args.max_plots,
        rng_seed=args.rng_seed)
    print("Saved outputs to:", res["outdir"])

if __name__ == "__main__":
    main()
