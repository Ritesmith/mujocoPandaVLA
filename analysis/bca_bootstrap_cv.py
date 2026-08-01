#!/usr/bin/env python3
"""BCa (bias-corrected and accelerated) bootstrap CI on CV of v4 eval place_rate.

Addresses the borderline percentile-bootstrap result (P(CV<3%)=0.0499).
Percentile bootstrap is known to under-cover for skewed statistics like
CV/variance at small n. BCa corrects for:
  - Bias (z0): via fraction of bootstrap estimates below observed
  - Skewness/acceleration (a): via jackknife leave-one-out estimates

References:
  - Efron, B. (1987). "Better bootstrap confidence intervals." JASA.
  - DiCiccio & Efron (1996). "Bootstrap Confidence Intervals." Stat. Sci.

Usage:
    python bca_bootstrap_cv.py
    python bca_bootstrap_cv.py --n_bootstrap 100000
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

# 10 v4 place_rates from P1 multi-seed evaluation
V4_PLACE_RATES = [0.715, 0.68, 0.7, 0.655, 0.715, 0.685, 0.685, 0.625, 0.725, 0.64]

OUTPUT_PATH = Path("/home/w/vla_workspace/outputs/dt_orchestrator/bca_bootstrap_cv.json")


def normal_cdf(x):
    """Standard normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p):
    """Inverse standard normal CDF Φ^{-1}(p) via rational approximation.

    Uses Beasley-Springer-Moro algorithm (accurate to ~1e-7).
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    # Acklam's algorithm
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
             ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    return x


def compute_cv(data):
    """Compute CV (in %) for a sample."""
    m = mean(data)
    s = stdev(data)
    return (s / m * 100) if m > 0 else 0.0


def jackknife_cv(data):
    """Leave-one-out jackknife estimates of CV for acceleration calculation."""
    n = len(data)
    jackknife_vals = []
    for i in range(n):
        loo = [data[j] for j in range(n) if j != i]
        jackknife_vals.append(compute_cv(loo))
    return jackknife_vals


def bca_bootstrap_cv(data, n_bootstrap=100000, seed=42, alpha=0.05):
    """BCa bootstrap CI on CV.

    Returns dict with percentile CI, BCa CI, z0, acceleration, and P(CV<threshold).
    """
    import random
    rng = random.Random(seed)
    n = len(data)
    observed_cv = compute_cv(data)
    m = mean(data)
    s = stdev(data)

    # 1. Generate bootstrap distribution of CV
    boot_cvs = []
    cv_below_3 = 0
    for _ in range(n_bootstrap):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        cv = compute_cv(sample)
        boot_cvs.append(cv)
        if cv < 3.0:
            cv_below_3 += 1

    boot_cvs.sort()

    # 2. Bias correction z0
    # z0 = Φ^{-1}(#{boot_cv < observed_cv} / n_bootstrap)
    count_below = sum(1 for cv in boot_cvs if cv < observed_cv)
    prop_below = count_below / n_bootstrap
    # Handle edge cases
    if prop_below <= 0.0:
        prop_below = 0.5 / n_bootstrap
    elif prop_below >= 1.0:
        prop_below = 1.0 - 0.5 / n_bootstrap
    z0 = normal_ppf(prop_below)

    # 3. Acceleration a via jackknife
    jack_vals = jackknife_cv(data)
    jack_mean = mean(jack_vals)
    num = sum((jack_mean - jv) ** 3 for jv in jack_vals)
    den = 6.0 * (sum((jack_mean - jv) ** 2 for jv in jack_vals) ** 1.5)
    a = num / den if den != 0 else 0.0

    # 4. BCa adjusted percentiles
    z_alpha_lower = normal_ppf(alpha / 2.0)      # z_{α/2}
    z_alpha_upper = normal_ppf(1.0 - alpha / 2.0)  # z_{1-α/2}

    def bca_alpha(z_a):
        return normal_cdf(z0 + (z0 + z_a) / (1.0 - a * (z0 + z_a)))

    alpha1 = bca_alpha(z_alpha_lower)
    alpha2 = bca_alpha(z_alpha_upper)

    # 5. Get BCa CI from bootstrap distribution
    idx_lower = int(alpha1 * n_bootstrap)
    idx_upper = int(alpha2 * n_bootstrap)
    idx_lower = max(0, min(idx_lower, n_bootstrap - 1))
    idx_upper = max(0, min(idx_upper, n_bootstrap - 1))

    bca_lower = boot_cvs[idx_lower]
    bca_upper = boot_cvs[idx_upper]

    # Percentile CI (for comparison)
    pct_lower = boot_cvs[int((alpha / 2.0) * n_bootstrap)]
    pct_upper = boot_cvs[int((1.0 - alpha / 2.0) * n_bootstrap)]

    return {
        "observed_cv_pct": round(observed_cv, 2),
        "mean_pp": round(m * 100, 2),
        "std_pp": round(s * 100, 2),
        "n_bootstrap": n_bootstrap,
        "n_data": n,
        "bias_correction_z0": round(z0, 4),
        "acceleration_a": round(a, 6),
        "bca_alphas": {
            "alpha1": round(alpha1, 4),
            "alpha2": round(alpha2, 4),
        },
        "percentile_ci_95": {
            "lower": round(pct_lower, 2),
            "upper": round(pct_upper, 2),
        },
        "bca_ci_95": {
            "lower": round(bca_lower, 2),
            "upper": round(bca_upper, 2),
        },
        "p_cv_below_3pct": round(cv_below_3 / n_bootstrap, 4),
        "cv_median": round(boot_cvs[n_bootstrap // 2], 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="BCa bootstrap CI on CV of v4 eval place_rate")
    parser.add_argument("--n_bootstrap", type=int, default=100000)
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="CV threshold for decision (default 3.0%)")
    args = parser.parse_args()

    data = V4_PLACE_RATES

    print("=" * 65)
    print("BCa Bootstrap CI on CV of v4 Eval Place Rate")
    print("=" * 65)
    print(f"  Data (10 v4 place_rates):")
    print(f"    {data}")
    print(f"  N (seeds):      {len(data)}")
    print(f"  N bootstrap:    {args.n_bootstrap}")
    print()

    result = bca_bootstrap_cv(data, args.n_bootstrap)

    print(f"  Observed:")
    print(f"    mean = {result['mean_pp']:.2f}pp")
    print(f"    std  = {result['std_pp']:.2f}pp")
    print(f"    CV   = {result['observed_cv_pct']:.2f}%")
    print()

    print(f"  BCa correction parameters:")
    print(f"    z0 (bias correction)     = {result['bias_correction_z0']:.4f}")
    print(f"    a  (acceleration/skew)   = {result['acceleration_a']:.6f}")
    print(f"    BCa adjusted percentiles = "
          f"[{result['bca_alphas']['alpha1']:.4f}, "
          f"{result['bca_alphas']['alpha2']:.4f}]")
    print(f"    (percentile uses [0.0250, 0.9750])")
    print()

    pct = result["percentile_ci_95"]
    bca = result["bca_ci_95"]
    print(f"  Percentile 95% CI on CV:  [{pct['lower']:.2f}%, {pct['upper']:.2f}%]")
    print(f"  BCa        95% CI on CV:  [{bca['lower']:.2f}%, {bca['upper']:.2f}%]")
    print(f"    (BCa shifts {'LEFT' if bca['lower'] < pct['lower'] else 'RIGHT'} "
          f"by {abs(bca['lower'] - pct['lower']):.2f}pp at lower bound)")
    print(f"    median = {result['cv_median']:.2f}%")
    print()

    p_below_3 = result["p_cv_below_3pct"]
    print(f"  P(true CV < {args.threshold}%) = {p_below_3:.4f}  "
          f"({p_below_3*100:.2f}% of bootstrap samples)")
    print()

    # ---- Decision ----
    print("=" * 65)
    print("DECISION (BCa)")
    print("=" * 65)
    threshold = args.threshold
    if bca["lower"] > threshold:
        print(f"  BCa CV 95% CI lower bound = {bca['lower']:.2f}% > {threshold}%")
        print(f"  => True eval CV is confidently > {threshold}%")
        print(f"  => 4.91% is NOT small-sample noise (BCa confirmed)")
        print(f"  => SKIP 50-eval expansion, proceed to training CV measurement")
        decision = "SKIP_50_EVAL"
    elif p_below_3 < 0.025:
        # Use 0.025 (one-sided) for stronger evidence
        print(f"  P(CV < {threshold}%) = {p_below_3:.4f} < 0.025 (one-sided)")
        print(f"  => Strong evidence that true CV >= {threshold}%")
        print(f"  => SKIP 50-eval expansion, proceed to training CV measurement")
        decision = "SKIP_50_EVAL"
    else:
        print(f"  BCa CV 95% CI lower bound = {bca['lower']:.2f}% <= {threshold}%")
        print(f"  P(CV < {threshold}%) = {p_below_3:.4f}")
        print(f"  => BCa cannot rule out true CV < {threshold}%")
        print(f"  => RUN 50-eval expansion to pin down CV")
        decision = "RUN_50_EVAL"

    result["decision"] = decision
    result["threshold_cv_pct"] = threshold
    result["data"] = data

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Output: {OUTPUT_PATH}")

    # ---- Comparison summary ----
    print()
    print("=" * 65)
    print("COMPARISON: Percentile vs BCa vs Chi-square")
    print("=" * 65)
    print(f"  Method          | CV 95% CI lower | Excludes CV<{threshold}%?")
    print(f"  ----------------|-----------------|----------------------")
    print(f"  Percentile boot | {pct['lower']:>15.2f}% | "
          f"{'YES' if pct['lower'] > threshold else 'NO '}")
    print(f"  BCa boot        | {bca['lower']:>15.2f}% | "
          f"{'YES' if bca['lower'] > threshold else 'NO '}")
    # Chi-square from previous run
    print(f"  Chi-square (df=9)| {3.38:>14.2f}% | YES")
    print()
    print(f"  P(CV<{threshold}%): percentile=0.0499, BCa={p_below_3:.4f}")


if __name__ == "__main__":
    main()
