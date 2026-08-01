#!/usr/bin/env python3
"""Bootstrap 95% CI on CV (and std) of v4 eval place_rate.

Answers: "Could the true eval CV be < 3%, with 4.91% just small-sample noise?"

Uses the 10 v4 place_rates from P1 multi-seed evaluation. Bootstraps
the CV (std/mean) to get a confidence interval. If the lower bound of
the 95% CI on CV is > 3%, we can confidently say eval variance is real
and skip the proposed 50-eval expansion.

Also reports:
  - Bootstrap CI on std (pp)
  - Bootstrap CI on mean (pp)
  - Probability that true CV < 3% (fraction of bootstrap samples with CV < 3%)
  - Analytical chi-square CI on std (for cross-validation)

Usage:
    python bootstrap_cv_ci.py
    python bootstrap_cv_ci.py --n_bootstrap 50000
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

# 10 v4 place_rates from P1 multi-seed evaluation
# (extracted from outputs/dt_orchestrator/multi_seed_results.json)
V4_PLACE_RATES = [0.715, 0.68, 0.7, 0.655, 0.715, 0.685, 0.685, 0.625, 0.725, 0.64]

OUTPUT_PATH = Path("/home/w/vla_workspace/outputs/dt_orchestrator/bootstrap_cv_ci.json")


def bootstrap_cv_ci(data, n_bootstrap=50000, seed=42):
    """Bootstrap CI on CV, std, and mean."""
    import random
    rng = random.Random(seed)
    n = len(data)
    m = mean(data)
    s = stdev(data)  # sample std (n-1 denominator)

    cv_samples = []
    std_samples = []
    mean_samples = []
    cv_below_3 = 0

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = [data[rng.randrange(n)] for _ in range(n)]
        sm = mean(sample)
        ss = stdev(sample) if len(sample) > 1 else 0.0
        cv = (ss / sm * 100) if sm > 0 else 0
        cv_samples.append(cv)
        std_samples.append(ss * 100)  # in pp
        mean_samples.append(sm * 100)  # in pp
        if cv < 3.0:
            cv_below_3 += 1

    cv_samples.sort()
    std_samples.sort()
    mean_samples.sort()

    def pct(sorted_list, p):
        idx = int(len(sorted_list) * p)
        return sorted_list[min(idx, len(sorted_list) - 1)]

    return {
        "n_bootstrap": n_bootstrap,
        "n_data": n,
        "observed": {
            "mean_pp": round(m * 100, 2),
            "std_pp": round(s * 100, 2),
            "cv_pct": round(s / m * 100, 2),
        },
        "cv_ci_95": {
            "lower": round(cv_samples[int(0.025 * n_bootstrap)], 2),
            "upper": round(cv_samples[int(0.975 * n_bootstrap)], 2),
        },
        "std_ci_95": {
            "lower": round(std_samples[int(0.025 * n_bootstrap)], 2),
            "upper": round(std_samples[int(0.975 * n_bootstrap)], 2),
        },
        "mean_ci_95": {
            "lower": round(mean_samples[int(0.025 * n_bootstrap)], 2),
            "upper": round(mean_samples[int(0.975 * n_bootstrap)], 2),
        },
        "p_cv_below_3pct": round(cv_below_3 / n_bootstrap, 4),
        "cv_median": round(cv_samples[n_bootstrap // 2], 2),
        "cv_p5": round(pct(cv_samples, 0.05), 2),
        "cv_p95": round(pct(cv_samples, 0.95), 2),
    }


def analytical_std_ci(data, alpha=0.05):
    """Analytical chi-square CI on std (assumes normality, for cross-check)."""
    n = len(data)
    s = stdev(data)
    # Chi-square critical values for df = n-1
    # For df=9, alpha=0.05: chi2_lower=2.700, chi2_upper=19.023
    df = n - 1
    # Use scipy if available, otherwise hardcode for df=9
    try:
        from scipy.stats import chi2
        chi2_lower = chi2.ppf(alpha / 2, df)
        chi2_upper = chi2.ppf(1 - alpha / 2, df)
    except ImportError:
        # Hardcoded for df=9 (n=10)
        if df == 9:
            chi2_lower = 2.700
            chi2_upper = 19.023
        else:
            return None
    std_lower = s * math.sqrt(df / chi2_upper)
    std_upper = s * math.sqrt(df / chi2_lower)
    m = mean(data)
    return {
        "method": f"chi-square (df={df}, assumes normality)",
        "std_lower_pp": round(std_lower * 100, 2),
        "std_upper_pp": round(std_upper * 100, 2),
        "cv_lower_pct": round(std_lower / m * 100, 2),
        "cv_upper_pct": round(std_upper / m * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap 95% CI on CV of v4 eval place_rate")
    parser.add_argument("--n_bootstrap", type=int, default=50000)
    parser.add_argument("--data", type=str, default=None,
                        help="Comma-separated place_rates (default: P1 v4 data)")
    args = parser.parse_args()

    if args.data:
        data = [float(x.strip()) for x in args.data.split(",")]
    else:
        data = V4_PLACE_RATES

    print("=" * 65)
    print("Bootstrap CI on CV of v4 Eval Place Rate")
    print("=" * 65)
    print(f"  Data:           {data}")
    print(f"  N (seeds):      {len(data)}")
    print(f"  N bootstrap:    {args.n_bootstrap}")
    print()

    result = bootstrap_cv_ci(data, args.n_bootstrap)
    analytical = analytical_std_ci(data)

    obs = result["observed"]
    print(f"  Observed:")
    print(f"    mean = {obs['mean_pp']:.2f}pp")
    print(f"    std  = {obs['std_pp']:.2f}pp")
    print(f"    CV   = {obs['cv_pct']:.2f}%")
    print()

    cv_ci = result["cv_ci_95"]
    std_ci = result["std_ci_95"]
    mean_ci = result["mean_ci_95"]
    print(f"  Bootstrap 95% CI on CV:")
    print(f"    [{cv_ci['lower']:.2f}%, {cv_ci['upper']:.2f}%]")
    print(f"    median = {result['cv_median']:.2f}%")
    print()
    print(f"  Bootstrap 95% CI on std:")
    print(f"    [{std_ci['lower']:.2f}pp, {std_ci['upper']:.2f}pp]")
    print()
    print(f"  Bootstrap 95% CI on mean:")
    print(f"    [{mean_ci['lower']:.2f}pp, {mean_ci['upper']:.2f}pp]")
    print()

    p_below_3 = result["p_cv_below_3pct"]
    print(f"  P(true CV < 3%) = {p_below_3:.4f}  "
          f"({p_below_3*100:.2f}% of bootstrap samples)")
    print()

    if analytical:
        print(f"  Analytical chi-square 95% CI on std (normality assumption):")
        print(f"    [{analytical['std_lower_pp']:.2f}pp, "
              f"{analytical['std_upper_pp']:.2f}pp]")
        print(f"    => CV CI: [{analytical['cv_lower_pct']:.2f}%, "
              f"{analytical['cv_upper_pct']:.2f}%]")
        print()

    # ---- Decision ----
    print("=" * 65)
    print("DECISION")
    print("=" * 65)
    threshold = 3.0
    if cv_ci["lower"] > threshold:
        print(f"  CV 95% CI lower bound = {cv_ci['lower']:.2f}% > {threshold}%")
        print(f"  => True eval CV is confidently > {threshold}%")
        print(f"  => 4.91% is NOT small-sample noise")
        print(f"  => SKIP 50-eval expansion, proceed to training CV measurement")
        decision = "SKIP_50_EVAL"
    elif p_below_3 < 0.05:
        print(f"  P(CV < {threshold}%) = {p_below_3:.4f} < 0.05")
        print(f"  => Unlikely that true CV < {threshold}%")
        print(f"  => SKIP 50-eval expansion, proceed to training CV measurement")
        decision = "SKIP_50_EVAL"
    else:
        print(f"  CV 95% CI lower bound = {cv_ci['lower']:.2f}% <= {threshold}%")
        print(f"  P(CV < {threshold}%) = {p_below_3:.4f} >= 0.05")
        print(f"  => Cannot rule out true CV < {threshold}%")
        print(f"  => RUN 50-eval expansion to pin down CV")
        decision = "RUN_50_EVAL"

    result["analytical"] = analytical
    result["decision"] = decision
    result["threshold_cv_pct"] = threshold
    result["data"] = data

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
