#!/usr/bin/env python3
"""Phase 7 Round 1 A/B TOST analysis: EMA-only / Huber-only vs baseline v4.

Three complementary tests, all leveraging PAIRED structure (same 10 seeds):

  1. Paired mean TOST on place_rate
     - H0: |mean_diff| >= δ_mean   (NOT equivalent within δ_mean)
     - H1: |mean_diff| <  δ_mean   (equivalent within δ_mean)
     - δ_mean = 1.0pp (project standard)
     - Passes → cannot distinguish configs (mean-wise)
     - Fails + diff<0 → config is significantly WORSE
     - Fails + diff>0 → config is significantly BETTER

  2. Bootstrap TOST on CV (place_rate CV across seeds)
     - H0: |CV_diff| >= δ_cv        (CV difference at least δ_cv)
     - H1: |CV_diff| <  δ_cv        (CV equivalent within δ_cv)
     - δ_cv = 2.0pp
     - Bootstrap: resample 10 seeds with replacement B=10000 times,
       compute CV_baseline and CV_config per resample, take diff.
     - One-sided 95% CI from bootstrap distribution.
     - Passes → CVs are equivalent (no meaningful CV change)
     - Fails + CV_diff<0 → config significantly REDUCED CV
     - Fails + CV_diff>0 → config significantly INCREASED CV

  3. Pitman-Morgan paired variance test
     - Tests H0: var1 == var2 given paired samples (x_i, y_i)
     - t = (s1^2 - s2^2) * sqrt(n-2) / (2 * s1 * s2 * sqrt(1 - r^2))
     - df = n - 2
     - r = Pearson correlation between x and y
     - More powerful than F-test for paired data.

Decision synthesis:
  - CV_REDUCTION_CLAIMED  ⟺ test 2 fails with CV_diff<0  (significant reduction)
                           AND test 3 rejects var-equality in same direction
  - MEAN_CHANGE_CLAIMED   ⟺ test 1 fails (significant mean difference)
  - Otherwise: inconclusive (N too small, or effect too small)

Usage:
    python phase7_round1_tost_analysis.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

WORKSPACE = Path("/home/w/vla_workspace")
BASELINE_PATH = WORKSPACE / "outputs/phase7_variance_decomposition/training_cv_results.json"
EMA_PATH = WORKSPACE / "outputs/phase7_round1_ema_only/training_cv_results.json"
HUBER_PATH = WORKSPACE / "outputs/phase7_round1_huber_only/training_cv_results.json"

# Pre-registered equivalence margins (must be set BEFORE looking at data)
DELTA_MEAN_PP = 1.0     # place_rate mean equivalence: ±1pp
DELTA_CV_PP = 2.0       # CV equivalence: ±2pp CV
ALPHA = 0.05            # one-sided alpha per TOST arm (overall 0.05)
BOOTSTRAP_B = 10000
RNG_SEED = 42


def load_place_rates(path):
    """Load place rates keyed by seed (for paired matching).

    Uses the dict key (format: 'seed{int}') as the canonical seed source,
    since some legacy result files (e.g. baseline v4) have entries missing
    the inner 'seed' field.
    """
    data = json.load(open(path))
    out = {}
    for key, v in data["eval_runs"].items():
        # key format: 'seed3757552657' → 3757552657
        seed_int = int(key.replace("seed", ""))
        out[seed_int] = v["place_rate"] * 100
    return out


def load_qv_cv(path, key="v_mean"):
    """Load Q/V diagnostic values across seeds for CV computation."""
    data = json.load(open(path))
    vals = []
    for trun in data["training_runs"].values():
        qv = trun.get("qv_diagnostics", {})
        if key in qv:
            vals.append(qv[key])
    return vals


def cv(x):
    """Coefficient of variation in percent."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    s = x.std(ddof=1)
    return (s / m * 100) if m != 0 else float("inf")


def paired_tost_mean(x, y, delta):
    """Paired TOST on mean difference.

    H0: |mean_diff| >= delta  (not equivalent)
    H1: |mean_diff| <  delta  (equivalent)

    Returns (mean_diff, t1, t2, p1, p2, tost_passes, tcrit).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y  # paired differences
    n = len(d)
    mean_diff = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    df = n - 1
    tcrit = stats.t.ppf(1 - ALPHA, df)

    # TOST: two one-sided tests
    # H0_1: mean_diff <= -delta  → t1 = (mean_diff - (-delta)) / se
    # H0_2: mean_diff >= +delta  → t2 = (delta - mean_diff) / se
    t1 = (mean_diff - (-delta)) / se
    t2 = (delta - mean_diff) / se
    p1 = 1 - stats.t.cdf(t1, df)   # reject H0_1 if t1 > tcrit
    p2 = stats.t.cdf(t2, df)       # reject H0_2 if t2 < -tcrit

    tost_passes = (t1 > tcrit) and (t2 < -tcrit)
    return dict(mean_diff=mean_diff, se=se, t1=t1, t2=t2, p1=p1, p2=p2,
                tcrit=tcrit, df=df, tost_passes=tost_passes)


def bootstrap_tost_cv(x, y, delta, B=BOOTSTRAP_B, seed=RNG_SEED):
    """Bootstrap TOST on CV difference (paired resampling).

    H0: |CV_x - CV_y| >= delta  (CV not equivalent)
    H1: |CV_x - CV_y| <  delta  (CV equivalent within delta)

    Bootstrap procedure:
      - Resample indices [0, n) with replacement, B times
      - For each resample, compute CV_x and CV_y on the resampled pairs
      - Build bootstrap distribution of CV_diff = CV_x - CV_y
      - TOST: test CV_diff against -delta and +delta using bootstrap percentiles

    Returns observed CV_diff, bootstrap CI, and TOST verdict.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    obs_cv_x = cv(x)
    obs_cv_y = cv(y)
    obs_cv_diff = obs_cv_x - obs_cv_y

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = cv(x[idx]) - cv(y[idx])

    # TOST via bootstrap percentile method:
    # H0_1: CV_diff <= -delta  → reject if (1-alpha) lower CI of CV_diff > -delta
    # H0_2: CV_diff >= +delta  → reject if (1-alpha) upper CI of CV_diff < +delta
    # Equivalently: 90% CI (two 5% tails) must lie entirely within (-delta, +delta)
    lower = np.percentile(boot_diffs, 100 * ALPHA)
    upper = np.percentile(boot_diffs, 100 * (1 - ALPHA))

    tost_passes = (lower > -delta) and (upper < delta)

    # One-sided p-values (for reduction claim)
    p_reduction = np.mean(boot_diffs >= 0)   # H0: CV_diff >= 0, H1: CV_diff < 0
    p_increase = np.mean(boot_diffs <= 0)    # H0: CV_diff <= 0, H1: CV_diff > 0

    return dict(
        obs_cv_x=obs_cv_x, obs_cv_y=obs_cv_y, obs_cv_diff=obs_cv_diff,
        boot_mean=float(boot_diffs.mean()),
        boot_std=float(boot_diffs.std(ddof=1)),
        ci_lower=float(lower), ci_upper=float(upper),
        p_reduction=p_reduction, p_increase=p_increase,
        tost_passes=tost_passes,
        boot_diffs=boot_diffs,
    )


def pitman_morgan_test(x, y):
    """Pitman-Morgan paired variance test.

    H0: var(x) == var(y)
    H1: var(x) != var(y)

    t = (s_x^2 - s_y^2) * sqrt(n-2) / (2 * s_x * s_y * sqrt(1 - r^2))
    df = n - 2
    r = Pearson correlation between x and y

    Returns t-statistic, p-value (two-sided), and direction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    r, _ = stats.pearsonr(x, y)

    var_diff = sx**2 - sy**2
    denom = 2 * sx * sy * math.sqrt(1 - r**2)
    t = var_diff * math.sqrt(n - 2) / denom
    df = n - 2
    p = 2 * (1 - stats.t.cdf(abs(t), df))

    direction = "x>y" if var_diff > 0 else "x<y"
    return dict(t=t, p=p, df=df, r=r, sx=sx, sy=sy,
                var_diff=var_diff, direction=direction)


def analyze_config(name, baseline_pr, config_pr, baseline_qv, config_qv):
    """Run full TOST analysis for one config vs baseline."""
    print("=" * 72)
    print(f"  CONFIG: {name}  vs  baseline v4   (N={len(baseline_pr)} paired seeds)")
    print("=" * 72)

    # Align by seed
    common_seeds = sorted(set(baseline_pr.keys()) & set(config_pr.keys()))
    b = np.array([baseline_pr[s] for s in common_seeds])
    c = np.array([config_pr[s] for s in common_seeds])

    print(f"\n  Per-seed place_rates (pp):")
    print(f"    {'seed':>12s}  baseline  config  diff(config-baseline)")
    for s, bv, cv in zip(common_seeds, b, c):
        print(f"    {s:>12d}  {bv:7.1f}  {cv:6.1f}  {cv-bv:+6.1f}")

    print(f"\n  Summary statistics:")
    print(f"    baseline: mean={b.mean():.2f}pp  std={b.std(ddof=1):.2f}  CV={cv_fn(b):.2f}%")
    print(f"    config:   mean={c.mean():.2f}pp  std={c.std(ddof=1):.2f}  CV={cv_fn(c):.2f}%")
    print(f"    diff:     mean={c.mean()-b.mean():+.2f}pp  CV={cv_fn(c)-cv_fn(b):+.2f}pp")

    # ---- Test 1: Paired mean TOST ----
    print(f"\n  --- Test 1: Paired mean TOST (δ={DELTA_MEAN_PP}pp) ---")
    r1 = paired_tost_mean(c, b, DELTA_MEAN_PP)
    print(f"    mean_diff (config - baseline) = {r1['mean_diff']:+.3f}pp")
    print(f"    SE = {r1['se']:.3f}, df = {r1['df']}, tcrit = {r1['tcrit']:.3f}")
    print(f"    t1 (H0: diff<=-δ) = {r1['t1']:.3f}   reject? {r1['t1'] > r1['tcrit']}")
    print(f"    t2 (H0: diff>=+δ) = {r1['t2']:.3f}   reject? {r1['t2'] < -r1['tcrit']}")
    if r1["tost_passes"]:
        print(f"    => TOST PASSES: means equivalent within ±{DELTA_MEAN_PP}pp")
        mean_verdict = "EQUIVALENT"
    else:
        # TOST failed → means are NOT equivalent → check if significantly different
        # Use paired t-test for direction
        ttest = stats.ttest_rel(c, b)
        sig = abs(ttest.statistic) > r1["tcrit"]
        if sig:
            direction = "BETTER" if r1["mean_diff"] > 0 else "WORSE"
            print(f"    => TOST FAILS + paired t-test significant (t={ttest.statistic:.3f}, p={ttest.pvalue:.4f})")
            print(f"    => Config is significantly {direction} on mean place_rate")
            mean_verdict = direction
        else:
            print(f"    => TOST FAILS but paired t-test not significant (t={ttest.statistic:.3f}, p={ttest.pvalue:.4f})")
            print(f"    => Inconclusive (N too small OR effect in ±{DELTA_MEAN_PP}pp but noisy)")
            mean_verdict = "INCONCLUSIVE"

    # ---- Test 2: Bootstrap TOST on CV ----
    print(f"\n  --- Test 2: Bootstrap TOST on CV (δ={DELTA_CV_PP}pp, B={BOOTSTRAP_B}) ---")
    r2 = bootstrap_tost_cv(c, b, DELTA_CV_PP)
    print(f"    CV_config   = {r2['obs_cv_x']:.2f}%")
    print(f"    CV_baseline = {r2['obs_cv_y']:.2f}%")
    print(f"    CV_diff     = {r2['obs_cv_diff']:+.2f}pp")
    print(f"    Bootstrap: mean={r2['boot_mean']:+.2f}  std={r2['boot_std']:.2f}")
    print(f"    90% CI (one-sided 5% tails): [{r2['ci_lower']:+.2f}, {r2['ci_upper']:+.2f}]")
    print(f"    p(reduction) = {r2['p_reduction']:.4f}  (H0: CV_diff>=0, H1: CV_diff<0)")
    print(f"    p(increase)  = {r2['p_increase']:.4f}  (H0: CV_diff<=0, H1: CV_diff>0)")
    if r2["tost_passes"]:
        print(f"    => TOST PASSES: CVs equivalent within ±{DELTA_CV_PP}pp")
        cv_verdict = "EQUIVALENT"
    else:
        if r2["obs_cv_diff"] < 0 and r2["p_reduction"] < ALPHA:
            print(f"    => TOST FAILS + significant reduction (p={r2['p_reduction']:.4f} < {ALPHA})")
            print(f"    => Config significantly REDUCED CV")
            cv_verdict = "REDUCED"
        elif r2["obs_cv_diff"] > 0 and r2["p_increase"] < ALPHA:
            print(f"    => TOST FAILS + significant increase (p={r2['p_increase']:.4f} < {ALPHA})")
            print(f"    => Config significantly INCREASED CV")
            cv_verdict = "INCREASED"
        else:
            print(f"    => TOST FAILS but no significant direction")
            cv_verdict = "INCONCLUSIVE"

    # ---- Test 3: Pitman-Morgan paired variance test ----
    print(f"\n  --- Test 3: Pitman-Morgan paired variance test ---")
    r3 = pitman_morgan_test(c, b)
    print(f"    var(config)   = {r3['sx']**2:.3f}")
    print(f"    var(baseline) = {r3['sy']**2:.3f}")
    print(f"    r (correlation) = {r3['r']:.4f}")
    print(f"    t = {r3['t']:.3f}, df = {r3['df']}, p = {r3['p']:.4f}")
    print(f"    direction: var(config) {r3['direction']}")
    if r3["p"] < ALPHA:
        var_verdict = "VAR_REDUCED" if r3["direction"] == "x<y" else "VAR_INCREASED"
        print(f"    => Significant variance difference ({var_verdict})")
    else:
        var_verdict = "VAR_EQUIVALENT"
        print(f"    => No significant variance difference")

    # ---- Decision synthesis ----
    print(f"\n  === DECISION SYNTHESIS ===")
    print(f"    Mean:  {mean_verdict}")
    print(f"    CV:    {cv_verdict}")
    print(f"    Var:   {var_verdict}")

    if cv_verdict == "REDUCED" and var_verdict == "VAR_REDUCED":
        overall = "CV_REDUCTION_CLAIMED"
    elif cv_verdict == "INCREASED" and var_verdict == "VAR_INCREASED":
        overall = "CV_INCREASE_CLAIMED"
    elif cv_verdict == "EQUIVALENT":
        overall = "CV_NO_CHANGE"
    else:
        overall = "INCONCLUSIVE"
    print(f"    Overall: {overall}")

    return dict(
        config=name, n=len(common_seeds),
        mean_diff=r1["mean_diff"], mean_verdict=mean_verdict,
        cv_diff=r2["obs_cv_diff"], cv_verdict=cv_verdict,
        var_verdict=var_verdict, overall=overall,
        details=dict(tost_mean=r1, tost_cv={k: v for k, v in r2.items() if k != "boot_diffs"},
                     pitman_morgan=r3),
    )


def cv_fn(x):
    """Helper for inline CV display."""
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return (x.std(ddof=1) / m * 100) if m != 0 else 0


def main():
    print("=" * 72)
    print("  Phase 7 Round 1 A/B — TOST Analysis vs Baseline v4")
    print("=" * 72)
    print(f"  Pre-registered margins: δ_mean={DELTA_MEAN_PP}pp, δ_cv={DELTA_CV_PP}pp")
    print(f"  Alpha (per arm): {ALPHA}")
    print(f"  Bootstrap B: {BOOTSTRAP_B}, RNG seed: {RNG_SEED}")

    # Load data
    baseline_pr = load_place_rates(BASELINE_PATH)
    ema_pr = load_place_rates(EMA_PATH)
    huber_pr = load_place_rates(HUBER_PATH)

    # Q/V diagnostics (v_mean) for supplementary analysis
    baseline_qv_v = load_qv_cv(BASELINE_PATH, "v_mean")
    ema_qv_v = load_qv_cv(EMA_PATH, "v_mean")
    huber_qv_v = load_qv_cv(HUBER_PATH, "v_mean")

    results = {}
    results["ema_only"] = analyze_config("EMA-only", baseline_pr, ema_pr,
                                          baseline_qv_v, ema_qv_v)
    print()
    results["huber_only"] = analyze_config("Huber-only", baseline_pr, huber_pr,
                                            baseline_qv_v, huber_qv_v)

    # ---- Supplementary: Q/V CV comparison ----
    print("\n" + "=" * 72)
    print("  Supplementary: v_mean CV across seeds (value-function stability)")
    print("=" * 72)
    for name, qv_v in [("baseline", baseline_qv_v),
                        ("ema_only", ema_qv_v),
                        ("huber_only", huber_qv_v)]:
        arr = np.array(qv_v)
        print(f"    {name:12s}: v_mean mean={arr.mean():+.2f}  std={arr.std(ddof=1):.2f}  "
              f"CV={abs(arr.std(ddof=1)/arr.mean()*100):.1f}%")

    # ---- Save ----
    out_path = WORKSPACE / "outputs/phase7_round1_tost_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Analysis saved: {out_path}")

    # ---- Final summary table ----
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  {'config':12s}  {'mean_diff':>10s}  {'cv_diff':>8s}  {'mean_verdict':>14s}  {'cv_verdict':>14s}  {'overall':>22s}")
    for name, r in results.items():
        print(f"  {name:12s}  {r['mean_diff']:+9.2f}pp  {r['cv_diff']:+7.2f}pp  {r['mean_verdict']:>14s}  {r['cv_verdict']:>14s}  {r['overall']:>22s}")


if __name__ == "__main__":
    main()
