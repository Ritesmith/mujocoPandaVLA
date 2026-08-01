#!/usr/bin/env python3
"""Phase 7 Round 2a TOST analysis: tau0.5 / tau0.6 vs baseline v4 (tau=0.7).

Pre-registered decision rules (outputs/phase7_round2a_preregistration.json):
  R1 [WINNER-TAU]: A τ config WINS if (a) bootstrap TOST on CV shows significant
                   reduction (p_reduction<α AND 90% CI upper bound<0), AND
                   (b) paired t-test on mean_diff is not significant at α=0.05.
  R2 [FALLBACK]:   If τ=0.5 passes (a) but fails (b) with mean_loss>2pp,
                   fall back to τ=0.6.
  R3 [PIVOT]:      If NEITHER passes (a), pivot to V L2 regularization.
  R4 [TIE-BREAK]:  If BOTH pass (a)+(b), pick lower τ.
  R5 [MEAN-PROT]:  If mean_loss>2pp (paired t-test sig AND mean_diff<-2pp),
                   REJECT that τ regardless of CV improvement.

Pre-registered margins (RELAXED from Round 1):
  δ_mean = 2.0pp (was 1.0pp in Round 1; lower τ is EXPECTED to cost some mean)
  δ_cv   = 2.0pp
  α      = 0.05 (one-sided per TOST arm)

Usage:
    python phase7_round2a_tost_analysis.py
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

WORKSPACE = Path("/home/w/vla_workspace")
BASELINE_PATH = WORKSPACE / "outputs/phase7_variance_decomposition/training_cv_results.json"
TAU05_PATH = WORKSPACE / "outputs/phase7_round2a_tau0.5/training_cv_results.json"
TAU06_PATH = WORKSPACE / "outputs/phase7_round2a_tau0.6/training_cv_results.json"

# Pre-registered margins (relaxed from Round 1)
DELTA_MEAN_PP = 2.0     # was 1.0 in Round 1
DELTA_CV_PP = 2.0
ALPHA = 0.05
BOOTSTRAP_B = 10000
RNG_SEED = 42


def load_place_rates(path):
    """Load place rates keyed by seed (for paired matching)."""
    data = json.load(open(path))
    out = {}
    for key, v in data["eval_runs"].items():
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


def cv_fn(x):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    return (x.std(ddof=1) / m * 100) if m != 0 else 0


def paired_tost_mean(x, y, delta):
    """Paired TOST on mean difference.

    H0: |mean_diff| >= delta  (not equivalent)
    H1: |mean_diff| <  delta  (equivalent)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    n = len(d)
    mean_diff = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    df = n - 1
    tcrit = stats.t.ppf(1 - ALPHA, df)

    t1 = (mean_diff - (-delta)) / se
    t2 = (delta - mean_diff) / se
    p1 = 1 - stats.t.cdf(t1, df)
    p2 = stats.t.cdf(t2, df)

    tost_passes = (t1 > tcrit) and (t2 < -tcrit)

    # Paired t-test for direction
    ttest = stats.ttest_rel(x, y)

    return dict(
        mean_diff=float(mean_diff), se=float(se), t1=float(t1), t2=float(t2),
        p1=float(p1), p2=float(p2), tcrit=float(tcrit), df=df,
        tost_passes=bool(tost_passes),
        paired_t_stat=float(ttest.statistic),
        paired_t_p=float(ttest.pvalue),
    )


def bootstrap_tost_cv(x, y, delta, B=BOOTSTRAP_B, seed=RNG_SEED):
    """Bootstrap TOST on CV difference (paired resampling)."""
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

    lower = float(np.percentile(boot_diffs, 100 * ALPHA))
    upper = float(np.percentile(boot_diffs, 100 * (1 - ALPHA)))

    tost_passes = (lower > -delta) and (upper < delta)

    p_reduction = float(np.mean(boot_diffs >= 0))
    p_increase = float(np.mean(boot_diffs <= 0))

    return dict(
        obs_cv_x=float(obs_cv_x), obs_cv_y=float(obs_cv_y),
        obs_cv_diff=float(obs_cv_diff),
        boot_mean=float(boot_diffs.mean()),
        boot_std=float(boot_diffs.std(ddof=1)),
        ci_lower=lower, ci_upper=upper,
        p_reduction=p_reduction, p_increase=p_increase,
        tost_passes=bool(tost_passes),
    )


def pitman_morgan_test(x, y):
    """Pitman-Morgan paired variance test."""
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
    return dict(t=float(t), p=float(p), df=df, r=float(r),
                sx=float(sx), sy=float(sy), var_diff=float(var_diff),
                direction=direction)


def analyze_config(name, baseline_pr, config_pr):
    """Run full TOST analysis for one config vs baseline. Returns results dict."""
    print("=" * 72)
    print(f"  CONFIG: {name}  vs  baseline v4 (τ=0.7)   (N={len(baseline_pr)} paired seeds)")
    print("=" * 72)

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
    print(f"    Paired t-test: t={r1['paired_t_stat']:.3f}, p={r1['paired_t_p']:.4f}")

    # R1(b): paired t-test NOT significant at α
    r1_b_passes = r1["paired_t_p"] >= ALPHA  # not significant → mean preserved
    # R5: mean_loss>2pp AND paired t-test significant
    r5_triggers = (r1["mean_diff"] < -DELTA_MEAN_PP) and (r1["paired_t_p"] < ALPHA)

    if r1["tost_passes"]:
        print(f"    => TOST PASSES: means equivalent within ±{DELTA_MEAN_PP}pp")
        mean_verdict = "EQUIVALENT"
    elif r1["paired_t_p"] < ALPHA:
        direction = "BETTER" if r1["mean_diff"] > 0 else "WORSE"
        print(f"    => TOST FAILS + paired t-test significant → {direction}")
        mean_verdict = direction
    else:
        print(f"    => TOST FAILS but paired t-test not significant → inconclusive")
        mean_verdict = "INCONCLUSIVE"

    print(f"    R1(b) [mean preserved]: {r1_b_passes} (paired t p={r1['paired_t_p']:.4f} >= α={ALPHA})")
    print(f"    R5 [mean_loss>2pp + sig]: {r5_triggers}")

    # ---- Test 2: Bootstrap TOST on CV ----
    print(f"\n  --- Test 2: Bootstrap TOST on CV (δ={DELTA_CV_PP}pp, B={BOOTSTRAP_B}) ---")
    r2 = bootstrap_tost_cv(c, b, DELTA_CV_PP)
    print(f"    CV_config   = {r2['obs_cv_x']:.2f}%")
    print(f"    CV_baseline = {r2['obs_cv_y']:.2f}%")
    print(f"    CV_diff     = {r2['obs_cv_diff']:+.2f}pp")
    print(f"    Bootstrap: mean={r2['boot_mean']:+.2f}  std={r2['boot_std']:.2f}")
    print(f"    90% CI: [{r2['ci_lower']:+.2f}, {r2['ci_upper']:+.2f}]")
    print(f"    p(reduction) = {r2['p_reduction']:.4f}  (H0: CV_diff>=0)")
    print(f"    p(increase)  = {r2['p_increase']:.4f}  (H0: CV_diff<=0)")

    # R1(a): significant CV reduction = (p_reduction < α) AND (90% CI upper < 0)
    r1_a_passes = (r2["p_reduction"] < ALPHA) and (r2["ci_upper"] < 0)

    if r2["tost_passes"]:
        cv_verdict = "EQUIVALENT"
        print(f"    => TOST PASSES: CVs equivalent within ±{DELTA_CV_PP}pp")
    elif r2["obs_cv_diff"] < 0 and r2["p_reduction"] < ALPHA:
        cv_verdict = "REDUCED"
        print(f"    => TOST FAILS + significant reduction (p={r2['p_reduction']:.4f})")
    elif r2["obs_cv_diff"] > 0 and r2["p_increase"] < ALPHA:
        cv_verdict = "INCREASED"
        print(f"    => TOST FAILS + significant increase")
    else:
        cv_verdict = "INCONCLUSIVE"
        print(f"    => TOST FAILS, no significant direction")

    print(f"    R1(a) [CV sig reduced]: {r1_a_passes} "
          f"(p_red={r2['p_reduction']:.4f}<{ALPHA} AND CI_upper={r2['ci_upper']:+.2f}<0)")

    # ---- Test 3: Pitman-Morgan ----
    print(f"\n  --- Test 3: Pitman-Morgan paired variance test ---")
    r3 = pitman_morgan_test(c, b)
    print(f"    r (correlation) = {r3['r']:.4f}")
    print(f"    t = {r3['t']:.3f}, df = {r3['df']}, p = {r3['p']:.4f}")
    print(f"    direction: var(config) {r3['direction']}")
    var_verdict = "VAR_REDUCED" if (r3["p"] < ALPHA and r3["direction"] == "x<y") else \
                  ("VAR_INCREASED" if (r3["p"] < ALPHA and r3["direction"] == "x>y") else "VAR_EQUIVALENT")
    print(f"    => {var_verdict}")

    # ---- Decision synthesis ----
    print(f"\n  === DECISION SYNTHESIS ===")
    print(f"    R1(a) CV sig reduced:    {r1_a_passes}")
    print(f"    R1(b) mean preserved:    {r1_b_passes}")
    print(f"    R1 WINNER passes (a∧b):  {r1_a_passes and r1_b_passes}")
    print(f"    R5 REJECT (mean_loss>2): {r5_triggers}")

    if r5_triggers:
        overall = "REJECTED_BY_R5"
    elif r1_a_passes and r1_b_passes:
        overall = "R1_WINNER"
    elif r1_a_passes and not r1_b_passes:
        overall = "CV_REDUCED_BUT_MEAN_LOSS"  # triggers R2 fallback if τ=0.5
    else:
        overall = "CV_NOT_REDUCED"  # triggers R3 pivot if both fail

    print(f"    Overall: {overall}")

    return dict(
        config=name, n=len(common_seeds),
        mean_diff=r1["mean_diff"], mean_verdict=mean_verdict,
        cv_diff=r2["obs_cv_diff"], cv_verdict=cv_verdict,
        var_verdict=var_verdict, overall=overall,
        r1_a_passes=r1_a_passes, r1_b_passes=r1_b_passes, r5_triggers=r5_triggers,
        details=dict(tost_mean=r1, tost_cv=r2, pitman_morgan=r3),
    )


def main():
    print("=" * 72)
    print("  Phase 7 Round 2a — TOST Analysis: τ scan vs Baseline v4 (τ=0.7)")
    print("=" * 72)
    print(f"  Pre-registered margins: δ_mean={DELTA_MEAN_PP}pp (relaxed from 1.0), δ_cv={DELTA_CV_PP}pp")
    print(f"  Alpha (per arm): {ALPHA}")
    print(f"  Bootstrap B: {BOOTSTRAP_B}, RNG seed: {RNG_SEED}")
    print(f"  Preregistration: outputs/phase7_round2a_preregistration.json")

    baseline_pr = load_place_rates(BASELINE_PATH)
    tau05_pr = load_place_rates(TAU05_PATH)
    tau06_pr = load_place_rates(TAU06_PATH)

    baseline_qv_v = load_qv_cv(BASELINE_PATH, "v_mean")
    tau05_qv_v = load_qv_cv(TAU05_PATH, "v_mean")
    tau06_qv_v = load_qv_cv(TAU06_PATH, "v_mean")

    results = {}
    results["tau0.5"] = analyze_config("τ=0.5", baseline_pr, tau05_pr)
    print()
    results["tau0.6"] = analyze_config("τ=0.6", baseline_pr, tau06_pr)

    # ---- Apply R1-R5 decision rules ----
    print("\n" + "=" * 72)
    print("  APPLYING PRE-REGISTERED DECISION RULES R1-R5")
    print("=" * 72)

    t5 = results["tau0.5"]
    t6 = results["tau0.6"]

    verdict = None
    reason = None

    # R5: REJECT any τ with mean_loss > 2pp (sig)
    t5_rejected = t5["r5_triggers"]
    t6_rejected = t6["r5_triggers"]

    print(f"\n  R5 [MEAN-PROTECTION]:")
    print(f"    τ=0.5: mean_diff={t5['mean_diff']:+.2f}pp, R5 reject = {t5_rejected}")
    print(f"    τ=0.6: mean_diff={t6['mean_diff']:+.2f}pp, R5 reject = {t6_rejected}")

    # R1: WINNER = (a) CV sig reduced AND (b) mean preserved
    t5_r1 = t5["r1_a_passes"] and t5["r1_b_passes"]
    t6_r1 = t6["r1_a_passes"] and t6["r1_b_passes"]

    print(f"\n  R1 [WINNER-TAU]: (a) CV sig reduced AND (b) mean preserved")
    print(f"    τ=0.5: R1(a)={t5['r1_a_passes']}, R1(b)={t5['r1_b_passes']} → R1 passes = {t5_r1}")
    print(f"    τ=0.6: R1(a)={t6['r1_a_passes']}, R1(b)={t6['r1_b_passes']} → R1 passes = {t6_r1}")

    # R4: TIE-BREAK — if both pass R1, pick lower τ
    if t5_r1 and t6_r1:
        verdict = "tau0.5_WINNER"
        reason = "R4 TIE-BREAK: both τ=0.5 and τ=0.6 pass R1, pick lower τ (more aggressive V smoothing)"
    elif t5_r1 and not t6_r1:
        verdict = "tau0.5_WINNER"
        reason = "R1: τ=0.5 passes (CV reduced + mean preserved), τ=0.6 fails R1"
    elif t6_r1 and not t5_r1:
        verdict = "tau0.6_WINNER"
        reason = "R1: τ=0.6 passes, τ=0.5 fails R1"
    elif t5["r1_a_passes"] and not t5["r1_b_passes"] and not t5_rejected:
        # τ=0.5 reduced CV but mean loss within 2pp but significant — check R2
        if t6_r1:
            verdict = "tau0.6_WINNER"
            reason = "R2 FALLBACK: τ=0.5 reduced CV but mean loss significant (within 2pp), τ=0.6 passes R1"
        else:
            verdict = "INCONCLUSIVE"
            reason = "R2 FALLBACK attempted but τ=0.6 also fails R1"
    elif not t5["r1_a_passes"] and not t6["r1_a_passes"]:
        verdict = "PIVOT_TO_V_L2"
        reason = "R3 PIVOT: NEITHER τ=0.5 NOR τ=0.6 significantly reduced CV → pivot to V L2 regularization"
    else:
        verdict = "INCONCLUSIVE"
        reason = "No pre-registered rule covers this combination"

    print(f"\n  >>> VERDICT: {verdict}")
    print(f"  >>> Reason:  {reason}")

    # ---- Supplementary: v_mean CV ----
    print("\n" + "=" * 72)
    print("  Supplementary: v_mean CV across seeds (value-function stability)")
    print("=" * 72)
    for name, qv_v in [("baseline τ=0.7", baseline_qv_v),
                        ("τ=0.5", tau05_qv_v),
                        ("τ=0.6", tau06_qv_v)]:
        arr = np.array(qv_v)
        if len(arr) > 0:
            print(f"    {name:18s}: v_mean mean={arr.mean():+.2f}  std={arr.std(ddof=1):.2f}  "
                  f"CV={abs(arr.std(ddof=1)/arr.mean()*100):.1f}%")

    # ---- Catastrophic seed highlight ----
    print("\n" + "=" * 72)
    print("  Catastrophic seed analysis (seed 2976135721)")
    print("=" * 72)
    cat_seed = 2976135721
    if cat_seed in baseline_pr and cat_seed in tau05_pr and cat_seed in tau06_pr:
        b_v = baseline_pr[cat_seed]
        t5_v = tau05_pr[cat_seed]
        t6_v = tau06_pr[cat_seed]
        print(f"    baseline τ=0.7: {b_v:.1f}%")
        print(f"    τ=0.5:          {t5_v:.1f}%  (diff {t5_v-b_v:+.1f}pp)")
        print(f"    τ=0.6:          {t6_v:.1f}%  (diff {t6_v-b_v:+.1f}pp)")
        print(f"    → τ=0.5 救回 {t5_v-b_v:+.1f}pp, τ=0.6 救回 {t6_v-b_v:+.1f}pp")

    # ---- Save ----
    output = {
        "analysis": "Phase 7 Round 2a TOST",
        "preregistration": "outputs/phase7_round2a_preregistration.json",
        "margins": {"delta_mean_pp": DELTA_MEAN_PP, "delta_cv_pp": DELTA_CV_PP,
                    "alpha": ALPHA, "bootstrap_B": BOOTSTRAP_B},
        "results": results,
        "verdict": verdict,
        "reason": reason,
    }
    out_path = WORKSPACE / "outputs/phase7_round2a_tost_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Analysis saved: {out_path}")

    # ---- Final summary ----
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  {'config':12s}  {'mean_diff':>10s}  {'cv_diff':>8s}  {'R1(a)':>6s}  {'R1(b)':>6s}  {'R5':>6s}  {'overall':>22s}")
    for name, r in results.items():
        print(f"  {name:12s}  {r['mean_diff']:+9.2f}pp  {r['cv_diff']:+7.2f}pp  "
              f"{str(r['r1_a_passes']):>6s}  {str(r['r1_b_passes']):>6s}  "
              f"{str(r['r5_triggers']):>6s}  {r['overall']:>22s}")
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
