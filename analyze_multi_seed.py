#!/usr/bin/env python3
"""Statistical analysis & visualization for multi-seed evaluation results.

Reads outputs/dt_orchestrator/multi_seed_results.json (produced by
multi_seed_eval.py) and generates:
  1. Text summary: mean±std, median±IQR, CV, 95% CI, paired t-test
  2. Bootstrap 95% CI (per user suggestion, no scipy dependency)
  3. Box plot + per-seed line plot (matplotlib)

Usage:
    python analyze_multi_seed.py
    python analyze_multi_seed.py --results <path> --output_dir <dir>
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

WORKSPACE = Path(__file__).parent
DEFAULT_RESULTS = WORKSPACE / "outputs" / "dt_orchestrator" / "multi_seed_results.json"
DEFAULT_OUT_DIR = WORKSPACE / "outputs" / "dt_orchestrator"


def quantile(sorted_data, q):
    """Compute quantile (q in [0,1]) via linear interpolation."""
    if not sorted_data:
        return float("nan")
    if len(sorted_data) == 1:
        return sorted_data[0]
    pos = q * (len(sorted_data) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_data[lo]
    frac = pos - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def bootstrap_ci(data, n_boot=10000, ci=0.95, stat="mean", seed=42):
    """Bootstrap confidence interval (no scipy dependency).

    Args:
        data: list of floats
        n_boot: number of bootstrap resamples
        ci: confidence level (0.95 → 95% CI)
        stat: "mean" or "median"
        seed: RNG seed for reproducibility
    Returns:
        (lower, upper, point_estimate)
    """
    if len(data) < 2:
        return (float("nan"), float("nan"), data[0] if data else float("nan"))
    rng = random.Random(seed)
    n = len(data)
    boot_stats = []
    for _ in range(n_boot):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        if stat == "mean":
            boot_stats.append(sum(sample) / n)
        else:  # median
            boot_stats.append(median(sample))
    boot_stats.sort()
    alpha = (1 - ci) / 2
    lo = quantile(boot_stats, alpha)
    hi = quantile(boot_stats, 1 - alpha)
    pt = mean(data) if stat == "mean" else median(data)
    return (lo, hi, pt)


def paired_t_test(diffs):
    """Manual paired t-test (no scipy). Returns (t_stat, df, p_approx).

    p_approx uses a crude two-sided approximation via t-distribution table
    lookup for common df values; for other df, reports whether |t| exceeds
    critical values at p=0.10, 0.05, 0.01.
    """
    n = len(diffs)
    if n < 2:
        return (float("nan"), 0, "n/a")
    m = mean(diffs)
    s = stdev(diffs)
    if s == 0:
        return (float("inf") if m != 0 else 0, n - 1, "perfect")
    t_stat = m / (s / math.sqrt(n))
    df = n - 1
    # Critical two-sided t-values at p=0.05 for df 1..30
    t_crit_05 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    }
    t_crit_01 = {
        1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032,
        6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
        15: 2.947, 20: 2.845, 25: 2.787, 30: 2.750,
    }
    def lookup(table, df):
        if df in table:
            return table[df]
        keys = sorted(table.keys())
        if df < keys[0]:
            return table[keys[0]]
        if df > keys[-1]:
            return table[keys[-1]]
        return table[min(keys, key=lambda k: abs(k - df))]
    tc05 = lookup(t_crit_05, df)
    tc01 = lookup(t_crit_01, df)
    abs_t = abs(t_stat)
    if abs_t > tc01:
        sig = f"p < 0.01 (significant, |t|={abs_t:.2f} > {tc01})"
    elif abs_t > tc05:
        sig = f"p < 0.05 (significant, |t|={abs_t:.2f} > {tc05})"
    else:
        sig = f"p >= 0.05 (NOT significant, |t|={abs_t:.2f} <= {tc05})"
    return (t_stat, df, sig)


# One-sided t critical values at alpha=0.05 (for TOST equivalence test)
_T_CRIT_ONE_SIDED_05 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
    6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
    15: 1.753, 20: 1.725, 25: 1.708, 30: 1.697,
}


def tost_equivalence(diffs, delta, alpha=0.05):
    """Two One-Sided Tests (TOST) for equivalence.

    Tests whether the mean of paired differences is within (-delta, +delta).

    H0: |mean(diff)| >= delta  (NOT equivalent)
    H1: |mean(diff)| <  delta  (equivalent)

    Reject H0 (declare equivalence) only if BOTH one-sided tests are
    significant. This closes the gap between "failed to reject H0"
    (current paired t-test) and "demonstrated equivalence".

    Args:
        diffs: list of paired differences (b - a)
        delta: equivalence boundary in same units as diffs
        alpha: per-test significance level (default 0.05)
    Returns:
        dict: {equivalent, t1, t2, tcrit, mean_diff, delta, n, interpretation}
    """
    n = len(diffs)
    if n < 2:
        return {"equivalent": False, "error": "n < 2", "n": n}
    m = mean(diffs)
    s = stdev(diffs)
    df = n - 1
    # Lookup one-sided t critical
    keys = sorted(_T_CRIT_ONE_SIDED_05.keys())
    if df in _T_CRIT_ONE_SIDED_05:
        tcrit = _T_CRIT_ONE_SIDED_05[df]
    elif df < keys[0]:
        tcrit = _T_CRIT_ONE_SIDED_05[keys[0]]
    elif df > keys[-1]:
        tcrit = _T_CRIT_ONE_SIDED_05[keys[-1]]
    else:
        tcrit = _T_CRIT_ONE_SIDED_05[min(keys, key=lambda k: abs(k - df))]

    if s == 0:
        eq = abs(m) < delta
        return {
            "equivalent": eq, "t1": float("inf"), "t2": float("-inf"),
            "tcrit": tcrit, "mean_diff": m, "delta": delta, "n": n, "df": df,
            "interpretation": (f"零方差, |mean diff|={abs(m):.4f} "
                               f"{'<' if eq else '>='} delta={delta} → "
                               f"{'等价' if eq else '不等价'}"),
        }
    se = s / math.sqrt(n)
    # H01: mean <= -delta  →  reject if t1 > tcrit
    t1 = (m - (-delta)) / se  # = (m + delta) / se
    # H02: mean >= +delta  →  reject if t2 < -tcrit
    t2 = (m - delta) / se
    reject1 = t1 > tcrit
    reject2 = t2 < -tcrit
    equivalent = reject1 and reject2

    if equivalent:
        interp = (f"TOST 等价 (delta={delta}): |mean diff|={abs(m):.4f} < {delta}, "
                  f"t1={t1:.2f}>{tcrit}, t2={t2:.2f}<-{tcrit} → "
                  f"可声明统计等价 (±{delta} 内)")
    else:
        interp = (f"TOST 不等价 (delta={delta}): |mean diff|={abs(m):.4f}, "
                  f"t1={t1:.2f}{'>' if reject1 else '<='}{tcrit} "
                  f"({'reject' if reject1 else 'fail'}), "
                  f"t2={t2:.2f}{'<' if reject2 else '>='}-{tcrit} "
                  f"({'reject' if reject2 else 'fail'}) → "
                  f"无法声明等价 (真实差异可能达 ±{delta})")
    return {
        "equivalent": equivalent, "t1": t1, "t2": t2, "tcrit": tcrit,
        "mean_diff": m, "delta": delta, "n": n, "df": df,
        "interpretation": interp,
    }


def analyze(results_path, out_dir):
    """Main analysis routine."""
    with open(results_path) as f:
        results = json.load(f)

    seeds = results.get("seeds", [])
    configs = results.get("configs", [])
    runs = results.get("runs", {})

    print("=" * 70)
    print("Multi-Seed Statistical Analysis")
    print("=" * 70)
    print(f"  Results:   {results_path}")
    print(f"  Seeds:     {seeds}")
    print(f"  Configs:   {configs}")
    print(f"  N_eps:     {results.get('n_episodes', '?')}")

    # Collect per-config metrics
    config_data = defaultdict(lambda: {
        "place_rates": [], "drifts": [], "near_miss": [],
        "seeds_done": [], "final_dists": []
    })
    for config in configs:
        for seed in seeds:
            key = f"{config}_seed{seed}"
            if key not in runs:
                continue
            m = runs[key]["metrics"]
            config_data[config]["place_rates"].append(m["place_rate"])
            config_data[config]["drifts"].append(m["n_drift"])
            config_data[config]["near_miss"].append(m["n_near_miss"])
            config_data[config]["seeds_done"].append(seed)
            if m.get("mean_final_dist_cm") is not None:
                config_data[config]["final_dists"].append(m["mean_final_dist_cm"])

    # Per-config summary
    print("\n" + "-" * 70)
    print("PER-CONFIG SUMMARY")
    print("-" * 70)
    summary = {}
    for config in configs:
        d = config_data[config]
        prs = d["place_rates"]
        if not prs:
            print(f"\n  {config}: NO DATA")
            continue
        n = len(prs)
        pr_mean = mean(prs)
        pr_std = stdev(prs) if n > 1 else 0.0
        pr_median = median(prs)
        pr_cv = (pr_std / pr_mean * 100) if pr_mean > 0 else 0
        # 95% CI via bootstrap (mean)
        ci_lo, ci_hi, _ = bootstrap_ci(prs, stat="mean")
        # IQR
        prs_sorted = sorted(prs)
        q1 = quantile(prs_sorted, 0.25)
        q3 = quantile(prs_sorted, 0.75)
        iqr = q3 - q1

        dr_mean = mean(d["drifts"])
        dr_std = stdev(d["drifts"]) if n > 1 else 0.0

        print(f"\n  {config} ({n}/{len(seeds)} seeds completed):")
        print(f"    place_rate:")
        print(f"      mean   ± std:   {pr_mean*100:.1f}% ± {pr_std*100:.2f}pp  (CV={pr_cv:.2f}%)")
        print(f"      median ± IQR:   {pr_median*100:.1f}% ± {iqr*100:.2f}pp  (Q1={q1*100:.1f}, Q3={q3*100:.1f})")
        print(f"      95% CI (bootstrap, mean): [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
        print(f"      per-seed (pp): {[f'{p*100:.1f}' for p in prs]}")
        print(f"    drift:     {dr_mean:.1f} ± {dr_std:.1f}")
        print(f"    near_miss: {mean(d['near_miss']):.1f} ± "
              f"{stdev(d['near_miss']) if n>1 else 0:.1f}")
        if d["final_dists"]:
            print(f"    final_dist_cm: {mean(d['final_dists']):.2f} ± "
                  f"{stdev(d['final_dists']) if n>1 else 0:.2f}")

        summary[config] = {
            "n": n, "mean": pr_mean, "std": pr_std, "median": pr_median,
            "cv": pr_cv, "ci_lo": ci_lo, "ci_hi": ci_hi, "iqr": iqr,
            "q1": q1, "q3": q3, "per_seed": prs,
        }

    # Paired comparison
    common_seeds = [s for s in seeds
                    if f"v4_seed{s}" in runs and f"dt_router_v2_seed{s}" in runs]
    if len(common_seeds) >= 2:
        print("\n" + "-" * 70)
        print(f"PAIRED COMPARISON ({len(common_seeds)} common seeds)")
        print("-" * 70)
        v4_prs = [runs[f"v4_seed{s}"]["metrics"]["place_rate"] for s in common_seeds]
        dt_prs = [runs[f"dt_router_v2_seed{s}"]["metrics"]["place_rate"] for s in common_seeds]
        diffs = [d - v for d, v in zip(dt_prs, v4_prs)]

        diff_mean = mean(diffs)
        diff_std = stdev(diffs) if len(diffs) > 1 else 0
        diff_median = median(diffs)
        ci_lo, ci_hi, _ = bootstrap_ci(diffs, stat="mean")
        t_stat, df, sig = paired_t_test(diffs)

        print(f"  DT Router v2 - v4 (per seed, pp): "
              f"{[f'{d*100:+.1f}' for d in diffs]}")
        print(f"  Mean diff:   {diff_mean*100:+.2f}pp ± {diff_std*100:.2f}pp")
        print(f"  Median diff: {diff_median*100:+.2f}pp")
        print(f"  95% CI (bootstrap): [{ci_lo*100:+.2f}pp, {ci_hi*100:+.2f}pp]")
        print(f"  Paired t-test: t={t_stat:.3f}, df={df}")
        print(f"  Significance: {sig}")

        # Effect size (Cohen's d for paired)
        if diff_std > 0:
            cohen_d = diff_mean / diff_std
            print(f"  Cohen's d (paired): {cohen_d:.2f} "
                  f"({'small' if abs(cohen_d)<0.5 else 'medium' if abs(cohen_d)<0.8 else 'large'})")

        # TOST equivalence test (closes "statistical equivalence" claim)
        # delta = 0.01 (1pp) — the expected DT Router gain magnitude
        tost_delta = 0.01
        tost_result = tost_equivalence(diffs, tost_delta)
        print(f"\n  TOST 等价检验 (delta=±{tost_delta*100:.1f}pp):")
        print(f"    {tost_result['interpretation']}")
        if not tost_result.get("equivalent"):
            print(f"    → 当前 N={len(diffs)} 无法声明统计等价")
            print(f"    → 'DT Router 无增益' 应表述为 '在 ±{tost_delta*100:.1f}pp 内无法分辨'")

        # CI overlap check
        v4_ci = summary.get("v4", {})
        dt_ci = summary.get("dt_router_v2", {})
        if v4_ci and dt_ci:
            overlap = not (v4_ci["ci_hi"] < dt_ci["ci_lo"] or
                          dt_ci["ci_hi"] < v4_ci["ci_lo"])
            print(f"\n  CI overlap check:")
            print(f"    v4  95% CI:           [{v4_ci['ci_lo']*100:.1f}%, {v4_ci['ci_hi']*100:.1f}%]")
            print(f"    dt2 95% CI:           [{dt_ci['ci_lo']*100:.1f}%, {dt_ci['ci_hi']*100:.1f}%]")
            print(f"    Overlap: {'YES (cannot reject equivalence)' if overlap else 'NO (significant difference)'}")

        # Decision gate
        print("\n" + "-" * 70)
        print("DECISION GATE")
        print("-" * 70)
        v4_cv = summary.get("v4", {}).get("cv", 999)
        dt_cv = summary.get("dt_router_v2", {}).get("cv", 999)
        max_cv = max(v4_cv, dt_cv)
        v4_mean = summary.get("v4", {}).get("mean", 0)

        # CV threshold provenance: 3% at ~68% place_rate ≈ 2.04pp std,
        # which is the smallest effect we'd want to reliably detect
        # (DT Router expected gain +1~2pp). CV threshold = "effect < noise 1σ".
        cv_threshold = 3.0
        cv_implied_std_pp = v4_mean * cv_threshold / 100 * 100  # mean*cv% in pp
        print(f"  CV 阈值溯源: {cv_threshold}% @ {v4_mean*100:.1f}% place_rate "
              f"→ 隐含 std ≈ {cv_implied_std_pp:.2f}pp "
              f"(= 期望检测的最小效应 {tost_delta*100:.1f}~2pp 的可检测性边界)")
        if max_cv > cv_threshold:
            print(f"  ⚠ CV={max_cv:.2f}% > {cv_threshold}% → 优先优化训练稳定性")
        if not tost_result.get("equivalent"):
            print(f"  ⚠ TOST 未通过 (delta=±{tost_delta*100:.1f}pp) → 无法声明统计等价")
            print(f"    → 'DT Router 无增益' 应表述为 '±{tost_delta*100:.1f}pp 内无法分辨'")
            print(f"    → 需 ≥30 seeds + 降低 CV 后用 TOST 复核再正式归档")
        elif "p < 0.05" in sig or "p < 0.01" in sig:
            if diff_mean > 0:
                print(f"  → v2 显著优于 v4 (+{diff_mean*100:.2f}pp, p<0.05) → 启动 v4.2 Critic 重训")
            else:
                print(f"  → v4 显著优于 v2 ({diff_mean*100:+.2f}pp, p<0.05) → 接受 v4 进 P3 sim-to-real")
        else:
            print(f"  → v2 vs v4 差异不显著 (p>=0.05) 且 TOST 等价 (±{tost_delta*100:.1f}pp)")
            print(f"    → 可声明统计等价，DT Router 路由无真实增益")
            v4_ci_lo = summary.get("v4", {}).get("ci_lo", 0)
            if v4_ci_lo >= 0.68:
                print(f"    → v4 CI 下界 {v4_ci_lo*100:.1f}% ≥ 68%: 接受 v4 进 P3 (已超越 V59 56%)")
            else:
                print(f"    → v4 CI 下界 {v4_ci_lo*100:.1f}% < 68%: 需更多 seed 或稳定性优化")

    # Visualization
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Box plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        data_box = [config_data[c]["place_rates"] for c in configs if config_data[c]["place_rates"]]
        labels = [c for c in configs if config_data[c]["place_rates"]]
        if data_box:
            axes[0].boxplot(data_box, labels=labels, showmeans=True)
            axes[0].set_ylabel("Place Rate")
            axes[0].set_title("Place Rate Distribution (box=IQR, ◆=mean)")
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
            for i, d in enumerate(data_box, 1):
                x = [i] * len(d)
                axes[0].scatter(x, d, alpha=0.6, zorder=3)

        # Per-seed line plot (paired)
        if len(common_seeds) >= 2:
            x = list(range(len(common_seeds)))
            axes[1].plot(x, [p*100 for p in v4_prs], "o-", label="v4", color="steelblue")
            axes[1].plot(x, [p*100 for p in dt_prs], "s-", label="dt_router_v2", color="coral")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([str(s) for s in common_seeds], rotation=45)
            axes[1].set_xlabel("Seed")
            axes[1].set_ylabel("Place Rate (%)")
            axes[1].set_title("Paired Per-Seed Comparison")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = out_dir / "multi_seed_analysis.png"
        plt.savefig(plot_path, dpi=120)
        print(f"\n  Plot saved: {plot_path}")
        plt.close()
    except ImportError:
        print("\n  (matplotlib not available, skipping plots)")

    # Save analysis summary
    analysis_path = out_dir / "multi_seed_analysis.json"
    analysis = {
        "summary": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                     for k, v in summary.items()},
        "per_seed": {k: v["per_seed"] for k, v in summary.items()},
        "common_seeds": common_seeds,
    }
    if len(common_seeds) >= 2:
        analysis["paired"] = {
            "diffs": diffs,
            "mean_diff": diff_mean,
            "std_diff": diff_std,
            "median_diff": diff_median,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "t_stat": t_stat, "df": df, "significance": sig,
        }
        if "tost_result" in locals():
            analysis["tost"] = {
                "delta": tost_result.get("delta"),
                "equivalent": tost_result.get("equivalent"),
                "t1": tost_result.get("t1"),
                "t2": tost_result.get("t2"),
                "tcrit": tost_result.get("tcrit"),
                "interpretation": tost_result.get("interpretation"),
            }
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  Analysis JSON: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of multi-seed evaluation results")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    if not Path(args.results).exists():
        print(f"ERROR: results file not found: {args.results}", flush=True)
        print("Run multi_seed_eval.py first.")
        return
    analyze(args.results, args.output_dir)


if __name__ == "__main__":
    main()
