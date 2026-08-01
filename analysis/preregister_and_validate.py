#!/usr/bin/env python3
"""Pre-registration manifest + formalized decision gate for RL experiments.

Closes three gaps from the methodology review:
  1. Pre-registration: write expected effect/delta/power BEFORE running,
     preventing post-hoc analysis adjustment.
  2. Formalized decision gate: rules are code, not human judgment. Every
     trigger is logged and traceable.
  3. Anchor regression: every new config is compared against a fixed
     anchor (v4), not just the previous version, preventing drift.

Usage:
    # Step 1: Pre-register before running (writes manifest JSON)
    python preregister_and_validate.py --init \\
        --name v4.2_vs_v4 \\
        --expected_effect 2.0 \\
        --delta 1.0 \\
        --target_power 0.8 \\
        --planned_seeds 30 \\
        --anchor v4

    # Step 2: After experiment completes, validate against manifest
    python preregister_and_validate.py --validate \\
        --manifest experiment_manifest_v4.2_vs_v4.json \\
        --results multi_seed_results.json

Decision gate rules (formalized, configurable):
    R1: cv > cv_threshold           → TRIGGER stability_optimization
    R2: not tost_equivalent         → CANNOT declare equivalence
    R3: tost_equivalent & not sig   → DECLARE equivalent
    R4: significant & diff > 0      → treatment better, proceed
    R5: significant & diff < 0      → anchor better, accept anchor
    R6: anchor_ci_lower < min       → insufficient, need more seeds
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "outputs" / "dt_orchestrator"

# Import TOST from analyze_multi_seed.py
sys.path.insert(0, str(WORKSPACE))
from analysis.analyze_multi_seed import tost_equivalence, bootstrap_ci, paired_t_test  # noqa


def init_manifest(name, expected_effect, delta, target_power,
                  planned_seeds, anchor, n_episodes=200):
    """Create a pre-registration manifest BEFORE running the experiment."""
    manifest = {
        "experiment_name": name,
        "preregistered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hypothesis": (f"Treatment config place_rate exceeds {anchor} "
                       f"by >= {expected_effect}pp"),
        "expected_effect_pp": expected_effect,
        "equivalence_delta_pp": delta,
        "target_power": target_power,
        "alpha": 0.05,
        "anchor_config": anchor,
        "planned_n_seeds": planned_seeds,
        "n_episodes": n_episodes,
        "decision_gate_config": {
            "cv_threshold": 3.0,
            "tost_delta": delta / 100.0,  # pp → fraction
            "min_anchor_ci_lower": 0.68,
            "t_crit_p05_two_sided": 2.262,  # df=9; will be recalculated
        },
        "status": "preregistered",
    }
    path = OUTPUT_DIR / f"experiment_manifest_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Pre-registration manifest created: {path}")
    print(f"  Experiment: {name}")
    print(f"  Hypothesis: {manifest['hypothesis']}")
    print(f"  Expected effect: +{expected_effect}pp")
    print(f"  Equivalence delta: ±{delta}pp (TOST)")
    print(f"  Target power: {target_power}")
    print(f"  Planned seeds: {planned_seeds}")
    print(f"  Anchor: {anchor}")
    print(f"\n  → Run the experiment, then validate with --validate")
    return path


def validate(manifest_path, results_path):
    """Validate experiment results against pre-registered manifest.

    Executes the formalized decision gate and reports which rules triggered.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    dg = manifest["decision_gate_config"]
    anchor = manifest["anchor_config"]
    treatment = None
    # Infer treatment config (the one that's not the anchor)
    for c in results.get("configs", []):
        if c != anchor:
            treatment = c
            break
    if treatment is None:
        print(f"ERROR: no treatment config found (all configs == anchor '{anchor}')")
        return

    seeds = results.get("seeds", [])
    runs = results.get("runs", {})

    # Collect paired diffs
    common_seeds = [s for s in seeds
                    if f"{anchor}_seed{s}" in runs
                    and f"{treatment}_seed{s}" in runs]
    if len(common_seeds) < 2:
        print(f"ERROR: need >= 2 common seeds, got {len(common_seeds)}")
        return

    anchor_prs = [runs[f"{anchor}_seed{s}"]["metrics"]["place_rate"]
                  for s in common_seeds]
    treat_prs = [runs[f"{treatment}_seed{s}"]["metrics"]["place_rate"]
                 for s in common_seeds]
    diffs = [t - a for t, a in zip(treat_prs, anchor_prs)]

    # Statistics
    diff_mean = mean(diffs)
    diff_std = stdev(diffs) if len(diffs) > 1 else 0
    anchor_mean = mean(anchor_prs)
    anchor_std = stdev(anchor_prs) if len(anchor_prs) > 1 else 0
    anchor_cv = anchor_std / anchor_mean * 100 if anchor_mean > 0 else 0
    treat_cv = (stdev(treat_prs) / mean(treat_prs) * 100) if len(treat_prs) > 1 else 0
    max_cv = max(anchor_cv, treat_cv)

    # TOST
    tost_delta = dg["tost_delta"]
    tost_result = tost_equivalence(diffs, tost_delta)

    # Paired t-test
    t_stat, df, sig = paired_t_test(diffs)
    significant = "p < 0.05" in sig or "p < 0.01" in sig

    # Bootstrap CI for anchor
    anchor_ci_lo, anchor_ci_hi, _ = bootstrap_ci(anchor_prs)

    # ---- Formalized decision gate ----
    rules = []
    # R1: CV threshold
    r1_triggered = max_cv > dg["cv_threshold"]
    rules.append({
        "id": "R1", "name": "cv_threshold",
        "condition": f"max_cv ({max_cv:.2f}%) > {dg['cv_threshold']}%",
        "triggered": r1_triggered,
        "action": "require_stability_optimization" if r1_triggered else "pass",
        "message": (f"CV={max_cv:.2f}% > {dg['cv_threshold']}% → "
                    f"优先优化训练稳定性" if r1_triggered else
                    f"CV={max_cv:.2f}% <= {dg['cv_threshold']}% (OK)"),
    })
    # R2/R3: TOST
    r2_triggered = not tost_result.get("equivalent", False)
    rules.append({
        "id": "R2", "name": "tost_not_equivalent",
        "condition": f"not TOST equivalent (delta=±{tost_delta*100:.1f}pp)",
        "triggered": r2_triggered,
        "action": "cannot_declare_equivalence" if r2_triggered else "pass",
        "message": tost_result.get("interpretation", ""),
    })
    # R4/R5: significance
    if significant and diff_mean > 0:
        rules.append({
            "id": "R4", "name": "treatment_better",
            "condition": f"significant & diff > 0 ({diff_mean*100:+.2f}pp)",
            "triggered": True, "action": "proceed_with_treatment",
            "message": (f"{treatment} 显著优于 {anchor} "
                        f"(+{diff_mean*100:.2f}pp, p<0.05) → 采纳 treatment"),
        })
    elif significant and diff_mean < 0:
        rules.append({
            "id": "R5", "name": "anchor_better",
            "condition": f"significant & diff < 0 ({diff_mean*100:+.2f}pp)",
            "triggered": True, "action": "accept_anchor",
            "message": (f"{anchor} 显著优于 {treatment} "
                        f"({diff_mean*100:+.2f}pp, p<0.05) → 接受 {anchor}"),
        })
    else:
        r3_triggered = tost_result.get("equivalent", False) and not significant
        rules.append({
            "id": "R3", "name": "declare_equivalent",
            "condition": "TOST equivalent & not significant",
            "triggered": r3_triggered,
            "action": "declare_equivalent" if r3_triggered else "inconclusive",
            "message": (f"可声明统计等价 (±{tost_delta*100:.1f}pp)" if r3_triggered
                        else f"无法声明等价也无法拒绝 (diff={diff_mean*100:+.2f}pp, "
                             f"p>0.05, TOST fail) → 需更多 seed"),
        })
    # R6: anchor CI lower bound
    r6_triggered = anchor_ci_lo < dg["min_anchor_ci_lower"]
    rules.append({
        "id": "R6", "name": "anchor_ci_lower",
        "condition": (f"anchor CI lower ({anchor_ci_lo*100:.1f}%) < "
                      f"{dg['min_anchor_ci_lower']*100:.1f}%"),
        "triggered": r6_triggered,
        "action": "need_more_seeds" if r6_triggered else "pass",
        "message": (f"{anchor} CI 下界 {anchor_ci_lo*100:.1f}% < "
                    f"{dg['min_anchor_ci_lower']*100:.1f}% → 需更多 seed"
                    if r6_triggered else
                    f"{anchor} CI 下界 {anchor_ci_lo*100:.1f}% >= "
                    f"{dg['min_anchor_ci_lower']*100:.1f}% (OK)"),
    })

    # ---- Report ----
    print("=" * 70)
    print(f"Validation: {manifest['experiment_name']}")
    print("=" * 70)
    print(f"  Manifest:    {manifest_path}")
    print(f"  Results:     {results_path}")
    print(f"  Anchor:      {anchor} ({len(common_seeds)} seeds)")
    print(f"  Treatment:   {treatment} ({len(common_seeds)} seeds)")
    print(f"  Preregistered: effect=+{manifest['expected_effect_pp']}pp, "
          f"delta=±{manifest['equivalence_delta_pp']}pp, "
          f"power={manifest['target_power']}, "
          f"planned_seeds={manifest['planned_n_seeds']}")
    print(f"\n  Observed:")
    print(f"    {anchor}:    {anchor_mean*100:.1f}% ± {anchor_std*100:.2f}pp "
          f"(CV={anchor_cv:.2f}%, CI=[{anchor_ci_lo*100:.1f}, {anchor_ci_hi*100:.1f}])")
    print(f"    {treatment}: {mean(treat_prs)*100:.1f}% ± "
          f"{stdev(treat_prs)*100:.2f}pp (CV={treat_cv:.2f}%)")
    print(f"    diff:        {diff_mean*100:+.2f}pp ± {diff_std*100:.2f}pp "
          f"(t={t_stat:.3f}, {sig})")

    print(f"\n  {'DECISION GATE (formalized)':^66s}")
    print(f"  {'-'*66}")
    for r in rules:
        status = "⚠ TRIGGERED" if r["triggered"] else "✓ pass"
        print(f"  {r['id']} [{r['name']:24s}] {status}")
        print(f"     cond: {r['condition']}")
        print(f"     msg:  {r['message']}")
    print(f"  {'-'*66}")

    # Final verdict
    triggered = [r for r in rules if r["triggered"]]
    print(f"\n  FINAL VERDICT ({len(triggered)} rule(s) triggered):")
    if any(r["id"] == "R1" for r in triggered):
        print(f"    → BLOCKED: 优先优化训练稳定性 (CV > {dg['cv_threshold']}%)")
    elif any(r["id"] == "R4" for r in triggered):
        print(f"    → PROCEED: 采纳 {treatment}")
    elif any(r["id"] == "R5" for r in triggered):
        print(f"    → ACCEPT: 接受 {anchor} 作为基线")
    elif any(r["id"] == "R3" for r in triggered):
        print(f"    → EQUIVALENT: {treatment} 与 {anchor} 统计等价")
    else:
        print(f"    → INCONCLUSIVE: 需更多 seed 或降低 CV")

    # Check if planned seeds matched actual
    actual_seeds = len(common_seeds)
    planned = manifest["planned_n_seeds"]
    if actual_seeds < planned:
        print(f"\n  ⚠ 实际 seed 数 ({actual_seeds}) < 计划 ({planned}): 功效可能不足")
    else:
        print(f"\n  ✓ 实际 seed 数 ({actual_seeds}) >= 计划 ({planned})")

    # Update manifest status
    manifest["status"] = "validated"
    manifest["validation_result"] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "actual_n_seeds": actual_seeds,
        "diff_mean_pp": diff_mean * 100,
        "diff_std_pp": diff_std * 100,
        "t_stat": t_stat, "significance": sig,
        "tost_equivalent": tost_result.get("equivalent"),
        "anchor_cv": anchor_cv, "treatment_cv": treat_cv,
        "rules_triggered": [r["id"] for r in triggered],
        "final_verdict": ("blocked_stability" if any(r["id"]=="R1" for r in triggered)
                          else "proceed" if any(r["id"]=="R4" for r in triggered)
                          else "accept_anchor" if any(r["id"]=="R5" for r in triggered)
                          else "equivalent" if any(r["id"]=="R3" for r in triggered)
                          else "inconclusive"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest updated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-registration + formalized decision gate")
    sub = parser.add_subparsers(dest="mode")

    p_init = sub.add_parser("init", help="Create pre-registration manifest")
    p_init.add_argument("--name", required=True)
    p_init.add_argument("--expected_effect", type=float, default=2.0,
                        help="Expected effect in pp (default: 2.0)")
    p_init.add_argument("--delta", type=float, default=1.0,
                        help="TOST equivalence delta in pp (default: 1.0)")
    p_init.add_argument("--target_power", type=float, default=0.8)
    p_init.add_argument("--planned_seeds", type=int, default=30)
    p_init.add_argument("--anchor", default="v4")
    p_init.add_argument("--n_episodes", type=int, default=200)

    p_val = sub.add_parser("validate", help="Validate results against manifest")
    p_val.add_argument("--manifest", required=True)
    p_val.add_argument("--results", default=str(
        OUTPUT_DIR / "multi_seed_results.json"))

    args = parser.parse_args()
    if args.mode == "init":
        init_manifest(args.name, args.expected_effect, args.delta,
                      args.target_power, args.planned_seeds,
                      args.anchor, args.n_episodes)
    elif args.mode == "validate":
        validate(args.manifest, args.results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
