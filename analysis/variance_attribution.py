#!/usr/bin/env python3
"""Variance attribution analysis for multi-seed RL evaluation.

CRITICAL DISTINCTION:
  P1 measured EVALUATION variance (fixed checkpoint, different eval seeds).
  This is NOT the same as TRAINING variance (different training seeds →
  different checkpoints → different place rates).

This script does two things:

PART A — Evaluation variance source decomposition (runs immediately on
existing P1 data):
  Decomposes the observed CV=4.91% into:
    - Between-seed variance: env initial-state distribution differences
      (different eval seed → different episode initial conditions)
    - Within-seed variance: finite-sample sampling variance
      (200 episodes is a sample, not the full population)
  Answers: "Is the CV from env randomness or from too few episodes?"

PART B — Training variance attribution experiment DESIGN (generates a
command matrix, does NOT run):
  Designs a 4-dimensional seed decomposition for future training runs:
    init_seed (weight init), data_seed (shuffle), env_seed (env interaction),
    grad_seed (dropout/gradient noise)
  Each run fixes 3 dims, varies 1 → ~40 training runs needed.
  Answers: "Which training randomness source contributes most to
  checkpoint-to-checkpoint place_rate variance?"

Usage:
    python variance_attribution.py                # Part A (analysis) + Part B (design)
    python variance_attribution.py --part a       # Part A only
    python variance_attribution.py --part b       # Part B only (design)
"""
import argparse
import json
import random
from pathlib import Path
from statistics import mean, stdev

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "outputs" / "dt_orchestrator"
SEEDS = [42, 123, 456, 789, 2024, 314, 271, 1618, 9999, 7777]

import sys
sys.path.insert(0, str(WORKSPACE))
from dt_router.dt_outcomes_parser import parse_log  # noqa: E402


def part_a_evaluation_variance(n_bootstrap=5000, seed=42):
    """Decompose evaluation variance into between-seed and within-seed.

    Uses per-episode outcomes from the 10 completed v4 eval runs
    (multi_seed_v4_seed{S}.log, 200 episodes each).
    """
    print("=" * 70)
    print("PART A: Evaluation Variance Source Decomposition")
    print("=" * 70)

    # Parse per-episode outcomes for each seed
    seed_episodes = {}  # seed -> list of outcome strings
    for s in SEEDS:
        log_path = OUTPUT_DIR / f"multi_seed_v4_seed{s}.log"
        if not log_path.exists():
            print(f"  WARNING: {log_path} not found, skipping seed {s}")
            continue
        entries = parse_log(str(log_path))
        outcomes = [e["outcome"] for e in entries]
        seed_episodes[s] = outcomes

    if len(seed_episodes) < 2:
        print("  ERROR: need >= 2 seeds with per-episode data", flush=True)
        return None

    n_seeds = len(seed_episodes)
    n_eps = len(next(iter(seed_episodes.values())))

    print(f"  Seeds analyzed: {n_seeds} (each with {n_eps} episodes)")
    print(f"  Bootstrap iterations: {n_bootstrap}")

    # Per-seed observed place_rate
    seed_prs = {}
    for s, outcomes in seed_episodes.items():
        pr = sum(1 for o in outcomes if o == "placed") / len(outcomes)
        seed_prs[s] = pr

    observed_prs = list(seed_prs.values())
    obs_mean = mean(observed_prs)
    obs_std = stdev(observed_prs) if len(observed_prs) > 1 else 0
    obs_cv = obs_std / obs_mean * 100 if obs_mean > 0 else 0

    print(f"\n  Observed place_rates per seed (pp):")
    for s in SEEDS:
        if s in seed_prs:
            print(f"    seed {s:>5d}: {seed_prs[s]*100:.1f}%")
    print(f"  Mean: {obs_mean*100:.1f}%, Std: {obs_std*100:.2f}pp, CV: {obs_cv:.2f}%")

    # ---- Between-seed variance (env initial-state distribution) ----
    # This is the variance of the TRUE per-seed place_rate.
    # Estimated as: Var(observed_prs) - mean(within-seed variance) / n_eps
    # (correcting for finite-sample bias: observed Var includes both
    #  true between-seed Var AND within-seed sampling Var/n_eps)

    # ---- Within-seed variance (finite-sample sampling) ----
    # For each seed, bootstrap-resample n_eps episodes, compute place_rate,
    # repeat B times → within-seed variance.
    rng = random.Random(seed)
    within_variances = {}
    within_stds = {}
    for s, outcomes in seed_episodes.items():
        n = len(outcomes)
        boot_prs = []
        for _ in range(n_bootstrap):
            sample = [outcomes[rng.randrange(n)] for _ in range(n)]
            pr = sum(1 for o in sample if o == "placed") / n
            boot_prs.append(pr)
        w_var = sum((p - mean(boot_prs))**2 for p in boot_prs) / len(boot_prs)
        within_variances[s] = w_var
        within_stds[s] = w_var ** 0.5

    mean_within_var = mean(within_variances.values())
    mean_within_std = mean(within_stds.values())

    # Between-seed variance (deconvolved)
    observed_var = obs_std ** 2
    between_var = observed_var - mean_within_var / n_eps
    between_var = max(between_var, 0)  # can't be negative
    between_std = between_var ** 0.5

    total_var = between_var + mean_within_var
    between_pct = between_var / total_var * 100 if total_var > 0 else 0
    within_pct = mean_within_var / total_var * 100 if total_var > 0 else 0

    print(f"\n  {'Variance Decomposition':^60s}")
    print(f"  {'-'*60}")
    print(f"  Between-seed (env initial-state):  {between_std*100:.2f}pp "
          f"(Var={between_var*1e4:.4f}pp², {between_pct:.1f}%)")
    print(f"  Within-seed (finite-sample):       {mean_within_std*100:.2f}pp "
          f"(Var={mean_within_var*1e4:.4f}pp², {within_pct:.1f}%)")
    print(f"  {'-'*60}")
    print(f"  Total (observed):                  {obs_std*100:.2f}pp")

    print(f"\n  INTERPRETATION:")
    if between_pct > 60:
        print(f"    → {between_pct:.0f}% 方差来自 env 初始状态分布差异 (between-seed)")
        print(f"    → 增加 N_episodes 收益有限 (within 仅 {within_pct:.0f}%)")
        print(f"    → 降低 CV 需要改变 env 初始状态采样策略 (如分层采样/重要性采样)")
    elif within_pct > 60:
        print(f"    → {within_pct:.0f}% 方差来自有限样本采样 (within-seed)")
        print(f"    → 增加 N_episodes (200→1000) 可有效降低 CV")
        print(f"    → 当前 200 episodes 不足以稳定估计真实 place_rate")
    else:
        print(f"    → 两类方差贡献接近 (between {between_pct:.0f}% / within {within_pct:.0f}%)")
        print(f"    → 需同时增加 N_episodes 和改进 env 采样")

    # What N_episodes would reduce within-seed std to target?
    target_within_std = 0.01  # 1pp
    if mean_within_std > 0:
        n_needed = int(n_eps * (mean_within_std / target_within_std) ** 2)
        print(f"\n  要将 within-seed std 降至 {target_within_std*100:.1f}pp:")
        print(f"    需要 N_episodes ≈ {n_needed} (当前 {n_eps})")

    return {
        "n_seeds": n_seeds, "n_episodes": n_eps,
        "observed_mean_pp": obs_mean * 100,
        "observed_std_pp": obs_std * 100,
        "observed_cv_pct": obs_cv,
        "between_seed_std_pp": between_std * 100,
        "within_seed_std_pp": mean_within_std * 100,
        "between_pct": between_pct,
        "within_pct": within_pct,
        "per_seed_pr_pp": {str(s): pr * 100 for s, pr in seed_prs.items()},
        "n_episodes_for_1pp_within": n_needed if mean_within_std > 0 else None,
    }


def part_b_training_variance_design():
    """Design a 4-dimensional training seed decomposition experiment.

    Generates a JSON matrix of ~40 training configs where each config
    fixes 3 of {init, data, env, grad} seed dims and varies 1.

    Requires train_iql.py to support --init_seed, --data_seed, --env_seed,
    --grad_seed (currently it only has --seed).
    """
    print("\n" + "=" * 70)
    print("PART B: Training Variance Attribution Experiment DESIGN")
    print("=" * 70)

    dims = ["init_seed", "data_seed", "env_seed", "grad_seed"]
    base_seed = 42
    varied_seeds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    configs = []
    config_id = 0

    # For each dimension to vary:
    for vary_dim in dims:
        # Fix the other 3 dims to base_seed, vary this dim across 10 values
        for vs in varied_seeds:
            config_id += 1
            cfg = {"id": config_id, "vary_dim": vary_dim}
            for d in dims:
                cfg[d] = vs if d == vary_dim else base_seed
            configs.append(cfg)

    # Also add 5 "all-same" baseline runs (all dims = same seed)
    for i, s in enumerate([42, 123, 456, 789, 2024]):
        config_id += 1
        cfg = {"id": config_id, "vary_dim": "all"}
        for d in dims:
            cfg[d] = s
        configs.append(cfg)

    print(f"  Total training configs: {len(configs)}")
    print(f"  Dimensions: {dims}")
    print(f"  Base seed (fixed): {base_seed}")
    print(f"  Varied seeds: {varied_seeds} ({len(varied_seeds)} per dim)")
    print(f"\n  Breakdown:")
    for d in dims:
        n = sum(1 for c in configs if c["vary_dim"] == d)
        print(f"    Vary {d:12s}: {n} runs")
    n_all = sum(1 for c in configs if c["vary_dim"] == "all")
    print(f"    {'all (baseline)':12s}: {n_all} runs")

    print(f"\n  Analysis plan (after all runs complete):")
    print(f"    For each vary_dim d:")
    print(f"      place_rates_d = [result for configs where vary_dim == d]")
    print(f"      var_d = Var(place_rates_d)  # variance attributable to dim d")
    print(f"    Total training variance = sum(var_d for all d)")
    print(f"    Contribution of dim d = var_d / total")

    print(f"\n  PREREQUISITE: train_iql.py must support:")
    print(f"    --init_seed (torch.manual_seed + nn.init)")
    print(f"    --data_seed  (DataLoader shuffle / replay buffer)")
    print(f"    --env_seed   (gym env.seed for training rollouts)")
    print(f"    --grad_seed  (dropout / gradient noise, if applicable)")

    # Save design
    design_path = OUTPUT_DIR / "training_variance_design.json"
    design = {
        "description": "4-dim training seed decomposition experiment",
        "dims": dims,
        "base_seed": base_seed,
        "varied_seeds": varied_seeds,
        "n_configs": len(configs),
        "configs": configs,
        "analysis_plan": (
            "For each vary_dim d: var_d = Var(place_rates where vary_dim==d). "
            "Contribution of d = var_d / sum(var_all_dims). "
            "Identifies which training randomness source to fix first."
        ),
    }
    with open(design_path, "w") as f:
        json.dump(design, f, indent=2)
    print(f"\n  Design saved: {design_path}")
    return design


def main():
    parser = argparse.ArgumentParser(
        description="Variance attribution: evaluation (Part A) + training design (Part B)")
    parser.add_argument("--part", choices=["a", "b", "both"], default="both")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "variance_attribution.json"))
    args = parser.parse_args()

    result = {}
    if args.part in ("a", "both"):
        result["part_a_evaluation"] = part_a_evaluation_variance(
            n_bootstrap=args.n_bootstrap)

    if args.part in ("b", "both"):
        result["part_b_training_design"] = part_b_training_variance_design()

    if result:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Full output: {args.output}")


if __name__ == "__main__":
    main()
