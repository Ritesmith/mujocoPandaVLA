#!/usr/bin/env python3
"""Phase 8 PILOT evaluation: v4 (τ=0.5) vs DT Router v3.

Purpose: measure actual paired diff_std to reverse-engineer correlation r
between v4 and dt_router_v3 runs. This determines whether N=5 PILOT is
sufficient or we need to expand to N=30.

Key differences from multi_seed_eval.py:
  - V4_CHECKPOINT points to τ=0.5 seed1822509288 (Phase 7 final baseline)
  - V4_MODEL points to dt_model_v3.pkl (trained on τ=0.5 features)
  - PILOT seeds EXCLUDE 42 (used for codebook training → data leakage)
  - Output to outputs/phase8_dt_router_v3/

Usage:
    python phase8_pilot_eval.py --seeds 123,456,789,2024,314 --n_episodes 200
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "outputs" / "phase8_dt_router_v3"

# Phase 8 paths (τ=0.5 baseline)
V4_CHECKPOINT = WORKSPACE / "outputs" / "phase7_round2a_tau0.5_N30" / \
    "train_seed1822509288" / "final_model.pt"
V4_MODEL = OUTPUT_DIR / "dt_model_v3.pkl"

CONDA_ACTIVATE = "source /home/w/miniconda3/etc/profile.d/conda.sh && conda activate vla"

# PILOT seeds: exclude 42 (used in codebook training to avoid leakage)
PILOT_SEEDS = [123, 456, 789, 2024, 314]

sys.path.insert(0, str(WORKSPACE))
from dt_router.dt_outcomes_parser import parse_log  # noqa: E402
from collections import Counter
from statistics import mean, stdev


def build_command(config, seed, n_episodes, log_path):
    """Build eval command for v4_tau05 or dt_router_v3."""
    base = (f"python evaluate_iql_env.py "
            f"--n_episodes {n_episodes} "
            f"--seed {seed} "
            f"--checkpoint {V4_CHECKPOINT} "
            f"--chunk_size 4 "
            f"--early_abort --abort_patience 30 --abort_drift 0.5 "
            f"--log_q_values ")
    if config == "v4_tau05":
        cmd = base
    elif config == "dt_router_v3":
        cmd = base + f"--dt_router {V4_MODEL} --dt_confidence 0.65 --dt_warmup_steps 20 "
    else:
        raise ValueError(f"Unknown config: {config}")
    # Wrap in braces to avoid pipe deadlock (memory lesson)
    return f"{{ {CONDA_ACTIVATE} && cd {WORKSPACE} && {cmd} ; }} > {log_path} 2>&1"


def extract_metrics(log_path):
    """Parse eval log and extract per-seed metrics."""
    entries = parse_log(log_path)
    n = len(entries)
    if n == 0:
        return None
    outcomes = Counter(e["outcome"] for e in entries)
    dists = [e["final_dist_cm"] for e in entries if e["final_dist_cm"] is not None]
    return {
        "n_episodes": n,
        "n_placed": outcomes.get("placed", 0),
        "place_rate": outcomes.get("placed", 0) / n,
        "n_near_miss": outcomes.get("near_miss", 0),
        "n_drift": outcomes.get("drift", 0),
        "n_grasp_fail": outcomes.get("grasp_fail", 0),
        "mean_final_dist_cm": round(mean(dists), 2) if dists else None,
        "std_final_dist_cm": round(stdev(dists), 2) if len(dists) > 1 else 0.0,
        "outcome_counts": dict(outcomes),
    }


def load_existing_results(path):
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"seeds": [], "configs": [], "runs": {}}


def is_completed(results, config, seed):
    return f"{config}_seed{seed}" in results.get("runs", {})


def save_results(path, results):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.rename(tmp, path)


def run_eval(config, seed, n_episodes, log_path, timeout=3600):
    cmd = build_command(config, seed, n_episodes, log_path)
    print(f"  Running {config} seed={seed}...", end="", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, timeout=timeout, executable="/bin/bash",
            capture_output=True, text=True)
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f" FAILED (exit {result.returncode}, {elapsed:.0f}s)")
            return None, elapsed
        metrics = extract_metrics(log_path)
        if metrics is None:
            print(f" PARSE_FAIL ({elapsed:.0f}s)")
            return None, elapsed
        print(f" {metrics['place_rate']*100:.1f}% "
              f"({metrics['n_placed']}/{metrics['n_episodes']}, "
              f"drift={metrics['n_drift']}, {elapsed:.0f}s)")
        return metrics, elapsed
    except subprocess.TimeoutExpired:
        print(f" TIMEOUT ({timeout}s)")
        return None, timeout


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 PILOT: v4 (τ=0.5) vs DT Router v3")
    parser.add_argument("--seeds", type=str,
                        default=",".join(map(str, PILOT_SEEDS)),
                        help=f"Comma-separated eval seeds (default PILOT: {PILOT_SEEDS})")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--configs", type=str, default="v4_tau05,dt_router_v3",
                        help="Comma-separated configs to evaluate")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "pilot_results.json"))
    parser.add_argument("--timeout", type=int, default=1200,
                        help="Per-run timeout in seconds (default 1200)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    configs = [c.strip() for c in args.configs.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 8 PILOT Evaluation: v4 (τ=0.5) vs DT Router v3")
    print("=" * 70)
    print(f"  V4 checkpoint: {V4_CHECKPOINT}")
    print(f"  DT model:      {V4_MODEL}")
    print(f"  Seeds:         {seeds}")
    print(f"  Configs:       {configs}")
    print(f"  N_episodes:    {args.n_episodes}")
    print(f"  Output:        {args.output}")
    print(f"  Timeout:       {args.timeout}s per run")

    results = load_existing_results(args.output)
    results["seeds"] = seeds
    results["configs"] = configs
    results["n_episodes"] = args.n_episodes
    results["v4_checkpoint"] = str(V4_CHECKPOINT)
    results["dt_model"] = str(V4_MODEL)

    total_runs = len(seeds) * len(configs)
    completed = sum(1 for s in seeds for c in configs
                    if is_completed(results, c, s))
    print(f"  Progress:      {completed}/{total_runs} already done (resuming)\n")

    run_idx = completed
    for seed in seeds:
        for config in configs:
            key = f"{config}_seed{seed}"
            if is_completed(results, config, seed):
                continue
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {key}")
            log_path = OUTPUT_DIR / f"pilot_{config}_seed{seed}.log"
            metrics, elapsed = run_eval(config, seed, args.n_episodes,
                                        log_path, timeout=args.timeout)
            if metrics is not None:
                results["runs"][key] = {
                    "config": config,
                    "seed": seed,
                    "metrics": metrics,
                    "log": str(log_path),
                    "elapsed_s": round(elapsed, 1),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_results(args.output, results)
            else:
                print(f"  WARNING: {key} failed, skipping (will retry on resume)")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)

    for config in configs:
        place_rates = []
        drifts = []
        for seed in seeds:
            key = f"{config}_seed{seed}"
            if key in results["runs"]:
                m = results["runs"][key]["metrics"]
                place_rates.append(m["place_rate"])
                drifts.append(m["n_drift"])
        if place_rates:
            pr_mean = mean(place_rates) * 100
            pr_std = stdev(place_rates) * 100 if len(place_rates) > 1 else 0
            pr_cv = pr_std / pr_mean * 100 if pr_mean > 0 else 0
            print(f"  {config:15s}: {pr_mean:.2f}% ± {pr_std:.2f}pp "
                  f"(CV={pr_cv:.2f}%, drift_mean={mean(drifts):.1f}, N={len(place_rates)})")

    # Paired diff analysis (if both configs have data for same seeds)
    if len(configs) == 2:
        c1, c2 = configs
        diffs = []
        for seed in seeds:
            k1, k2 = f"{c1}_seed{seed}", f"{c2}_seed{seed}"
            if k1 in results["runs"] and k2 in results["runs"]:
                pr1 = results["runs"][k1]["metrics"]["place_rate"]
                pr2 = results["runs"][k2]["metrics"]["place_rate"]
                diffs.append((pr2 - pr1) * 100)  # pp
        if len(diffs) >= 2:
            d_mean = mean(diffs)
            d_std = stdev(diffs) if len(diffs) > 1 else 0
            print(f"\n  Paired diff ({c2} - {c1}):")
            print(f"    N = {len(diffs)}")
            print(f"    mean = {d_mean:+.2f}pp")
            print(f"    std  = {d_std:.2f}pp")
            print(f"    diffs = {[round(d, 2) for d in diffs]}")

            # Reverse-engineer r from diff_std
            # σ_d = σ_place * sqrt(2 - 2r) → r = 1 - σ_d²/(2*σ_place²)
            # Need σ_place from individual configs (use c1 as reference)
            prs1 = [results["runs"][f"{c1}_seed{s}"]["metrics"]["place_rate"]
                    for s in seeds if f"{c1}_seed{s}" in results["runs"]]
            if len(prs1) > 1:
                sigma_place = stdev(prs1) * 100  # in pp
                if sigma_place > 0:
                    r_estimated = 1 - (d_std ** 2) / (2 * sigma_place ** 2)
                    print(f"\n  Reverse-engineered correlation r:")
                    print(f"    σ_place ({c1}) = {sigma_place:.2f}pp")
                    print(f"    σ_diff  = {d_std:.2f}pp")
                    print(f"    r ≈ 1 - σ_d²/(2·σ_place²) = {r_estimated:.4f}")

                    # Power analysis for +1pp and +2pp at N=10, N=30
                    from math import sqrt, erf
                    def power(n, effect, sigma_d, alpha=0.05):
                        # two-sided paired t-test, approximate with normal
                        ncp = effect / sigma_d * sqrt(n)
                        t_crit = 1.96  # approximate for two-sided alpha=0.05
                        return 1 - 0.5 * (1 + erf((t_crit - ncp) / sqrt(2)))

                    print(f"\n  Power analysis (based on PILOT σ_d={d_std:.2f}pp):")
                    for eff in [1.0, 2.0]:
                        for n in [10, 30]:
                            p = power(n, eff, d_std)
                            print(f"    +{eff:.0f}pp @ N={n:2d}: power={p:.3f} "
                                  f"{'✓ PASS' if p >= 0.8 else '✗ fail'}")

    print(f"\n  Results: {args.output}")
    print(f"  Logs:    {OUTPUT_DIR}/pilot_*.log")


if __name__ == "__main__":
    main()
