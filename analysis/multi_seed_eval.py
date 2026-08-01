#!/usr/bin/env python3
"""Multi-seed evaluation for statistical comparison of v4 vs DT Router v2.

Since P0 (diagnose_nondeterminism.py) confirmed that fixed-seed evaluations
are FULLY DETERMINISTIC (0 mismatch across repeat runs), the only source of
variance to measure is SEED variance. This script runs v4 and DT Router v2
across multiple seeds and collects per-seed metrics for paired statistical
analysis.

Design
------
- Paired design: same seed → same initial episode conditions for both configs
- Incremental save: after each run, append result to JSON (crash-safe)
- Resume capability: skip seeds already present in the output JSON
- Per-seed metrics: place_rate, n_placed, n_drift, n_near_miss, n_grasp_fail,
  mean_final_dist_cm, std_final_dist_cm

Usage
-----
    # Full 10-seed run (default seeds from user spec)
    python multi_seed_eval.py

    # Quick 3-seed test
    python multi_seed_eval.py --seeds 42,123,456 --n_episodes 200

    # Resume after interruption (automatically skips completed seeds)
    python multi_seed_eval.py  # just re-run, it will skip done seeds

Output
------
- outputs/dt_orchestrator/multi_seed_results.json  (incremental)
- outputs/dt_orchestrator/multi_seed_v4_seed{S}.log  (per-run logs)
- outputs/dt_orchestrator/multi_seed_dt_router_v2_seed{S}.log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent))
from dt_router.dt_outcomes_parser import parse_log  # noqa: E402

WORKSPACE = Path(__file__).parent
OUTPUT_DIR = WORKSPACE / "outputs" / "dt_orchestrator"
V4_CHECKPOINT = WORKSPACE / "outputs" / "iql_v4_chunking" / "final_model.pt"
V4_MODEL = WORKSPACE / "outputs" / "dt_orchestrator" / "dt_model_v2.pkl"
CONDA_ACTIVATE = "source /home/w/miniconda3/etc/profile.d/conda.sh && conda activate vla"

DEFAULT_SEEDS = [42, 123, 456, 789, 2024, 314, 271, 1618, 9999, 7777]


def build_command(config, seed, n_episodes, log_path):
    """Build eval command for a given config + seed."""
    base = (f"python evaluate_iql_env.py "
            f"--n_episodes {n_episodes} "
            f"--seed {seed} "
            f"--checkpoint {V4_CHECKPOINT} "
            f"--chunk_size 4 "
            f"--early_abort --abort_patience 30 --abort_drift 0.5 "
            f"--log_q_values ")
    if config == "v4":
        cmd = base
    elif config == "dt_router_v2":
        cmd = base + f"--dt_router {V4_MODEL} --dt_confidence 0.65 --dt_warmup_steps 20 "
    else:
        raise ValueError(f"Unknown config: {config}")
    return f"{CONDA_ACTIVATE} && cd {WORKSPACE} && {cmd} > {log_path} 2>&1"


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
    """Load existing results JSON for resume capability."""
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {"seeds": [], "configs": {}, "runs": {}}


def is_completed(results, config, seed):
    """Check if a specific (config, seed) run is already done."""
    key = f"{config}_seed{seed}"
    return key in results.get("runs", {})


def save_results(path, results):
    """Save results JSON (atomic write)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.rename(tmp, path)


def run_eval(config, seed, n_episodes, log_path, timeout=3600):
    """Run a single eval and return metrics dict."""
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
        description="Multi-seed evaluation: v4 vs DT Router v2")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Comma-separated seeds (default: 10 seeds from user spec)")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--configs", type=str, default="v4,dt_router_v2",
                        help="Comma-separated configs to evaluate")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "multi_seed_results.json"))
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    configs = [c.strip() for c in args.configs.split(",")]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Multi-Seed Evaluation")
    print("=" * 70)
    print(f"  Seeds:    {seeds}")
    print(f"  Configs:  {configs}")
    print(f"  N_eps:    {args.n_episodes}")
    print(f"  Output:   {args.output}")

    results = load_existing_results(args.output)
    results["seeds"] = seeds
    results["configs"] = configs
    results["n_episodes"] = args.n_episodes

    total_runs = len(seeds) * len(configs)
    completed = sum(1 for s in seeds for c in configs
                    if is_completed(results, c, s))
    print(f"  Progress: {completed}/{total_runs} already done (resuming)\n")

    run_idx = completed
    for seed in seeds:
        for config in configs:
            key = f"{config}_seed{seed}"
            if is_completed(results, config, seed):
                continue
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {key}")
            log_path = OUTPUT_DIR / f"multi_seed_{config}_seed{seed}.log"
            metrics, elapsed = run_eval(config, seed, args.n_episodes, log_path)
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
    print("SUMMARY")
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
            pr_cv = (pr_std / pr_mean * 100) if pr_mean > 0 else 0
            dr_mean = mean(drifts)
            print(f"\n  {config} ({len(place_rates)} seeds):")
            print(f"    place_rate: {pr_mean:.1f}% ± {pr_std:.1f}pp  (CV={pr_cv:.1f}%)")
            print(f"    per-seed: {[f'{p*100:.1f}' for p in place_rates]}")
            print(f"    drift:     {dr_mean:.1f} ± {stdev(drifts):.1f}" if len(drifts) > 1
                  else f"    drift:     {dr_mean:.1f}")

    # ---- Paired comparison (if both configs have data for same seeds) ----
    common_seeds = [s for s in seeds
                    if f"v4_seed{s}" in results["runs"]
                    and f"dt_router_v2_seed{s}" in results["runs"]]
    if len(common_seeds) >= 2:
        print(f"\n  Paired comparison ({len(common_seeds)} common seeds):")
        v4_prs = [results["runs"][f"v4_seed{s}"]["metrics"]["place_rate"]
                  for s in common_seeds]
        dt_prs = [results["runs"][f"dt_router_v2_seed{s}"]["metrics"]["place_rate"]
                  for s in common_seeds]
        diffs = [d - v for d, v in zip(dt_prs, v4_prs)]
        diff_mean = mean(diffs) * 100
        diff_std = stdev(diffs) * 100 if len(diffs) > 1 else 0
        print(f"    DT Router v2 - v4: {diff_mean:+.1f}pp ± {diff_std:.1f}pp")
        print(f"    per-seed diffs (pp): "
              f"{[f'{d*100:+.1f}' for d in diffs]}")
        # Simple paired t-test (manual, no scipy dependency)
        if len(diffs) > 1 and diff_std > 0:
            t_stat = diff_mean / (diff_std / (len(diffs) ** 0.5))
            print(f"    t-statistic: {t_stat:.2f} (df={len(diffs)-1})")
            # Rough significance: |t| > 2.262 for p<0.05 at df=9
            if abs(t_stat) > 2.262:
                print(f"    => p < 0.05 (significant at 10 seeds)")
            else:
                print(f"    => p >= 0.05 (NOT significant, need more seeds)")

    print(f"\n  Full results: {args.output}")
    print(f"  Per-run logs: {OUTPUT_DIR}/multi_seed_*.log")


if __name__ == "__main__":
    main()
