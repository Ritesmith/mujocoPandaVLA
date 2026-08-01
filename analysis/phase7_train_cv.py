#!/usr/bin/env python3
"""Phase 7 Step 1: Training CV measurement for v4 IQL.

Measures training variance by training v4 with 10 different seeds
(generated via SeedSequence(0).spawn(10)), then evaluating each
checkpoint with a fixed eval seed.

Pipeline:
  1. Generate 10 seeds via SeedSequence(0).spawn(10) → seed_manifest.json
  2. Dry-run: 2 seeds × 2 epochs, verify hash differs (seed independence)
  3. Full training: 10 seeds × 100 epochs (v4 frozen config)
  4. Evaluation: each checkpoint, fixed eval seed=42, N=200, deterministic
  5. Analysis: train CV, Q/V diagnostics CV, decision gate

Usage:
    # Dry-run only (verify seed control, ~10 min)
    python phase7_train_cv.py --dry_run

    # Full 10-seed training + eval (~6 hours)
    python phase7_train_cv.py

    # Resume after interruption (skips completed seeds)
    python phase7_train_cv.py
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np

WORKSPACE = Path("/home/w/vla_workspace")
OUTPUT_DIR = WORKSPACE / "outputs" / "phase7_variance_decomposition"
CONDA_ACTIVATE = "source /home/w/miniconda3/etc/profile.d/conda.sh && conda activate vla"

# v4 frozen config
V4_CONFIG = dict(
    n_epochs=100,
    batch_size=256,
    n_step=5,
    chunk_size=4,
    tau=0.7,
    beta=3.0,
    gamma=0.99,
    lr=3e-4,
    polyak=0.005,
    oversample_dist="0.20,0.40",
    oversample_factor=3,
)

# Fixed eval config
EVAL_SEED = 42
EVAL_N_EPISODES = 200


def generate_seeds(n=10, base_seed=0):
    """Generate n independent seeds via SeedSequence.spawn.

    Uses child.generate_state() to derive a unique integer per child.
    Note: child.entropy is the PARENT's entropy (same for all children);
    the spawn_key differentiates them. generate_state() mixes both.
    """
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(n)
    seeds = []
    for i, child in enumerate(children):
        # generate_state mixes entropy + spawn_key → unique per child
        state = child.generate_state(1, dtype=np.uint32)[0]
        seed_int = int(state)
        seeds.append({
            "spawn_index": i,
            "entropy": int(child.entropy),
            "spawn_key": list(child.spawn_key),
            "child_seed_int": seed_int,
        })
    return seeds


def save_seed_manifest(seeds, path):
    """Save seed manifest for reproducibility."""
    manifest = {
        "base_seed": 0,
        "method": "np.random.SeedSequence(0).spawn(10)",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Seed manifest saved: {path}")
    print(f"  Seeds: {[s['child_seed_int'] for s in seeds]}")


def build_train_command(seed, output_dir, n_epochs=100, dry_run=False,
                        extra_train_args=""):
    """Build train_iql.py command for a given seed.

    extra_train_args: string appended to the train_iql.py command, used for
    Phase 7 Round 1 A/B configs (e.g. "--ema_v" or "--huber_loss"). Empty
    string = v4 baseline.
    """
    cmd = (f"python -u train_iql.py "
           f"--n_epochs {n_epochs} "
           f"--batch_size {V4_CONFIG['batch_size']} "
           f"--n_step {V4_CONFIG['n_step']} "
           f"--chunk_size {V4_CONFIG['chunk_size']} "
           f"--tau {V4_CONFIG['tau']} --beta {V4_CONFIG['beta']} "
           f"--gamma {V4_CONFIG['gamma']} --lr {V4_CONFIG['lr']} "
           f"--polyak {V4_CONFIG['polyak']} "
           f"--oversample_dist {V4_CONFIG['oversample_dist']} "
           f"--oversample_factor {V4_CONFIG['oversample_factor']} "
           f"--seed {seed} "
           f"--output_dir {output_dir}")
    if extra_train_args:
        cmd += f" {extra_train_args}"
    return f"{CONDA_ACTIVATE} && cd {WORKSPACE} && {cmd}"


def build_eval_command(checkpoint_path, log_path, seed=EVAL_SEED):
    """Build evaluate_iql_env.py command for a trained checkpoint."""
    cmd = (f"python -u evaluate_iql_env.py "
           f"--n_episodes {EVAL_N_EPISODES} "
           f"--seed {seed} "
           f"--checkpoint {checkpoint_path} "
           f"--chunk_size 4 "
           f"--early_abort --abort_patience 30 --abort_drift 0.5 "
           f"--log_q_values")
    return f"{CONDA_ACTIVATE} && cd {WORKSPACE} && {cmd} > {log_path} 2>&1"


def run_command(cmd, log_path=None, timeout=7200):
    """Run a shell command, optionally logging to file.

    When log_path is provided, the ENTIRE command chain (including source,
    conda activate, cd) is wrapped in { ...; } and redirected to the log
    file. This is critical: if only the last command is redirected, the
    source/conda output goes to the pipes created by capture_output=True,
    which can fill the 64KB pipe buffer and deadlock over long runs (this
    caused seeds 2+ to hang for 2738s with 0-byte logs in Round 1 A/B).
    """
    t0 = time.time()
    if log_path:
        # Wrap entire chain in { ...; } so ALL output goes to log file
        cmd = f"{{ {cmd}; }} > {log_path} 2>&1"
        # Discard pipe output (it's in the file) to prevent buffer deadlock
        stdout_dest, stderr_dest = subprocess.DEVNULL, subprocess.DEVNULL
    else:
        # No log file: capture output for inspection
        stdout_dest, stderr_dest = subprocess.PIPE, subprocess.PIPE
    try:
        result = subprocess.run(
            cmd, shell=True, timeout=timeout, executable="/bin/bash",
            stdout=stdout_dest, stderr=stderr_dest, text=True)
        elapsed = time.time() - t0
        return (result.returncode == 0, elapsed,
                result.stdout or "", result.stderr or "")
    except subprocess.TimeoutExpired:
        return False, timeout, "", "TIMEOUT"


def dry_run_verification(seeds, output_dir, extra_train_args=""):
    """Run 2 seeds × 2 epochs to verify seed control.

    Checks:
      - Same seed → same init hash (reproducibility)
      - Different seeds → different init hash (independence)
    """
    print("\n" + "=" * 65)
    print("DRY-RUN: Seed Control Verification")
    if extra_train_args:
        print(f"  A/B config extra args: {extra_train_args}")
    print("=" * 65)

    dry_dir = output_dir / "dry_run"
    dry_dir.mkdir(parents=True, exist_ok=True)

    hashes = {}
    for seed_info in seeds[:2]:  # First 2 seeds
        seed = seed_info["child_seed_int"]
        seed_dir = dry_dir / f"seed{seed}"
        log_path = dry_dir / f"train_seed{seed}.log"

        print(f"\n  Training seed={seed} (2 epochs)...")
        cmd = build_train_command(seed, str(seed_dir), n_epochs=2, dry_run=True,
                                  extra_train_args=extra_train_args)
        success, elapsed, stdout, stderr = run_command(cmd, log_path=str(log_path))

        if not success:
            print(f"    FAILED (exit code != 0)")
            print(f"    stderr: {stderr[:500]}")
            return False

        # Extract init hash from log
        init_hash = None
        with open(log_path) as f:
            for line in f:
                if "Init Q/V hash:" in line:
                    init_hash = line.strip().split("Init Q/V hash:")[1].strip()
                    break

        if init_hash is None:
            print(f"    FAILED: no init hash found in log")
            return False

        hashes[seed] = init_hash
        print(f"    Init hash: {init_hash} ({elapsed:.0f}s)")

    # Verify: different seeds → different hashes
    seed_list = list(hashes.keys())
    if len(seed_list) >= 2:
        if hashes[seed_list[0]] != hashes[seed_list[1]]:
            print(f"\n  ✓ PASS: Different seeds → different hashes")
            print(f"    seed {seed_list[0]}: {hashes[seed_list[0]]}")
            print(f"    seed {seed_list[1]}: {hashes[seed_list[1]]}")
        else:
            print(f"\n  ✗ FAIL: Different seeds → SAME hash (seed control broken!)")
            return False

    # Verify reproducibility: same seed → same hash
    print(f"\n  Re-running seed={seed_list[0]} for reproducibility check...")
    seed = seed_list[0]
    seed_dir = dry_dir / f"seed{seed}_repeat"
    log_path = dry_dir / f"train_seed{seed}_repeat.log"
    cmd = build_train_command(seed, str(seed_dir), n_epochs=2, dry_run=True,
                              extra_train_args=extra_train_args)
    success, elapsed, stdout, stderr = run_command(cmd, log_path=str(log_path))

    if success:
        with open(log_path) as f:
            for line in f:
                if "Init Q/V hash:" in line:
                    repeat_hash = line.strip().split("Init Q/V hash:")[1].strip()
                    break

        if repeat_hash == hashes[seed]:
            print(f"    ✓ PASS: Same seed → same hash ({repeat_hash})")
        else:
            print(f"    ✗ FAIL: Same seed → different hash!")
            print(f"      first:  {hashes[seed]}")
            print(f"      repeat: {repeat_hash}")
            return False

    print(f"\n  Dry-run PASSED. Seed control verified.")
    return True


def run_full_training(seeds, output_dir, extra_train_args=""):
    """Run full 10-seed training + evaluation.

    extra_train_args: passed through to train_iql.py for A/B configs.
    """
    results = {
        "seeds": [],
        "training_runs": {},
        "eval_runs": {},
        "config": {
            "v4_frozen": V4_CONFIG,
            "extra_train_args": extra_train_args,
            "description": ("v4 baseline" if not extra_train_args
                            else f"A/B config: {extra_train_args}"),
        },
    }
    results_path = output_dir / "training_cv_results.json"

    # Load existing results for resume
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"  Resuming: {len(results.get('training_runs', {}))} seeds already done")

    n_total = len(seeds)
    for i, seed_info in enumerate(seeds):
        seed = seed_info["child_seed_int"]
        key = f"seed{seed}"

        if key in results.get("training_runs", {}):
            print(f"  [{i+1}/{n_total}] {key}: SKIP (already done)")
            continue

        print(f"\n  [{i+1}/{n_total}] Training {key}...")
        seed_dir = output_dir / f"train_{key}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_log = output_dir / f"train_{key}.log"

        cmd = build_train_command(seed, str(seed_dir), n_epochs=V4_CONFIG["n_epochs"],
                                  extra_train_args=extra_train_args)
        success, elapsed, stdout, stderr = run_command(
            cmd, log_path=str(train_log), timeout=7200)

        if not success:
            print(f"    TRAINING FAILED ({elapsed:.0f}s)")
            print(f"    Check: {train_log}")
            continue

        # Load training results
        train_results_path = seed_dir / "training_results.json"
        qv_diag_path = seed_dir / "qv_diagnostics.json"
        train_data = {}
        if train_results_path.exists():
            with open(train_results_path) as f:
                train_data = json.load(f)
        qv_data = {}
        if qv_diag_path.exists():
            with open(qv_diag_path) as f:
                qv_data = json.load(f)

        results["training_runs"][key] = {
            "seed": seed,
            "spawn_index": seed_info["spawn_index"],
            "output_dir": str(seed_dir),
            "log": str(train_log),
            "elapsed_s": round(elapsed, 1),
            "best_q_gap": train_data.get("best_q_gap"),
            "init_hash": train_data.get("init_hash"),
            "final_hash": train_data.get("final_hash"),
            "qv_diagnostics": qv_data.get("final_qv"),
        }

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"    Training done ({elapsed:.0f}s), best_q_gap={train_data.get('best_q_gap', 0):.1f}")

        # ---- Evaluation ----
        print(f"    Evaluating (eval_seed={EVAL_SEED}, N={EVAL_N_EPISODES})...")
        checkpoint = seed_dir / "final_model.pt"
        eval_log = output_dir / f"eval_{key}.log"
        eval_cmd = build_eval_command(str(checkpoint), str(eval_log), seed=EVAL_SEED)
        eval_success, eval_elapsed, _, _ = run_command(
            eval_cmd, timeout=3600)

        if eval_success:
            # Parse eval results
            sys.path.insert(0, str(WORKSPACE))
            from dt_router.dt_outcomes_parser import parse_log
            entries = parse_log(str(eval_log))
            if entries:
                from collections import Counter
                outcomes = Counter(e["outcome"] for e in entries)
                n = len(entries)
                place_rate = outcomes.get("placed", 0) / n
                results["eval_runs"][key] = {
                    "seed": seed,
                    "place_rate": place_rate,
                    "n_placed": outcomes.get("placed", 0),
                    "n_episodes": n,
                    "n_drift": outcomes.get("drift", 0),
                    "n_near_miss": outcomes.get("near_miss", 0),
                    "n_grasp_fail": outcomes.get("grasp_fail", 0),
                    "log": str(eval_log),
                    "elapsed_s": round(eval_elapsed, 1),
                }
                print(f"    Eval: {place_rate*100:.1f}% "
                      f"({outcomes.get('placed',0)}/{n}, {eval_elapsed:.0f}s)")
            else:
                print(f"    EVAL PARSE FAILED")
        else:
            print(f"    EVAL FAILED ({eval_elapsed:.0f}s)")

        # Save incrementally
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    return results


def analyze_training_cv(results):
    """Analyze training CV and Q/V diagnostics."""
    print("\n" + "=" * 65)
    print("TRAINING CV ANALYSIS")
    print("=" * 65)

    eval_runs = results.get("eval_runs", {})
    train_runs = results.get("training_runs", {})

    if len(eval_runs) < 2:
        print("  Not enough eval results for CV analysis")
        return

    place_rates = [v["place_rate"] for v in eval_runs.values()]
    pr_mean = mean(place_rates) * 100
    pr_std = stdev(place_rates) * 100 if len(place_rates) > 1 else 0
    pr_cv = (pr_std / pr_mean * 100) if pr_mean > 0 else 0

    print(f"\n  Place Rate (train CV, {len(place_rates)} seeds):")
    print(f"    mean = {pr_mean:.2f}pp")
    print(f"    std  = {pr_std:.2f}pp")
    print(f"    CV   = {pr_cv:.2f}%")
    print(f"    range: [{min(place_rates)*100:.1f}%, {max(place_rates)*100:.1f}%]")
    print(f"    per-seed: {[f'{p*100:.1f}' for p in place_rates]}")

    # Q/V diagnostics CV
    qv_means = {"q1_mean": [], "q2_mean": [], "v_mean": [], "q1_q2_gap_mean": []}
    qv_stds = {"q1_std": [], "q2_std": [], "v_std": []}
    for key, trun in train_runs.items():
        qv = trun.get("qv_diagnostics")
        if qv:
            for k in qv_means:
                qv_means[k].append(qv.get(k, 0))
            for k in qv_stds:
                qv_stds[k].append(qv.get(k, 0))

    print(f"\n  Q/V Diagnostics CV (across {len(qv_means['q1_mean'])} seeds):")
    for k, vals in qv_means.items():
        if len(vals) > 1:
            m = mean(vals)
            s = stdev(vals)
            cv = (s / abs(m) * 100) if m != 0 else 0
            print(f"    {k:20s}: {m:.4f} ± {s:.4f}  (CV={cv:.1f}%)")

    print(f"\n  Q/V Within-seed std (avg across seeds):")
    for k, vals in qv_stds.items():
        if vals:
            print(f"    {k:20s}: {mean(vals):.4f}")

    # ---- Decision Gate ----
    print(f"\n  {'='*60}")
    print(f"  DECISION GATE (pre-registered)")
    print(f"  {'='*60}")

    threshold_low = 3.0
    threshold_high = 8.0
    qv_instability_threshold = 15.0

    if pr_cv < threshold_low:
        # Check Q/V instability sub-rule
        qv_cv_max = 0
        for k, vals in qv_means.items():
            if len(vals) > 1 and mean(vals) != 0:
                cv = stdev(vals) / abs(mean(vals)) * 100
                qv_cv_max = max(qv_cv_max, cv)

        if qv_cv_max > qv_instability_threshold:
            print(f"  train CV = {pr_cv:.2f}% < {threshold_low}%")
            print(f"  BUT Q/V CV max = {qv_cv_max:.1f}% > {qv_instability_threshold}%")
            print(f"  => VALUE INSTABILITY ALERT")
            print(f"  => Proceed to Option B (init×data 2×2 factorial)")
            decision = "VALUE_INSTABILITY"
        else:
            print(f"  train CV = {pr_cv:.2f}% < {threshold_low}%")
            print(f"  Q/V CV max = {qv_cv_max:.1f}% <= {qv_instability_threshold}%")
            print(f"  => Bottleneck on EVAL side")
            print(f"  => Next: increase episodes / env sampling / FQE")
            decision = "EVAL_SIDE"
    elif pr_cv < threshold_high:
        print(f"  train CV = {pr_cv:.2f}% in [{threshold_low}%, {threshold_high}%)")
        print(f"  => Training variance notable but not dominant")
        print(f"  => Next: Option B (init×data 2×2 factorial, 4 cells × 5 reps)")
        decision = "OPTION_B"
    else:
        print(f"  train CV = {pr_cv:.2f}% >= {threshold_high}%")
        print(f"  => Training variance DOMINANT")
        print(f"  => Next: fix training stability first (loss spike / Q divergence)")
        decision = "FIX_STABILITY"

    # Save analysis
    analysis = {
        "place_rate_cv": round(pr_cv, 2),
        "place_rate_mean": round(pr_mean, 2),
        "place_rate_std": round(pr_std, 2),
        "place_rates": [round(p * 100, 1) for p in place_rates],
        "qv_cv": {
            k: round(stdev(v) / abs(mean(v)) * 100, 1) if len(v) > 1 and mean(v) != 0 else 0
            for k, v in qv_means.items()
        },
        "decision": decision,
        "thresholds": {
            "cv_low": threshold_low,
            "cv_high": threshold_high,
            "qv_instability": qv_instability_threshold,
        },
        "config": results.get("config", {}),
    }
    analysis_path = OUTPUT_DIR / "training_cv_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis saved: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Training CV measurement for v4 IQL")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only run dry-run verification (2 seeds × 2 epochs)")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: outputs/phase7_variance_decomposition")
    parser.add_argument("--skip_dry_run", action="store_true", default=False,
                        help="Skip dry-run verification (use when seed control already verified)")
    parser.add_argument("--extra_train_args", type=str, default="",
                        help="Extra args appended to train_iql.py for A/B configs. "
                             "Examples: '--ema_v' (Round 1 EMA-only), "
                             "'--huber_loss' (Round 1 Huber-only). "
                             "Empty string = v4 baseline.")
    args = parser.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Phase 7: Training CV Measurement")
    print("=" * 65)
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  N seeds:    {args.n_seeds}")
    print(f"  Base seed:  {args.base_seed}")
    if args.extra_train_args:
        print(f"  A/B config: {args.extra_train_args}")
    else:
        print(f"  A/B config: (none — v4 baseline)")

    # Step 1: Generate seeds
    print(f"\n--- Step 1: Generate seeds ---")
    seeds = generate_seeds(n=args.n_seeds, base_seed=args.base_seed)
    manifest_path = OUTPUT_DIR / "seed_manifest.json"
    save_seed_manifest(seeds, manifest_path)

    # Step 2: Dry-run verification
    if not args.skip_dry_run:
        print(f"\n--- Step 2: Dry-run verification ---")
        if not dry_run_verification(seeds, OUTPUT_DIR,
                                    extra_train_args=args.extra_train_args):
            print("\n  Dry-run FAILED. Fix seed control before proceeding.")
            sys.exit(1)
    else:
        print(f"\n--- Step 2: Dry-run SKIPPED (--skip_dry_run) ---")

    if args.dry_run:
        print("\n  Dry-run mode: stopping before full training.")
        return

    # Step 3: Full training + eval
    print(f"\n--- Step 3: Full {args.n_seeds}-seed training + eval ---")
    results = run_full_training(seeds, OUTPUT_DIR,
                                extra_train_args=args.extra_train_args)

    # Step 4: Analysis
    print(f"\n--- Step 4: Training CV analysis ---")
    analyze_training_cv(results)

    print(f"\n  Full results: {OUTPUT_DIR / 'training_cv_results.json'}")
    print(f"  Log location: {OUTPUT_DIR}/train_seed*.log, eval_seed*.log")


if __name__ == "__main__":
    main()
