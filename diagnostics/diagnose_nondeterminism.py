#!/usr/bin/env python3
"""Diagnose non-determinism in IQL environment evaluation.

This tool runs the same evaluation config multiple times with an identical
seed and compares per-episode outcomes to quantify run-to-run variance.
It localizes the non-determinism source (cudnn / DataLoader / floating-point
reduction order) and optionally validates that deterministic-GPU flags fix it.

Rationale
---------
v1 DT Router's 75.5% contained +7pp non-determinism noise inflation (17 non-
switched episodes mismatched the v4 baseline: 12 gain - 5 loss). Before
trusting ANY single-run "performance improvement", this diagnostic must
confirm reproducibility. See CHANGELOG.md (2026-07-15) and Colas et al.
"How Many Random Seeds?" for the statistical methodology.

Layers of comparison
--------------------
1. Checkpoint hash  — confirm final_model.pt is byte-identical across runs
2. Per-episode outcome — parse eval log, compare placed/near_miss/drift per ep
3. Per-episode final_dist — quantify trajectory divergence magnitude
4. Source isolation — enable deterministic GPU flags, re-run, check if fixed

Usage
-----
    # Baseline: run v4 eval twice, compare
    python diagnose_nondeterminism.py --config v4 --n_episodes 200

    # Also test deterministic GPU fix (3rd run with flags enabled)
    python diagnose_nondeterminism.py --config v4 --n_episodes 200 --deterministic_gpu

    # Quick smoke (N=20)
    python diagnose_nondeterminism.py --config v4 --n_episodes 20

    # Test DT Router v2 reproducibility
    python diagnose_nondeterminism.py --config dt_router_v2 --n_episodes 200

Output
------
- Console: mismatch summary + likely non-determinism source
- JSON report at --output (default: outputs/dt_orchestrator/nondeterminism_report.json)
- Raw eval logs at outputs/dt_orchestrator/nondet_run_{i}.log
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# Reuse the proven parser from dt_outcomes_parser.py
sys.path.insert(0, str(Path(__file__).parent))
from dt_router.dt_outcomes_parser import parse_log, classify  # noqa: E402

WORKSPACE = Path(__file__).parent
EVAL_SCRIPT = WORKSPACE / "evaluate_iql_env.py"
OUTPUT_DIR = WORKSPACE / "outputs" / "dt_orchestrator"

# IQL v4 checkpoint (shared by v4, v4.1, warmup_switch, DT router)
V4_CHECKPOINT = WORKSPACE / "outputs" / "iql_v4_chunking" / "final_model.pt"
V4_MODEL = WORKSPACE / "outputs" / "dt_orchestrator" / "dt_model_v2.pkl"

# Conda env activation
CONDA_ACTIVATE = "source /home/w/miniconda3/etc/profile.d/conda.sh && conda activate vla"


def build_eval_command(config, n_episodes, seed, log_path, deterministic_gpu=False):
    """Build the shell command to run evaluate_iql_env.py for a given config.

    Args:
        config: "v4" or "dt_router_v2"
        n_episodes: number of episodes
        seed: random seed
        log_path: where to redirect stdout
        deterministic_gpu: if True, wrap with torch deterministic flags
    Returns:
        (command_str, env_dict) — env_dict has extra env vars to set
    """
    base = (f"python evaluate_iql_env.py "
            f"--n_episodes {n_episodes} "
            f"--seed {seed} "
            f"--checkpoint {V4_CHECKPOINT} "
            f"--chunk_size 4 "
            f"--early_abort --abort_patience 30 --abort_drift 0.5 "
            f"--log_q_values ")

    if config == "v4":
        # Pure v4: chunk_size=4 only (no adaptive, no router)
        cmd = base
    elif config == "dt_router_v2":
        cmd = base + f"--dt_router {V4_MODEL} --dt_confidence 0.65 --dt_warmup_steps 20 "
    else:
        raise ValueError(f"Unknown config: {config}")

    if deterministic_gpu:
        # Wrap in a python -c that sets torch flags before running eval.
        # CUBLAS_WORKSPACE_CONFIG must be set as env var.
        wrapper = (
            "python -c \""
            "import torch; "
            "torch.backends.cudnn.deterministic=True; "
            "torch.backends.cudnn.benchmark=False; "
            "import os; os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); "
            "try: torch.use_deterministic_algorithms(True, warn_only=True); "
            "except Exception as e: print('WARN det-alg:', e); "
            "import runpy; runpy.run_path('evaluate_iql_env.py', run_name='__main__')\" "
        )
        # Strip leading "python evaluate_iql_env.py" from base since wrapper runs it
        args_after_script = cmd[len("python evaluate_iql_env.py"):]
        cmd = wrapper + args_after_script.lstrip()

    full = f"{CONDA_ACTIVATE} && cd {WORKSPACE} && {cmd} > {log_path} 2>&1"
    env = os.environ.copy()
    if deterministic_gpu:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return full, env


def hash_file(path, algo="md5", chunk_size=65536):
    """Return hex digest of file content, or None if file missing."""
    if not Path(path).exists():
        return None
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def outcomes_to_dict(entries):
    """Convert list of entry dicts to {ep: entry} for easy comparison."""
    return {e["ep"]: e for e in entries}


def compare_runs(run_entries_list, run_labels=None):
    """Compare per-episode outcomes across multiple runs.

    Args:
        run_entries_list: list of lists of entry dicts (one per run)
        run_labels: optional list of labels for each run
    Returns:
        dict with comparison results
    """
    n_runs = len(run_entries_list)
    if run_labels is None:
        run_labels = [f"run_{i}" for i in range(n_runs)]

    run_dicts = [outcomes_to_dict(e) for e in run_entries_list]
    # Union of all ep ids
    all_eps = sorted(set().union(*[set(d.keys()) for d in run_dicts]))

    # Per-run place_rate
    place_rates = []
    outcome_counters = []
    for entries in run_entries_list:
        n = len(entries)
        n_placed = sum(1 for e in entries if e["outcome"] == "placed")
        place_rates.append(n_placed / n if n else 0.0)
        outcome_counters.append(Counter(e["outcome"] for e in entries))

    # Find episodes where outcome differs across runs
    mismatch_eps = []
    dist_diff_eps = []
    for ep in all_eps:
        outcomes_for_ep = []
        dists_for_ep = []
        for d in run_dicts:
            if ep in d:
                outcomes_for_ep.append(d[ep]["outcome"])
                dists_for_ep.append(d[ep].get("final_dist_cm"))
            else:
                outcomes_for_ep.append(None)
                dists_for_ep.append(None)
        if len(set(outcomes_for_ep)) > 1:
            mismatch_eps.append({
                "ep": ep,
                "outcomes": dict(zip(run_labels, outcomes_for_ep)),
                "final_dists_cm": dict(zip(run_labels, dists_for_ep)),
            })
        # Also flag episodes where outcome same but dist differs a lot (>2cm)
        elif len(set(outcomes_for_ep)) == 1 and outcomes_for_ep[0] is not None:
            valid_dists = [d for d in dists_for_ep if d is not None]
            if valid_dists and max(valid_dists) - min(valid_dists) > 2.0:
                dist_diff_eps.append({
                    "ep": ep,
                    "outcome": outcomes_for_ep[0],
                    "final_dists_cm": dict(zip(run_labels, dists_for_ep)),
                    "dist_range_cm": max(valid_dists) - min(valid_dists),
                })

    return {
        "n_runs": n_runs,
        "run_labels": run_labels,
        "n_episodes_per_run": [len(e) for e in run_entries_list],
        "place_rates": dict(zip(run_labels, place_rates)),
        "outcome_counts": {label: dict(c) for label, c in zip(run_labels, outcome_counters)},
        "n_outcome_mismatches": len(mismatch_eps),
        "outcome_mismatches": mismatch_eps,
        "n_dist_divergent": len(dist_diff_eps),
        "dist_divergent_eps": dist_diff_eps[:20],  # cap output
    }


def identify_source(comparison, deterministic_comparison=None):
    """Heuristically identify the likely non-determinism source.

    Returns a dict with 'likely_source' and 'evidence'.
    """
    n_mismatch = comparison["n_outcome_mismatches"]
    n_dist_div = comparison["n_dist_divergent"]
    max_pr = max(comparison["place_rates"].values())
    min_pr = min(comparison["place_rates"].values())
    pr_spread_pp = (max_pr - min_pr) * 100

    sources = []

    if n_mismatch == 0 and n_dist_div == 0:
        sources.append({
            "source": "none (fully deterministic)",
            "evidence": "0 outcome mismatches, 0 dist divergences across runs",
            "severity": "none",
        })
    else:
        # Check if deterministic GPU flags fixed it
        if deterministic_comparison is not None:
            det_mismatch = deterministic_comparison["n_outcome_mismatches"]
            if det_mismatch == 0 and n_mismatch > 0:
                sources.append({
                    "source": "GPU non-determinism (cudnn / CUBLAS reduction order)",
                    "evidence": f"{n_mismatch} mismatches without flags → 0 with "
                                f"cudnn.deterministic + use_deterministic_algorithms + "
                                f"CUBLAS_WORKSPACE_CONFIG",
                    "severity": "high" if pr_spread_pp > 2 else "medium",
                    "fix": "Set deterministic GPU flags in evaluate_iql_env.py "
                           "(add torch.backends.cudnn.deterministic=True, "
                           "torch.use_deterministic_algorithms(True), "
                           "torch.backends.cudnn.benchmark=False at top of script; "
                           "export CUBLAS_WORKSPACE_CONFIG=:4096:8)",
                })
            elif det_mismatch < n_mismatch:
                sources.append({
                    "source": "partially GPU non-determinism + residual",
                    "evidence": f"{n_mismatch} → {det_mismatch} mismatches with flags "
                                f"(partial fix; residual likely from env physics solver "
                                f"or MuJoCo contact randomness)",
                    "severity": "medium",
                    "fix": "Apply GPU flags + investigate MuJoCo solver settings "
                           "(solver iterations, contact randomness)",
                })
            else:
                sources.append({
                    "source": "NOT GPU (cudnn flags did not help)",
                    "evidence": f"{n_mismatch} mismatches persist with deterministic flags",
                    "severity": "high",
                    "fix": "Investigate: (1) MuJoCo physics solver non-determinism, "
                           "(2) env.seed() not fully controlling all randomness, "
                           "(3) unseeded python random module",
                })
        else:
            # No deterministic test run — infer from pattern
            if n_dist_div > 0 and n_mismatch == 0:
                sources.append({
                    "source": "likely GPU floating-point reduction order (mild)",
                    "evidence": f"0 outcome mismatches but {n_dist_div} episodes with "
                                f">2cm dist divergence — trajectory micro-divergence "
                                f"not yet flipping outcomes",
                    "severity": "low" if pr_spread_pp < 1 else "medium",
                    "fix": "Run with --deterministic_gpu to confirm; apply cudnn flags",
                })
            else:
                sources.append({
                    "source": "GPU non-determinism (suspected) — run --deterministic_gpu to confirm",
                    "evidence": f"{n_mismatch} outcome mismatches, {n_dist_div} dist divergences, "
                                f"place_rate spread {pr_spread_pp:.1f}pp",
                    "severity": "high" if pr_spread_pp > 2 else "medium",
                    "fix": "Re-run with --deterministic_gpu flag to isolate source",
                })

    return {
        "likely_sources": sources,
        "place_rate_spread_pp": round(pr_spread_pp, 2),
        "n_outcome_mismatches": n_mismatch,
        "n_dist_divergent": n_dist_div,
    }


def run_eval_subprocess(command, env, label, timeout=3600):
    """Run the eval command and return (success, elapsed_s, returncode)."""
    print(f"\n[{label}] Starting eval subprocess...")
    t0 = time.time()
    try:
        result = subprocess.run(
            command, shell=True, env=env, timeout=timeout,
            executable="/bin/bash",
        )
        elapsed = time.time() - t0
        print(f"[{label}] Done in {elapsed:.0f}s, exit={result.returncode}")
        return result.returncode == 0, elapsed, result.returncode
    except subprocess.TimeoutExpired:
        print(f"[{label}] TIMEOUT after {timeout}s")
        return False, timeout, -1


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose non-determinism in IQL environment evaluation")
    parser.add_argument("--config", choices=["v4", "dt_router_v2"], default="v4",
                        help="Config to test (default: v4)")
    parser.add_argument("--n_episodes", type=int, default=200,
                        help="Episodes per run (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed seed for all runs (default: 42)")
    parser.add_argument("--runs", type=int, default=2,
                        help="Number of repeat runs (default: 2)")
    parser.add_argument("--deterministic_gpu", action="store_true",
                        help="Also run one extra run with deterministic GPU flags "
                             "to test if they fix non-determinism")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "nondeterminism_report.json"),
                        help="Output JSON report path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Non-Determinism Diagnostic")
    print("=" * 70)
    print(f"  Config:          {args.config}")
    print(f"  N_episodes:      {args.n_episodes}")
    print(f"  Seed (fixed):    {args.seed}")
    print(f"  Baseline runs:   {args.runs}")
    print(f"  Det-GPU test:    {args.deterministic_gpu}")
    print(f"  Output:          {args.output}")

    # ---- Layer 1: Checkpoint hash ----
    print("\n--- Layer 1: Checkpoint hash ---")
    ckpt_hash = hash_file(V4_CHECKPOINT)
    print(f"  V4 checkpoint: {V4_CHECKPOINT.name}  md5={ckpt_hash}")
    if args.config == "dt_router_v2":
        model_hash = hash_file(V4_MODEL)
        print(f"  DT model v2:   {V4_MODEL.name}  md5={model_hash}")

    # ---- Layer 2-3: Run eval N times, parse, compare ----
    print(f"\n--- Layer 2-3: {args.runs} baseline runs ---")
    run_entries_list = []
    run_labels = []
    run_logs = []
    for i in range(args.runs):
        label = f"baseline_run{i}"
        log_path = OUTPUT_DIR / f"nondet_{args.config}_{label}.log"
        run_logs.append(str(log_path))
        cmd, env = build_eval_command(
            args.config, args.n_episodes, args.seed, log_path,
            deterministic_gpu=False)
        ok, elapsed, rc = run_eval_subprocess(cmd, env, label)
        if not ok:
            print(f"  ERROR: {label} failed (exit {rc}). See {log_path}")
            sys.exit(1)
        entries = parse_log(log_path)
        print(f"  [{label}] parsed {len(entries)} episodes")
        run_entries_list.append(entries)
        run_labels.append(label)

    comparison = compare_runs(run_entries_list, run_labels)

    # ---- Layer 4: Optional deterministic GPU test ----
    det_comparison = None
    det_label = None
    if args.deterministic_gpu:
        print("\n--- Layer 4: Deterministic GPU test run ---")
        det_label = "det_gpu_run0"
        det_log = OUTPUT_DIR / f"nondet_{args.config}_{det_label}.log"
        cmd, env = build_eval_command(
            args.config, args.n_episodes, args.seed, det_log,
            deterministic_gpu=True)
        ok, elapsed, rc = run_eval_subprocess(cmd, env, det_label)
        if not ok:
            print(f"  WARNING: {det_label} failed (exit {rc}). See {det_log}")
            print("  Continuing without deterministic comparison.")
        else:
            det_entries = parse_log(det_log)
            print(f"  [{det_label}] parsed {len(det_entries)} episodes")
            # Compare det run against the first baseline run
            det_comparison = compare_runs(
                [run_entries_list[0], det_entries],
                [run_labels[0], det_label])

    # ---- Source identification ----
    source_analysis = identify_source(comparison, det_comparison)

    # ---- Report ----
    report = {
        "tool": "diagnose_nondeterminism.py",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "config": args.config,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "n_baseline_runs": args.runs,
            "deterministic_gpu_tested": args.deterministic_gpu,
        },
        "layer1_checkpoint_hash": {
            "v4_checkpoint": ckpt_hash,
            "dt_model_v2": hash_file(V4_MODEL) if args.config == "dt_router_v2" else None,
        },
        "layer2_baseline_comparison": comparison,
        "layer4_deterministic_comparison": det_comparison,
        "source_analysis": source_analysis,
        "eval_logs": run_logs + ([str(OUTPUT_DIR / f'nondet_{args.config}_{det_label}.log')]
                                 if args.deterministic_gpu and det_label else []),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Config: {args.config}, N={args.n_episodes}, seed={args.seed}")
    print(f"\n  Place rates per run:")
    for label, pr in comparison["place_rates"].items():
        print(f"    {label}: {pr*100:.1f}%")
    print(f"  Place rate spread: {source_analysis['place_rate_spread_pp']:.2f}pp")

    print(f"\n  Per-run outcome counts:")
    for label, counts in comparison["outcome_counts"].items():
        print(f"    {label}: {counts}")

    print(f"\n  Outcome mismatches: {comparison['n_outcome_mismatches']}")
    if comparison["n_outcome_mismatches"] > 0:
        print(f"  Mismatched episodes (first 10):")
        for m in comparison["outcome_mismatches"][:10]:
            print(f"    ep {m['ep']:3d}: {m['outcomes']}  dists={m['final_dists_cm']}")

    print(f"\n  Dist-divergent episodes (>2cm, same outcome): {comparison['n_dist_divergent']}")
    if det_comparison:
        print(f"\n  Deterministic-GPU run vs baseline_run0:")
        print(f"    mismatches: {det_comparison['n_outcome_mismatches']} "
              f"(baseline had {comparison['n_outcome_mismatches']} between runs)")

    print(f"\n  Likely source(s):")
    for s in source_analysis["likely_sources"]:
        print(f"    [{s['severity'].upper()}] {s['source']}")
        print(f"           evidence: {s['evidence']}")
        if "fix" in s:
            print(f"           fix: {s['fix']}")

    print(f"\n  Full report: {out_path}")
    print(f"  Eval logs: {', '.join(report['eval_logs'])}")


if __name__ == "__main__":
    main()
