#!/usr/bin/env python3
"""Bootstrap power analysis for multi-seed RL evaluation.

Answers: "How many seeds do I need to reliably detect a +X pp effect?"

Uses the ACTUAL paired-diff variance from the completed 10-seed v4 vs
dt_router_v2 experiment as the noise model, then simulates future paired
experiments with an injected effect size to compute statistical power via
Monte Carlo resampling.

Method (per-cell of the N × effect grid):
  1. Bootstrap-resample N diffs (with replacement) from the empirical
     diff distribution (10 observed diffs, std≈3.55pp).
  2. Add the target effect shift to each sampled diff.
  3. Run a paired t-test on the shifted diffs; record p < 0.05.
  4. Repeat B=2000 times; power = fraction of rejections.

This is non-parametric (no normality assumption on the diff distribution)
and uses the REAL noise observed in this project, not a theoretical sigma.

Usage:
    python bootstrap_power_analysis.py
    python bootstrap_power_analysis.py --results <multi_seed_results.json>
"""
import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev

WORKSPACE = Path(__file__).parent
DEFAULT_RESULTS = WORKSPACE / "outputs" / "dt_orchestrator" / "multi_seed_results.json"

# Two-sided t critical values at p=0.05
_T_CRIT_TWO_SIDED_05 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 50: 2.009, 100: 1.984,
}


def t_crit(df):
    if df in _T_CRIT_TWO_SIDED_05:
        return _T_CRIT_TWO_SIDED_05[df]
    keys = sorted(_T_CRIT_TWO_SIDED_05.keys())
    if df < keys[0]:
        return _T_CRIT_TWO_SIDED_05[keys[0]]
    if df > keys[-1]:
        return _T_CRIT_TWO_SIDED_05[keys[-1]]
    return _T_CRIT_TWO_SIDED_05[min(keys, key=lambda k: abs(k - df))]


def paired_t_p05(diffs, tcrit):
    """Return True if paired t-test rejects H0 at p<0.05 (two-sided)."""
    n = len(diffs)
    if n < 2:
        return False
    m = mean(diffs)
    s = stdev(diffs)
    if s == 0:
        return m != 0  # perfect: reject if nonzero
    t = abs(m / (s / math.sqrt(n)))
    return t > tcrit


def simulate_power(empirical_diffs, n_seeds, effect_pp, n_sim=2000, seed=42):
    """Monte Carlo power simulation for a given (N, effect) cell.

    Args:
        empirical_diffs: list of observed diffs (the noise distribution)
        n_seeds: number of seeds in the simulated experiment
        effect_pp: injected effect in percentage points (e.g. 1.0 = +1pp)
        n_sim: number of Monte Carlo simulations
        seed: RNG seed
    Returns:
        power (0..1): fraction of simulations where p < 0.05
    """
    rng = random.Random(seed)
    effect = effect_pp / 100.0  # pp → fraction
    n_emp = len(empirical_diffs)
    tcrit = t_crit(n_seeds - 1)
    rejections = 0
    for _ in range(n_sim):
        # Bootstrap-resample n_seeds diffs from empirical distribution
        sample = [empirical_diffs[rng.randrange(n_emp)] for _ in range(n_seeds)]
        # Inject the effect (shift all diffs by +effect)
        shifted = [d + effect for d in sample]
        if paired_t_p05(shifted, tcrit):
            rejections += 1
    return rejections / n_sim


def find_min_n_for_power(empirical_diffs, effect_pp, target_power=0.8,
                         n_candidates=None, n_sim=2000):
    """Find minimum N achieving target power via grid search."""
    if n_candidates is None:
        n_candidates = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]
    for n in n_candidates:
        pwr = simulate_power(empirical_diffs, n, effect_pp, n_sim=n_sim)
        if pwr >= target_power:
            return n, pwr
    return None, pwr  # last tried


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap power analysis: how many seeds for +X pp?")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS))
    parser.add_argument("--n_sim", type=int, default=2000,
                        help="Monte Carlo simulations per cell")
    parser.add_argument("--target_power", type=float, default=0.8)
    parser.add_argument("--output", type=str,
                        default=str(WORKSPACE / "outputs" / "dt_orchestrator"
                                    / "power_analysis.json"))
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    # Extract empirical diffs (dt_router_v2 - v4) per common seed
    seeds = results.get("seeds", [])
    runs = results.get("runs", {})
    diffs = []
    for s in seeds:
        k1, k2 = f"v4_seed{s}", f"dt_router_v2_seed{s}"
        if k1 in runs and k2 in runs:
            d = runs[k2]["metrics"]["place_rate"] - runs[k1]["metrics"]["place_rate"]
            diffs.append(d)

    if len(diffs) < 2:
        print(f"ERROR: need >= 2 paired diffs, got {len(diffs)}", flush=True)
        return

    diff_mean_pp = mean(diffs) * 100
    diff_std_pp = stdev(diffs) * 100

    print("=" * 70)
    print("Bootstrap Power Analysis")
    print("=" * 70)
    print(f"  Empirical diffs (n={len(diffs)}):")
    print(f"    mean = {diff_mean_pp:+.2f}pp, std = {diff_std_pp:.2f}pp")
    print(f"    values (pp): {[f'{d*100:+.1f}' for d in diffs]}")
    print(f"  Target power: {args.target_power}")
    print(f"  Simulations per cell: {args.n_sim}")

    # Power grid: N × effect
    n_values = [5, 10, 15, 20, 30, 50, 75, 100, 200]
    effects_pp = [0.5, 1.0, 2.0, 3.0, 5.0]

    print("\n" + "-" * 70)
    print("POWER GRID (fraction of simulations with p < 0.05)")
    print("-" * 70)
    n_effect_label = "N \\ effect"
    header = f"{n_effect_label:>12s}"
    for e in effects_pp:
        header += f" | +{e:.1f}pp"
    print(header)
    print("-" * len(header))

    grid = {}
    for n in n_values:
        row = f"{n:>12d}"
        for e in effects_pp:
            pwr = simulate_power(diffs, n, e, n_sim=args.n_sim)
            grid[f"N{n}_eff{e}"] = pwr
            mark = "*" if pwr >= args.target_power else " "
            row += f" | {pwr:.2f}{mark}"
        print(row)
    print(f"\n  (* = power >= {args.target_power})")

    # Minimum N for each effect
    print("\n" + "-" * 70)
    print(f"MINIMUM N FOR power={args.target_power}")
    print("-" * 70)
    min_n_results = {}
    for e in effects_pp:
        min_n, last_pwr = find_min_n_for_power(diffs, e, args.target_power,
                                                n_sim=args.n_sim)
        if min_n is not None:
            print(f"  +{e:.1f}pp → N = {min_n} seeds (power={last_pwr:.2f})")
            min_n_results[f"eff{e}"] = {"min_n": min_n, "power": last_pwr}
        else:
            print(f"  +{e:.1f}pp → N > 200 (last: N=200 power={last_pwr:.2f})")
            min_n_results[f"eff{e}"] = {"min_n": None, "power": last_pwr,
                                         "note": "N > 200 needed"}

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print(f"  Empirical diff std = {diff_std_pp:.2f}pp")
    n_for_1pp = min_n_results.get("eff1.0", {}).get("min_n")
    n_for_2pp = min_n_results.get("eff2.0", {}).get("min_n")
    print(f"  To detect +1.0pp (DT Router expected gain): "
          f"N = {n_for_1pp if n_for_1pp else '>200'} seeds")
    print(f"  To detect +2.0pp (moderate effect): "
          f"N = {n_for_2pp if n_for_2pp else '>200'} seeds")
    print(f"  Current experiment: N = {len(diffs)} seeds")
    # What can current N detect?
    current_power = grid.get(f"N{len(diffs)}_eff2.0", 0)
    print(f"  Current N can detect +2.0pp with power={current_power:.2f} "
          f"({'sufficient' if current_power >= 0.8 else 'INSUFFICIENT'})")

    # Save
    out = {
        "empirical_diffs_pp": [d * 100 for d in diffs],
        "diff_mean_pp": diff_mean_pp,
        "diff_std_pp": diff_std_pp,
        "target_power": args.target_power,
        "n_sim": args.n_sim,
        "power_grid": grid,
        "min_n_for_power": min_n_results,
        "n_values": n_values,
        "effects_pp": effects_pp,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Output: {args.output}")


if __name__ == "__main__":
    main()
