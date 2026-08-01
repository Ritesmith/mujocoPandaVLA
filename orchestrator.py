#!/usr/bin/env python3
"""Experiment Orchestrator: automated decision-tree branch selection.

Reads:
  1. experiment_registry.yaml (pre-registered hypothesis + decision tree)
  2. evaluation results JSON (actual metrics)

Matches actual metrics against the pre-registered decision tree conditions
(top-to-bottom, first match wins), outputs the selected branch, and updates
the YAML with actual_result + selected_branch.

This implements the user's L1 automation level:
  "注册表+决策树人工填写，orchestrator 自动匹配分支并生成下一实验配置"

Usage:
    python orchestrator.py \\
        --registry outputs/iql_v4_1_adaptive/experiment_registry.yaml \\
        --results outputs/iql_v4_1_adaptive/env_eval_n200.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


def extract_metrics(results_path: str) -> dict:
    """Extract key metrics from evaluation results JSON."""
    with open(results_path) as f:
        r = json.load(f)

    place_rate = r.get("place_rate", 0.0)
    failure_modes = r.get("failure_modes", {})
    drift_abs = failure_modes.get("drift", 0)
    near_miss_abs = failure_modes.get("near_miss", 0)
    n_placed = r.get("n_placed", 0)
    n_entered = r.get("n_entered_place", 0)
    n_failed = n_entered - n_placed

    # Adaptive chunk stats (if present)
    adaptive_stats = r.get("adaptive_chunk_stats", {})
    chunk_steps = adaptive_stats.get("total_steps_per_chunk_size", {})
    n_switches = adaptive_stats.get("n_switch_events", 0)

    # Q-value diagnostics (if present)
    q_diag = r.get("q_value_diagnostics", {})
    q_at_best = q_diag.get("q_at_best_dist", [])

    return {
        "place_rate": place_rate,
        "place_rate_pct": round(place_rate * 100, 1),
        "drift_abs": drift_abs,
        "near_miss_abs": near_miss_abs,
        "n_placed": n_placed,
        "n_entered": n_entered,
        "n_failed": n_failed,
        "drift_pct_of_failures": round(100 * drift_abs / max(1, n_failed), 1),
        "near_miss_pct_of_failures": round(100 * near_miss_abs / max(1, n_failed), 1),
        "chunk_steps": chunk_steps,
        "n_switch_events": n_switches,
        "n_q_logged": q_diag.get("n_q_logged", 0),
    }


def evaluate_condition(condition: str, metrics: dict) -> bool:
    """Evaluate a decision tree condition string against actual metrics.

    Supports simple boolean expressions with metrics as variables:
        "place_rate >= 0.78 AND drift_abs <= 10"
        "place_rate < 0.72"
        "drift_abs >= 18 AND place_rate >= 0.70"
        "DEFAULT"
    """
    if condition.strip() == "DEFAULT":
        return True

    # Replace metric names with values
    expr = condition
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            expr = re.sub(rf'\b{key}\b', str(val), expr)

    # Replace AND/OR with Python operators
    expr = re.sub(r'\bAND\b', 'and', expr, flags=re.IGNORECASE)
    expr = re.sub(r'\bOR\b', 'or', expr, flags=re.IGNORECASE)

    try:
        result = eval(expr)
        return bool(result)
    except Exception as e:
        print(f"  WARNING: Failed to evaluate condition '{condition}': {e}")
        print(f"    Transformed expr: {expr}")
        return False


def match_branch(decision_tree: list, metrics: dict) -> dict:
    """Match metrics against decision tree. First match wins."""
    for branch in decision_tree:
        condition = branch.get("condition", "DEFAULT")
        if evaluate_condition(condition, metrics):
            return branch
    return {"branch_id": "NONE", "next_exp": "manual_review",
            "rationale": "No branch matched", "action": "halt_for_review"}


def check_expected_vs_actual(expected: dict, metrics: dict) -> dict:
    """Compare actual metrics against pre-registered expected intervals."""
    comparison = {}
    for key, exp in expected.items():
        if isinstance(exp, dict) and "min" in exp and "max" in exp:
            actual = metrics.get(key)
            if actual is not None:
                in_range = exp["min"] <= actual <= exp["max"]
                comparison[key] = {
                    "expected_min": exp["min"],
                    "expected_max": exp["max"],
                    "actual": actual,
                    "in_range": in_range,
                }
    return comparison


def run_orchestrator(registry_path: str, results_path: str,
                     update_yaml: bool = True) -> dict:
    """Main orchestrator: match results to decision tree, output selected branch."""
    print("=" * 70)
    print("EXPERIMENT ORCHESTRATOR")
    print("=" * 70)

    # Load registry
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    print(f"Registry: {registry_path}")
    print(f"  exp_id: {registry.get('exp_id', '?')}")
    print(f"  hypothesis: {registry.get('hypothesis', '?')[:80]}...")

    # Extract metrics
    metrics = extract_metrics(results_path)
    print(f"\nResults: {results_path}")
    print(f"  place_rate:    {metrics['place_rate_pct']}% "
          f"({metrics['n_placed']}/{metrics['n_entered']})")
    print(f"  drift_abs:     {metrics['drift_abs']} "
          f"({metrics['drift_pct_of_failures']}% of failures)")
    print(f"  near_miss_abs: {metrics['near_miss_abs']} "
          f"({metrics['near_miss_pct_of_failures']}% of failures)")
    if metrics["chunk_steps"]:
        print(f"  chunk_steps:   {metrics['chunk_steps']}")
        print(f"  switch_events: {metrics['n_switch_events']}")
    print()

    # Check expected vs actual
    expected = registry.get("expected_metrics", {})
    if expected:
        print("Expected vs Actual:")
        comparison = check_expected_vs_actual(expected, metrics)
        for key, cmp in comparison.items():
            status = "✓ IN RANGE" if cmp["in_range"] else "✗ OUTSIDE"
            print(f"  {key}: expected [{cmp['expected_min']}, {cmp['expected_max']}], "
                  f"actual {cmp['actual']} → {status}")
        print()

    # Match decision tree
    decision_tree = registry.get("decision_tree", [])
    selected = match_branch(decision_tree, metrics)

    print("=" * 70)
    print("SELECTED BRANCH")
    print("=" * 70)
    print(f"  branch_id:   {selected.get('branch_id', '?')}")
    print(f"  condition:   {selected.get('condition', '?')}")
    print(f"  next_exp:    {selected.get('next_exp', '?')}")
    print(f"  action:      {selected.get('action', '?')}")
    print(f"  rationale:   {selected.get('rationale', '?')[:120]}...")
    print()

    # Prepare actual_result for YAML update
    actual_result = {
        "place_rate": metrics["place_rate"],
        "place_rate_pct": metrics["place_rate_pct"],
        "drift_abs": metrics["drift_abs"],
        "near_miss_abs": metrics["near_miss_abs"],
        "n_placed": metrics["n_placed"],
        "n_entered": metrics["n_entered"],
        "n_failed": metrics["n_failed"],
        "drift_pct_of_failures": metrics["drift_pct_of_failures"],
        "near_miss_pct_of_failures": metrics["near_miss_pct_of_failures"],
        "chunk_steps": metrics["chunk_steps"],
        "n_switch_events": metrics["n_switch_events"],
        "n_q_logged": metrics["n_q_logged"],
        "expected_vs_actual": comparison if expected else {},
    }

    # Update YAML
    if update_yaml:
        registry["actual_result"] = actual_result
        registry["selected_branch"] = {
            "branch_id": selected.get("branch_id"),
            "condition": selected.get("condition"),
            "next_exp": selected.get("next_exp"),
            "action": selected.get("action"),
            "rationale": selected.get("rationale"),
        }
        with open(registry_path, "w") as f:
            yaml.dump(registry, f, default_flow_style=False, allow_unicode=True,
                      sort_keys=False)
        print(f"Registry updated: {registry_path}")
        print(f"  actual_result + selected_branch written to YAML")

    return {
        "metrics": metrics,
        "selected_branch": selected,
        "actual_result": actual_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment orchestrator")
    parser.add_argument("--registry", type=str, required=True,
                        help="Path to experiment_registry.yaml")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to evaluation results JSON")
    parser.add_argument("--no-update", action="store_true",
                        help="Don't update the YAML file (dry run)")
    args = parser.parse_args()

    result = run_orchestrator(
        registry_path=args.registry,
        results_path=args.results,
        update_yaml=not args.no_update,
    )

    # Exit code: 0 if branch found, 1 if fallback
    branch_id = result["selected_branch"].get("branch_id", "")
    if branch_id in ("E", "NONE"):
        sys.exit(1)
    sys.exit(0)
