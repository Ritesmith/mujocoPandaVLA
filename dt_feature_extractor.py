#!/usr/bin/env python3
"""DT Orchestrator Task 1: Per-episode feature extraction from v4 warmup JSON.

Reads the per-episode early-distance JSON produced by evaluate_iql_env.py
--dt_features_path (which records the first 20 place-step block-target
distances for EVERY episode that entered place phase) and computes the 7-dim
feature vector defined in the DT Orchestrator spec.

P0 constraint: features MUST come from the warmup config (v4) JSON, NOT v4.1.
This script only processes v4 (or any single warmup config) features.

Usage:
    python dt_feature_extractor.py \
        --features_json outputs/dt_orchestrator/v4_features.json \
        --config v4 \
        --output outputs/dt_orchestrator/v4_extracted_features.json

Output format:
    {
      "version": "1.0",
      "config": "v4",
      "feature_source": "v4",
      "n_episodes": 200,
      "feature_names": [7 names],
      "entries": [
        {
          "ep": 0,
          "config": "v4",
          "features": {
            "dist_at_step20": 5.2,
            "dist_change_rate": 0.94,
            "dist_variance_early": 45.3,
            "early_drift_signal": 0,
            "q1_at_step20": 123.45,
            "best_dist_early": 3.9,
            "has_q_value": 1
          },
          "outcome": "placed",
          "place_steps": 45,
          "final_dist_cm": 3.2,
          "best_dist_cm": 2.1
        },
        ...
      ]
    }
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "dist_at_step20",
    "dist_change_rate",
    "dist_variance_early",
    "early_drift_signal",
    "q1_at_step20",
    "best_dist_early",
    "has_q_value",
]


def compute_features(entry):
    """Compute the 7-dim feature vector from a single dt_features entry.

    Args:
        entry: dict with keys ep, early_dists (list of cm), q1_at_step20
               (float or None), outcome, place_steps, final_dist_cm,
               best_dist_cm.

    Returns:
        dict of 7 features, or None if early_dists is empty (grasp fail).
    """
    early_dists = entry.get("early_dists", [])
    if not early_dists:
        return None  # episode didn't enter place phase (grasp failure)

    dists = np.array(early_dists, dtype=float)
    n = len(dists)

    # 1. dist_at_step20: distance at step 20 (or last available step)
    dist_at_step20 = float(dists[-1])  # last available = step 20 if n>=20

    # 2. dist_change_rate: (first - last) / n, positive = approaching target
    dist_change_rate = float((dists[0] - dists[-1]) / n)

    # 3. dist_variance_early: variance of early distances
    dist_variance_early = float(np.var(dists))

    # 4. early_drift_signal: 1 if moving away from target (rate < 0)
    early_drift_signal = int(dist_change_rate < 0)

    # 5-7. Q-value features with imputation
    q1_raw = entry.get("q1_at_step20")
    if q1_raw is not None:
        q1_at_step20 = float(q1_raw)
        has_q_value = 1
    else:
        q1_at_step20 = 0.0  # imputation: fill 0.0
        has_q_value = 0

    # 6. best_dist_early: minimum distance in the first 20 steps
    best_dist_early = float(np.min(dists))

    return {
        "dist_at_step20": round(dist_at_step20, 4),
        "dist_change_rate": round(dist_change_rate, 6),
        "dist_variance_early": round(dist_variance_early, 4),
        "early_drift_signal": early_drift_signal,
        "q1_at_step20": round(q1_at_step20, 4),
        "best_dist_early": round(best_dist_early, 4),
        "has_q_value": has_q_value,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DT Orchestrator: extract 7-dim features from v4 warmup JSON")
    parser.add_argument("--features_json", type=str, required=True,
                        help="Path to the dt_features JSON produced by "
                             "evaluate_iql_env.py --dt_features_path")
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (e.g. v4). Must match the warmup config.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for extracted features JSON")
    args = parser.parse_args()

    # Load the dt_features JSON
    features_path = Path(args.features_json)
    if not features_path.exists():
        print(f"ERROR: features JSON not found: {features_path}", file=sys.stderr)
        sys.exit(1)

    with open(features_path) as f:
        data = json.load(f)

    entries = data.get("entries", [])
    print(f"Loaded {len(entries)} episodes from {features_path}")
    print(f"  warmup_switch: {data.get('warmup_switch')}")
    print(f"  log_q_values: {data.get('log_q_values')}")
    print(f"  chunk_size: {data.get('chunk_size')}")

    # Compute features for each episode
    extracted = []
    skipped = 0
    outcome_counts = {"placed": 0, "near_miss": 0, "drift": 0}

    for entry in entries:
        features = compute_features(entry)
        if features is None:
            skipped += 1
            continue

        outcome = entry.get("outcome", "unknown")
        if outcome in outcome_counts:
            outcome_counts[outcome] += 1

        extracted.append({
            "ep": entry["ep"],
            "config": args.config,
            "features": features,
            "outcome": outcome,
            "place_steps": entry.get("place_steps"),
            "final_dist_cm": entry.get("final_dist_cm"),
            "best_dist_cm": entry.get("best_dist_cm"),
        })

    # Build output
    output = {
        "version": "1.0",
        "config": args.config,
        "feature_source": args.config,
        "n_episodes": len(extracted),
        "n_skipped": skipped,
        "feature_names": FEATURE_NAMES,
        "outcome_distribution": outcome_counts,
        "entries": extracted,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Feature extraction complete")
    print(f"{'='*60}")
    print(f"  Config:            {args.config}")
    print(f"  Episodes:          {len(extracted)} (skipped {skipped} grasp fails)")
    print(f"  Outcome dist:      {outcome_counts}")
    print(f"  Q-value coverage:  {sum(1 for e in extracted if e['features']['has_q_value'])}/{len(extracted)}")
    print(f"  Output:            {out_path}")

    # Quick feature statistics
    if extracted:
        feat_arr = np.array([[e["features"][n] for n in FEATURE_NAMES]
                             for e in extracted])
        print(f"\n  Feature statistics (mean / std / min / max):")
        for i, name in enumerate(FEATURE_NAMES):
            col = feat_arr[:, i]
            print(f"    {name:25s}: {col.mean():10.4f} / {col.std():10.4f} "
                  f"/ {col.min():10.4f} / {col.max():10.4f}")


if __name__ == "__main__":
    main()
