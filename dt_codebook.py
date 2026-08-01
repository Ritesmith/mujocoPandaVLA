#!/usr/bin/env python3
"""DT Orchestrator Task 2: Cross-config codebook builder.

Builds the decision-tree training codebook by combining:
  - Features from the warmup config (v4) — extracted by dt_feature_extractor.py
  - Outcomes from ALL config dt_features JSONs (v4, v4.1, ...)

P0 constraint: features come ONLY from v4. Other configs' dt_features JSONs
are read solely for their outcome labels (placed/near_miss/drift) and
auxiliary metrics (place_steps, best_dist).

Label rules (priority high → low):
  1. One config placed, others not        → label that config (weak=false)
  2. Multiple placed                       → label fewest place_steps (weak=false)
  3. All failed, some near_miss some drift → label a near_miss config (weak=false)
  4. All near_miss                         → label best_dist minimum (weak=true, weight=0.5)
  5. All drift                             → exclude (needs_new_config)

Usage:
    python dt_codebook.py \
        --features outputs/dt_orchestrator/v4_extracted_features.json \
        --outcomes v4.1:outputs/dt_orchestrator/v4_1_features.json \
        --output outputs/dt_orchestrator/codebook.json

    # v4 outcomes come from the extracted features JSON itself.
    # --outcomes adds additional configs (v4.1, v4.2, ...) for outcome comparison.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

NEAR_MISS_DIST_CM = 15.0  # 15cm threshold (matches evaluate_iql_env.py NEAR_MISS_DIST)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_outcome_args(outcome_specs):
    """Parse --outcomes args like 'v4.1:path/to/json' into {config: path}."""
    result = {}
    for spec in outcome_specs:
        if ":" not in spec:
            print(f"ERROR: invalid --outcomes spec '{spec}', expected 'config:path'",
                  file=sys.stderr)
            sys.exit(1)
        name, path = spec.split(":", 1)
        result[name.strip()] = path.strip()
    return result


def extract_outcome_map(dt_features_json):
    """Extract per-episode outcome info from a dt_features JSON.

    Returns {ep: {outcome, place_steps, final_dist_cm, best_dist_cm}}.
    """
    entries = dt_features_json.get("entries", [])
    result = {}
    for e in entries:
        ep = e["ep"]
        result[ep] = {
            "outcome": e.get("outcome", "unknown"),
            "place_steps": e.get("place_steps"),
            "final_dist_cm": e.get("final_dist_cm"),
            "best_dist_cm": e.get("best_dist_cm"),
        }
    return result


def apply_label_rules(configs_for_ep):
    """Apply the 5-level label priority to determine optimal_config.

    Args:
        configs_for_ep: dict {config_name: {outcome, place_steps, best_dist_cm, ...}}
                        Only configs that have data for this ep are included.

    Returns:
        (optimal_config, label_reason, weak_label, exclude)
        exclude=True means this ep should be labeled "needs_new_config".
    """
    placed_configs = [c for c, d in configs_for_ep.items()
                      if d["outcome"] == "placed"]
    near_miss_configs = [c for c, d in configs_for_ep.items()
                         if d["outcome"] == "near_miss"]
    drift_configs = [c for c, d in configs_for_ep.items()
                     if d["outcome"] == "drift"]

    # Rule 1: exactly one placed
    if len(placed_configs) == 1:
        c = placed_configs[0]
        others = [k for k in configs_for_ep if k != c]
        return c, f"{c} placed, others not placed ({','.join(others)})", False, False

    # Rule 2: multiple placed → fewest place_steps
    if len(placed_configs) >= 2:
        best = min(placed_configs,
                   key=lambda c: configs_for_ep[c].get("place_steps", 9999))
        steps = configs_for_ep[best].get("place_steps", "?")
        return best, f"multiple placed, {best} fewest steps ({steps})", False, False

    # No config placed
    # Rule 3: some near_miss, some drift → label a near_miss config
    if near_miss_configs and drift_configs:
        # If multiple near_miss, pick best_dist minimum
        best = min(near_miss_configs,
                   key=lambda c: configs_for_ep[c].get("best_dist_cm", 9999))
        return best, f"{best} near_miss, others drift", False, False

    # Rule 4: all near_miss → best_dist minimum, weak_label
    if near_miss_configs and not drift_configs and not placed_configs:
        best = min(near_miss_configs,
                   key=lambda c: configs_for_ep[c].get("best_dist_cm", 9999))
        bd = configs_for_ep[best].get("best_dist_cm", "?")
        return best, f"all near_miss, {best} best_dist ({bd}cm)", True, False

    # Rule 5: all drift → exclude
    if drift_configs and not near_miss_configs and not placed_configs:
        return None, "all drift — needs new config", False, True

    # Fallback (shouldn't happen with valid data)
    return None, "unclassified", False, True


def main():
    parser = argparse.ArgumentParser(
        description="DT Orchestrator: build cross-config codebook")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to extracted features JSON (from "
                             "dt_feature_extractor.py). Contains v4 features "
                             "+ v4 outcome per episode.")
    parser.add_argument("--outcomes", type=str, nargs="*", default=[],
                        help="Additional config outcome JSONs in 'name:path' "
                             "format. e.g. v4.1:outputs/dt_orchestrator/v4_1_features.json")
    parser.add_argument("--output", type=str, required=True,
                        help="Output codebook JSON path")
    parser.add_argument("--warmup_steps", type=int, default=20,
                        help="Warmup steps used in online routing (metadata)")
    parser.add_argument("--version", type=str, default="1.0",
                        help="Codebook version string (default: 1.0). "
                             "Use 2.0 when labels come from warmup_switch.")
    args = parser.parse_args()

    # Load extracted features (v4)
    feat_data = load_json(args.features)
    feature_source = feat_data.get("config", "v4")
    feature_entries = {e["ep"]: e for e in feat_data.get("entries", [])}
    feature_names = feat_data.get("feature_names", [])
    print(f"Loaded {len(feature_entries)} feature entries from {args.features}")
    print(f"  Feature source: {feature_source}")
    print(f"  Features: {feature_names}")

    # Load additional outcome JSONs
    outcome_maps = {}  # {config_name: {ep: outcome_info}}
    config_set = [feature_source]  # v4 is always included
    for name, path in parse_outcome_args(args.outcomes).items():
        data = load_json(path)
        outcome_maps[name] = extract_outcome_map(data)
        config_set.append(name)
        print(f"  Outcomes for {name}: {len(outcome_maps[name])} episodes from {path}")

    # v4 outcome map comes from the features JSON itself
    v4_outcome_map = {}
    for ep, e in feature_entries.items():
        v4_outcome_map[ep] = {
            "outcome": e.get("outcome", "unknown"),
            "place_steps": e.get("place_steps"),
            "final_dist_cm": e.get("final_dist_cm"),
            "best_dist_cm": e.get("best_dist_cm"),
        }
    outcome_maps[feature_source] = v4_outcome_map

    # Build codebook entries
    all_eps = sorted(feature_entries.keys())
    codebook_entries = []
    stats = {
        "v4_only_win": 0,       # v4 placed, v4.1 not
        "v4_1_only_win": 0,     # v4.1 placed, v4 not
        "both_win": 0,          # both placed
        "both_lose": 0,         # both failed
        "weak_label": 0,        # rule 4 (all near_miss)
        "needs_new_config": 0,  # rule 5 (all drift)
    }

    for ep in all_eps:
        feat_entry = feature_entries[ep]
        features = feat_entry["features"]

        # Gather outcomes for this ep across all configs
        configs_for_ep = {}
        for cfg in config_set:
            if ep in outcome_maps.get(cfg, {}):
                configs_for_ep[cfg] = outcome_maps[cfg][ep]

        if len(configs_for_ep) < 2:
            # Only one config has data (shouldn't happen with same seed)
            continue

        # Apply label rules
        optimal, reason, weak, exclude = apply_label_rules(configs_for_ep)

        # Build outcomes dict for codebook
        outcomes = {cfg: d["outcome"] for cfg, d in configs_for_ep.items()}
        place_steps_dict = {cfg: d.get("place_steps") for cfg, d in configs_for_ep.items()}
        best_dists_dict = {cfg: d.get("best_dist_cm") for cfg, d in configs_for_ep.items()}

        # Update stats
        v4_placed = outcomes.get(feature_source) == "placed"
        v41_placed = any(outcomes.get(c) == "placed" for c in config_set
                         if c != feature_source)
        if v4_placed and v41_placed:
            stats["both_win"] += 1
        elif v4_placed and not v41_placed:
            stats["v4_only_win"] += 1
        elif not v4_placed and v41_placed:
            stats["v4_1_only_win"] += 1
        else:
            stats["both_lose"] += 1

        if weak:
            stats["weak_label"] += 1
        if exclude:
            stats["needs_new_config"] += 1

        if exclude:
            entry = {
                "ep": ep,
                "features": features,
                "outcomes": outcomes,
                "place_steps": place_steps_dict,
                "best_dists": best_dists_dict,
                "optimal_config": None,
                "label_reason": reason,
                "weak_label": False,
                "exclude": True,
            }
        else:
            entry = {
                "ep": ep,
                "features": features,
                "outcomes": outcomes,
                "place_steps": place_steps_dict,
                "best_dists": best_dists_dict,
                "optimal_config": optimal,
                "label_reason": reason,
                "weak_label": weak,
                "exclude": False,
            }
        codebook_entries.append(entry)

    # Build codebook with version metadata (P1-4.5)
    codebook = {
        "version": args.version,
        "config_set": config_set,
        "feature_source": feature_source,
        "warmup_steps": args.warmup_steps,
        "created": str(date.today()),
        "feature_names": feature_names,
        "n_entries": len(codebook_entries),
        "label_stats": stats,
        "entries": codebook_entries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(codebook, f, indent=2, default=str)

    # Print summary
    trainable = [e for e in codebook_entries if not e.get("exclude")]
    print(f"\n{'='*60}")
    print(f"Codebook built")
    print(f"{'='*60}")
    print(f"  Version:          {codebook['version']}")
    print(f"  Config set:       {config_set}")
    print(f"  Feature source:   {feature_source}")
    print(f"  Total entries:    {len(codebook_entries)}")
    print(f"  Trainable:        {len(trainable)} (excluded {stats['needs_new_config']})")
    print(f"\n  Label statistics:")
    print(f"    v4-only-win:       {stats['v4_only_win']}")
    print(f"    v4.1-only-win:     {stats['v4_1_only_win']}")
    print(f"    both-win:          {stats['both_win']}")
    print(f"    both-lose:         {stats['both_lose']}")
    print(f"    weak-label:        {stats['weak_label']}")
    print(f"    needs-new-config:  {stats['needs_new_config']}")

    # Label distribution
    if trainable:
        from collections import Counter
        label_dist = Counter(e["optimal_config"] for e in trainable)
        print(f"\n  Optimal config distribution (trainable):")
        for cfg, count in sorted(label_dist.items()):
            print(f"    {cfg:10s}: {count:3d} ({100*count/len(trainable):.1f}%)")

    print(f"\n  Output: {out_path}")


if __name__ == "__main__":
    main()
