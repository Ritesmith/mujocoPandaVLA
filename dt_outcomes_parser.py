#!/usr/bin/env python3
"""DT Orchestrator Task 2.1: Parse per-episode outcomes from an evaluation log.

Reads an evaluation log (e.g. reachability.log produced by
evaluate_iql_env.py) and extracts per-episode outcome labels compatible with
dt_codebook.py's expected outcomes JSON format.

Log line formats handled:
    Ep   0: place_steps=126, lift=4.8cm, dist=7.1cm, FAIL [ABORT:no_improvement_30steps]
    Ep   2: place_steps= 85, lift=6.2cm, dist=3.6cm, PLACE
    Ep  50: GRASP FAIL (lift=2.8cm)
    Ep  56: place_steps=  2, lift=6.2cm, dist=50.3cm, FAIL [ABORT:drift>0.5m]
    Ep 192: place_steps= 93, lift=6.1cm, dist=4.6cm, PLACE [ABORT:no_improvement_30steps]

Classification rules (priority high → low):
    1. grasp_fail  — line contains "GRASP FAIL"
    2. placed      — line contains "PLACE"
    3. drift       — line contains "FAIL" and (dist > 10cm or ABORT:drift)
    4. near_miss   — line contains "FAIL" and dist <= 10cm

Usage:
    python dt_outcomes_parser.py --log <path_to_log> --output <output_json>
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# 10cm threshold separating near_miss from drift (per SubTask 2.1 spec).
NEAR_MISS_DIST_CM = 10.0

# Ep N: place_steps=X, lift=Ycm, dist=Zcm, PLACE|FAIL [ABORT:reason]
PLACE_FAIL_RE = re.compile(
    r"Ep\s+(\d+):\s+"
    r"place_steps=\s*(\d+),\s+"
    r"lift=([\d.]+)cm,\s+"
    r"dist=([\d.]+)cm,\s+"
    r"(PLACE|FAIL)"
    r"(?:\s+\[ABORT:([^\]]+)\])?"
)

# Ep N: GRASP FAIL (lift=Ycm)
GRASP_FAIL_RE = re.compile(
    r"Ep\s+(\d+):\s+GRASP\s+FAIL(?:\s+\(lift=([\d.]+)cm\))?"
)


def classify(outcome_kw, dist_cm, abort_reason):
    """Classify an episode outcome from keyword, final distance and abort reason.

    Args:
        outcome_kw: "PLACE", "FAIL", or "GRASP FAIL".
        dist_cm: final distance in cm (float or None).
        abort_reason: ABORT reason string or None (e.g. "drift>0.5m").

    Returns:
        one of "placed", "drift", "near_miss", "grasp_fail".
    """
    if outcome_kw == "GRASP FAIL":
        return "grasp_fail"
    if outcome_kw == "PLACE":
        return "placed"
    # FAIL branch
    is_drift_abort = bool(abort_reason and "drift" in abort_reason.lower())
    if is_drift_abort:
        return "drift"
    if dist_cm is not None and dist_cm > NEAR_MISS_DIST_CM:
        return "drift"
    return "near_miss"


def parse_line(line):
    """Parse a single log line. Returns an entry dict or None if no match."""
    m = GRASP_FAIL_RE.search(line)
    if m:
        ep = int(m.group(1))
        return {
            "ep": ep,
            "outcome": "grasp_fail",
            "final_dist_cm": None,
            "best_dist_cm": None,
            "place_steps": None,
        }

    m = PLACE_FAIL_RE.search(line)
    if not m:
        return None
    ep = int(m.group(1))
    place_steps = int(m.group(2))
    # lift = float(m.group(3))  # parsed but not needed in output
    dist_cm = float(m.group(4))
    outcome_kw = m.group(5)
    abort_reason = m.group(6)

    outcome = classify(outcome_kw, dist_cm, abort_reason)
    # The log only reports final distance; best_dist is not available, so use
    # final_dist as a proxy (keeps the value a float so dt_codebook.py's
    # min() tie-break over near_miss configs never compares None vs float).
    return {
        "ep": ep,
        "outcome": outcome,
        "final_dist_cm": dist_cm,
        "best_dist_cm": dist_cm,
        "place_steps": place_steps,
    }


def parse_log(log_path):
    """Parse the log file and return a list of per-episode entry dicts."""
    entries = []
    seen_eps = set()
    with open(log_path) as f:
        for line in f:
            entry = parse_line(line)
            if entry is None:
                continue
            if entry["ep"] in seen_eps:
                print(f"WARNING: duplicate ep {entry['ep']} in log, "
                      f"overwriting previous entry", file=sys.stderr)
            seen_eps.add(entry["ep"])
            entries.append(entry)
    entries.sort(key=lambda e: e["ep"])
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Parse per-episode outcomes from an evaluation log into "
                    "a JSON compatible with dt_codebook.py")
    parser.add_argument("--log", type=str, required=True,
                        help="Path to evaluation log file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path")
    parser.add_argument("--config", type=str, default="warmup_switch",
                        help="Config name recorded in the output JSON "
                             "(default: warmup_switch)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    entries = parse_log(log_path)
    if not entries:
        print(f"ERROR: no episode lines parsed from {log_path}", file=sys.stderr)
        sys.exit(1)

    payload = {
        "source": log_path.name,
        "config": args.config,
        "n_episodes": len(entries),
        "entries": entries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Summary
    dist = Counter(e["outcome"] for e in entries)
    ep_ids = [e["ep"] for e in entries]
    print(f"{'='*60}")
    print(f"Parsed {len(entries)} episodes from {log_path}")
    print(f"{'='*60}")
    print(f"  Source:   {payload['source']}")
    print(f"  Config:   {payload['config']}")
    print(f"  Episodes: {len(entries)} (ep {min(ep_ids)} .. {max(ep_ids)})")
    print(f"  Outcome distribution:")
    for k in ("placed", "near_miss", "drift", "grasp_fail"):
        if dist.get(k):
            print(f"    {k:11s}: {dist[k]:3d} "
                  f"({100*dist[k]/len(entries):.1f}%)")
    print(f"  Output:   {out_path}")


if __name__ == "__main__":
    main()
