#!/usr/bin/env python3
"""Quick overlap analysis: compute naive oracle bound and DT router upper bound.

Compares per-episode outcomes across v4, v4.1, and warmup_switch to determine:
1. naive_oracle_bound = (v4_placed ∪ v4.1_placed) / N  (unreachable, no warmup damage)
2. dt_router_upper_bound = (v4_placed ∪ warmup_switch_placed) / N  (reachable with perfect DT)
3. v4_only_win, v4.1_only_win, both_win, both_lose counts
"""
import json
import sys
from pathlib import Path


def load_outcomes(path):
    """Load per-episode outcomes from a dt_features JSON file."""
    with open(path) as f:
        data = json.load(f)
    outcomes = {}
    for e in data.get("entries", []):
        outcomes[e["ep"]] = e.get("outcome", "unknown")
    return outcomes


def main():
    base = Path("/home/w/vla_workspace/outputs/dt_orchestrator")
    
    v4_outcomes = load_outcomes(base / "v4_features.json")
    v41_outcomes = load_outcomes(base / "v4_1_features.json")
    
    # warmup_switch features may not exist yet
    ws_path = base / "warmup_switch_features.json"
    ws_outcomes = load_outcomes(ws_path) if ws_path.exists() else {}
    
    print("=" * 60)
    print("Overlap Analysis")
    print("=" * 60)
    print(f"  v4 episodes:              {len(v4_outcomes)}")
    print(f"  v4.1 episodes:            {len(v41_outcomes)}")
    print(f"  warmup_switch episodes:   {len(ws_outcomes)}")
    
    # v4 vs v4.1 overlap (naive oracle bound)
    all_eps = sorted(set(v4_outcomes.keys()) | set(v41_outcomes.keys()))
    v4_placed = {ep for ep, o in v4_outcomes.items() if o == "placed"}
    v41_placed = {ep for ep, o in v41_outcomes.items() if o == "placed"}
    
    both_placed = v4_placed & v41_placed
    v4_only = v4_placed - v41_placed
    v41_only = v41_placed - v4_placed
    neither = set(all_eps) - v4_placed - v41_placed
    
    naive_oracle = len(v4_placed | v41_placed)
    
    print(f"\n  v4 vs v4.1 (naive oracle, IGNORES warmup damage):")
    print(f"    v4 placed:          {len(v4_placed)}")
    print(f"    v4.1 placed:        {len(v41_placed)}")
    print(f"    both placed:        {len(both_placed)}")
    print(f"    v4-only-win:        {len(v4_only)}")
    print(f"    v4.1-only-win:      {len(v41_only)}")
    print(f"    both-lose:          {len(neither)}")
    print(f"    naive_oracle_bound: {naive_oracle}/{len(all_eps)} = {100*naive_oracle/len(all_eps):.1f}%")
    
    # DT router upper bound (with warmup_switch data)
    if ws_outcomes:
        ws_placed = {ep for ep, o in ws_outcomes.items() if o == "placed"}
        dt_upper = len(v4_placed | ws_placed)
        ws_only = ws_placed - v4_placed  # episodes warmup_switch wins but v4 loses
        
        print(f"\n  v4 vs warmup_switch (DT router reachable bound):")
        print(f"    v4 placed:              {len(v4_placed)}")
        print(f"    warmup_switch placed:   {len(ws_placed)}")
        print(f"    both placed:            {len(v4_placed & ws_placed)}")
        print(f"    v4-only-win:            {len(v4_placed - ws_placed)}")
        print(f"    ws-only-win:            {len(ws_only)}")
        print(f"    both-lose:              {len(set(all_eps) - v4_placed - ws_placed)}")
        print(f"    DT_upper_bound:         {dt_upper}/{len(all_eps)} = {100*dt_upper/len(all_eps):.1f}%")
        print(f"    v4 baseline:            {len(v4_placed)}/{len(v4_outcomes)} = {100*len(v4_placed)/len(v4_outcomes):.1f}%")
        
        if dt_upper > len(v4_placed):
            print(f"\n  → DT router CAN potentially beat v4 by {dt_upper - len(v4_placed)} episodes")
            print(f"    (if it perfectly identifies the {len(ws_only)} ws-only-win episodes)")
        else:
            print(f"\n  → DT router CANNOT beat v4 (warmup_switch wins are subset of v4 wins)")
        
        # Also compute v4.1 vs warmup_switch overlap
        v41_ws_both = v41_placed & ws_placed
        v41_only_placed = v41_placed - ws_placed
        ws_only_placed = ws_placed - v41_placed
        print(f"\n  v4.1 vs warmup_switch (warmup damage analysis):")
        print(f"    v4.1 placed:            {len(v41_placed)}")
        print(f"    warmup_switch placed:   {len(ws_placed)}")
        print(f"    both placed:            {len(v41_ws_both)}")
        print(f"    v4.1-only (damaged):    {len(v41_only_placed)} ← warmup killed these")
        print(f"    ws-only (rescued):      {len(ws_only_placed)} ← warmup helped these")
    else:
        print(f"\n  warmup_switch per-episode data not yet available.")
        print(f"  Re-running reachability with --dt_features_path...")
    
    print(f"\n  Summary:")
    print(f"    v4 baseline:             71.9% (143/199)")
    print(f"    v4.1 baseline:           64.5% (129/200)")
    print(f"    warmup_switch:           65.5% (131/200) ← P0 GATE FAILED")
    print(f"    naive_oracle_bound:      {100*naive_oracle/len(all_eps):.1f}% (unreachable)")


if __name__ == "__main__":
    main()
