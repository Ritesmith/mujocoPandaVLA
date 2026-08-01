#!/usr/bin/env python3
"""Phase 8 三节点监测分析.

Pre-registered monitoring nodes (from project_memory):
  1. Router 置信度分布 — V_CV=30% 是否导致 DT 阈值分布剧烈漂移
     (proxy: fallback_rate 跨 seeds 的稳定性; 低 fallback_rate = 高置信度)
  2. 路由决策一致性 — 不同 seed 训练的 Router 在"何时切换"上是否分歧
     (proxy: n_switched 跨 seeds 的 CV; 低 CV = 高一致性)
  3. 端到端 place_rate CV — 若 ≤ 8%, V_CV=30% 可被系统吸收 (最终试金石)

Usage:
    python phase8_3node_monitor.py --results outputs/phase8_dt_router_v3/n30_results.json
    python phase8_3node_monitor.py --results outputs/phase8_dt_router_v3/pilot_results.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, stdev

WORKSPACE = Path(__file__).parent


def parse_routing_stats_from_log(log_path):
    """Extract routing stats from evaluate_iql_env.py log.

    Parses lines like:
      Switched to v4.1:  49
      Fallbacks:         96 (48.0%)
    """
    if not Path(log_path).exists():
        return None
    text = Path(log_path).read_text()
    switched = None
    fallback_count = None
    fallback_rate = None
    m = re.search(r"Switched to v4\.1:\s*(\d+)", text)
    if m:
        switched = int(m.group(1))
    m = re.search(r"Fallbacks:\s*(\d+)\s*\(([\d.]+)%\)", text)
    if m:
        fallback_count = int(m.group(1))
        fallback_rate = float(m.group(2)) / 100.0
    if switched is None and fallback_count is None:
        return None
    return {
        "n_switched_to_v4_1": switched,
        "n_fallback": fallback_count,
        "fallback_rate": fallback_rate,
    }


def analyze_node1_confidence(routing_stats_list):
    """Node 1: Router 置信度分布 (via fallback_rate).

    High fallback_rate = low confidence. CV of fallback_rate across seeds
    indicates whether V_CV=30% causes confidence drift.
    """
    frs = [r["fallback_rate"] for r in routing_stats_list
           if r and r.get("fallback_rate") is not None]
    if not frs:
        return {"status": "no_data"}
    fr_mean = mean(frs)
    fr_std = stdev(frs) if len(frs) > 1 else 0
    fr_cv = fr_std / fr_mean * 100 if fr_mean > 0 else 0
    # Threshold: fallback_rate CV > 30% = confidence drift (V_CV impact)
    status = "PASS" if fr_cv < 30 else "FAIL"
    return {
        "n_seeds": len(frs),
        "fallback_rate_mean": round(fr_mean, 4),
        "fallback_rate_std": round(fr_std, 4),
        "fallback_rate_cv": round(fr_cv, 2),
        "fallback_rates": [round(fr, 4) for fr in frs],
        "status": status,
        "interpretation": (
            f"fallback_rate CV={fr_cv:.1f}% (< 30% threshold) → "
            f"Router 置信度跨 seeds 稳定, V_CV=30% 未导致置信度漂移"
            if status == "PASS"
            else f"fallback_rate CV={fr_cv:.1f}% (≥ 30% threshold) → "
            f"Router 置信度跨 seeds 不稳定, V_CV=30% 可能影响置信度"
        ),
    }


def analyze_node2_consistency(routing_stats_list, n_episodes=200):
    """Node 2: 路由决策一致性 (via n_switched CV).

    Low CV of n_switched across seeds = high decision consistency.
    """
    nsw = [r["n_switched_to_v4_1"] for r in routing_stats_list
           if r and r.get("n_switched_to_v4_1") is not None]
    if not nsw:
        return {"status": "no_data"}
    ns_mean = mean(nsw)
    ns_std = stdev(nsw) if len(nsw) > 1 else 0
    ns_cv = ns_std / ns_mean * 100 if ns_mean > 0 else 0
    switch_rate_mean = ns_mean / n_episodes
    # Threshold: n_switched CV > 40% = inconsistent routing
    status = "PASS" if ns_cv < 40 else "FAIL"
    return {
        "n_seeds": len(nsw),
        "n_switched_mean": round(ns_mean, 1),
        "n_switched_std": round(ns_std, 2),
        "n_switched_cv": round(ns_cv, 2),
        "switch_rate_mean": round(switch_rate_mean, 4),
        "n_switched_list": nsw,
        "status": status,
        "interpretation": (
            f"n_switched CV={ns_cv:.1f}% (< 40% threshold) → "
            f"路由决策跨 seeds 一致, Router 对'何时切换'有稳定判断"
            if status == "PASS"
            else f"n_switched CV={ns_cv:.1f}% (≥ 40% threshold) → "
            f"路由决策跨 seeds 不一致, Router 判断受 seed 影响"
        ),
    }


def analyze_node3_e2e_cv(results, config_name="dt_router_v3"):
    """Node 3: 端到端 place_rate CV (最终试金石).

    If CV ≤ 8%, V_CV=30% is absorbed by system robustness.
    """
    runs = results.get("runs", {})
    seeds = results.get("seeds", [])
    place_rates = []
    for s in seeds:
        key = f"{config_name}_seed{s}"
        if key in runs:
            pr = runs[key]["metrics"]["place_rate"]
            place_rates.append(pr)
    if not place_rates:
        return {"status": "no_data"}
    pr_mean = mean(place_rates)
    pr_std = stdev(place_rates) if len(place_rates) > 1 else 0
    pr_cv = pr_std / pr_mean * 100 if pr_mean > 0 else 0
    # Threshold: CV ≤ 8% (from project_memory hard constraint)
    status = "PASS" if pr_cv <= 8.0 else "FAIL"
    return {
        "n_seeds": len(place_rates),
        "place_rate_mean": round(pr_mean * 100, 2),
        "place_rate_std_pp": round(pr_std * 100, 2),
        "place_rate_cv": round(pr_cv, 2),
        "place_rates": [round(pr * 100, 1) for pr in place_rates],
        "threshold_cv": 8.0,
        "status": status,
        "interpretation": (
            f"端到端 CV={pr_cv:.2f}% (≤ 8% threshold) → "
            f"V_CV=30% 被系统鲁棒性吸收, 无需回 Round 2c 优化 V"
            if status == "PASS"
            else f"端到端 CV={pr_cv:.2f}% (> 8% threshold) → "
            f"V_CV=30% 影响端到端稳定性, 需回 Round 2c 启动 V 预训练"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8 三节点监测分析")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to n30_results.json or pilot_results.json")
    parser.add_argument("--config", type=str, default="dt_router_v3",
                        help="Treatment config name (default: dt_router_v3)")
    parser.add_argument("--n_episodes", type=int, default=200)
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    seeds = results.get("seeds", [])
    runs = results.get("runs", {})
    log_dir = Path(args.results).parent

    print("=" * 70)
    print("Phase 8 三节点监测分析")
    print("=" * 70)
    print(f"  Results: {args.results}")
    print(f"  Config:  {args.config}")
    print(f"  Seeds:   {seeds}")
    print(f"  N_eps:   {args.n_episodes}")

    # Collect routing stats from logs
    routing_stats = []
    for s in seeds:
        key = f"{args.config}_seed{s}"
        if key not in runs:
            routing_stats.append(None)
            continue
        log_path = runs[key].get("log", "")
        if not log_path:
            # Try default log path
            log_path = str(log_dir / f"pilot_{args.config}_seed{s}.log")
        stats = parse_routing_stats_from_log(log_path)
        routing_stats.append(stats)
        if stats:
            print(f"  seed={s}: switched={stats['n_switched_to_v4_1']}, "
                  f"fallback={stats['n_fallback']} ({stats['fallback_rate']*100:.1f}%)")
        else:
            print(f"  seed={s}: no routing stats (log not found: {log_path})")

    valid_stats = [r for r in routing_stats if r]
    print(f"\n  Valid routing stats: {len(valid_stats)}/{len(seeds)} seeds")

    # ---- Node 1: Router 置信度分布 ----
    print(f"\n{'='*70}")
    print(f"节点 1: Router 置信度分布 (fallback_rate 代理)")
    print(f"{'='*70}")
    node1 = analyze_node1_confidence(valid_stats)
    for k, v in node1.items():
        print(f"  {k}: {v}")

    # ---- Node 2: 路由决策一致性 ----
    print(f"\n{'='*70}")
    print(f"节点 2: 路由决策一致性 (n_switched CV)")
    print(f"{'='*70}")
    node2 = analyze_node2_consistency(valid_stats, args.n_episodes)
    for k, v in node2.items():
        print(f"  {k}: {v}")

    # ---- Node 3: 端到端 place_rate CV ----
    print(f"\n{'='*70}")
    print(f"节点 3: 端到端 place_rate CV (最终试金石)")
    print(f"{'='*70}")
    node3 = analyze_node3_e2e_cv(results, args.config)
    for k, v in node3.items():
        print(f"  {k}: {v}")

    # ---- Final verdict ----
    print(f"\n{'='*70}")
    print(f"三节点综合判定")
    print(f"{'='*70}")
    nodes = {"node1_confidence": node1, "node2_consistency": node2,
             "node3_e2e_cv": node3}
    all_pass = all(n.get("status") == "PASS" for n in nodes.values())
    for name, n in nodes.items():
        status = n.get("status", "no_data")
        icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "?"
        print(f"  {icon} {name}: {status}")
    print(f"\n  {'→ 项目推进 (V_CV=30% 被吸收)' if all_pass else '→ 需回 Round 2c 优化 V'}")

    # Save report
    report = {
        "results_file": str(args.results),
        "config": args.config,
        "n_seeds": len(seeds),
        "nodes": nodes,
        "all_pass": all_pass,
    }
    report_path = log_dir / f"3node_monitor_{Path(args.results).stem}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
