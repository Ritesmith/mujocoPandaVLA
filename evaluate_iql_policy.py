"""P0.3: Offline IQL policy evaluation.

Validates that the trained IQL policy actually learned useful action selection,
not just Q-value separation. Computes:

  1. Action MSE:       ||π(s) - a_oracle||² on D_expert (target < 0.05)
  2. Top-1 hit rate:   fraction of dimensions within tolerance (target > 70%)
  3. Direction match:  sign agreement between predicted and oracle actions
  4. Log-prob split:   success vs failure trajectory log π(a_oracle|s)
  5. Q-stratified:     action MSE and hit rate by Q-value decile
  6. AWR weight dist:  exp(β·A) distribution statistics

Decision tree (per user spec):
  - MSE < 0.05 AND success/failure log-prob diff significant → P1 (env eval)
  - MSE OK but success/failure diff weak → increase β, retrain policy only
  - MSE not OK but Q_gap healthy → check network capacity or extend training
  - All fail → data/hyperparameter issue

Usage:
    cd /home/w/vla_workspace
    python evaluate_iql_policy.py
    python evaluate_iql_policy.py --checkpoint outputs/iql_v1/best_q_gap.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats as scipy_stats

from iql_dataset import OfflineDataset
from iql_agent import IQLAgent

WORKSPACE = Path("/home/w/vla_workspace")
OUTPUT_DIR = WORKSPACE / "outputs" / "iql_v1"


def load_agent(checkpoint_path: str, device: str = "cpu") -> IQLAgent:
    """Load trained IQL agent from checkpoint."""
    agent = IQLAgent(state_dim=12, action_dim=8, hidden_dim=256,
                     tau=0.7, beta=3.0, gamma=0.99, polyak=0.005,
                     device=device)
    agent.load(checkpoint_path)
    agent.v_net.eval()
    agent.q1_net.eval()
    agent.q2_net.eval()
    agent.policy.eval()
    return agent


@torch.no_grad()
def evaluate(agent: IQLAgent, dataset: OfflineDataset, device: str = "cpu"):
    """Run full offline evaluation on D_expert.npz."""

    data = dataset.get_all_data()
    states = data["states"].to(device)
    actions = data["actions"].to(device)
    rewards = data["rewards"].to(device)
    next_states = data["next_states"].to(device)
    dones = data["dones"].to(device)

    n_transitions = len(states)

    # Build success mask (per-transition)
    success_mask = np.zeros(n_transitions, dtype=bool)
    for ep_id in np.unique(dataset.episode_ids):
        if ep_id < len(dataset.success_flags) and dataset.success_flags[ep_id]:
            success_mask |= (dataset.episode_ids == ep_id)
    success_mask_t = torch.BoolTensor(success_mask).to(device)

    print(f"  Transitions: {n_transitions}")
    print(f"  Success:     {success_mask.sum()} ({success_mask.mean():.1%})")
    print(f"  Failure:     {(~success_mask).sum()} ({(~success_mask).mean():.1%})")
    print()

    # ================================================================
    # 1. Policy predictions (deterministic: tanh(mean))
    # ================================================================
    mean, log_std = agent.policy(states)
    predicted_actions = torch.tanh(mean)  # deterministic policy

    # ================================================================
    # 2. Action MSE
    # ================================================================
    action_diff = predicted_actions - actions  # (N, 8)
    action_mse_per_dim = (action_diff ** 2).mean(dim=0).cpu().numpy()
    action_mse_per_sample = (action_diff ** 2).mean(dim=1)  # (N,)
    action_mse_mean = action_mse_per_sample.mean().item()
    action_mse_std = action_mse_per_sample.std().item()

    # L2 distance
    action_l2 = torch.norm(action_diff, dim=1)  # (N,)
    action_l2_mean = action_l2.mean().item()

    # ================================================================
    # 3. Top-1 hit rate (per-dimension within tolerance)
    # ================================================================
    tol = 0.1
    per_dim_hit = (action_diff.abs() < tol)  # (N, 8) bool
    hit_rate_per_dim = per_dim_hit.float().mean(dim=0).cpu().numpy()
    # Full match: all 8 dims within tolerance
    full_match = per_dim_hit.all(dim=1).float().mean().item()
    # Partial match: at least 6/8 dims within tolerance
    partial_match = (per_dim_hit.sum(dim=1) >= 6).float().mean().item()

    # Direction match: sign agreement
    dir_match_per_dim = (torch.sign(predicted_actions) == torch.sign(actions)).float()
    # Handle zero actions (sign(0)=0, treat as match)
    zero_mask = (actions.abs() < 1e-6)
    dir_match_per_dim[zero_mask] = 1.0
    dir_match_rate = dir_match_per_dim.mean(dim=1)
    dir_match_full = (dir_match_rate >= 6.0/8.0).float().mean().item()
    dir_match_mean = dir_match_rate.mean().item()

    # ================================================================
    # 4. Log-prob of oracle actions
    # ================================================================
    log_prob = agent.policy.log_prob(states, actions)  # (N, 1)
    log_prob = log_prob.squeeze(1)  # (N,)

    # ================================================================
    # 5. Q values and advantages
    # ================================================================
    sa = torch.cat([states, actions], dim=-1)
    q1 = agent.q1_net(sa)
    q2 = agent.q2_net(sa)
    q = torch.min(q1, q2).squeeze(1)
    v = agent.v_net(states).squeeze(1)
    advantage = q - v
    awr_weight = torch.exp(agent.beta * advantage).clamp(max=100.0)

    # ================================================================
    # 6. Success vs Failure analysis
    # ================================================================
    log_prob_success = log_prob[success_mask_t].cpu().numpy()
    log_prob_failure = log_prob[~success_mask_t].cpu().numpy()

    mse_success = action_mse_per_sample[success_mask_t].cpu().numpy()
    mse_failure = action_mse_per_sample[~success_mask_t].cpu().numpy()

    q_success = q[success_mask_t].cpu().numpy()
    q_failure = q[~success_mask_t].cpu().numpy()

    adv_success = advantage[success_mask_t].cpu().numpy()
    adv_failure = advantage[~success_mask_t].cpu().numpy()

    awr_success = awr_weight[success_mask_t].cpu().numpy()
    awr_failure = awr_weight[~success_mask_t].cpu().numpy()

    # Statistical test: is log-prob significantly different?
    t_stat, p_value = scipy_stats.ttest_ind(log_prob_success, log_prob_failure,
                                              equal_var=False)
    # Effect size (Cohen's d)
    pooled_std = math.sqrt((log_prob_success.std()**2 + log_prob_failure.std()**2) / 2)
    cohens_d = (log_prob_success.mean() - log_prob_failure.mean()) / (pooled_std + 1e-8)

    # ================================================================
    # 7. Q-stratified analysis (by Q decile)
    # ================================================================
    q_np = q.cpu().numpy()
    q_deciles = np.percentile(q_np, np.arange(0, 101, 10))
    q_bins = np.digitize(q_np, q_deciles[1:-1])  # 0-9

    stratified = []
    for bin_idx in range(10):
        mask = q_bins == bin_idx
        if mask.sum() == 0:
            continue
        stratified.append({
            "q_decile": bin_idx + 1,
            "n": int(mask.sum()),
            "q_mean": float(q_np[mask].mean()),
            "mse_mean": float(action_mse_per_sample[mask].cpu().numpy().mean()),
            "hit_rate": float(per_dim_hit[mask].float().mean(dim=1).mean().item()),
            "dir_match": float(dir_match_rate[mask].mean().item()),
            "log_prob_mean": float(log_prob[mask].cpu().numpy().mean()),
            "success_ratio": float(success_mask[mask].mean()),
        })

    # ================================================================
    # 8. AWR weight distribution
    # ================================================================
    awr_np = awr_weight.squeeze().cpu().numpy()
    awr_entropy = -(awr_np / awr_np.sum() * np.log(awr_np / awr_np.sum() + 1e-8)).sum()
    awr_max_entropy = math.log(len(awr_np))
    awr_entropy_ratio = awr_entropy / awr_max_entropy
    ess = (awr_np.sum() ** 2) / (awr_np ** 2).sum()

    # ================================================================
    # Print results
    # ================================================================
    print("=" * 70)
    print("P0.3: Offline IQL Policy Evaluation")
    print("=" * 70)

    # --- Action MSE ---
    print("\n--- Action MSE (π(s) vs a_oracle) ---")
    print(f"  Overall MSE:  {action_mse_mean:.6f}  (target < 0.05)")
    print(f"  Overall L2:   {action_l2_mean:.6f}")
    print(f"  Per-dim MSE:  {action_mse_per_dim}")
    mse_pass = action_mse_mean < 0.05
    print(f"  Status: {'✓ PASS' if mse_pass else '✗ FAIL'}")

    # --- Hit rate ---
    print("\n--- Hit Rate (tolerance-based) ---")
    print(f"  Tolerance:    {tol}")
    print(f"  Per-dim hit:  {hit_rate_per_dim}")
    print(f"  Full match (8/8):  {full_match*100:.1f}%")
    print(f"  Partial (≥6/8):    {partial_match*100:.1f}%  (target > 70%)")
    print(f"  Direction match:   {dir_match_mean*100:.1f}%")
    print(f"  Dir match (≥6/8):  {dir_match_full*100:.1f}%")
    hit_pass = partial_match > 0.70
    print(f"  Status: {'✓ PASS' if hit_pass else '✗ FAIL'}")

    # --- Success vs Failure ---
    print("\n--- Success vs Failure Differentiation ---")
    print(f"  {'Metric':<20} {'Success':>12} {'Failure':>12} {'Diff':>12} {'p-value':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Log-prob':<20} {log_prob_success.mean():>12.4f} {log_prob_failure.mean():>12.4f} "
          f"{log_prob_success.mean()-log_prob_failure.mean():>12.4f} {p_value:>12.2e}")
    print(f"  {'Action MSE':<20} {mse_success.mean():>12.6f} {mse_failure.mean():>12.6f} "
          f"{mse_success.mean()-mse_failure.mean():>12.6f} {'':>12}")
    print(f"  {'Q-value':<20} {q_success.mean():>12.4f} {q_failure.mean():>12.4f} "
          f"{q_success.mean()-q_failure.mean():>12.4f} {'':>12}")
    print(f"  {'Advantage':<20} {adv_success.mean():>12.4f} {adv_failure.mean():>12.4f} "
          f"{adv_success.mean()-adv_failure.mean():>12.4f} {'':>12}")
    print(f"  {'AWR weight':<20} {awr_success.mean():>12.4f} {awr_failure.mean():>12.4f} "
          f"{awr_success.mean()-awr_failure.mean():>12.4f} {'':>12}")

    print(f"\n  Statistical test (log-prob):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value:     {p_value:.2e}  ({'significant' if p_value < 0.05 else 'NOT significant'})")
    print(f"    Cohen's d:   {cohens_d:.4f}  ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

    lp_diff = log_prob_success.mean() - log_prob_failure.mean()
    lp_significant = p_value < 0.05 and lp_diff > 0
    print(f"  Status: {'✓ PASS' if lp_significant else '✗ FAIL'} "
          f"(success log-prob {'higher' if lp_diff > 0 else 'LOWER'} than failure)")

    # --- Q-stratified ---
    print("\n--- Q-Stratified Analysis (by Q decile) ---")
    print(f"  {'Decile':>7} {'N':>6} {'Q_mean':>10} {'MSE':>10} {'Hit%':>8} {'Dir%':>8} "
          f"{'LogProb':>10} {'Succ%':>8}")
    print(f"  {'-'*7} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for s in stratified:
        print(f"  {s['q_decile']:>7} {s['n']:>6} {s['q_mean']:>10.2f} {s['mse_mean']:>10.6f} "
              f"{s['hit_rate']*100:>7.1f}% {s['dir_match']*100:>7.1f}% "
              f"{s['log_prob_mean']:>10.4f} {s['success_ratio']*100:>7.1f}%")

    # Check if high-Q bucket has better hit rate than low-Q bucket
    if len(stratified) >= 2:
        low_q_hit = stratified[0]["hit_rate"]
        high_q_hit = stratified[-1]["hit_rate"]
        print(f"\n  Low-Q hit rate:  {low_q_hit*100:.1f}%")
        print(f"  High-Q hit rate: {high_q_hit*100:.1f}%")
        print(f"  High > Low: {'✓' if high_q_hit > low_q_hit else '✗'}")

    # --- AWR weight distribution ---
    print("\n--- AWR Weight Distribution ---")
    print(f"  Mean:      {awr_np.mean():.4f}")
    print(f"  Std:       {awr_np.std():.4f}")
    print(f"  Min:       {awr_np.min():.4f}")
    print(f"  Max:       {awr_np.max():.4f}")
    print(f"  Median:    {np.median(awr_np):.4f}")
    print(f"  Entropy:   {awr_entropy:.4f} / {awr_max_entropy:.4f} ({awr_entropy_ratio*100:.1f}%)")
    print(f"  ESS:       {ess:.1f} / {n_transitions}")

    # ================================================================
    # Decision Tree
    # ================================================================
    print("\n" + "=" * 70)
    print("Decision Tree Assessment")
    print("=" * 70)

    q_gap = q_success.mean() - q_failure.mean()
    results = {
        "action_mse": action_mse_mean,
        "mse_pass": mse_pass,
        "hit_rate": partial_match,
        "hit_pass": hit_pass,
        "log_prob_diff": lp_diff,
        "log_prob_significant": lp_significant,
        "q_gap": q_gap,
        "cohens_d": cohens_d,
        "p_value": p_value,
    }

    if mse_pass and lp_significant:
        print("  → Branch 1: MSE < 0.05 AND success/failure log-prob diff significant")
        print("  → Policy is EFFECTIVE")
        print("  → RECOMMENDATION: Proceed to P1 (environment evaluation)")
        recommendation = "P1_ENV_EVAL"
    elif mse_pass and not lp_significant:
        print("  → Branch 2: MSE OK but success/failure log-prob diff NOT significant")
        print("  → Actor did not absorb Q-V signal")
        print("  → RECOMMENDATION: Increase β (3.0→5.0), retrain policy network only")
        recommendation = "RETRAIN_POLICY_BETA"
    elif not mse_pass and q_gap > 50:
        print("  → Branch 3: MSE not OK but Q_gap healthy (>50)")
        print("  → Policy expressiveness insufficient")
        print("  → RECOMMENDATION: Check network capacity or extend training 100 epochs")
        recommendation = "EXTEND_TRAINING"
    else:
        print("  → Branch 4: All metrics fail")
        print("  → Data/hyperparameter issue")
        print("  → RECOMMENDATION: Review τ=0.7, check D_expert success trajectory quality")
        recommendation = "DATA_HYPERPARAM_REVIEW"

    print(f"\n  Summary: MSE={'✓' if mse_pass else '✗'} | "
          f"HitRate={'✓' if hit_pass else '✗'} | "
          f"LogProbSplit={'✓' if lp_significant else '✗'} | "
          f"Q_gap={q_gap:.1f}")
    print(f"  Decision: {recommendation}")

    # Save results
    eval_results = {
        "action_mse": action_mse_mean,
        "action_mse_per_dim": action_mse_per_dim.tolist(),
        "action_l2": action_l2_mean,
        "hit_rate_full": full_match,
        "hit_rate_partial": partial_match,
        "direction_match": dir_match_mean,
        "direction_match_full": dir_match_full,
        "success_vs_failure": {
            "log_prob_success": float(log_prob_success.mean()),
            "log_prob_failure": float(log_prob_failure.mean()),
            "log_prob_diff": float(lp_diff),
            "mse_success": float(mse_success.mean()),
            "mse_failure": float(mse_failure.mean()),
            "q_success": float(q_success.mean()),
            "q_failure": float(q_failure.mean()),
            "q_gap": float(q_gap),
            "adv_success": float(adv_success.mean()),
            "adv_failure": float(adv_failure.mean()),
            "awr_success": float(awr_success.mean()),
            "awr_failure": float(awr_failure.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
        },
        "stratified": stratified,
        "awr_distribution": {
            "mean": float(awr_np.mean()),
            "std": float(awr_np.std()),
            "min": float(awr_np.min()),
            "max": float(awr_np.max()),
            "entropy": float(awr_entropy),
            "entropy_ratio": float(awr_entropy_ratio),
            "ess": float(ess),
        },
        "decision": recommendation,
        "metrics_pass": {
            "mse": mse_pass,
            "hit_rate": hit_pass,
            "log_prob_split": lp_significant,
        },
    }

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Offline IQL policy evaluation")
    parser.add_argument("--checkpoint", type=str,
                        default=str(OUTPUT_DIR / "final_model.pt"),
                        help="Path to IQL checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = OfflineDataset(
        data_path=str(WORKSPACE / "data" / "D_expert.npz"),
        normalize_states=True,
        normalize_actions=False,
    )

    # Load agent
    print("\nLoading IQL agent...")
    agent = load_agent(args.checkpoint, device=args.device)

    # Run evaluation
    results = evaluate(agent, dataset, device=args.device)

    # Save
    output_path = OUTPUT_DIR / "offline_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
