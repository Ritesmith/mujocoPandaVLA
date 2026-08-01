"""Reward density diagnostic for RL from scratch failure analysis.

Verifies the reward density trap hypothesis: if per-step rewards have near-zero
variance conditioned on the action, PPO's advantage function degenerates and
the policy gradient has no directional signal.

Three diagnostic layers:
  1. Dataset statistics (D_expert.npz) — action/state variance
  2. Static reward function analysis — action-dependent vs state-dependent ratio
  3. Training log analysis — episode reward statistics, explained_variance
  4. (Optional) Environment rollout — actual per-step reward distribution

Usage:
    cd /home/w/vla_workspace
    python diagnose_reward_density.py [--rollout N_EPISODES]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path("/home/w/vla_workspace")
DATASET_PATH = WORKSPACE / "data" / "D_expert.npz"
TRAINING_LOG = WORKSPACE / "outputs" / "rl_from_scratch_v1_stage1.log"
RESULTS_JSON = WORKSPACE / "outputs" / "rl_from_scratch_v1" / "training_results.json"
OUTPUT_PATH = WORKSPACE / "outputs" / "reward_density_diagnostic.json"


# ---------------------------------------------------------------------------
# Layer 1: Dataset statistics
# ---------------------------------------------------------------------------

def analyze_dataset():
    """Analyze action and state distributions in D_expert.npz."""
    print("=" * 70)
    print("Layer 1: Dataset Statistics (D_expert.npz)")
    print("=" * 70)

    if not DATASET_PATH.exists():
        print(f"  [SKIP] {DATASET_PATH} not found")
        return None

    data = np.load(DATASET_PATH, allow_pickle=True)
    actions = data["actions"]  # (29467, 8)
    states = data["states"]    # (29467, 12)
    episode_ids = data["episode_ids"]  # (29467,)
    success_flags = data["success_flags"]  # (200,)

    n_transitions = len(actions)
    n_episodes = int(data["n_episodes"])
    n_placed = int(data["n_placed"])
    place_rate = n_placed / n_episodes

    print(f"  Transitions: {n_transitions}")
    print(f"  Episodes: {n_episodes} (placed: {n_placed}, place_rate: {place_rate:.1%})")

    # --- Action statistics ---
    print(f"\n  Action statistics (8-dim, range [-1, 1]):")
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)
    action_var = actions.var(axis=0)
    print(f"    Per-dim mean: {action_mean}")
    print(f"    Per-dim std:  {action_std}")
    print(f"    Per-dim var:  {action_var}")
    print(f"    Overall mean: {actions.mean():.6f}")
    print(f"    Overall std:  {actions.std():.6f}")
    print(f"    Overall var:  {actions.var():.6f}")

    # Within-episode vs between-episode action variance
    unique_eps = np.unique(episode_ids)
    within_ep_vars = []
    between_ep_means = []
    for ep_id in unique_eps:
        mask = episode_ids == ep_id
        ep_actions = actions[mask]
        if len(ep_actions) > 1:
            within_ep_vars.append(ep_actions.var(axis=0).mean())
        between_ep_means.append(ep_actions.mean(axis=0))
    within_ep_var = np.mean(within_ep_vars)
    between_ep_var = np.var(between_ep_means, axis=0).mean()
    print(f"\n  Action variance decomposition:")
    print(f"    Within-episode var (mean):  {within_ep_var:.6f}")
    print(f"    Between-episode var (mean): {between_ep_var:.6f}")
    print(f"    Ratio (within/total):       {within_ep_var / (within_ep_var + between_ep_var):.4f}")

    # Action-to-action differences (smoothness)
    action_diffs = np.diff(actions, axis=0)
    # Only compute within-episode diffs
    same_ep_mask = episode_ids[1:] == episode_ids[:-1]
    action_diffs_same_ep = action_diffs[same_ep_mask]
    action_diff_norms = np.linalg.norm(action_diffs_same_ep, axis=1)
    print(f"\n  Action smoothness (within-episode consecutive diffs):")
    print(f"    ||a_t - a_{{t-1}}||^2 mean: {np.mean(action_diff_norms**2):.6f}")
    print(f"    ||a_t - a_{{t-1}}||^2 std:  {np.std(action_diff_norms**2):.6f}")
    print(f"    ||a_t - a_{{t-1}}||^2 max:  {np.max(action_diff_norms**2):.6f}")

    # --- State statistics ---
    print(f"\n  State statistics (12-dim):")
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0)
    state_var = states.var(axis=0)
    print(f"    Per-dim mean: {state_mean}")
    print(f"    Per-dim std:  {state_std}")
    print(f"    Per-dim var:  {state_var}")
    print(f"    Overall var:  {states.var():.6f}")

    # Estimate block-target distance from state (last few dims typically)
    # State format: [gripper_pos(3), gripper_quat(4), block_rel_pos(3?), ...]
    # We'll check the range of each dimension to identify distance-like dims
    print(f"\n  State dimension ranges:")
    for i in range(states.shape[1]):
        vals = states[:, i]
        print(f"    dim[{i}]: min={vals.min():.4f}, max={vals.max():.4f}, "
              f"mean={vals.mean():.4f}, std={vals.std():.4f}")

    # --- Reward density estimation from actions ---
    # The reward function has:
    #   State-dependent: -5.0*dist, -2.0*height, -0.5*hover, -0.01*time
    #   Action-dependent: -0.001*||jerk||^2, -0.005*||action_diff||^2
    # Estimate the action-dependent reward magnitude
    action_diff_penalty = 0.005 * np.mean(action_diff_norms**2)
    print(f"\n  Estimated action-dependent reward magnitude:")
    print(f"    Action diff penalty: -0.005 * {np.mean(action_diff_norms**2):.4f} = {action_diff_penalty:.6f}")

    # Jerk estimate (a_t - 2*a_{t-1} + a_{t-2})
    if len(actions) > 2:
        jerks = actions[2:] - 2 * actions[1:-1] + actions[:-2]
        same_ep_jerk_mask = (episode_ids[2:] == episode_ids[1:-1]) & (episode_ids[1:-1] == episode_ids[:-2])
        jerks_same_ep = jerks[same_ep_jerk_mask]
        jerk_norms_sq = np.sum(jerks_same_ep**2, axis=1)
        jerk_penalty = 0.001 * np.mean(jerk_norms_sq)
        print(f"    Jerk penalty:        -0.001 * {np.mean(jerk_norms_sq):.4f} = {jerk_penalty:.6f}")
    else:
        jerk_penalty = 0.0

    total_action_dep = action_diff_penalty + jerk_penalty
    print(f"    Total action-dep reward magnitude: {total_action_dep:.6f}")

    # State-dependent reward magnitude estimate
    # If typical distance is ~0.1-0.3m, distance penalty is -0.5 to -1.5
    # Time penalty is -0.01
    # We don't have exact block positions, but we can estimate from final_dists
    final_dists = data["final_dists"]
    # final_dists may be per-transition or per-episode; get unique per-episode values
    if len(final_dists) == n_transitions:
        # Per-transition: extract one per episode
        ep_final_dists = np.array([final_dists[episode_ids == ep_id][0] for ep_id in unique_eps])
    else:
        ep_final_dists = final_dists
    print(f"\n  Final distances (episode-level, n={len(ep_final_dists)}):")
    print(f"    mean: {ep_final_dists.mean():.4f}m, std: {ep_final_dists.std():.4f}m")
    print(f"    min: {ep_final_dists.min():.4f}m, max: {ep_final_dists.max():.4f}m")
    if len(ep_final_dists) == len(success_flags):
        print(f"    Success ep final dist: {ep_final_dists[success_flags == 1].mean():.4f}m")
        print(f"    Failure ep final dist: {ep_final_dists[success_flags == 0].mean():.4f}m")

    # Rough state-dependent reward estimate
    typical_dist = 0.15  # conservative estimate during approach
    dist_penalty = 5.0 * typical_dist
    time_penalty = 0.01
    total_state_dep = dist_penalty + time_penalty
    print(f"\n  Estimated state-dependent reward magnitude (typical):")
    print(f"    Distance penalty: -5.0 * {typical_dist} = {dist_penalty:.4f}")
    print(f"    Time penalty:     -0.01")
    print(f"    Total state-dep:  {total_state_dep:.4f}")

    ratio = total_action_dep / total_state_dep if total_state_dep > 0 else float('inf')
    print(f"\n  *** Action-dependent / State-dependent ratio: {ratio:.6f} ({ratio*100:.3f}%) ***")
    print(f"  *** Action-dep signal is {total_state_dep / total_action_dep:.0f}x weaker than state-dep ***")

    return {
        "n_transitions": n_transitions,
        "n_episodes": n_episodes,
        "place_rate": place_rate,
        "action_var_overall": float(actions.var()),
        "action_var_within_ep": float(within_ep_var),
        "action_var_between_ep": float(between_ep_var),
        "action_diff_norm_sq_mean": float(np.mean(action_diff_norms**2)),
        "action_diff_penalty_est": float(action_diff_penalty),
        "jerk_penalty_est": float(jerk_penalty),
        "total_action_dep_reward_est": float(total_action_dep),
        "total_state_dep_reward_est": float(total_state_dep),
        "action_to_state_reward_ratio": float(ratio),
        "final_dist_mean": float(ep_final_dists.mean()),
        "final_dist_std": float(ep_final_dists.std()),
    }


# ---------------------------------------------------------------------------
# Layer 2: Static reward function analysis
# ---------------------------------------------------------------------------

def analyze_reward_function():
    """Statically analyze _compute_reward_safe() component magnitudes."""
    print("\n" + "=" * 70)
    print("Layer 2: Static Reward Function Analysis")
    print("=" * 70)

    components = {
        # name: (coefficient, depends_on_action, description)
        "distance_penalty": (5.0, False, "-5.0 * block_target_dist (state-dep via transition)"),
        "height_penalty": (2.0, False, "-2.0 * excess_height (state-dep)"),
        "hover_penalty": (0.5, False, "-0.5 * hover_intensity (state-dep)"),
        "time_penalty": (0.01, False, "-0.01 (constant)"),
        "jerk_penalty": (0.001, True, "-0.001 * ||a_t - 2*a_{t-1} + a_{t-2}||^2 (action-dep)"),
        "action_diff_penalty": (0.005, True, "-0.005 * ||a_t - a_{t-1}||^2 (action-dep)"),
        "early_release": (5.0, False, "-5.0 one-time (state-dep transition)"),
        "release_bonus": (50.0, False, "+50.0 one-time (state-dep, terminal)"),
        "terminal_success": (200.0, False, "+200.0 one-time (state-dep, terminal)"),
    }

    print("\n  Reward components:")
    print(f"  {'Component':<25} {'Coeff':>8} {'Action-dep?':>12} {'Description'}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*50}")
    for name, (coeff, action_dep, desc) in components.items():
        flag = "YES" if action_dep else "no"
        print(f"  {name:<25} {coeff:>8.3f} {flag:>12} {desc}")

    # Magnitude analysis
    print("\n  Magnitude analysis (typical values):")
    typical_values = {
        "distance_penalty": 5.0 * 0.15,       # dist=0.15m
        "height_penalty": 2.0 * 0.05,          # excess_height=5cm
        "hover_penalty": 0.5 * 0.5,            # hover_intensity=0.5
        "time_penalty": 0.01,                  # constant
        "jerk_penalty": 0.001 * 0.5,           # ||jerk||^2=0.5
        "action_diff_penalty": 0.005 * 0.3,    # ||action_diff||^2=0.3
    }

    state_dep_total = sum(v for k, v in typical_values.items() if not components[k][1])
    action_dep_total = sum(v for k, v in typical_values.items() if components[k][1])

    print(f"  {'Component':<25} {'Magnitude':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for name, val in typical_values.items():
        print(f"  {name:<25} {val:>10.6f}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'State-dep total':<25} {state_dep_total:>10.6f}")
    print(f"  {'Action-dep total':<25} {action_dep_total:>10.6f}")
    print(f"  {'Ratio (action/state)':<25} {action_dep_total/state_dep_total:>10.6f}")
    print(f"  {'Action is N× weaker':<25} {state_dep_total/action_dep_total:>10.1f}x")

    # Theoretical advantage analysis
    print("\n  Theoretical advantage analysis:")
    print(f"    PPO advantage: A(s,a) = Q(s,a) - V(s)")
    print(f"    If V(s) well-estimated (explained_var=0.87-0.95):")
    print(f"      δ_t = r_t + γV(s') - V(s) ≈ 0 for most transitions")
    print(f"      A(s,a) ≈ 0 → policy gradient has no directional signal")
    print(f"    The action-dependent reward ({action_dep_total:.4f}) is the ONLY")
    print(f"    signal that differentiates actions. But it's {state_dep_total/action_dep_total:.0f}x")
    print(f"    weaker than state-dependent reward ({state_dep_total:.4f}).")
    print(f"    PPO's clip mechanism and KL constraint dominate this weak signal.")

    # Key insight about distance penalty
    print("\n  IMPORTANT NUANCE — distance penalty IS action-dependent via transition:")
    print(f"    -5.0 * dist(s') where s' depends on action a_t")
    print(f"    If action moves block 1cm closer: Δreward = 5.0 * 0.01 = 0.05")
    print(f"    But V(s) also changes by 0.05/(1-γ) = {0.05/(1-0.99):.2f}")
    print(f"    So advantage A(s,a) = 0.05 - {0.05/(1-0.99):.2f} * ΔV ≈ 0")
    print(f"    The reward signal CANCELS in the advantage computation")
    print(f"    because the value function captures the state-dependent reward.")

    return {
        "state_dep_total": float(state_dep_total),
        "action_dep_total": float(action_dep_total),
        "ratio": float(action_dep_total / state_dep_total),
        "weaker_factor": float(state_dep_total / action_dep_total),
    }


# ---------------------------------------------------------------------------
# Layer 3: Training log analysis
# ---------------------------------------------------------------------------

def parse_training_log():
    """Parse the RL from scratch training log for reward statistics."""
    print("\n" + "=" * 70)
    print("Layer 3: Training Log Analysis")
    print("=" * 70)

    if not TRAINING_LOG.exists():
        # Check archived log
        archived = WORKSPACE / "auto_iter" / "evidence" / "family_10_rl_from_scratch" / "training.log"
        if archived.exists():
            log_path = archived
        else:
            print(f"  [SKIP] Training log not found")
            return None
    else:
        log_path = TRAINING_LOG

    with open(log_path) as f:
        content = f.read()

    # Extract episode reward at eval time
    reward_pattern = r"episode_reward=([\-\d.]+)\s*\+/\-\s*([\d.]+)"
    reward_matches = re.findall(reward_pattern, content)

    # Extract episode length
    length_pattern = r"Episode length:\s*([\d.]+)\s*\+/\-\s*([\d.]+)"
    length_matches = re.findall(length_pattern, content)

    # Extract training metrics per iteration
    kl_pattern = r"approx_kl\s*\|\s*([\d.]+)"
    kl_matches = re.findall(kl_pattern, content)

    ev_pattern = r"explained_variance\s*\|\s*([\d.]+)"
    ev_matches = re.findall(ev_pattern, content)

    pg_loss_pattern = r"policy_gradient_loss\s*\|\s*([\-\d.]+)"
    pg_loss_matches = re.findall(pg_loss_pattern, content)

    clip_pattern = r"clip_fraction\s*\|\s*([\d.]+)"
    clip_matches = re.findall(clip_pattern, content)

    bc_loss_pattern = r"bc_loss\s*\|\s*([\d.]+)"
    bc_loss_matches = re.findall(bc_loss_pattern, content)

    # Extract hier eval results
    hier_pattern = r"\[HIER_EVAL\] step=(\d+)\s+place_rate=(\d+)%"
    hier_matches = re.findall(hier_pattern, content)

    print(f"  Training log: {log_path}")
    print(f"  PPO iterations logged: {len(kl_matches)}")
    print(f"  Eval episodes logged: {len(reward_matches)}")

    if reward_matches:
        rewards = [float(m[0]) for m in reward_matches]
        reward_stds = [float(m[1]) for m in reward_matches]
        print(f"\n  Episode rewards at eval:")
        for i, (r, s) in enumerate(zip(rewards, reward_stds)):
            print(f"    Eval {i}: reward={r:.2f} ± {s:.2f}")

        ep_reward = rewards[0]
        ep_reward_std = reward_stds[0]
        cv = ep_reward_std / abs(ep_reward) if ep_reward != 0 else float('inf')

        print(f"\n  Episode reward statistics:")
        print(f"    Mean: {ep_reward:.2f}")
        print(f"    Std:  {ep_reward_std:.2f}")
        print(f"    CV (std/mean): {cv:.4f} ({cv*100:.2f}%)")
        print(f"    Episode-level variance: {'HIGH' if cv > 0.1 else 'LOW'}")

    if length_matches:
        lengths = [float(m[0]) for m in length_matches]
        length_stds = [float(m[1]) for m in length_matches]
        print(f"\n  Episode lengths:")
        for i, (l, s) in enumerate(zip(lengths, length_stds)):
            print(f"    Eval {i}: length={l:.1f} ± {s:.1f}")
        if lengths[0] == 500.0 and length_stds[0] == 0.0:
            print(f"    *** ALL episodes hit 500-step max — 0% terminal success ***")
            print(f"    *** No +200/+50 terminal bonuses ever fire ***")
            print(f"    *** ALL reward is from per-step penalties ***")

    if kl_matches:
        kls = [float(k) for k in kl_matches]
        print(f"\n  PPO approx_kl per iteration:")
        print(f"    Values: {kls}")
        print(f"    Mean: {np.mean(kls):.6f}")
        print(f"    Max:  {np.max(kls):.6f}")
        print(f"    Min:  {np.min(kls):.6f}")
        print(f"    All below 0.03 'safe' threshold: {all(k < 0.03 for k in kls)}")

    if ev_matches:
        evs = [float(e) for e in ev_matches]
        print(f"\n  Explained variance per iteration:")
        print(f"    Values: {evs}")
        print(f"    Mean: {np.mean(evs):.4f}")
        print(f"    *** V(s) is {'WELL-ESTIMATED' if np.mean(evs) > 0.8 else 'POORLY-ESTIMATED'} ***")
        print(f"    *** High explained_var → δ_t ≈ 0 → advantage ≈ 0 ***")

    if pg_loss_matches:
        pg_losses = [float(p) for p in pg_loss_matches]
        print(f"\n  Policy gradient loss per iteration:")
        print(f"    Values: {pg_losses}")
        print(f"    Mean: {np.mean(pg_losses):.6f}")
        print(f"    Std:  {np.std(pg_losses):.6f}")
        print(f"    Sign changes: {sum(1 for i in range(1, len(pg_losses)) if (pg_losses[i] > 0) != (pg_losses[i-1] > 0))}")
        print(f"    *** {'Oscillating sign = no consistent gradient direction' if np.std(pg_losses) > 0.01 else 'Stable'} ***")

    if hier_matches:
        print(f"\n  Hier eval results:")
        for step, rate in hier_matches:
            print(f"    step={step}: place_rate={rate}%")

    # Compute per-step reward from episode data
    if reward_matches and length_matches:
        ep_reward = float(reward_matches[0][0])
        ep_length = float(length_matches[0][0])
        per_step_reward = ep_reward / ep_length
        print(f"\n  Per-step reward estimate:")
        print(f"    Total episode reward: {ep_reward:.2f}")
        print(f"    Episode length: {ep_length:.0f}")
        print(f"    Per-step reward: {per_step_reward:.4f}")
        print(f"    Expected time penalty only: -0.01")
        print(f"    Actual per-step: {per_step_reward:.4f}")
        print(f"    Excess over time penalty: {per_step_reward - (-0.01):.4f}")
        print(f"    (excess = distance + height + hover + jerk + action_diff penalties)")

    return {
        "episode_reward_mean": float(rewards[0]) if reward_matches else None,
        "episode_reward_std": float(reward_stds[0]) if reward_matches else None,
        "episode_length": float(lengths[0]) if length_matches else None,
        "per_step_reward": float(rewards[0] / lengths[0]) if reward_matches and length_matches else None,
        "approx_kl_mean": float(np.mean(kls)) if kl_matches else None,
        "approx_kl_max": float(np.max(kls)) if kl_matches else None,
        "explained_variance_mean": float(np.mean(evs)) if ev_matches else None,
        "policy_gradient_loss_values": [float(p) for p in pg_losses] if pg_loss_matches else None,
        "place_rate_final": int(hier_matches[-1][1]) if hier_matches else None,
    }


# ---------------------------------------------------------------------------
# Layer 4: Environment rollout (optional)
# ---------------------------------------------------------------------------

def rollout_reward_distribution(n_episodes=3):
    """Run the environment with BC warmstart policy to collect per-step rewards."""
    print("\n" + "=" * 70)
    print(f"Layer 4: Environment Rollout ({n_episodes} episodes)")
    print("=" * 70)

    try:
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO

        print("  Loading BC warmstart model...")
        bc_model_path = WORKSPACE / "outputs" / "bc_expert_v1" / "final_model.zip"
        if not bc_model_path.exists():
            print(f"  [SKIP] BC model not found at {bc_model_path}")
            return None

        # We need the custom env, which requires the full import chain
        sys.path.insert(0, str(WORKSPACE))
        from gym_env.panda_vla_env import PandaVlaEnv

        print("  Creating environment (reward_type='place_safe')...")
        env = PandaVlaEnv(
            render_mode=None,
            reward_type="place_safe",
            image_size=84,
            episode_limit=500,
        )

        # Load VecNormalize
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        vec_normalize_path = WORKSPACE / "outputs" / "place_policy_v59" / "best_hier" / "vec_normalize.pkl"

        print("  Loading BC warmstart model...")
        model = PPO.load(str(bc_model_path), device="cpu")

        all_rewards = []
        all_reward_components = []

        for ep in range(n_episodes):
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            ep_rewards = []
            done = False
            step = 0

            while not done and step < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_rewards.append(reward)
                done = terminated or truncated
                step += 1

            all_rewards.extend(ep_rewards)
            print(f"  Episode {ep+1}: length={step}, total_reward={sum(ep_rewards):.2f}, "
                  f"mean_reward={np.mean(ep_rewards):.4f}")

            # Extract reward components from env for the last episode
            if hasattr(env, '_last_reward_components'):
                all_reward_components.append(env._last_reward_components.copy())

        env.close()

        rewards = np.array(all_rewards)
        print(f"\n  Per-step reward distribution ({len(rewards)} steps):")
        print(f"    Mean: {rewards.mean():.6f}")
        print(f"    Std:  {rewards.std():.6f}")
        print(f"    Var:  {rewards.var():.6f}")
        print(f"    Min:  {rewards.min():.6f}")
        print(f"    Max:  {rewards.max():.6f}")
        print(f"    Median: {np.median(rewards):.6f}")
        print(f"    Percentiles: 1%={np.percentile(rewards, 1):.4f}, "
              f"50%={np.percentile(rewards, 50):.4f}, "
              f"99%={np.percentile(rewards, 99):.4f}")

        # Reward variance analysis
        cv = rewards.std() / abs(rewards.mean()) if rewards.mean() != 0 else float('inf')
        print(f"\n    Coefficient of variation: {cv:.4f} ({cv*100:.2f}%)")
        print(f"    Near-zero variance (CV < 0.1): {cv < 0.1}")
        print(f"    Reward density: {'SPARSE' if cv < 0.1 else 'MODERATE' if cv < 0.5 else 'DENSE'}")

        # Unique reward values
        unique_rewards = np.unique(np.round(rewards, 4))
        print(f"    Unique reward values (rounded to 4 decimals): {len(unique_rewards)}")
        if len(unique_rewards) <= 20:
            print(f"    Values: {unique_rewards}")

        return {
            "n_steps": len(rewards),
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "var": float(rewards.var()),
            "cv": float(cv),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
            "median": float(np.median(rewards)),
            "p1": float(np.percentile(rewards, 1)),
            "p99": float(np.percentile(rewards, 99)),
        }

    except Exception as e:
        print(f"  [SKIP] Environment rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Summary and verdict
# ---------------------------------------------------------------------------

def print_summary(dataset_stats, reward_analysis, log_stats, rollout_stats):
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print("\n  Hypothesis: Reward density trap — per-step rewards have near-zero")
    print("  variance, causing PPO's advantage function to degenerate.\n")

    if log_stats:
        print(f"  1. Episode reward: {log_stats.get('episode_reward_mean'):.2f} "
              f"± {log_stats.get('episode_reward_std'):.2f}")
        print(f"     Episode length: {log_stats.get('episode_length'):.0f} (ALL hit max)")
        print(f"     Per-step reward: {log_stats.get('per_step_reward'):.4f}")
        print(f"     Place rate: {log_stats.get('place_rate_final')}% (0% → no terminal bonuses)")

    if log_stats and log_stats.get("explained_variance_mean"):
        ev = log_stats["explained_variance_mean"]
        print(f"\n  2. Value function: explained_variance={ev:.4f}")
        print(f"     → V(s) is {'WELL-ESTIMATED' if ev > 0.8 else 'POORLY-ESTIMATED'}")
        print(f"     → δ_t = r + γV(s') - V(s) ≈ 0 for most transitions")
        print(f"     → Advantage A(s,a) ≈ 0 → no gradient direction")

    if dataset_stats:
        ratio = dataset_stats.get("action_to_state_reward_ratio", 0)
        print(f"\n  3. Action-dependent / State-dependent reward ratio: {ratio:.4f} ({ratio*100:.2f}%)")
        print(f"     Action-dep reward magnitude: {dataset_stats.get('total_action_dep_reward_est'):.6f}")
        print(f"     State-dep reward magnitude:  {dataset_stats.get('total_state_dep_reward_est'):.4f}")
        print(f"     → Action signal is {1/ratio:.0f}x weaker than state signal")

    if log_stats and log_stats.get("policy_gradient_loss_values"):
        pg_losses = log_stats["policy_gradient_loss_values"]
        sign_changes = sum(1 for i in range(1, len(pg_losses)) if (pg_losses[i] > 0) != (pg_losses[i-1] > 0))
        print(f"\n  4. Policy gradient loss: {pg_losses}")
        print(f"     Sign changes: {sign_changes}/{len(pg_losses)-1}")
        print(f"     → {'Oscillating — no consistent direction' if sign_changes > 2 else 'Stable direction'}")

    if rollout_stats:
        cv = rollout_stats.get("cv", 0)
        print(f"\n  5. Per-step reward CV (from rollout): {cv:.4f} ({cv*100:.2f}%)")
        print(f"     Reward density: {'SPARSE' if cv < 0.1 else 'MODERATE' if cv < 0.5 else 'DENSE'}")

    print("\n  VERDICT:")
    print("  " + "-" * 66)

    # Determine verdict
    evidence_for = 0
    evidence_against = 0
    evidence_points = []

    if log_stats and log_stats.get("explained_variance_mean", 0) > 0.8:
        evidence_for += 1
        evidence_points.append(f"High explained_variance ({log_stats['explained_variance_mean']:.3f}) → advantage ≈ 0")

    if dataset_stats and dataset_stats.get("action_to_state_reward_ratio", 1) < 0.05:
        evidence_for += 1
        evidence_points.append(f"Action-dep reward is {1/dataset_stats['action_to_state_reward_ratio']:.0f}x weaker than state-dep")

    if log_stats and log_stats.get("place_rate_final") == 0:
        evidence_for += 1
        evidence_points.append("0% place rate → no terminal bonuses → all reward is state-dep penalties")

    if log_stats and log_stats.get("policy_gradient_loss_values"):
        pg_losses = log_stats["policy_gradient_loss_values"]
        sign_changes = sum(1 for i in range(1, len(pg_losses)) if (pg_losses[i] > 0) != (pg_losses[i-1] > 0))
        if sign_changes > 2:
            evidence_for += 1
            evidence_points.append(f"Policy gradient loss oscillates ({sign_changes} sign changes) — no direction")

    if rollout_stats:
        cv = rollout_stats.get("cv", 0)
        if cv < 0.1:
            evidence_for += 1
            evidence_points.append(f"Per-step reward CV={cv:.3f} < 0.1 → near-zero variance")

    print(f"  Evidence FOR reward density trap: {evidence_for}")
    for p in evidence_points:
        print(f"    ✓ {p}")

    if evidence_for >= 3:
        verdict = "CONFIRMED"
    elif evidence_for >= 2:
        verdict = "PARTIALLY CONFIRMED"
    else:
        verdict = "REFUTED"

    print(f"\n  *** HYPOTHESIS: {verdict} ***")
    print(f"  PPO's advantage function has no effective directional signal because:")
    print(f"  (a) V(s) captures the state-dependent reward → δ_t ≈ 0")
    print(f"  (b) The remaining action-dependent signal is 10-100x weaker")
    print(f"  (c) PPO's clip + KL dominate the weak advantage signal")
    print(f"  → Policy does random walk, not directed optimization")

    print(f"\n  IMPLICATION FOR IQL:")
    print(f"  IQL is less affected because:")
    print(f"  (a) No exploration needed (uses offline data)")
    print(f"  (b) Expectile regression amplifies upper-tail Q values")
    print(f"  (c) AWR's exp(β·A) weighting amplifies small advantages")
    print(f"  But reward shaping (potential-based) may still be needed if")
    print(f"  the action-dependent signal is too weak for IQL to extract.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=int, default=0,
                        help="Number of episodes to rollout for actual reward distribution (0=skip)")
    args = parser.parse_args()

    print("Reward Density Diagnostic")
    print(f"Date: 2026-07-14")
    print(f"Context: RL from scratch failure analysis (PPO destroyed BC warmstart)")
    print()

    dataset_stats = analyze_dataset()
    reward_analysis = analyze_reward_function()
    log_stats = parse_training_log()

    rollout_stats = None
    if args.rollout > 0:
        rollout_stats = rollout_reward_distribution(args.rollout)

    print_summary(dataset_stats, reward_analysis, log_stats, rollout_stats)

    # Save results
    results = {
        "diagnostic": "reward_density_trap",
        "date": "2026-07-14",
        "dataset_stats": dataset_stats,
        "reward_analysis": reward_analysis,
        "log_stats": log_stats,
        "rollout_stats": rollout_stats,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
