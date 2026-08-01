"""IQL training script for offline RL on D_expert.npz.

Implements the three-step IQL training loop (V → Q → Policy) with
comprehensive monitoring of the key metrics identified in the reward
density diagnostic.

Monitoring metrics (per user spec):
  1. Expectile gap: Q_target.mean() - V.mean()  (should be positive, stable)
  2. AWR weight entropy ratio  (low = signal concentrated = good)
  3. Q-value separation: Q_success - Q_failure  (should be > 100)
  4. Advantage distribution  (should have non-zero variance)
  5. ESS (Effective Sample Size)  (should not collapse to 1)

Risk detection:
  - AWR weight collapse: weight_entropy_ratio > 0.95 → IQL degenerates to BC
  - V non-convergence: expectile_gap < 0 → V not learning upper tail
  - Q no separation: q_gap < 50 → Q can't distinguish success/failure

Usage:
    cd /home/w/vla_workspace
    python train_iql.py --n_epochs 100 --batch_size 256
    python train_iql.py --n_epochs 200 --beta 10.0  # if AWR weights too flat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from iql_dataset import OfflineDataset
from iql_agent import IQLAgent

WORKSPACE = Path("/home/w/vla_workspace")
OUTPUT_DIR = WORKSPACE / "outputs" / "iql_v1"


def set_global_seed(seed: int, init_seed: int = None, data_seed: int = None):
    """Set all RNG seeds for reproducibility.

    For Option A (single seed): init_seed=data_seed=None → all use `seed`.
    For Option B (decoupled): init_seed controls torch (weight init),
    data_seed controls numpy (data shuffling order).
    """
    s_init = init_seed if init_seed is not None else seed
    s_data = data_seed if data_seed is not None else seed
    torch.manual_seed(s_init)
    np.random.seed(s_data)
    random.seed(seed)
    print(f"  Seed control: seed={seed}, init_seed={s_init}, data_seed={s_data}")


def compute_qv_diagnostics(agent, probe_states_np, device="cpu"):
    """Compute Q1/Q2/V statistics on fixed probe states.

    Uses policy's deterministic mean action for Q evaluation:
      Q(s, π_mean(s)) where π_mean = tanh(μ(s))

    Returns dict with q1/q2/v mean±std and q1_q2_gap_mean.
    """
    states_t = torch.FloatTensor(probe_states_np).to(device)
    with torch.no_grad():
        # V values
        v = agent.v_net(states_t).squeeze(-1)

        # Policy deterministic mean action (tanh squashed)
        mean, _ = agent.policy(states_t)
        action_chunk = torch.tanh(mean)

        # Q values: (state, action_chunk) → Q
        sa = torch.cat([states_t, action_chunk], dim=-1)
        q1 = agent.q1_net(sa).squeeze(-1)
        q2 = agent.q2_net(sa).squeeze(-1)

    q1_np = q1.cpu().numpy()
    q2_np = q2.cpu().numpy()
    v_np = v.cpu().numpy()

    return {
        "n_probe_states": len(probe_states_np),
        "q1_mean": float(np.mean(q1_np)),
        "q1_std": float(np.std(q1_np)),
        "q2_mean": float(np.mean(q2_np)),
        "q2_std": float(np.std(q2_np)),
        "v_mean": float(np.mean(v_np)),
        "v_std": float(np.std(v_np)),
        "q1_q2_gap_mean": float(np.mean(np.abs(q1_np - q2_np))),
    }


def compute_init_hash(agent, probe_states_np, device="cpu"):
    """Compute hash of Q/V outputs right after agent initialization.

    Used for dry-run seed control verification:
      - Same seed → same hash (reproducibility)
      - Different seeds → different hash (independence)
    """
    diag = compute_qv_diagnostics(agent, probe_states_np, device)
    hash_str = (f"q1_mean={diag['q1_mean']:.6f},"
                f"q2_mean={diag['q2_mean']:.6f},"
                f"v_mean={diag['v_mean']:.6f}")
    return hashlib.md5(hash_str.encode()).hexdigest()[:12], diag


def sample_probe_states(dataset, n=100, seed=0):
    """Sample n fixed probe states from dataset (in-distribution).

    Uses fixed seed=0 so all training runs share the same probe set.
    States are already normalized (dataset.normalize_states=True).
    """
    rng = np.random.RandomState(seed)
    n_total = len(dataset.states)
    indices = rng.choice(n_total, size=n, replace=False)
    indices.sort()
    probe_states = dataset.states[indices]
    return probe_states, indices


def train(
    n_epochs: int = 100,
    batch_size: int = 256,
    tau: float = 0.7,
    beta: float = 3.0,
    gamma: float = 0.99,
    lr: float = 3e-4,
    polyak: float = 0.005,
    eval_every: int = 10,
    save_every: int = 50,
    n_step: int = 1,
    oversample_dist: str = None,
    oversample_factor: int = 3,
    reward_shaping: bool = False,
    chunk_size: int = 1,
    output_dir: str = None,
    device: str = "cpu",
    seed: int = 42,
    init_seed: int = None,
    data_seed: int = None,
    probe_states_path: str = None,
    normalize_rewards: bool = False,
    awr_hist_epochs: tuple = (1, 50, 100),
    # Phase 7 Round 1 A/B: stability options (消融式, each independently testable)
    ema_v: bool = False,
    ema_tau: float = 0.005,
    huber_loss: bool = False,
    huber_delta: float = 10.0,
):
    """Run IQL training loop."""
    print("=" * 70)
    print("IQL Training — Offline RL on D_expert.npz")
    print("=" * 70)

    # --- Seed control (must be before any RNG usage) ---
    set_global_seed(seed, init_seed=init_seed, data_seed=data_seed)
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  τ (expectile): {tau}")
    print(f"  β (AWR temperature): {beta}")
    print(f"  γ (discount): {gamma}")
    print(f"  n_step (Q-chunking): {n_step}")
    print(f"  Oversample dist: {oversample_dist} (factor={oversample_factor})")
    print(f"  Reward shaping: {reward_shaping}")
    print(f"  Chunk size: {chunk_size} (true action chunking)")
    print(f"  Learning rate: {lr}")
    print(f"  Polyak: {polyak}")
    print(f"  Normalize rewards: {normalize_rewards}")
    print(f"  EMA on V network: {ema_v} (τ_ema={ema_tau})")
    print(f"  Huber loss on Q:  {huber_loss} (δ={huber_delta})")
    print(f"  Device: {device}")
    print()

    # Override output directory if specified
    global OUTPUT_DIR
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
    print(f"  Output dir: {OUTPUT_DIR}")

    # --- Load dataset ---
    print("Loading dataset...")
    oversample_range = None
    if oversample_dist:
        parts = [float(x) for x in oversample_dist.split(",")]
        oversample_range = (parts[0], parts[1])

    dataset = OfflineDataset(
        data_path=str(WORKSPACE / "data" / "D_expert.npz"),
        normalize_states=True,
        normalize_actions=False,  # Keep actions in [-1, 1] for tanh policy
        n_step=n_step,
        gamma=gamma,
        oversample_dist_range=oversample_range,
        oversample_factor=oversample_factor,
        reward_shaping=reward_shaping,
        chunk_size=chunk_size,
        normalize_rewards=normalize_rewards,
    )
    print(f"  Dataset size: {len(dataset)} transitions")
    print()

    # Save normalize_dict for evaluation consistency (P-1b)
    # Evaluation must reuse these stats to avoid train/eval distribution mismatch
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    normalize_dict_path = OUTPUT_DIR / "normalize_dict.json"
    with open(normalize_dict_path, "w") as f:
        json.dump(dataset.get_normalize_dict(), f, indent=2)
    print(f"  normalize_dict saved to {normalize_dict_path}")
    print()

    # Prepare success mask for Q separation evaluation
    # Use original states (not oversampled) for eval
    success_mask = np.zeros(len(dataset.states), dtype=bool)
    for ep_id in np.unique(dataset.episode_ids):
        ep_success = int(dataset.success_flags[ep_id]) if ep_id < len(dataset.success_flags) else 0
        if ep_success:
            mask = dataset.episode_ids == ep_id
            success_mask |= mask
    print(f"  Success transitions: {success_mask.sum()} / {len(success_mask)} "
          f"({success_mask.mean():.1%})")
    print()

    # --- Initialize agent ---
    print("Initializing IQL agent...")
    agent = IQLAgent(
        state_dim=12,
        action_dim=8,
        hidden_dim=256,
        tau=tau,
        beta=beta,
        gamma=gamma,
        polyak=polyak,
        lr_v=lr,
        lr_q=lr,
        lr_policy=lr,
        n_step=n_step,
        chunk_size=chunk_size,
        device=device,
        ema_v=ema_v,
        ema_tau=ema_tau,
        huber_loss=huber_loss,
        huber_delta=huber_delta,
    )

    # Count parameters
    n_params = sum(p.numel() for p in agent.v_net.parameters())
    n_params += sum(p.numel() for p in agent.q1_net.parameters()) * 2  # Q1+Q2
    n_params += sum(p.numel() for p in agent.policy.parameters())
    print(f"  Total parameters: {n_params:,}")
    print()

    # Ensure OUTPUT_DIR exists before any file writes (probe states, checkpoints)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Probe states for Q/V diagnostics ---
    if probe_states_path and Path(probe_states_path).exists():
        probe_data = np.load(probe_states_path)
        probe_states = probe_data["states"]
        probe_indices = probe_data["indices"]
        print(f"  Loaded probe states: {len(probe_states)} from {probe_states_path}")
    else:
        probe_states, probe_indices = sample_probe_states(dataset, n=100, seed=0)
        print(f"  Sampled probe states: {len(probe_states)} (seed=0, in-distribution)")
        # Save for reuse across training runs
        probe_save = OUTPUT_DIR / "probe_states.npz"
        if not probe_save.exists():
            np.savez(probe_save, states=probe_states, indices=probe_indices)
            print(f"    Saved to {probe_save}")

    # --- Dry-run: init hash for seed control verification ---
    init_hash, init_diag = compute_init_hash(agent, probe_states, device)
    print(f"  Init Q/V hash: {init_hash}")
    print(f"    (q1={init_diag['q1_mean']:.4f}, q2={init_diag['q2_mean']:.4f}, "
          f"v={init_diag['v_mean']:.4f})")
    print(f"    Verify: same seed→same hash, diff seed→diff hash")
    print()

    # --- Training loop ---
    print("Starting training...")
    print(f"  {'Epoch':>5} {'V_loss':>10} {'Q_loss':>10} {'P_loss':>10} "
          f"{'Exp_gap':>10} {'Adv_mean':>10} {'Adv_std':>10} "
          f"{'W_ent%':>8} {'ESS':>8} {'Q_gap':>8}")
    print(f"  {'-'*5} {'-'*10} {'-'*10} {'-'*10} "
          f"{'-'*10} {'-'*10} {'-'*10} "
          f"{'-'*8} {'-'*8} {'-'*8}")

    dataloader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)
    n_batches = len(dataloader)
    all_data = dataset.get_all_data()

    history = []
    awr_histograms = {}  # P-1c: AWR weight distribution at key epochs
    best_q_gap = 0
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_metrics = defaultdict(list)

        for batch in dataloader:
            metrics = agent.train_step(batch)
            for k, v in metrics.items():
                epoch_metrics[k].append(v)

        # Compute epoch averages
        avg = {k: np.mean(v) for k, v in epoch_metrics.items()}

        # Evaluate Q separation
        if epoch % eval_every == 0 or epoch == 1:
            q_eval = agent.evaluate_q_separation(all_data, success_mask=success_mask)
            avg.update(q_eval)
            if q_eval["q_gap"] > best_q_gap:
                best_q_gap = q_eval["q_gap"]
                agent.save(str(OUTPUT_DIR / "best_q_gap.pt"))
        else:
            q_eval = {"q_gap": 0}

        # Print metrics
        if epoch % eval_every == 0 or epoch <= 5 or epoch == n_epochs:
            print(f"  {epoch:5d} {avg.get('v_loss', 0):10.4f} "
                  f"{avg.get('q1_loss', 0):10.4f} "
                  f"{avg.get('policy_loss', 0):10.4f} "
                  f"{avg.get('expectile_gap', 0):10.4f} "
                  f"{avg.get('advantage_mean', 0):10.4f} "
                  f"{avg.get('advantage_std', 0):10.4f} "
                  f"{avg.get('weight_entropy_ratio', 0)*100:7.2f}% "
                  f"{avg.get('ess', 0):8.1f} "
                  f"{q_eval.get('q_gap', 0):8.1f}")

        # Risk detection
        if epoch % eval_every == 0:
            risks = []
            if avg.get("weight_entropy_ratio", 0) > 0.95:
                risks.append("AWR_WEIGHT_COLLAPSE (entropy > 95%)")
            # Expectile gap < 0 is EXPECTED for τ > 0.5 (V is upper-tail expectile).
            # Only flag if gap is extremely negative (> 2×advantage_std below 0).
            exp_gap = avg.get("expectile_gap", 0)
            adv_std = avg.get("advantage_std", 1.0)
            if exp_gap < -2.0 * adv_std:
                risks.append(f"EXPECTILE_GAP_TOO_NEGATIVE (gap={exp_gap:.2f} < -2×std)")
            if q_eval.get("q_gap", 0) < 10 and epoch > 20:
                risks.append("Q_NO_SEPARATION (gap < 10)")
            if avg.get("ess", 0) < 5 and epoch > 20:
                risks.append("ESS_COLLAPSE (< 5)")

            if risks:
                print(f"  ⚠ RISK: {'; '.join(risks)}")
                if "AWR_WEIGHT_COLLAPSE" in ";".join(risks) and beta < 10:
                    print(f"    → Consider increasing β from {beta} to {beta*3:.0f}")

        # Save checkpoint
        if epoch % save_every == 0 or epoch == n_epochs:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            agent.save(str(OUTPUT_DIR / f"checkpoint_epoch_{epoch}.pt"))

        # P-1c: AWR weight histogram at key epochs (sanity check for binarization)
        if epoch in awr_hist_epochs or epoch == n_epochs:
            awr_dist = agent.compute_awr_weight_distribution(all_data)
            awr_histograms[epoch] = awr_dist
            if epoch == 1 or epoch == n_epochs or epoch in awr_hist_epochs:
                print(f"  [AWR hist @ep{epoch}] adv=[{awr_dist['advantage_min']:.2f},"
                      f"{awr_dist['advantage_max']:.2f}] "
                      f"weight_clamp={awr_dist['frac_at_clamp_ceiling']:.1%} "
                      f"near_zero={awr_dist['frac_near_zero']:.1%} "
                      f"continuous={awr_dist['frac_continuous_middle']:.1%}")

        # Record history
        history.append({
            "epoch": epoch,
            **avg,
            "q_gap": q_eval.get("q_gap", 0),
        })

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/n_epochs:.2f}s/epoch)")
    print(f"Best Q-gap: {best_q_gap:.2f}")

    # --- Final evaluation ---
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    final_eval = agent.evaluate_q_separation(all_data, success_mask=success_mask)
    print(f"  Q success:      {final_eval['q_success']:.2f}")
    print(f"  Q failure:      {final_eval['q_failure']:.2f}")
    print(f"  Q gap:          {final_eval['q_gap']:.2f}")
    print(f"  Adv success:    {final_eval['advantage_success']:.4f}")
    print(f"  Adv failure:    {final_eval['advantage_failure']:.4f}")
    print(f"  Adv gap:        {final_eval['advantage_gap']:.4f}")

    # Final risk assessment
    print("\n  Risk Assessment:")
    last_metrics = history[-1]
    if last_metrics.get("weight_entropy_ratio", 0) > 0.95:
        print(f"    ⚠ AWR weight collapse: IQL degenerated to BC")
    else:
        print(f"    ✓ AWR weights have structure (entropy={last_metrics.get('weight_entropy_ratio', 0)*100:.1f}%)")

    if final_eval["q_gap"] > 100:
        print(f"    ✓ Q separates success/failure (gap={final_eval['q_gap']:.1f} > 100)")
    elif final_eval["q_gap"] > 50:
        print(f"    ~ Q separation moderate (gap={final_eval['q_gap']:.1f}, want > 100)")
    else:
        print(f"    ⚠ Q does not separate success/failure (gap={final_eval['q_gap']:.1f} < 50)")

    exp_gap = last_metrics.get("expectile_gap", 0)
    adv_std = last_metrics.get("advantage_std", 1.0)
    if exp_gap > -2.0 * adv_std:
        print(f"    ✓ V at expected level for τ>0.5 (expectile_gap={exp_gap:.4f}, "
              f"negative is normal — V is upper-tail expectile)")
    else:
        print(f"    ⚠ V too far above Q_target (expectile_gap={exp_gap:.4f}, "
              f"adv_std={adv_std:.4f}, threshold=-{2*adv_std:.4f})")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    agent.save(str(OUTPUT_DIR / "final_model.pt"))

    # --- Q/V diagnostics on probe states (Phase 7 training CV) ---
    final_qv_diag = compute_qv_diagnostics(agent, probe_states, device)
    final_hash, _ = compute_init_hash(agent, probe_states, device)
    print(f"\n  Q/V Probe Diagnostics (n={final_qv_diag['n_probe_states']}):")
    print(f"    Q1: {final_qv_diag['q1_mean']:.4f} ± {final_qv_diag['q1_std']:.4f}")
    print(f"    Q2: {final_qv_diag['q2_mean']:.4f} ± {final_qv_diag['q2_std']:.4f}")
    print(f"    V:  {final_qv_diag['v_mean']:.4f} ± {final_qv_diag['v_std']:.4f}")
    print(f"    Q1-Q2 gap: {final_qv_diag['q1_q2_gap_mean']:.4f}")
    print(f"    Final Q/V hash: {final_hash}")

    qv_diag_path = OUTPUT_DIR / "qv_diagnostics.json"
    with open(qv_diag_path, "w") as f:
        json.dump({
            "seed": seed,
            "init_seed": init_seed if init_seed is not None else seed,
            "data_seed": data_seed if data_seed is not None else seed,
            "init_hash": init_hash,
            "final_hash": final_hash,
            "init_qv": init_diag,
            "final_qv": final_qv_diag,
            "probe_states_count": final_qv_diag["n_probe_states"],
        }, f, indent=2)
    print(f"    Saved to {qv_diag_path}")

    results = {
        "n_epochs": n_epochs,
        "config": {
            "tau": tau, "beta": beta, "gamma": gamma,
            "lr": lr, "polyak": polyak, "batch_size": batch_size,
            "n_step": n_step,
            "oversample_dist": oversample_dist,
            "oversample_factor": oversample_factor,
            "reward_shaping": reward_shaping,
            "chunk_size": chunk_size,
            "seed": seed,
            "init_seed": init_seed if init_seed is not None else seed,
            "data_seed": data_seed if data_seed is not None else seed,
            "normalize_rewards": normalize_rewards,
            "ema_v": ema_v,
            "ema_tau": ema_tau,
            "huber_loss": huber_loss,
            "huber_delta": huber_delta,
        },
        "final_eval": final_eval,
        "best_q_gap": best_q_gap,
        "history": history,
        "init_hash": init_hash,
        "final_hash": final_hash,
    }
    with open(OUTPUT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_DIR}")

    # P-1c: Save AWR weight histograms (binarization sanity check)
    awr_hist_path = OUTPUT_DIR / "awr_weight_histograms.json"
    with open(awr_hist_path, "w") as f:
        json.dump(awr_histograms, f, indent=2)
    print(f"  AWR weight histograms saved to {awr_hist_path}")
    # Summary: compare epoch 1 vs final binarization
    if 1 in awr_histograms and n_epochs in awr_histograms:
        ep1 = awr_histograms[1]
        epf = awr_histograms[n_epochs]
        print(f"\n  AWR Binarization Summary (ep1 → ep{n_epochs}):")
        print(f"    frac@clamp_ceiling: {ep1['frac_at_clamp_ceiling']:.1%} → {epf['frac_at_clamp_ceiling']:.1%}")
        print(f"    frac@near_zero:     {ep1['frac_near_zero']:.1%} → {epf['frac_near_zero']:.1%}")
        print(f"    frac@continuous:    {ep1['frac_continuous_middle']:.1%} → {epf['frac_continuous_middle']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="IQL training")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.7, help="Expectile coefficient")
    parser.add_argument("--beta", type=float, default=3.0, help="AWR temperature")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--polyak", type=float, default=0.005)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--n_step", type=int, default=1,
                        help="Q-chunking bootstrap length h (1=standard IQL, 5=recommended)")
    parser.add_argument("--oversample_dist", type=str, default=None,
                        help="Distant state oversampling range as 'lo,hi' in meters "
                             "(e.g. '0.20,0.40' for 20-40cm)")
    parser.add_argument("--oversample_factor", type=int, default=3,
                        help="Oversampling multiplier for distant states")
    parser.add_argument("--reward_shaping", action="store_true", default=False,
                        help="Enable direction-aware dense reward shaping (Step 2c): "
                             "overshoot penalty + stay bonus + leave penalty")
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="v4: True action chunking. Actor outputs action_dim*h actions, "
                             "Critic takes (state, action_chunk). h=1=standard IQL.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: outputs/iql_v1")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master seed for all RNG (default 42). "
                             "Controls torch + numpy + random.")
    parser.add_argument("--init_seed", type=int, default=None,
                        help="Override seed for weight init (torch.manual_seed). "
                             "Default: use --seed. For Option B decoupling.")
    parser.add_argument("--data_seed", type=int, default=None,
                        help="Override seed for data shuffling (np.random.seed). "
                             "Default: use --seed. For Option B decoupling.")
    parser.add_argument("--probe_states_path", type=str, default=None,
                        help="Path to pre-saved probe_states.npz. If None, "
                             "samples 100 states from D_expert with seed=0.")
    parser.add_argument("--normalize_rewards", action="store_true", default=False,
                        help="Enable reward normalization (std-based). "
                             "Phase 7 P-0 found this is COUNTERPRODUCTIVE alone — "
                             "AWR binarization served as implicit regularizer. "
                             "Kept opt-in for ablation; default OFF.")
    parser.add_argument("--ema_v", action="store_true", default=False,
                        help="Phase 7 Round 1 A/B: maintain EMA copy of V network, "
                             "use it for Q-targets and advantage computation. "
                             "Stabilizes V multi-solution oscillation.")
    parser.add_argument("--ema_tau", type=float, default=0.005,
                        help="EMA soft-update rate for V network (default 0.005).")
    parser.add_argument("--huber_loss", action="store_true", default=False,
                        help="Phase 7 Round 1 A/B: replace MSE with smooth_L1 "
                             "(Huber) on Q update. Robust to heavy-tailed Q targets.")
    parser.add_argument("--huber_delta", type=float, default=10.0,
                        help="Huber delta (beta) for smooth_L1_loss. "
                             "Errors < delta → MSE-like; errors >= delta → L1-like. "
                             "Default 10.0 (proportional to reward std~17).")
    args = parser.parse_args()

    train(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        tau=args.tau,
        beta=args.beta,
        gamma=args.gamma,
        lr=args.lr,
        polyak=args.polyak,
        eval_every=args.eval_every,
        save_every=args.save_every,
        n_step=args.n_step,
        oversample_dist=args.oversample_dist,
        oversample_factor=args.oversample_factor,
        reward_shaping=args.reward_shaping,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        init_seed=args.init_seed,
        data_seed=args.data_seed,
        probe_states_path=args.probe_states_path,
        normalize_rewards=args.normalize_rewards,
        ema_v=args.ema_v,
        ema_tau=args.ema_tau,
        huber_loss=args.huber_loss,
        huber_delta=args.huber_delta,
    )


if __name__ == "__main__":
    main()
