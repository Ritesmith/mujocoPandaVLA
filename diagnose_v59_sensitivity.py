"""Deep diagnostic: Why is V59 so sensitive to gradient perturbation?

Priority 1 analyses:
1. Action comparison: V59 vs DAGGER_V1 on identical states
2. Per-dimension action shift analysis
3. State-correlated failure analysis (which states cause largest shift)
4. Loss landscape perturbation (place_rate vs noise magnitude)

Usage:
    python diagnose_v59_sensitivity.py
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecTransposeImage


def load_dagger_data(path):
    """Load D_dagger.npz."""
    data = np.load(path, allow_pickle=True)
    images = torch.as_tensor(data["images"], dtype=torch.float32)  # (N, 84, 84, 3)
    states = torch.as_tensor(data["states"], dtype=torch.float32)  # (N, 12)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)  # (N, 8) oracle actions
    print(f"Data loaded: {len(actions)} transitions")
    print(f"  Images: {images.shape}, States: {states.shape}, Actions: {actions.shape}")
    return images, states, actions


def normalize_states(states, vecnorm_path):
    """Normalize states using V59's VecNormalize statistics."""
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        print("Warning: no vec_normalize.pkl, states used as-is")
        return states
    import pickle
    with open(vecnorm_path, "rb") as f:
        vecnorm = pickle.load(f)
    obs_rms = vecnorm.obs_rms
    epsilon = vecnorm.epsilon
    clip_obs = vecnorm.clip_obs
    if isinstance(obs_rms, dict) and "state" in obs_rms:
        rms = obs_rms["state"]
        mean = torch.as_tensor(rms.mean, dtype=torch.float32)
        var = torch.as_tensor(rms.var, dtype=torch.float32)
        states = (states - mean) / torch.sqrt(var + epsilon)
        states = torch.clamp(states, -clip_obs, clip_obs)
    else:
        print(f"Warning: obs_rms format not recognized: {type(obs_rms)}")
    return states


def load_model(path, vecnorm_path, device="cuda"):
    """Load a DAPGPPO model with VecNormalize."""
    from train_dapg import DAPGPPO
    from train_place_policy import make_env
    import functools
    import pickle

    grasp_states_path = str(WORKSPACE / "outputs/grasp_states_v5_500.pkl")
    grasp_states = None
    if os.path.exists(grasp_states_path):
        with open(grasp_states_path, "rb") as f:
            grasp_states = pickle.load(f)

    target_pos_range = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]
    env_kwargs = dict(
        grasp_states=grasp_states,
        release_threshold=0.05,
        target_pos_range=target_pos_range,
        vision_mode=True,
        domain_randomize=False,
        better_reward=False,
        use_pbrs=False,
        pbrs_alpha=1.0, pbrs_beta=0.0, pbrs_scale=0.5,
    )
    train_env = DummyVecEnv([functools.partial(make_env, **env_kwargs)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        norm_obs_keys=["state"],
    )
    if vecnorm_path and os.path.exists(vecnorm_path):
        train_env = VecNormalize.load(vecnorm_path, train_env)
    train_env = VecTransposeImage(train_env)

    model = DAPGPPO.load(
        path,
        env=train_env,
        device=device,
        demo_obs=None,
        demo_actions=None,
    )
    return model


def get_model_actions(model, images, states, device, batch_size=320):
    """Get model's predicted actions for all data points."""
    model.policy.set_training_mode(False)
    model.policy.eval()

    all_actions = []
    n = len(images)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_imgs = images[i:i+batch_size].permute(0, 3, 1, 2).contiguous().to(device)
            batch_states = states[i:i+batch_size].to(device)
            obs = {"image": batch_imgs, "state": batch_states}
            features = model.policy.extract_features(obs)
            latent = model.policy.mlp_extractor.forward_actor(features)
            pred_actions = model.policy.action_net(latent)
            all_actions.append(pred_actions.cpu())

    return torch.cat(all_actions, dim=0)


def analysis_1_action_comparison(v59_model, dagger_model, images, states, oracle_actions, device):
    """Analysis 1: Compare V59 vs DAGGER_V1 vs Oracle actions on identical states."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Action Comparison (V59 vs DAGGER_V1 vs Oracle)")
    print("=" * 70)

    # Sample a subset for comparison (full 45k would be slow)
    n_total = len(images)
    n_sample = min(2000, n_total)
    indices = np.random.RandomState(42).choice(n_total, n_sample, replace=False)
    indices = np.sort(indices)

    sample_imgs = images[indices]
    sample_states = states[indices]
    sample_oracle = oracle_actions[indices]

    print(f"Sampling {n_sample} transitions for comparison...")

    # Get V59 actions
    print("Computing V59 actions...")
    v59_actions = get_model_actions(v59_model, sample_imgs, sample_states, device)

    # Get DAGGER_V1 actions
    print("Computing DAGGER_V1 actions...")
    dagger_actions = get_model_actions(dagger_model, sample_imgs, sample_states, device)

    # Compute differences (convert all to numpy)
    v59_np = v59_actions.numpy()
    dagger_np = dagger_actions.numpy()
    oracle_np = sample_oracle.numpy()

    v59_vs_dagger = dagger_np - v59_np
    v59_vs_oracle = oracle_np - v59_np
    dagger_vs_oracle = oracle_np - dagger_np

    # Per-dimension analysis
    dim_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]

    print(f"\n--- Per-Dimension Action Means ---")
    print(f"{'Dim':<10} {'V59 mean':>10} {'DAGGER mean':>12} {'Oracle mean':>12} {'V59→D shift':>12} {'D-Oracle':>10}")
    for i, name in enumerate(dim_names):
        v59_m = v59_actions[:, i].mean().item()
        dag_m = dagger_actions[:, i].mean().item()
        orc_m = sample_oracle[:, i].mean().item()
        shift = dag_m - v59_m
        d_orc = (dagger_actions[:, i] - sample_oracle[:, i]).mean().item()
        print(f"{name:<10} {v59_m:>10.4f} {dag_m:>12.4f} {orc_m:>12.4f} {shift:>+12.4f} {d_orc:>+10.4f}")

    print(f"\n--- Per-Dimension Action Std ---")
    print(f"{'Dim':<10} {'V59 std':>10} {'DAGGER std':>12} {'Oracle std':>12}")
    for i, name in enumerate(dim_names):
        v59_s = v59_actions[:, i].std().item()
        dag_s = dagger_actions[:, i].std().item()
        orc_s = sample_oracle[:, i].std().item()
        print(f"{name:<10} {v59_s:>10.4f} {dag_s:>12.4f} {orc_s:>12.4f}")

    # MSE comparisons
    mse_v59_dagger = np.mean(v59_vs_dagger ** 2)
    mse_v59_oracle = np.mean(v59_vs_oracle ** 2)
    mse_dagger_oracle = np.mean(dagger_vs_oracle ** 2)

    print(f"\n--- MSE Comparisons ---")
    print(f"  V59 vs DAGGER_V1:  {mse_v59_dagger:.6f}  (policy shift)")
    print(f"  V59 vs Oracle:     {mse_v59_oracle:.6f}  (original gap)")
    print(f"  DAGGER_V1 vs Oracle: {mse_dagger_oracle:.6f}  (remaining gap)")
    print(f"  Gap closed: {(1 - mse_dagger_oracle/mse_v59_oracle)*100:.1f}%")

    # Per-dimension MSE
    print(f"\n--- Per-Dimension MSE (V59 vs DAGGER_V1) ---")
    for i, name in enumerate(dim_names):
        dim_mse = np.mean(v59_vs_dagger[:, i] ** 2)
        print(f"  {name:<10}: {dim_mse:.6f}")

    # Identify states with largest action shift
    action_shifts = np.linalg.norm(v59_vs_dagger, axis=1)  # (N,)
    sorted_idx = np.argsort(action_shifts)[::-1]  # largest shift first

    print(f"\n--- Top 10 States with Largest V59→DAGGER Action Shift ---")
    print(f"{'Rank':<6} {'Shift':>8} {'State (12-dim)':>60}")
    for rank, idx in enumerate(sorted_idx[:10]):
        shift = action_shifts[idx]
        state = sample_states[idx].numpy()
        state_str = "[" + ",".join(f"{x:.3f}" for x in state) + "]"
        print(f"{rank+1:<6} {shift:>8.4f} {state_str:>60}")

    # Correlation between action shift and state features
    print(f"\n--- Correlation: State Features vs Action Shift ---")
    state_feature_names = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "grip", "dist", "tx", "ty", "tz"]
    state_np = sample_states.numpy()
    for i, name in enumerate(state_feature_names):
        corr = np.corrcoef(state_np[:, i], action_shifts)[0, 1]
        print(f"  {name:<6}: r={corr:>+.4f}")

    # Analyze: does DAGGER_V1 shift toward oracle?
    print(f"\n--- Direction Analysis ---")
    # V59→Oracle direction
    v59_to_oracle = oracle_np - v59_np
    # V59→DAGGER direction
    v59_to_dagger = dagger_np - v59_np

    # Cosine similarity between the two directions
    dot_product = np.sum(v59_to_oracle * v59_to_dagger, axis=1)
    norm_oracle = np.linalg.norm(v59_to_oracle, axis=1)
    norm_dagger = np.linalg.norm(v59_to_dagger, axis=1)
    cosine_sim = dot_product / (norm_oracle * norm_dagger + 1e-8)

    print(f"  Cosine sim (V59→Oracle vs V59→DAGGER): mean={cosine_sim.mean():.4f}, std={cosine_sim.std():.4f}")
    print(f"  Fraction with cos > 0.5: {(cosine_sim > 0.5).mean()*100:.1f}%")
    print(f"  Fraction with cos < 0:   {(cosine_sim < 0).mean()*100:.1f}%")

    if cosine_sim.mean() > 0.5:
        print(f"  → DAGGER_V1 shifted TOWARD oracle (correct direction)")
    elif cosine_sim.mean() > 0:
        print(f"  → DAGGER_V1 shifted partially toward oracle (weak alignment)")
    else:
        print(f"  → DAGGER_V1 shifted AWAY from oracle (wrong direction!)")

    # Gripper action analysis (critical for dropping)
    print(f"\n--- Gripper Action Analysis ---")
    v59_grip = v59_actions[:, 7].numpy()
    dag_grip = dagger_actions[:, 7].numpy()
    orc_grip = sample_oracle[:, 7].numpy()

    print(f"  V59 gripper:     mean={v59_grip.mean():.4f}, std={v59_grip.std():.4f}, min={v59_grip.min():.4f}, max={v59_grip.max():.4f}")
    print(f"  DAGGER gripper:  mean={dag_grip.mean():.4f}, std={dag_grip.std():.4f}, min={dag_grip.min():.4f}, max={dag_grip.max():.4f}")
    print(f"  Oracle gripper:  mean={orc_grip.mean():.4f}, std={orc_grip.std():.4f}, min={orc_grip.min():.4f}, max={orc_grip.max():.4f}")

    # Fraction where DAGGER gripper < 0 (open) but oracle says close (>0)
    dagger_open_wrong = ((dag_grip < 0) & (orc_grip > 0)).mean()
    v59_open_wrong = ((v59_grip < 0) & (orc_grip > 0)).mean()
    print(f"  V59 opens gripper when oracle says close: {v59_open_wrong*100:.1f}%")
    print(f"  DAGGER opens gripper when oracle says close: {dagger_open_wrong*100:.1f}%")

    # Check if DAGGER_V1 ever outputs negative gripper (premature release)
    dagger_negative_grip = (dag_grip < 0).mean()
    v59_negative_grip = (v59_grip < 0).mean()
    print(f"  V59 negative gripper output: {v59_negative_grip*100:.1f}%")
    print(f"  DAGGER negative gripper output: {dagger_negative_grip*100:.1f}%")

    return {
        "mse_v59_dagger": float(mse_v59_dagger),
        "mse_v59_oracle": float(mse_v59_oracle),
        "mse_dagger_oracle": float(mse_dagger_oracle),
        "gap_closed_pct": float((1 - mse_dagger_oracle/mse_v59_oracle)*100),
        "cosine_sim_mean": float(cosine_sim.mean()),
        "v59_gripper_mean": float(v59_grip.mean()),
        "dagger_gripper_mean": float(dag_grip.mean()),
        "oracle_gripper_mean": float(orc_grip.mean()),
    }


def analysis_2_weight_perturbation(v59_model, images, states, device):
    """Analysis 2: How sensitive is V59 to weight perturbation?

    Add Gaussian noise of varying magnitude to MLP head weights,
    measure how much the action output changes.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Weight Perturbation Sensitivity")
    print("=" * 70)

    # Get baseline actions
    n_sample = min(500, len(images))
    indices = np.random.RandomState(42).choice(len(images), n_sample, replace=False)
    indices = np.sort(indices)
    sample_imgs = images[indices]
    sample_states = states[indices]

    print(f"Sampling {n_sample} transitions for perturbation analysis...")

    print("Computing baseline V59 actions...")
    baseline_actions = get_model_actions(v59_model, sample_imgs, sample_states, device)

    # Get MLP head parameters (trainable params)
    mlp_params = [(name, p) for name, p in v59_model.policy.named_parameters() if p.requires_grad]
    print(f"MLP head parameters: {len(mlp_params)} tensors, {sum(p.numel() for _, p in mlp_params)} total params")

    # Save original weights
    original_weights = {name: p.data.clone() for name, p in mlp_params}

    noise_levels = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    results = []

    for noise_std in noise_levels:
        if noise_std == 0.0:
            perturbed_actions = baseline_actions
            mse = 0.0
        else:
            # Add noise to MLP weights
            for name, p in mlp_params:
                noise = torch.randn_like(p.data) * noise_std * p.data.std()
                p.data = original_weights[name] + noise

            perturbed_actions = get_model_actions(v59_model, sample_imgs, sample_states, device)
            diff = (perturbed_actions - baseline_actions).numpy()
            mse = float(np.mean(diff ** 2))

            # Restore original weights
            for name, p in mlp_params:
                p.data = original_weights[name].clone()

        results.append({"noise_std": noise_std, "action_mse": mse})
        print(f"  Noise std={noise_std:.1e}: action MSE={mse:.6f}")

    # Restore original weights (safety)
    for name, p in mlp_params:
        p.data = original_weights[name].clone()

    print(f"\n--- Sensitivity Summary ---")
    # Find the noise level where action MSE exceeds 0.1 (significant change)
    threshold = 0.1
    for r in results:
        if r["action_mse"] > threshold:
            print(f"  Action MSE > {threshold} at noise_std={r['noise_std']:.1e}")
            break
    else:
        print(f"  Action MSE never exceeded {threshold} (V59 is robust to weight perturbation)")

    return results


def analysis_3_layer_wise_sensitivity(v59_model, dagger_model, images, states, device):
    """Analysis 3: Which MLP layers shifted the most during DAgger training?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Layer-Wise Weight Shift (V59 → DAGGER_V1)")
    print("=" * 70)

    v59_params = dict(v59_model.policy.named_parameters())
    dagger_params = dict(dagger_model.policy.named_parameters())

    # Compare only trainable params (MLP head)
    trainable_names = [name for name, p in v59_params.items() if p.requires_grad]

    print(f"{'Layer':<45} {'V59 norm':>10} {'DAGGER norm':>12} {'Shift norm':>12} {'Shift %':>10}")
    print("-" * 90)

    layer_shifts = []
    for name in trainable_names:
        v59_w = v59_params[name].data
        dag_w = dagger_params[name].data
        shift = dag_w - v59_w

        v59_norm = v59_w.norm().item()
        dag_norm = dag_w.norm().item()
        shift_norm = shift.norm().item()
        shift_pct = (shift_norm / (v59_norm + 1e-8)) * 100

        print(f"{name:<45} {v59_norm:>10.4f} {dag_norm:>12.4f} {shift_norm:>12.4f} {shift_pct:>9.1f}%")

        layer_shifts.append({
            "layer": name,
            "v59_norm": v59_norm,
            "dagger_norm": dag_norm,
            "shift_norm": shift_norm,
            "shift_pct": shift_pct,
        })

    # Identify most-shifted layers
    layer_shifts.sort(key=lambda x: x["shift_pct"], reverse=True)
    print(f"\n--- Most Shifted Layers ---")
    for ls in layer_shifts[:5]:
        print(f"  {ls['layer']:<45} shift={ls['shift_pct']:.1f}%")

    return layer_shifts


def main():
    print("=" * 70)
    print("V59 Sensitivity Deep Diagnostic")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    data_path = str(WORKSPACE / "data/D_dagger.npz")
    images, states, oracle_actions = load_dagger_data(data_path)

    # Normalize states
    vecnorm_path = str(WORKSPACE / "outputs/place_policy_v59/best_hier/vec_normalize.pkl")
    states = normalize_states(states, vecnorm_path)

    # Load models
    v59_model_path = str(WORKSPACE / "outputs/place_policy_v59/best_hier/best_model.zip")
    dagger_model_path = str(WORKSPACE / "outputs/place_policy_dagger_v1/best_model.zip")

    print(f"\nLoading V59 model...")
    v59_model = load_model(v59_model_path, vecnorm_path, device=str(device))
    print(f"Loading DAGGER_V1 model...")
    dagger_model = load_model(dagger_model_path, vecnorm_path, device=str(device))

    # Run analyses
    results_1 = analysis_1_action_comparison(v59_model, dagger_model, images, states, oracle_actions, device)
    results_2 = analysis_2_weight_perturbation(v59_model, images, states, device)
    results_3 = analysis_3_layer_wise_sensitivity(v59_model, dagger_model, images, states, device)

    # Save results
    output_path = str(WORKSPACE / "outputs/diagnose_v59_sensitivity_results.json")
    all_results = {
        "analysis_1_action_comparison": results_1,
        "analysis_2_weight_perturbation": results_2,
        "analysis_3_layer_wise_sensitivity": results_3,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("Diagnostic Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
