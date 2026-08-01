#!/usr/bin/env python3
"""BC training on expert demos (oracle IK trajectories).

Trains V59's MLP head on expert demonstrations collected by running the IK
oracle as the place policy. The oracle achieves ~80% place rate (vs V59's 56%),
providing genuinely NEW information that V59 cannot self-generate.

Key differences from train_bc_only.py (self-imitation):
  - Data: D_expert.npz (oracle actions) instead of D_succ.npz (V59 actions)
  - Evaluation: 15-ep eval at each epoch, keep best model
  - Safety: rollback if place rate drops below 40%
  - log_std: set to -2.0 (std=0.135) after training (deterministic is optimal)

The backbone probe (backbone_probe.py) confirmed V59's ResNet-18 representation
is sufficient (89.2% success prediction). The bottleneck is in the policy head.
This script trains a better policy head using external expert information.

Usage:
    python train_bc_expert.py --demos data/D_expert.npz --save_path outputs/bc_expert_v1
    python train_bc_expert.py --demos data/D_expert.npz --n_epochs 30 --lr 1e-5
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).parent.resolve()

V59_MODEL_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/best_model.zip")
V59_VECNORM_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/vec_normalize.pkl")
GRASP_MODEL_PATH = str(WORKSPACE / "outputs/dapg_800k_v5/best/best_model.zip")
GRASP_VECNORM_PATH = str(WORKSPACE / "outputs/dapg_800k_v5/vec_normalize.pkl")


def load_expert_demos(path):
    """Load D_expert.npz → (images, states, actions) tensors."""
    data = np.load(path, allow_pickle=True)
    images = torch.as_tensor(data["images"], dtype=torch.float32)  # (N, 84, 84, 3)
    states = torch.as_tensor(data["states"], dtype=torch.float32)  # (N, 12)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)  # (N, 8)
    print(f"Expert demos loaded: {len(actions)} transitions")
    if "success_flags" in data:
        success_rate = float(data["success_flags"].mean())
        print(f"  Oracle success rate: {100*success_rate:.1f}%")
    if "n_placed" in data.files and "n_episodes" in data.files:
        n_placed = int(data["n_placed"])
        n_eps = int(data["n_episodes"])
        print(f"  Oracle place rate: {n_placed}/{n_eps} ({100*n_placed/n_eps:.1f}%)")
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


def freeze_backbone(model):
    """Freeze ResNet-18 features_extractor, only train MLP head."""
    fe = model.policy.features_extractor
    for p in fe.parameters():
        p.requires_grad = False
    for m in fe.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"Backbone frozen: {total} total params, {trainable} trainable")
    return trainable


def compute_bc_loss(model, images, states, actions, device):
    """Forward pass through policy → MSE between predicted and expert actions."""
    obs = {"image": images.to(device), "state": states.to(device)}
    features = model.policy.extract_features(obs)
    latent = model.policy.mlp_extractor.forward_actor(features)
    pred_actions = model.policy.action_net(latent)
    return nn.functional.mse_loss(pred_actions, actions.to(device))


def train_bc(model, images, states, actions, n_epochs, batch_size, lr, device,
             log_interval=1):
    """Supervised BC training loop. Returns loss history."""
    dataset = TensorDataset(images, states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    model.policy.set_training_mode(True)
    fe = model.policy.features_extractor
    fe.eval()  # Keep backbone in eval mode (frozen BN)

    loss_history = []
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_imgs, batch_states, batch_acts in loader:
            batch_imgs = batch_imgs.permute(0, 3, 1, 2).contiguous()
            loss = compute_bc_loss(model, batch_imgs, batch_states, batch_acts, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.3)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: bc_loss={avg_loss:.6f}")

    return loss_history


def quick_eval(model, vec_env, n_episodes=15, release_threshold=0.05):
    """Quick 15-episode evaluation. Returns (place_rate, mean_dist)."""
    from hierarchical_policy import HierarchicalPickPlacePolicy
    from gym_env.wrappers import VisionObs, FlattenObs
    import gymnasium
    import gym_env  # noqa: F401

    # Load grasp model
    grasp_factory = lambda: (lambda: (
        FlattenObs(gymnasium.make("PandaVLA-v0", reward_type="dense",
                                   gravity_comp=True,
                                   target_pos_range=[[0.35,0.15,0.22],[0.65,0.45,0.22]],
                                   domain_randomize=False))
    ))
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    grasp_vec = DummyVecEnv([grasp_factory()])
    grasp_vec = VecNormalize.load(GRASP_VECNORM_PATH, grasp_vec)
    grasp_vec.norm_reward = False
    grasp_vec.training = False
    grasp_model = __import__('stable_baselines3').PPO.load(GRASP_MODEL_PATH, env=grasp_vec, device="auto")

    policy = HierarchicalPickPlacePolicy(grasp_model, model)

    raw_env = DummyVecEnv([grasp_factory()])
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = release_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    n_placed = 0
    n_grabbed = 0
    final_dists = []

    for ep in range(n_episodes):
        inner.place_mode = False
        inner._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        prev_info = None
        max_lift = 0.0
        block_target_dist = float("inf")

        for step in range(500):
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                inner.place_mode = True
                inner._place_gravcomp_active = True
                inner.snap_block_to_hand()
                inner._arm_target = inner.data.qpos[inner._arm_qpos_adrs].copy()
                inner._gripper_target = float(inner.data.qpos[inner._finger_qpos_adrs].mean())
                inner.reward_type = "place_only"
                inner._place_approach_bonus_given = False
                inner._place_proximity_15_given = False
                inner._place_proximity_10_given = False
                inner._place_success = False
                inner._prev_block_target_dist = None
                inner._prev_block_height = None
                inner._use_gripper_target_check = True
                flatten_wrapper = raw_env.envs[0]
                inner_obs = inner._get_obs()
                raw_obs = flatten_wrapper.observation(inner_obs)[np.newaxis, :].astype(np.float32)

            if phase == "place":
                vision_obs = place_vision.observation(inner._get_obs())
                obs_batched = {
                    "image": vision_obs["image"][np.newaxis, ...],
                    "state": vision_obs["state"][np.newaxis, ...],
                }
                obs = vec_env.normalize_obs(obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)
            else:
                raw_obs_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_grasp[0, 8:11]
                raw_obs_grasp[0, 15] = np.linalg.norm(block_pos - np.array([0.5, 0.3, 0.2]))
                obs = grasp_vec.normalize_obs(raw_obs_grasp)
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(info[0].get("block_target_distance", block_target_dist))
            lift = max(0.0, float(info[0].get("block_height", 0.0)) - 0.22)
            if lift > max_lift:
                max_lift = lift
            if done[0]:
                break

        if first_place_step is not None and max_lift > 0.03:
            n_grabbed += 1
            if block_target_dist < release_threshold:
                n_placed += 1
        final_dists.append(block_target_dist)

    place_rate = n_placed / max(1, n_grabbed)
    mean_dist = float(np.mean(final_dists))
    return place_rate, mean_dist, n_placed, n_grabbed


def main():
    parser = argparse.ArgumentParser(description="BC training on expert demos")
    parser.add_argument("--demos", type=str, default="data/D_expert.npz")
    parser.add_argument("--load_model", type=str, default=V59_MODEL_PATH)
    parser.add_argument("--load_vecnorm", type=str, default=V59_VECNORM_PATH)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, default="BC_EXPERT_V1")
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--final_log_std", type=float, default=-2.0,
                        help="Set log_std after training (default -2.0, std=0.135)")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Eval every N epochs")
    parser.add_argument("--safety_threshold", type=float, default=0.40,
                        help="Rollback if place rate drops below this")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = WORKSPACE / args.save_path
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"BC Training on Expert Demos: {args.experiment_id}")
    print("=" * 60)
    print(f"Demos: {args.demos}")
    print(f"Model: {args.load_model}")
    print(f"Epochs: {args.n_epochs}, batch_size: {args.batch_size}, lr: {args.lr}")
    print(f"Save: {save_dir}")
    print(f"Eval interval: {args.eval_interval} epochs")
    print(f"Safety threshold: {args.safety_threshold}")
    print()

    # ---- Load expert demos ----
    print("Loading expert demos...")
    images, states, actions = load_expert_demos(str(WORKSPACE / args.demos))
    states = normalize_states(states, args.load_vecnorm)

    # ---- Load V59 model ----
    print("\nLoading V59 model...")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
    import gymnasium
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs

    env_factory = lambda: VisionObs(
        gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True,
                       target_pos_range=[[0.35,0.15,0.22],[0.65,0.45,0.22]],
                       domain_randomize=False),
        image_size=84)
    vec_env = DummyVecEnv([env_factory])
    vec_env = VecNormalize.load(args.load_vecnorm, vec_env)
    vec_env.norm_reward = False
    vec_env.training = False
    vec_env = VecTransposeImage(vec_env)

    model = PPO.load(args.load_model, env=vec_env, device=args.device)
    print("V59 model loaded.")

    # ---- Freeze backbone ----
    freeze_backbone(model)

    # ---- Initial BC loss ----
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_imgs = images[:320].permute(0, 3, 1, 2).contiguous()
    test_states = states[:320]
    test_acts = actions[:320]
    with torch.no_grad():
        init_loss = compute_bc_loss(model, test_imgs, test_states, test_acts, device)
    print(f"\nInitial BC loss (V59 vs expert): {init_loss.item():.6f}")
    print(f"  (This should be > 0 — expert actions differ from V59)")

    # ---- Check V59's current log_std ----
    log_std = model.policy.log_std.data
    print(f"V59 log_std: {log_std.mean().item():.4f} (std={log_std.exp().mean().item():.4f})")

    # ---- Initial evaluation ----
    print("\n--- Initial Evaluation (V59 baseline) ---")
    t0 = time.time()
    init_place_rate, init_mean_dist, n_placed, n_grabbed = quick_eval(
        model, vec_env, n_episodes=15)
    print(f"  V59 baseline: {n_placed}/{n_grabbed} placed ({100*init_place_rate:.1f}%), "
          f"mean_dist={init_mean_dist*100:.1f}cm ({time.time()-t0:.0f}s)")

    best_place_rate = init_place_rate
    best_epoch = 0
    best_state = None

    # ---- Training loop ----
    print(f"\n--- Training ({args.n_epochs} epochs) ---")
    all_results = {'init_loss': init_loss.item(), 'init_place_rate': init_place_rate,
                   'epochs': []}

    for epoch_block in range(0, args.n_epochs, args.eval_interval):
        end_epoch = min(epoch_block + args.eval_interval, args.n_epochs)
        n_train_epochs = end_epoch - epoch_block

        # Train for eval_interval epochs
        losses = train_bc(model, images, states, actions,
                          n_epochs=n_train_epochs, batch_size=args.batch_size,
                          lr=args.lr, device=device, log_interval=1)

        # Evaluate
        current_epoch = end_epoch
        print(f"\n--- Eval at epoch {current_epoch} ---")
        t0 = time.time()
        place_rate, mean_dist, n_placed, n_grabbed = quick_eval(
            model, vec_env, n_episodes=15)
        print(f"  Epoch {current_epoch}: {n_placed}/{n_grabbed} placed "
              f"({100*place_rate:.1f}%), mean_dist={mean_dist*100:.1f}cm "
              f"({time.time()-t0:.0f}s)")

        # Check BC loss
        with torch.no_grad():
            current_loss = compute_bc_loss(model, test_imgs, test_states, test_acts, device)
        print(f"  BC loss: {current_loss.item():.6f}")

        all_results['epochs'].append({
            'epoch': current_epoch,
            'bc_loss': current_loss.item(),
            'place_rate': place_rate,
            'mean_dist': mean_dist,
            'n_placed': n_placed,
            'n_grabbed': n_grabbed,
        })

        # Keep best model
        if place_rate > best_place_rate:
            best_place_rate = place_rate
            best_epoch = current_epoch
            best_state = {k: v.clone() for k, v in model.policy.state_dict().items()}
            print(f"  ★ NEW BEST! place_rate={100*place_rate:.1f}%")
            # Save best model
            model.save(str(save_dir / "best_model.zip"))
            print(f"  Saved best model to {save_dir / 'best_model.zip'}")

        # Safety rollback
        if place_rate < args.safety_threshold:
            print(f"\n  ⚠ SAFETY ROLLBACK: place_rate={100*place_rate:.1f}% < "
                  f"{100*args.safety_threshold:.0f}% threshold")
            if best_state is not None:
                model.policy.load_state_dict(best_state)
                print(f"  Rolled back to epoch {best_epoch} (place_rate={100*best_place_rate:.1f}%)")
            break

    # ---- Set log_std to smaller value ----
    print(f"\n--- Setting log_std to {args.final_log_std} (std={np.exp(args.final_log_std):.4f}) ---")
    with torch.no_grad():
        model.policy.log_std.data.fill_(args.final_log_std)

    # ---- Final evaluation ----
    print("\n--- Final Evaluation ---")
    t0 = time.time()
    final_place_rate, final_mean_dist, n_placed, n_grabbed = quick_eval(
        model, vec_env, n_episodes=15)
    print(f"  Final: {n_placed}/{n_grabbed} placed ({100*final_place_rate:.1f}%), "
          f"mean_dist={final_mean_dist*100:.1f}cm ({time.time()-t0:.0f}s)")

    all_results['final_place_rate'] = final_place_rate
    all_results['final_mean_dist'] = final_mean_dist
    all_results['best_place_rate'] = best_place_rate
    all_results['best_epoch'] = best_epoch

    # ---- Save final model ----
    model.save(str(save_dir / "final_model.zip"))
    print(f"\nSaved final model to {save_dir / 'final_model.zip'}")

    # ---- Save results ----
    results_path = save_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {results_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"V59 baseline:    {100*init_place_rate:.1f}% place rate")
    print(f"Best (epoch {best_epoch}): {100*best_place_rate:.1f}% place rate")
    print(f"Final:           {100*final_place_rate:.1f}% place rate")
    print(f"Improvement:     {100*(best_place_rate - init_place_rate):+.1f}%")

    if best_place_rate > init_place_rate + 0.05:
        print(f"\n✓ SUCCESS: Expert demos improved V59 by {100*(best_place_rate - init_place_rate):+.1f}%")
    elif best_place_rate > init_place_rate - 0.05:
        print(f"\n~ NEUTRAL: Expert demos did not significantly change V59 (within ±5%)")
    else:
        print(f"\n✗ FAILED: Expert demos degraded V59 by {100*(best_place_rate - init_place_rate):+.1f}%")


if __name__ == "__main__":
    main()
