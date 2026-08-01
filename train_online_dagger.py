#!/usr/bin/env python3
"""Online iterative DAgger training for the place policy.

DAgger (Ross et al. 2011) solves covariate shift by iteratively:
  1. Running the CURRENT policy to collect states from its distribution
  2. Labeling those states with EXPERT actions (analytical IK oracle)
  3. Aggregating with the existing dataset and retraining

This is the online variant: instead of collecting all DAgger data upfront
with a fixed policy, we re-collect after every retrain round so the state
distribution tracks the improving policy.

Pipeline:
  D = D_expert  (29,467 oracle transitions, pre-collected)
  model = BC epoch-5 model (22% place rate, NOT V59)
  for iter in range(10):
    1. COLLECT: Run current model for 50 episodes
       - At each place-phase step:
         a. Record (image, state) from env
         b. Call oracle.get_expert_action() to get expert label
         c. Execute MODEL's action (NOT oracle's) to advance env
       - Produces D_new with states from BC policy's distribution
    2. AGGREGATE: D = concat(D, D_new)
    3. RETRAIN: BC on aggregated D (5 epochs, lr=1e-5, frozen backbone)
    4. EVAL: 50-episode place rate
    5. SAFETY: if place_rate < 0.40, rollback to best model, break
    6. TRACK BEST: if place_rate > best, save best model
    7. LOG: per-iteration bc_loss, place_rate, n_collected, dataset_size

Key design choices:
  - Init from BC epoch-5 (NOT V59) — BC on expert demos failed at 22% due
    to covariate shift; DAgger fixes this by collecting on-policy states.
  - V59's VecNormalize statistics are reused for state normalization
    (the BC epoch-5 model was trained with these stats).
  - The oracle reads state from self.data (MuJoCo sim state) — it always
    labels the CURRENT env state, regardless of which policy advanced it.
  - 50-ep eval (not 15) to avoid high-variance false positives.

Usage:
    python train_online_dagger.py
    python train_online_dagger.py --max_iters 5 --n_collect_episodes 30
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
INIT_MODEL_PATH = str(WORKSPACE / "outputs/bc_expert_v1/final_model.zip")
EXPERT_DEMOS_PATH = str(WORKSPACE / "data/D_expert.npz")

LIFT_THRESHOLD = 0.03
TABLE_Z = 0.22
MAX_STEPS = 500
TARGET_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]


# ---------------------------------------------------------------------------
# Helper functions (copied from train_bc_expert.py)
# ---------------------------------------------------------------------------

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
    """Forward pass through policy -> MSE between predicted and expert actions.

    images: (B, C, H, W) — already in NCHW format.
    """
    obs = {"image": images.to(device), "state": states.to(device)}
    features = model.policy.extract_features(obs)
    latent = model.policy.mlp_extractor.forward_actor(features)
    pred_actions = model.policy.action_net(latent)
    return nn.functional.mse_loss(pred_actions, actions.to(device))


def train_bc(model, images, states, actions, n_epochs, batch_size, lr, device,
             log_interval=1):
    """Supervised BC training loop. Returns loss history.

    images: (N, H, W, C) float32 — will be permuted to NCHW per batch.
    """
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


# ---------------------------------------------------------------------------
# Evaluation (adapted from train_bc_expert.py, n_episodes=50)
# ---------------------------------------------------------------------------

def quick_eval(model, vec_env, n_episodes=50, release_threshold=0.05,
               grasp_model=None, grasp_vec=None, raw_env=None, inner=None,
               place_vision=None):
    """Evaluate place rate over n_episodes. Default 50 (avoids 15-ep variance).

    Returns (place_rate, mean_dist, n_placed, n_grabbed).

    Optionally accepts pre-loaded resources (grasp_model, grasp_vec, raw_env,
    inner, place_vision) to avoid reloading on every call. If any are None,
    they are created fresh inside this function.
    """
    from hierarchical_policy import HierarchicalPickPlacePolicy
    from gym_env.wrappers import VisionObs, FlattenObs
    import gymnasium
    import gym_env  # noqa: F401
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _grasp_factory():
        return lambda: FlattenObs(
            gymnasium.make("PandaVLA-v0", reward_type="dense",
                           gravity_comp=True,
                           target_pos_range=TARGET_RANGE,
                           domain_randomize=False))

    # Set up grasp model if not provided
    if grasp_model is None or grasp_vec is None:
        grasp_vec = DummyVecEnv([_grasp_factory()])
        grasp_vec = VecNormalize.load(GRASP_VECNORM_PATH, grasp_vec)
        grasp_vec.norm_reward = False
        grasp_vec.training = False
        grasp_model = PPO.load(GRASP_MODEL_PATH, env=grasp_vec, device="auto")

    # Set up eval env if not provided
    if raw_env is None or inner is None or place_vision is None:
        raw_env = DummyVecEnv([_grasp_factory()])
        inner = raw_env.envs[0].env.unwrapped
        inner._release_dist_threshold = release_threshold
        inner._release_height_threshold = float('inf')
        place_vision = VisionObs(inner, image_size=84)

    policy = HierarchicalPickPlacePolicy(grasp_model, model)

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

        for step in range(MAX_STEPS):
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

        if first_place_step is not None and max_lift > LIFT_THRESHOLD:
            n_grabbed += 1
            if block_target_dist < release_threshold:
                n_placed += 1
        final_dists.append(block_target_dist)

    place_rate = n_placed / max(1, n_grabbed)
    mean_dist = float(np.mean(final_dists))
    return place_rate, mean_dist, n_placed, n_grabbed


# ---------------------------------------------------------------------------
# DAgger data collection
# ---------------------------------------------------------------------------

def collect_dagger_data(model, oracle, place_vision, raw_env, inner, policy,
                        grasp_vec, vec_env, n_episodes=50, max_steps=MAX_STEPS):
    """Collect DAgger data: run BC policy, label with oracle actions.

    CRITICAL DIFFERENCE from collect_expert_demos.py:
      - Execute MODEL's action (BC policy) to advance env
      - Record ORACLE's action as the label

    This produces (state, oracle_action) pairs where states come from
    the BC policy's state distribution, solving covariate shift.

    Parameters
    ----------
    model : PPO
        The BC place model (updated in-place by the outer DAgger loop).
    oracle : DAggerOracle
        Analytical IK oracle bound to `inner`. Reads state from self.data.
    place_vision : VisionObs
        Wrapper to extract (image, state) from inner._get_obs().
    raw_env : DummyVecEnv
        Flatten-obs env used for stepping.
    inner : PandaVLAEnv
        Unwrapped inner env (for place-mode flags, phase transitions).
    policy : HierarchicalPickPlacePolicy
        Hierarchical policy using grasp_model + current BC model.
    grasp_vec : VecNormalize
        For normalizing grasp-phase observations.
    vec_env : VecNormalize / VecTransposeImage
        For normalizing place-phase observations (image + state).
    n_episodes : int
        Number of episodes to collect.
    max_steps : int
        Max steps per episode.

    Returns
    -------
    dict with keys:
        images  : (N, 84, 84, 3) uint8
        states  : (N, 12) float32  — raw, unnormalized
        actions : (N, 8) float32   — oracle labels
        n_grabbed : int
        n_placed  : int
    """
    all_images = []
    all_states = []
    all_actions = []
    n_grabbed = 0
    n_placed = 0

    for ep in range(n_episodes):
        # Reset (same as collect_expert_demos.py lines 162-170)
        inner.place_mode = False
        inner._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        prev_info = None
        max_lift = 0.0
        block_target_dist = float("inf")

        ep_images = []
        ep_states = []
        ep_actions = []

        for step in range(max_steps):
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                # Transition to place phase (same as collect_expert_demos.py 178-196)
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
                # --- DAgger: record oracle label, execute BC action ---
                vision_obs = place_vision.observation(inner._get_obs())

                # 1. Record (image, state) — raw, unnormalized (for training)
                ep_images.append(vision_obs["image"].copy())
                ep_states.append(vision_obs["state"].copy())

                # 2. Get oracle label (expert action for THIS state)
                #    Oracle reads from inner.data (MuJoCo sim state) — always
                #    labels the current env state regardless of which policy
                #    advanced the env.
                oracle_action = oracle.get_expert_action()
                ep_actions.append(oracle_action.copy())

                # 3. Get BC model's action (what the model WANTS to do)
                obs_batched = {
                    "image": vision_obs["image"][np.newaxis, ...],
                    "state": vision_obs["state"][np.newaxis, ...],
                }
                obs = vec_env.normalize_obs(obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

                # 4. Execute BC action (NOT oracle action) to advance env.
                #    This ensures states come from BC policy's distribution.
            else:
                # Grasp phase: use V59 grasp model (collect_expert_demos.py 212-217)
                raw_obs_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_grasp[0, 8:11]
                raw_obs_grasp[0, 15] = np.linalg.norm(block_pos - np.array([0.5, 0.3, 0.2]))
                obs = grasp_vec.normalize_obs(raw_obs_grasp)
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(info[0].get("block_target_distance", block_target_dist))
            lift = max(0.0, float(info[0].get("block_height", 0.0)) - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            if done[0]:
                break

        # Episode finished (same as collect_expert_demos.py 229-254)
        if first_place_step is not None and max_lift > LIFT_THRESHOLD:
            n_grabbed += 1
            placed = block_target_dist < oracle.release_threshold
            if placed:
                n_placed += 1
            # Only record data from episodes that entered the place phase
            # and successfully lifted the block (matching collect_expert_demos)
            all_images.extend(ep_images)
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)

    return {
        "images": np.array(all_images, dtype=np.uint8),
        "states": np.array(all_states, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "n_grabbed": n_grabbed,
        "n_placed": n_placed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Online iterative DAgger training for place policy")
    parser.add_argument("--init_model", type=str, default=INIT_MODEL_PATH,
                        help="Initial BC model (BC epoch-5, NOT V59)")
    parser.add_argument("--expert_demos", type=str, default=EXPERT_DEMOS_PATH,
                        help="Path to D_expert.npz (initial dataset)")
    parser.add_argument("--max_iters", type=int, default=10,
                        help="Max DAgger iterations")
    parser.add_argument("--n_collect_episodes", type=int, default=50,
                        help="Episodes per iteration for data collection")
    parser.add_argument("--n_train_epochs", type=int, default=5,
                        help="BC training epochs per iteration")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="BC learning rate")
    parser.add_argument("--batch_size", type=int, default=320,
                        help="BC batch size")
    parser.add_argument("--eval_episodes", type=int, default=50,
                        help="Episodes per evaluation (50 to avoid 15-ep variance)")
    parser.add_argument("--safety_threshold", type=float, default=0.40,
                        help="Rollback and stop if place rate drops below this")
    parser.add_argument("--save_path", type=str, default="outputs/online_dagger_v1",
                        help="Output directory for saved models")
    parser.add_argument("--oracle_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="v1=binary gripper, v2=V59-style gripper (-0.07)")
    parser.add_argument("--oracle_gain", type=float, default=2.0,
                        help="Proportional gain for oracle velocity control")
    parser.add_argument("--oracle_max_speed", type=float, default=0.5,
                        help="Max Cartesian speed (m/s) for oracle")
    parser.add_argument("--release_threshold", type=float, default=0.05,
                        help="Block-target distance threshold for place success")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = WORKSPACE / args.save_path
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Online Iterative DAgger Training")
    print("=" * 60)
    print(f"Init model (BC epoch-5): {args.init_model}")
    print(f"Expert demos (D_expert): {args.expert_demos}")
    print(f"VecNormalize stats:      {V59_VECNORM_PATH}")
    print(f"Max iterations:          {args.max_iters}")
    print(f"Collect episodes/iter:   {args.n_collect_episodes}")
    print(f"Train epochs/iter:       {args.n_train_epochs}")
    print(f"LR: {args.lr}, batch_size: {args.batch_size}")
    print(f"Eval episodes:           {args.eval_episodes}")
    print(f"Safety threshold:        {args.safety_threshold}")
    print(f"Oracle: v{args.oracle_version}, gain={args.oracle_gain}, "
          f"max_speed={args.oracle_max_speed}")
    print(f"Save path:               {save_dir}")
    print(f"Device:                  {args.device}")
    print()

    # ------------------------------------------------------------------
    # 1. Setup: load models, env, oracle
    # ------------------------------------------------------------------
    print("--- Loading models and environment ---")

    # Lazy imports (after env vars are set)
    import gymnasium
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs, FlattenObs
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, VecNormalize, VecTransposeImage)
    from hierarchical_policy import HierarchicalPickPlacePolicy
    from dagger_oracle import DAggerOracle, DAggerOracleV2

    def _grasp_factory():
        return lambda: FlattenObs(
            gymnasium.make("PandaVLA-v0", reward_type="dense",
                           gravity_comp=True,
                           target_pos_range=TARGET_RANGE,
                           domain_randomize=False))

    def _place_factory():
        return lambda: VisionObs(
            gymnasium.make("PandaVLA-v0", reward_type="dense",
                           gravity_comp=True,
                           target_pos_range=TARGET_RANGE,
                           domain_randomize=False),
            image_size=84)

    # --- Grasp model (for grasp phase) ---
    print("Loading grasp model...")
    grasp_vec = DummyVecEnv([_grasp_factory()])
    grasp_vec = VecNormalize.load(GRASP_VECNORM_PATH, grasp_vec)
    grasp_vec.norm_reward = False
    grasp_vec.training = False
    grasp_model = PPO.load(GRASP_MODEL_PATH, env=grasp_vec, device="auto")
    print(f"  Grasp model loaded from {GRASP_MODEL_PATH}")

    # --- BC epoch-5 model with V59's VecNormalize stats ---
    # The BC epoch-5 model was trained with V59's VecNormalize statistics,
    # so we use V59_VECNORM_PATH for the wrapper but load weights from
    # INIT_MODEL_PATH.
    print(f"Loading BC epoch-5 model from {args.init_model}...")
    print(f"  (using V59 VecNormalize stats from {V59_VECNORM_PATH})")
    place_vec = DummyVecEnv([_place_factory()])
    place_vec = VecNormalize.load(V59_VECNORM_PATH, place_vec)
    place_vec.norm_reward = False
    place_vec.training = False
    place_vec = VecTransposeImage(place_vec)
    model = PPO.load(args.init_model, env=place_vec, device=args.device)
    print("  BC epoch-5 model loaded.")

    # --- Raw eval/collection env (shared) ---
    # The oracle binds to `inner`, so we use the same inner for both
    # collection and evaluation. Both paths reset the env at the start
    # of each episode, so no state leaks between them.
    print("Setting up raw env for collection + eval...")
    raw_env = DummyVecEnv([_grasp_factory()])
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = args.release_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    # --- DAgger oracle (bound to inner) ---
    if args.oracle_version == "v2":
        oracle = DAggerOracleV2(
            inner, gain=args.oracle_gain,
            max_speed=args.oracle_max_speed,
            release_threshold=args.release_threshold)
        print(f"  Oracle: DAggerOracleV2 (Jacobian IK arm + V59-style gripper)")
    else:
        oracle = DAggerOracle(
            inner, gain=args.oracle_gain,
            max_speed=args.oracle_max_speed,
            release_threshold=args.release_threshold)
        print(f"  Oracle: DAggerOracle (Jacobian IK, binary gripper)")

    # --- Hierarchical policy (grasp_model + BC model) ---
    # The policy stores a reference to `model`, so after train_bc updates
    # the model in-place, the policy automatically uses the new weights.
    policy = HierarchicalPickPlacePolicy(grasp_model, model)

    # --- Freeze backbone of BC model ---
    freeze_backbone(model)

    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 2. Load D_expert as initial dataset
    # ------------------------------------------------------------------
    print("\n--- Loading D_expert (initial dataset) ---")
    data = np.load(args.expert_demos, allow_pickle=True)
    D_images = data["images"].astype(np.float32)   # (N, 84, 84, 3)
    D_states_raw = data["states"].astype(np.float32)  # (N, 12) raw
    D_actions = data["actions"].astype(np.float32)    # (N, 8)
    print(f"  D_expert: {len(D_actions)} transitions")
    if "n_placed" in data.files and "n_episodes" in data.files:
        n_placed_expert = int(data["n_placed"])
        n_eps_expert = int(data["n_episodes"])
        print(f"  Oracle place rate (at collection): "
              f"{n_placed_expert}/{n_eps_expert} "
              f"({100*n_placed_expert/max(1,n_eps_expert):.1f}%)")

    # Normalize states with V59 VecNormalize stats (same as D_new will be)
    D_states = normalize_states(
        torch.as_tensor(D_states_raw), V59_VECNORM_PATH).numpy()
    print(f"  States normalized with V59 VecNormalize stats.")
    print(f"  Shapes: images={D_images.shape}, states={D_states.shape}, "
          f"actions={D_actions.shape}")

    # Fixed test set for BC loss tracking (first 320 transitions of D_expert)
    n_test = min(320, len(D_images))
    test_imgs = torch.as_tensor(D_images[:n_test], dtype=torch.float32)
    test_imgs = test_imgs.permute(0, 3, 1, 2).contiguous()  # NCHW
    test_states = torch.as_tensor(D_states[:n_test], dtype=torch.float32)
    test_acts = torch.as_tensor(D_actions[:n_test], dtype=torch.float32)

    # ------------------------------------------------------------------
    # 3. Initial eval (BC epoch-5 baseline)
    # ------------------------------------------------------------------
    print(f"\n--- Initial Evaluation (BC epoch-5, {args.eval_episodes} eps) ---")
    t0 = time.time()
    init_place_rate, init_mean_dist, init_n_placed, init_n_grabbed = quick_eval(
        model, place_vec, n_episodes=args.eval_episodes,
        release_threshold=args.release_threshold,
        grasp_model=grasp_model, grasp_vec=grasp_vec,
        raw_env=raw_env, inner=inner, place_vision=place_vision)
    print(f"  BC epoch-5: {init_n_placed}/{init_n_grabbed} placed "
          f"({100*init_place_rate:.1f}%), "
          f"mean_dist={init_mean_dist*100:.1f}cm "
          f"({time.time()-t0:.0f}s)")

    # Initial BC loss
    with torch.no_grad():
        init_bc_loss = compute_bc_loss(
            model, test_imgs, test_states, test_acts, device)
    print(f"  Initial BC loss: {init_bc_loss.item():.6f}")

    best_place_rate = init_place_rate
    best_state = None
    best_iter = 0

    # ------------------------------------------------------------------
    # 4. DAgger loop
    # ------------------------------------------------------------------
    all_results = {
        "init_place_rate": init_place_rate,
        "init_bc_loss": init_bc_loss.item(),
        "iterations": [],
        "config": {
            "init_model": args.init_model,
            "expert_demos": args.expert_demos,
            "max_iters": args.max_iters,
            "n_collect_episodes": args.n_collect_episodes,
            "n_train_epochs": args.n_train_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "eval_episodes": args.eval_episodes,
            "safety_threshold": args.safety_threshold,
            "oracle_version": args.oracle_version,
            "oracle_gain": args.oracle_gain,
            "oracle_max_speed": args.oracle_max_speed,
            "release_threshold": args.release_threshold,
        },
    }

    place_rate = init_place_rate  # for final summary if loop doesn't run

    for iter_idx in range(args.max_iters):
        iter_num = iter_idx + 1
        print(f"\n{'='*60}")
        print(f"DAgger Iteration {iter_num}/{args.max_iters}")
        print(f"{'='*60}")

        # 4a. COLLECT
        print(f"\n[1/5] Collecting DAgger data ({args.n_collect_episodes} eps)...")
        t0 = time.time()
        D_new = collect_dagger_data(
            model, oracle, place_vision, raw_env, inner, policy,
            grasp_vec, place_vec,
            n_episodes=args.n_collect_episodes, max_steps=MAX_STEPS)
        n_new = len(D_new["images"])
        print(f"  Collected: {n_new} transitions, "
              f"{D_new['n_placed']}/{D_new['n_grabbed']} placed "
              f"({time.time()-t0:.0f}s)")
        if n_new == 0:
            print("  WARNING: No transitions collected this iteration!")

        # 4b. AGGREGATE
        print(f"\n[2/5] Aggregating dataset...")
        if n_new > 0:
            # Normalize D_new states with V59 VecNormalize stats before
            # aggregating with D_expert (which was also normalized).
            D_new_states_norm = normalize_states(
                torch.as_tensor(D_new["states"], dtype=torch.float32),
                V59_VECNORM_PATH).numpy()
            D_images = np.concatenate([D_images, D_new["images"].astype(np.float32)])
            D_states = np.concatenate([D_states, D_new_states_norm])
            D_actions = np.concatenate([D_actions, D_new["actions"].astype(np.float32)])
        print(f"  Aggregated dataset: {len(D_images)} transitions")

        # 4c. RETRAIN
        print(f"\n[3/5] BC training ({args.n_train_epochs} epochs, "
              f"lr={args.lr}, batch_size={args.batch_size})...")
        images_t = torch.as_tensor(D_images, dtype=torch.float32)
        states_t = torch.as_tensor(D_states, dtype=torch.float32)
        actions_t = torch.as_tensor(D_actions, dtype=torch.float32)
        t0 = time.time()
        losses = train_bc(
            model, images_t, states_t, actions_t,
            n_epochs=args.n_train_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, log_interval=1)
        print(f"  Training done ({time.time()-t0:.0f}s), "
              f"final loss={losses[-1]:.6f}")

        # 4d. EVAL
        print(f"\n[4/5] Evaluation ({args.eval_episodes} eps)...")
        t0 = time.time()
        place_rate, mean_dist, n_placed, n_grabbed = quick_eval(
            model, place_vec, n_episodes=args.eval_episodes,
            release_threshold=args.release_threshold,
            grasp_model=grasp_model, grasp_vec=grasp_vec,
            raw_env=raw_env, inner=inner, place_vision=place_vision)
        print(f"  Iter {iter_num}: {n_placed}/{n_grabbed} placed "
              f"({100*place_rate:.1f}%), "
              f"mean_dist={mean_dist*100:.1f}cm ({time.time()-t0:.0f}s)")

        # 4e. BC loss (on fixed test set)
        with torch.no_grad():
            bc_loss = compute_bc_loss(
                model, test_imgs, test_states, test_acts, device)
        print(f"  BC loss (test): {bc_loss.item():.6f}")

        print(f"\n  >> Iter {iter_num}: place_rate={100*place_rate:.1f}%, "
              f"bc_loss={bc_loss.item():.6f}, "
              f"n_collected={n_new}, dataset_size={len(D_images)}")

        # 4f. TRACK BEST
        if place_rate > best_place_rate:
            best_place_rate = place_rate
            best_iter = iter_num
            best_state = {k: v.clone() for k, v in model.policy.state_dict().items()}
            model.save(str(save_dir / "best_model.zip"))
            print(f"  * NEW BEST! {100*place_rate:.1f}% "
                  f"(saved to {save_dir / 'best_model.zip'})")

        # 4g. SAFETY
        if place_rate < args.safety_threshold:
            print(f"  ! SAFETY ROLLBACK: {100*place_rate:.1f}% < "
                  f"{100*args.safety_threshold:.0f}% threshold")
            if best_state is not None:
                model.policy.load_state_dict(best_state)
                print(f"  Rolled back to iter {best_iter} "
                      f"({100*best_place_rate:.1f}%)")
            all_results["iterations"].append({
                "iter": iter_num,
                "place_rate": place_rate,
                "bc_loss": bc_loss.item(),
                "n_collected": n_new,
                "dataset_size": len(D_images),
                "mean_dist": mean_dist,
                "rolled_back": True,
            })
            break

        # 4h. RECORD
        all_results["iterations"].append({
            "iter": iter_num,
            "place_rate": place_rate,
            "bc_loss": bc_loss.item(),
            "n_collected": n_new,
            "dataset_size": len(D_images),
            "mean_dist": mean_dist,
            "rolled_back": False,
        })

        # 4i. BREAKTHROUGH check
        if place_rate > 0.56:
            print(f"  ** BREAKTHROUGH! Exceeded V59 baseline (56%)!")

    # ------------------------------------------------------------------
    # 5. Save final model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Saving final model...")
    print(f"{'='*60}")
    model.save(str(save_dir / "final_model.zip"))
    print(f"  Saved to {save_dir / 'final_model.zip'}")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    all_results["best_place_rate"] = best_place_rate
    all_results["best_iter"] = best_iter
    all_results["final_place_rate"] = place_rate
    all_results["final_dataset_size"] = len(D_images)
    results_path = save_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved to {results_path}")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Initial (BC epoch-5): ~22% place rate "
          f"(measured: {100*init_place_rate:.1f}%)")
    print(f"Best (iter {best_iter}): {100*best_place_rate:.1f}% place rate")
    print(f"Final:                {100*place_rate:.1f}% place rate")
    print(f"V59 baseline:         56%")
    print(f"Dataset size:         {len(D_images)} transitions "
          f"(started with {len(data['actions'])})")

    if best_place_rate > 0.56:
        print(f"\n** BREAKTHROUGH! DAgger exceeded V59 by "
              f"{100*(best_place_rate-0.56):+.1f}%")
    elif best_place_rate > 0.40:
        print(f"\n~ PARTIAL: DAgger improved but did not exceed V59 "
              f"(best={100*best_place_rate:.1f}% vs 56%)")
    else:
        print(f"\nFAILED: DAgger did not improve over BC epoch-5 "
              f"(best={100*best_place_rate:.1f}%)")


if __name__ == "__main__":
    main()
