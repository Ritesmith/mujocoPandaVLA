#!/usr/bin/env python3
"""Train ConditionalUNet1D Diffusion Policy on cached V59 ResNet-18 features.

Implements the full training loop per the user's spec:
  - DDPMScheduler (squaredcos_cap_v2, 100 train steps, epsilon prediction)
  - DDIMScheduler (20 inference steps for prototype)
  - EMAModel (decay=0.9999, diffusers API)
  - AdamW (lr=1e-4, betas=[0.95,0.999], weight_decay=1e-6)
  - Cosine LR scheduler with 500-step warmup
  - 8-bit AdamW (bitsandbytes) — halves optimizer state to fit [512,1024,2048] on 6GB GPU
  - AMP disabled (float16 overflows in FiLM MLP with ResNet features up to ±7.2)
  - Gradient clipping (max_norm=1.0)
  - Top-K checkpoint saving (top-3 by place_rate)
  - Trajectory smoothness metrics (action_jerk, chunk_transition_variance)
  - Inline eval every N epochs (5-15 episodes)
  - --finetune_backbone flag (reserved for full config)

Breakthrough: place_rate >= 60% → save breakthrough.ckpt, stop early.
Safety: place_rate < 5% for 3 consecutive evals → stop early.

Usage:
    # Smoke test (1 epoch, 5-ep eval)
    python train_diffusion_policy.py --n_epochs 1 --eval_every 1 --eval_episodes 5

    # Prototype (50 epochs, eval every 10, 5-ep eval)
    python train_diffusion_policy.py --n_epochs 50 --eval_every 10 --eval_episodes 5

    # Full prototype (100 epochs, eval every 5, 15-ep eval)
    python train_diffusion_policy.py --n_epochs 100 --eval_every 5 --eval_episodes 15
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

# Fix bitsandbytes CUDA 13 library path (must preload before bitsandbytes import)
import ctypes
_nvidia_cu13 = "/home/w/miniconda3/envs/vla/lib/python3.10/site-packages/nvidia/cu13/lib"
for _lib in ["libnvJitLink.so.13", "libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"]:
    _full = os.path.join(_nvidia_cu13, _lib)
    if os.path.exists(_full):
        try:
            ctypes.cdll.LoadLibrary(_full)
        except OSError:
            pass

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

WORKSPACE = Path(__file__).parent.resolve()

V59_MODEL_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/best_model.zip")
V59_VECNORM_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/vec_normalize.pkl")
GRASP_MODEL_PATH = str(WORKSPACE / "outputs/dapg_800k_v5/best/best_model.zip")
GRASP_VECNORM_PATH = str(WORKSPACE / "outputs/dapg_800k_v5/vec_normalize.pkl")
FEATURES_PATH = str(WORKSPACE / "data/D_expert_features.npz")
DEFAULT_SAVE_PATH = str(WORKSPACE / "outputs/diffusion_policy_v1")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActionChunkDataset(Dataset):
    """Builds (obs_window, action_chunk) pairs from sequential feature data.

    For each episode, slides a window of size (T_obs + T_pred) over the
    sequence, producing (obs_window[T_obs, 524], action_chunk[T_pred, 8]) pairs.
    """

    def __init__(self, features, actions, episode_ids,
                 T_obs=2, T_pred=16, action_min=None, action_max=None):
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.pairs = []  # list of (obs_window_idx_start, action_chunk_idx_start)

        # Group transitions by episode
        ep_to_indices = {}
        for i, ep in enumerate(episode_ids):
            ep_to_indices.setdefault(int(ep), []).append(i)

        # Build sliding window pairs
        for ep_id, indices in ep_to_indices.items():
            ep_len = len(indices)
            window = T_obs + T_pred
            if ep_len < window:
                continue  # skip episodes too short
            for t in range(ep_len - window + 1):
                self.pairs.append((indices[t], indices[t + T_obs]))

        self.features = features  # (N, 524)
        self.actions = actions    # (N, 8)

        # Action normalization: per-dim min/max → [-1, 1]
        if action_min is None:
            action_min = actions.min(axis=0)
        if action_max is None:
            action_max = actions.max(axis=0)
        self.action_min = action_min
        self.action_max = action_max
        # Avoid division by zero
        denom = np.maximum(action_max - action_min, 1e-6)
        self.actions_norm = 2.0 * (actions - action_min) / denom - 1.0
        # Clip to [-1, 1]
        self.actions_norm = np.clip(self.actions_norm, -1.0, 1.0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        obs_start, act_start = self.pairs[idx]
        obs_window = self.features[obs_start: obs_start + self.T_obs]      # (T_obs, 524)
        action_chunk = self.actions_norm[act_start: act_start + self.T_pred]  # (T_pred, 8)
        return (
            torch.as_tensor(obs_window, dtype=torch.float32),
            torch.as_tensor(action_chunk, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Cosine LR scheduler with warmup
# ---------------------------------------------------------------------------

class CosineWarmupScheduler:
    """Cosine learning rate schedule with linear warmup."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            factor = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            factor = max(factor, self.min_lr / max(self.base_lrs[0], 1e-8))

        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base_lr * factor

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Top-K checkpoint manager
# ---------------------------------------------------------------------------

class TopKCheckpoints:
    """Keeps top-K checkpoints by place_rate. Saves to save_dir/."""

    def __init__(self, save_dir, k=3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.k = k
        self.checkpoints = []  # list of (place_rate, filename)

    def update(self, model, ema, epoch, place_rate, action_min, action_max,
               extra_state=None):
        """Consider adding a new checkpoint. Removes worst if over K."""
        filename = self.save_dir / f"ckpt_epoch{epoch}_place{place_rate*100:.0f}.pt"
        state = {
            'epoch': epoch,
            'place_rate': place_rate,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict() if hasattr(ema, 'state_dict') else None,
            'action_min': action_min,
            'action_max': action_max,
            'extra': extra_state or {},
        }
        torch.save(state, filename)
        self.checkpoints.append((place_rate, str(filename)))

        # Sort by place_rate descending, keep top K
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)
        while len(self.checkpoints) > self.k:
            _, removed_file = self.checkpoints.pop()
            removed_path = Path(removed_file)
            if removed_path.exists():
                removed_path.unlink()

    def best(self):
        """Return path to best checkpoint, or None if empty."""
        if not self.checkpoints:
            return None
        return self.checkpoints[0][1]


# ---------------------------------------------------------------------------
# Inline evaluation
# ---------------------------------------------------------------------------

def inline_eval(model, ema, ddim_scheduler, device, n_episodes=5,
                release_threshold=0.05):
    """Run N-episode hierarchical eval with the diffusion policy.

    Returns dict with:
      place_rate, mean_dist, n_placed, n_grabbed,
      action_jerk, chunk_transition_variance, action_range_violation
    """
    import gymnasium
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs, FlattenObs
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
    from hierarchical_policy import HierarchicalPickPlacePolicy
    from backbone_probe import load_v59_feature_extractor
    from diffusion_policy_model import DiffusionPlacePolicy

    TARGET_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]

    # Load grasp model
    grasp_factory = lambda: FlattenObs(gymnasium.make(
        "PandaVLA-v0", reward_type="dense", gravity_comp=True,
        target_pos_range=TARGET_RANGE, domain_randomize=False))
    grasp_vec = DummyVecEnv([grasp_factory])
    grasp_vec = VecNormalize.load(GRASP_VECNORM_PATH, grasp_vec)
    grasp_vec.norm_reward = False
    grasp_vec.training = False
    grasp_model = PPO.load(GRASP_MODEL_PATH, env=grasp_vec, device="cuda")

    # Load V59 features extractor (for online feature extraction during eval)
    feat_extractor, _ = load_v59_feature_extractor(V59_MODEL_PATH, device=device)

    # Load V59 vecnorm (for normalizing obs)
    place_factory = lambda: VisionObs(
        gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True,
                       target_pos_range=TARGET_RANGE, domain_randomize=False),
        image_size=84)
    place_vec = DummyVecEnv([place_factory])
    place_vec = VecNormalize.load(V59_VECNORM_PATH, place_vec)
    place_vec.norm_reward = False
    place_vec.training = False
    place_vec = VecTransposeImage(place_vec)

    # Get action normalization stats from the dataset (saved in checkpoint)
    # For inline eval, we need to pass action_min/max to the policy
    # This is handled by the caller via closure
    # Here we use the EMA model for evaluation

    # Build diffusion policy wrapper — use EMA shadow params if available
    # diffusers EMAModel stores shadow_params (list of tensors), not averaged_model
    eval_model = model
    if hasattr(ema, 'shadow_params') and ema.shadow_params:
        # Copy shadow params into a temporary model for eval
        eval_model = model
        orig_params = [p.data.clone() for p in model.parameters()]
        for p, shadow in zip(model.parameters(), ema.shadow_params):
            p.data.copy_(shadow)
    else:
        orig_params = None
    action_min = getattr(inline_eval, '_action_min', np.zeros(8))
    action_max = getattr(inline_eval, '_action_max', np.ones(8))

    diffusion_policy = DiffusionPlacePolicy(
        unet=eval_model,
        ddim_scheduler=ddim_scheduler,
        feature_extractor=feat_extractor,
        vecnorm=place_vec,
        action_min=action_min,
        action_max=action_max,
        T_obs=2, T_pred=16, T_exec=8,
        num_inference_steps=20,
        device=device,
    )

    policy = HierarchicalPickPlacePolicy(grasp_model, diffusion_policy)

    # Eval environment
    raw_env = DummyVecEnv([grasp_factory])
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = release_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    n_placed = 0
    n_grabbed = 0
    final_dists = []
    all_actions = []
    chunk_transitions = []

    np.random.seed(42)
    try:
        raw_env.seed(42)
    except Exception:
        pass

    for ep in range(n_episodes):
        inner.place_mode = False
        inner._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        prev_info = None
        max_lift = 0.0
        block_target_dist = float("inf")
        ep_actions = []
        prev_chunk_end = None

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
                obs = place_vec.normalize_obs(obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
                # Track buffer state before predict to detect chunk transitions
                buf_was_empty = len(diffusion_policy._buffer) == 0
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)
                ep_actions.append(action[0].copy())
                # Check for chunk transition (buffer was empty, now has items → new chunk started)
                if buf_was_empty and len(diffusion_policy._buffer) > 0 and len(ep_actions) > 1:
                    # This action is the first of a new chunk; previous was last of old chunk
                    chunk_transitions.append(
                        np.linalg.norm(ep_actions[-1] - ep_actions[-2]))
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
        if len(ep_actions) > 2:
            all_actions.extend(ep_actions)

    place_rate = n_placed / max(1, n_grabbed)
    mean_dist = float(np.mean(final_dists)) if final_dists else 0.0

    # Compute smoothness metrics
    action_jerk = 0.0
    chunk_transition_variance = 0.0
    action_range_violation = 0.0

    if len(all_actions) >= 3:
        actions_arr = np.array(all_actions)
        # Jerk: 3rd difference
        jerk = actions_arr[2:] - 2 * actions_arr[1:-1] + actions_arr[:-2]
        action_jerk = float(np.mean(np.sum(jerk ** 2, axis=-1)))

    if len(chunk_transitions) >= 2:
        chunk_transition_variance = float(np.var(chunk_transitions))

    if len(all_actions) > 0:
        actions_arr = np.array(all_actions)
        violations = np.mean(np.any(np.abs(actions_arr) > 1.1, axis=-1))
        action_range_violation = float(violations)

    # Restore original model params (we temporarily swapped in EMA shadow params)
    if orig_params is not None:
        for p, orig in zip(model.parameters(), orig_params):
            p.data.copy_(orig)

    return {
        'place_rate': place_rate,
        'mean_dist': mean_dist,
        'n_placed': n_placed,
        'n_grabbed': n_grabbed,
        'action_jerk': action_jerk,
        'chunk_transition_variance': chunk_transition_variance,
        'action_range_violation': action_range_violation,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=FEATURES_PATH)
    parser.add_argument("--save_path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=15)
    parser.add_argument("--T_obs", type=int, default=2)
    parser.add_argument("--T_pred", type=int, default=16)
    parser.add_argument("--T_exec", type=int, default=8)
    parser.add_argument("--num_train_timesteps", type=int, default=100)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--down_dims", type=int, nargs="+", default=[512, 1024, 2048],
                        help="Channel dims (community standard [512,1024,2048], fits with 8-bit AdamW)")
    parser.add_argument("--diffusion_step_embed_dim", type=int, default=128)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finetune_backbone", action="store_true",
                        help="Enable backbone fine-tuning (full config only)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 70)
    print("Diffusion Policy Training — Break the 56% Unimodal Ceiling")
    print("=" * 70)
    print(f"Config: {vars(args)}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device

    # --- Load cached features ---
    print(f"\nLoading cached features from {args.features}...")
    data = np.load(args.features)
    features = data["features"]      # (N, 524)
    actions = data["actions"]        # (N, 8)
    episode_ids = data["episode_ids"]  # (N,)
    print(f"  {len(actions)} transitions, {len(np.unique(episode_ids))} episodes")
    print(f"  features: {features.shape}, actions: {actions.shape}")
    print(f"  actions range: [{actions.min():.3f}, {actions.max():.3f}]")

    # --- Build dataset ---
    print("\nBuilding action chunk dataset...")
    full_dataset = ActionChunkDataset(
        features, actions, episode_ids,
        T_obs=args.T_obs, T_pred=args.T_pred,
    )
    print(f"  {len(full_dataset)} (obs_window, action_chunk) pairs")

    # Episode-aware 90/10 split
    from backbone_probe import split_by_episode
    train_idx, test_idx = split_by_episode(episode_ids, train_frac=0.9)
    # Filter pairs to train/test by checking if their episode is in train/test
    train_eps = set(episode_ids[train_idx])
    test_eps = set(episode_ids[test_idx])

    # Rebuild train/test datasets with shared action normalization stats
    action_min = full_dataset.action_min
    action_max = full_dataset.action_max
    print(f"  Action normalization: min={action_min}, max={action_max}")

    # Create train/test by filtering the full dataset's pairs
    train_pairs = [(s, a) for (s, a) in full_dataset.pairs
                   if int(episode_ids[s]) in train_eps]
    test_pairs = [(s, a) for (s, a) in full_dataset.pairs
                  if int(episode_ids[s]) in test_eps]

    train_dataset = ActionChunkDataset(
        features, actions, episode_ids,
        T_obs=args.T_obs, T_pred=args.T_pred,
        action_min=action_min, action_max=action_max,
    )
    train_dataset.pairs = train_pairs
    test_dataset = ActionChunkDataset(
        features, actions, episode_ids,
        T_obs=args.T_obs, T_pred=args.T_pred,
        action_min=action_min, action_max=action_max,
    )
    test_dataset.pairs = test_pairs

    print(f"  Train: {len(train_dataset)} pairs, Test: {len(test_dataset)} pairs")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True,
                              pin_memory=True)

    # --- Build model ---
    print(f"\nBuilding ConditionalUNet1D (down_dims={args.down_dims})...")
    from diffusion_policy_model import ConditionalUNet1D
    model = ConditionalUNet1D(
        input_dim=8,
        down_dims=args.down_dims,
        T_pred=args.T_pred,
        cond_dim=524,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # --- Diffusion schedulers ---
    from diffusers import DDPMScheduler, DDIMScheduler
    from diffusers.training_utils import EMAModel

    ddpm = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
    )
    ddim = DDIMScheduler.from_config(ddpm.config)
    ddim.set_timesteps(args.num_inference_steps)

    # EMA
    ema = EMAModel(parameters=model.parameters(), decay=args.ema_decay,
                   update_after_step=0)

    # --- Optimizer + scheduler ---
    # 8-bit AdamW: halves optimizer state memory, enables [512,1024,2048] on 6GB GPU
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=args.lr,
            betas=(0.95, 0.999), weight_decay=1e-6,
        )
        print("  Optimizer: bitsandbytes AdamW8bit")
    except Exception:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr,
            betas=(0.95, 0.999), weight_decay=1e-6,
        )
        print("  Optimizer: torch AdamW (bitsandbytes unavailable)")
    total_steps = len(train_loader) * args.n_epochs
    lr_scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps,
    )

    # --- Top-K checkpoint manager ---
    ckpt_manager = TopKCheckpoints(args.save_path, k=3)

    # --- Training state ---
    best_place_rate = 0.0
    best_epoch = 0
    consecutive_failures = 0
    history = []
    instability_flags = []

    # Store action stats for inline eval
    inline_eval._action_min = action_min
    inline_eval._action_max = action_max

    print(f"\nStarting training: {args.n_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"  Eval every {args.eval_every} epochs, {args.eval_episodes} episodes per eval")
    print(f"  Breakthrough: place_rate >= 60% → save breakthrough.ckpt, stop")
    print(f"  Safety: place_rate < 5% for 3 consecutive evals → stop")
    print("=" * 70)

    train_start = time.time()

    for epoch in range(args.n_epochs):
        model.train()
        epoch_losses = []
        epoch_grad_norms = []
        epoch_start = time.time()

        for batch_idx, (obs_window, action_chunk) in enumerate(train_loader):
            obs_window = obs_window.to(device, non_blocking=True)    # (B, T_obs, 524)
            action_chunk = action_chunk.to(device, non_blocking=True)  # (B, T_pred, 8)

            B = action_chunk.shape[0]
            optimizer.zero_grad()

            # No AMP — float16 overflows in FiLM MLP with feature values up to 7.18.
            # Model is only 66M params (~1.5GB), fits in 6GB VRAM without mixed precision.
            t = torch.randint(0, args.num_train_timesteps, (B,), device=device)
            noise = torch.randn_like(action_chunk)
            noisy_actions = ddpm.add_noise(action_chunk, noise, t)
            noise_pred = model(noisy_actions, t.float(), obs_window)
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            ema.step(model.parameters())
            lr_scheduler.step()

            epoch_losses.append(loss.item())
            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                           if p.grad is not None) ** 0.5
            epoch_grad_norms.append(grad_norm)

        avg_loss = np.mean(epoch_losses)
        avg_grad_norm = np.mean(epoch_grad_norms)
        epoch_time = time.time() - epoch_start

        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.n_epochs}: loss={avg_loss:.6f}, "
              f"grad_norm={avg_grad_norm:.4f}, lr={lr:.2e}, "
              f"time={epoch_time:.1f}s")

        # --- Evaluation ---
        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == 0)
        eval_result = None
        if do_eval:
            print(f"  Running inline eval ({args.eval_episodes} episodes)...")
            eval_start = time.time()
            try:
                eval_result = inline_eval(
                    model, ema, ddim, device,
                    n_episodes=args.eval_episodes,
                )
                eval_time = time.time() - eval_start
                place_rate = eval_result['place_rate']
                jerk = eval_result['action_jerk']
                ctv = eval_result['chunk_transition_variance']

                print(f"  Eval: place_rate={100*place_rate:.1f}% "
                      f"({eval_result['n_placed']}/{eval_result['n_grabbed']}), "
                      f"mean_dist={eval_result['mean_dist']*100:.1f}cm, "
                      f"jerk={jerk:.3f}, ctv={ctv:.3f}, "
                      f"time={eval_time:.0f}s")

                # Instability flag
                unstable = (jerk > 0.5) or (ctv > 0.3)
                if unstable:
                    msg = (f"WARNING: Training instability detected at epoch {epoch+1}: "
                           f"jerk={jerk:.3f} (>0.5={jerk>0.5}), ctv={ctv:.3f} (>0.3={ctv>0.3})")
                    print(f"  {msg}")
                    instability_flags.append({
                        'epoch': epoch + 1, 'jerk': jerk, 'ctv': ctv,
                        'place_rate': place_rate,
                    })

                # Update best
                if place_rate > best_place_rate:
                    best_place_rate = place_rate
                    best_epoch = epoch + 1
                    # Save top-K checkpoint
                    ckpt_manager.update(
                        model, ema, epoch + 1, place_rate,
                        action_min, action_max,
                        extra_state={'loss': avg_loss, 'jerk': jerk, 'ctv': ctv},
                    )

                # Breakthrough
                if place_rate >= 0.60:
                    print(f"\n*** BREAKTHROUGH! place_rate={100*place_rate:.1f}% >= 60% ***")
                    breakthrough_path = Path(args.save_path) / "breakthrough.ckpt"
                    torch.save({
                        'epoch': epoch + 1,
                        'place_rate': place_rate,
                        'model_state_dict': model.state_dict(),
                        'ema_state_dict': ema.state_dict() if hasattr(ema, 'state_dict') else None,
                        'action_min': action_min,
                        'action_max': action_max,
                    }, breakthrough_path)
                    print(f"  Saved to {breakthrough_path}")
                    history.append({
                        'epoch': epoch + 1, 'loss': avg_loss,
                        'place_rate': place_rate, **eval_result,
                    })
                    break

                # Safety check
                if place_rate < 0.05:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print(f"\n*** SAFETY STOP: place_rate < 5% for 3 consecutive evals ***")
                        break
                else:
                    consecutive_failures = 0

            except Exception as e:
                print(f"  Eval failed: {e}")
                import traceback
                traceback.print_exc()
                eval_result = None

        history.append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'grad_norm': float(avg_grad_norm),
            'lr': float(lr),
            'time': float(epoch_time),
            'eval': eval_result,
        })

    total_time = time.time() - train_start

    # --- Save final results ---
    results = {
        'config': vars(args),
        'n_params': n_params,
        'best_place_rate': best_place_rate,
        'best_epoch': best_epoch,
        'total_epochs_run': len(history),
        'total_time_sec': total_time,
        'instability_flags': instability_flags,
        'action_min': action_min.tolist(),
        'action_max': action_max.tolist(),
        'top_k_checkpoints': [
            {'place_rate': pr, 'path': p} for pr, p in ckpt_manager.checkpoints
        ],
        'history': history,
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results_path = Path(args.save_path) / "training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best place_rate: {100*best_place_rate:.1f}% (epoch {best_epoch})")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Instability flags: {len(instability_flags)}")
    print(f"  Top-K checkpoints: {len(ckpt_manager.checkpoints)}")
    print(f"  Results saved to: {results_path}")

    if best_place_rate >= 0.60:
        print("\n  STATUS: BREAKTHROUGH — diffusion policy exceeded 60% place rate!")
    elif best_place_rate >= 0.45:
        print("\n  STATUS: PROMISING — prototype shows trend above 45%, "
              "consider full config training")
    else:
        print(f"\n  STATUS: INSUFFICIENT — best {100*best_place_rate:.1f}% < 45%, "
              "archive as family_09 failure")


if __name__ == "__main__":
    main()
