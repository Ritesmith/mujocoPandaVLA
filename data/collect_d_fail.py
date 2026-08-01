#!/usr/bin/env python3
"""Collect V59 failure trajectories (D_fail) for failure-mode clustering.

Runs V59 in the hierarchical pick-and-place env for N episodes, collecting
per-step transition data with rich annotations (critic value, log-prob,
gripper width, EE position, contact flag, task stage). Only transitions from
FAILURE episodes (final_dist >= PLACE_THRESHOLD) are saved to D_fail.npz.

Fields saved (per transition):
  state_img      (84,84,3) uint8   -- raw HWC image
  state_vec      (12,)     float32 -- joint_pos(7) + gripper(1) + dist(1) + target(3)
  action         (8,)      float32 -- V59 deterministic action
  reward         float32           -- env reward
  critic_V59     float32           -- V59 Critic V(s)
  logpi_V59      float32           -- log pi_V59(a|s) under DiagGaussian
  task_stage     int               -- 0=grasp, 1=place
  gripper_width  float32           -- gripper opening (m)
  EE_pos         (3,)     float32  -- end-effector position
  contact_flag   int               -- 1 if hand-block distance < 0.05m
  ep_id          int               -- episode index
  t_within_ep    int               -- step index within episode
  success_flag   int               -- 1 if episode succeeded (final_dist < 0.05)
  final_dist     float32           -- final block-target distance (m)

Usage:
    python collect_d_fail.py --n_episodes 100
    python collect_d_fail.py --n_episodes 50 --device auto
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

# Reuse infrastructure from voronoi_partition (envs, model loading, helpers).
from diagnostics.voronoi_partition import (
    GRASP_MODEL_PATH, GRASP_VECNORM_PATH,
    PLACE_MODEL_PATH, PLACE_VECNORM_PATH,
    TARGET_POS_RANGE, TABLE_Z, MAX_STEPS, SEED,
    PLACE_THRESHOLD, LIFT_THRESHOLD,
    _build_collect_envs, freeze_backbone,
    extract_v59_latent, v59_value,
)

D_FAIL_PATH = WORKSPACE / "data" / "D_fail.npz"
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def compute_logpi(mean: torch.Tensor, log_std: torch.Tensor,
                  action: torch.Tensor) -> torch.Tensor:
    """Log density of a diagonal Gaussian: log N(a | mean, exp(log_std)^2).

    All tensors are (N, D). Returns (N,) tensor.
    """
    std = torch.exp(log_std)
    diff = action - mean
    log_pi = -0.5 * ((diff / std) ** 2).sum(dim=-1) \
             - log_std.sum(dim=-1) \
             - 0.5 * mean.shape[-1] * float(np.log(2.0 * np.pi))
    return log_pi


def collect_d_fail(
    n_episodes: int = 100,
    max_steps: int = MAX_STEPS,
    place_model_path: str = PLACE_MODEL_PATH,
    place_vecnorm_path: str = PLACE_VECNORM_PATH,
    grasp_model_path: str = GRASP_MODEL_PATH,
    grasp_vecnorm_path: str = GRASP_VECNORM_PATH,
    release_threshold: float = PLACE_THRESHOLD,
    seed: int = SEED,
    device: str = "auto",
    output_path: Optional[str] = None,
) -> dict:
    """Run V59 for n_episodes, collect failure trajectories -> D_fail.npz.

    Returns a summary dict with collection statistics.
    """
    print("=" * 60)
    print("D_fail Collection (V59 Failure Trajectories)")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Place model: {place_model_path}")
    print(f"Image augmentation: DISABLED (project_memory hard constraint)")
    print(f"BN running stats: FROZEN (features_extractor.eval())")
    print(f"V59 weights: FROZEN -- inference only")
    print()

    # Resolve device.
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Build envs.
    args_ns = argparse.Namespace(
        place_model=place_model_path,
        place_vecnorm=place_vecnorm_path,
        grasp_model=grasp_model_path,
        grasp_vecnorm=grasp_vecnorm_path,
        release_threshold=release_threshold,
    )
    envs = _build_collect_envs(args_ns, device=device)
    policy = envs["policy"]
    place_model = envs["place_model"]
    raw_env = envs["raw_env"]
    _inner_env = envs["inner_env"]
    place_vec_env = envs["place_vec_env"]
    place_vision_wrapper = envs["place_vision_wrapper"]

    # Freeze V59.
    freeze_backbone(place_model)
    place_model.policy.features_extractor.eval()
    for m in place_model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    v59_policy = place_model.policy
    v59_log_std = v59_policy.log_std.detach().clamp(LOG_STD_MIN, LOG_STD_MAX)

    np.random.seed(seed)
    try:
        raw_env.seed(seed)
    except Exception:
        pass

    # Storage for ALL transitions (filter to failures at the end).
    all_img: list[np.ndarray] = []
    all_vec: list[np.ndarray] = []
    all_action: list[np.ndarray] = []
    all_reward: list[float] = []
    all_critic: list[float] = []
    all_logpi: list[float] = []
    all_stage: list[int] = []
    all_gripper: list[float] = []
    all_ee: list[np.ndarray] = []
    all_contact: list[int] = []
    all_ep_id: list[int] = []
    all_t: list[int] = []
    all_success: list[int] = []
    all_final_dist: list[float] = []

    t_start = time.time()
    n_placed = 0
    n_entered_place = 0

    for ep in range(n_episodes):
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        prev_info = None
        first_place_step = None
        ep_transitions = 0
        ep_final_dist = 1.0  # default: far
        ep_placed = False

        for t in range(max_steps):
            phase = policy._detect_phase(prev_info)

            # Enter place phase: configure inner env for place dynamics.
            if phase == "place" and first_place_step is None:
                first_place_step = t
                _inner_env.place_mode = True
                _inner_env._place_gravcomp_active = True
                _inner_env.snap_block_to_hand()
                _inner_env._arm_target = _inner_env.data.qpos[
                    _inner_env._arm_qpos_adrs
                ].copy()
                _inner_env._gripper_target = float(
                    _inner_env.data.qpos[_inner_env._finger_qpos_adrs].mean()
                )
                _inner_env.reward_type = "place_only"
                _inner_env._place_approach_bonus_given = False
                _inner_env._place_proximity_15_given = False
                _inner_env._place_proximity_10_given = False
                _inner_env._place_success = False
                _inner_env._prev_block_target_dist = None
                _inner_env._prev_block_height = None
                _inner_env._use_gripper_target_check = True

                flatten_wrapper = raw_env.envs[0]
                inner_obs = _inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

            # Construct obs based on phase.
            if phase == "place":
                vision_obs = place_vision_wrapper.observation(_inner_env._get_obs())
                vision_obs_batched = {
                    "image": vision_obs["image"][np.newaxis, ...],
                    "state": vision_obs["state"][np.newaxis, ...],
                }
                obs = place_vec_env.normalize_obs(vision_obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
            else:
                raw_obs_for_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_for_grasp[0, 8:11]
                default_target = np.array([0.5, 0.3, 0.2])
                raw_obs_for_grasp[0, 15] = np.linalg.norm(block_pos - default_target)
                obs = envs["grasp_vec_env"].normalize_obs(raw_obs_for_grasp)

            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            # Record place-phase transitions (where V59 place policy is active).
            if phase == "place":
                with torch.no_grad():
                    img_t = torch.as_tensor(
                        obs["image"], dtype=torch.float32, device=device)
                    st_t = torch.as_tensor(
                        obs["state"], dtype=torch.float32, device=device)
                    obs_t = {"image": img_t, "state": st_t}
                    latent = extract_v59_latent(v59_policy, obs_t)
                    a_mean = v59_policy.action_net(latent)
                    value = float(v59_value(v59_policy, obs_t).cpu().item())
                    log_pi = float(compute_logpi(
                        a_mean, v59_log_std.expand_as(a_mean),
                        torch.as_tensor(action, dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    ).cpu().item())

                info_dict = prev_info or {}
                gripper_w = float(info_dict.get("gripper_opening", 0.04))
                ee_pos = np.array(info_dict.get("hand_position", [0, 0, 0]),
                                  dtype=np.float32)
                hand_block_dist = float(info_dict.get("hand_block_distance", 1.0))
                contact = 1 if hand_block_dist < 0.05 else 0

                all_img.append(vision_obs["image"].copy())
                all_vec.append(vision_obs["state"].copy())
                all_action.append(action[0].copy())
                all_critic.append(value)
                all_logpi.append(log_pi)
                all_stage.append(1)  # place
                all_gripper.append(gripper_w)
                all_ee.append(ee_pos)
                all_contact.append(contact)
                all_ep_id.append(ep)
                all_t.append(t)
                ep_transitions += 1

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(
                prev_info.get("block_target_distance", 1.0))
            ep_final_dist = block_target_dist

            # Record per-step reward for place-phase transitions.
            if phase == "place":
                all_reward.append(float(reward[0]))

            if done[0]:
                break

        if ep_final_dist < PLACE_THRESHOLD:
            ep_placed = True
            n_placed += 1
        if first_place_step is not None:
            n_entered_place += 1

        # Tag all transitions from this episode with success/failure.
        for _ in range(ep_transitions):
            all_success.append(1 if ep_placed else 0)
            all_final_dist.append(ep_final_dist)

        elapsed = time.time() - t_start
        status = "PLACED" if ep_placed else "FAIL"
        print(f"Ep {ep:3d}: {phase:5s} dist={ep_final_dist*100:5.1f}cm "
              f"trans={ep_transitions:4d} | {status} "
              f"({n_placed}/{ep+1}={n_placed/(ep+1)*100:.0f}%) "
              f"[{elapsed:.0f}s]")

    elapsed_total = time.time() - t_start

    # Convert to arrays.
    all_img_arr = np.array(all_img, dtype=np.uint8) if all_img else np.zeros((0, 84, 84, 3), dtype=np.uint8)
    all_vec_arr = np.array(all_vec, dtype=np.float32) if all_vec else np.zeros((0, 12), dtype=np.float32)
    all_action_arr = np.array(all_action, dtype=np.float32) if all_action else np.zeros((0, 8), dtype=np.float32)
    all_reward_arr = np.array(all_reward, dtype=np.float32)
    all_critic_arr = np.array(all_critic, dtype=np.float32)
    all_logpi_arr = np.array(all_logpi, dtype=np.float32)
    all_stage_arr = np.array(all_stage, dtype=np.int32)
    all_gripper_arr = np.array(all_gripper, dtype=np.float32)
    all_ee_arr = np.array(all_ee, dtype=np.float32) if all_ee else np.zeros((0, 3), dtype=np.float32)
    all_contact_arr = np.array(all_contact, dtype=np.int32)
    all_ep_id_arr = np.array(all_ep_id, dtype=np.int32)
    all_t_arr = np.array(all_t, dtype=np.int32)
    all_success_arr = np.array(all_success, dtype=np.int32)
    all_final_dist_arr = np.array(all_final_dist, dtype=np.float32)

    # Filter to failure episodes only (D_fail).
    fail_mask = all_success_arr == 0
    n_fail = int(fail_mask.sum())
    n_total = len(fail_mask)

    print()
    print("=" * 60)
    print("D_fail Collection Complete")
    print("=" * 60)
    print(f"Episodes run:        {n_episodes}")
    print(f"Entered place phase: {n_entered_place}/{n_episodes}")
    print(f"Placed (dist<5cm):   {n_placed}/{n_episodes} "
          f"({n_placed/n_episodes*100:.1f}%)")
    print(f"Total transitions:   {n_total}")
    print(f"Failure transitions: {n_fail} (kept)")
    print(f"Success transitions: {n_total - n_fail} (discarded)")
    print(f"Elapsed: {elapsed_total:.0f}s")

    if n_fail == 0:
        print("WARNING: No failure transitions collected!")
        return {"n_fail": 0}

    # Save D_fail (failure transitions only).
    save_path = output_path or str(D_FAIL_PATH)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_path,
        state_img=all_img_arr[fail_mask],
        state_vec=all_vec_arr[fail_mask],
        action=all_action_arr[fail_mask],
        reward=all_reward_arr[fail_mask],
        critic_V59=all_critic_arr[fail_mask],
        logpi_V59=all_logpi_arr[fail_mask],
        task_stage=all_stage_arr[fail_mask],
        gripper_width=all_gripper_arr[fail_mask],
        EE_pos=all_ee_arr[fail_mask],
        contact_flag=all_contact_arr[fail_mask],
        ep_id=all_ep_id_arr[fail_mask],
        t_within_ep=all_t_arr[fail_mask],
        success_flag=all_success_arr[fail_mask],
        final_dist=all_final_dist_arr[fail_mask],
    )
    print(f"Saved to {save_path} "
          f"({Path(save_path).stat().st_size / 1e6:.1f} MB)")

    # Print summary stats for GMM clustering.
    print("\n--- D_fail Summary Stats (for GMM clustering) ---")
    print(f"  gripper_width:   min={all_gripper_arr[fail_mask].min():.3f} "
          f"max={all_gripper_arr[fail_mask].max():.3f} "
          f"mean={all_gripper_arr[fail_mask].mean():.3f}")
    ee_dists = np.linalg.norm(all_ee_arr[fail_mask] -
                              np.array([0.5, 0.3, 0.22]), axis=1)
    print(f"  EE_dist_to_tgt:  min={ee_dists.min():.3f} "
          f"max={ee_dists.max():.3f} mean={ee_dists.mean():.3f}")
    print(f"  action_L2:       mean={np.linalg.norm(all_action_arr[fail_mask], axis=1).mean():.3f}")
    print(f"  critic_V59:      min={all_critic_arr[fail_mask].min():.3f} "
          f"max={all_critic_arr[fail_mask].max():.3f} "
          f"mean={all_critic_arr[fail_mask].mean():.3f}")
    print(f"  logpi_V59:       min={all_logpi_arr[fail_mask].min():.3f} "
          f"max={all_logpi_arr[fail_mask].max():.3f} "
          f"mean={all_logpi_arr[fail_mask].mean():.3f}")
    print(f"  contact_rate:    {all_contact_arr[fail_mask].mean()*100:.1f}%")
    print(f"  n_episodes:      {len(np.unique(all_ep_id_arr[fail_mask]))}")

    return {
        "n_episodes": n_episodes,
        "n_placed": n_placed,
        "n_fail_transitions": n_fail,
        "n_total_transitions": n_total,
        "elapsed_s": elapsed_total,
        "save_path": save_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect V59 failure trajectories (D_fail) for clustering."
    )
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of episodes to run (default: 100)")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH)
    parser.add_argument("--place_vecnorm", type=str, default=PLACE_VECNORM_PATH)
    parser.add_argument("--grasp_model", type=str, default=GRASP_MODEL_PATH)
    parser.add_argument("--grasp_vecnorm", type=str, default=GRASP_VECNORM_PATH)
    parser.add_argument("--release_threshold", type=float,
                        default=PLACE_THRESHOLD)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="auto",
                        help="torch device (default: auto)")
    parser.add_argument("--output", type=str, default=str(D_FAIL_PATH))
    args = parser.parse_args()

    collect_d_fail(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        place_model_path=args.place_model,
        place_vecnorm_path=args.place_vecnorm,
        grasp_model_path=args.grasp_model,
        grasp_vecnorm_path=args.grasp_vecnorm,
        release_threshold=args.release_threshold,
        seed=args.seed,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
