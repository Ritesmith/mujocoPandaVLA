#!/usr/bin/env python3
"""Collect expert demonstrations by running the IK oracle as the place policy.

Unlike DAgger (which runs V59 and labels with oracle), this script runs the
ORACLE as the executed policy during the place phase. This produces "pure
expert demonstrations" where the state distribution matches the oracle's
policy, avoiding the distribution shift that doomed DAgger V1/V2A/V2B.

Pipeline:
  1. Grasp phase: use V59's grasp model (100% grasp rate)
  2. Place phase: use DAggerOracle (Jacobian IK) as the EXECUTED policy
  3. Record (image, state, oracle_action) for each place-phase step
  4. Measure oracle's place rate (should be high if IK controller works)
  5. Save to D_expert.npz

The oracle provides genuinely EXTERNAL information (analytical IK solution,
not from V59's policy distribution). This is the "human demo" equivalent
without requiring a teleop interface.

Usage:
    python collect_expert_demos.py --n_episodes 200
    python collect_expert_demos.py --n_episodes 50 --oracle_gain 3.0
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs, VisionObs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hierarchical_policy import HierarchicalPickPlacePolicy
from dagger_oracle import DAggerOracle, DAggerOracleV2

GRASP_MODEL = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"
OUTPUT = "/home/w/vla_workspace/data/D_expert.npz"

LIFT_THRESHOLD = 0.03
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42
TARGET_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]


def make_env(vision_mode=False, target_pos_range=None):
    kwargs = dict(reward_type="dense", gravity_comp=True)
    if target_pos_range:
        kwargs["target_pos_range"] = target_pos_range
    kwargs["domain_randomize"] = False
    env = gymnasium.make("PandaVLA-v0", **kwargs)
    if vision_mode:
        env = VisionObs(env, image_size=84)
    else:
        env = FlattenObs(env)
    return env


def load_model(path, vecnorm_path, vision_mode=False, target_pos_range=None):
    factory = lambda: make_env(vision_mode=vision_mode,
                                target_pos_range=target_pos_range)
    vec_env = DummyVecEnv([factory])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False
    else:
        norm_keys = ["state"] if vision_mode else None
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, norm_obs_keys=norm_keys)
        vec_env.training = False
    model = PPO.load(path, env=vec_env, device="auto")
    return model, vec_env


def main():
    parser = argparse.ArgumentParser(description="Collect expert demos (oracle as policy)")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default=OUTPUT)
    parser.add_argument("--release_threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--oracle_gain", type=float, default=2.0,
                        help="Proportional gain for oracle velocity control")
    parser.add_argument("--oracle_max_speed", type=float, default=0.5,
                        help="Max Cartesian speed (m/s) for oracle")
    parser.add_argument("--oracle_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="v1=binary gripper, v2=V59-style gripper (-0.07)")
    args = parser.parse_args()

    print("=" * 60)
    print("Expert Demo Collection (Oracle as Place Policy)")
    print("=" * 60)
    print(f"Episodes: {args.n_episodes}")
    print(f"Grasp: V59 grasp model (deterministic)")
    print(f"Place: DAggerOracle v{args.oracle_version} (Jacobian IK)")
    print(f"Oracle gain: {args.oracle_gain}, max_speed: {args.oracle_max_speed}")
    print(f"Output: {args.output}")
    print()

    # Load grasp model
    print("Loading grasp model...")
    grasp_model, grasp_vec_env = load_model(
        GRASP_MODEL, GRASP_VECNORM, vision_mode=False,
        target_pos_range=TARGET_RANGE)

    # We need the place model only for the HierarchicalPickPlacePolicy's
    # phase detection logic (grasp → place switch). But during the place
    # phase, we OVERRIDE V59's action with the oracle's action.
    print("Loading place model (for phase detection only)...")
    place_model, place_vec_env = load_model(
        PLACE_MODEL, PLACE_VECNORM, vision_mode=True,
        target_pos_range=TARGET_RANGE)

    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    # Raw eval env
    raw_env = DummyVecEnv([lambda: make_env(vision_mode=False,
                                              target_pos_range=TARGET_RANGE)])
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = args.release_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    # DAgger oracle
    if args.oracle_version == "v2":
        oracle = DAggerOracleV2(
            inner, gain=args.oracle_gain, max_speed=args.oracle_max_speed,
            release_threshold=args.release_threshold)
    else:
        oracle = DAggerOracle(
            inner, gain=args.oracle_gain, max_speed=args.oracle_max_speed,
            release_threshold=args.release_threshold)
    print(f"Oracle initialized (v{args.oracle_version})")
    print()

    np.random.seed(args.seed)
    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    all_images = []
    all_states = []
    all_actions = []
    all_episode_ids = []
    all_final_dists = []
    all_success = []

    n_grabbed = 0
    n_placed = 0
    n_entered_place = 0
    t0 = time.time()

    for ep in range(args.n_episodes):
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

        for step in range(MAX_STEPS):
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                inner.place_mode = True
                inner._place_gravcomp_active = True
                inner.snap_block_to_hand()
                inner._arm_target = inner.data.qpos[inner._arm_qpos_adrs].copy()
                inner._gripper_target = float(
                    inner.data.qpos[inner._finger_qpos_adrs].mean())
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
                # --- ORACLE AS POLICY (key difference from DAgger) ---
                # Execute oracle action, NOT V59's action
                oracle_action = oracle.get_expert_action()

                # Record observation + oracle action
                vision_obs = place_vision.observation(inner._get_obs())
                ep_images.append(vision_obs["image"].copy())
                ep_states.append(vision_obs["state"].copy())
                ep_actions.append(oracle_action.copy())

                # Execute oracle action (reshape for vec_env: (8,) -> (1, 8))
                action = oracle_action[np.newaxis, :]
            else:
                # Grasp phase: use V59 grasp model
                raw_obs_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_grasp[0, 8:11]
                raw_obs_grasp[0, 15] = np.linalg.norm(block_pos - np.array([0.5, 0.3, 0.2]))
                obs = grasp_vec_env.normalize_obs(raw_obs_grasp)
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(info[0].get("block_target_distance", block_target_dist))
            lift = max(0.0, float(info[0].get("block_height", 0.0)) - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            if done[0]:
                break

        # Episode finished
        if first_place_step is not None and max_lift > LIFT_THRESHOLD:
            n_entered_place += 1
            n_grabbed += 1

            # Check place success
            placed = block_target_dist < args.release_threshold
            if placed:
                n_placed += 1

            success = 1 if placed else 0
            all_success.append(success)

            # Record episode data
            n_steps = len(ep_images)
            all_images.extend(ep_images)
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_episode_ids.extend([ep] * n_steps)
            all_final_dists.extend([block_target_dist] * n_steps)

            status = "PLACE" if placed else "FAIL"
            print(f"Ep {ep:3d}: steps={step:3d}, lift={max_lift*100:.1f}cm, "
                  f"dist={block_target_dist*100:.1f}cm, {status}")
        else:
            all_success.append(0)
            print(f"Ep {ep:3d}: GRASP FAIL (lift={max_lift*100:.1f}cm)")

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{ep+1}/{args.n_episodes}] {elapsed:.0f}s, "
                  f"place_rate={100*n_placed/max(1,n_entered_place):.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Expert Demo Collection Complete ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"Total episodes:    {args.n_episodes}")
    print(f"Grasp success:     {n_grabbed}/{args.n_episodes} ({100*n_grabbed/args.n_episodes:.1f}%)")
    print(f"Entered place:     {n_entered_place}/{args.n_episodes}")
    print(f"Oracle place rate: {n_placed}/{n_entered_place} ({100*n_placed/max(1,n_entered_place):.1f}%)")
    print(f"Total transitions: {len(all_images)}")

    if len(all_images) > 0:
        all_images = np.array(all_images, dtype=np.uint8)
        all_states = np.array(all_states, dtype=np.float32)
        all_actions = np.array(all_actions, dtype=np.float32)
        all_episode_ids = np.array(all_episode_ids, dtype=np.int64)
        all_final_dists = np.array(all_final_dists, dtype=np.float32)

        print(f"\nData shapes:")
        print(f"  images:      {all_images.shape}")
        print(f"  states:      {all_states.shape}")
        print(f"  actions:     {all_actions.shape}")
        print(f"  episode_ids: {all_episode_ids.shape}")
        print(f"  final_dists: {all_final_dists.shape}")

        print(f"\nAction statistics:")
        print(f"  mean: {all_actions.mean(axis=0)}")
        print(f"  std:  {all_actions.std(axis=0)}")
        print(f"  min:  {all_actions.min(axis=0)}")
        print(f"  max:  {all_actions.max(axis=0)}")

        np.savez_compressed(
            args.output,
            images=all_images,
            states=all_states,
            actions=all_actions,
            episode_ids=all_episode_ids,
            final_dists=all_final_dists,
            success_flags=np.array(all_success, dtype=np.int64),
            n_episodes=args.n_episodes,
            n_grabbed=n_grabbed,
            n_placed=n_placed,
            oracle_version=args.oracle_version,
            oracle_gain=args.oracle_gain,
            oracle_max_speed=args.oracle_max_speed,
        )
        print(f"\nSaved to {args.output}")

        # Compare oracle actions vs V59 actions (if we have D_csil for comparison)
        d_csil_path = "/home/w/vla_workspace/data/D_csil.npz"
        if os.path.exists(d_csil_path):
            d_csil = np.load(d_csil_path)
            v59_actions = d_csil['actions']
            print(f"\nOracle vs V59 action comparison:")
            print(f"  Oracle mean: {all_actions.mean(axis=0)}")
            print(f"  V59 mean:    {v59_actions.mean(axis=0)}")
            print(f"  Mean diff:   {(all_actions.mean(axis=0) - v59_actions.mean(axis=0))}")
            print(f"  Diff norm:   {np.linalg.norm(all_actions.mean(axis=0) - v59_actions.mean(axis=0)):.4f}")
    else:
        print("\nWARNING: No place-phase data collected. Check grasp model.")


if __name__ == "__main__":
    main()
