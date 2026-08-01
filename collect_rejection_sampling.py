"""Rejection sampling: collect stochastic trajectories, keep the best.

Runs V59 with STOCHASTIC policy (deterministic=False), collects place-phase
transitions from ALL trajectories, then filters for the top-K with lowest
final block-target distance. These "lucky" trajectories represent noise
directions that happened to improve placement precision.

SFT on this filtered set shifts the policy mean toward those directions.
This is the Llama2 rejection-sampling + SFT method.

Output: D_reject.npz with same format as D_succ.npz (images, states, actions)
but only from the top-K% best trajectories.

Usage:
    python collect_rejection_sampling.py --n_episodes 500 --top_k_pct 20
"""

import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
import time
import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs, VisionObs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hierarchical_policy import HierarchicalPickPlacePolicy

GRASP_MODEL = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"
OUTPUT = "/home/w/vla_workspace/data/D_reject.npz"

LIFT_THRESHOLD = 0.03
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42


def make_env(vision_mode=False, target_pos_range=None, domain_randomize=False):
    kwargs = dict(reward_type="dense", gravity_comp=True)
    if target_pos_range:
        kwargs["target_pos_range"] = target_pos_range
    kwargs["domain_randomize"] = domain_randomize
    env = gymnasium.make("PandaVLA-v0", **kwargs)
    if vision_mode:
        env = VisionObs(env, image_size=84)
    else:
        env = FlattenObs(env)
    return env


def load_model(path, vecnorm_path, vision_mode=False, target_pos_range=None):
    factory = lambda: make_env(
        vision_mode=vision_mode, target_pos_range=target_pos_range)
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
    parser = argparse.ArgumentParser(description="Rejection sampling trajectory collection")
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--top_k_pct", type=int, default=20,
                        help="Keep top K%% of trajectories by final_dist (default 20%%)")
    parser.add_argument("--dist_threshold", type=float, default=0.05,
                        help="Only keep trajectories with dist < this (default 5cm = success)")
    parser.add_argument("--output", type=str, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    target_range = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]

    print("=" * 60)
    print("Rejection Sampling Trajectory Collection")
    print("=" * 60)
    print(f"Place model: {PLACE_MODEL}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Policy: STOCHASTIC (deterministic=False)")
    print(f"Filter: top {args.top_k_pct}% with dist < {args.dist_threshold}m")
    print(f"Output: {args.output}")
    print()

    # Load models
    print("Loading grasp model...")
    grasp_model, grasp_vec_env = load_model(
        GRASP_MODEL, GRASP_VECNORM, vision_mode=False, target_pos_range=target_range)
    print("Loading place model...")
    place_model, place_vec_env = load_model(
        PLACE_MODEL, PLACE_VECNORM, vision_mode=True, target_pos_range=target_range)

    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    # Raw eval env
    raw_env = DummyVecEnv([lambda: make_env(target_pos_range=target_range)])
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = args.dist_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    np.random.seed(args.seed)
    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    # Collect ALL trajectories with their final_dist
    all_trajectories = []  # list of (final_dist, images, states, actions)
    n_placed = 0
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

        ep_images, ep_states, ep_actions = [], [], []

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
                vision_obs = place_vision.observation(inner._get_obs())
                obs_batched = {
                    "image": vision_obs["image"][np.newaxis, ...],
                    "state": vision_obs["state"][np.newaxis, ...],
                }
                obs = place_vec_env.normalize_obs(obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
            else:
                raw_obs_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_grasp[0, 8:11]
                raw_obs_grasp[0, 15] = np.linalg.norm(block_pos - np.array([0.5, 0.3, 0.2]))
                obs = grasp_vec_env.normalize_obs(raw_obs_grasp)

            # STOCHASTIC action (key difference from D_succ collection)
            action, _ = policy.predict(obs, info=prev_info, deterministic=False)

            if phase == "place":
                ep_images.append(vision_obs["image"].copy())
                ep_states.append(vision_obs["state"].copy())
                ep_actions.append(action[0].copy())

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(info[0].get("block_target_distance", block_target_dist))
            lift = max(0.0, float(info[0].get("block_height", 0.0)) - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            if done[0]:
                break

        # Only keep trajectories that entered place phase and placed successfully
        if first_place_step is not None and max_lift > LIFT_THRESHOLD:
            if block_target_dist < args.dist_threshold:
                all_trajectories.append((
                    block_target_dist,
                    np.array(ep_images, dtype=np.uint8),
                    np.array(ep_states, dtype=np.float32),
                    np.array(ep_actions, dtype=np.float32),
                ))
                n_placed += 1

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Ep {ep+1}/{args.n_episodes}: {n_placed} placed, "
                  f"dist={block_target_dist:.3f}m, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nCollection complete: {n_placed} successful trajectories in {elapsed:.0f}s")

    if not all_trajectories:
        print("ERROR: No successful trajectories collected!")
        return

    # Sort by final_dist (ascending = best first)
    all_trajectories.sort(key=lambda x: x[0])

    # Keep top K%
    k = max(1, int(len(all_trajectories) * args.top_k_pct / 100))
    best_trajs = all_trajectories[:k]

    print(f"\nRejection sampling results:")
    print(f"  Total successful: {len(all_trajectories)}")
    print(f"  Best {args.top_k_pct}% kept: {k}")
    print(f"  Dist range (all):    {all_trajectories[0][0]*100:.1f}cm - {all_trajectories[-1][0]*100:.1f}cm")
    print(f"  Dist range (kept):   {best_trajs[0][0]*100:.1f}cm - {best_trajs[-1][0]*100:.1f}cm")

    # Concatenate into arrays
    images = np.concatenate([t[1] for t in best_trajs], axis=0)
    states = np.concatenate([t[2] for t in best_trajs], axis=0)
    actions = np.concatenate([t[3] for t in best_trajs], axis=0)

    print(f"  Total transitions: {len(actions)}")
    print(f"  Images: {images.shape}, States: {states.shape}, Actions: {actions.shape}")

    # Save
    np.savez(args.output, images=images, states=states, actions=actions)
    print(f"\nSaved to {args.output}")

    # Also save per-trajectory dist stats
    stats = {
        "n_episodes": args.n_episodes,
        "n_successful": len(all_trajectories),
        "top_k_pct": args.top_k_pct,
        "n_kept": k,
        "n_transitions": len(actions),
        "dist_all": [t[0] for t in all_trajectories],
        "dist_kept": [t[0] for t in best_trajs],
        "stochastic": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    stats_path = args.output.replace(".npz", "_stats.json")
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
