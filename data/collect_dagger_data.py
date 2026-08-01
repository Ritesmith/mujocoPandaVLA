"""Collect DAgger training data: V59 states + oracle expert actions.

Runs the hierarchical policy (V59) to visit states, but labels each
place-phase state with the analytical oracle's expert action (not V59's
own action). This is the DAgger principle: train on the policy's state
distribution with expert labels.

Output: D_dagger.npz (images, states, actions) where actions are ORACLE
actions (not V59 actions). bc_loss will be > 0 because the oracle's
analytical solution differs from V59's learned mean.

Usage:
    python collect_dagger_data.py --n_episodes 200
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
from core.hierarchical_policy import HierarchicalPickPlacePolicy
from data.dagger_oracle import DAggerOracle, DAggerOracleV2

GRASP_MODEL = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"
OUTPUT = "/home/w/vla_workspace/data/D_dagger.npz"

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
    parser = argparse.ArgumentParser(description="Collect DAgger training data")
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
                        help="Oracle version: v1=binary gripper, v2=V59-style gripper")
    args = parser.parse_args()

    # When v2 is selected and output not explicitly overridden, use v2 default
    if args.oracle_version == "v2" and args.output == OUTPUT:
        args.output = "/home/w/vla_workspace/data/D_dagger_v2.npz"

    target_range = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]

    print("=" * 60)
    print("DAgger Data Collection")
    print("=" * 60)
    print(f"Place model: {PLACE_MODEL}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Policy: V59 deterministic (collects states it visits)")
    print(f"Labels: Analytical oracle (Jacobian-based IK)")
    print(f"Oracle version: {args.oracle_version}")
    print(f"Oracle gain: {args.oracle_gain}, max_speed: {args.oracle_max_speed}")
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
    inner._release_dist_threshold = args.release_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    # DAgger oracle
    if args.oracle_version == "v2":
        oracle = DAggerOracleV2(
            inner, gain=args.oracle_gain, max_speed=args.oracle_max_speed,
            release_threshold=args.release_threshold)
        print("Oracle initialized (DAggerOracleV2: Jacobian IK arm + V59-style gripper)")
    else:
        oracle = DAggerOracle(
            inner, gain=args.oracle_gain, max_speed=args.oracle_max_speed,
            release_threshold=args.release_threshold)
        print("Oracle initialized (Jacobian-based proportional controller)")
    print()

    np.random.seed(args.seed)
    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    all_images = []
    all_states = []
    all_actions = []
    all_oracle_dists = []

    n_entered_place = 0
    n_collected = 0
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
        ep_dists = []

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

                # V59 action (deterministic) — this is what gets executed
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

                # Oracle expert action — this is the LABEL for training
                oracle_action = oracle.get_expert_action()
                oracle_dist = float(np.linalg.norm(
                    inner.data.xpos[inner._red_block_id] - inner._target_pos))

                # Record (image, state, oracle_action) — NOT V59's action
                ep_images.append(vision_obs["image"].copy())
                ep_states.append(vision_obs["state"].copy())
                ep_actions.append(oracle_action)
                ep_dists.append(oracle_dist)
            else:
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

        if first_place_step is not None and max_lift > LIFT_THRESHOLD:
            n_entered_place += 1
            all_images.extend(ep_images)
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
            all_oracle_dists.extend(ep_dists)
            n_collected += len(ep_images)

        elapsed = time.time() - t0
        ep_status = "place" if first_place_step is not None else "no_place"
        print(f"Ep {ep:3d}: {ep_status:9s}  dist={block_target_dist*100:5.1f}cm  "
              f"steps={len(ep_images):3d}  | total={n_collected}  [{elapsed:.0f}s]")

        if (ep + 1) % 50 == 0:
            print(f"  >> Checkpoint: {n_collected} transitions from {n_entered_place} episodes")

    elapsed = time.time() - t0
    print(f"\nCollection complete: {n_collected} transitions in {elapsed:.0f}s")

    if n_collected == 0:
        print("ERROR: No transitions collected!")
        return

    images = np.array(all_images, dtype=np.uint8)
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    dists = np.array(all_oracle_dists, dtype=np.float32)

    print(f"\nDataset statistics:")
    print(f"  Transitions: {len(actions)}")
    print(f"  Images: {images.shape}, States: {states.shape}, Actions: {actions.shape}")
    print(f"  Oracle action mean: {actions.mean(axis=0)}")
    print(f"  Oracle action std:  {actions.std(axis=0)}")
    print(f"  Oracle action abs mean: {np.abs(actions).mean(axis=0)}")
    print(f"  Dist range: {dists.min()*100:.1f}cm - {dists.max()*100:.1f}cm")
    print(f"  Dist mean: {dists.mean()*100:.1f}cm")

    np.savez(args.output, images=images, states=states, actions=actions)
    print(f"\nSaved to {args.output}")

    import json
    stats = {
        "n_episodes": args.n_episodes,
        "n_entered_place": n_entered_place,
        "n_transitions": n_collected,
        "oracle_version": args.oracle_version,
        "oracle_gain": args.oracle_gain,
        "oracle_max_speed": args.oracle_max_speed,
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).tolist(),
        "dist_min": float(dists.min()),
        "dist_max": float(dists.max()),
        "dist_mean": float(dists.mean()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    stats_path = args.output.replace(".npz", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
