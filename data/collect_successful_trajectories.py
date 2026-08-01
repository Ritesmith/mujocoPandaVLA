#!/usr/bin/env python3
"""Collect successful place-phase trajectories from V59 for OPR self-imitation.

Runs the hierarchical pick-and-place policy (v5 grasp + V59 vision place) in
eval mode for N episodes. During the place phase, captures (image, state,
action) transitions from the V59 place policy. After each episode, if the
block was placed successfully (dist < 5cm), the place-phase trajectory is
saved to D_succ.pkl.

These self-generated successful trajectories form the OPR (Off-Policy
Regularization) self-imitation buffer: unlike external demos (bc_loss=0.503,
RMS error ~0.7), the policy's own successful actions are by definition
aligned with its learned strategy, so BC anchoring pulls toward better (not
worse) behavior.

Output format (D_succ.pkl): list of trajectory dicts, each =
    {
        "image":       np.array (N, 84, 84, 3) uint8,   # raw HWC, unnormalized
        "state":       np.array (N, 12)       float32,  # raw, unnormalized
        "action":      np.array (N, 8)        float32,  # [-1, 1], deterministic
        "target_pos":  np.array (3,)          float32,
        "final_dist":  float,                            # m, block-target dist
        "place_steps": int,                              # steps in place phase
        "max_lift":    float,                            # m, max lift in episode
    }

Usage:
    python collect_successful_trajectories.py
    python collect_successful_trajectories.py --n_episodes 200 --target_success 80
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
import time
import numpy as np
import gymnasium
import gym_env  # noqa: F401  registers PandaVLA-v0
from gym_env.wrappers import FlattenObs

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from core.hierarchical_policy import HierarchicalPickPlacePolicy


# ---- Model paths (same as eval_hierarchical.py V59 deployment) ----
GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"

TARGET_RANGE = "0.35,0.15,0.22,0.65,0.45,0.22"
OUTPUT_PATH = "/home/w/vla_workspace/data/D_succ.pkl"

LIFT_THRESHOLD = 0.03   # m, grab success
PLACE_THRESHOLD = 0.05  # m, place success
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42
PHASE_SWITCH_LIFT = 0.02  # matches hierarchical_policy.GRASP_TO_PLACE_LIFT


def make_env(vision_mode=False, target_pos_range=None, domain_randomize=False):
    kwargs = dict(reward_type="dense", gravity_comp=True)
    if target_pos_range is not None:
        kwargs["target_pos_range"] = target_pos_range
    kwargs["domain_randomize"] = domain_randomize
    env = gymnasium.make("PandaVLA-v0", **kwargs)
    if vision_mode:
        from gym_env.wrappers import VisionObs
        env = VisionObs(env, image_size=84)
    else:
        env = FlattenObs(env)
    return env


def load_model(model_path, vecnorm_path, vision_mode=False,
               target_pos_range=None, domain_randomize=False):
    env_factory = lambda: make_env(
        vision_mode=vision_mode,
        target_pos_range=target_pos_range,
        domain_randomize=domain_randomize,
    )
    vec_env = DummyVecEnv([env_factory])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False
    else:
        norm_obs_keys = ["state"] if vision_mode else None
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, norm_obs_keys=norm_obs_keys)
        vec_env.training = False
    model = PPO.load(model_path, env=vec_env, device="auto")
    return model, vec_env


def main():
    parser = argparse.ArgumentParser(
        description="Collect successful place trajectories for OPR self-imitation")
    parser.add_argument('--place_model', type=str, default=PLACE_MODEL_PATH)
    parser.add_argument('--place_vecnorm', type=str, default=PLACE_VECNORM_PATH)
    parser.add_argument('--grasp_model', type=str, default=GRASP_MODEL_PATH)
    parser.add_argument('--grasp_vecnorm', type=str, default=GRASP_VECNORM_PATH)
    parser.add_argument('--n_episodes', type=int, default=200,
                        help='Total episodes to run (default 200)')
    parser.add_argument('--target_success', type=int, default=80,
                        help='Stop early once this many successful trajectories collected')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--release_threshold', type=float, default=0.05,
                        help='Release distance threshold (m). Must match eval '
                             'conditions that produced V59 56% place rate.')
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    parser.add_argument('--checkpoint_freq', type=int, default=50,
                        help='Save partial results every N episodes')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()

    target_pos_range = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]

    print("=" * 60)
    print("OPR Self-Imitation Trajectory Collection")
    print("=" * 60)
    print(f"Place model: {args.place_model}")
    print(f"Grasp model: {args.grasp_model}")
    print(f"Target range: {TARGET_RANGE}")
    print(f"Episodes: {args.n_episodes} (stop early at {args.target_success} successes)")
    print(f"Output: {args.output}")
    print(f"Domain randomization: disabled (matches V59 training/eval)")
    print()

    # ---- Load grasp model (state-only, 16-dim) ----
    print(f"Loading grasp model: {args.grasp_model}")
    grasp_model, grasp_vec_env = load_model(
        args.grasp_model, args.grasp_vecnorm,
        vision_mode=False,
        target_pos_range=target_pos_range,
        domain_randomize=False,
    )

    # ---- Load place model (vision, Dict {image, state}) ----
    print(f"Loading place model: {args.place_model}")
    place_model, place_vec_env = load_model(
        args.place_model, args.place_vecnorm,
        vision_mode=True,
        target_pos_range=target_pos_range,
        domain_randomize=False,
    )

    # ---- Build hierarchical policy ----
    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    # ---- Eval environment (raw, unwrapped — FlattenObs 16-dim) ----
    raw_env = DummyVecEnv([lambda: make_env(
        target_pos_range=target_pos_range,
        domain_randomize=False,
    )])

    _inner_env = raw_env.envs[0].env.unwrapped
    _inner_env._release_dist_threshold = args.release_threshold
    _inner_env._release_height_threshold = float('inf')

    # Vision wrapper for constructing Dict obs during place phase
    from gym_env.wrappers import VisionObs
    place_vision_wrapper = VisionObs(_inner_env, image_size=84)
    print("Vision mode: place model uses Dict obs {image, state}")
    print()

    np.random.seed(args.seed)
    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    # ---- Run episodes and collect successful trajectories ----
    successful_trajectories = []
    n_placed = 0
    n_grabbed = 0
    n_entered_place = 0
    total_place_transitions = 0

    t0 = time.time()

    for ep in range(args.n_episodes):
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        ep_target_pos = _inner_env._target_pos.copy()
        max_lift = 0.0
        block_target_dist = float("inf")
        first_place_step = None
        prev_info = None

        # Per-episode place-phase transition buffers
        ep_images = []
        ep_states = []
        ep_actions = []

        for step in range(args.max_steps):
            phase = policy._detect_phase(prev_info)

            # ---- Phase switch: activate place_mode, snap block to hand ----
            if phase == "place" and first_place_step is None:
                first_place_step = step
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

                # Re-get raw_obs through FlattenObs (reflects snapped block)
                flatten_wrapper = raw_env.envs[0]
                inner_obs = _inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

            # ---- Construct obs for the active phase ----
            if phase == "place":
                # Vision obs: Dict {image, state} from raw env obs
                vision_obs = place_vision_wrapper.observation(
                    _inner_env._get_obs()
                )
                vision_obs_batched = {
                    "image": vision_obs["image"][np.newaxis, ...],
                    "state": vision_obs["state"][np.newaxis, ...],
                }
                obs = place_vec_env.normalize_obs(vision_obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))  # HWC→CHW
            else:
                raw_obs_for_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_for_grasp[0, 8:11]
                default_target = np.array([0.5, 0.3, 0.2])
                raw_obs_for_grasp[0, 15] = np.linalg.norm(block_pos - default_target)
                obs = grasp_vec_env.normalize_obs(raw_obs_for_grasp)

            # ---- Predict action ----
            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            # ---- Capture (image, state, action) during place phase ----
            # Store RAW (unnormalized) data — training pipeline normalizes
            # internally via _normalize_demo_state() and permute(0,3,1,2).
            if phase == "place":
                ep_images.append(vision_obs["image"].copy())    # (84,84,3) uint8
                ep_states.append(vision_obs["state"].copy())    # (12,) float32
                ep_actions.append(action[0].copy())             # (8,) float32

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]

            i = info[0]
            block_h = float(i.get("block_height", 0.0))
            block_target_dist = float(
                i.get("block_target_distance", block_target_dist)
            )
            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift

            if done[0]:
                break

        # ---- Episode end: check success ----
        grabbed = max_lift > LIFT_THRESHOLD
        entered_place = first_place_step is not None
        placed = block_target_dist < PLACE_THRESHOLD

        if grabbed:
            n_grabbed += 1
        if entered_place:
            n_entered_place += 1

        if placed and len(ep_images) > 0:
            n_placed += 1
            traj = {
                "image": np.array(ep_images, dtype=np.uint8),       # (N,84,84,3)
                "state": np.array(ep_states, dtype=np.float32),     # (N,12)
                "action": np.array(ep_actions, dtype=np.float32),   # (N,8)
                "target_pos": ep_target_pos.copy(),
                "final_dist": float(block_target_dist),
                "place_steps": len(ep_images),
                "max_lift": float(max_lift),
            }
            successful_trajectories.append(traj)
            total_place_transitions += len(ep_images)

        ep_status = "PLACED" if placed else ("grabbed" if grabbed else "failed")
        elapsed = time.time() - t0
        print(f"Ep {ep:3d}: {ep_status:7s}  dist={block_target_dist*100:5.1f}cm  "
              f"lift={max_lift*100:5.1f}cm  place_steps={len(ep_images):3d}  "
              f"| succ={n_placed}/{ep+1}  transitions={total_place_transitions}  "
              f"[{elapsed:.0f}s]")

        # ---- Checkpoint ----
        if (ep + 1) % args.checkpoint_freq == 0:
            _save_trajectories(successful_trajectories, args.output)
            print(f"  >> Checkpoint saved: {len(successful_trajectories)} trajectories "
                  f"({total_place_transitions} transitions) to {args.output}")

        # ---- Early stop if target reached ----
        if n_placed >= args.target_success:
            print(f"\nTarget reached: {n_placed} successful trajectories collected.")
            break

    # ---- Final save ----
    _save_trajectories(successful_trajectories, args.output)

    print()
    print("=" * 60)
    print("Collection Complete")
    print("=" * 60)
    print(f"Episodes run:        {ep + 1}")
    print(f"Grabbed (lift>3cm):  {n_grabbed}/{ep+1} ({100*n_grabbed/(ep+1):.0f}%)")
    print(f"Entered place phase: {n_entered_place}/{ep+1} ({100*n_entered_place/(ep+1):.0f}%)")
    print(f"Placed (dist<5cm):   {n_placed}/{ep+1} ({100*n_placed/(ep+1):.0f}%)")
    print(f"Successful traj:     {len(successful_trajectories)}")
    print(f"Total transitions:   {total_place_transitions}")
    if successful_trajectories:
        steps_list = [t["place_steps"] for t in successful_trajectories]
        dist_list = [t["final_dist"] for t in successful_trajectories]
        print(f"Place steps/traj:    mean={np.mean(steps_list):.1f}  "
              f"min={min(steps_list)}  max={max(steps_list)}")
        print(f"Final dist (cm):     mean={np.mean(dist_list)*100:.2f}  "
              f"min={min(dist_list)*100:.2f}  max={max(dist_list)*100:.2f}")
    print(f"Output: {args.output}")
    print(f"Elapsed: {time.time() - t0:.0f}s")

    raw_env.close()


def _save_trajectories(trajectories, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(trajectories, f)


if __name__ == "__main__":
    main()
