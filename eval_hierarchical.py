#!/usr/bin/env python3
"""Evaluate the hierarchical pick-and-place policy.

Loads the v3 grasp model and a place sub-policy, wraps them in
HierarchicalPickPlacePolicy, and runs N episodes tracking:
  - Grab  : max lift height > 3cm  (block was picked up)
  - Place : final block-target dist < 5cm  (block was placed)
  - Pick+Place : both conditions in the same episode (in order)

Output format mirrors analyze_pickplace.py.

Usage:
    python eval_hierarchical.py
    python eval_hierarchical.py --place_model outputs/place_policy/best/best_model.zip
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import gymnasium
import gym_env  # noqa: F401  registers PandaVLA-v0
from gym_env.wrappers import FlattenObs
from hierarchical_policy import HierarchicalPickPlacePolicy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# v3 grasp model (65% grab rate, 0% place rate)
GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/vec_normalize.pkl"

# Default place model path (may not exist yet -> falls back to grasp model)
PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v13/best/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v13/best/vec_normalize.pkl"

LIFT_THRESHOLD = 0.03   # m above table (table_z=0.22)
PLACE_THRESHOLD = 0.05  # m block-target distance
TABLE_Z = 0.22
MAX_STEPS = 500
N_EPISODES = 20
SEED = 42


def make_env():
    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    return FlattenObs(env)


def load_model(model_path, vecnorm_path, env_factory):
    """Load an SB3 PPO model with its VecNormalize stats."""
    vec_env = DummyVecEnv([env_factory])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0)
        vec_env.training = False
    model = PPO.load(model_path, env=vec_env, device="auto")
    return model, vec_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical pick-place")
    parser.add_argument('--place_model', type=str, default=PLACE_MODEL_PATH,
                        help='Path to place policy .zip')
    parser.add_argument('--place_vecnorm', type=str, default=PLACE_VECNORM_PATH,
                        help='Path to place policy vec_normalize.pkl')
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--release_threshold', type=float, default=0.05,
                        help='Release distance threshold (m) for place phase')
    parser.add_argument('--release_height', type=float, default=0.0,
                        help='Max block height above table (m) to allow release. 0=no gate')
    parser.add_argument('--freeze_arm_on_release', action='store_true',
                        help='Freeze arm movement after gripper opens to prevent pushing block')
    args = parser.parse_args()

    # ---- Load grasp model (v3) ----
    print(f"Loading grasp model: {GRASP_MODEL_PATH}")
    grasp_model, grasp_vec_env = load_model(
        GRASP_MODEL_PATH, GRASP_VECNORM_PATH, make_env
    )

    # ---- Load place model (or fall back to grasp model) ----
    place_fallback = False
    if os.path.exists(args.place_model):
        print(f"Loading place model: {args.place_model}")
        place_model, place_vec_env = load_model(
            args.place_model, args.place_vecnorm, make_env
        )
    else:
        print(f"WARNING: place model not found at {args.place_model}")
        print("  Falling back to grasp model for the place phase.")
        place_model = grasp_model
        place_vec_env = grasp_vec_env
        place_fallback = True

    # ---- Build hierarchical policy ----
    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    # ---- Eval environment (raw, unwrapped) ----
    # We normalize observations manually per-phase using grasp_vec_env /
    # place_vec_env stats, so the eval env must NOT wrap VecNormalize. This
    # fixes the VecNormalize mismatch bug where the place model was receiving
    # observations normalized with the grasp model's stats.
    raw_env = DummyVecEnv([make_env])

    # Access the unwrapped PandaVLAEnv to toggle place_mode mid-episode.
    # The place policy was trained in place_mode (block hard-attached to
    # hand). Switching the env to place_mode during the place phase ensures
    # the block follows the hand as the policy expects, fixing the
    # train-eval environment mismatch that caused 0% place rate.
    _inner_env = raw_env.envs[0].env.unwrapped
    # Set configurable release threshold for the place phase
    _inner_env._release_dist_threshold = args.release_threshold
    # Set height gate: only allow release when block is near the table.
    # 0 means no height gate (use infinity). Otherwise set to the value.
    if args.release_height > 0:
        _inner_env._release_height_threshold = args.release_height
    else:
        _inner_env._release_height_threshold = float('inf')
    print(f"Release threshold: {args.release_threshold}m, height gate: "
          f"{args.release_height if args.release_height > 0 else 'off'}m")

    np.random.seed(SEED)
    try:
        raw_env.seed(SEED)
    except Exception:
        pass

    # ---- Run episodes ----
    grab_flags, place_flags, pickplace_flags = [], [], []
    max_lifts, final_dists = [], []
    phase_switch_steps = []

    for ep in range(args.n_episodes):
        # Reset place_mode to False for the grasp phase (block on table)
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        ep_reward = 0.0
        max_lift = 0.0
        block_target_dist = float("inf")
        block_grabbed_at = None
        first_place_step = None
        prev_info = None  # info from previous step (None on first step)
        arm_frozen = False  # freeze arm after gripper opens

        for step in range(args.max_steps):
            # Detect phase from previous step's info. This call is idempotent
            # with the _detect_phase call inside policy.predict() below, so
            # calling it twice with the same info is safe (the second call
            # sees the phase already updated and makes no further change).
            phase = policy._detect_phase(prev_info)

            # When place_mode first activates, snap the block to the hand
            # immediately so the first place-policy observation reflects
            # the block at its trained position (hand_pos - 5cm). Without
            # this, the first place action is based on the grasp-phase obs
            # where the block is still at its old position.
            if phase == "place" and first_place_step is None:
                first_place_step = step
                _inner_env.place_mode = True
                _inner_env._place_gravcomp_active = True
                _inner_env.snap_block_to_hand()

                # CRITICAL: sync _arm_target and _gripper_target with the
                # current qpos. During the grasp phase, _arm_target drifts
                # from arm_qpos (due to action clipping, joint limits, etc).
                # Without this sync, the first place-model action (arm_delta)
                # is added to the stale _arm_target, causing the arm to jump
                # to a wrong position. This was the root cause of 0% place
                # rate in hierarchical eval despite place-only eval succeeding.
                _inner_env._arm_target = _inner_env.data.qpos[
                    _inner_env._arm_qpos_adrs
                ].copy()
                _inner_env._gripper_target = float(
                    _inner_env.data.qpos[_inner_env._finger_qpos_adrs].mean()
                )

                # CRITICAL: switch reward_type to "place_only" so that
                # _place_success is set when the block is placed (dist<5cm,
                # on table, gripper open). Without this, the episode never
                # terminates on place success, the place model keeps running,
                # and the block drifts away from the target — causing 0%
                # place rate despite the place model reaching <5cm.
                # Also reset the place reward state variables so the
                # one-time bonuses and progress tracking start fresh.
                _inner_env.reward_type = "place_only"
                _inner_env._place_approach_bonus_given = False
                _inner_env._place_proximity_15_given = False
                _inner_env._place_proximity_10_given = False
                _inner_env._place_success = False
                _inner_env._prev_block_target_dist = None
                _inner_env._prev_block_height = None
                # Use _gripper_target for the gripper_open check in eval
                # (immediate termination on success). Training uses
                # data.qpos (allows multi-step release bonus for stronger
                # gradient). See panda_vla_env.py _compute_reward_place
                # for details.
                _inner_env._use_gripper_target_check = True

                # Re-get raw_obs through the FlattenObs wrapper so the
                # block_position / hand_block_distance / block_target_distance
                # dims reflect the snapped block.
                flatten_wrapper = raw_env.envs[0]
                inner_obs = _inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

            # Normalize obs with the VecNormalize stats matching the active
            # sub-policy. grasp_vec_env holds the grasp model's stats;
            # place_vec_env holds the place model's stats. Feeding the place
            # model grasp-normalized obs (the old bug) produced garbage
            # actions during the place phase.
            if phase == "place":
                obs = place_vec_env.normalize_obs(raw_obs)
            else:
                obs = grasp_vec_env.normalize_obs(raw_obs)

            # Predict (calls _detect_phase again — idempotent, see above).
            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            # Freeze arm after gripper opens to prevent pushing the
            # falling/landed block. Only active during place phase.
            if (args.freeze_arm_on_release and not arm_frozen
                    and first_place_step is not None):
                if _inner_env._gripper_target > 0.02:
                    arm_frozen = True
            if arm_frozen:
                action[0][:] = 0.0  # zero arm deltas + gripper cmd

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]  # for next prediction
            ep_reward += float(reward[0])
            i = info[0]

            block_h = float(i.get("block_height", 0.0))
            block_target_dist = float(
                i.get("block_target_distance", block_target_dist)
            )

            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            if lift > LIFT_THRESHOLD and block_grabbed_at is None:
                block_grabbed_at = step

            if done[0]:
                break

        grabbed = max_lift > LIFT_THRESHOLD
        placed = block_target_dist < PLACE_THRESHOLD
        pickplace = grabbed and placed

        grab_flags.append(grabbed)
        place_flags.append(placed)
        pickplace_flags.append(pickplace)
        max_lifts.append(max_lift)
        final_dists.append(block_target_dist)
        phase_switch_steps.append(first_place_step)

        print(f"Ep {ep:2d}: max_lift={max_lift*100:5.1f}cm  "
              f"final_dist={block_target_dist*100:5.1f}cm  "
              f"place_step={first_place_step if first_place_step is not None else '  -'}  "
              f"grab={'Y' if grabbed else 'N'} place={'Y' if placed else 'N'}")

    raw_env.close()

    # ---- Summary ----
    print()
    print("=" * 64)
    title = "Hierarchical Pick-and-Place Summary"
    if place_fallback:
        title += "  (place model = grasp fallback)"
    print(title)
    print("=" * 64)
    print(f"Episodes              : {args.n_episodes}")
    print(f"Grab  (lift>{LIFT_THRESHOLD*100:.0f}cm)   : "
          f"{sum(grab_flags)}/{args.n_episodes} "
          f"({100*sum(grab_flags)/args.n_episodes:.0f}%)")
    print(f"Place (dist<{PLACE_THRESHOLD*100:.0f}cm)  : "
          f"{sum(place_flags)}/{args.n_episodes} "
          f"({100*sum(place_flags)/args.n_episodes:.0f}%)")
    print(f"Pick+Place (both)     : "
          f"{sum(pickplace_flags)}/{args.n_episodes} "
          f"({100*sum(pickplace_flags)/args.n_episodes:.0f}%)")
    print(f"Mean max lift         : {np.mean(max_lifts)*100:.1f} cm")
    print(f"Best max lift         : {np.max(max_lifts)*100:.1f} cm")
    print(f"Mean final dist       : {np.mean(final_dists)*100:.1f} cm")
    print(f"Best final dist       : {np.min(final_dists)*100:.1f} cm")

    valid_switches = [s for s in phase_switch_steps if s is not None]
    if valid_switches:
        print(f"Phase switches (grasp->place): {len(valid_switches)}/"
              f"{args.n_episodes} episodes")
        print(f"  Mean switch step : {np.mean(valid_switches):.1f}")
    else:
        print("Phase switches (grasp->place): 0 (never reached place phase)")


if __name__ == "__main__":
    main()
