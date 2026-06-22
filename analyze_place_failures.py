#!/usr/bin/env python3
"""Analyze place-phase failures in hierarchical eval.

Runs v11 + MIN_GRASP_STEPS=20 and logs per-episode details:
  - Block height / dist at phase switch
  - Min dist during place phase
  - Gripper open step + dist at that step
  - Final dist / height / on_table
  - Episode termination reason

Usage:
    python analyze_place_failures.py --n_episodes 50
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs
from hierarchical_policy import HierarchicalPickPlacePolicy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vola_workspace/outputs/dapg_500k_v3/vec_normalize.pkl"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/vec_normalize.pkl"

PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v11/best/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v11/best/vec_normalize.pkl"

LIFT_THRESHOLD = 0.03
PLACE_THRESHOLD = 0.05
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42


def make_env():
    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    return FlattenObs(env)


def load_model(model_path, vecnorm_path, env_factory):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--release_threshold', type=float, default=0.10,
                        help='Max block-target dist to allow gripper open (m)')
    parser.add_argument('--smart_release', action='store_true',
                        help='Force gripper open when block starts moving away from target')
    args = parser.parse_args()

    print("Loading grasp model...")
    grasp_model, grasp_vec_env = load_model(
        GRASP_MODEL_PATH, GRASP_VECNORM_PATH, make_env
    )
    print("Loading place model...")
    place_model, place_vec_env = load_model(
        PLACE_MODEL_PATH, PLACE_VECNORM_PATH, make_env
    )

    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)
    raw_env = DummyVecEnv([make_env])
    _inner_env = raw_env.envs[0].env.unwrapped

    np.random.seed(SEED)
    try:
        raw_env.seed(SEED)
    except Exception:
        pass

    results = []

    for ep in range(args.n_episodes):
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()

        max_lift = 0.0
        block_target_dist = float("inf")
        first_place_step = None
        prev_info = None

        # Place-phase tracking
        switch_height = None
        switch_dist = None
        min_place_dist = float("inf")
        min_place_dist_step = None
        gripper_open_step = None
        gripper_open_dist = None
        gripper_open_height = None
        final_height = None
        done_reason = None
        total_steps = 0

        for step in range(args.max_steps):
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                _inner_env.place_mode = True
                _inner_env._place_gravcomp_active = True
                _inner_env.snap_block_to_hand()
                _inner_env._arm_target = _inner_env.data.qpos[
                    _inner_env._arm_qpos_adrs].copy()
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
                _inner_env._release_dist_threshold = args.release_threshold

                flatten_wrapper = raw_env.envs[0]
                inner_obs = _inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

                # Record switch state
                i = prev_info or {}
                bh = float(i.get("block_height", TABLE_Z))
                bd = float(i.get("block_target_distance", float("inf")))
                switch_height = max(0.0, bh - TABLE_Z)
                switch_dist = bd

            if phase == "place":
                obs = place_vec_env.normalize_obs(raw_obs)
            else:
                obs = grasp_vec_env.normalize_obs(raw_obs)

            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            # Smart release: force gripper open when block starts moving
            # away from target after being close. The model often navigates
            # to 2-3cm from target, then overshoots to 7-8cm before opening
            # the gripper. This catches the optimal release moment.
            if (args.smart_release and first_place_step is not None
                    and gripper_open_step is None
                    and min_place_dist < 0.05
                    and block_target_dist > min_place_dist + 0.01):
                action[0][-1] = -1.0  # force gripper open

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            total_steps = step + 1

            i = info[0]
            block_h = float(i.get("block_height", 0.0))
            block_target_dist = float(
                i.get("block_target_distance", block_target_dist)
            )
            gripper_opening = float(i.get("gripper_opening", 0.04))

            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift

            # Track place-phase details
            if first_place_step is not None:
                if block_target_dist < min_place_dist:
                    min_place_dist = block_target_dist
                    min_place_dist_step = step - first_place_step

                # Detect gripper opening (first time opening beyond threshold)
                if gripper_open_step is None and gripper_opening > 0.03:
                    gripper_open_step = step - first_place_step
                    gripper_open_dist = block_target_dist
                    gripper_open_height = lift

            if done[0]:
                if _inner_env._place_success:
                    done_reason = "place_success"
                elif lift < 0.01:
                    done_reason = "block_dropped"
                else:
                    done_reason = "other_done"
                break

        final_height = lift
        placed = block_target_dist < PLACE_THRESHOLD
        grabbed = max_lift > LIFT_THRESHOLD

        r = {
            "ep": ep,
            "grabbed": grabbed,
            "placed": placed,
            "max_lift": max_lift,
            "final_dist": block_target_dist,
            "final_height": final_height,
            "switch_step": first_place_step,
            "switch_height": switch_height,
            "switch_dist": switch_dist,
            "min_place_dist": min_place_dist if min_place_dist < 1e6 else None,
            "min_place_dist_step": min_place_dist_step,
            "gripper_open_step": gripper_open_step,
            "gripper_open_dist": gripper_open_dist,
            "gripper_open_height": gripper_open_height,
            "done_reason": done_reason,
            "total_steps": total_steps,
        }
        results.append(r)

        status = "OK" if placed else "FAIL"
        sh = f"{switch_height*100:.1f}" if switch_height is not None else "--"
        sd = f"{switch_dist*100:.1f}" if switch_dist is not None else "--"
        md = f"{min_place_dist*100:.1f}" if min_place_dist < 1e6 else "--"
        go = str(gripper_open_step) if gripper_open_step is not None else "--"
        od = f"{gripper_open_dist*100:.1f}" if gripper_open_dist is not None else "--"
        print(f"Ep {ep:2d} [{status}] max_lift={max_lift*100:5.1f}cm "
              f"final_d={block_target_dist*100:5.1f}cm "
              f"final_h={final_height*100:4.1f}cm "
              f"sw_h={sh:>4}cm sw_d={sd:>5}cm "
              f"min_d={md:>5}cm "
              f"grip_open={go:>3} open_d={od:>5}cm "
              f"done={done_reason} steps={total_steps}")

    raw_env.close()

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 80)

    successes = [r for r in results if r["placed"]]
    failures = [r for r in results if not r["placed"]]
    print(f"Total: {len(results)}, Success: {len(successes)}, "
          f"Failure: {len(failures)}")
    print(f"Place rate: {100*len(successes)/len(results):.0f}%")

    if successes:
        print(f"\n--- Successes ({len(successes)} eps) ---")
        print(f"  switch_height: {np.mean([r['switch_height'] for r in successes])*100:.1f} cm")
        print(f"  switch_dist:   {np.mean([r['switch_dist'] for r in successes])*100:.1f} cm")
        print(f"  min_place_dist: {np.mean([r['min_place_dist'] for r in successes])*100:.1f} cm")
        print(f"  grip_open_step: {np.mean([r['gripper_open_step'] for r in successes if r['gripper_open_step'] is not None]):.1f}")
        print(f"  grip_open_dist: {np.mean([r['gripper_open_dist'] for r in successes if r['gripper_open_dist'] is not None])*100:.1f} cm")
        print(f"  final_dist:    {np.mean([r['final_dist'] for r in successes])*100:.1f} cm")
        print(f"  final_height:  {np.mean([r['final_height'] for r in successes])*100:.1f} cm")
        print(f"  total_steps:   {np.mean([r['total_steps'] for r in successes]):.1f}")

    if failures:
        print(f"\n--- Failures ({len(failures)} eps) ---")
        print(f"  switch_height: {np.mean([r['switch_height'] for r in failures if r['switch_height'] is not None])*100:.1f} cm")
        print(f"  switch_dist:   {np.mean([r['switch_dist'] for r in failures if r['switch_dist'] is not None])*100:.1f} cm")
        print(f"  min_place_dist: {np.mean([r['min_place_dist'] for r in failures if r['min_place_dist'] is not None])*100:.1f} cm")
        print(f"  grip_open_step: {np.mean([r['gripper_open_step'] for r in failures if r['gripper_open_step'] is not None]):.1f}")
        print(f"  grip_open_dist: {np.mean([r['gripper_open_dist'] for r in failures if r['gripper_open_dist'] is not None])*100:.1f} cm")
        print(f"  final_dist:    {np.mean([r['final_dist'] for r in failures])*100:.1f} cm")
        print(f"  final_height:  {np.mean([r['final_height'] for r in failures])*100:.1f} cm")
        print(f"  total_steps:   {np.mean([r['total_steps'] for r in failures]):.1f}")

        # Categorize failures
        print(f"\n--- Failure Categories ---")
        never_opened = [r for r in failures if r["gripper_open_step"] is None]
        opened_too_early = [r for r in failures
                           if r["gripper_open_step"] is not None
                           and r["gripper_open_dist"] is not None
                           and r["gripper_open_dist"] > 0.05]
        opened_close = [r for r in failures
                       if r["gripper_open_step"] is not None
                       and r["gripper_open_dist"] is not None
                       and r["gripper_open_dist"] <= 0.05]
        min_dist_close = [r for r in failures
                         if r["min_place_dist"] is not None
                         and r["min_place_dist"] <= 0.07]
        min_dist_far = [r for r in failures
                       if r["min_place_dist"] is not None
                       and r["min_place_dist"] > 0.07]

        print(f"  Never opened gripper:     {len(never_opened)}")
        print(f"  Opened too early (>5cm):  {len(opened_too_early)}")
        print(f"  Opened close (<=5cm):     {len(opened_close)}")
        print(f"  Got close (min<=7cm):     {len(min_dist_close)}")
        print(f"  Never got close (min>7cm): {len(min_dist_far)}")

        # Done reasons
        print(f"\n--- Done Reasons (failures) ---")
        for reason in set(r["done_reason"] for r in failures):
            count = sum(1 for r in failures if r["done_reason"] == reason)
            print(f"  {reason}: {count}")

        # Final height distribution
        print(f"\n--- Final Height Distribution (failures) ---")
        for r in failures:
            print(f"  Ep {r['ep']:2d}: final_h={r['final_height']*100:5.1f}cm "
                  f"final_d={r['final_dist']*100:5.1f}cm "
                  f"min_d={r['min_place_dist']*100 if r['min_place_dist'] else '--':>5.1f}cm "
                  f"open_step={r['gripper_open_step']} "
                  f"open_d={r['gripper_open_dist']*100 if r['gripper_open_dist'] else '--':>5.1f}cm")


if __name__ == "__main__":
    main()
