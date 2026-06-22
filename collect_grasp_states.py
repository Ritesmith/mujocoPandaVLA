#!/usr/bin/env python3
"""Collect arm states from the grasp policy for place training.

Runs the v3 grasp model in the normal (non-place_mode) environment and
records the arm configuration whenever the block is successfully lifted
(lift > 3cm) and held (gripper closed). These states are then used as
realistic starting positions for place policy training, bridging the
train-eval mismatch: during hierarchical eval, place_mode activates
mid-episode when the grasp policy has already positioned the arm.

Usage:
    python collect_grasp_states.py --n_episodes 100
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle

import gymnasium
import numpy as np
import gym_env  # noqa: F401  registers PandaVLA-v0
from gym_env.wrappers import FlattenObs

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/vec_normalize.pkl"
OUTPUT_PATH = "/home/w/vla_workspace/outputs/grasp_states.pkl"

LIFT_THRESHOLD = 0.03   # m: same as hierarchical policy phase switch
GRIPPER_OPEN_THRESHOLD = 0.03  # m: gripper considered open above this
TABLE_Z = 0.22
MAX_STEPS = 500


def make_env():
    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    return FlattenObs(env)


def main():
    parser = argparse.ArgumentParser(description="Collect grasp states")
    parser.add_argument('--n_episodes', type=int, default=100)
    parser.add_argument('--output', type=str, default=OUTPUT_PATH)
    parser.add_argument('--grasp_model', type=str, default=GRASP_MODEL_PATH)
    parser.add_argument('--grasp_vecnorm', type=str, default=GRASP_VECNORM_PATH)
    args = parser.parse_args()

    # Load grasp model
    print(f"Loading grasp model: {args.grasp_model}")
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load(args.grasp_vecnorm, vec_env)
    vec_env.norm_reward = False
    vec_env.training = False
    model = PPO.load(args.grasp_model, env=vec_env, device="cpu")

    # Access inner env for qpos access
    inner_env = vec_env.envs[0].env.unwrapped

    collected_states = []
    np.random.seed(42)

    for ep in range(args.n_episodes):
        obs = vec_env.reset()
        max_lift = 0.0
        ep_states = []

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            i = info[0]

            block_h = float(i.get("block_height", TABLE_Z))
            gripper_opening = float(i.get("gripper_opening", 0.04))
            lift = max(0.0, block_h - TABLE_Z)

            if lift > max_lift:
                max_lift = lift

            # Collect state when block is lifted and gripper is closed
            if lift > LIFT_THRESHOLD and gripper_opening < GRIPPER_OPEN_THRESHOLD:
                arm_qpos = inner_env.data.qpos[inner_env._arm_qpos_adrs].copy()
                finger_qpos = inner_env.data.qpos[inner_env._finger_qpos_adrs].copy()
                block_pos = i.get("block_position", np.zeros(3)).copy()
                hand_pos = i.get("hand_position", np.zeros(3)).copy()
                target_pos = inner_env._target_pos.copy()

                ep_states.append({
                    'arm_qpos': arm_qpos,
                    'finger_qpos': finger_qpos,
                    'block_pos': block_pos,
                    'hand_pos': hand_pos,
                    'target_pos': target_pos,
                    'lift': lift,
                    'step': step,
                })

            if done[0]:
                break

        if ep_states:
            # Pick one state from this episode (the one with highest lift)
            best_state = max(ep_states, key=lambda s: s['lift'])
            collected_states.append(best_state)
            print(f"Ep {ep:3d}: max_lift={max_lift*100:5.1f}cm  "
                  f"collected lift={best_state['lift']*100:.1f}cm  "
                  f"arm_qpos={best_state['arm_qpos']}")
        else:
            print(f"Ep {ep:3d}: max_lift={max_lift*100:5.1f}cm  "
                  f"(no state collected - block not lifted)")

    vec_env.close()

    # Save collected states
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(collected_states, f)

    print(f"\n{'='*60}")
    print(f"Collected {len(collected_states)} grasp states from {args.n_episodes} episodes")
    print(f"Saved to: {args.output}")

    if collected_states:
        arm_qpos_arr = np.array([s['arm_qpos'] for s in collected_states])
        print(f"\nArm qpos statistics:")
        print(f"  Mean: {arm_qpos_arr.mean(axis=0)}")
        print(f"  Std:  {arm_qpos_arr.std(axis=0)}")
        print(f"  Min:  {arm_qpos_arr.min(axis=0)}")
        print(f"  Max:  {arm_qpos_arr.max(axis=0)}")

        lifts = np.array([s['lift'] for s in collected_states])
        print(f"\nLift statistics:")
        print(f"  Mean: {lifts.mean()*100:.1f}cm")
        print(f"  Min:  {lifts.min()*100:.1f}cm")
        print(f"  Max:  {lifts.max()*100:.1f}cm")


if __name__ == "__main__":
    main()
