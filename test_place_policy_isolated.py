#!/usr/bin/env python3
"""Test place policy v3 in isolation (place_mode_realistic)."""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
sys.path.insert(0, "/home/w/vla_workspace")

import numpy as np
import gymnasium
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PLACE_MODEL = "/home/w/vla_workspace/outputs/place_policy_v3/best/best_model.zip"
PLACE_VECNORM = "/home/w/vla_workspace/outputs/place_policy_v3/vec_normalize.pkl"
TARGET = np.array([0.5, 0.3, 0.2])
TABLE_Z = 0.22
N_EPISODES = 10
MAX_STEPS = 200

def make_env():
    env = gymnasium.make(
        "PandaVLA-v0",
        reward_type="place_only",
        place_mode_realistic=True,
        gravity_comp=True,
        target_pos=TARGET,
    )
    return FlattenObs(env)

def main():
    # Create env with VecNormalize
    vec_env = DummyVecEnv([make_env])
    if os.path.exists(PLACE_VECNORM):
        vec_env = VecNormalize.load(PLACE_VECNORM, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vec_env.training = False

    model = PPO.load(PLACE_MODEL, env=vec_env, device="cpu")

    print(f"Testing place policy in place_mode_realistic")
    print(f"Target: {TARGET}")
    print(f"Episodes: {N_EPISODES}, Max steps: {MAX_STEPS}")
    print("=" * 60)

    placed_count = 0
    for ep in range(N_EPISODES):
        obs = vec_env.reset()
        block_h = TABLE_Z
        block_target_dist = float("inf")
        min_dist = float("inf")
        gripper_opened = False

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            i = info[0]
            block_h = float(i.get("block_height", TABLE_Z))
            block_target_dist = float(i.get("block_target_distance", block_target_dist))
            gripper_opening = float(i.get("gripper_opening", 0.04))

            if block_target_dist < min_dist:
                min_dist = block_target_dist

            if gripper_opening > 0.03:
                gripper_opened = True

            if done[0]:
                break

        placed = block_target_dist < 0.05
        if placed:
            placed_count += 1

        print(f"Ep {ep:2d}: final_dist={block_target_dist*100:5.1f}cm  "
              f"min_dist={min_dist*100:5.1f}cm  "
              f"block_h={block_h*100:5.1f}cm  "
              f"gripper_opened={'Y' if gripper_opened else 'N'}  "
              f"place={'Y' if placed else 'N'}")

    print("=" * 60)
    print(f"Place rate: {placed_count}/{N_EPISODES} ({100*placed_count/N_EPISODES:.0f}%)")
    print(f"Mean min dist: {np.mean([min_dist]):.1f}cm")  # This is wrong, but just for summary

if __name__ == "__main__":
    main()
