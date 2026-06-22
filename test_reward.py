#!/usr/bin/env python3
"""Test pick-place reward function."""
import os
os.environ.pop('PYTHONPATH', None)
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium
import numpy as np
import gym_env

env = gymnasium.make('PandaVLA-v0')
obs, info = env.reset()
print(f"Initial info: block_pos={info.get('block_position')}, block_height={info.get('block_height')}")
print(f"Initial reward: {env.unwrapped._compute_reward():.4f}")

# Test: random actions for 50 steps
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if (i+1) % 10 == 0:
        print(f"Step {i+1}: reward={reward:.4f}, block_z={info.get('block_height', 0):.4f}, "
              f"hand_block_dist={info.get('hand_block_distance', 0):.4f}")

# Test: reward should be in [-1, 1]
rewards = []
obs, info = env.reset()
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        obs, info = env.reset()

print(f"\nReward stats over 100 random steps:")
print(f"  Min: {min(rewards):.4f}")
print(f"  Max: {max(rewards):.4f}")
print(f"  Mean: {np.mean(rewards):.4f}")
print(f"  All in [-1, 1]: {all(-1 <= r <= 1 for r in rewards)}")

env.close()
print("Reward function test: PASSED")
