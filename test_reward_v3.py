#!/usr/bin/env python3
"""Test improved dense reward function."""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import gymnasium
import gym_env
from gym_env.wrappers import FlattenObs

# Test 1: Do nothing (zero action)
print("=" * 60)
print("Test 1: Zero action (do nothing)")
env = gymnasium.make('PandaVLA-v0', reward_type='dense')
obs, info = env.reset()
rewards = []
for i in range(100):
    action = np.zeros(8, dtype=np.float32)  # Do nothing
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        break
print(f"  Mean reward per step: {np.mean(rewards):.4f}")
print(f"  Min: {np.min(rewards):.4f}, Max: {np.max(rewards):.4f}")
print(f"  Total reward: {np.sum(rewards):.2f}")
print(f"  Expected: ~0.1 per step (close to 0)")

# Test 2: Move toward block
print("\nTest 2: Move toward block")
env2 = gymnasium.make('PandaVLA-v0', reward_type='dense')
obs2, info2 = env2.reset()
rewards2 = []
prev_dist = info2.get('hand_block_distance', 1.0)
for i in range(100):
    # Simple heuristic: move toward block
    hand_pos = info2.get('hand_position', np.zeros(3))
    block_pos = info2.get('block_position', np.zeros(3))
    direction = block_pos - hand_pos
    # Map to action space (rough heuristic)
    action = np.zeros(8, dtype=np.float32)
    action[:3] = np.clip(direction[:3] * 2.0, -1.0, 1.0)  # Move arm
    action[7] = -1.0  # Open gripper

    obs2, reward, terminated, truncated, info2 = env2.step(action)
    rewards2.append(reward)
    curr_dist = info2.get('hand_block_distance', 1.0)
    if i < 10 or i % 20 == 0:
        print(f"  Step {i}: reward={reward:.4f}, hand_block_dist={curr_dist:.4f}, progress={prev_dist-curr_dist:.4f}")
    prev_dist = curr_dist
    if terminated or truncated:
        break

print(f"  Mean reward per step: {np.mean(rewards2):.4f}")
print(f"  Min: {np.min(rewards2):.4f}, Max: {np.max(rewards2):.4f}")
print(f"  Total reward: {np.sum(rewards2):.2f}")
print(f"  Expected: >0 per step (positive progress reward)")

# Test 3: Random action
print("\nTest 3: Random action")
env3 = gymnasium.make('PandaVLA-v0', reward_type='dense')
obs3, info3 = env3.reset()
rewards3 = []
for i in range(100):
    action = env3.action_space.sample()
    obs3, reward, terminated, truncated, info3 = env3.step(action)
    rewards3.append(reward)
    if terminated or truncated:
        break
print(f"  Mean reward per step: {np.mean(rewards3):.4f}")
print(f"  Min: {np.min(rewards3):.4f}, Max: {np.max(rewards3):.4f}")
print(f"  Total reward: {np.sum(rewards3):.2f}")

env.close()
env2.close()
env3.close()
print("\nDone!")
