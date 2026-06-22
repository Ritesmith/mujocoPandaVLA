#!/usr/bin/env python3
"""Test Gymnasium + SmolVLA closed loop."""
import os
os.environ.pop('PYTHONPATH', None)
os.environ['MUJOCO_GL'] = 'egl'

import gymnasium
import sys
sys.path.insert(0, '/home/w/vla_workspace')
import gym_env

# Test with VLA enabled - use unwrapped env for VLA methods
env = gymnasium.make('PandaVLA-v0', vla_enabled=True,
                     vla_model_path='/home/w/vla_workspace/models/smolvla_base',
                     task_instruction='pick up the red block')
obs, info = env.reset()
print(f"Reset OK, observation keys: {list(obs.keys())}")
print(f"Image shape: {obs['image'].shape}")
print(f"Joint positions: {obs['joint_positions']}")
print(f"Gripper: {obs['gripper']}")

# Access unwrapped env for VLA-specific methods
raw_env = env.unwrapped

# Run 5 VLA steps
for i in range(5):
    obs, reward, terminated, truncated, info = raw_env.vla_step()
    print(f"Step {i+1}: reward={reward:.4f}, terminated={terminated}, "
          f"joint_pos={obs['joint_positions'][:3]}")
    if terminated or truncated:
        print("Episode ended, resetting...")
        obs, info = env.reset()

print("VLA closed-loop test: PASSED")
env.close()
