#!/usr/bin/env python3
"""Debug place_mode_realistic in the actual env."""
import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.insert(0, "/home/w/vla_workspace")

import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs

TABLE_Z = 0.22

env = gymnasium.make(
    "PandaVLA-v0",
    reward_type="place_only",
    place_mode_realistic=True,
    gravity_comp=True,
    target_pos=np.array([0.5, 0.3, 0.2]),
)
env = FlattenObs(env)

# Access raw env to check internal state
raw_env = env.unwrapped

obs, info = env.reset()

# Check pad positions after reset
import mujoco
lf_pad_gid = raw_env._lf_pad1_gid
rf_pad_gid = raw_env._rf_pad1_gid
lf_pad_pos = raw_env.data.geom_xpos[lf_pad_gid].copy()
rf_pad_pos = raw_env.data.geom_xpos[rf_pad_gid].copy()
pad_center = (lf_pad_pos + rf_pad_pos) / 2

print(f"After reset:")
print(f"  lf_pad1: ({lf_pad_pos[0]:.4f}, {lf_pad_pos[1]:.4f}, {lf_pad_pos[2]:.4f})")
print(f"  rf_pad1: ({rf_pad_pos[0]:.4f}, {rf_pad_pos[1]:.4f}, {rf_pad_pos[2]:.4f})")
print(f"  pad_center: ({pad_center[0]:.4f}, {pad_center[1]:.4f}, {pad_center[2]:.4f})")
print(f"  pad Y gap: {abs(lf_pad_pos[1] - rf_pad_pos[1]):.4f}m")
print(f"  block pos: ({info['block_position'][0]:.4f}, {info['block_position'][1]:.4f}, {info['block_position'][2]:.4f})")
print(f"  block height: {info['block_height']:.4f}m")
print(f"  gripper opening: {info['gripper_opening']:.4f}")

# Check finger positions
lf_pos = raw_env.data.xpos[raw_env._left_finger_id].copy()
rf_pos = raw_env.data.xpos[raw_env._right_finger_id].copy()
print(f"  left_finger: ({lf_pos[0]:.4f}, {lf_pos[1]:.4f}, {lf_pos[2]:.4f})")
print(f"  right_finger: ({rf_pos[0]:.4f}, {rf_pos[1]:.4f}, {rf_pos[2]:.4f})")
print(f"  finger Y gap: {abs(lf_pos[1] - rf_pos[1]):.4f}m")

# Run 10 steps with no action and track block
print(f"\n10 steps with zero action:")
for step in range(10):
    action = np.zeros(8, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {step}: block_z={info['block_height']:.4f}m, gripper={info['gripper_opening']:.4f}")
    if terminated:
        print(f"  Terminated!")
        break

env.close()
