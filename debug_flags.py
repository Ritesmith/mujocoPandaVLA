#!/usr/bin/env python3
"""Debug: check place_mode flags in env."""
import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.insert(0, "/home/w/vla_workspace")

import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs

env = gymnasium.make(
    "PandaVLA-v0",
    reward_type="place_only",
    place_mode_realistic=True,
    gravity_comp=True,
    target_pos=np.array([0.5, 0.3, 0.2]),
)
env = FlattenObs(env)
raw_env = env.unwrapped

print(f"place_mode: {raw_env.place_mode}")
print(f"place_mode_realistic: {raw_env.place_mode_realistic}")
print(f"_place_gravcomp_active (before reset): {raw_env._place_gravcomp_active}")

obs, info = env.reset()

print(f"\nAfter reset:")
print(f"_place_gravcomp_active: {raw_env._place_gravcomp_active}")
print(f"block pos: {info.get('block_position', 'N/A')}")
print(f"block height: {info.get('block_height', 'N/A')}")

# Check pad positions after reset
import mujoco
if raw_env._lf_pad1_gid >= 0:
    lf_pad_pos = raw_env.data.geom_xpos[raw_env._lf_pad1_gid].copy()
    rf_pad_pos = raw_env.data.geom_xpos[raw_env._rf_pad1_gid].copy()
    pad_center = (lf_pad_pos + rf_pad_pos) / 2
    print(f"lf_pad1: ({lf_pad_pos[0]:.4f}, {lf_pad_pos[1]:.4f}, {lf_pad_pos[2]:.4f})")
    print(f"rf_pad1: ({rf_pad_pos[0]:.4f}, {rf_pad_pos[1]:.4f}, {rf_pad_pos[2]:.4f})")
    print(f"pad_center: ({pad_center[0]:.4f}, {pad_center[1]:.4f}, {pad_center[2]:.4f})")

# Check arm qpos
print(f"arm qpos: {raw_env.data.qpos[raw_env._arm_qpos_adrs]}")
print(f"finger qpos: {raw_env.data.qpos[raw_env._finger_qpos_adrs]}")

env.close()
