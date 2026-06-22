#!/usr/bin/env python3
"""Debug: check env internal state during place_mode_realistic reset."""
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

# Check arm qpos addresses
print(f"Arm qpos addresses: {raw_env._arm_qpos_adrs}")
print(f"Finger qpos addresses: {raw_env._finger_qpos_adrs}")
print(f"Block qpos address: {raw_env._red_block_qpos_adr}")
print(f"lf_pad1 gid: {raw_env._lf_pad1_gid}")
print(f"rf_pad1 gid: {raw_env._rf_pad1_gid}")

# Manually replicate reset logic to debug
import mujoco
mujoco.mj_resetData(raw_env.model, raw_env.data)

# Home keyframe
home_key_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_KEY, "home")
print(f"\nHome key id: {home_key_id}")
if home_key_id >= 0:
    raw_env.data.qpos[:] = raw_env.model.key_qpos[home_key_id]
    raw_env.data.ctrl[:] = raw_env.model.key_ctrl[home_key_id]
    print(f"Home qpos[:9]: {raw_env.data.qpos[:9]}")
    print(f"Home ctrl[:8]: {raw_env.data.ctrl[:8]}")

# Set block on table
if raw_env._red_block_qpos_adr is not None:
    raw_env.data.qpos[raw_env._red_block_qpos_adr + 0] = 0.5
    raw_env.data.qpos[raw_env._red_block_qpos_adr + 1] = 0.0
    raw_env.data.qpos[raw_env._red_block_qpos_adr + 2] = 0.24
    raw_env.data.qpos[raw_env._red_block_qpos_adr + 3] = 1.0
    raw_env.data.qpos[raw_env._red_block_qpos_adr + 4:7] = 0.0

# Set lifted arm
lifted_qpos = np.array([0.5, 0.3, 0.0, -1.57079, 0.0, 1.57079, 0.7854])
raw_env.data.qpos[raw_env._arm_qpos_adrs] = lifted_qpos
raw_env.data.ctrl[raw_env._arm_actuator_ids] = lifted_qpos
print(f"\nAfter lifted_qpos:")
print(f"  qpos[arm]: {raw_env.data.qpos[raw_env._arm_qpos_adrs]}")
print(f"  ctrl[arm]: {raw_env.data.ctrl[raw_env._arm_actuator_ids]}")

# Set gripper
grasp_finger_pos = 0.02
raw_env.data.qpos[raw_env._finger_qpos_adrs] = [grasp_finger_pos, grasp_finger_pos]
raw_env.data.ctrl[raw_env._gripper_actuator_id] = grasp_finger_pos

mujoco.mj_forward(raw_env.model, raw_env.data)

# Check positions
hand_pos = raw_env.data.xpos[raw_env._hand_id].copy()
print(f"\nAfter mj_forward:")
print(f"  hand pos: ({hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f})")

if raw_env._lf_pad1_gid >= 0:
    lf_pad_pos = raw_env.data.geom_xpos[raw_env._lf_pad1_gid].copy()
    rf_pad_pos = raw_env.data.geom_xpos[raw_env._rf_pad1_gid].copy()
    pad_center = (lf_pad_pos + rf_pad_pos) / 2
    print(f"  lf_pad1: ({lf_pad_pos[0]:.4f}, {lf_pad_pos[1]:.4f}, {lf_pad_pos[2]:.4f})")
    print(f"  rf_pad1: ({rf_pad_pos[0]:.4f}, {rf_pad_pos[1]:.4f}, {rf_pad_pos[2]:.4f})")
    print(f"  pad_center: ({pad_center[0]:.4f}, {pad_center[1]:.4f}, {pad_center[2]:.4f})")
    print(f"  pad Y gap: {abs(lf_pad_pos[1] - rf_pad_pos[1]):.4f}m")

# Check finger positions
lf_pos = raw_env.data.xpos[raw_env._left_finger_id].copy()
rf_pos = raw_env.data.xpos[raw_env._right_finger_id].copy()
print(f"  left_finger: ({lf_pos[0]:.4f}, {lf_pos[1]:.4f}, {lf_pos[2]:.4f})")
print(f"  right_finger: ({rf_pos[0]:.4f}, {rf_pos[1]:.4f}, {rf_pos[2]:.4f})")

env.close()
