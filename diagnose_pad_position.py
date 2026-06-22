#!/usr/bin/env python3
"""Diagnose finger and pad positions correctly using geom lookup."""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
from mujoco import MjModel, MjData

SCENE_XML = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"

model = MjModel.from_xml_path(SCENE_XML)
data = MjData(model)

# Set lifted arm configuration
arm_qpos_adrs = []
for j in range(1, 8):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}")
    arm_qpos_adrs.append(model.jnt_qposadr[jid])

lifted_qpos = np.array([0.5, 0.3, 0.0, -1.57079, 0.0, 1.57079, 0.7854])
data.qpos[arm_qpos_adrs] = lifted_qpos
data.ctrl[:7] = lifted_qpos

# Close gripper
finger_qpos_adrs = []
for name in ["finger_joint1", "finger_joint2"]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    finger_qpos_adrs.append(model.jnt_qposadr[jid])
data.qpos[finger_qpos_adrs] = [0.02, 0.02]
gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint1")
data.ctrl[gripper_act_id] = 0.02

mujoco.mj_forward(model, data)

# Get body positions
bodies = ["hand", "left_finger", "right_finger"]
print("Body positions (x, y, z):")
for name in bodies:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid >= 0:
        pos = data.xpos[bid]
        print(f"  {name:15s}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

# Get geom positions for pads
print("\nGeom positions (x, y, z):")
for name in ["lf_pad1", "rf_pad1", "lf_pad2", "rf_pad2", "lf_pad3", "rf_pad3"]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid >= 0:
        pos = data.geom_xpos[gid]
        size = model.geom_size[gid]
        print(f"  {name:15s}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) size=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")

# Calculate finger center
lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
lf_pos = data.xpos[lf_id]
rf_pos = data.xpos[rf_id]
finger_center = (lf_pos + rf_pos) / 2
print(f"\nFinger center: ({finger_center[0]:.4f}, {finger_center[1]:.4f}, {finger_center[2]:.4f})")

# Calculate pad center (using pad1 geoms)
lf_pad1_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lf_pad1")
rf_pad1_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_pad1")
lf_pad1_pos = data.geom_xpos[lf_pad1_gid]
rf_pad1_pos = data.geom_xpos[rf_pad1_gid]
pad_center = (lf_pad1_pos + rf_pad1_pos) / 2
print(f"Pad1 center: ({pad_center[0]:.4f}, {pad_center[1]:.4f}, {pad_center[2]:.4f})")
print(f"Pad1 Y gap: {abs(lf_pad1_pos[1] - rf_pad1_pos[1]):.4f}m")

# Place block at pad center and test
block_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
block_qpos_adr = model.jnt_qposadr[block_joint_id]
block_dof_adr = model.jnt_dofadr[block_joint_id]
block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")

print(f"\n--- Test: block at pad center ---")
data.qpos[block_qpos_adr + 0] = pad_center[0]
data.qpos[block_qpos_adr + 1] = pad_center[1]
data.qpos[block_qpos_adr + 2] = pad_center[2]
data.qpos[block_qpos_adr + 3] = 1.0
data.qpos[block_qpos_adr + 4:7] = 0.0
data.qvel[block_dof_adr:block_dof_adr + 6] = 0

mujoco.mj_forward(model, data)
initial_z = data.xpos[block_id][2]
print(f"Initial block z: {initial_z:.4f}m")

# Run 200 steps with gravity comp for arm
for step in range(200):
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:7] = data.qfrc_bias[:7]
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0

final_z = data.xpos[block_id][2]
print(f"After 200 steps: block z={final_z:.4f}m (delta={final_z - initial_z:.4f}m)")
print(f"Block {'STAYED' if abs(final_z - initial_z) < 0.01 else 'FELL'}")

# Try with tighter gripper (0.01)
print(f"\n--- Test: block at pad center, gripper=0.01 ---")
data.qpos[finger_qpos_adrs] = [0.01, 0.01]
data.ctrl[gripper_act_id] = 0.01
mujoco.mj_forward(model, data)

# Recalculate pad center with tighter gripper
lf_pad1_pos = data.geom_xpos[lf_pad1_gid]
rf_pad1_pos = data.geom_xpos[rf_pad1_gid]
pad_center = (lf_pad1_pos + rf_pad1_pos) / 2
print(f"Pad1 center (gripper=0.01): ({pad_center[0]:.4f}, {pad_center[1]:.4f}, {pad_center[2]:.4f})")
print(f"Pad1 Y gap: {abs(lf_pad1_pos[1] - rf_pad1_pos[1]):.4f}m")

data.qpos[block_qpos_adr + 0] = pad_center[0]
data.qpos[block_qpos_adr + 1] = pad_center[1]
data.qpos[block_qpos_adr + 2] = pad_center[2]
data.qpos[block_qpos_adr + 3] = 1.0
data.qpos[block_qpos_adr + 4:7] = 0.0
data.qvel[block_dof_adr:block_dof_adr + 6] = 0

mujoco.mj_forward(model, data)
initial_z = data.xpos[block_id][2]
print(f"Initial block z: {initial_z:.4f}m")

for step in range(200):
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:7] = data.qfrc_bias[:7]
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0

final_z = data.xpos[block_id][2]
print(f"After 200 steps: block z={final_z:.4f}m (delta={final_z - initial_z:.4f}m)")
print(f"Block {'STAYED' if abs(final_z - initial_z) < 0.01 else 'FELL'}")
