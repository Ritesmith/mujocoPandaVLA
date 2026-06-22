#!/usr/bin/env python3
"""Test block holding with gripper=0.025 in raw MuJoCo (no env wrapper)."""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
from mujoco import MjModel, MjData

SCENE_XML = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"

model = MjModel.from_xml_path(SCENE_XML)
data = MjData(model)

# Lookups
arm_qpos_adrs = []
for j in range(1, 8):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}")
    arm_qpos_adrs.append(model.jnt_qposadr[jid])

finger_qpos_adrs = []
for name in ["finger_joint1", "finger_joint2"]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    finger_qpos_adrs.append(model.jnt_qposadr[jid])

gripper_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint1")
block_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
block_qpos_adr = model.jnt_qposadr[block_joint_id]
block_dof_adr = model.jnt_dofadr[block_joint_id]
block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
lf_pad1_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "lf_pad1")
rf_pad1_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rf_pad1")

# Reset
mujoco.mj_resetData(model, data)

# Set lifted arm
lifted_qpos = np.array([0.5, 0.3, 0.0, -1.57079, 0.0, 1.57079, 0.7854])
data.qpos[arm_qpos_adrs] = lifted_qpos
data.ctrl[:7] = lifted_qpos

# Set gripper to 0.025
data.qpos[finger_qpos_adrs] = [0.025, 0.025]
data.ctrl[gripper_act_id] = 0.025

mujoco.mj_forward(model, data)

# Get pad positions
lf_pad_pos = data.geom_xpos[lf_pad1_gid].copy()
rf_pad_pos = data.geom_xpos[rf_pad1_gid].copy()
pad_center = (lf_pad_pos + rf_pad_pos) / 2
pad_y_gap = abs(lf_pad_pos[1] - rf_pad_pos[1])
pad_inner_gap = pad_y_gap - 2 * 0.015  # pad Y half-width = 0.015

print(f"Gripper=0.025:")
print(f"  lf_pad1: ({lf_pad_pos[0]:.4f}, {lf_pad_pos[1]:.4f}, {lf_pad_pos[2]:.4f})")
print(f"  rf_pad1: ({rf_pad_pos[0]:.4f}, {rf_pad_pos[1]:.4f}, {rf_pad_pos[2]:.4f})")
print(f"  pad_center: ({pad_center[0]:.4f}, {pad_center[1]:.4f}, {pad_center[2]:.4f})")
print(f"  pad Y gap: {pad_y_gap*1000:.1f}mm")
print(f"  pad inner gap: {pad_inner_gap*1000:.1f}mm (block=40mm)")

# Place block at pad center
data.qpos[block_qpos_adr + 0] = pad_center[0]
data.qpos[block_qpos_adr + 1] = pad_center[1]
data.qpos[block_qpos_adr + 2] = pad_center[2]
data.qpos[block_qpos_adr + 3] = 1.0
data.qpos[block_qpos_adr + 4:7] = 0.0
data.qvel[block_dof_adr:block_dof_adr + 6] = 0

mujoco.mj_forward(model, data)
initial_z = data.xpos[block_id][2]
print(f"\nBlock placed at z={initial_z:.4f}m")

# Run 200 steps with gravity comp and arm fixed
print("\n200 steps (arm fixed, gravity comp):")
for step in range(200):
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:7] = data.qfrc_bias[:7]
    # Fix arm position
    data.qpos[arm_qpos_adrs] = lifted_qpos
    data.qvel[:7] = 0
    # Keep gripper closed
    data.ctrl[gripper_act_id] = 0.025
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0

    if step % 50 == 0 or step == 199:
        bz = data.xpos[block_id][2]
        fp = data.qpos[finger_qpos_adrs].mean()
        print(f"  Step {step:3d}: block_z={bz:.4f}m, finger={fp:.4f}")

final_z = data.xpos[block_id][2]
print(f"\nFinal block z: {final_z:.4f}m (delta={final_z - initial_z:.4f}m)")
print(f"Block {'STAYED' if abs(final_z - initial_z) < 0.01 else 'FELL'}")

# Now test WITHOUT fixing arm position
print("\n\n--- Test WITHOUT arm fixing ---")
mujoco.mj_resetData(model, data)
data.qpos[arm_qpos_adrs] = lifted_qpos
data.ctrl[:7] = lifted_qpos
data.qpos[finger_qpos_adrs] = [0.025, 0.025]
data.ctrl[gripper_act_id] = 0.025
mujoco.mj_forward(model, data)

lf_pad_pos = data.geom_xpos[lf_pad1_gid].copy()
rf_pad_pos = data.geom_xpos[rf_pad1_gid].copy()
pad_center = (lf_pad_pos + rf_pad_pos) / 2

data.qpos[block_qpos_adr + 0] = pad_center[0]
data.qpos[block_qpos_adr + 1] = pad_center[1]
data.qpos[block_qpos_adr + 2] = pad_center[2]
data.qpos[block_qpos_adr + 3] = 1.0
data.qpos[block_qpos_adr + 4:7] = 0.0
data.qvel[block_dof_adr:block_dof_adr + 6] = 0

mujoco.mj_forward(model, data)
initial_z = data.xpos[block_id][2]
print(f"Block placed at z={initial_z:.4f}m")

for step in range(200):
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:7] = data.qfrc_bias[:7]
    # Do NOT fix arm position — let PD controller handle it
    data.ctrl[gripper_act_id] = 0.025
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0

    if step % 50 == 0 or step == 199:
        bz = data.xpos[block_id][2]
        aq = data.qpos[arm_qpos_adrs]
        print(f"  Step {step:3d}: block_z={bz:.4f}m, arm_q=[{aq[0]:.3f},{aq[1]:.3f},{aq[2]:.3f}]")

final_z = data.xpos[block_id][2]
print(f"\nFinal block z: {final_z:.4f}m (delta={final_z - initial_z:.4f}m)")
print(f"Block {'STAYED' if abs(final_z - initial_z) < 0.01 else 'FELL'}")
