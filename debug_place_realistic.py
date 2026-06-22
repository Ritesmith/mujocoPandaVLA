#!/usr/bin/env python3
"""Debug place_mode_realistic: check block position at each stage."""
import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.insert(0, "/home/w/vla_workspace")

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
lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

# Reset to home
mujoco.mj_resetData(model, data)
home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if home_key_id >= 0:
    data.qpos[:] = model.key_qpos[home_key_id]
    data.ctrl[:] = model.key_ctrl[home_key_id]

# Set block on table
data.qpos[block_qpos_adr + 0] = 0.5
data.qpos[block_qpos_adr + 1] = 0.0
data.qpos[block_qpos_adr + 2] = 0.24
data.qpos[block_qpos_adr + 3] = 1.0
data.qpos[block_qpos_adr + 4:7] = 0.0

# Set lifted arm
lifted_qpos = np.array([0.5, 0.3, 0.0, -1.57079, 0.0, 1.57079, 0.7854])
data.qpos[arm_qpos_adrs] = lifted_qpos
data.ctrl[:7] = lifted_qpos

# Close gripper to 0.01
data.qpos[finger_qpos_adrs] = [0.01, 0.01]
data.ctrl[gripper_act_id] = 0.01

mujoco.mj_forward(model, data)

# Get finger positions
lf_pos = data.xpos[lf_id].copy()
rf_pos = data.xpos[rf_id].copy()
finger_center = (lf_pos + rf_pos) / 2
print(f"Finger center: ({finger_center[0]:.4f}, {finger_center[1]:.4f}, {finger_center[2]:.4f})")
print(f"  lf: ({lf_pos[0]:.4f}, {lf_pos[1]:.4f}, {lf_pos[2]:.4f})")
print(f"  rf: ({rf_pos[0]:.4f}, {rf_pos[1]:.4f}, {rf_pos[2]:.4f})")
print(f"  Y gap: {abs(lf_pos[1] - rf_pos[1]):.4f}m")

# Place block at finger center
data.qpos[block_qpos_adr + 0] = finger_center[0]
data.qpos[block_qpos_adr + 1] = finger_center[1]
data.qpos[block_qpos_adr + 2] = finger_center[2]
data.qpos[block_qpos_adr + 3] = 1.0
data.qpos[block_qpos_adr + 4:7] = 0.0
data.qvel[block_dof_adr:block_dof_adr + 6] = 0

mujoco.mj_forward(model, data)
print(f"\nBlock placed at: ({data.xpos[block_id][0]:.4f}, {data.xpos[block_id][1]:.4f}, {data.xpos[block_id][2]:.4f})")

# Run settle steps and track block position
print("\nSettle steps (50):")
for step in range(50):
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:7] = data.qfrc_bias[:7]
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0

    if step % 10 == 0 or step == 49:
        bz = data.xpos[block_id][2]
        fp = data.qpos[finger_qpos_adrs].mean()
        print(f"  Step {step:3d}: block_z={bz:.4f}m, finger_pos={fp:.4f}")

# Check contact info
print(f"\nAfter settle:")
print(f"  Block pos: ({data.xpos[block_id][0]:.4f}, {data.xpos[block_id][1]:.4f}, {data.xpos[block_id][2]:.4f})")
print(f"  Finger pos: {data.qpos[finger_qpos_adrs]}")

# Check contacts
n_contacts = data.ncon
print(f"  Contacts: {n_contacts}")
for i in range(min(n_contacts, 10)):
    c = data.contact[i]
    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
    print(f"    {g1} <-> {g2}  dist={c.dist:.6f}")

# Run 100 more steps with gripper closed
print("\n100 steps with gripper closed (ctrl=0.01):")
for step in range(100):
    mujoco.mj_forward(model, data)
    data.qfrc_applied[:7] = data.qfrc_bias[:7]
    # Keep gripper closed
    data.ctrl[gripper_act_id] = 0.01
    mujoco.mj_step(model, data)
    data.qfrc_applied[:] = 0

    if step % 20 == 0 or step == 99:
        bz = data.xpos[block_id][2]
        fp = data.qpos[finger_qpos_adrs].mean()
        print(f"  Step {step:3d}: block_z={bz:.4f}m, finger_pos={fp:.4f}")

print(f"\nFinal block pos: ({data.xpos[block_id][0]:.4f}, {data.xpos[block_id][1]:.4f}, {data.xpos[block_id][2]:.4f})")
