#!/usr/bin/env python3
"""Verify collision filtering and grasp with proper orientation."""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

SCENE_XML = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"

def solve_ik(model, data, target_pos, body_name="hand", max_iter=200, tol=0.005, step_size=0.5, target_rot=None):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return False, 0

    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}") for j in range(1, 8)]

    # Fix j7=0 for proper grasp orientation (fingers open along y-axis)
    data.qpos[6] = 0.0

    # Only optimize j1-j6
    opt_joint_ids = arm_joint_ids[:6]
    opt_dof_adrs = [model.jnt_dofadr[jid] for jid in opt_joint_ids]

    for i in range(max_iter):
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[body_id].copy()
        error_pos = target_pos - current_pos

        # Orientation error
        error_rot = np.zeros(3)
        if target_rot is not None:
            current_rot = data.xmat[body_id].reshape(3, 3)
            R_err = target_rot @ current_rot.T
            angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
            if abs(angle) > 1e-6:
                axis = np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1]
                ]) / (2 * np.sin(angle))
                error_rot = axis * angle

        error = np.concatenate([error_pos, error_rot])

        if np.linalg.norm(error_pos) < tol and np.linalg.norm(error_rot) < 0.05:
            return True, i

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

        J_pos = jacp[:, opt_dof_adrs]
        J_rot = jacr[:, opt_dof_adrs]

        if target_rot is not None:
            J = np.vstack([J_pos, J_rot])
        else:
            J = J_pos

        lam = 0.1
        n = J.shape[0]
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(n), error)

        for j, jid in enumerate(opt_joint_ids):
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] += dq[j] * step_size
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)

    return False, max_iter


def test_grasp():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    # Fix gripper actuator: must update BOTH gainprm AND biasprm for position actuator
    # Force = gainprm[0]*ctrl + biasprm[1]*qpos + biasprm[2]*qvel
    # For proper position control: gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd
    # Use low kp for gentle closing, with critical damping to prevent overshoot
    new_kp = 100.0
    model.actuator_gainprm[7, 0] = new_kp
    model.actuator_biasprm[7, 1] = -new_kp
    # Critical damping: kd = 2*sqrt(kp*m), m~0.015kg → kd~2.45
    model.actuator_biasprm[7, 2] = -2.5
    # Also reduce finger joint damping for faster closing
    finger_dof_adr1 = model.jnt_dofadr[7]
    finger_dof_adr2 = model.jnt_dofadr[8]
    print(f"Finger DOF damping before: [{model.dof_damping[finger_dof_adr1]:.2f}, {model.dof_damping[finger_dof_adr2]:.2f}]")
    model.dof_damping[finger_dof_adr1] = 1.0
    model.dof_damping[finger_dof_adr2] = 1.0
    data = mujoco.MjData(model)

    # Reset
    mujoco.mj_resetData(model, data)

    # Set j1=-pi/4, j7=0 so fingers open along world y-axis
    data.qpos[:7] = [-0.7854, 0, 0, -1.57079, 0, 1.57079, 0]
    data.qpos[7:9] = [0.04, 0.04]
    data.ctrl[:7] = [-0.7854, 0, 0, -1.57079, 0, 1.57079, 0]
    data.ctrl[7] = 0.04

    # Set block on table
    block_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    block_qpos_adr = model.jnt_qposadr[block_joint_id]
    data.qpos[block_qpos_adr:block_qpos_adr+7] = [0.5, 0.0, 0.24, 1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)

    block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    lf_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

    block_pos = data.xpos[block_body_id].copy()
    print(f"Block position: {block_pos}")
    print(f"Hand position: {data.xpos[hand_body_id]}")
    print(f"LF: {data.xpos[lf_body_id]}")
    print(f"RF: {data.xpos[rf_body_id]}")
    print(f"Finger y-gap: {abs(data.xpos[lf_body_id][1] - data.xpos[rf_body_id][1])*1000:.1f}mm")

    # Compute grasp height offset:
    # hand_z - lf_z = 0.0584 (finger body below hand)
    # fingertip pad is at local z=0.0445 from finger body (further down in world -z)
    # Total: hand to pad center = 0.0584 + 0.0445 = 0.1029
    hand_z = data.xpos[hand_body_id][2]
    lf_z = data.xpos[lf_body_id][2]
    fingertip_offset = hand_z - lf_z  # 0.0584
    pad_z_local = 0.0445  # fingertip pad z offset in finger local frame
    grasp_offset = fingertip_offset + pad_z_local  # total offset from hand to pad center
    print(f"Fingertip offset (hand_z - lf_z): {fingertip_offset:.4f}")
    print(f"Pad z local offset: {pad_z_local:.4f}")
    print(f"Total grasp offset: {grasp_offset:.4f}")

    # Store target orientation from home config (hand pointing down, fingers along y)
    target_rot = data.xmat[hand_body_id].reshape(3, 3).copy()
    print(f"Target rotation:\n{target_rot}")
    print(f"Hand y-axis (finger open dir): {target_rot[:, 1]}")
    print(f"Hand z-axis (approach dir): {target_rot[:, 2]}")

    # Phase 1: Move above block (10cm above)
    above_pos = block_pos.copy()
    above_pos[2] += 0.10
    success, iters = solve_ik(model, data, above_pos, target_rot=target_rot)
    print(f"\nIK to above block: success={success}, iters={iters}")
    print(f"  Hand: {data.xpos[hand_body_id]}")
    print(f"  LF: {data.xpos[lf_body_id]}")
    print(f"  RF: {data.xpos[rf_body_id]}")
    print(f"  Finger y-gap: {abs(data.xpos[lf_body_id][1] - data.xpos[rf_body_id][1])*1000:.1f}mm")
    print(f"  LF z - RF z: {abs(data.xpos[lf_body_id][2] - data.xpos[rf_body_id][2])*1000:.1f}mm (should be small)")

    # Step simulation to settle
    data.ctrl[:7] = data.qpos[:7]
    data.ctrl[7] = 0.04
    for si in range(200):
        mujoco.mj_step(model, data)
        if si < 5 or si % 50 == 0:
            for ci in range(data.ncon):
                c = data.contact[ci]
                b1 = model.geom_bodyid[c.geom1]
                b2 = model.geom_bodyid[c.geom2]
                bn1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or 'world'
                bn2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or 'world'
                if 'block' in bn1 or 'block' in bn2:
                    other = bn2 if 'block' in bn1 else bn1
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, force)
                    print(f"  Step {si}: block contact with {other}, force={np.linalg.norm(force[:3]):.2f}N")

    block_pos_after_above = data.xpos[block_body_id].copy()
    print(f"  Block after above: {block_pos_after_above}")
    print(f"  Block displacement: {np.linalg.norm(block_pos_after_above - block_pos)*1000:.1f}mm")

    # Phase 2: Lower to block level
    # Hand target z = block_z + grasp_offset so fingertip pad is at block center height
    grasp_pos = data.xpos[block_body_id].copy()
    grasp_pos[2] += grasp_offset

    success, iters = solve_ik(model, data, grasp_pos, target_rot=target_rot)
    print(f"\nIK to grasp position: success={success}, iters={iters}")
    print(f"  Hand: {data.xpos[hand_body_id]}")
    print(f"  LF: {data.xpos[lf_body_id]}")
    print(f"  RF: {data.xpos[rf_body_id]}")
    print(f"  Finger y-gap: {abs(data.xpos[lf_body_id][1] - data.xpos[rf_body_id][1])*1000:.1f}mm")
    print(f"  LF z - RF z: {abs(data.xpos[lf_body_id][2] - data.xpos[rf_body_id][2])*1000:.1f}mm (should be small)")
    # Verify fingertip pad would be at block center height
    lf_world = data.xpos[lf_body_id]
    pad_z_world = lf_world[2] - pad_z_local
    print(f"  Fingertip pad z (world): {pad_z_world:.4f} (block center z: {data.xpos[block_body_id][2]:.4f})")

    # Check if fingers surround block
    lf_y = data.xpos[lf_body_id][1]
    rf_y = data.xpos[rf_body_id][1]
    block_y = data.xpos[block_body_id][1]
    print(f"  Block between fingers (y): {min(lf_y, rf_y) < block_y < max(lf_y, rf_y)}")

    # Step simulation
    data.ctrl[:7] = data.qpos[:7]
    data.ctrl[7] = 0.04  # Keep open
    for _ in range(200):
        mujoco.mj_step(model, data)

    block_pos_after_lower = data.xpos[block_body_id].copy()
    print(f"  Block after lower: {block_pos_after_lower}")
    print(f"  Block displacement: {np.linalg.norm(block_pos_after_lower - block_pos)*1000:.1f}mm")

    # Phase 3: Close gripper using actuator (gentle closing with low kp)
    # With kp=100 and ctrl=0.0, fingers close gently until contact stops them
    print("\n=== Closing gripper ===")
    data.ctrl[7] = 0.0  # Fully close target - contact will stop fingers at block surface
    for step in range(1000):
        mujoco.mj_step(model, data)
        if step % 100 == 0:
            finger_block = 0
            for i in range(data.ncon):
                c = data.contact[i]
                b1 = model.geom_bodyid[c.geom1]
                b2 = model.geom_bodyid[c.geom2]
                bn1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1) or 'world'
                bn2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2) or 'world'
                if ('finger' in bn1 or 'finger' in bn2) and ('block' in bn1 or 'block' in bn2):
                    finger_block += 1
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
            lf_y = data.xpos[lf_body_id][1]
            rf_y = data.xpos[rf_body_id][1]
            print(f"  Step {step}: fb={finger_block}, qpos[7:9]=[{data.qpos[7]:.4f},{data.qpos[8]:.4f}], y-gap={abs(lf_y-rf_y)*1000:.1f}mm, bz={data.xpos[block_body_id][2]:.4f}")

    # Check final state
    print(f"\nAfter gripper close:")
    print(f"  LF: {data.xpos[lf_body_id]}")
    print(f"  RF: {data.xpos[rf_body_id]}")
    print(f"  Block: {data.xpos[block_body_id]}")
    print(f"  Finger y-gap: {abs(data.xpos[lf_body_id][1] - data.xpos[rf_body_id][1])*1000:.1f}mm")

    # Phase 4: Lift
    print("\n=== Lifting ===")
    lift_pos = data.xpos[hand_body_id].copy()
    lift_pos[2] += 0.10
    success, iters = solve_ik(model, data, lift_pos, target_rot=target_rot)
    print(f"IK to lift: success={success}, iters={iters}")

    data.ctrl[:7] = data.qpos[:7]
    data.ctrl[7] = 0.018  # Keep gripper closed on block
    for _ in range(300):
        mujoco.mj_step(model, data)

    block_pos_after_lift = data.xpos[block_body_id].copy()
    print(f"Block after lift: {block_pos_after_lift}")
    print(f"Block z change: {(block_pos_after_lift[2] - block_pos[2])*1000:.1f}mm")

    if block_pos_after_lift[2] > block_pos[2] + 0.02:
        print("\n SUCCESS: Block lifted!")
    else:
        print(f"\n FAILURE: Block not lifted. z={block_pos_after_lift[2]:.4f}, init_z={block_pos[2]:.4f}")


if __name__ == "__main__":
    test_grasp()
