#!/usr/bin/env python3
"""Test grasping a block on a table using the modified Panda MuJoCo model.

Uses Jacobian-based IK to position the hand, then closes the gripper
and attempts to lift the block. Monitors block position, contact forces,
and lift success throughout the sequence.

Usage:
    cd /home/w/vla_workspace
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla
    python test_grasp_table.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from mujoco import MjModel, MjData

SCENE_PATH = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"


def ik_step(model, data, target_pos, body_name="hand", step_size=0.3, damping=0.01):
    """Perform one Jacobian-based IK step targeting a body to a position."""
    body_id = model.body(body_name).id
    mujoco.mj_forward(model, data)
    current_pos = data.body(body_id).xpos.copy()
    error = target_pos - current_pos

    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_joint_ids = [model.joint(jn).id for jn in arm_joint_names]
    arm_qpos_adrs = [model.jnt_dofadr[jid] for jid in arm_joint_ids]
    arm_dof_adrs = [model.jnt_dofadr[jid] for jid in arm_joint_ids]
    arm_ranges = np.array([[model.jnt_range[jid][0], model.jnt_range[jid][1]] for jid in arm_joint_ids])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, target_pos, body_id)

    J = jacp[:, arm_dof_adrs]
    JJT = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(JJT, error)

    current_qpos = data.qpos[arm_qpos_adrs].copy()
    new_qpos = current_qpos + step_size * dq
    new_qpos = np.clip(new_qpos, arm_ranges[:, 0], arm_ranges[:, 1])
    data.qpos[arm_qpos_adrs] = new_qpos
    return np.linalg.norm(error)


def get_arm_qpos_adrs(model):
    """Get arm joint qpos addresses."""
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_joint_ids = [model.joint(jn).id for jn in arm_joint_names]
    return [model.jnt_dofadr[jid] for jid in arm_joint_ids]


def get_contact_info(model, data):
    """Get detailed contact information for all active contacts."""
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = model.geom(c.geom1).name or f"geom_{c.geom1}"
        g2 = model.geom(c.geom2).name or f"geom_{c.geom2}"
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        normal_force = np.linalg.norm(force[:3])
        friction_force = np.linalg.norm(force[3:6])
        contacts.append({
            'geom1': g1, 'geom2': g2,
            'normal': normal_force, 'friction': friction_force,
            'force_vec': force[:3].copy(),
        })
    return contacts


def is_hand_block_contact(contact):
    """Check if a contact involves hand/finger geoms and the block."""
    hand_names = {
        'hand_c', 'finger_0',
        'fingertip_pad_collision_1', 'fingertip_pad_collision_2',
        'fingertip_pad_collision_3', 'fingertip_pad_collision_4',
        'fingertip_pad_collision_5',
    }
    block_names = {'red_block_geom'}
    g1, g2 = contact['geom1'], contact['geom2']
    return (g1 in hand_names and g2 in block_names) or \
           (g2 in hand_names and g1 in block_names)


def compute_finger_offset(model, data):
    """Compute the offset from hand body to the midpoint between fingertips."""
    hand_pos = data.body('hand').xpos.copy()
    lfinger_pos = data.body('left_finger').xpos.copy()
    rfinger_pos = data.body('right_finger').xpos.copy()
    finger_mid = (lfinger_pos + rfinger_pos) / 2
    return finger_mid - hand_pos


def ik_with_stepping(model, data, target_pos, arm_qpos_adrs,
                     max_iter=50, step_size=0.15, damping=0.05,
                     settle_steps=20, gripper_ctrl=0.04,
                     block_pos_init=None, block_disp_limit=0.02):
    """Run IK with simulation stepping for smooth, collision-aware motion."""
    block_displaced = False
    max_block_disp = 0.0
    final_err = float('inf')

    for i in range(max_iter):
        err = ik_step(model, data, target_pos, step_size=step_size, damping=damping)
        final_err = err

        data.ctrl[:7] = data.qpos[arm_qpos_adrs]
        data.ctrl[7] = gripper_ctrl
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)

        if block_pos_init is not None:
            block_pos = data.body('red_block').xpos.copy()
            block_disp = np.linalg.norm(block_pos - block_pos_init)
            max_block_disp = max(max_block_disp, block_disp)
            if block_disp > block_disp_limit:
                block_displaced = True

        if err < 0.005:
            break

    return final_err, block_displaced, max_block_disp


def main():
    print("=" * 80)
    print("Panda Grasp Table Test")
    print("=" * 80)

    print(f"\nLoading model from: {SCENE_PATH}")
    model = MjModel.from_xml_path(SCENE_PATH)
    data = MjData(model)

    print(f"Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, ngeom={model.ngeom}")

    arm_qpos_adrs = get_arm_qpos_adrs(model)

    # ========================================================================
    # Phase 0: Reset to home and settle
    # ========================================================================
    print("\n--- Phase 0: Reset to home position ---")
    mujoco.mj_resetData(model, data)

    home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04,
                          0.5, 0.0, 0.24, 1, 0, 0, 0])
    data.qpos[:] = home_qpos
    data.ctrl[:7] = home_qpos[:7]
    data.ctrl[7] = 0.04

    for _ in range(1000):
        mujoco.mj_step(model, data)

    block_pos_init = data.body('red_block').xpos.copy()
    hand_pos = data.body('hand').xpos.copy()
    finger_offset = compute_finger_offset(model, data)
    print(f"  Block position (settled): [{block_pos_init[0]:.4f}, {block_pos_init[1]:.4f}, {block_pos_init[2]:.4f}]")
    print(f"  Hand position (home):     [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Finger offset from hand:  [{finger_offset[0]:.4f}, {finger_offset[1]:.4f}, {finger_offset[2]:.4f}]")

    # ========================================================================
    # Phase 1: IK to 15cm above block center [0.5, 0.0, 0.39]
    # ========================================================================
    target_above = np.array([0.5, 0.0, 0.39])
    print(f"\n--- Phase 1: IK to 15cm above block center [{target_above[0]}, {target_above[1]}, {target_above[2]}] ---")

    # Use a "vertical reach" seed that keeps arm links high
    seed_qpos = np.array([0, -0.3, 0, -2.356, 0, 2.0, 0.785])
    data.qpos[arm_qpos_adrs] = seed_qpos
    data.ctrl[:7] = seed_qpos
    data.ctrl[7] = 0.04
    mujoco.mj_forward(model, data)

    err, displaced, max_disp = ik_with_stepping(
        model, data, target_above, arm_qpos_adrs,
        max_iter=100, step_size=0.15, damping=0.05,
        settle_steps=30, gripper_ctrl=0.04,
        block_pos_init=block_pos_init, block_disp_limit=0.02
    )

    for _ in range(500):
        mujoco.mj_step(model, data)

    hand_pos = data.body('hand').xpos.copy()
    block_pos = data.body('red_block').xpos.copy()
    finger_offset = compute_finger_offset(model, data)
    finger_mid = hand_pos + finger_offset
    block_disp = np.linalg.norm(block_pos - block_pos_init)
    print(f"  Hand position:  [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Finger midpoint:[{finger_mid[0]:.4f}, {finger_mid[1]:.4f}, {finger_mid[2]:.4f}]")
    print(f"  Block position: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
    print(f"  IK error: {err:.4f}m, Block displacement: {block_disp:.4f}m")

    block_pos_init = data.body('red_block').xpos.copy()

    # ========================================================================
    # Phase 2: IK to 2cm above block top [0.5, 0.0, 0.28]
    #   Adjust hand target to compensate for finger offset so fingertips
    #   end up at the block. Descend gradually with block monitoring.
    # ========================================================================
    # Finger offset is primarily in -z (~5.8cm below hand). When hand is at
    # z=0.28, fingers are at z≈0.222 (block bottom level). The x/y offsets
    # are small (~1cm). We adjust the hand target to place fingers at block.
    adjusted_x = 0.5 - finger_offset[0]
    adjusted_y = 0.0 - finger_offset[1]
    target_grasp_hand = np.array([adjusted_x, adjusted_y, 0.28])

    print(f"\n--- Phase 2: IK to 2cm above block top [{target_grasp_hand[0]:.4f}, {target_grasp_hand[1]:.4f}, {target_grasp_hand[2]:.4f}] ---")
    print(f"  (Adjusted from [0.5, 0.0, 0.28] to place fingertips at block)")

    # Gradual descent with IK stepping
    current_hand_z = hand_pos[2]
    target_z = target_grasp_hand[2]
    z_step = 0.005  # 5mm per waypoint for very careful descent
    block_displaced = False
    max_block_disp_overall = 0.0
    waypoint = 0
    displacement_waypoint = -1

    while current_hand_z > target_z + 0.002:
        waypoint += 1
        next_z = max(current_hand_z - z_step, target_z)
        waypoint_target = np.array([target_grasp_hand[0], target_grasp_hand[1], next_z])

        err, wp_displaced, wp_max_disp = ik_with_stepping(
            model, data, waypoint_target, arm_qpos_adrs,
            max_iter=20, step_size=0.05, damping=0.05,
            settle_steps=10, gripper_ctrl=0.04,
            block_pos_init=block_pos_init, block_disp_limit=0.02
        )

        current_hand_z = data.body('hand').xpos[2]
        max_block_disp_overall = max(max_block_disp_overall, wp_max_disp)

        if wp_displaced and not block_displaced:
            block_displaced = True
            displacement_waypoint = waypoint

        if waypoint % 10 == 0 or wp_displaced or next_z <= target_z + 0.002:
            block_pos = data.body('red_block').xpos.copy()
            block_disp = np.linalg.norm(block_pos - block_pos_init)
            hand_pos_now = data.body('hand').xpos.copy()
            finger_mid_now = hand_pos_now + compute_finger_offset(model, data)
            print(f"  WP {waypoint:3d}: target_z={next_z:.3f} hand_z={current_hand_z:.4f} "
                  f"finger_z={finger_mid_now[2]:.3f} block_disp={block_disp:.4f}m displaced={wp_displaced}")

        if block_displaced:
            print(f"  WARNING: Block displaced >2cm at waypoint {waypoint}!")
            # Don't break - continue to see how far we can get
            # but stop if displacement is very large
            block_pos = data.body('red_block').xpos.copy()
            if np.linalg.norm(block_pos - block_pos_init) > 0.05:
                print(f"  Block displaced >5cm, stopping descent.")
                break

    # Additional settling
    for _ in range(300):
        data.ctrl[:7] = data.qpos[arm_qpos_adrs]
        data.ctrl[7] = 0.04
        mujoco.mj_step(model, data)

    hand_pos = data.body('hand').xpos.copy()
    block_pos = data.body('red_block').xpos.copy()
    block_disp = np.linalg.norm(block_pos - block_pos_init)
    finger1_pos = data.body('left_finger').xpos.copy()
    finger2_pos = data.body('right_finger').xpos.copy()
    finger_mid = (finger1_pos + finger2_pos) / 2
    print(f"\n  Final positions after Phase 2:")
    print(f"  Hand position:  [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Lfinger:        [{finger1_pos[0]:.4f}, {finger1_pos[1]:.4f}, {finger1_pos[2]:.4f}]")
    print(f"  Rfinger:        [{finger2_pos[0]:.4f}, {finger2_pos[1]:.4f}, {finger2_pos[2]:.4f}]")
    print(f"  Finger midpoint:[{finger_mid[0]:.4f}, {finger_mid[1]:.4f}, {finger_mid[2]:.4f}]")
    print(f"  Block position: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
    print(f"  Finger-to-block dist: {np.linalg.norm(finger_mid - block_pos):.4f}m")
    print(f"  Block displacement: {block_disp:.4f}m (max: {max_block_disp_overall:.4f}m)")
    if block_displaced:
        print(f"  Block displaced >2cm at waypoint {displacement_waypoint}")

    # Check contacts before closing gripper
    contacts = get_contact_info(model, data)
    hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
    print(f"  Contacts: total={len(contacts)}, hand-block={len(hb_contacts)}")
    for c in hb_contacts:
        print(f"    {c['geom1']} <-> {c['geom2']} normal={c['normal']:.4f} friction={c['friction']:.4f}")

    # ========================================================================
    # Phase 3: Close gripper (ctrl from 0.04 to 0.0 over 300 steps)
    # ========================================================================
    print("\n--- Phase 3: Closing gripper ---")
    max_hand_block_force = 0.0
    contact_detected_during_close = False

    for step in range(300):
        alpha = step / 299.0
        data.ctrl[7] = 0.04 * (1.0 - alpha)
        mujoco.mj_step(model, data)

        contacts = get_contact_info(model, data)
        hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
        block_z = data.body('red_block').xpos[2]

        if hb_contacts:
            contact_detected_during_close = True
            for c in hb_contacts:
                if c['normal'] > max_hand_block_force:
                    max_hand_block_force = c['normal']

        if step % 50 == 0 or step == 299:
            block_pos = data.body('red_block').xpos.copy()
            print(f"  Step {step:3d}: ctrl={data.ctrl[7]:.4f} block_z={block_z:.4f} "
                  f"hand-block contacts={len(hb_contacts)}")
            for c in hb_contacts:
                print(f"    {c['geom1']} <-> {c['geom2']} normal={c['normal']:.4f}")

    block_pos_after_close = data.body('red_block').xpos.copy()
    print(f"  Block position after close: [{block_pos_after_close[0]:.4f}, "
          f"{block_pos_after_close[1]:.4f}, {block_pos_after_close[2]:.4f}]")
    print(f"  Contact detected during close: {contact_detected_during_close}")
    print(f"  Max hand-block normal force: {max_hand_block_force:.4f}N")

    # ========================================================================
    # Phase 4: Lift hand 15cm upward using IK
    # ========================================================================
    hand_pos_current = data.body('hand').xpos.copy()
    target_lift = hand_pos_current.copy()
    target_lift[2] += 0.15
    print(f"\n--- Phase 4: Lift hand 15cm upward ---")
    print(f"  Current hand: [{hand_pos_current[0]:.4f}, {hand_pos_current[1]:.4f}, {hand_pos_current[2]:.4f}]")
    print(f"  Target hand:  [{target_lift[0]:.4f}, {target_lift[1]:.4f}, {target_lift[2]:.4f}]")

    err, _, _ = ik_with_stepping(
        model, data, target_lift, arm_qpos_adrs,
        max_iter=60, step_size=0.2, damping=0.01,
        settle_steps=20, gripper_ctrl=0.0,
        block_pos_init=None, block_disp_limit=999
    )

    block_lifted = False
    max_lift_force = 0.0
    block_z_during_lift = []

    for step in range(500):
        data.ctrl[:7] = data.qpos[arm_qpos_adrs]
        data.ctrl[7] = 0.0
        mujoco.mj_step(model, data)

        block_z = data.body('red_block').xpos[2]
        block_z_during_lift.append(block_z)
        hand_z = data.body('hand').xpos[2]

        contacts = get_contact_info(model, data)
        hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
        for c in hb_contacts:
            if c['normal'] > max_lift_force:
                max_lift_force = c['normal']

        if block_z > 0.27:
            block_lifted = True

        if step % 100 == 0 or step == 499:
            block_pos = data.body('red_block').xpos.copy()
            print(f"  Step {step:3d}: hand_z={hand_z:.4f} block_z={block_z:.4f} "
                  f"hand-block contacts={len(hb_contacts)} lifted={block_lifted}")
            for c in hb_contacts:
                print(f"    {c['geom1']} <-> {c['geom2']} normal={c['normal']:.4f}")

    # ========================================================================
    # Final Results
    # ========================================================================
    block_pos_final = data.body('red_block').xpos.copy()
    hand_pos_final = data.body('hand').xpos.copy()

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"  Block initial position:  [{block_pos_init[0]:.4f}, {block_pos_init[1]:.4f}, {block_pos_init[2]:.4f}]")
    print(f"  Block final position:    [{block_pos_final[0]:.4f}, {block_pos_final[1]:.4f}, {block_pos_final[2]:.4f}]")
    print(f"  Block z change:          {block_pos_final[2] - block_pos_init[2]:.4f}m")
    print(f"  Hand final position:     [{hand_pos_final[0]:.4f}, {hand_pos_final[1]:.4f}, {hand_pos_final[2]:.4f}]")
    print(f"  Block displaced >2cm during approach: {block_displaced}")
    if block_displaced:
        print(f"  Displacement first detected at waypoint: {displacement_waypoint}")
        print(f"  Max block displacement during approach: {max_block_disp_overall:.4f}m")
    print(f"  Contact detected during gripper close: {contact_detected_during_close}")
    print(f"  Max hand-block force (close): {max_hand_block_force:.4f}N")
    print(f"  Max hand-block force (lift):  {max_lift_force:.4f}N")
    print(f"  Block lifted (z > 0.27): {block_lifted}")
    if block_z_during_lift:
        print(f"  Block z range during lift: [{min(block_z_during_lift):.4f}, {max(block_z_during_lift):.4f}]")

    contacts = get_contact_info(model, data)
    hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
    print(f"  Final hand-block contacts: {len(hb_contacts)}")
    for c in hb_contacts:
        print(f"    {c['geom1']} <-> {c['geom2']} normal={c['normal']:.4f} friction={c['friction']:.4f}")

    print("\n--- VERDICT ---")
    if block_lifted:
        print("  SUCCESS: Block was lifted (z > 0.27)")
    elif contact_detected_during_close and not block_lifted:
        print("  PARTIAL: Contact was detected during grasp, but block was NOT lifted.")
        print("  Possible causes: insufficient friction, block slipped, or grip force too low.")
    else:
        print("  FAILURE: No contact detected or block not lifted.")
        if block_displaced:
            print("  Root cause: Arm links (link5/link6) collide with the block during")
            print("  the descent phase, pushing it away before the fingers can engage.")
            print("  The block at x=0.5 is in the path of the arm links when the hand")
            print("  descends from above. A motion planner or different approach angle")
            print("  would be needed to avoid this collision.")
        else:
            print("  Possible causes: hand not positioned correctly, fingers not reaching block.")


if __name__ == "__main__":
    main()
