#!/usr/bin/env python3
"""MuJoCo contact force diagnostic for Panda gripper + red_block.

Uses Jacobian-based IK to precisely position the hand above the block,
then tests whether the gripper can make contact and lift the block.

Usage:
    cd /home/w/vla_workspace
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla
    python test_contact_force.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from mujoco import MjModel, MjData

SCENE_PATH = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"


def print_geom_info(model):
    """Print all geom names, contype, conaffinity."""
    print("\n" + "="*80)
    print("GEOM COLLISION PROPERTIES")
    print("="*80)
    print(f"{'Geom Name':<40} {'contype':>8} {'conaffinity':>12} {'group':>6}")
    print("-"*80)
    for i in range(model.ngeom):
        name = model.geom(i).name or f"(unnamed_{i})"
        contype = model.geom_contype[i]
        conaffinity = model.geom_conaffinity[i]
        group = model.geom_group[i]
        print(f"{name:<40} {contype:>8} {conaffinity:>12} {group:>6}")
    print()


def print_body_positions(data, model):
    """Print key body positions."""
    bodies_of_interest = [
        "red_block", "hand", "left_finger", "right_finger",
        "link7", "link0",
    ]
    print("\n  Body positions:")
    for bname in bodies_of_interest:
        try:
            body_id = model.body(bname).id
            pos = data.body(body_id).xpos
            print(f"    {bname:<20} pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        except KeyError:
            pass


def get_contact_info(model, data):
    """Get detailed contact information."""
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


def is_finger_block_contact(contact, model):
    """Check if a contact involves a finger and the block."""
    g1, g2 = contact['geom1'], contact['geom2']
    finger_names = {'finger_0', 'fingertip_pad_collision_1', 'fingertip_pad_collision_2',
                    'fingertip_pad_collision_3', 'fingertip_pad_collision_4',
                    'fingertip_pad_collision_5'}
    block_names = {'red_block_geom'}
    return (g1 in finger_names and g2 in block_names) or \
           (g2 in finger_names and g1 in block_names)


def is_hand_block_contact(contact, model):
    """Check if a contact involves hand/finger and the block."""
    g1, g2 = contact['geom1'], contact['geom2']
    hand_names = {'hand_c', 'finger_0', 'fingertip_pad_collision_1', 'fingertip_pad_collision_2',
                  'fingertip_pad_collision_3', 'fingertip_pad_collision_4',
                  'fingertip_pad_collision_5'}
    block_names = {'red_block_geom'}
    return (g1 in hand_names and g2 in block_names) or \
           (g2 in hand_names and g1 in block_names)


def solve_ik(model, data, target_pos, body_name="hand", max_iter=100, step_size=0.5, damping=0.01):
    """Jacobian-based IK to move a body to target_pos.

    Returns the target joint positions for the 7 arm joints.
    """
    body_id = model.body(body_name).id

    # Get arm joint indices
    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_joint_ids = [model.joint(jn).id for jn in arm_joint_names]
    arm_qpos_adrs = [model.jnt_dofadr[jid] for jid in arm_joint_ids]

    # Get joint ranges
    arm_ranges = np.array([[
        model.jnt_range[jid][0], model.jnt_range[jid][1]
    ] for jid in arm_joint_ids])

    for iteration in range(max_iter):
        mujoco.mj_forward(model, data)
        current_pos = data.body(body_id).xpos.copy()
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm < 0.001:  # 1mm tolerance
            break

        # Compute Jacobian
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, target_pos, body_id)

        # Extract arm columns
        arm_dof_adrs = [model.jnt_dofadr[jid] for jid in arm_joint_ids]
        J = jacp[:, arm_dof_adrs]  # 3 x 7

        # Damped least-squares
        JJT = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        # Update qpos
        current_qpos = data.qpos[arm_qpos_adrs].copy()
        new_qpos = current_qpos + step_size * dq
        new_qpos = np.clip(new_qpos, arm_ranges[:, 0], arm_ranges[:, 1])
        data.qpos[arm_qpos_adrs] = new_qpos

    mujoco.mj_forward(model, data)
    return data.qpos[arm_qpos_adrs].copy()


def run_ik_sequence(model, data):
    """Run an IK-based grasp sequence and monitor contacts."""
    # Reset
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Set home position first
    home_ctrl = np.array([0.0, -1.57079, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.04])
    data.ctrl[:] = home_ctrl
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Get block position
    block_pos = data.body('red_block').xpos.copy()
    print(f"\n  Block position: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")

    # Phase tracking
    max_finger_block_force = 0.0
    total_finger_block_contacts = 0
    any_finger_block_contact = False
    max_block_z = block_pos[2]
    block_z_init = block_pos[2]

    # --- Phase 1: IK to above block (5cm above) ---
    above_pos = block_pos.copy()
    above_pos[2] += 0.10  # 10cm above block center

    print("\n" + "="*80)
    print("PHASE 1: IK to above block")
    print("="*80)
    print(f"  Target: [{above_pos[0]:.4f}, {above_pos[1]:.4f}, {above_pos[2]:.4f}]")

    target_qpos = solve_ik(model, data, above_pos, body_name="hand", max_iter=200, step_size=0.3)
    data.ctrl[:7] = target_qpos
    data.ctrl[7] = 0.04  # Open gripper
    for _ in range(300):
        mujoco.mj_step(model, data)

    hand_pos = data.body('hand').xpos.copy()
    print(f"  Hand position: [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Error: {np.linalg.norm(hand_pos - above_pos):.4f}")
    print_body_positions(data, model)

    # --- Phase 2: IK to block level (lower hand to just above block) ---
    grasp_pos = block_pos.copy()
    grasp_pos[2] += 0.02  # Just above block top (block half-size is 0.02)

    print("\n" + "="*80)
    print("PHASE 2: IK to grasp position (just above block)")
    print("="*80)
    print(f"  Target: [{grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}]")

    target_qpos = solve_ik(model, data, grasp_pos, body_name="hand", max_iter=200, step_size=0.3)
    data.ctrl[:7] = target_qpos
    data.ctrl[7] = 0.04  # Open gripper
    for _ in range(300):
        mujoco.mj_step(model, data)

    hand_pos = data.body('hand').xpos.copy()
    finger1_pos = data.body('left_finger').xpos.copy()
    finger2_pos = data.body('right_finger').xpos.copy()
    print(f"  Hand position: [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Left finger:   [{finger1_pos[0]:.4f}, {finger1_pos[1]:.4f}, {finger1_pos[2]:.4f}]")
    print(f"  Right finger:  [{finger2_pos[0]:.4f}, {finger2_pos[1]:.4f}, {finger2_pos[2]:.4f}]")
    print(f"  Error: {np.linalg.norm(hand_pos - grasp_pos):.4f}")

    # Check finger-to-block distance
    block_top = block_pos[2] + 0.02  # block half-size
    finger_y_gap = abs(finger1_pos[1] - finger2_pos[1])
    print(f"  Block top z: {block_top:.4f}")
    print(f"  Finger y gap: {finger_y_gap:.4f} (block width: 0.04)")

    contacts = get_contact_info(model, data)
    hb_contacts = [c for c in contacts if is_hand_block_contact(c, model)]
    print(f"  Total contacts: {len(contacts)}, hand-block: {len(hb_contacts)}")
    if hb_contacts:
        for c in hb_contacts:
            print(f"    CONTACT: {c['geom1']} <-> {c['geom2']} normal={c['normal']:.4f}")

    print_body_positions(data, model)

    # --- Phase 3: Close gripper ---
    print("\n" + "="*80)
    print("PHASE 3: Closing gripper")
    print("="*80)
    for step in range(300):
        # Gradually close gripper
        alpha = min(step / 150.0, 1.0)
        data.ctrl[7] = 0.04 * (1.0 - alpha)  # Close from 0.04 to 0.0
        mujoco.mj_step(model, data)

        contacts = get_contact_info(model, data)
        hb_contacts = [c for c in contacts if is_hand_block_contact(c, model)]
        fb_contacts = [c for c in contacts if is_finger_block_contact(c, model)]

        if hb_contacts:
            any_finger_block_contact = True
            for c in hb_contacts:
                total_finger_block_contacts += 1
                max_finger_block_force = max(max_finger_block_force, c['normal'])

        block_z = data.body('red_block').xpos[2]
        max_block_z = max(max_block_z, block_z)

        if step % 50 == 0:
            finger1_pos = data.body('left_finger').xpos
            finger2_pos = data.body('right_finger').xpos
            print(f"  Step {step}: ctrl_finger={data.ctrl[7]:.4f} block_z={block_z:.4f} "
                  f"Lfinger_y={finger1_pos[1]:.4f} Rfinger_y={finger2_pos[1]:.4f} "
                  f"gap={abs(finger1_pos[1]-finger2_pos[1]):.4f} "
                  f"contacts={len(contacts)} hand-block={len(hb_contacts)} finger-block={len(fb_contacts)}")
            if hb_contacts:
                for c in hb_contacts:
                    print(f"    CONTACT: {c['geom1']} <-> {c['geom2']} "
                          f"normal={c['normal']:.4f} friction={c['friction']:.4f}")

    print_body_positions(data, model)

    # --- Phase 4: Lift ---
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.15  # Lift 15cm

    print("\n" + "="*80)
    print("PHASE 4: Lifting")
    print("="*80)
    print(f"  Target: [{lift_pos[0]:.4f}, {lift_pos[1]:.4f}, {lift_pos[2]:.4f}]")

    # Use IK to compute lift target
    target_qpos_lift = solve_ik(model, data, lift_pos, body_name="hand", max_iter=200, step_size=0.3)

    # Interpolate to lift position
    start_qpos = data.ctrl[:7].copy()
    for step in range(300):
        alpha = min(step / 150.0, 1.0)
        data.ctrl[:7] = start_qpos + alpha * (target_qpos_lift[:7] - start_qpos[:7])
        data.ctrl[7] = 0.0  # Keep gripper closed
        mujoco.mj_step(model, data)

        block_z = data.body('red_block').xpos[2]
        max_block_z = max(max_block_z, block_z)

        contacts = get_contact_info(model, data)
        hb_contacts = [c for c in contacts if is_hand_block_contact(c, model)]

        if hb_contacts:
            any_finger_block_contact = True
            for c in hb_contacts:
                total_finger_block_contacts += 1
                max_finger_block_force = max(max_finger_block_force, c['normal'])

        if step % 50 == 0:
            hand_pos = data.body('hand').xpos
            block_pos_now = data.body('red_block').xpos
            print(f"  Step {step}: hand_z={hand_pos[2]:.4f} block_z={block_pos_now[2]:.4f} "
                  f"contacts={len(contacts)} hand-block={len(hb_contacts)}")
            if hb_contacts:
                for c in hb_contacts:
                    print(f"    CONTACT: {c['geom1']} <-> {c['geom2']} "
                          f"normal={c['normal']:.4f} friction={c['friction']:.4f}")

    # --- Summary ---
    block_z_final = data.body('red_block').xpos[2]
    block_lifted = block_z_final > block_z_init + 0.01

    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"  Block initial z:     {block_z_init:.4f}")
    print(f"  Block final z:       {block_z_final:.4f}")
    print(f"  Block max z:         {max_block_z:.4f}")
    print(f"  Block lifted:        {block_lifted}")
    print(f"  Any hand-block contact:    {any_finger_block_contact}")
    print(f"  Total hand-block contacts: {total_finger_block_contacts}")
    print(f"  Max hand-block force:      {max_finger_block_force:.4f}")

    if not any_finger_block_contact:
        print("\n  *** ERROR: No contact force between gripper and block! ***")
        print("  This means the gripper cannot physically grasp the block.")
        print("  Possible causes:")
        print("    1. Finger collision geoms have contype=0 or conaffinity=0")
        print("    2. Finger geoms are too small to touch the block")
        print("    3. Gripper is not positioned correctly over the block")
        print("    4. Hand orientation is wrong (fingers not aligned with block)")
    elif not block_lifted:
        print("\n  *** WARNING: Contact detected but block not lifted! ***")
        print("  This means the gripper touches the block but cannot lift it.")
        print("  Possible causes:")
        print("    1. Grip force too low (actuator kp too small)")
        print("    2. Friction too low (block slides out)")
        print("    3. Block too heavy")
        print("    4. Gripper closing not tight enough")
    else:
        print("\n  *** SUCCESS: Block was lifted by the gripper! ***")
        print("  The physics simulation is working correctly for grasping.")

    # Print all contacts at the end
    print("\n  All active contacts at final step:")
    contacts = get_contact_info(model, data)
    if contacts:
        for c in contacts:
            print(f"    {c['geom1']:<35} <-> {c['geom2']:<35} "
                  f"normal={c['normal']:.4f} friction={c['friction']:.4f}")
    else:
        print("    (none)")

    return any_finger_block_contact, block_lifted


def main():
    print("Loading MuJoCo model from:", SCENE_PATH)
    model = MjModel.from_xml_path(SCENE_PATH)
    data = MjData(model)

    print(f"Model: {model.nq} DOF, {model.nu} actuators, {model.ngeom} geoms")
    print(f"Timestep: {model.opt.timestep}")

    # Print actuator info
    print("\nActuators:")
    for i in range(model.nu):
        name = model.actuator(i).name or f"(unnamed_{i})"
        ctrlrange = model.actuator_ctrlrange[i]
        print(f"  {name:<20} ctrlrange=[{ctrlrange[0]:.4f}, {ctrlrange[1]:.4f}]")

    # Print geom collision properties
    print_geom_info(model)

    # Run the IK-based diagnostic sequence
    has_contact, block_lifted = run_ik_sequence(model, data)


if __name__ == "__main__":
    main()
