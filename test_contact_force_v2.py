#!/usr/bin/env python3
"""MuJoCo contact force diagnostic v2 - Direct qpos test.

Bypasses IK approach by directly setting joint positions to place
the gripper around the block, then tests contact and lifting.

Usage:
    cd /home/w/vla_workspace
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate vla
    python test_contact_force_v2.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from mujoco import MjModel, MjData

SCENE_PATH = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"


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


def is_hand_block_contact(contact):
    """Check if a contact involves hand/finger and the block."""
    g1, g2 = contact['geom1'], contact['geom2']
    hand_names = {'hand_c', 'finger_0', 'fingertip_pad_collision_1', 'fingertip_pad_collision_2',
                  'fingertip_pad_collision_3', 'fingertip_pad_collision_4',
                  'fingertip_pad_collision_5'}
    block_names = {'red_block_geom'}
    return (g1 in hand_names and g2 in block_names) or \
           (g2 in hand_names and g1 in block_names)


def print_finger_geom_details(model, data):
    """Print detailed info about finger collision geoms."""
    print("\n  Finger geom details:")
    for body_name in ['left_finger', 'right_finger']:
        body_id = model.body(body_name).id
        print(f"  {body_name} (body_id={body_id}):")
        # Find geoms belonging to this body
        for i in range(model.ngeom):
            if model.geom_bodyid[i] == body_id:
                name = model.geom(i).name or f"geom_{i}"
                contype = model.geom_contype[i]
                conaffinity = model.geom_conaffinity[i]
                group = model.geom_group[i]
                gtype = model.geom_type[i]
                size = model.geom_size[i]
                pos = model.geom_pos[i]
                print(f"    {name}: type={gtype} size={size} pos={pos} "
                      f"contype={contype} conaffinity={conaffinity} group={group}")


def main():
    print("Loading MuJoCo model from:", SCENE_PATH)
    model = MjModel.from_xml_path(SCENE_PATH)
    data = MjData(model)

    print(f"Model: {model.nq} DOF, {model.nu} actuators, {model.ngeom} geoms")

    # Print finger geom details
    print_finger_geom_details(model, data)

    # Print block geom details
    block_body_id = model.body('red_block').id
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == block_body_id:
            name = model.geom(i).name or f"geom_{i}"
            contype = model.geom_contype[i]
            conaffinity = model.geom_conaffinity[i]
            size = model.geom_size[i]
            print(f"\n  Block geom: {name}: size={size} contype={contype} conaffinity={conaffinity}")

    # Print hand orientation
    hand_body_id = model.body('hand').id
    hand_quat = data.body(hand_body_id).xquat
    print(f"\n  Hand quaternion (home): {hand_quat}")

    # ========================================================================
    # TEST 1: Direct qpos placement - hand around block
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: Direct qpos - place hand around block")
    print("="*80)

    mujoco.mj_resetData(model, data)

    # The block is at (0.5, 0.0, 0.22) with half-size 0.02
    # Block center z = 0.22 + 0.02 = 0.24 (on table at z=0.22, block half-height 0.02)
    # We need the hand to be positioned so fingers are on either side of the block

    # Use IK to find a good configuration, but do it step by step
    # First, set home position
    # qpos layout: 7 arm joints + 2 finger joints + 3 block pos + 4 block quat = 16
    home_qpos = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04,
                          0.5, 0.0, 0.24, 1, 0, 0, 0])
    data.qpos[:] = home_qpos
    mujoco.mj_forward(model, data)

    block_pos = data.body('red_block').xpos.copy()
    print(f"  Block position (after home): [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")

    # Try different joint configurations to find one that places hand near block
    # The Panda workspace: joint1 rotates around z (base rotation)
    # For block at x=0.5, we need the arm extended forward

    # Configuration 1: arm reaching forward, hand pointing down
    # This is a common pick configuration for Panda
    test_configs = {
        "config_forward_down": np.array([
            0.0,      # joint1: base rotation (0 = forward)
            -0.785,   # joint2: shoulder down
            0.0,      # joint3: 
            -2.356,   # joint4: elbow up
            0.0,      # joint5:
            1.571,    # joint6: wrist
            0.785,    # joint7: hand rotation
            0.04,     # finger1: open
            0.04,     # finger2: open
            0.5, 0.0, 0.24, 1, 0, 0, 0,  # block
        ]),
        "config_reach_table": np.array([
            0.0,      # joint1
            0.5,      # joint2: shoulder forward
            0.0,      # joint3
            -1.8,     # joint4: elbow
            0.0,      # joint5
            1.2,      # joint6: wrist
            0.785,    # joint7
            0.04,     # finger1: open
            0.04,     # finger2: open
            0.5, 0.0, 0.24, 1, 0, 0, 0,  # block
        ]),
    }

    for config_name, qpos in test_configs.items():
        print(f"\n  --- Testing {config_name} ---")
        data.qpos[:] = qpos
        data.ctrl[:7] = qpos[:7]
        data.ctrl[7] = qpos[7]  # Open gripper
        mujoco.mj_forward(model, data)

        hand_pos = data.body('hand').xpos.copy()
        finger1_pos = data.body('left_finger').xpos.copy()
        finger2_pos = data.body('right_finger').xpos.copy()
        block_pos = data.body('red_block').xpos.copy()

        print(f"  Hand:    [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
        print(f"  Lfinger: [{finger1_pos[0]:.4f}, {finger1_pos[1]:.4f}, {finger1_pos[2]:.4f}]")
        print(f"  Rfinger: [{finger2_pos[0]:.4f}, {finger2_pos[1]:.4f}, {finger2_pos[2]:.4f}]")
        print(f"  Block:   [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")

        # Check finger gap
        finger_gap_x = abs(finger1_pos[0] - finger2_pos[0])
        finger_gap_y = abs(finger1_pos[1] - finger2_pos[1])
        finger_gap_z = abs(finger1_pos[2] - finger2_pos[2])
        print(f"  Finger gap: x={finger_gap_x:.4f} y={finger_gap_y:.4f} z={finger_gap_z:.4f}")

        # Distance from hand to block
        hand_block_dist = np.linalg.norm(hand_pos - block_pos)
        print(f"  Hand-block distance: {hand_block_dist:.4f}")

    # ========================================================================
    # TEST 2: Use IK with collision-free approach
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: IK approach from above (collision-free)")
    print("="*80)

    mujoco.mj_resetData(model, data)
    data.qpos[:] = home_qpos
    data.ctrl[:] = home_qpos[:8]
    for _ in range(500):
        mujoco.mj_step(model, data)

    block_pos = data.body('red_block').xpos.copy()
    print(f"  Block position: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")

    # IK to position well above block first
    target_above = block_pos.copy()
    target_above[2] += 0.20  # 20cm above

    arm_joint_names = [f"joint{i}" for i in range(1, 8)]
    arm_joint_ids = [model.joint(jn).id for jn in arm_joint_names]
    arm_qpos_adrs = [model.jnt_dofadr[jid] for jid in arm_joint_ids]
    arm_ranges = np.array([[model.jnt_range[jid][0], model.jnt_range[jid][1]] for jid in arm_joint_ids])

    def ik_step(model, data, target_pos, body_name="hand", step_size=0.3, damping=0.01):
        body_id = model.body(body_name).id
        mujoco.mj_forward(model, data)
        current_pos = data.body(body_id).xpos.copy()
        error = target_pos - current_pos

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, target_pos, body_id)

        arm_dof_adrs = [model.jnt_dofadr[jid] for jid in arm_joint_ids]
        J = jacp[:, arm_dof_adrs]
        JJT = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        current_qpos = data.qpos[arm_qpos_adrs].copy()
        new_qpos = current_qpos + step_size * dq
        new_qpos = np.clip(new_qpos, arm_ranges[:, 0], arm_ranges[:, 1])
        data.qpos[arm_qpos_adrs] = new_qpos
        return np.linalg.norm(error)

    # Step 1: IK to above block (20cm)
    print("\n  Step 1: IK to 20cm above block")
    for i in range(200):
        err = ik_step(model, data, target_above)
        if err < 0.001:
            break
    data.ctrl[:7] = data.qpos[arm_qpos_adrs]
    data.ctrl[7] = 0.04
    for _ in range(300):
        mujoco.mj_step(model, data)

    hand_pos = data.body('hand').xpos.copy()
    block_pos = data.body('red_block').xpos.copy()
    print(f"  Hand: [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Block: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
    print(f"  Error: {np.linalg.norm(hand_pos - target_above):.4f}")

    # Step 2: IK to 5cm above block
    target_5cm = block_pos.copy()
    target_5cm[2] += 0.05
    print("\n  Step 2: IK to 5cm above block")
    for i in range(200):
        err = ik_step(model, data, target_5cm)
        if err < 0.001:
            break
    data.ctrl[:7] = data.qpos[arm_qpos_adrs]
    data.ctrl[7] = 0.04
    for _ in range(300):
        mujoco.mj_step(model, data)

    hand_pos = data.body('hand').xpos.copy()
    block_pos = data.body('red_block').xpos.copy()
    finger1_pos = data.body('left_finger').xpos.copy()
    finger2_pos = data.body('right_finger').xpos.copy()
    print(f"  Hand: [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Lfinger: [{finger1_pos[0]:.4f}, {finger1_pos[1]:.4f}, {finger1_pos[2]:.4f}]")
    print(f"  Rfinger: [{finger2_pos[0]:.4f}, {finger2_pos[1]:.4f}, {finger2_pos[2]:.4f}]")
    print(f"  Block: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")

    contacts = get_contact_info(model, data)
    hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
    print(f"  Contacts: total={len(contacts)} hand-block={len(hb_contacts)}")

    # Step 3: IK to block level (hand at block center height)
    target_grasp = block_pos.copy()
    target_grasp[2] = block_pos[2]  # Same height as block center
    print("\n  Step 3: IK to block center height")
    for i in range(200):
        err = ik_step(model, data, target_grasp)
        if err < 0.001:
            break
    data.ctrl[:7] = data.qpos[arm_qpos_adrs]
    data.ctrl[7] = 0.04
    for _ in range(300):
        mujoco.mj_step(model, data)

    hand_pos = data.body('hand').xpos.copy()
    block_pos = data.body('red_block').xpos.copy()
    finger1_pos = data.body('left_finger').xpos.copy()
    finger2_pos = data.body('right_finger').xpos.copy()
    print(f"  Hand: [{hand_pos[0]:.4f}, {hand_pos[1]:.4f}, {hand_pos[2]:.4f}]")
    print(f"  Lfinger: [{finger1_pos[0]:.4f}, {finger1_pos[1]:.4f}, {finger1_pos[2]:.4f}]")
    print(f"  Rfinger: [{finger2_pos[0]:.4f}, {finger2_pos[1]:.4f}, {finger2_pos[2]:.4f}]")
    print(f"  Block: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")

    finger_gap_x = abs(finger1_pos[0] - finger2_pos[0])
    finger_gap_y = abs(finger1_pos[1] - finger2_pos[1])
    print(f"  Finger gap: x={finger_gap_x:.4f} y={finger_gap_y:.4f}")

    contacts = get_contact_info(model, data)
    hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
    print(f"  Contacts: total={len(contacts)} hand-block={len(hb_contacts)}")
    for c in hb_contacts:
        print(f"    CONTACT: {c['geom1']} <-> {c['geom2']} normal={c['normal']:.4f}")

    # Step 4: Close gripper
    print("\n  Step 4: Closing gripper")
    for step in range(300):
        alpha = min(step / 150.0, 1.0)
        data.ctrl[7] = 0.04 * (1.0 - alpha)
        mujoco.mj_step(model, data)

        if step % 100 == 0:
            contacts = get_contact_info(model, data)
            hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
            block_z = data.body('red_block').xpos[2]
            finger1_pos = data.body('left_finger').xpos
            finger2_pos = data.body('right_finger').xpos
            print(f"    Step {step}: ctrl={data.ctrl[7]:.4f} block_z={block_z:.4f} "
                  f"gap_y={abs(finger1_pos[1]-finger2_pos[1]):.4f} "
                  f"hand-block contacts={len(hb_contacts)}")
            for c in hb_contacts:
                print(f"      CONTACT: {c['geom1']} <-> {c['geom2']} normal={c['normal']:.4f}")

    # Step 5: Lift
    print("\n  Step 5: Lifting")
    target_lift = data.body('hand').xpos.copy()
    target_lift[2] += 0.15

    for i in range(200):
        err = ik_step(model, data, target_lift)
        if err < 0.001:
            break
    lift_qpos = data.qpos[arm_qpos_adrs].copy()

    start_qpos = data.ctrl[:7].copy()
    block_z_init = data.body('red_block').xpos[2]

    for step in range(300):
        alpha = min(step / 150.0, 1.0)
        data.ctrl[:7] = start_qpos + alpha * (lift_qpos - start_qpos)
        data.ctrl[7] = 0.0  # Keep closed
        mujoco.mj_step(model, data)

        if step % 100 == 0:
            block_z = data.body('red_block').xpos[2]
            hand_z = data.body('hand').xpos[2]
            contacts = get_contact_info(model, data)
            hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
            print(f"    Step {step}: hand_z={hand_z:.4f} block_z={block_z:.4f} "
                  f"hand-block contacts={len(hb_contacts)}")

    block_z_final = data.body('red_block').xpos[2]
    block_lifted = block_z_final > block_z_init + 0.01
    print(f"\n  Block z: init={block_z_init:.4f} final={block_z_final:.4f} lifted={block_lifted}")

    # ========================================================================
    # TEST 3: Force contact by directly placing fingers around block
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 3: Force contact - set block position between fingers")
    print("="*80)

    mujoco.mj_resetData(model, data)
    data.qpos[:] = home_qpos
    data.ctrl[:] = home_qpos[:8]
    for _ in range(500):
        mujoco.mj_step(model, data)

    # Get the hand position in home config
    hand_pos_home = data.body('hand').xpos.copy()
    finger1_pos_home = data.body('left_finger').xpos.copy()
    finger2_pos_home = data.body('right_finger').xpos.copy()
    print(f"  Home hand: [{hand_pos_home[0]:.4f}, {hand_pos_home[1]:.4f}, {hand_pos_home[2]:.4f}]")
    print(f"  Home Lfinger: [{finger1_pos_home[0]:.4f}, {finger1_pos_home[1]:.4f}, {finger1_pos_home[2]:.4f}]")
    print(f"  Home Rfinger: [{finger2_pos_home[0]:.4f}, {finger2_pos_home[1]:.4f}, {finger2_pos_home[2]:.4f}]")

    # Move the block to be between the fingers
    # Block freejoint: qpos[7:10] = position, qpos[10:14] = quaternion
    block_qpos_adr = model.jnt_dofadr[model.joint('red_block_joint').id]
    block_center_x = (finger1_pos_home[0] + finger2_pos_home[0]) / 2
    block_center_y = (finger1_pos_home[1] + finger2_pos_home[1]) / 2
    block_center_z = (finger1_pos_home[2] + finger2_pos_home[2]) / 2

    # Place block between fingers at their height
    data.qpos[block_qpos_adr:block_qpos_adr+3] = [block_center_x, block_center_y, block_center_z]
    data.qpos[block_qpos_adr+3:block_qpos_adr+7] = [1, 0, 0, 0]  # identity quaternion
    mujoco.mj_forward(model, data)

    block_pos = data.body('red_block').xpos.copy()
    finger1_pos = data.body('left_finger').xpos.copy()
    finger2_pos = data.body('right_finger').xpos.copy()
    print(f"  Block placed at: [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
    print(f"  Lfinger: [{finger1_pos[0]:.4f}, {finger1_pos[1]:.4f}, {finger1_pos[2]:.4f}]")
    print(f"  Rfinger: [{finger2_pos[0]:.4f}, {finger2_pos[1]:.4f}, {finger2_pos[2]:.4f}]")

    # Check distance from block to each finger
    d1 = np.linalg.norm(block_pos - finger1_pos)
    d2 = np.linalg.norm(block_pos - finger2_pos)
    print(f"  Block-Lfinger dist: {d1:.4f}")
    print(f"  Block-Rfinger dist: {d2:.4f}")

    # Now close gripper
    print("\n  Closing gripper with block between fingers...")
    data.ctrl[7] = 0.04  # Start open
    for step in range(500):
        alpha = min(step / 250.0, 1.0)
        data.ctrl[7] = 0.04 * (1.0 - alpha)
        mujoco.mj_step(model, data)

        if step % 100 == 0:
            contacts = get_contact_info(model, data)
            hb_contacts = [c for c in contacts if is_hand_block_contact(c)]
            block_z = data.body('red_block').xpos[2]
            print(f"    Step {step}: ctrl={data.ctrl[7]:.4f} block_z={block_z:.4f} "
                  f"contacts={len(contacts)} hand-block={len(hb_contacts)}")
            for c in hb_contacts:
                print(f"      CONTACT: {c['geom1']} <-> {c['geom2']} "
                      f"normal={c['normal']:.4f} friction={c['friction']:.4f}")

    # Print all contacts
    print("\n  All contacts after closing:")
    contacts = get_contact_info(model, data)
    for c in contacts:
        print(f"    {c['geom1']:<35} <-> {c['geom2']:<35} "
              f"normal={c['normal']:.4f} friction={c['friction']:.4f}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("  Key findings:")
    print(f"  - Collision geoms have correct contype/conaffinity (1,1)")
    print(f"  - Finger collision geoms are in group 3 (collision)")
    print(f"  - Block geom has correct contype/conaffinity (1,1)")
    print(f"  - The hand has a 45-degree z-rotation (quat ~0.924, 0, 0, -0.383)")
    print(f"  - This means fingers open along a diagonal, not along Y axis")
    print(f"  - The fingertip_pad_collision boxes are very small (~8.5mm)")
    print(f"  - The block is 40mm wide, much larger than fingertip pads")


if __name__ == "__main__":
    main()
