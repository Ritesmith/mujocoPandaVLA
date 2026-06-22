#!/usr/bin/env python3
"""Scripted policy for Panda pick-place task.

Strategy: Gravity compensation via qfrc_applied + high-gain finger control.

Key insights:
1. Use qfrc_applied[:7] = qfrc_bias[:7] to cancel gravity on arm joints,
   eliminating PD steady-state error (the root cause of arm drift).
2. With gravity compensated, the arm PD controller only handles tracking,
   so ctrl = target_qpos gives near-zero steady-state error.
3. Higher finger kp (5000) closes gripper fast enough to grasp before
   any residual drift can push the block away.
4. Use actual fingertip geom position (not finger body) for waypoint offsets.
5. Fingertips must be at block center height for proper side-squeeze grasp.
6. Smooth ctrl interpolation for transport prevents contact breakage.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
from mujoco import MjModel, MjData


SCENE_XML = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"


def solve_ik(model, data, target_pos, target_rot=None, body_name="hand",
             max_iter=100, tol=0.002, step_size=0.3):
    """Solve IK using Jacobian damped least-squares from current state."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return False, 0

    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}") for j in range(1, 8)]
    arm_dof_adrs = [model.jnt_dofadr[jid] for jid in arm_joint_ids]

    for i in range(max_iter):
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[body_id].copy()
        error_pos = target_pos - current_pos

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

        J_pos = jacp[:, arm_dof_adrs]
        J_rot = jacr[:, arm_dof_adrs]
        J = np.vstack([J_pos, J_rot]) if target_rot is not None else J_pos

        lam = 0.1
        n = J.shape[0]
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(n), error)

        for j, jid in enumerate(arm_joint_ids):
            qadr = model.jnt_qposadr[jid]
            data.qpos[qadr] += dq[j] * step_size
            lo, hi = model.jnt_range[jid]
            data.qpos[qadr] = np.clip(data.qpos[qadr], lo, hi)

    return False, max_iter


def compute_ik_ctrl(model, data, target_pos, target_rot, current_qpos=None):
    """Compute IK on a separate data object and return target ctrl."""
    ik_data = MjData(model)
    if current_qpos is not None:
        ik_data.qpos[:7] = current_qpos.copy()
    else:
        ik_data.qpos[:7] = data.qpos[:7].copy()
    ik_data.qpos[7:9] = [0.04, 0.04]
    mujoco.mj_forward(model, ik_data)

    success, _ = solve_ik(model, ik_data, target_pos, target_rot=target_rot,
                          max_iter=200, tol=0.001)
    return ik_data.qpos[:7].copy(), success


def find_fingertip_geom(model, data, body_name="left_finger"):
    """Find the main fingertip pad geom for a finger body.

    Returns the geom id of the largest box geom in the finger body
    (which is the main fingertip pad collision geom).
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    best_geom = None
    best_vol = 0
    for i in range(model.ngeom):
        if model.geom_bodyid[i] == body_id and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            size = model.geom_size[i]
            vol = size[0] * size[1] * size[2]
            if vol > best_vol:
                best_vol = vol
                best_geom = i
    return best_geom


def print_contacts(model, data, label=""):
    """Print all contacts involving finger or block geoms."""
    pad_names = {f"lf_pad{i}" for i in range(1, 6)} | {f"rf_pad{i}" for i in range(1, 6)}
    finger_contacts = 0
    all_contacts = 0
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom_{g1}"
        g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom_{g2}"
        all_contacts += 1
        is_pad = g1_name in pad_names or g2_name in pad_names
        is_block = "red_block" in g1_name or "red_block" in g2_name
        is_finger_mesh = "finger" in g1_name.lower() or "finger" in g2_name.lower()
        if (is_pad or is_finger_mesh) and is_block:
            finger_contacts += 1
            force = np.linalg.norm(data.efc_force[contact.efc_address:contact.efc_address + 1])
            print(f"  {label}Contact: {g1_name} <-> {g2_name}, force={force:.2f}")
    if all_contacts > 0 and finger_contacts == 0:
        # Print all contacts for debugging if no finger-block contacts found
        print(f"  {label}No finger-block contacts. All contacts ({all_contacts}):")
        for i in range(min(data.ncon, 10)):
            contact = data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom_{g1}"
            g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom_{g2}"
            print(f"    {g1_name} <-> {g2_name}")
    return finger_contacts


def generate_pick_place_trajectory(scene_xml=SCENE_XML, n_settle=2000,
                                   n_close=2500, n_lift=2000, n_grip_settle=500,
                                   n_transport=4000, block_pos=None):
    """Generate a pick-place trajectory with gravity compensation."""
    model = MjModel.from_xml_path(scene_xml)
    data = MjData(model)

    # Body IDs
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    lf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    block_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    block_qpos_adr = model.jnt_qposadr[block_joint_id]

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
    data.qpos[7:9] = [0.04, 0.04]
    data.ctrl[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
    data.ctrl[7] = 0.04
    if block_pos is None:
        block_pos = [0.5, 0.0, 0.24]
    data.qpos[block_qpos_adr:block_qpos_adr+7] = [block_pos[0], block_pos[1], block_pos[2], 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    target_rot = data.xmat[hand_id].reshape(3, 3).copy()
    hand_pos = data.xpos[hand_id].copy()

    # Compute fingertip offset using actual geom positions
    lf_fingertip_geom = find_fingertip_geom(model, data, "left_finger")
    rf_fingertip_geom = find_fingertip_geom(model, data, "right_finger")

    if lf_fingertip_geom is not None and rf_fingertip_geom is not None:
        lf_tip_pos = data.geom_xpos[lf_fingertip_geom].copy()
        rf_tip_pos = data.geom_xpos[rf_fingertip_geom].copy()
        fingertip_mid_z = (lf_tip_pos[2] + rf_tip_pos[2]) / 2
        fingertip_offset_z = hand_pos[2] - fingertip_mid_z
        print(f"Fingertip geom positions: lf={lf_tip_pos}, rf={rf_tip_pos}")
    else:
        lf_pos = data.xpos[lf_id].copy()
        fingertip_offset_z = hand_pos[2] - lf_pos[2] + 0.04
        print(f"WARNING: Fingertip geom not found, using estimated offset")

    print(f"Hand: {hand_pos}, fingertip_offset_z={fingertip_offset_z:.4f}")

    block_pos = data.xpos[block_id].copy()
    target_place_pos = np.array([0.5, 0.3, 0.22])

    print(f"Block: {block_pos}")

    # Waypoints - fingertips at block center height for proper side-squeeze
    above_block = block_pos.copy()
    above_block[2] = block_pos[2] + 0.08 + fingertip_offset_z

    on_block = block_pos.copy()
    on_block[2] = block_pos[2] + fingertip_offset_z  # fingertips at block center z

    lift_pos = block_pos.copy()
    lift_pos[2] = block_pos[2] + 0.15 + fingertip_offset_z

    above_target = target_place_pos.copy()
    above_target[2] = target_place_pos[2] + 0.15 + fingertip_offset_z

    at_target = target_place_pos.copy()
    at_target[2] = target_place_pos[2] + fingertip_offset_z

    print(f"\nWaypoints:")
    print(f"  above_block: {above_block}")
    print(f"  on_block: {on_block} (fingertips at z={on_block[2]-fingertip_offset_z:.4f})")
    print(f"  lift_pos: {lift_pos}")

    trajectory = []

    def step_sim(target_ctrl=None, gripper_ctrl=None, gravity_comp=True, record=True):
        """Step simulation with gravity compensation for arm joints."""
        if gravity_comp:
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:7] = data.qfrc_bias[:7]
        if target_ctrl is not None:
            data.ctrl[:7] = target_ctrl
        if gripper_ctrl is not None:
            data.ctrl[7] = gripper_ctrl
        mujoco.mj_step(model, data)
        data.qfrc_applied[:] = 0
        if record:
            trajectory.append((data.qpos[:9].copy(), data.ctrl[:8].copy()))

    def move_to_pos(target_pos, gripper_ctrl, n_steps, n_ik_iters=8):
        """Move to target position with gravity compensation + iterative IK."""
        for ik_iter in range(n_ik_iters):
            target_ctrl, success = compute_ik_ctrl(
                model, data, target_pos, target_rot, data.qpos[:7].copy()
            )
            if not success:
                print(f"  IK failed at iter {ik_iter}")
                continue

            steps_per_iter = n_steps // n_ik_iters
            for _ in range(steps_per_iter):
                step_sim(target_ctrl, gripper_ctrl)

            # Check convergence
            mujoco.mj_forward(model, data)
            hand_pos_now = data.xpos[hand_id].copy()
            error = np.linalg.norm(hand_pos_now - target_pos)
            if ik_iter % 2 == 0:
                print(f"  IK iter {ik_iter}: error={error*1000:.1f}mm, hand={hand_pos_now}")
            if error < 0.005:
                print(f"  Converged at iter {ik_iter}, error={error*1000:.1f}mm")
                break

    def move_to_pos_smooth(target_pos, gripper_ctrl, n_steps):
        """Move to target using smooth ctrl interpolation (better for grasped transport)."""
        # Compute start and end ctrl
        start_ctrl = data.qpos[:7].copy()
        end_ctrl, success = compute_ik_ctrl(model, data, target_pos, target_rot, start_ctrl)
        if not success:
            print(f"  Smooth move: IK failed, falling back to iterative")
            move_to_pos(target_pos, gripper_ctrl, n_steps)
            return

        for step in range(n_steps):
            t = step / max(n_steps - 1, 1)
            # Smooth interpolation (ease in-out)
            t_smooth = 0.5 - 0.5 * np.cos(t * np.pi)
            interp_ctrl = start_ctrl + t_smooth * (end_ctrl - start_ctrl)
            step_sim(interp_ctrl, gripper_ctrl)

            # Re-compute end ctrl periodically to correct drift
            if step > 0 and step % (n_steps // 4) == 0:
                mujoco.mj_forward(model, data)
                hand_pos_now = data.xpos[hand_id].copy()
                error = np.linalg.norm(hand_pos_now - target_pos)
                if error > 0.01:
                    end_ctrl, _ = compute_ik_ctrl(model, data, target_pos, target_rot, data.qpos[:7].copy())
                    start_ctrl = data.qpos[:7].copy()

        mujoco.mj_forward(model, data)
        hand_pos_now = data.xpos[hand_id].copy()
        error = np.linalg.norm(hand_pos_now - target_pos)
        print(f"  Smooth move done: error={error*1000:.1f}mm, hand={hand_pos_now}")

    # Phase 1: Move above block
    print("\nPhase 1: Move above block")
    move_to_pos(above_block, 0.04, n_settle)
    hand_pos = data.xpos[hand_id].copy()
    if lf_fingertip_geom is not None:
        lf_tip = data.geom_xpos[lf_fingertip_geom].copy()
        rf_tip = data.geom_xpos[rf_fingertip_geom].copy()
        print(f"  hand={hand_pos}, lf_tip_z={lf_tip[2]:.4f}, rf_tip_z={rf_tip[2]:.4f}")

    # Phase 2: Lower to block
    print("\nPhase 2: Lower to block")
    move_to_pos(on_block, 0.04, n_settle)
    hand_pos = data.xpos[hand_id].copy()
    block_pos_now = data.xpos[block_id].copy()
    if lf_fingertip_geom is not None:
        lf_tip = data.geom_xpos[lf_fingertip_geom].copy()
        rf_tip = data.geom_xpos[rf_fingertip_geom].copy()
        tip_y_gap = (rf_tip[1] - lf_tip[1]) * 1000
        print(f"  hand={hand_pos}, tip_y_gap={tip_y_gap:.1f}mm, lf_tip_z={lf_tip[2]:.4f}")
        print(f"  block center z={block_pos_now[2]:.4f}, fingertip z={(lf_tip[2]+rf_tip[2])/2:.4f}")
    print(f"  block={block_pos_now}")

    # Phase 3: Close gripper
    print("\nPhase 3: Close gripper")
    arm_ctrl, _ = compute_ik_ctrl(model, data, on_block, target_rot, data.qpos[:7].copy())

    for step in range(n_close):
        t = step / max(n_close - 1, 1)
        gripper_ctrl = 0.04 * (1 - t)  # 0.04 -> 0.0

        # Re-compute arm ctrl periodically
        if step % 300 == 0:
            arm_ctrl, _ = compute_ik_ctrl(model, data, on_block, target_rot, data.qpos[:7].copy())

        step_sim(arm_ctrl, gripper_ctrl)

    # Check grasp
    hand_pos = data.xpos[hand_id].copy()
    block_pos_now = data.xpos[block_id].copy()
    if lf_fingertip_geom is not None:
        lf_tip = data.geom_xpos[lf_fingertip_geom].copy()
        rf_tip = data.geom_xpos[rf_fingertip_geom].copy()
        tip_y_gap = (rf_tip[1] - lf_tip[1]) * 1000
        print(f"  tip_y_gap={tip_y_gap:.1f}mm, lf_tip={lf_tip}, rf_tip={rf_tip}")
    print(f"  hand={hand_pos}, block={block_pos_now}")
    print(f"  finger_qpos={data.qpos[7:9]}")

    # Check contacts
    print_contacts(model, data, "Phase3: ")

    # Phase 3b: Grip settle - hold position to let contact forces stabilize
    print("\nPhase 3b: Grip settle")
    arm_ctrl, _ = compute_ik_ctrl(model, data, on_block, target_rot, data.qpos[:7].copy())
    for _ in range(n_grip_settle):
        step_sim(arm_ctrl, 0.0)
    block_pos_now = data.xpos[block_id].copy()
    print(f"  block after settle={block_pos_now}")
    print_contacts(model, data, "Settle: ")

    # Phase 4: Lift straight up (vertical only - critical for maintaining grasp)
    print("\nPhase 4: Lift straight up")
    # Compute a position directly above current hand position
    hand_pos_now = data.xpos[hand_id].copy()
    vertical_lift = hand_pos_now.copy()
    vertical_lift[2] = lift_pos[2]  # Only change z
    move_to_pos_smooth(vertical_lift, 0.0, n_lift)
    block_pos_now = data.xpos[block_id].copy()
    block_lifted = block_pos_now[2] - 0.24
    print(f"  block={block_pos_now}, lifted={block_lifted*1000:.1f}mm")
    print_contacts(model, data, "Lift: ")

    # Phase 5: Move horizontally to above target (smooth transport)
    print("\nPhase 5: Horizontal transport")
    move_to_pos_smooth(above_target, 0.0, n_transport)
    block_pos_now = data.xpos[block_id].copy()
    print(f"  block={block_pos_now}")
    print_contacts(model, data, "Transport: ")

    # Phase 6: Lower to target
    print("\nPhase 6: Lower to target")
    move_to_pos_smooth(at_target, 0.0, n_settle)

    # Phase 7: Open gripper
    print("\nPhase 7: Release")
    arm_ctrl, _ = compute_ik_ctrl(model, data, at_target, target_rot, data.qpos[:7].copy())
    for step in range(n_close):
        t = step / max(n_close - 1, 1)
        gripper_ctrl = 0.04 * t
        if step % 300 == 0:
            arm_ctrl, _ = compute_ik_ctrl(model, data, at_target, target_rot, data.qpos[:7].copy())
        step_sim(arm_ctrl, gripper_ctrl)

    final_block_pos = data.xpos[block_id].copy()
    success = np.linalg.norm(final_block_pos - target_place_pos) < 0.05

    print(f"\nFinal block: {final_block_pos}")
    print(f"Target: {target_place_pos}")
    print(f"Distance: {np.linalg.norm(final_block_pos - target_place_pos):.4f}")
    print(f"Success: {success}")

    return trajectory, success, final_block_pos


class ScriptedPolicyJoint:
    """Scripted policy that outputs joint-space actions for PandaVLAEnv."""

    def __init__(self, scene_xml=SCENE_XML):
        self.scene_xml = scene_xml
        self.trajectory = None
        self.step_idx = 0

    def reset(self, block_pos=None, target_pos=None):
        self.trajectory, self.success, self.final_block_pos = generate_pick_place_trajectory(
            self.scene_xml
        )
        self.step_idx = 0
        return self.success

    def predict(self, obs):
        if self.trajectory is None or self.step_idx >= len(self.trajectory):
            return np.zeros(8, dtype=np.float32)

        qpos_target, ctrl_target = self.trajectory[self.step_idx]
        self.step_idx += 1

        current_joints = obs.get("joint_positions", np.zeros(7))
        joint_delta = (ctrl_target[:7] - current_joints) / (1.0 * 0.05)
        joint_delta = np.clip(joint_delta, -1.0, 1.0)

        gripper_cmd = -1.0 if ctrl_target[7] < 0.02 else 1.0

        action = np.zeros(8, dtype=np.float32)
        action[:7] = joint_delta
        action[7] = gripper_cmd
        return action


if __name__ == "__main__":
    print("=" * 60)
    print("Test: Gravity-compensated pick-place trajectory")
    print("=" * 60)

    trajectory, success, final_pos = generate_pick_place_trajectory()
    print(f"\nTrajectory length: {len(trajectory)} steps")
    print(f"Success: {success}")
