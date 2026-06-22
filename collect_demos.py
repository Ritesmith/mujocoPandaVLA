#!/usr/bin/env python3
"""Collect demo trajectories from scripted policy for DAPG training.

Generates successful pick-place trajectories using the gravity-compensated
scripted policy, then saves them as transition tuples (obs, action, reward,
next_obs, done) compatible with Stable-Baselines3 replay buffers.

The scripted policy runs its own MuJoCo simulation with gravity compensation.
We collect (qpos, ctrl) pairs and convert them to the Gym environment's
observation/action format.

Usage:
    python collect_demos.py --n_demos 50 --output demos/panda_pickplace.npz
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import mujoco
from mujoco import MjModel, MjData
from scripted_policy import generate_pick_place_trajectory


SCENE_XML = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"


def compute_reward_dense(model, data, block_id, hand_id, target_pos,
                         prev_hand_block_dist=None, prev_block_height=None):
    """Compute dense reward v3 matching PandaVLAEnv."""
    block_pos = data.xpos[block_id].copy()
    hand_pos = data.xpos[hand_id].copy()

    hand_block_dist = np.linalg.norm(hand_pos - block_pos)
    block_target_dist = np.linalg.norm(block_pos - target_pos)
    block_z = block_pos[2]
    table_z = 0.22

    finger_qpos_adrs = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")],
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")],
    ]
    gripper_open = data.qpos[finger_qpos_adrs].mean() > 0.02
    lift_height = max(0, block_z - table_z)

    reward = 0.0

    # 1. Hand progress reward
    if prev_hand_block_dist is not None:
        progress = prev_hand_block_dist - hand_block_dist
        reward += 3.0 * progress

    # 2. Lifting progress reward
    if prev_block_height is not None:
        height_progress = lift_height - max(0, prev_block_height - table_z)
        reward += 20.0 * height_progress

    # 3. Proximity bonus
    if hand_block_dist < 0.05:
        reward += 0.05
    elif hand_block_dist < 0.10:
        reward += 0.02

    # 4. Grasp bonus
    block_in_hand = hand_block_dist < 0.05 and not gripper_open
    if block_in_hand:
        reward += 0.1

    # 5. Lifting bonus
    if lift_height > 0.02:
        reward += 1.0
    if lift_height > 0.05:
        reward += 2.0
    if lift_height > 0.10:
        reward += 3.0

    # 6. Placing bonus
    if lift_height > 0.03:
        if block_target_dist < 0.05:
            reward += 5.0
        elif block_target_dist < 0.10:
            reward += 2.0

    reward = float(np.clip(reward, -1.0, 15.0))
    return reward, hand_block_dist, block_z


def collect_demos(n_demos=50, output_path="demos/panda_pickplace.npz",
                  verbose=True):
    """Collect demo trajectories using the scripted policy.

    Runs the scripted policy's internal simulation and collects
    (obs, action, reward, next_obs, done) transitions.
    """
    model = MjModel.from_xml_path(SCENE_XML)
    data = MjData(model)

    # Body IDs
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    block_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint")
    block_qpos_adr = model.jnt_qposadr[block_joint_id]

    arm_joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}") for j in range(1, 8)]
    arm_qpos_adrs = [model.jnt_qposadr[jid] for jid in arm_joint_ids]
    finger_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
    ]
    finger_qpos_adrs = [model.jnt_qposadr[jid] for jid in finger_joint_ids]

    target_pos = np.array([0.5, 0.3, 0.22])

    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_dones = []
    all_episode_returns = []

    successful = 0
    total = 0

    for demo_idx in range(n_demos):
        # Generate trajectory using scripted policy
        trajectory, traj_success, final_block_pos = generate_pick_place_trajectory()

        if not traj_success:
            if verbose:
                print(f"Demo {demo_idx}: scripted policy failed, skipping")
            continue

        # Replay trajectory and collect transitions
        # Reset simulation
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
        data.qpos[7:9] = [0.04, 0.04]
        data.ctrl[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
        data.ctrl[7] = 0.04
        data.qpos[block_qpos_adr:block_qpos_adr+7] = [0.5, 0.0, 0.24, 1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)

        episode_reward = 0.0
        prev_hand_block_dist = None
        prev_block_height = None

        for step_idx, (qpos_target, ctrl_target) in enumerate(trajectory):
            # Current observation: 16-dim matching FlattenObs
            # joint_positions(7) + gripper(1) + block_pos(3) + hand_pos(3)
            # + hand_block_dist(1) + block_target_dist(1) = 16
            joint_pos = data.qpos[arm_qpos_adrs].copy()
            gripper = np.array([data.qpos[finger_qpos_adrs].mean()])
            block_pos_now = data.xpos[block_id].copy()
            hand_pos_now = data.xpos[hand_id].copy()
            hand_block_dist = np.linalg.norm(hand_pos_now - block_pos_now)
            block_target_dist = np.linalg.norm(block_pos_now - target_pos)
            obs = np.concatenate([
                joint_pos, gripper,
                block_pos_now, hand_pos_now,
                [hand_block_dist], [block_target_dist]
            ]).astype(np.float32)

            # Action: ctrl_target is the desired ctrl values
            # Convert to env action format (velocity delta + gripper command)
            current_ctrl = data.ctrl[:8].copy()
            arm_delta = (ctrl_target[:7] - current_ctrl[:7]) / 0.05  # scale by control_dt
            arm_delta = np.clip(arm_delta, -1.0, 1.0)

            gripper_cmd = -1.0 if ctrl_target[7] < 0.02 else 1.0
            action = np.zeros(8, dtype=np.float32)
            action[:7] = arm_delta
            action[7] = gripper_cmd

            # Step simulation with gravity compensation
            data.ctrl[:7] = ctrl_target[:7]
            data.ctrl[7] = ctrl_target[7]
            mujoco.mj_forward(model, data)
            data.qfrc_applied[:7] = data.qfrc_bias[:7]
            mujoco.mj_step(model, data)
            data.qfrc_applied[:] = 0

            # Next observation (16-dim)
            joint_pos_next = data.qpos[arm_qpos_adrs].copy()
            gripper_next = np.array([data.qpos[finger_qpos_adrs].mean()])
            block_pos_next = data.xpos[block_id].copy()
            hand_pos_next = data.xpos[hand_id].copy()
            hand_block_dist_next = np.linalg.norm(hand_pos_next - block_pos_next)
            block_target_dist_next = np.linalg.norm(block_pos_next - target_pos)
            next_obs = np.concatenate([
                joint_pos_next, gripper_next,
                block_pos_next, hand_pos_next,
                [hand_block_dist_next], [block_target_dist_next]
            ]).astype(np.float32)

            # Reward
            reward, prev_hand_block_dist, prev_block_height = compute_reward_dense(
                model, data, block_id, hand_id, target_pos,
                prev_hand_block_dist, prev_block_height
            )

            # Done
            done = step_idx == len(trajectory) - 1

            all_obs.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_obs.append(next_obs)
            all_dones.append(float(done))

            episode_reward += reward

        total += 1

        # Check success
        final_block = data.xpos[block_id].copy()
        dist = np.linalg.norm(final_block - target_pos)
        is_success = dist < 0.05
        if is_success:
            successful += 1

        all_episode_returns.append(episode_reward)

        if verbose:
            status = "OK" if is_success else "FAIL"
            print(f"Demo {demo_idx}: {status}, reward={episode_reward:.1f}, "
                  f"steps={len(trajectory)}, dist={dist:.3f}")

    # Convert to arrays
    observations = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    next_observations = np.array(all_next_obs, dtype=np.float32)
    dones = np.array(all_dones, dtype=np.float32)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
        n_demos=total,
        n_successful=successful,
    )

    if verbose:
        print(f"\n{'='*50}")
        print(f"Collected {total} demos ({successful} successful)")
        print(f"Total transitions: {len(observations)}")
        print(f"Mean episode return: {np.mean(all_episode_returns):.1f}")
        print(f"Saved to: {output_path}")
        print(f"{'='*50}")

    return observations, actions, rewards, next_observations, dones


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_demos", type=int, default=50)
    parser.add_argument("--output", type=str, default="demos/panda_pickplace.npz")
    args = parser.parse_args()

    collect_demos(args.n_demos, args.output)

