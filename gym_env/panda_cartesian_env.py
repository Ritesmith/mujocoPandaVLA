"""Panda environment with Cartesian (end-effector) action space.

Instead of joint-space deltas, actions are Cartesian space deltas:
- [dx, dy, dz]: position delta in world frame (meters)
- [droll, dpitch, dyaw]: orientation delta (not implemented yet, keep current)
- [gripper]: gripper command

Uses Jacobian-based IK to convert Cartesian targets to joint targets.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData


class PandaCartesianEnv(gym.Env):
    """Panda environment with Cartesian action space for easier RL learning."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, image_size=256, control_dt=0.05,
                 model_path=None, max_episode_steps=500,
                 reward_type="dense", target_pos=None):
        super().__init__()

        if model_path is None:
            model_path = os.path.join(
                "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf",
                "franka_emika_panda/scene.xml",
            )

        self.render_mode = render_mode
        self.image_size = image_size
        self.control_dt = control_dt
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Load MuJoCo model
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.model.opt.timestep = 0.002
        self.n_substeps = int(control_dt / self.model.opt.timestep)

        # Discover joint/actuator indices (same as PandaVLAEnv)
        self.n_arm_joints = 7
        self.n_finger_joints = 2
        self.n_actuators = self.model.nu

        self._arm_joint_ids = []
        self._arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for name in self._arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found")
            self._arm_joint_ids.append(jid)

        self._finger_joint_ids = []
        for name in ["finger_joint1", "finger_joint2"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found")
            self._finger_joint_ids.append(jid)

        self._arm_qpos_adrs = [self.model.jnt_qposadr[jid] for jid in self._arm_joint_ids]
        self._finger_qpos_adrs = [self.model.jnt_qposadr[jid] for jid in self._finger_joint_ids]

        self._arm_actuator_ids = []
        for name in self._arm_joint_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise ValueError(f"Actuator '{name}' not found")
            self._arm_actuator_ids.append(aid)

        self._gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint1"
        )
        if self._gripper_actuator_id == -1:
            self._gripper_actuator_id = self.n_actuators - 1

        self._arm_joint_ranges = np.zeros((self.n_arm_joints, 2))
        for i, jid in enumerate(self._arm_joint_ids):
            self._arm_joint_ranges[i, 0] = self.model.jnt_range[jid, 0]
            self._arm_joint_ranges[i, 1] = self.model.jnt_range[jid, 1]

        self._gripper_ctrl_range = self.model.actuator_ctrlrange[self._gripper_actuator_id].copy()

        # Find hand body for IK
        self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self._hand_body_id == -1:
            # Try "panda_hand"
            self._hand_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")

        # Action space: [dx, dy, dz, gripper] = 4D (simplified: no orientation control)
        # dx/dy/dz: position delta in world frame, max 5cm per step
        # gripper: >0 close, <0 open
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation space (same as PandaVLAEnv)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            ),
            "joint_positions": gym.spaces.Box(
                low=-np.pi, high=np.pi, shape=(self.n_arm_joints,), dtype=np.float32
            ),
            "gripper": gym.spaces.Box(
                low=0.0, high=0.04, shape=(1,), dtype=np.float32
            ),
        })

        self._renderer = None
        self._arm_target = np.zeros(self.n_arm_joints)
        self._gripper_target = 0.04

        # Reward parameters
        self.reward_type = reward_type
        self._red_block_id = None
        self._hand_id = None
        self._target_pos = target_pos if target_pos is not None else np.array([0.5, 0.3, 0.2])
        self._prev_hand_block_dist = None
        self._prev_block_height = None
        self._last_action = None
        self._find_bodies()

        self._red_block_qpos_adr = None
        if self._red_block_id is not None:
            try:
                rj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'red_block_joint')
                if rj_id >= 0:
                    self._red_block_qpos_adr = self.model.jnt_qposadr[rj_id]
            except Exception:
                pass

    def _find_bodies(self):
        try:
            self._red_block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
        except Exception:
            self._red_block_id = None
        try:
            self._hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        except Exception:
            self._hand_id = None

    def _get_hand_pos(self):
        """Get current hand position in world frame."""
        if self._hand_body_id >= 0:
            return self.data.xpos[self._hand_body_id].copy()
        elif self._hand_id is not None:
            return self.data.xpos[self._hand_id].copy()
        return np.zeros(3)

    def _ik_step(self, target_pos, step_size=0.5):
        """One step of Jacobian-based IK.

        Computes the Jacobian of the hand position w.r.t. arm joints,
        then uses pseudo-inverse to find joint deltas that move hand toward target.
        """
        # Get current hand position
        current_pos = self._get_hand_pos()
        error = target_pos - current_pos

        # Compute Jacobian for hand position (3 x nv)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        body_id = self._hand_body_id if self._hand_body_id >= 0 else self._hand_id
        if body_id is None or body_id < 0:
            return self.data.qpos[self._arm_qpos_adrs].copy()

        mujoco.mj_jac(self.model, self.data, jacp, jacr, target_pos, body_id)

        # Extract arm joint columns from Jacobian
        # Jacobian is w.r.t. all DOFs; we need only arm joint DOFs
        arm_dof_adrs = [self.model.jnt_dofadr[jid] for jid in self._arm_joint_ids]
        J = jacp[:, arm_dof_adrs]  # 3 x 7

        # Pseudo-inverse IK: dq = J^+ * error
        # Use damped least squares for stability
        damping = 0.01
        JJT = J @ J.T + damping * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        # Scale by step size
        dq = step_size * dq

        # Apply to current joint positions
        new_qpos = self.data.qpos[self._arm_qpos_adrs].copy() + dq

        # Clip to joint limits
        new_qpos = np.clip(new_qpos, self._arm_joint_ranges[:, 0], self._arm_joint_ranges[:, 1])

        return new_qpos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            self.data.qpos[:] = self.model.key_qpos[home_key_id]
            self.data.ctrl[:] = self.model.key_ctrl[home_key_id]
        else:
            self.data.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
            self.data.qpos[7:9] = [0.04, 0.04]
            self.data.ctrl[:7] = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
            self.data.ctrl[7] = 0.04

        if self._red_block_qpos_adr is not None:
            self.data.qpos[self._red_block_qpos_adr + 0] = 0.5
            self.data.qpos[self._red_block_qpos_adr + 1] = 0.0
            self.data.qpos[self._red_block_qpos_adr + 2] = 0.24  # table_top=0.22 + block_half=0.02
            self.data.qpos[self._red_block_qpos_adr + 3] = 1.0
            self.data.qpos[self._red_block_qpos_adr + 4] = 0.0
            self.data.qpos[self._red_block_qpos_adr + 5] = 0.0
            self.data.qpos[self._red_block_qpos_adr + 6] = 0.0

        if self.np_random is not None:
            self.data.qpos[self._arm_qpos_adrs] += self.np_random.uniform(-0.05, 0.05, self.n_arm_joints)

        mujoco.mj_forward(self.model, self.data)

        self._arm_target = self.data.qpos[self._arm_qpos_adrs].copy()
        self._gripper_target = self.data.ctrl[self._gripper_actuator_id]
        self.step_count = 0
        self._prev_hand_block_dist = None
        self._prev_block_height = None
        self._last_action = None

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.copy()

        # Cartesian action: [dx, dy, dz, gripper]
        # Convert position delta to world frame target
        max_pos_delta = 0.05  # 5cm per step
        pos_delta = action[:3] * max_pos_delta

        # Get current hand position and compute target
        current_hand_pos = self._get_hand_pos()
        target_hand_pos = current_hand_pos + pos_delta

        # Use IK to find joint positions that achieve target
        new_arm_target = self._ik_step(target_hand_pos, step_size=1.0)
        self._arm_target = new_arm_target

        # Gripper
        gripper_cmd = action[3]
        if gripper_cmd > 0:
            self._gripper_target = max(0.0, self._gripper_target - 0.02)
        else:
            self._gripper_target = min(0.04, self._gripper_target + 0.02)
        self._gripper_target = np.clip(
            self._gripper_target,
            self._gripper_ctrl_range[0],
            self._gripper_ctrl_range[1],
        )

        # Set control inputs
        for i, aid in enumerate(self._arm_actuator_ids):
            self.data.ctrl[aid] = self._arm_target[i]
        self.data.ctrl[self._gripper_actuator_id] = self._gripper_target

        # Step physics
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_episode_steps
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self):
        """Lifting-focused dense reward (same as PandaVLAEnv v3)."""
        if self._red_block_id is None:
            return 0.0

        block_pos = self.data.xpos[self._red_block_id].copy()
        hand_pos = self._get_hand_pos()

        hand_block_dist = np.linalg.norm(hand_pos - block_pos)
        block_target_dist = np.linalg.norm(block_pos - self._target_pos)
        block_z = block_pos[2]
        table_z = 0.22

        gripper_open = self.data.qpos[self._finger_qpos_adrs].mean() > 0.02
        lift_height = max(0, block_z - table_z)

        reward = 0.0

        # 1. Hand progress reward
        if self._prev_hand_block_dist is not None:
            progress = self._prev_hand_block_dist - hand_block_dist
            reward += 3.0 * progress

        # 2. Lifting progress reward
        if self._prev_block_height is not None:
            height_progress = lift_height - max(0, self._prev_block_height - table_z)
            reward += 20.0 * height_progress

        # 3. Small proximity bonus
        if hand_block_dist < 0.05:
            reward += 0.05
        elif hand_block_dist < 0.10:
            reward += 0.02

        # 4. Small grasp bonus
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

        # 7. Action penalty
        if self._last_action is not None:
            action_penalty = -0.005 * np.sum(np.square(self._last_action))
            reward += action_penalty

        self._prev_hand_block_dist = hand_block_dist
        self._prev_block_height = block_z

        return float(np.clip(reward, -1.0, 15.0))

    def _check_terminated(self):
        for i, adr in enumerate(self._arm_qpos_adrs):
            q = self.data.qpos[adr]
            lo, hi = self._arm_joint_ranges[i]
            if q < lo - 0.1 or q > hi + 0.1:
                return True
        return False

    def _get_obs(self):
        image = self._render_image()
        joint_positions = self.data.qpos[self._arm_qpos_adrs].copy().astype(np.float32)
        gripper_opening = np.array([self.data.qpos[self._finger_qpos_adrs].mean()], dtype=np.float32)
        return {
            "image": image,
            "joint_positions": joint_positions,
            "gripper": gripper_opening,
        }

    def _get_info(self):
        info = {
            "joint_positions": self.data.qpos[self._arm_qpos_adrs].copy(),
            "joint_velocities": self.data.qvel[self._arm_qpos_adrs].copy(),
            "gripper_opening": self.data.qpos[self._finger_qpos_adrs].mean(),
            "step_count": self.step_count,
        }
        if self._red_block_id is not None:
            block_pos = self.data.xpos[self._red_block_id].copy()
            info["block_position"] = block_pos
            info["block_height"] = block_pos[2]
            hand_pos = self._get_hand_pos()
            info["hand_position"] = hand_pos
            info["hand_block_distance"] = np.linalg.norm(hand_pos - block_pos)
            info["block_target_distance"] = np.linalg.norm(block_pos - self._target_pos)
        return info

    def _render_image(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_image()
        return None

    def close(self):
        if self._renderer is not None:
            del self._renderer
            self._renderer = None
