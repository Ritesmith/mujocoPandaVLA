import os

# Set EGL for headless rendering BEFORE importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData


class PandaVLAEnv(gym.Env):
    """Gymnasium environment for MuJoCo Panda robot with VLA integration.

    Observation: RGB image (H, W, 3) + joint positions (7,) + gripper (1,)
    Action: 7 joint position deltas + 1 gripper = 8-dim continuous

    The Panda MJCF model uses position actuators (kp-based PD control).
    Actions are interpreted as position deltas (desired velocity * dt),
    which are added to the current position target and clipped to joint limits.

    This follows the standard Gymnasium API:
    - reset() -> observation, info
    - step(action) -> observation, reward, terminated, truncated, info
    - render() -> RGB image
    - close()

    VLA integration:
    - When vla_enabled=True, loads SmolVLA model for closed-loop control.
    - vla_predict() runs inference on current observation.
    - vla_step() runs inference and executes the predicted action.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, image_size=256, control_dt=0.05,
                 model_path=None, max_episode_steps=500,
                 vla_enabled=False, vla_model_path=None,
                 task_instruction="pick up the red block",
                 reward_type="dense", target_pos=None,
                 gravity_comp=True, domain_randomize=False,
                 place_mode=False, place_mode_realistic=False,
                 grasp_states=None, target_pos_range=None):
        super().__init__()

        # Model path
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
        self.gravity_comp = gravity_comp
        self.domain_randomize = domain_randomize
        self.place_mode = place_mode
        self.place_mode_realistic = place_mode_realistic
        self._grasp_states = grasp_states  # collected grasp states for realistic init

        # VLA integration
        self.vla_enabled = vla_enabled
        self.task_instruction = task_instruction
        self._vla_policy = None
        self._vla_preprocess = None
        self._vla_postprocess = None
        self._vla_img_key = None
        self._vla_state_dim = None

        if vla_enabled:
            self._load_vla_model(vla_model_path)

        # Load MuJoCo model
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.model.opt.timestep = 0.002  # 2ms physics timestep
        self.n_substeps = int(control_dt / self.model.opt.timestep)

        # Discover joint/actuator indices from the model
        # Panda has 7 arm joints + 2 finger joints (coupled via tendon)
        # Actuators: 7 position actuators for arm + 1 tendon actuator for gripper
        self.n_arm_joints = 7
        self.n_finger_joints = 2  # finger_joint1, finger_joint2
        self.n_actuators = self.model.nu  # should be 8

        # Find joint qpos indices by name
        self._arm_joint_ids = []
        self._arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        for name in self._arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self._arm_joint_ids.append(jid)

        # Find finger joint ids
        self._finger_joint_ids = []
        for name in ["finger_joint1", "finger_joint2"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found in model")
            self._finger_joint_ids.append(jid)

        # Map joint id -> qpos index
        self._arm_qpos_adrs = [self.model.jnt_qposadr[jid] for jid in self._arm_joint_ids]
        self._finger_qpos_adrs = [self.model.jnt_qposadr[jid] for jid in self._finger_joint_ids]

        # Find actuator ids
        self._arm_actuator_ids = []
        for name in self._arm_joint_names:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid == -1:
                raise ValueError(f"Actuator '{name}' not found in model")
            self._arm_actuator_ids.append(aid)

        # Gripper actuator (tendon "split")
        self._gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "finger_joint1"
        )
        if self._gripper_actuator_id == -1:
            # Try by index - last actuator
            self._gripper_actuator_id = self.n_actuators - 1

        # Joint limits for clipping
        self._arm_joint_ranges = np.zeros((self.n_arm_joints, 2))
        for i, jid in enumerate(self._arm_joint_ids):
            self._arm_joint_ranges[i, 0] = self.model.jnt_range[jid, 0]
            self._arm_joint_ranges[i, 1] = self.model.jnt_range[jid, 1]

        # Gripper range (ctrl for tendon actuator)
        self._gripper_ctrl_range = self.model.actuator_ctrlrange[self._gripper_actuator_id].copy()

        # Action space: 7 joint position deltas + 1 gripper command
        self.n_action = self.n_arm_joints + 1  # 8
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_action,), dtype=np.float32
        )

        # Observation space: image + joint positions + gripper opening
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

        # Renderer (created once, reused)
        self._renderer = None

        # Current position targets for incremental control
        self._arm_target = np.zeros(self.n_arm_joints)
        self._gripper_target = 0.04  # start open

        # Reward-related parameters
        self.reward_type = reward_type
        self._red_block_id = None
        self._hand_id = None
        self._left_finger_id = None
        self._right_finger_id = None
        self._place_gravcomp_active = False
        self._target_pos = target_pos if target_pos is not None else np.array([0.5, 0.3, 0.2])
        # target_pos_range: [[x_low, y_low, z_low], [x_high, y_high, z_high]]
        # When set, _target_pos is randomized within this range on each reset.
        self._target_pos_range = target_pos_range
        self._prev_hand_block_dist = None
        self._prev_block_height = None
        self._prev_block_target_dist = None
        self._place_approach_bonus_given = False
        self._place_proximity_15_given = False
        self._place_proximity_10_given = False
        self._place_success = False
        self._find_bodies()

        # Save red_block freejoint qpos address for reset
        self._red_block_qpos_adr = None
        self._red_block_dof_adr = None
        self._red_block_mass = 0.0
        if self._red_block_id is not None:
            try:
                rj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'red_block_joint')
                if rj_id >= 0:
                    self._red_block_qpos_adr = self.model.jnt_qposadr[rj_id]
                    self._red_block_dof_adr = self.model.jnt_dofadr[rj_id]
                    self._red_block_mass = self.model.body_mass[self._red_block_id]
            except Exception:
                pass

        # Save default physics properties for domain randomization restore
        self._default_geom_friction = self.model.geom_friction.copy()
        self._default_body_mass = self.model.body_mass.copy()
        self._default_body_inertia = self.model.body_inertia.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Restore default physics properties before applying randomization
        self._restore_default_physics()

        # Randomize target position if target_pos_range is set
        if self._target_pos_range is not None:
            low = np.array(self._target_pos_range[0])
            high = np.array(self._target_pos_range[1])
            self._target_pos = self.np_random.uniform(low, high)
            # Keep z at table height (0.22) for placing
            self._target_pos[2] = 0.22

        # Set to home keyframe
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            self.data.qpos[:] = self.model.key_qpos[home_key_id]
            self.data.ctrl[:] = self.model.key_ctrl[home_key_id]
        else:
            # Default home position
            self.data.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
            self.data.qpos[7:9] = [0.04, 0.04]
            self.data.ctrl[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
            self.data.ctrl[7] = 0.04

        # Restore red_block freejoint position (home keyframe may reset it to origin)
        if self._red_block_qpos_adr is not None:
            self.data.qpos[self._red_block_qpos_adr + 0] = 0.5   # x
            self.data.qpos[self._red_block_qpos_adr + 1] = 0.0   # y
            self.data.qpos[self._red_block_qpos_adr + 2] = 0.24  # z (on table: table_top=0.22 + block_half=0.02)
            self.data.qpos[self._red_block_qpos_adr + 3] = 1.0   # qw
            self.data.qpos[self._red_block_qpos_adr + 4] = 0.0   # qx
            self.data.qpos[self._red_block_qpos_adr + 5] = 0.0   # qy
            self.data.qpos[self._red_block_qpos_adr + 6] = 0.0   # qz

        # Domain randomization
        if self.domain_randomize:
            # Randomize block position
            block_x = self.np_random.uniform(0.35, 0.65)
            block_y = self.np_random.uniform(-0.15, 0.15)
            self.data.qpos[self._red_block_qpos_adr + 0] = block_x
            self.data.qpos[self._red_block_qpos_adr + 1] = block_y
            # z stays at 0.24 (on table)

            # Randomize block friction (geom id for red_block_geom)
            block_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "red_block_geom")
            if block_geom_id >= 0:
                base_friction = np.array([1.0, 0.5, 0.005])
                friction_scale = self.np_random.uniform(0.5, 2.0)
                self.model.geom_friction[block_geom_id] = base_friction * friction_scale

            # Randomize fingertip friction
            for pad_name in [f"lf_pad{i}" for i in range(1,6)] + [f"rf_pad{i}" for i in range(1,6)]:
                pad_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, pad_name)
                if pad_geom_id >= 0:
                    base_friction = np.array([2.0, 0.5, 0.005])
                    friction_scale = self.np_random.uniform(0.5, 2.0)
                    self.model.geom_friction[pad_geom_id] = base_friction * friction_scale

            # Randomize block mass
            block_body_id = self._red_block_id
            if block_body_id is not None and block_body_id >= 0:
                new_mass = self.np_random.uniform(0.02, 0.10)
                self.model.body_mass[block_body_id] = new_mass
                # Update inertia: for a box with side 2*s, I = m*(2s)^2/12 * diag(1,1,1)
                s = 0.02  # half-size
                I = new_mass * (2*s)**2 / 12.0
                self.model.body_inertia[block_body_id] = np.array([I, I, I])

        # Add small random perturbation to initial position
        if self.np_random is not None:
            self.data.qpos[self._arm_qpos_adrs] += self.np_random.uniform(
                -0.05, 0.05, self.n_arm_joints
            )

        # Place mode: start from a pre-grasped, lifted state for training/eval
        # the placing sub-policy independently of the grasping policy.
        if self.place_mode:
            if self._grasp_states and len(self._grasp_states) > 0:
                # Use a collected grasp state as the starting arm
                # configuration. This bridges the train-eval mismatch:
                # during hierarchical eval, place_mode activates
                # mid-episode when the grasp policy has already
                # positioned the arm. Training from these realistic
                # states ensures the place policy sees the same arm
                # distribution as eval.
                state = self._grasp_states[self.np_random.integers(
                    len(self._grasp_states)
                )]
                lifted_qpos = np.array(state['arm_qpos'])
                grasp_finger_pos = float(
                    np.array(state['finger_qpos']).mean()
                )
                # Noise matched to the std of 167 collected grasp states
                # (500-episode collection). Joints 4-6 have high variance
                # (0.65, 0.41, 0.69) because the wrist orientation varies
                # significantly across grasp attempts.
                arm_noise = np.array([0.05, 0.04, 0.10, 0.10, 0.65, 0.41, 0.69])
                lifted_qpos = lifted_qpos + self.np_random.uniform(
                    -arm_noise, arm_noise
                )
                # Clamp finger to valid range
                grasp_finger_pos = float(np.clip(
                    grasp_finger_pos + self.np_random.uniform(-0.005, 0.005),
                    0.0, 0.04
                ))
            else:
                # Fallback: lifted arm configuration with randomization
                lifted_qpos = np.array(
                    [0.5, 0.3, 0.0, -1.57079, 0.0, 1.57079, 0.7854]
                )
                arm_noise = np.array([0.3, 0.2, 0.2, 0.3, 0.2, 0.3, 0.2])
                lifted_qpos = lifted_qpos + self.np_random.uniform(
                    -arm_noise, arm_noise
                )
                grasp_finger_pos = 0.02

            self.data.qpos[self._arm_qpos_adrs] = lifted_qpos
            self.data.ctrl[self._arm_actuator_ids] = lifted_qpos

            # Set gripper to match block width (40mm block → 0.02 per finger).
            self.data.qpos[self._finger_qpos_adrs] = [grasp_finger_pos, grasp_finger_pos]
            self.data.ctrl[self._gripper_actuator_id] = grasp_finger_pos

            # Forward to get actual hand position after arm config
            mujoco.mj_forward(self.model, self.data)

            # Place block at hand position (between fingers).
            # With gravity compensation enabled, the block stays at this
            # position without needing physical finger contact.
            if self._red_block_qpos_adr is not None and self._hand_id is not None:
                hand_pos = self.data.xpos[self._hand_id].copy()
                # Offset slightly below hand to be at finger height
                grasp_pos = hand_pos.copy()
                grasp_pos[2] -= 0.05  # ~5cm below hand = finger center
                self.data.qpos[self._red_block_qpos_adr + 0] = grasp_pos[0]
                self.data.qpos[self._red_block_qpos_adr + 1] = grasp_pos[1]
                self.data.qpos[self._red_block_qpos_adr + 2] = grasp_pos[2]
                self.data.qpos[self._red_block_qpos_adr + 3] = 1.0   # qw
                self.data.qpos[self._red_block_qpos_adr + 4] = 0.0   # qx
                self.data.qpos[self._red_block_qpos_adr + 5] = 0.0   # qy
                self.data.qpos[self._red_block_qpos_adr + 6] = 0.0   # qz

            # Enable gravity compensation for the block in place_mode.
            # We use qfrc_applied in step() to counteract gravity on the
            # block while the gripper is "holding" it. When the policy
            # opens the gripper, the counter force is removed so the
            # block falls realistically.
            self._place_gravcomp_active = True

        # Place mode realistic: start from a pre-grasped, lifted state but
        # rely on real gripper friction to hold the block (no hard
        # attachment, no gravity compensation). This trains the place
        # policy under the same physics as evaluation.
        elif self.place_mode_realistic:
            # Lifted arm configuration (same as place_mode)
            lifted_qpos = np.array(
                [0.5, 0.3, 0.0, -1.57079, 0.0, 1.57079, 0.7854]
            )
            self.data.qpos[self._arm_qpos_adrs] = lifted_qpos
            self.data.ctrl[self._arm_actuator_ids] = lifted_qpos

            # Close gripper to hold block.
            # finger_joint=0.02 gives ~23mm pad inner gap (too tight, ejects block).
            # finger_joint=0.025 gives ~37mm pad inner gap (slightly < 40mm block,
            # providing gentle grip without excessive force).
            grasp_finger_pos = 0.025
            self.data.qpos[self._finger_qpos_adrs] = [grasp_finger_pos, grasp_finger_pos]
            self.data.ctrl[self._gripper_actuator_id] = grasp_finger_pos

            # Forward to get actual pad positions after arm config
            mujoco.mj_forward(self.model, self.data)

            # Place block at pad center (between lf_pad1 and rf_pad1).
            # Pad center z (~0.35m) differs from finger center z (~0.39m)
            # and hand z (~0.45m). Using pad center ensures the block is
            # within the pad contact range.
            if (self._red_block_qpos_adr is not None
                    and self._lf_pad1_gid >= 0
                    and self._rf_pad1_gid >= 0):
                lf_pad_pos = self.data.geom_xpos[self._lf_pad1_gid].copy()
                rf_pad_pos = self.data.geom_xpos[self._rf_pad1_gid].copy()
                grasp_pos = (lf_pad_pos + rf_pad_pos) / 2.0
                self.data.qpos[self._red_block_qpos_adr + 0] = grasp_pos[0]
                self.data.qpos[self._red_block_qpos_adr + 1] = grasp_pos[1]
                self.data.qpos[self._red_block_qpos_adr + 2] = grasp_pos[2]
                self.data.qpos[self._red_block_qpos_adr + 3] = 1.0   # qw
                self.data.qpos[self._red_block_qpos_adr + 4] = 0.0   # qx
                self.data.qpos[self._red_block_qpos_adr + 5] = 0.0   # qy
                self.data.qpos[self._red_block_qpos_adr + 6] = 0.0   # qz
                # Zero out block velocity to prevent drift
                if self._red_block_dof_adr is not None:
                    self.data.qvel[self._red_block_dof_adr:self._red_block_dof_adr + 6] = 0

            # Run a few physics steps to let contacts settle so the
            # gripper actually grips the block before the policy acts.
            # Fix arm position during settle to prevent contact forces
            # from pushing the arm away from the lifted configuration.
            for _ in range(50):
                if self.gravity_comp:
                    mujoco.mj_forward(self.model, self.data)
                    self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
                # Re-pin arm joints to prevent drift from block contact
                self.data.qpos[self._arm_qpos_adrs] = lifted_qpos
                self.data.qvel[:7] = 0
                mujoco.mj_step(self.model, self.data)
                if self.gravity_comp:
                    self.data.qfrc_applied[:] = 0

            # No hard attachment — block position is determined by physics.
            # Use gravity compensation for the block (upward force) so it
            # doesn't fall due to insufficient gripper friction. The block
            # still moves with the gripper via contact forces, and falls
            # when the gripper opens (gravcomp disabled). This is more
            # realistic than place_mode's hard position attachment but
            # still trainable.
            self._place_gravcomp_active = True

        mujoco.mj_forward(self.model, self.data)

        # Initialize position targets from current state
        self._arm_target = self.data.qpos[self._arm_qpos_adrs].copy()
        self._gripper_target = self.data.ctrl[self._gripper_actuator_id]
        self.step_count = 0
        self._prev_hand_block_dist = None
        self._prev_block_height = None
        self._prev_block_target_dist = None
        self._place_approach_bonus_given = False
        self._place_proximity_15_given = False
        self._place_proximity_10_given = False
        self._place_success = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._last_action = action.copy()

        # Arm: interpret action as velocity -> position delta
        # max velocity ~1 rad/s, scaled by control_dt
        arm_delta = action[:self.n_arm_joints] * 1.0 * self.control_dt  # max 1 rad/s
        self._arm_target += arm_delta
        # Clip to joint limits
        self._arm_target = np.clip(
            self._arm_target,
            self._arm_joint_ranges[:, 0],
            self._arm_joint_ranges[:, 1],
        )

        # Gripper: action[-1] > 0 -> close, < 0 -> open
        # ctrl range is [0, 0.04]: 0=closed, 0.04=open
        # Distance-gated: only allow closing when hand is near the block.
        # This prevents the policy from closing the gripper prematurely
        # during approach (which left the gripper too narrow ~15mm to
        # fit around the 40mm block).
        gripper_cmd = action[self.n_arm_joints]
        if self._hand_id is not None and self._red_block_id is not None:
            hand_pos = self.data.xpos[self._hand_id]
            block_pos = self.data.xpos[self._red_block_id]
            hand_block_dist = float(np.linalg.norm(hand_pos - block_pos))
        else:
            hand_block_dist = float("inf")

        if self.place_mode:
            # In place_mode, block is held by gravity compensation force.
            # gripper_cmd > 0: close (keep holding), < 0: open (release),
            # == 0: hold current position (don't drift).
            # Release is gated: only allow opening when the block is near
            # the target to prevent premature release that caused 0% place
            # rate in v2/v3. The threshold is configurable via
            # _release_dist_threshold (default 0.10m). Tightening to 0.05m
            # in eval forces the model to navigate closer before releasing,
            # reducing post-release drift (the main failure mode at 58%
            # place rate: block gets to 2-3cm, overshoots to 7-8cm, then
            # releases and drifts to 5-9cm).
            _release_thresh = getattr(self, '_release_dist_threshold', 0.10)
            _release_height = getattr(self, '_release_height_threshold', float('inf'))
            if self._red_block_id is not None:
                _bp = self.data.xpos[self._red_block_id]
                block_target_dist = float(np.linalg.norm(_bp - self._target_pos))
                block_height_above_table = float(_bp[2]) - 0.22
            else:
                block_target_dist = float("inf")
                block_height_above_table = float("inf")
            # Release gated on both horizontal distance to target AND
            # block height above table. The height gate forces the model
            # to lower the block near the table before releasing,
            # reducing post-release drift (block falls 12-14cm and
            # drifts 2-5cm when released from high up).
            if (gripper_cmd < 0
                    and block_target_dist < _release_thresh
                    and block_height_above_table < _release_height):
                self._gripper_target = min(0.04, self._gripper_target + 0.02)
                # Opening gripper → remove gravity compensation so block falls
                self._place_gravcomp_active = False
            elif gripper_cmd > 0:
                self._gripper_target = max(0.0, self._gripper_target - 0.005)
            # else: keep current target (no change)
        elif self.place_mode_realistic:
            # In place_mode_realistic, block is held by gravity compensation
            # (upward force) + gripper contact force, not hard-attached.
            # gripper_cmd > 0: close (keep holding), < 0: open (release).
            # No distance gating — block is already in the gripper.
            if gripper_cmd < 0:
                self._gripper_target = min(0.04, self._gripper_target + 0.02)
                # Opening gripper → remove gravity compensation so block falls
                self._place_gravcomp_active = False
            elif gripper_cmd > 0:
                self._gripper_target = max(0.0, self._gripper_target - 0.005)
            # else: keep current target (no change)
        else:
            # Normal grasping mode: distance-gated gripper control.
            # Only allow closing when hand is near the block to prevent
            # premature closing during approach. When the block is lifted
            # (in gripper), gripper_cmd=0 holds to prevent dropping.
            # Release is gated: only allow opening when block is near the
            # target (dist < 0.10m) to prevent premature release that
            # caused 0% place rate in hierarchical evaluation.
            block_lifted = (self._red_block_id is not None
                            and self.data.xpos[self._red_block_id][2] > 0.25)
            if gripper_cmd > 0 and hand_block_dist < 0.06:
                self._gripper_target = max(0.0, self._gripper_target - 0.02)
            elif gripper_cmd < 0:
                if not block_lifted:
                    # Block on table: allow opening
                    self._gripper_target = min(0.04, self._gripper_target + 0.02)
                elif self._red_block_id is not None:
                    # Block lifted: only allow opening near target
                    _bd = float(np.linalg.norm(
                        self.data.xpos[self._red_block_id] - self._target_pos))
                    if _bd < 0.10:
                        self._gripper_target = min(0.04, self._gripper_target + 0.02)
                    # else: ignore open command (keep holding)
            elif not block_lifted:
                # Block on table: gripper_cmd=0 opens (for approach)
                self._gripper_target = min(0.04, self._gripper_target + 0.02)
            # else: block lifted and gripper_cmd=0 -> hold
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
            if self.gravity_comp:
                mujoco.mj_forward(self.model, self.data)
                self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
            # place_mode_realistic: apply upward force to block to counteract gravity
            if (self.place_mode_realistic and self._place_gravcomp_active
                    and self._red_block_dof_adr is not None
                    and self._red_block_mass > 0):
                gravity_force = self._red_block_mass * 9.81
                self.data.qfrc_applied[self._red_block_dof_adr + 2] = gravity_force
            mujoco.mj_step(self.model, self.data)
            if self.gravity_comp:
                self.data.qfrc_applied[:] = 0
            # In place_mode, "attach" block to hand while gripper is holding.
            # Directly set block position to follow hand (hard attachment).
            # This is more reliable than force-based gravity compensation.
            if (self.place_mode and self._place_gravcomp_active
                    and self._red_block_qpos_adr is not None
                    and self._hand_id is not None):
                hand_pos = self.data.xpos[self._hand_id].copy()
                self.data.qpos[self._red_block_qpos_adr + 0] = hand_pos[0]
                self.data.qpos[self._red_block_qpos_adr + 1] = hand_pos[1]
                self.data.qpos[self._red_block_qpos_adr + 2] = hand_pos[2] - 0.05
                # Zero out block velocity to prevent drift
                if self._red_block_dof_adr is not None:
                    self.data.qvel[self._red_block_dof_adr:self._red_block_dof_adr + 6] = 0

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self.step_count >= self.max_episode_steps
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def snap_block_to_hand(self):
        """Immediately attach block to hand position (hand_pos - 5cm z).

        Called when place_mode activates mid-episode during hierarchical
        eval, so the first place-policy observation reflects the block
        at its trained position rather than wherever it was during the
        grasp phase. Also calls mj_forward to update derived quantities
        (xpos) used by _get_obs/_get_info.
        """
        if (self._red_block_qpos_adr is not None
                and self._hand_id is not None):
            hand_pos = self.data.xpos[self._hand_id].copy()
            self.data.qpos[self._red_block_qpos_adr + 0] = hand_pos[0]
            self.data.qpos[self._red_block_qpos_adr + 1] = hand_pos[1]
            self.data.qpos[self._red_block_qpos_adr + 2] = hand_pos[2] - 0.05
            if self._red_block_dof_adr is not None:
                self.data.qvel[
                    self._red_block_dof_adr:self._red_block_dof_adr + 6
                ] = 0
            mujoco.mj_forward(self.model, self.data)

    def _load_vla_model(self, model_path=None):
        """Load SmolVLA model for inference."""
        import sys
        sys.path.insert(0, '/home/w/vla_workspace/lerobot/src')
        from lerobot.policies.smolvla import SmolVLAPolicy
        from lerobot.policies import make_pre_post_processors

        if model_path is None:
            model_path = "/home/w/vla_workspace/models/smolvla_base"

        print(f"Loading VLA model from {model_path}...")
        self._vla_policy = SmolVLAPolicy.from_pretrained(model_path)
        self._vla_policy.eval()

        # Load pre/post processors
        self._vla_preprocess, self._vla_postprocess = make_pre_post_processors(
            self._vla_policy.config, model_path
        )

        # Discover image key and state dim from config
        self._vla_img_key = list(self._vla_policy.config.image_features.keys())[0]
        self._vla_state_dim = self._vla_policy.config.input_features["observation.state"].shape[0]

        print(f"VLA model loaded. Image key: {self._vla_img_key}, State dim: {self._vla_state_dim}")

    def vla_predict(self, task=None):
        """Run VLA inference on current observation, return action vector."""
        import torch
        from torchvision.transforms import ToTensor

        if self._vla_policy is None:
            raise RuntimeError("VLA model not loaded. Set vla_enabled=True to use vla_predict().")

        task = task or self.task_instruction

        # Get current image and convert to tensor
        image = self._render_image()
        pil_image = __import__("PIL").Image.fromarray(image)
        to_tensor = ToTensor()
        img_tensor = to_tensor(pil_image)  # [C, H, W] in [0, 1]

        # Get current state: map 7 arm joints + 1 gripper -> state_dim
        joint_pos = self.data.qpos[self._arm_qpos_adrs].copy()
        gripper = self.data.qpos[self._finger_qpos_adrs].mean()
        full_state = np.concatenate([joint_pos, [gripper]])
        state = np.zeros(self._vla_state_dim, dtype=np.float32)
        n_copy = min(len(full_state), self._vla_state_dim)
        state[:n_copy] = full_state[:n_copy]
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Build observation dict - replicate single camera to all image keys
        obs = {}
        for key in self._vla_policy.config.image_features.keys():
            obs[key] = img_tensor
        obs["observation.state"] = state_tensor
        obs["task"] = task

        # Preprocess
        processed = self._vla_preprocess(obs)

        # Run inference
        with torch.no_grad():
            action_chunk = self._vla_policy.predict_action_chunk(processed)

        # Postprocess (denormalize)
        action_final = self._vla_postprocess(action_chunk)

        # Extract first action from chunk: shape is (batch, n_action_steps, action_dim)
        action = action_final[0, 0, :].cpu().numpy()
        return action

    def vla_step(self, task=None):
        """Run one VLA inference step and execute the predicted action."""
        action = self.vla_predict(task)
        env_action = self._vla_action_to_env_action(action)
        return self.step(env_action)

    def _vla_action_to_env_action(self, vla_action):
        """Convert VLA action (6D) to env action (8D: 7 joint deltas + 1 gripper).

        SmolVLA outputs 6D actions. We map the first 6 dims to the first 6
        joint velocity deltas, leave joint 7 unchanged, and set gripper based
        on the last action dimension.
        """
        action = np.zeros(self.n_action, dtype=np.float32)
        # Map first 6 VLA action dims to first 6 arm joint deltas
        n_dims = min(len(vla_action), self.n_arm_joints)
        action[:n_dims] = np.array(vla_action[:n_dims], dtype=np.float32)
        # Use last VLA dim as gripper command if available
        if len(vla_action) > self.n_arm_joints:
            action[self.n_arm_joints] = vla_action[self.n_arm_joints - 1]
        # Clip to action space
        action = np.clip(action, -1.0, 1.0)
        return action

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_image()
        return None

    def close(self):
        if self._renderer is not None:
            del self._renderer
            self._renderer = None
        # Clean up VLA model
        if self._vla_policy is not None:
            import torch
            del self._vla_policy
            del self._vla_preprocess
            del self._vla_postprocess
            self._vla_policy = None
            self._vla_preprocess = None
            self._vla_postprocess = None
            torch.cuda.empty_cache()

    def _get_obs(self):
        image = self._render_image()
        joint_positions = self.data.qpos[self._arm_qpos_adrs].copy().astype(np.float32)
        # Gripper opening: average of two finger positions
        gripper_opening = np.array(
            [self.data.qpos[self._finger_qpos_adrs].mean()],
            dtype=np.float32,
        )

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
            if self._hand_id is not None:
                hand_pos = self.data.xpos[self._hand_id].copy()
                info["hand_position"] = hand_pos
                info["hand_block_distance"] = np.linalg.norm(hand_pos - block_pos)
            info["block_target_distance"] = np.linalg.norm(block_pos - self._target_pos)
            info["target_position"] = self._target_pos.copy()

        return info

    def _render_image(self):
        """Render RGB image using MuJoCo offscreen renderer (EGL)."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.image_size, width=self.image_size
            )
        self._renderer.update_scene(self.data)
        image = self._renderer.render()
        return image

    def _restore_default_physics(self):
        """Restore default physics properties (friction, mass, inertia)."""
        self.model.geom_friction[:] = self._default_geom_friction
        self.model.body_mass[:] = self._default_body_mass
        self.model.body_inertia[:] = self._default_body_inertia

    def _find_bodies(self):
        """Find body IDs for reward computation."""
        try:
            self._red_block_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "red_block"
            )
        except Exception:
            self._red_block_id = None
        try:
            self._hand_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
            )
        except Exception:
            self._hand_id = None
        try:
            self._left_finger_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
            )
        except Exception:
            self._left_finger_id = None
        try:
            self._right_finger_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
            )
        except Exception:
            self._right_finger_id = None
        # Pad geom IDs for place_mode_realistic block placement
        try:
            self._lf_pad1_gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "lf_pad1"
            )
        except Exception:
            self._lf_pad1_gid = -1
        try:
            self._rf_pad1_gid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "rf_pad1"
            )
        except Exception:
            self._rf_pad1_gid = -1

    def _compute_reward(self):
        if self.reward_type == "dense":
            return self._compute_reward_dense()
        elif self.reward_type == "curriculum_reach":
            return self._compute_reward_curriculum_reach()
        elif self.reward_type == "place_only":
            return self._compute_reward_place()
        else:  # "pick_place" (old)
            return self._compute_reward_pick_place()

    def _compute_reward_dense(self):
        """Dense pick-place reward v6: lifting-first, gentle placing.

        v5 showed that v4's placing reward (5.0/step + 5.0/m progress)
        still competed with lifting, dropping grab rate 65%->35%.
        v6 makes placing reward MUCH weaker and gates it higher:
        - Placing bonus only when lift > 0.10m (not 0.05m)
        - Placing weights reduced 5x (1.0/step max, 1.0/m progress)
        - Lifting bonus unchanged (6.0/step max) — stays dominant
        """
        if self._red_block_id is None:
            return 0.0

        block_pos = self.data.xpos[self._red_block_id].copy()
        hand_pos = self.data.xpos[self._hand_id].copy() if self._hand_id is not None else np.zeros(3)

        hand_block_dist = np.linalg.norm(hand_pos - block_pos)
        block_target_dist = np.linalg.norm(block_pos - self._target_pos)
        block_z = block_pos[2]
        table_z = 0.22

        gripper_open = self.data.qpos[self._finger_qpos_adrs].mean() > 0.02
        lift_height = max(0, block_z - table_z)

        reward = 0.0

        # 1. Hand progress reward (exploration signal)
        if self._prev_hand_block_dist is not None:
            progress = self._prev_hand_block_dist - hand_block_dist
            reward += 3.0 * progress  # +3.0 per meter toward block

        # 2. Lifting progress reward (PRIMARY SIGNAL)
        #    Reward the block moving upward — this is what we really want
        if self._prev_block_height is not None:
            height_progress = lift_height - max(0, self._prev_block_height - table_z)
            reward += 20.0 * height_progress  # +20.0 per meter of lifting

        # 3. Small proximity bonus (not dominant)
        if hand_block_dist < 0.05:
            reward += 0.05  # tiny per-step bonus
        elif hand_block_dist < 0.10:
            reward += 0.02

        # 4. Small grasp bonus (not dominant)
        block_in_hand = hand_block_dist < 0.05 and not gripper_open
        if block_in_hand:
            reward += 0.1  # tiny per-step bonus

        # 5. LIFTING BONUS — the key reward that makes lifting worthwhile
        #    v10: back to v3's original 6.0/step (v9's 12.0 caused over-lifting
        #    where block flew away). Use shorter fine-tuning to prevent decay.
        if lift_height > 0.02:  # Block slightly off table
            reward += 1.0  # +1.0/step for holding block up
        if lift_height > 0.05:  # Block clearly lifted
            reward += 2.0  # +2.0/step extra (total 3.0/step)
        if lift_height > 0.10:  # Block well above table
            reward += 3.0  # +3.0/step extra (total 6.0/step)

        # 6. Placing bonus — ONLY when block is well lifted (lift > 10cm)
        #    v6: reduced 5x vs v4, gated higher (0.10m vs 0.05m) so it
        #    never competes with the lifting phase.
        if lift_height > 0.10:
            if block_target_dist < 0.05:
                reward += 1.0  # +1.0/step for precise placing while lifted
            elif block_target_dist < 0.10:
                reward += 0.5  # +0.5/step for placing near target
            elif block_target_dist < 0.20:
                reward += 0.1  # +0.1/step for approaching target

        # 6b. Gentle placing progress — ONLY when block is well lifted
        #     v6: reduced 5x vs v4 (1.0/m vs 5.0/m).
        if self._prev_block_target_dist is not None and lift_height > 0.10:
            placing_progress = self._prev_block_target_dist - block_target_dist
            reward += 1.0 * placing_progress  # +1.0 per meter toward target

        # 7. Small action penalty
        if hasattr(self, '_last_action'):
            action_penalty = -0.005 * np.sum(np.square(self._last_action))
            reward += action_penalty

        # Update previous values for next step
        self._prev_hand_block_dist = hand_block_dist
        self._prev_block_height = block_z
        self._prev_block_target_dist = block_target_dist

        return float(np.clip(reward, -1.0, 15.0))

    def _compute_reward_place(self):
        """Place-only reward v5 for the placing sub-policy.

        v9 fix: v8's per-step proximity bonus (+5/step when dist<0.10)
        caused reward hacking — the model kept the block at ~5-10cm
        from target for 500 steps (collecting ~2000 reward) without
        placing. One-time bonuses prevent this.

        v9 changes:
        - REPLACED per-step proximity bonus with one-time bonuses:
          +20 when first reaching dist<0.15, +50 when dist<0.10
        - INCREASED approach bonus: +100 when first reaching dist<0.05
        - Kept lowering bonus, release bonus (gated on table), terminal

        Components:
        - Distance penalty: -10.0 * block_target_dist
        - Block progress: +20.0 * (prev_dist - curr_dist)
        - One-time proximity: dist<0.15 -> +20, dist<0.10 -> +50
        - Lowering bonus: +5.0 * (prev_z - curr_z) when descending
        - Approach bonus: dist<0.05 -> +100 (one-time)
        - Release bonus: dist<0.05 AND on table AND gripper open -> +50/step
        - Terminal success: dist<0.05 AND on table AND gripper open -> +200
        """
        if self._red_block_id is None:
            return 0.0

        block_pos = self.data.xpos[self._red_block_id].copy()
        block_target_dist = float(np.linalg.norm(block_pos - self._target_pos))
        block_z = float(block_pos[2])
        table_z = 0.22

        # Use _gripper_target instead of data.qpos for the gripper_open
        # check during EVALUATION. In place_mode, the actual finger
        # position (data.qpos) lags behind the target due to PD
        # controller dynamics. This lag caused _place_success to never
        # be set in hierarchical eval: the place model sends an open
        # command (gcmd < 0), which sets _gripper_target += 0.02, but
        # data.qpos takes several steps to reach 0.02. By the time it
        # does, the block has drifted away from the target.
        #
        # However, during TRAINING, using data.qpos is beneficial: the
        # delay means the episode continues for a few extra steps after
        # the model opens the gripper, accumulating the +50/step release
        # bonus. This provides a strong learning gradient that was
        # present in v10 but lost when we switched to _gripper_target.
        # v11 with _gripper_target in training failed to learn (rewards
        # stuck at -240 at 130K).
        #
        # Solution: use data.qpos in training (strong gradient from
        # multi-step release bonus) and _gripper_target in eval (correct
        # immediate termination). The _use_gripper_target_check flag is
        # set by eval_hierarchical.py.
        if getattr(self, '_use_gripper_target_check', False):
            gripper_open = self._gripper_target > 0.02
        else:
            gripper_open = self.data.qpos[self._finger_qpos_adrs].mean() > 0.02
        block_on_table = block_z < table_z + 0.03  # within 3cm of table

        reward = 0.0

        # 1. Continuous distance penalty
        reward -= 10.0 * block_target_dist

        # 2. Block progress toward target (velocity signal)
        if self._prev_block_target_dist is not None:
            progress = self._prev_block_target_dist - block_target_dist
            reward += 20.0 * progress

        # 3. One-time proximity bonuses (v9: replaced per-step to prevent hacking)
        if block_target_dist < 0.15 and not self._place_proximity_15_given:
            reward += 20.0
            self._place_proximity_15_given = True
        if block_target_dist < 0.10 and not self._place_proximity_10_given:
            reward += 50.0
            self._place_proximity_10_given = True

        # 4. Lowering bonus
        # Reward descending block toward table — provides Z-axis signal.
        if self._prev_block_height is not None and block_z > table_z + 0.03:
            height_progress = self._prev_block_height - block_z
            if height_progress > 0:
                reward += 5.0 * height_progress

        # 5. One-time approach bonus (v9: increased from +10 to +100)
        if block_target_dist < 0.05 and not self._place_approach_bonus_given:
            reward += 100.0
            self._place_approach_bonus_given = True

        # 6. Release bonus: block at target, ON TABLE, gripper open
        if block_target_dist < 0.05 and block_on_table and gripper_open:
            reward += 50.0

        # 7. Terminal success: block placed on target and on table
        if block_target_dist < 0.05 and block_on_table and gripper_open:
            reward += 200.0
            self._place_success = True

        # Update previous values for next step
        self._prev_block_target_dist = block_target_dist
        self._prev_block_height = block_z

        return float(np.clip(reward, -10.0, 400.0))

    def _compute_reward_curriculum_reach(self):
        """Curriculum stage 1: Only reaching reward."""
        if self._red_block_id is None:
            return 0.0

        block_pos = self.data.xpos[self._red_block_id].copy()
        hand_pos = self.data.xpos[self._hand_id].copy() if self._hand_id is not None else np.zeros(3)
        hand_block_dist = np.linalg.norm(hand_pos - block_pos)

        # Pure reaching reward: 0 to 1
        reward = 1.0 - min(hand_block_dist / 0.5, 1.0)

        # Big bonus for getting very close
        if hand_block_dist < 0.05:
            reward += 1.0

        return float(np.clip(reward, -1.0, 2.0))

    def _compute_reward_pick_place(self):
        """Compute pick-place reward based on object position (legacy progressive)."""
        if self._red_block_id is None:
            return 0.0

        # Get positions
        block_pos = self.data.xpos[self._red_block_id].copy()
        hand_pos = self.data.xpos[self._hand_id].copy() if self._hand_id is not None else np.zeros(3)

        # Block height (z-axis)
        block_z = block_pos[2]
        table_z = 0.22  # table surface height

        # Distance from hand to block
        hand_block_dist = np.linalg.norm(hand_pos - block_pos)

        # Distance from block to target
        block_target_dist = np.linalg.norm(block_pos - self._target_pos)

        # Reward components (normalized to roughly [-1, 1])
        # 1. Reaching reward: closer hand to block is better
        reaching_reward = 1.0 - min(hand_block_dist / 0.5, 1.0)

        # 2. Lifting reward: block above table is good
        lift_height = max(0, block_z - table_z)
        lifting_reward = min(lift_height / 0.2, 1.0)  # 0.2m = max lift for full reward

        # 3. Placing reward: block close to target is good
        placing_reward = 1.0 - min(block_target_dist / 0.5, 1.0)

        # Combined reward with progressive weighting
        # Phase 1: reach (hand close to block)
        # Phase 2: lift (block above table)
        # Phase 3: place (block at target)
        if lift_height < 0.01:  # Block on table - focus on reaching
            reward = 0.3 * reaching_reward
        elif block_target_dist > 0.1:  # Block lifted but not at target
            reward = 0.3 + 0.4 * lifting_reward
        else:  # Block near target
            reward = 0.7 + 0.3 * placing_reward

        # Clip to [-1, 1]
        reward = np.clip(reward, -1.0, 1.0)

        return float(reward)

    def _check_terminated(self):
        # Check if any arm joint exceeds safe limits (beyond model limits)
        for i, adr in enumerate(self._arm_qpos_adrs):
            q = self.data.qpos[adr]
            lo, hi = self._arm_joint_ranges[i]
            if q < lo - 0.1 or q > hi + 0.1:
                return True

        # Place mode: terminate on successful placement
        if self.place_mode and self._place_success:
            return True

        return False
