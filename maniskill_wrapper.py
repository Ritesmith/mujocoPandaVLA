"""ManiSkill3 environment adapter for PickCube-v1.

Wraps ManiSkill3's PickCube-v1 to match the existing PandaVLAEnv interface so
that downstream code (FlattenObs-style 16D observations, 8D actions, dense
pick-place reward) works unchanged.

Key bridging:
- Observation: 16D flat vector
    [joint_pos(7), gripper(1), block_pos(3), hand_pos(3),
     hand_block_dist(1), block_target_dist(1)]
- Action: 8D  [joint_delta(7), gripper_cmd(1)]  # gripper_cmd: -1=open, +1=close
- ManiSkill3 PickCube-v1 default action space is 8D (7 arm + 1 gripper via a
  MimicController). The wrapper also supports a 9D layout (7 arm + 2 gripper)
  should the controller be configured that way.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym

try:
    import torch
except ImportError:  # pragma: no cover - torch is a ManiSkill3 dependency
    torch = None


class ManiSkillWrapper(gym.Wrapper):
    """Wraps ManiSkill3 PickCube-v1 to match PandaVLAEnv interface.

    - Observation: 16D flat vector (same as FlattenObs(PandaVLAEnv))
    - Action: 8D (7 joint deltas + 1 gripper cmd)
    - Converts 8D action -> ManiSkill action (8D mimic or 9D two-finger)
    - Extracts 16D observation from ManiSkill dict obs
    """

    # Panda arm joint absolute limits (matches FlattenObs in gym_env/wrappers.py)
    _JOINT_LIMITS = np.array(
        [2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973],
        dtype=np.float32,
    )

    def __init__(self, env_id="PickCube-v1", num_envs=1, render_mode="rgb_array"):
        import mani_skill  # noqa: F401  registers ManiSkill envs with gymnasium

        self._num_envs = num_envs
        self._render_mode = render_mode

        # Create ManiSkill3 env with state_dict obs for easy component extraction.
        # pd_joint_delta_pos makes the arm action a joint-space delta, matching the
        # PandaVLAEnv convention where actions[:7] are interpreted as deltas.
        env = gym.make(
            env_id,
            num_envs=num_envs,
            obs_mode="state_dict",
            render_mode=render_mode,
            control_mode="pd_joint_delta_pos",
        )
        super().__init__(env)

        # Detect underlying ManiSkill action dim (8 for mimic, 9 for two-finger).
        self._ms_action_dim = int(np.prod(self.env.action_space.shape))

        # 8D action space: 7 joint deltas + 1 gripper cmd (-1=open, +1=close)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # 16D observation space matching FlattenObs(PandaVLAEnv).
        low = np.full(16, -np.inf, dtype=np.float32)
        high = np.full(16, np.inf, dtype=np.float32)
        low[:7] = -self._JOINT_LIMITS
        high[:7] = self._JOINT_LIMITS
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Reward / termination bookkeeping (mirrors PandaVLAEnv._compute_reward_dense)
        self._step_count = 0
        self._max_episode_steps = getattr(
            self.env.unwrapped, "max_episode_steps", 500
        )
        if self._max_episode_steps is None:
            self._max_episode_steps = 500
        self._prev_hand_block_dist = None
        self._prev_block_height = None
        self._initial_block_z = None
        self._last_action = None

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._step_count = 0
        self._prev_hand_block_dist = None
        self._prev_block_height = None
        self._last_action = None

        comps = self._get_components(obs)
        self._initial_block_z = float(comps["block_pos"][2])

        flat_obs = self._build_flat_obs(comps)
        out_info = self._build_info(obs, info, comps)
        return flat_obs, out_info

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._last_action = action.copy()

        ms_action = self._convert_action(action)
        obs, ms_reward, ms_terminated, ms_truncated, info = self.env.step(ms_action)

        self._step_count += 1
        comps = self._get_components(obs)

        flat_obs = self._build_flat_obs(comps)
        reward = self._compute_reward(comps)
        terminated = self._check_terminated(comps, ms_terminated)
        truncated = bool(self._to_scalar(ms_truncated)) or (
            self._step_count >= self._max_episode_steps
        )
        out_info = self._build_info(obs, info, comps)

        # Update previous-state trackers used by the dense reward.
        self._prev_hand_block_dist = float(comps["hand_block_dist"])
        self._prev_block_height = float(comps["block_pos"][2])

        return flat_obs, reward, terminated, truncated, out_info

    def render(self):
        img = self.env.render()
        return self._to_numpy(img)

    # ------------------------------------------------------------------ #
    # Observation extraction
    # ------------------------------------------------------------------ #
    def _extract_obs(self, obs):
        """Convert ManiSkill obs dict to 16D flat vector."""
        comps = self._get_components(obs)
        return self._build_flat_obs(comps)

    def _get_components(self, obs):
        """Extract scalar state components from a ManiSkill state_dict obs."""
        qpos = self._to_numpy(obs["agent"]["qpos"])  # (9,) or (N,9)
        joint_pos = qpos[..., :7].astype(np.float32)

        # Gripper opening: mean of the two finger joints (0=closed, 0.04=open),
        # matching PandaVLAEnv's gripper observation.
        fingers = qpos[..., 7:9]
        gripper = np.mean(fingers, axis=-1, keepdims=True).astype(np.float32)

        obj_pose = self._to_numpy(obs["extra"]["obj_pose"])  # (7,) pos+quat
        block_pos = obj_pose[..., :3].astype(np.float32)

        tcp_pose = self._to_numpy(obs["extra"]["tcp_pose"])  # (7,) pos+quat
        hand_pos = tcp_pose[..., :3].astype(np.float32)

        goal_pos = self._to_numpy(obs["extra"]["goal_pos"])[..., :3].astype(np.float32)

        hand_block_dist = np.linalg.norm(hand_pos - block_pos, axis=-1)
        block_target_dist = np.linalg.norm(block_pos - goal_pos, axis=-1)

        return {
            "joint_pos": joint_pos,
            "gripper": gripper,
            "block_pos": block_pos,
            "hand_pos": hand_pos,
            "goal_pos": goal_pos,
            "hand_block_dist": hand_block_dist,
            "block_target_dist": block_target_dist,
        }

    def _build_flat_obs(self, comps):
        """Assemble the 16D observation vector from extracted components."""
        joint_pos = comps["joint_pos"]
        gripper = comps["gripper"]
        block_pos = comps["block_pos"]
        hand_pos = comps["hand_pos"]
        hand_block_dist = np.atleast_1d(comps["hand_block_dist"]).astype(np.float32)
        block_target_dist = np.atleast_1d(comps["block_target_dist"]).astype(np.float32)

        flat = np.concatenate(
            [joint_pos, gripper, block_pos, hand_pos, hand_block_dist, block_target_dist]
        ).astype(np.float32)
        return flat

    # ------------------------------------------------------------------ #
    # Action conversion
    # ------------------------------------------------------------------ #
    def _convert_action(self, action):
        """Convert 8D action to ManiSkill action.

        Input action: [joint_delta(7), gripper_cmd(1)]
            gripper_cmd: -1 = open, +1 = close

        ManiSkill 8D (mimic): [joint_delta(7), gripper(1)]
            gripper: -1 = close, +1 = open  (opposite sign of gripper_cmd)
        ManiSkill 9D (two-finger): [joint_delta(7), gripper_left(1), gripper_right(1)]
            gripper values: 0 = open, 1 = close
        """
        joint_delta = action[:7]
        gripper_cmd = float(action[7])  # -1=open, +1=close

        if self._ms_action_dim == 8:
            # Mimic controller: negate so +1 cmd (close) -> -1 (close target).
            ms_gripper = -gripper_cmd
            ms_action = np.concatenate([joint_delta, [ms_gripper]])
        elif self._ms_action_dim == 9:
            # Two-finger: map -1..+1 (open..close) to 0..1 (open..close).
            g = (gripper_cmd + 1.0) / 2.0
            ms_action = np.concatenate([joint_delta, [g, g]])
        else:
            # Fallback: pass through the first ms_action_dim dims.
            ms_action = np.concatenate(
                [joint_delta, np.zeros(self._ms_action_dim - 7, dtype=np.float32)]
            )

        ms_action = ms_action.astype(np.float32)

        # ManiSkill3 expects torch tensors on the env's device.
        if torch is not None:
            device = self.env.unwrapped.device
            ms_action = torch.as_tensor(ms_action, device=device, dtype=torch.float32)
            if self._num_envs > 1:
                ms_action = ms_action.unsqueeze(0).repeat(self._num_envs, 1)
        return ms_action

    # ------------------------------------------------------------------ #
    # Reward & termination
    # ------------------------------------------------------------------ #
    def _compute_reward(self, comps):
        """Dense pick-place reward matching PandaVLAEnv._compute_reward_dense."""
        block_pos = comps["block_pos"]
        hand_pos = comps["hand_pos"]
        hand_block_dist = float(comps["hand_block_dist"])
        block_target_dist = float(comps["block_target_dist"])
        gripper_opening = float(np.ravel(comps["gripper"])[0])
        gripper_open = gripper_opening > 0.02

        block_z = float(block_pos[2])
        table_z = self._initial_block_z if self._initial_block_z is not None else 0.0
        lift_height = max(0.0, block_z - table_z)

        reward = 0.0

        # 1. Hand progress reward (exploration signal)
        if self._prev_hand_block_dist is not None:
            progress = self._prev_hand_block_dist - hand_block_dist
            reward += 3.0 * progress

        # 2. Lifting progress reward (PRIMARY signal)
        if self._prev_block_height is not None:
            prev_lift = max(0.0, self._prev_block_height - table_z)
            height_progress = lift_height - prev_lift
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

        # 5. Lifting bonus (per-step reward for holding block up)
        if lift_height > 0.02:
            reward += 1.0
        if lift_height > 0.05:
            reward += 2.0
        if lift_height > 0.10:
            reward += 3.0

        # 6. Placing bonus (only when block is lifted)
        if lift_height > 0.03:
            if block_target_dist < 0.05:
                reward += 5.0
            elif block_target_dist < 0.10:
                reward += 2.0

        # 7. Small action penalty
        if self._last_action is not None:
            reward += -0.005 * float(np.sum(np.square(self._last_action)))

        return float(np.clip(reward, -1.0, 15.0))

    def _check_terminated(self, comps, ms_terminated):
        """Check termination: ManiSkill success + joint-limit safety check."""
        if bool(self._to_scalar(ms_terminated)):
            return True
        joint_pos = comps["joint_pos"]
        # Joint-limit safety check (mirrors PandaVLAEnv._check_terminated).
        if np.any(np.abs(joint_pos) > self._JOINT_LIMITS + 0.1):
            return True
        return False

    # ------------------------------------------------------------------ #
    # Info assembly
    # ------------------------------------------------------------------ #
    def _build_info(self, obs, ms_info, comps):
        """Build an info dict compatible with PandaVLAEnv._get_info."""
        is_grasped = False
        success = False
        if "extra" in obs and "is_grasped" in obs["extra"]:
            is_grasped = bool(self._to_scalar(obs["extra"]["is_grasped"]))
        if isinstance(ms_info, dict):
            if "success" in ms_info:
                success = bool(self._to_scalar(ms_info["success"]))

        info = {
            "joint_positions": comps["joint_pos"].copy(),
            "gripper_opening": float(np.ravel(comps["gripper"])[0]),
            "block_position": comps["block_pos"].copy(),
            "block_height": float(comps["block_pos"][2]),
            "hand_position": comps["hand_pos"].copy(),
            "hand_block_distance": float(comps["hand_block_dist"]),
            "block_target_distance": float(comps["block_target_dist"]),
            "goal_position": comps["goal_pos"].copy(),
            "is_grasped": is_grasped,
            "success": success,
            "step_count": self._step_count,
        }
        return info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _to_numpy(self, x):
        """Convert torch/numpy to numpy, squeezing the batch dim for single env."""
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if self._num_envs == 1 and x.ndim >= 1 and x.shape[0] == 1:
            x = x[0]
        return x

    def _to_scalar(self, x):
        """Convert a 0/1-d torch/numpy value to a python scalar."""
        if torch is not None and isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 0:
            return x.item()
        return x.reshape(-1)[0].item()


if __name__ == "__main__":
    # Test the wrapper
    env = ManiSkillWrapper()
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}")
    assert obs.shape == (16,), f"Expected (16,), got {obs.shape}"
    assert env.action_space.shape == (8,), f"Expected (8,), got {env.action_space.shape}"
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            obs, info = env.reset()
    # Verify render works
    img = env.render()
    assert img is not None and img.size > 0, "render() returned empty image"
    print(f"Render image shape: {img.shape}")
    env.close()
    print("ManiSkillWrapper test: OK")
