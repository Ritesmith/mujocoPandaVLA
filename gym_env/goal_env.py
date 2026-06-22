"""Goal-Conditioned PandaVLAEnv for HER training."""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gym_env.panda_vla_env import PandaVLAEnv


class PandaGoalEnv(gym.Env):
    """Goal-Conditioned wrapper around PandaVLAEnv for HER.

    Since gymnasium 1.3.0 does not have GoalEnv, we implement the
    required interface manually:
    - observation_space = Dict({"observation", "achieved_goal", "desired_goal"})
    - compute_reward(achieved_goal, desired_goal, info) method
    """

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(self, target_pos=None, **kwargs):
        super().__init__()

        # Set default target
        self._target_pos = target_pos if target_pos is not None else np.array([0.5, 0.3, 0.2], dtype=np.float32)

        # Create inner env
        self._env = PandaVLAEnv(reward_type='dense', **kwargs)

        # Observation space: joint_positions(7) + gripper(1) + block_pos(3) + hand_pos(3) = 14
        obs_dim = 14
        goal_dim = 3  # block target position

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(goal_dim,), dtype=np.float32),
        })

        self.action_space = self._env.action_space

    def _get_obs(self):
        """Get goal-conditioned observation."""
        info = self._env._get_info()
        joint_pos = self._env.data.qpos[self._env._arm_qpos_adrs].copy()
        gripper = np.array([self._env.data.qpos[self._env._finger_qpos_adrs].mean()])
        block_pos = info.get("block_position", np.zeros(3))
        hand_pos = info.get("hand_position", np.zeros(3))

        observation = np.concatenate([joint_pos, gripper, block_pos, hand_pos]).astype(np.float32)
        achieved_goal = block_pos.astype(np.float32)
        desired_goal = self._target_pos.astype(np.float32)

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Dense reward for HER: continuous distance-based reward."""
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        # Continuous: -1 at max distance, 0 at goal
        # Max distance ~0.5m, so reward = -d/0.5, clipped to [-1, 0]
        reward = -np.minimum(d / 0.5, 1.0).astype(np.float32)
        # Bonus for being very close
        if np.isscalar(d):
            if d < 0.05:
                reward = 0.0  # At goal
        else:
            reward = np.where(d < 0.05, 0.0, reward)
        return reward

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        goal_obs = self._get_obs()
        # Override reward with HER-compatible dense reward
        reward = self.compute_reward(goal_obs["achieved_goal"], goal_obs["desired_goal"], info)
        return goal_obs, float(reward), terminated, truncated, info

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped
