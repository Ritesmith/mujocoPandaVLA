import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenObs(gym.ObservationWrapper):
    """Flatten Dict observation to a single Box vector.

    Keeps: joint_positions (7,) + gripper (1,) = 8-dim vector
    Drops: image (not usable by MlpPolicy)
    Also appends: block_position (3,) + hand_position (3,) + hand_block_distance (1,)
                   + block_target_distance (1,) = 8 extra dims
    Optionally appends: target_position (3,) when include_target_pos=True
    Total: 16-dim (default) or 19-dim (with target_pos) observation vector
    """

    def __init__(self, env, include_target_pos=False):
        super().__init__(env)
        self.include_target_pos = include_target_pos
        obs_dict = env.observation_space
        joint_dim = obs_dict["joint_positions"].shape[0]  # 7
        gripper_dim = obs_dict["gripper"].shape[0]  # 1

        # Task-relevant info from _get_info():
        # block_position (3) + hand_position (3) + hand_block_distance (1)
        # + block_target_distance (1) = 8
        extra_dim = 8
        if include_target_pos:
            extra_dim += 3  # target_position (3)

        total_dim = joint_dim + gripper_dim + extra_dim
        low = np.full(total_dim, -np.inf, dtype=np.float32)
        high = np.full(total_dim, np.inf, dtype=np.float32)

        # Clip joint positions to known limits (first 7 dims)
        joint_limits = np.array(
            [2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973]
        )
        low[:7] = -joint_limits
        high[:7] = joint_limits

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        info = self.unwrapped._get_info()

        joint_pos = obs["joint_positions"]  # (7,)
        gripper = obs["gripper"]  # (1,)

        block_pos = info.get("block_position", np.zeros(3))
        hand_pos = info.get("hand_position", np.zeros(3))
        hand_block_dist = np.array([info.get("hand_block_distance", 0.0)])
        block_target_dist = np.array([info.get("block_target_distance", 0.0)])

        parts = [joint_pos, gripper, block_pos, hand_pos, hand_block_dist,
                 block_target_dist]

        if self.include_target_pos:
            target_pos = info.get("target_position", np.array([0.5, 0.3, 0.2]))
            parts.append(target_pos)

        flat_obs = np.concatenate(parts).astype(np.float32)

        return flat_obs


class FlattenObsCartesian(gym.ObservationWrapper):
    """Flatten Dict observation for PandaCartesianEnv to a single Box vector.

    Layout: joint_positions (7) + gripper (1) + block_position (3)
            + hand_position (3) + hand_block_distance (1)
            + block_target_distance (1) = 16-dim vector
    Drops: image (not usable by MlpPolicy)
    """

    def __init__(self, env):
        super().__init__(env)
        obs_dict = env.observation_space
        joint_dim = obs_dict["joint_positions"].shape[0]  # 7
        gripper_dim = obs_dict["gripper"].shape[0]  # 1

        # Task-relevant info from _get_info():
        # block_position (3) + hand_position (3) + hand_block_distance (1)
        # + block_target_distance (1) = 8
        extra_dim = 8

        total_dim = joint_dim + gripper_dim + extra_dim
        low = np.full(total_dim, -np.inf, dtype=np.float32)
        high = np.full(total_dim, np.inf, dtype=np.float32)

        # Clip joint positions to known limits (first 7 dims)
        joint_limits = np.array(
            [2.8973, 1.7628, 2.8973, 3.0718, 2.8973, 3.7525, 2.8973]
        )
        low[:7] = -joint_limits
        high[:7] = joint_limits

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        info = self.unwrapped._get_info()

        joint_pos = obs["joint_positions"]  # (7,)
        gripper = obs["gripper"]  # (1,)

        block_pos = info.get("block_position", np.zeros(3))
        hand_pos = info.get("hand_position", np.zeros(3))
        hand_block_dist = np.array([info.get("hand_block_distance", 0.0)])
        block_target_dist = np.array([info.get("block_target_distance", 0.0)])

        flat_obs = np.concatenate(
            [joint_pos, gripper, block_pos, hand_pos, hand_block_dist, block_target_dist]
        ).astype(np.float32)

        return flat_obs
