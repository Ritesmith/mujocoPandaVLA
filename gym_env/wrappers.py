import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PBRSShapingWrapper(gym.Wrapper):
    """Potential-Based Reward Shaping wrapper.

    Implements the policy-invariant shaping of Ng, Harada & Russell (1999):
        R'(s, a, s') = R(s, a, s') + gamma * Phi(s') - Phi(s)

    The shaping term telescopes over any trajectory, so it cannot change
    the optimal policy set. Two implementation invariants are critical:

    1. Phi must be a pure function of state (no action, no history).
    2. Phi(terminal) = 0 — enforced here on `terminated=True`. For
       `truncated=True` (timeout), Phi(s_T) is kept, because the
       trajectory did not actually reach a terminal state and the
       telescoping must continue into the next episode's bootstrap.

    The wrapper stores raw_reward, shaping_reward and potential in info
    so the trainer can monitor PBRS health (shaping should -> 0 as the
    policy converges; raw_reward should track true task progress).
    """

    def __init__(self, env, potential_fn, gamma=0.99, shaping_scale=1.0):
        super().__init__(env)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.shaping_scale = shaping_scale
        self._prev_potential = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = float(self.potential_fn(obs, info))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            # Terminal-state potential must be 0 (Wiewiora et al. 2003).
            current_potential = 0.0
        else:
            current_potential = float(self.potential_fn(obs, info))

        shaping = self.gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential

        # Apply scale: PBRS preserves policy invariance for ANY bounded Φ,
        # so scaling Φ (equivalently, scaling the shaping term) does not
        # change the optimal policy set. Scaling is purely a learning-
        # dynamics adjustment: at scale=1 the shaping signal is too weak
        # to influence the gradient relative to the raw reward, so we
        # multiply by `shaping_scale` to bring it into a useful range.
        scaled_shaping = self.shaping_scale * shaping

        # Diagnostics: keep raw_reward separate so the trainer can verify
        # that the policy is making real task progress, not just chasing
        # shaping. shaping_reward logs the SCALED value actually added to
        # the reward so the trainer can compare it directly to raw_reward.
        info = dict(info)
        info["raw_reward"] = float(reward)
        info["shaping_reward"] = float(scaled_shaping)
        info["potential"] = float(current_potential)
        info["unscaled_shaping"] = float(shaping)

        return obs, reward + scaled_shaping, terminated, truncated, info


def placement_potential(obs, info, alpha=1.0, beta=2.0):
    """Multi-scale potential function for the place task.

    Phi(s) = -(alpha * d(block, target) + beta * lift_deficit)

    where lift_deficit = max(0, target_z - block_z) penalises the block
    being below the target height. Both terms are non-positive, so Phi
    is bounded above by 0 and increases monotonically as the block
    approaches the target both horizontally and vertically.

    The wrapper enforces Phi(terminal)=0 separately, so this function
    only needs to handle non-terminal states.

    Args:
        obs: observation from the env (Dict for vision mode, Box for
            FlattenObs). When called from inside the wrapper, `obs` is
            the post-observation-wrapper output, but we always read
            ground-truth state from `info` (more reliable than parsing
            the flattened vector).
        info: info dict from _get_info(); must contain block_position,
            target_position, block_height.
        alpha: weight on horizontal distance to target.
        beta: weight on vertical lift deficit.
    """
    block_pos = info.get("block_position", None)
    target_pos = info.get("target_position", None)
    block_h = info.get("block_height", None)

    if block_pos is None or target_pos is None or block_h is None:
        return 0.0

    block_pos = np.asarray(block_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    # Horizontal distance (xy-plane) — placing happens at table level,
    # so horizontal proximity is the dominant placement signal.
    d_xy = np.linalg.norm(block_pos[:2] - target_pos[:2])

    # Vertical lift deficit: how far the block is below the target's z.
    # For place_mode, target_z is typically 0.22 (table). If the block
    # is lifted above target, deficit=0 (no further reward to gain from
    # height alone). If below, we encourage lifting.
    lift_deficit = max(0.0, float(target_pos[2]) - float(block_h))

    phi = -(alpha * float(d_xy) + beta * lift_deficit)

    # Hard bound for numerical stability. With alpha=1, beta=2 and
    # d_xy <= ~1m, lift_deficit <= ~0.2m, the natural range is [-1.4, 0].
    # The clip is a safety net only.
    return float(np.clip(phi, -10.0, 0.0))


class VisionObs(gym.ObservationWrapper):
    """Vision observation wrapper: outputs Dict {"image": (84,84,3), "state": (12,)}.

    image: RGB rendered from third_person camera, downsampled to 84x84
    state: joint_positions (7) + gripper (1) + block_target_distance (1)
           + target_position (3) = 12-dim
    Drops: block_pos, hand_pos (must be inferred from image)
    Keeps: target_position and block_target_distance (not visible in image,
           required for the policy to know where to place the block)
    """

    def __init__(self, env, image_size=84):
        super().__init__(env)
        self.image_size = image_size
        obs_dict = env.observation_space
        joint_dim = obs_dict["joint_positions"].shape[0]  # 7
        gripper_dim = obs_dict["gripper"].shape[0]  # 1
        # block_target_distance (1) + target_position (3) = 4
        state_dim = joint_dim + gripper_dim + 4  # 12

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32),
        })

    def observation(self, obs):
        # Downsample image from 256x256 to 84x84
        image = obs["image"]
        if image.shape[0] != self.image_size:
            # Simple stride-based downsampling
            step = image.shape[0] // self.image_size
            image = image[::step, ::step, :][:self.image_size, :self.image_size, :]
        image = image.astype(np.uint8)

        # State: joint_positions + gripper + block_target_distance + target_position
        # target_position is NOT visible in the image (it's a coordinate in
        # space, not a rendered object), so it must be in the state vector.
        # Without it, the policy cannot know where to place the block.
        info = self.unwrapped._get_info()
        block_target_dist = np.array([info.get("block_target_distance", 0.0)])
        target_pos = info.get("target_position", np.array([0.5, 0.3, 0.2]))

        state = np.concatenate([
            obs["joint_positions"],  # (7,)
            obs["gripper"],          # (1,)
            block_target_dist,       # (1,)
            target_pos,              # (3,)
        ]).astype(np.float32)

        return {"image": image, "state": state}


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
