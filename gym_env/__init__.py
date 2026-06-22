"""Panda VLA Gymnasium Environment."""
import gymnasium

gymnasium.register(
    id="PandaVLA-v0",
    entry_point="gym_env.panda_vla_env:PandaVLAEnv",
)

gymnasium.register(
    id="PandaGoal-v0",
    entry_point="gym_env.goal_env:PandaGoalEnv",
    max_episode_steps=500,
)

gymnasium.register(
    id="PandaCartesian-v0",
    entry_point="gym_env.panda_cartesian_env:PandaCartesianEnv",
)

gymnasium.register(
    id='PandaVLA-Rand-v0',
    entry_point='gym_env.panda_vla_env:PandaVLAEnv',
    kwargs={'domain_randomize': True, 'gravity_comp': True},
    max_episode_steps=500,
)

from gym_env.panda_vla_env import PandaVLAEnv
from gym_env.goal_env import PandaGoalEnv
from gym_env.panda_cartesian_env import PandaCartesianEnv

__all__ = ["PandaVLAEnv", "PandaGoalEnv", "PandaCartesianEnv"]
