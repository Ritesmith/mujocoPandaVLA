#!/usr/bin/env python3
"""Evaluate SAC+HER model on PandaGoalEnv (Dict observation)."""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
from gym_env.goal_env import PandaGoalEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

MODEL_PATH = "/home/w/vla_workspace/outputs/rl_v2/sac_her/sac_her_final.zip"
N_EPISODES = 10
SUCCESS_HEIGHT = 0.27   # block_height > 0.27
SUCCESS_DIST = 0.1      # block_target_distance < 0.1


def make_env():
    return PandaGoalEnv()


def evaluate():
    env = DummyVecEnv([make_env])
    model = SAC.load(MODEL_PATH, env=env)
    print(f"Loaded SAC+HER from {MODEL_PATH}")

    episode_rewards = []
    successes = []

    for ep in range(N_EPISODES):
        obs = env.reset()
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = dones[0]

            if done:
                info = infos[0]
                block_height = info.get('block_height', 0)
                block_target_dist = info.get('block_target_distance', 1.0)
                success = block_height > SUCCESS_HEIGHT and block_target_dist < SUCCESS_DIST
                successes.append(success)
                tag = "SUCCESS" if success else "FAIL"
                print(
                    f"Episode {ep+1}: reward={total_reward:.2f}, "
                    f"steps={steps}, height={block_height:.3f}, "
                    f"dist={block_target_dist:.3f}, {tag}"
                )

        episode_rewards.append(total_reward)

    mean_reward = np.mean(episode_rewards)
    success_rate = np.mean(successes)

    print(f"\n{'='*60}")
    print(f"SAC+HER Evaluation Results ({N_EPISODES} episodes)")
    print(f"{'='*60}")
    print(f"Mean reward:  {mean_reward:.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"{'='*60}")

    env.close()
    return mean_reward, success_rate


if __name__ == '__main__':
    evaluate()
