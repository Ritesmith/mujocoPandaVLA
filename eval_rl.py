#!/usr/bin/env python3
"""Evaluate trained RL policy on PandaVLAEnv.

Usage:
    python eval_rl.py --model_path outputs/rl/ppo/best/best_model.zip --n_episodes 10
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import numpy as np
import gymnasium
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


TABLE_HEIGHT = 0.22


def make_env():
    env = gymnasium.make('PandaVLA-v0')
    env = FlattenObs(env)
    return env


def evaluate(model_path, vec_normalize_path, n_episodes=10):
    # Create env
    env = DummyVecEnv([make_env])

    # Load normalization stats
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        env.training = False
        env.norm_reward = False

    # Load model
    model = None
    for cls in [PPO, SAC]:
        try:
            model = cls.load(model_path, env=env)
            print(f"Loaded {cls.__name__} from {model_path}")
            break
        except Exception:
            continue

    if model is None:
        raise ValueError(f"Could not load model from {model_path}")

    # Evaluate
    episode_rewards = []
    episode_lengths = []
    successes = []

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_reward += reward[0]
            steps += 1
            done = dones[0]

            if done and infos and len(infos) > 0:
                info = infos[0]
                block_height = info.get('block_height', 0)
                block_target_dist = info.get('block_target_distance', 1.0)
                success = block_height > TABLE_HEIGHT + 0.05 and block_target_dist < 0.1
                successes.append(success)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_str = "SUCCESS" if successes[-1] else "FAIL"
        print(f"Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, {success_str}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Mean reward:  {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean length:  {np.mean(episode_lengths):.1f}")
    print(f"Success rate: {np.mean(successes)*100:.1f}%")
    print(f"{'='*60}")

    env.close()
    return np.mean(episode_rewards), np.mean(successes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vec_normalize_path', type=str, default=None)
    parser.add_argument('--n_episodes', type=int, default=10)
    args = parser.parse_args()

    # Auto-detect vec_normalize path
    if args.vec_normalize_path is None:
        model_dir = os.path.dirname(args.model_path)
        vnorm_path = os.path.join(model_dir, '..', 'vec_normalize.pkl')
        if os.path.exists(vnorm_path):
            args.vec_normalize_path = vnorm_path

    evaluate(args.model_path, args.vec_normalize_path, args.n_episodes)


if __name__ == '__main__':
    main()
