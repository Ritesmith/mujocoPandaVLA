#!/usr/bin/env python3
"""Train PPO on ManiSkill3 PickCube-v1 with GPU parallel simulation.

Uses ManiSkillWrapper to adapt ManiSkill3's observation/action format
to match PandaVLAEnv (16D obs, 8D action), then runs SB3 PPO.

Expected speedup: 15-30x vs MuJoCo CPU single-env.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


def make_maniskill_env(num_envs=1, render_mode="rgb_array"):
    """Create a ManiSkill3 env wrapped with ManiSkillWrapper."""
    from maniskill_wrapper import ManiSkillWrapper

    env = ManiSkillWrapper(
        env_id="PickCube-v1",
        num_envs=num_envs,
        render_mode=render_mode,
    )
    return env


def make_vec_env(num_envs=4):
    """Create a vectorized ManiSkill3 env for SB3."""
    def _env_fn():
        return make_maniskill_env(num_envs=1)

    vec_env = DummyVecEnv([_env_fn for _ in range(num_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def main():
    parser = argparse.ArgumentParser(description="Train PPO on ManiSkill3 PickCube")
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="outputs/maniskill_ppo")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(f"{args.save_path}/best", exist_ok=True)

    print(f"Creating {args.num_envs} ManiSkill3 envs...")
    vec_env = make_vec_env(num_envs=args.num_envs)
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Eval env (single, no normalization)
    eval_env = DummyVecEnv([lambda: make_maniskill_env(num_envs=1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Copy normalization stats from training env
    eval_env.obs_rms = vec_env.obs_rms
    eval_env.ret_rms = vec_env.ret_rms
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.95,
        ent_coef=0.01,
        verbose=1,
        device=args.device,
        tensorboard_log=f"{args.save_path}/tb_logs",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=10000,
        save_path=args.save_path,
        name_prefix="maniskill_ppo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.save_path}/best",
        log_path=f"{args.save_path}/eval_logs",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training PPO on ManiSkill3 for {args.total_timesteps} steps...")
    start_time = time.time()
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )
    elapsed = time.time() - start_time

    model.save(f"{args.save_path}/maniskill_ppo_final.zip")
    vec_env.save(f"{args.save_path}/vec_normalize.pkl")
    vec_env.close()
    eval_env.close()

    fps = args.total_timesteps / elapsed
    print(f"\nTraining complete in {elapsed:.1f}s ({fps:.0f} FPS)")
    print(f"Model saved to {args.save_path}/maniskill_ppo_final.zip")


if __name__ == "__main__":
    main()
