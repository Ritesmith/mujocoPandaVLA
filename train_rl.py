#!/usr/bin/env python3
"""Train RL policy on PandaVLAEnv using Stable-Baselines3.

Usage:
    python train_rl.py --algorithm ppo --total_timesteps 100000
    python train_rl.py --algorithm sac --total_timesteps 100000
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import functools
import gymnasium
import numpy as np
import torch
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3 import PPO, SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class CurriculumCallback(BaseCallback):
    """Switch reward type at a given timestep."""
    def __init__(self, switch_step, new_reward_type, verbose=0):
        super().__init__(verbose)
        self.switch_step = switch_step
        self.new_reward_type = new_reward_type
        self._switched = False

    def _on_step(self):
        if not self._switched and self.num_timesteps >= self.switch_step:
            # Access the underlying env through VecNormalize
            env = self.training_env
            try:
                # VecNormalize -> DummyVecEnv -> FlattenObs -> PandaVLAEnv
                base_env = env.envs[0]
                if hasattr(base_env, 'env'):
                    base_env = base_env.env
                if hasattr(base_env, 'unwrapped'):
                    base_env = base_env.unwrapped
                base_env.reward_type = self.new_reward_type
                print(f"\n[Curriculum] Switched reward_type to '{self.new_reward_type}' at step {self.num_timesteps}")
            except Exception as e:
                print(f"[Curriculum] Failed to switch: {e}")
            self._switched = True
        return True


def make_env(reward_type='dense', cartesian=False):
    """Create and wrap the environment."""
    if cartesian:
        env = gymnasium.make('PandaCartesian-v0', reward_type=reward_type)
        from gym_env.wrappers import FlattenObsCartesian
        env = FlattenObsCartesian(env)
    else:
        env = gymnasium.make('PandaVLA-v0', reward_type=reward_type)
        env = FlattenObs(env)
    return env


VLA_MODEL_PATHS = {
    'smolvla_base': '/home/w/vla_workspace/models/smolvla_base',
    'smolvla_finetuned': '/home/w/vla_workspace/models/smolvla_finetuned',
}


def vla_bc_pretrain(model, vla_init_name, n_episodes=5, max_steps=100):
    """Pretrain PPO policy with behavioral cloning from VLA demonstrations.

    Collects (obs, action) pairs by running the VLA model in the environment,
    then trains the PPO policy network to imitate those actions via MSE loss.
    """
    vla_model_path = VLA_MODEL_PATHS.get(vla_init_name, vla_init_name)
    print(f"[VLA Init] Collecting VLA demonstrations from {vla_model_path}...")

    # Create VLA-enabled env
    env = gymnasium.make('PandaVLA-v0', vla_enabled=True, vla_model_path=vla_model_path)
    wrapped_env = FlattenObs(env)

    # Collect (obs, action) pairs
    obs_list = []
    action_list = []

    for ep in range(n_episodes):
        obs, info = wrapped_env.reset()
        for step in range(max_steps):
            # Get VLA action (access through unwrapped since gymnasium wraps the env)
            raw_env = env.unwrapped
            vla_action = raw_env.vla_predict()
            env_action = raw_env._vla_action_to_env_action(vla_action)
            env_action = np.clip(env_action, -1.0, 1.0)

            obs_list.append(obs.copy())
            action_list.append(env_action.copy())

            next_obs, _, terminated, truncated, _ = wrapped_env.step(env_action)
            obs = next_obs
            if terminated or truncated:
                break

    wrapped_env.close()

    if len(obs_list) == 0:
        print("[VLA Init] No demos collected, skipping BC pretraining")
        return model

    print(f"[VLA Init] Collected {len(obs_list)} transitions, running BC pretraining...")

    # BC pretraining: minimize MSE between policy mean action and VLA actions
    obs_array = np.array(obs_list, dtype=np.float32)
    action_array = np.array(action_list, dtype=np.float32)

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)

    for epoch in range(50):
        indices = np.random.permutation(len(obs_array))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), 64):
            batch_idx = indices[i:i + 64]
            obs_batch = torch.tensor(obs_array[batch_idx], device=model.device, dtype=torch.float32)
            action_batch = torch.tensor(action_array[batch_idx], device=model.device, dtype=torch.float32)

            # Get mean actions from policy: extract_features -> mlp_extractor -> action_net
            features = model.policy.extract_features(obs_batch)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            mean_actions = model.policy.action_net(latent_pi)

            loss = torch.nn.functional.mse_loss(mean_actions, action_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch % 10 == 0:
            print(f"  BC Epoch {epoch}: loss = {total_loss / n_batches:.4f}")

    print(f"[VLA Init] BC pretraining complete!")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'sac', 'sac_her'])
    parser.add_argument('--total_timesteps', type=int, default=100000)
    parser.add_argument('--save_path', type=str, default='/home/w/vla_workspace/outputs/rl_v3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--curriculum', type=str, default='none', choices=['none', 'reach_then_pick'])
    parser.add_argument('--vla_init', type=str, default='none',
                        help='VLA model for BC pretraining (smolvla_base, smolvla_finetuned, or none)')
    parser.add_argument('--cartesian', action='store_true',
                        help='Use Cartesian action space (4D: dx,dy,dz,gripper) instead of joint space (8D)')
    args = parser.parse_args()

    algo_name = args.algorithm
    # Determine directory name based on algorithm + curriculum + cartesian combination
    if algo_name == 'ppo':
        if args.curriculum != 'none':
            dir_name = 'ppo_curriculum'
        else:
            dir_name = 'ppo_dense'
    elif algo_name == 'sac_her':
        dir_name = 'sac_her'
    else:
        dir_name = algo_name  # sac
    if args.cartesian:
        dir_name = dir_name + '_cartesian'
    save_dir = os.path.join(args.save_path, dir_name)
    log_dir = os.path.join(args.save_path, 'tb_logs', dir_name)
    os.makedirs(save_dir, exist_ok=True)

    # SAC+HER uses goal-conditioned env (Dict obs, no VecNormalize)
    if algo_name == 'sac_her':
        from gym_env.goal_env import PandaGoalEnv

        def make_goal_env():
            return PandaGoalEnv()

        train_env = DummyVecEnv([make_goal_env for _ in range(1)])
        eval_env = DummyVecEnv([make_goal_env])
    else:
        # Create vectorized env
        initial_reward_type = 'curriculum_reach' if args.curriculum == 'reach_then_pick' else 'dense'
        train_env = DummyVecEnv([functools.partial(make_env, initial_reward_type, args.cartesian) for _ in range(1)])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        # Eval env (separate instance, NOT reward-normalized)
        eval_env = DummyVecEnv([functools.partial(make_env, 'dense', args.cartesian)])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Create model
    if algo_name == 'ppo':
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=log_dir,
            seed=args.seed,
            device='cuda',
        )
    elif algo_name == 'sac':
        model = SAC(
            "MlpPolicy",
            train_env,
            batch_size=256,
            learning_rate=3e-4,
            buffer_size=50000,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            verbose=1,
            tensorboard_log=log_dir,
            seed=args.seed,
            device='cuda',
        )
    elif algo_name == 'sac_her':
        model = SAC(
            "MultiInputPolicy",
            train_env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            batch_size=256,
            learning_rate=3e-4,
            buffer_size=50000,
            learning_starts=500,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            verbose=1,
            tensorboard_log=log_dir,
            seed=args.seed,
            device='cuda',
        )

    # VLA BC pretraining (PPO only)
    if args.vla_init != 'none':
        if algo_name != 'ppo':
            print(f"[VLA Init] WARNING: --vla_init is only supported for PPO, skipping for {algo_name}")
        else:
            model = vla_bc_pretrain(model, args.vla_init)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, 'best'),
        log_path=os.path.join(save_dir, 'eval_logs'),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=os.path.join(save_dir, 'checkpoints'),
        name_prefix=f'{algo_name}_model',
    )

    callbacks = [eval_callback, checkpoint_callback]

    if args.curriculum == 'reach_then_pick':
        switch_step = int(args.total_timesteps * 0.4)
        print(f"[Curriculum] Starting with reward_type='curriculum_reach', will switch to 'dense' at step {switch_step}")
        curriculum_callback = CurriculumCallback(switch_step, 'dense')
        callbacks.append(curriculum_callback)

    # Train
    print(f"\n{'='*60}")
    print(f"Training {algo_name.upper()} for {args.total_timesteps} steps")
    print(f"Save dir: {save_dir}")
    print(f"TensorBoard: {log_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model and normalization stats
    model.save(os.path.join(save_dir, f'{algo_name}_final'))
    if algo_name != 'sac_her':
        train_env.save(os.path.join(save_dir, 'vec_normalize.pkl'))

    print(f"\nTraining complete! Model saved to {save_dir}")
    print(f"To view TensorBoard: tensorboard --logdir {os.path.join(args.save_path, 'tb_logs')}")


if __name__ == '__main__':
    main()
