#!/usr/bin/env python3
"""Demo-Augmented Policy Gradient (DAPG) training for Panda pick-place.

DAPG augments standard policy gradient with a BC regularization term from
demo trajectories, enabling faster and more stable learning:

    L = L_PG + lambda_bc * L_BC

Where:
    L_PG = standard PPO clipped objective
    L_BC = -log_pi(a_demo | s_demo)  (behavioral cloning loss on demo data)

The BC weight decays over training: lambda_bc = lambda_0 * decay^(t/T)

Usage:
    python train_dapg.py --demos demos/panda_pickplace.npz --total_timesteps 200000
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class DAPGCallback(BaseCallback):
    """Inject demo BC loss into PPO training at each gradient step.

    After PPO computes its loss on on-policy data, we add an additional
    BC loss term computed from the demo buffer. The BC weight decays
    exponentially over training.
    """

    def __init__(self, demo_obs, demo_actions, lambda_bc=0.5,
                 bc_decay=0.8, total_timesteps=100000, verbose=0):
        super().__init__(verbose)
        self.demo_obs = torch.tensor(demo_obs, dtype=torch.float32)
        self.demo_actions = torch.tensor(demo_actions, dtype=np.float32).float()
        self.lambda_bc_init = lambda_bc
        self.bc_decay = bc_decay
        self.total_timesteps = total_timesteps
        self.bc_loss_fn = nn.MSELoss()
        self._bc_losses = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        """Called after each rollout collection (before PPO update)."""
        pass  # We inject BC loss during the PPO update phase

    def get_bc_loss(self, batch_size=256):
        """Sample a batch of demo transitions and compute BC loss."""
        indices = np.random.randint(0, len(self.demo_obs), size=batch_size)
        obs_batch = self.demo_obs[indices].to(self.training_env.device)
        action_batch = self.demo_actions[indices].to(self.training_env.device)

        # Get policy mean actions
        with torch.no_grad():
            features = self.model.policy.extract_features(obs_batch)
        latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.model.policy.action_net(latent_pi)

        bc_loss = self.bc_loss_fn(mean_actions, action_batch)
        return bc_loss

    def current_lambda_bc(self):
        """Compute current BC weight with exponential decay."""
        progress = self.num_timesteps / self.total_timesteps
        return self.lambda_bc_init * (self.bc_decay ** (progress * 10))


class DAPGPPO(PPO):
    """PPO with Demo-Augmented Policy Gradient.

    Overrides the train() method to add BC regularization from demo data.
    """

    def __init__(self, *args, demo_obs=None, demo_actions=None,
                 lambda_bc=0.5, bc_decay=0.8, total_timesteps=100000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_obs = demo_obs
        self.demo_actions = demo_actions
        self.lambda_bc = lambda_bc
        self.bc_decay = bc_decay
        self.total_timesteps = total_timesteps
        self._bc_loss_fn = nn.MSELoss()

    def train(self):
        """PPO train with added BC regularization."""
        # Run standard PPO training
        super().train()

        # Add BC loss from demo data
        if self.demo_obs is None or len(self.demo_obs) == 0:
            return

        # Compute BC weight with decay
        progress = self.num_timesteps / max(self.total_timesteps, 1)
        current_lambda = self.lambda_bc * (self.bc_decay ** (progress * 10))

        # Sample demo batch
        batch_size = min(256, len(self.demo_obs))
        indices = np.random.randint(0, len(self.demo_obs), size=batch_size)
        demo_obs_batch = torch.tensor(
            self.demo_obs[indices], dtype=torch.float32, device=self.device
        )
        demo_act_batch = torch.tensor(
            self.demo_actions[indices], dtype=torch.float32, device=self.device
        )

        # Get policy mean actions on demo observations
        features = self.policy.extract_features(demo_obs_batch)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.policy.action_net(latent_pi)

        # BC loss: MSE between policy output and demo actions
        bc_loss = self._bc_loss_fn(mean_actions, demo_act_batch)

        # Add BC gradient to policy parameters
        total_loss = current_lambda * bc_loss
        total_loss.backward()

        # Log
        self.logger.record("train/bc_loss", bc_loss.item())
        self.logger.record("train/lambda_bc", current_lambda)


def make_env(env_id, reward_type='dense'):
    env = gymnasium.make(env_id, reward_type=reward_type, gravity_comp=True)
    env = FlattenObs(env)
    return env


def load_demos(demo_path):
    """Load demo transitions from npz file."""
    data = np.load(demo_path, allow_pickle=True)
    return (
        data['observations'].astype(np.float32),
        data['actions'].astype(np.float32),
        data['rewards'].astype(np.float32),
        data['next_observations'].astype(np.float32),
        data['dones'].astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser(description="DAPG Training")
    parser.add_argument('--env', type=str, default='PandaVLA-v0',
                        help='Environment ID (PandaVLA-v0 or PandaVLA-Rand-v0)')
    parser.add_argument('--demos', type=str, default='demos/panda_pickplace.npz',
                        help='Path to demo trajectories npz file')
    parser.add_argument('--total_timesteps', type=int, default=200000)
    parser.add_argument('--lambda_bc', type=float, default=0.5,
                        help='Initial BC regularization weight')
    parser.add_argument('--bc_decay', type=float, default=0.8,
                        help='BC weight decay factor (per 10% of training)')
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warm_start', type=str, default=None,
                        help='Path to pre-trained model .zip for warm start')
    parser.add_argument('--warm_start_vecnorm', type=str, default=None,
                        help='Path to pre-trained vec_normalize .pkl')
    parser.add_argument('--finetune_lr', type=float, default=None,
                        help='Learning rate for fine-tuning (default: 1e-4 if warm_start)')
    parser.add_argument('--finetune_lambda_bc', type=float, default=None,
                        help='BC weight for fine-tuning (default: 0.1 if warm_start)')
    parser.add_argument('--freeze_vecnorm', action='store_true',
                        help='Freeze VecNormalize obs stats during fine-tuning '
                             '(prevents catastrophic forgetting from input drift)')
    args = parser.parse_args()

    # Set default save_path based on env
    if args.save_path is None:
        if args.env == 'PandaVLA-Rand-v0':
            args.save_path = '/home/w/vla_workspace/outputs/dapg_rand'
        else:
            args.save_path = '/home/w/vla_workspace/outputs/dapg'

    # Load demos
    if not os.path.exists(args.demos):
        print(f"Demo file not found: {args.demos}")
        print("Run `python collect_demos.py` first to generate demos.")
        return

    demo_obs, demo_actions, _, _, _ = load_demos(args.demos)
    print(f"Loaded {len(demo_obs)} demo transitions from {args.demos}")

    # Create environments
    train_env = DummyVecEnv([functools.partial(make_env, args.env, 'dense')])
    if args.warm_start_vecnorm and os.path.exists(args.warm_start_vecnorm):
        train_env = VecNormalize.load(args.warm_start_vecnorm, train_env)
        train_env.norm_reward = True
        train_env.training = not args.freeze_vecnorm
        print(f"Loaded VecNormalize stats from {args.warm_start_vecnorm}"
              f" (training={'frozen' if args.freeze_vecnorm else 'active'})")
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([functools.partial(make_env, args.env, 'dense')])
    if args.warm_start_vecnorm and os.path.exists(args.warm_start_vecnorm):
        eval_env = VecNormalize.load(args.warm_start_vecnorm, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
    else:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Determine hyperparameters
    if args.warm_start:
        lr = args.finetune_lr if args.finetune_lr is not None else 1e-4
        lambda_bc = args.finetune_lambda_bc if args.finetune_lambda_bc is not None else 0.1
    else:
        lr = 3e-4
        lambda_bc = args.lambda_bc

    # Create or load DAPG-PPO model
    if args.warm_start and os.path.exists(args.warm_start):
        print(f"Warm starting from {args.warm_start}")
        model = DAPGPPO.load(
            args.warm_start,
            env=train_env,
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            lambda_bc=lambda_bc,
            bc_decay=args.bc_decay,
            total_timesteps=args.total_timesteps,
            learning_rate=lr,
            device='cuda',
        )
        print(f"  lr={lr}, lambda_bc={lambda_bc}")
    else:
        model = DAPGPPO(
            "MlpPolicy",
            train_env,
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            lambda_bc=lambda_bc,
            bc_decay=args.bc_decay,
            total_timesteps=args.total_timesteps,
            n_steps=2048,
            batch_size=64,
            learning_rate=lr,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(args.save_path, 'tb_logs'),
            seed=args.seed,
            device='cuda',
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_path, 'best'),
        log_path=os.path.join(args.save_path, 'eval_logs'),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # Train
    print(f"\n{'='*60}")
    if args.warm_start:
        print(f"DAPG Fine-tuning: {args.total_timesteps} steps (warm start)")
        print(f"Warm start: {args.warm_start}")
    else:
        print(f"DAPG Training: {args.total_timesteps} steps")
    print(f"LR: {lr}, BC weight: {lambda_bc} (decay={args.bc_decay})")
    print(f"Demos: {len(demo_obs)} transitions")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback],
        progress_bar=True,
    )

    # Save
    model.save(os.path.join(args.save_path, 'dapg_final'))
    train_env.save(os.path.join(args.save_path, 'vec_normalize.pkl'))
    print(f"\nTraining complete! Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
