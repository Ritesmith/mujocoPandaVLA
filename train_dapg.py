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
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import explained_variance
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
    Supports both flat state vectors (MlpPolicy) and Dict observations
    {"image", "state"} (MultiInputPolicy).
    """

    def __init__(self, *args, demo_obs=None, demo_actions=None,
                 lambda_bc=0.5, bc_decay=0.8, total_timesteps=100000,
                 image_augment=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_obs = demo_obs
        self.demo_actions = demo_actions
        self.lambda_bc = lambda_bc
        self.bc_decay = bc_decay
        self.total_timesteps = total_timesteps
        self.image_augment = image_augment
        self._bc_loss_fn = nn.MSELoss()

    def _get_vec_normalize(self):
        """Traverse the env wrapper chain to find VecNormalize.

        For vision mode the chain is VecTransposeImage -> VecNormalize ->
        DummyVecEnv, so ``self.env`` alone is not enough.
        """
        env = self.env
        while env is not None:
            if isinstance(env, VecNormalize):
                return env
            env = getattr(env, "venv", None)
        return None

    def _normalize_demo_state(self, states):
        """Normalize demo ``state`` using VecNormalize running statistics.

        VecNormalize normalizes on-policy state observations but not images.
        To keep the BC demo inputs on the same distribution as on-policy
        data, we replicate the normalization here. For Dict observations
        ``obs_rms`` is a dict keyed by the normalized observation keys.
        """
        vec_norm = self._get_vec_normalize()
        if vec_norm is None:
            return states
        obs_rms = vec_norm.obs_rms
        if isinstance(obs_rms, dict) and "state" in obs_rms:
            state_rms = obs_rms["state"]
            mean = torch.as_tensor(
                state_rms.mean, dtype=torch.float32, device=states.device
            )
            var = torch.as_tensor(
                state_rms.var, dtype=torch.float32, device=states.device
            )
            states = (states - mean) / torch.sqrt(var + vec_norm.epsilon)
            states = torch.clamp(states, -vec_norm.clip_obs, vec_norm.clip_obs)
        return states


    def _augment_images(self, images):
        """Apply random augmentations to a batch of images (CHW float).

        Safe augmentations only: random crop, brightness, contrast.
        No flips (would break spatial correspondence with actions/target).
        """
        import torch.nn.functional as F
        batch_size = images.shape[0]
        H, W = images.shape[2], images.shape[3]

        # 1. Random crop: pad by 4 on each side, then random crop back
        if torch.rand(1).item() < 0.5:
            padded = F.pad(images, (4, 4, 4, 4), mode='replicate')
            crops_h = torch.randint(0, 9, (batch_size,))
            crops_w = torch.randint(0, 9, (batch_size,))
            cropped = []
            for i in range(batch_size):
                cropped.append(
                    padded[i, :, crops_h[i]:crops_h[i]+H, crops_w[i]:crops_w[i]+W]
                )
            images = torch.stack(cropped)

        # 2. Random brightness: +/- 15 (on 0-255 scale)
        if torch.rand(1).item() < 0.5:
            brightness = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(-15, 15)
            images = images + brightness

        # 3. Random contrast: 0.85 to 1.15
        if torch.rand(1).item() < 0.5:
            contrast = torch.empty(batch_size, 1, 1, 1, device=images.device).uniform_(0.85, 1.15)
            images = images * contrast

        return images.clamp(0, 255)

    def _compute_bc_loss(self, batch_size):
        """Compute BC loss on a random mini-batch of demo data.

        Samples ``batch_size`` demo transitions, runs them through the
        policy's actor path (feature_extractor → mlp_extractor → action_net),
        and returns the MSE between policy mean_actions and demo actions.

        Returns None if no demo data is available.
        """
        if self.demo_obs is None:
            return None

        if isinstance(self.demo_obs, dict):
            n_demos = len(self.demo_obs["image"])
        else:
            n_demos = len(self.demo_obs)
        if n_demos == 0:
            return None

        bc_batch_size = min(batch_size, n_demos)
        indices = np.random.randint(0, n_demos, size=bc_batch_size)
        demo_act_batch = torch.tensor(
            self.demo_actions[indices], dtype=torch.float32, device=self.device
        )

        if isinstance(self.demo_obs, dict):
            demo_images = torch.as_tensor(
                self.demo_obs["image"][indices], device=self.device
            )
            demo_states = torch.as_tensor(
                self.demo_obs["state"][indices],
                dtype=torch.float32, device=self.device
            )
            demo_states = self._normalize_demo_state(demo_states)
            demo_images = demo_images.permute(0, 3, 1, 2).contiguous().float()
            if getattr(self, 'image_augment', False):
                demo_images = self._augment_images(demo_images)
            demo_obs_batch = {"image": demo_images, "state": demo_states}
        else:
            demo_obs_batch = torch.tensor(
                self.demo_obs[indices], dtype=torch.float32, device=self.device
            )

        features = self.policy.extract_features(demo_obs_batch)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        mean_actions = self.policy.action_net(latent_pi)

        bc_loss = self._bc_loss_fn(mean_actions, demo_act_batch)
        return bc_loss

    def train(self):
        """DAPG-PPO joint training: PPO clipped objective + BC regularization.

        BC loss is computed per mini-batch and ADDED to the PPO loss before
        a single optimizer.step(), so PPO and BC gradients are applied
        together in the same update. This is the standard DAPG formulation:

            L = L_policy + c_ent * L_entropy + c_vf * L_value + lambda_bc * L_BC

        V59 bug fix: the previous implementation called super().train() (full
        PPO loop) then computed BC loss.backward() WITHOUT optimizer.step().
        The next PPO update's zero_grad() cleared the BC gradients before they
        were ever applied — lambda_bc had zero effect throughout V50-V59.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        bc_losses = []

        # BC weight with exponential decay over training progress
        progress = self.num_timesteps / max(self.total_timesteps, 1)
        current_lambda = self.lambda_bc * (self.bc_decay ** (progress * 10))
        has_demos = self.demo_obs is not None

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Joint loss: PPO + BC (single backward + step applies both)
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if has_demos:
                    bc_loss = self._compute_bc_loss(self.batch_size)
                    if bc_loss is not None:
                        loss = loss + current_lambda * bc_loss
                        bc_losses.append(bc_loss.item())

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to max kl: {approx_kl_div:.4f}")
                    break

                # Optimization step: PPO + BC gradients applied together
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if bc_losses:
            self.logger.record("train/bc_loss", np.mean(bc_losses))
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
