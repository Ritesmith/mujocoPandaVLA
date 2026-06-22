#!/usr/bin/env python3
"""Group Relative Policy Optimization (GRPO) for VLA + RL fine-tuning.

GRPO improves VLA policies by generating multiple action candidates per
observation, ranking them by reward, and training the policy to prefer
high-reward actions using a relative advantage within each group.

Key idea:
1. For each observation, sample K action candidates from the VLA policy
2. Execute each candidate in simulation, collect rewards
3. Compute relative advantages: A_i = (R_i - mean(R)) / std(R)
4. Update policy using clipped PPO-style objective with these advantages

This avoids needing a separate value function (critic), making it
memory-efficient for large VLA models.

Usage:
    python train_grpo.py --n_groups 4 --group_size 8 --total_steps 5000
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import sys
import numpy as np
import mujoco
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium
import gym_env
from gym_env.wrappers import FlattenObs
from collections import deque


class GRPOPolicy(nn.Module):
    """Simple MLP policy for GRPO training (joint-space control).

    This is a standalone policy that can be initialized from VLA features
    or trained from scratch. For VLA integration, the VLA model would
    replace this policy's backbone.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        features = self.backbone(obs)
        mean = torch.tanh(self.mean_head(features))
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mean, std

    def get_action(self, obs, deterministic=False):
        mean, std = self.forward(obs)
        if deterministic:
            return mean
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, obs, actions):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return log_prob


class GRPOTrainer:
    """GRPO trainer for Panda pick-place task.

    For each training step:
    1. Sample an observation from the environment
    2. Generate K action candidates (group_size)
    3. Execute each in the env, collect rewards
    4. Compute group-relative advantages
    5. Update policy with clipped objective

    Supports two policy types:
    - 'mlp': GRPOPolicy (standalone MLP)
    - 'vla': VLAPolicyAdapter (wraps SmolVLA with LoRA)
    """

    def __init__(self, policy, env, lr=3e-4, clip_range=0.2,
                 group_size=8, n_groups=4, gamma=0.99,
                 entropy_coef=0.01, max_grad_norm=0.5,
                 device='cuda', policy_type='mlp',
                 task='pick up the red block'):
        self.policy = policy.to(device)
        self.env = env
        if policy_type == 'vla':
            self.optimizer = torch.optim.Adam(policy.get_lora_params(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_range = clip_range
        self.group_size = group_size
        self.n_groups = n_groups
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.policy_type = policy_type
        self.task = task

        # Logging
        self._rewards = deque(maxlen=100)
        self._policy_losses = deque(maxlen=100)
        self._entropies = deque(maxlen=100)

    def _get_raw_env(self):
        """Unwrap the env to get the underlying PandaVLAEnv."""
        raw_env = self.env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env
        if hasattr(raw_env, 'unwrapped'):
            raw_env = raw_env.unwrapped
        return raw_env

    def _get_image(self):
        """Render an image from the underlying env for VLA inference."""
        raw_env = self._get_raw_env()
        if hasattr(raw_env, '_render_image'):
            return raw_env._render_image()
        return None

    def _save_policy(self, path):
        """Save policy parameters (LoRA for VLA, full state_dict for MLP)."""
        if self.policy_type == 'vla' and hasattr(self.policy, 'save_lora'):
            self.policy.save_lora(path)
        else:
            torch.save(self.policy.state_dict(), path)

    def collect_group(self, obs_flat, n_actions):
        """Generate n_actions candidates and evaluate them.

        Saves/restores MuJoCo state so each candidate is evaluated
        from the same initial state for fair comparison.

        Args:
            obs_flat: flattened observation (obs_dim,)
            n_actions: number of action candidates to generate

        Returns:
            actions: (n_actions, act_dim)
            log_probs: (n_actions,)
            rewards: (n_actions,) - single-step rewards from executing each action
        """
        if self.policy_type == 'vla':
            image = self._get_image()
            with torch.no_grad():
                actions, log_probs = self.policy.get_action_group(
                    obs_flat, n_actions, image=image, task=self.task
                )
            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
        else:
            obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
            obs_batch = obs_tensor.unsqueeze(0).expand(n_actions, -1)

            # Sample actions
            with torch.no_grad():
                mean, std = self.policy(obs_batch)
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(dim=-1)

            actions_np = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

        # Save MuJoCo state for fair evaluation
        # Access the underlying MuJoCo model/data through the env wrapper chain
        raw_env = self.env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env
        if hasattr(raw_env, 'unwrapped'):
            raw_env = raw_env.unwrapped

        mj_model = raw_env.model
        mj_data = raw_env.data

        # Save full state: qpos, qvel, ctrl, act, warm_start
        state_size = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        saved_state = np.zeros(state_size)
        mujoco.mj_getState(mj_model, mj_data, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        # Evaluate each action in the environment
        rewards = np.zeros(n_actions, dtype=np.float32)
        for i in range(n_actions):
            # Restore state before each evaluation
            mujoco.mj_setState(mj_model, mj_data, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(mj_model, mj_data)
            # Sync env's _arm_target with restored qpos
            raw_env._arm_target = mj_data.qpos[raw_env._arm_qpos_adrs].copy()

            next_obs, reward, terminated, truncated, info = self.env.step(actions_np[i])
            rewards[i] = reward

        # Restore original state after all evaluations
        mujoco.mj_setState(mj_model, mj_data, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(mj_model, mj_data)
        # Sync env's _arm_target with restored qpos
        raw_env._arm_target = mj_data.qpos[raw_env._arm_qpos_adrs].copy()

        return actions_np, log_probs_np, rewards

    def compute_advantages(self, rewards):
        """Compute group-relative advantages.

        A_i = (R_i - mean(R)) / (std(R) + eps)

        This normalizes rewards within each group, so the policy learns
        relative preferences rather than absolute values.
        """
        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages = (rewards - mean_r) / std_r
        return advantages

    def update(self, obs_flat, actions, old_log_probs, advantages):
        """Update policy using GRPO clipped objective.

        L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
            - entropy_coef * H(pi)

        Where ratio = exp(new_log_prob - old_log_prob)
        """
        obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        if self.policy_type == 'vla':
            image = self._get_image()
            new_log_probs = self.policy.get_log_prob(
                obs_flat, actions_tensor, image=image, task=self.task
            )
            # Entropy of Gaussian depends only on std (not mu)
            std = torch.exp(self.policy.log_std.clamp(-5, 2))
            dummy_dist = torch.distributions.Normal(torch.zeros_like(std), std)
            entropy = dummy_dist.entropy().sum()
        else:
            # Expand obs to match actions
            obs_batch = obs_tensor.unsqueeze(0).expand(len(actions), -1)

            # Get new log probs
            mean, std = self.policy(obs_batch)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)

        # Clipped objective
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        # Total loss
        loss = policy_loss - self.entropy_coef * entropy

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        if self.policy_type == 'vla':
            nn.utils.clip_grad_norm_(self.policy.get_lora_params(), self.max_grad_norm)
        else:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self._policy_losses.append(policy_loss.item())
        self._entropies.append(entropy.item())

        return policy_loss.item(), entropy.item()

    def train(self, total_steps, eval_freq=500, save_path=None):
        """Main training loop."""
        obs, info = self.env.reset()
        obs_flat = obs.astype(np.float32) if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)

        best_reward = -float('inf')

        for step in range(total_steps):
            # Collect group of action candidates
            actions, log_probs, rewards = self.collect_group(obs_flat, self.group_size)

            # Compute group-relative advantages
            advantages = self.compute_advantages(rewards)

            # Update policy
            p_loss, entropy = self.update(obs_flat, actions, log_probs, advantages)

            # Step environment with best action
            best_idx = np.argmax(rewards)
            best_action = actions[best_idx]
            # Sync _arm_target with restored MuJoCo state before stepping
            raw_env = self.env
            while hasattr(raw_env, 'env'):
                raw_env = raw_env.env
            if hasattr(raw_env, 'unwrapped'):
                raw_env = raw_env.unwrapped
            raw_env._arm_target = raw_env.data.qpos[raw_env._arm_qpos_adrs].copy()
            next_obs, reward, terminated, truncated, info = self.env.step(best_action)
            self._rewards.append(reward)

            obs_flat = next_obs.astype(np.float32) if isinstance(next_obs, np.ndarray) else np.array(next_obs, dtype=np.float32)

            if terminated or truncated:
                obs_flat, info = self.env.reset()
                obs_flat = obs_flat.astype(np.float32) if isinstance(obs_flat, np.ndarray) else np.array(obs_flat, dtype=np.float32)

            # Logging
            if step % 100 == 0:
                mean_reward = np.mean(self._rewards) if self._rewards else 0
                mean_ploss = np.mean(self._policy_losses) if self._policy_losses else 0
                mean_ent = np.mean(self._entropies) if self._entropies else 0
                print(f"Step {step}/{total_steps} | "
                      f"reward={mean_reward:.3f} | "
                      f"p_loss={mean_ploss:.4f} | "
                      f"entropy={mean_ent:.4f}")

            # Eval and save
            if save_path and (step + 1) % eval_freq == 0:
                eval_reward = self.evaluate(n_episodes=5)
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    self._save_policy(os.path.join(save_path, 'grpo_best.pt'))
                    print(f"  New best model! eval_reward={eval_reward:.2f}")
                self._save_policy(os.path.join(save_path, f'grpo_step{step}.pt'))

        # Save final
        if save_path:
            self._save_policy(os.path.join(save_path, 'grpo_final.pt'))

    def evaluate(self, n_episodes=5, max_steps=500):
        """Evaluate policy over n episodes.

        Saves/restores MuJoCo state and _arm_target so that training
        can continue seamlessly after evaluation.
        """
        # Save current MuJoCo state and _arm_target
        raw_env = self.env
        while hasattr(raw_env, 'env'):
            raw_env = raw_env.env
        if hasattr(raw_env, 'unwrapped'):
            raw_env = raw_env.unwrapped

        mj_model = raw_env.model
        mj_data = raw_env.data
        state_size = mujoco.mj_stateSize(mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        saved_state = np.zeros(state_size)
        mujoco.mj_getState(mj_model, mj_data, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        saved_arm_target = raw_env._arm_target.copy()

        total_rewards = []
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            obs_flat = obs.astype(np.float32) if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)
            ep_reward = 0.0

            for step in range(max_steps):
                if self.policy_type == 'vla':
                    image = self._get_image()
                    with torch.no_grad():
                        action = self.policy.get_action(
                            obs_flat, image=image, task=self.task, deterministic=True
                        )
                    action_np = action.cpu().numpy()
                else:
                    obs_tensor = torch.tensor(obs_flat, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        action = self.policy.get_action(obs_tensor.unsqueeze(0), deterministic=True)
                    action_np = action[0].cpu().numpy()

                next_obs, reward, terminated, truncated, info = self.env.step(action_np)
                ep_reward += reward
                obs_flat = next_obs.astype(np.float32) if isinstance(next_obs, np.ndarray) else np.array(next_obs, dtype=np.float32)

                if terminated or truncated:
                    break

            total_rewards.append(ep_reward)

        # Restore MuJoCo state and _arm_target
        mujoco.mj_setState(mj_model, mj_data, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(mj_model, mj_data)
        raw_env._arm_target = saved_arm_target

        return np.mean(total_rewards)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training")
    parser.add_argument('--env', type=str, default='PandaVLA-v0',
                        help='Environment ID (PandaVLA-v0 or PandaVLA-Rand-v0)')
    parser.add_argument('--policy', type=str, choices=['mlp', 'vla'],
                        default='mlp', help='Policy type: mlp or vla')
    parser.add_argument('--vla_model_path', type=str,
                        default='/home/w/vla_workspace/models/smolvla_base',
                        help='Path to SmolVLA pretrained model directory')
    parser.add_argument('--task', type=str, default='pick up the red block',
                        help='Task instruction for VLA inference')
    parser.add_argument('--group_size', type=int, default=8,
                        help='Number of action candidates per group')
    parser.add_argument('--n_groups', type=int, default=4,
                        help='Number of groups per training step (unused in current impl)')
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Set default save_path based on env
    if args.save_path is None:
        if args.env == 'PandaVLA-Rand-v0':
            args.save_path = '/home/w/vla_workspace/outputs/grpo_rand'
        else:
            args.save_path = '/home/w/vla_workspace/outputs/grpo'

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env = gymnasium.make(args.env, reward_type='dense', gravity_comp=True)
    wrapped_env = FlattenObs(env)

    # Get dimensions
    obs, _ = wrapped_env.reset()
    obs_dim = len(obs)
    act_dim = wrapped_env.action_space.shape[0]
    print(f"obs_dim={obs_dim}, act_dim={act_dim}")

    # Create policy
    if args.policy == 'vla':
        try:
            from vla_policy_adapter import VLAPolicyAdapter
            print(f"Loading VLA policy from {args.vla_model_path} ...")
            policy = VLAPolicyAdapter(
                vla_model_path=args.vla_model_path,
                obs_dim=obs_dim,
                act_dim=act_dim,
                device=args.device,
            )
            print("VLA policy loaded successfully.")
        except FileNotFoundError as exc:
            print(f"\n[ERROR] VLA model not found:\n  {exc}")
            print("Make sure the SmolVLA model path is correct.")
            print("You can also fall back to --policy mlp.")
            sys.exit(1)
        except Exception as exc:
            print(f"\n[ERROR] Failed to load VLA policy: {exc}")
            print("Make sure the SmolVLA model path is correct and lerobot is installed.")
            print("You can also fall back to --policy mlp.")
            sys.exit(1)
    else:
        policy = GRPOPolicy(obs_dim, act_dim, hidden_dim=256)

    # Create trainer
    trainer = GRPOTrainer(
        policy=policy,
        env=wrapped_env,
        lr=args.lr,
        clip_range=args.clip_range,
        group_size=args.group_size,
        n_groups=args.n_groups,
        entropy_coef=args.entropy_coef,
        device=args.device,
        policy_type=args.policy,
        task=args.task,
    )

    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)

    # Train
    print(f"\n{'='*60}")
    print(f"GRPO Training: {args.total_steps} steps")
    print(f"Policy: {args.policy}")
    print(f"Group size: {args.group_size}")
    print(f"{'='*60}\n")

    trainer.train(
        total_steps=args.total_steps,
        eval_freq=500,
        save_path=args.save_path,
    )

    print(f"\nTraining complete! Models saved to {args.save_path}")


if __name__ == "__main__":
    main()
