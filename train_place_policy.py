#!/usr/bin/env python3
"""DAPG-PPO training for an independent place policy.

The place policy starts from an already-grasped state (place_mode=True)
and learns to move the block toward the target and release it. Only
placing rewards are used (reward_type="place_only").

The place_mode (hard-attached block) is used because v2 (hard-attached)
outperformed v3 (realistic physics) in hierarchical evaluation
(best dist 10.6cm vs 17.1cm). A release constraint in the environment
gates gripper opening to only when block_target_dist < 0.10m, preventing
the premature release that caused 0% place rate.

Demo data is filtered to the place phase: transitions after the block
has been lifted more than 5 cm above the table
(lift_height = obs[10] - 0.22 > 0.05, where obs[10] = block_z).

Usage:
    python train_place_policy.py --demos demos/panda_pickplace.npz \
        --total_timesteps 200000
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import pickle

import gymnasium
import numpy as np
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# DAPGPPO is defined in train_dapg.py. A dedicated dapg_ppo.py module does
# not exist yet; importing from train_dapg runs only harmless module-level
# setup (env vars + imports) since main() is guarded by __main__.
from train_dapg import DAPGPPO


GRASP_STATES_PATH = "/home/w/vla_workspace/outputs/grasp_states_500.pkl"


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats periodically so models can be evaluated
    mid-training (the final vec_normalize.pkl is only saved at the end)."""

    def __init__(self, save_path, save_freq=10000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.env.save(self.save_path)
        return True


class SaveVecNormalizeOnBest(BaseCallback):
    """Save VecNormalize stats when a new best model is found.

    This is passed as callback_on_new_best to EvalCallback, fixing the
    vec_normalize/best_model mismatch issue: previously, best_model.zip
    could be from step 90K while vec_normalize.pkl was from step 150K+,
    causing evaluation with mismatched normalization stats.
    """

    def __init__(self, vec_normalize_path, verbose=0):
        super().__init__(verbose)
        self.vec_normalize_path = vec_normalize_path

    def _on_step(self):
        os.makedirs(os.path.dirname(self.vec_normalize_path), exist_ok=True)
        self.model.env.save(self.vec_normalize_path)
        if self.verbose > 0:
            print(f"Saved vec_normalize.pkl alongside best_model to "
                  f"{self.vec_normalize_path}")
        return True


def make_env(env_id='PandaVLA-v0', reward_type='place_only',
             grasp_states=None, release_threshold=0.10):
    """Create a place-mode training/eval environment.

    Uses place_mode (hard-attached block): the block position is set
    to the hand each step, so the policy learns arm motion without
    dealing with grip physics. A release constraint in the env gates
    gripper opening to only when block_target_dist < release_threshold.
    The target matches the v3 grasp model's default target
    [0.5, 0.3, 0.2] so the place policy is compatible with the
    hierarchical eval.

    If grasp_states is provided, the env initializes the arm from
    collected grasp-policy states instead of a fixed lifted pose. This
    bridges the train-eval mismatch (see collect_grasp_states.py).

    Args:
        release_threshold: Distance (m) below which the gripper is
            allowed to open. Default 0.10m. Tightening to 0.05m for
            v13 forces the model to navigate closer before releasing,
            reducing post-release drift.
    """
    env = gymnasium.make(
        env_id,
        reward_type=reward_type,
        place_mode=True,
        gravity_comp=True,
        target_pos=np.array([0.5, 0.3, 0.2]),
        grasp_states=grasp_states,
    )
    env = FlattenObs(env)
    # Set configurable release threshold on the inner PandaVLAEnv
    env.unwrapped._release_dist_threshold = release_threshold
    return env


def load_place_demos(demo_path, lift_threshold=0.05):
    """Load demos and keep only the place phase.

    For each episode (split by the `dones` flag), the place phase starts
    at the first transition where the block has been lifted above
    `lift_threshold` meters from the table
    (lift_height = obs[10] - 0.22 > lift_threshold) and runs to the end
    of the episode.

    Args:
        demo_path: Path to the npz file with keys observations(N,16),
            actions(N,8), rewards, next_observations, dones.
        lift_threshold: Lift height (m) above the table that marks the
            start of the place phase.

    Returns:
        place_obs (np.ndarray, (M, 16)): place-phase observations.
        place_actions (np.ndarray, (M, 8)): place-phase actions.
    """
    data = np.load(demo_path, allow_pickle=True)
    observations = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    dones = data['dones'].astype(np.float32)

    table_z = 0.22
    # obs layout (FlattenObs): [joint(7), gripper(1), block_xyz(3),
    #                            hand_xyz(3), hand_block_dist(1),
    #                            block_target_dist(1)] -> obs[10] = block_z
    lift_height = observations[:, 10] - table_z

    # Episode boundaries from the dones flag
    done_idx = np.where(dones > 0.5)[0]
    ep_starts = np.concatenate([[0], done_idx + 1])
    ep_ends = np.concatenate([done_idx + 1, [len(dones)]])

    place_obs, place_actions = [], []
    for start, end in zip(ep_starts, ep_ends):
        if start >= end:
            continue
        ep_lift = lift_height[start:end]
        lifted = np.where(ep_lift > lift_threshold)[0]
        if len(lifted) == 0:
            continue
        place_start = start + lifted[0]
        place_obs.append(observations[place_start:end])
        place_actions.append(actions[place_start:end])

    if not place_obs:
        # Fallback: no episode structure found -> global filter
        mask = lift_height > lift_threshold
        if mask.sum() == 0:
            raise ValueError(
                f"No place-phase transitions (lift_height > {lift_threshold}m) "
                f"found in {demo_path}"
            )
        return observations[mask], actions[mask]

    place_obs = np.concatenate(place_obs, axis=0)
    place_actions = np.concatenate(place_actions, axis=0)
    return place_obs, place_actions


def main():
    parser = argparse.ArgumentParser(description="DAPG-PPO place policy training")
    parser.add_argument('--total_timesteps', type=int, default=300000)
    parser.add_argument(
        '--save_path', type=str,
        default='/home/w/vla_workspace/outputs/place_policy_v13',
    )
    parser.add_argument(
        '--grasp_states', type=str, default=GRASP_STATES_PATH,
        help='Path to collected grasp states pkl (for realistic init)',
    )
    parser.add_argument(
        '--demos', type=str,
        default='/home/w/vla_workspace/demos/panda_pickplace.npz',
    )
    parser.add_argument('--lambda_bc', type=float, default=0.5,
                        help='Initial BC regularization weight')
    parser.add_argument('--bc_decay', type=float, default=0.8,
                        help='BC weight decay factor (per 10% of training)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument(
        '--release_threshold', type=float, default=0.10,
        help='Release distance threshold (m). 0.05 for v13 tight release.',
    )
    parser.add_argument(
        '--load_model', type=str, default=None,
        help='Path to a saved model .zip to fine-tune from (e.g. v11 best_model)',
    )
    parser.add_argument(
        '--load_vecnorm', type=str, default=None,
        help='Path to vec_normalize.pkl matching --load_model',
    )
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Load and filter demos to the place phase
    if not os.path.exists(args.demos):
        print(f"Demo file not found: {args.demos}")
        print("Run `python collect_demos.py` first to generate demos.")
        return

    demo_obs, demo_actions = load_place_demos(args.demos)
    print(f"Loaded {len(demo_obs)} place-phase demo transitions from {args.demos}")

    # Load collected grasp states for realistic initialization
    grasp_states = None
    if os.path.exists(args.grasp_states):
        with open(args.grasp_states, 'rb') as f:
            grasp_states = pickle.load(f)
        print(f"Loaded {len(grasp_states)} grasp states from {args.grasp_states}")
    else:
        print(f"WARNING: grasp states not found at {args.grasp_states}")
        print("  Falling back to fixed lifted pose. Run collect_grasp_states.py first.")

    print(f"Release threshold: {args.release_threshold}m")

    # Create environments (place_mode + place_only reward).
    # See make_env docstring: requires PandaVLAEnv place_mode/place_only support.
    env_kwargs = dict(
        grasp_states=grasp_states,
        release_threshold=args.release_threshold,
    )
    train_env = DummyVecEnv([functools.partial(make_env, **env_kwargs)])
    eval_env = DummyVecEnv([functools.partial(make_env, **env_kwargs)])

    # Load VecNormalize stats if fine-tuning, otherwise create fresh
    if args.load_vecnorm and os.path.exists(args.load_vecnorm):
        print(f"Loading VecNormalize stats from {args.load_vecnorm}")
        train_env = VecNormalize.load(args.load_vecnorm, train_env)
        train_env.norm_reward = True
        train_env.training = True
        eval_env = VecNormalize.load(args.load_vecnorm, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Determine hyperparameters: use lower LR and BC weight for fine-tuning
    if args.load_model:
        lr = 1e-4
        lambda_bc = 0.1
    else:
        lr = args.learning_rate
        lambda_bc = args.lambda_bc

    # Create or load DAPG-PPO model
    if args.load_model and os.path.exists(args.load_model):
        print(f"Fine-tuning from {args.load_model}")
        print(f"  lr={lr}, lambda_bc={lambda_bc}")
        model = DAPGPPO.load(
            args.load_model,
            env=train_env,
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            lambda_bc=lambda_bc,
            bc_decay=args.bc_decay,
            total_timesteps=args.total_timesteps,
            learning_rate=lr,
            device=args.device,
        )
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
            seed=42,
            device=args.device,
        )

    # Eval callback: evaluate every 10000 steps, save best model
    # and vec_normalize stats together (fixes mismatch issue from v10)
    save_vecnorm_on_best = SaveVecNormalizeOnBest(
        os.path.join(args.save_path, 'best', 'vec_normalize.pkl'),
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_path, 'best'),
        log_path=os.path.join(args.save_path, 'eval_logs'),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=save_vecnorm_on_best,
        verbose=1,
    )

    # Save VecNormalize stats periodically for mid-training evaluation
    vecnorm_callback = SaveVecNormalizeCallback(
        os.path.join(args.save_path, 'vec_normalize.pkl'),
        save_freq=2048,  # save every rollout (n_steps=2048)
    )

    # Train
    print(f"\n{'='*60}")
    print(f"DAPG-PPO Place Training: {args.total_timesteps} steps")
    if args.load_model:
        print(f"Fine-tuning from: {args.load_model}")
    print(f"LR: {lr}, BC weight: {lambda_bc} (decay={args.bc_decay})")
    print(f"Device: {args.device}")
    print(f"Release threshold: {args.release_threshold}m")
    print(f"Place-phase demos: {len(demo_obs)} transitions")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, vecnorm_callback],
        progress_bar=False,
    )

    # Save final model + normalization stats
    model.save(os.path.join(args.save_path, 'place_final'))
    train_env.save(os.path.join(args.save_path, 'vec_normalize.pkl'))
    print(f"\nTraining complete! Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
