#!/usr/bin/env python3
"""RL from scratch training with BC warmstart + hack-free reward.

This is the implementation of the `different_paradigm → rl_from_scratch` path
in the experiment decision tree. It is the ONLY remaining open path after ALL
9 offline imitation learning method families were falsified.

Key differences from train_place_policy.py:
  1. Loads BC warmstart (outputs/bc_expert_v1/final_model.zip, 22% place rate)
     instead of V59 — provides guided exploration starting point
  2. Uses reward_type='place_safe' — ALL per-step rewards <= 0, closing all 4
     documented reward hacks (hover, unconditional lowering, distance norm,
     reward decoupling)
  3. Progressive training: 100K (smoke test) → 5M (main) → 15M (final push)
  4. SAC fallback if PPO plateaus at 5M steps
  5. Safety stop: place_rate < 5% for 3 consecutive hier evals → auto-stop

The BC warmstart solves the cold-start exploration problem: pure RL from random
init would never find the target in 8-dim continuous action space. BC provides
a 22% starting point, and RL fine-tuning uses environment interaction to
correct the BC policy's distribution mismatch (the failure mode of all offline
IL methods).

Usage:
    # Stage 1: 100K smoke test
    python train_rl_from_scratch.py --stage 1 --save_path outputs/rl_from_scratch_v1

    # Stage 2: 5M main training (load stage 1 best)
    python train_rl_from_scratch.py --stage 2 --load_model outputs/rl_from_scratch_v1/best_hier/best_model.zip \\
        --save_path outputs/rl_from_scratch_v2

    # SAC fallback (if PPO plateaus)
    python train_rl_from_scratch.py --algorithm sac --stage 2 --save_path outputs/rl_sac_v1
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import json
import pickle
import time
from pathlib import Path

import gymnasium
import numpy as np
import torch as th
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv, VecNormalize

# Reuse ALL infrastructure from train_place_policy.py
from train_place_policy import (
    linear_schedule, cosine_schedule,
    freeze_bn_running_stats, freeze_backbone_params,
    SaveVecNormalizeCallback, SaveVecNormalizeOnBest,
    ClipRangeScheduleCallback, PBRSDiagnosticsCallback,
    HierPlaceRateCallback, RollingCheckpointCallback,
    make_env, GRASP_STATES_PATH,
)
from train_dapg import DAPGPPO


WORKSPACE = Path(__file__).parent.resolve()

# BC warmstart model (22% place rate, from train_bc_expert.py)
BC_WARMSTART_PATH = str(WORKSPACE / "outputs/bc_expert_v1/final_model.zip")
# V59 VecNormalize stats (BC model uses V59's normalization)
V59_VECNORM_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/vec_normalize.pkl")
# Grasp model for hierarchical eval
GRASP_MODEL_PATH = str(WORKSPACE / "outputs/dapg_800k_v5/best/best_model.zip")
GRASP_VECNORM_PATH = str(WORKSPACE / "outputs/dapg_800k_v5/vec_normalize.pkl")
# Oracle IK demos for DAPG BC regularization
ORACLE_DEMOS_PATH = str(WORKSPACE / "data/D_expert.npz")

# Progressive training stages: (steps, lr, freeze_backbone, freeze_bn, target_kl)
STAGE_CONFIG = {
    1: {  # Smoke test: 100K steps, frozen backbone
        "total_timesteps": 100_000,
        "learning_rate": 3e-4,
        "freeze_backbone": True,
        "freeze_bn": True,
        "target_kl": 0.015,
        "n_steps": 1024,
        "batch_size": 32,
        "lambda_bc": 0.1,
        "description": "Stage 1: smoke test, frozen backbone, monitor for degradation",
    },
    2: {  # Main training: 5M steps, unfreeze layer4
        "total_timesteps": 5_000_000,
        "learning_rate": 1e-4,
        "freeze_backbone": False,  # unfreeze for main training
        "freeze_bn": True,         # keep BN frozen (V59 root cause fix)
        "target_kl": 0.015,
        "n_steps": 2048,
        "batch_size": 64,
        "lambda_bc": 0.05,         # decay BC weight
        "description": "Stage 2: main training, backbone trainable, BN frozen",
    },
    3: {  # Final push: 15M steps, full unfreeze
        "total_timesteps": 15_000_000,
        "learning_rate": 3e-5,
        "freeze_backbone": False,
        "freeze_bn": True,
        "target_kl": 0.015,
        "n_steps": 2048,
        "batch_size": 64,
        "lambda_bc": 0.02,         # minimal BC
        "description": "Stage 3: final push, low LR, minimal BC regularization",
    },
}

# Hierarchical eval target range (matches V59 training)
HIER_TARGET_POS_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]


def load_oracle_demos(demo_path):
    """Load Oracle IK demos for DAPG BC regularization.

    Returns (demo_obs, demo_actions) in the format expected by DAPGPPO.
    For vision mode, demo_obs is a dict {"image": ..., "state": ...}.
    """
    if not os.path.exists(demo_path):
        print(f"Warning: Oracle demos not found at {demo_path}")
        print("  DAPG BC regularization will be disabled (lambda_bc=0)")
        return None, None

    data = np.load(demo_path, allow_pickle=True)
    demo_obs = {
        "image": data["images"],   # (N, 84, 84, 3) uint8
        "state": data["states"],   # (N, 12) float32
    }
    demo_actions = data["actions"]  # (N, 8) float32
    print(f"Loaded {len(demo_actions)} Oracle IK demo transitions from {demo_path}")
    if "n_placed" in data.files and "n_episodes" in data.files:
        n_placed = int(data["n_placed"])
        n_eps = int(data["n_episodes"])
        print(f"  Oracle place rate: {n_placed}/{n_eps} ({100*n_placed/n_eps:.1f}%)")
    return demo_obs, demo_actions


def build_callbacks(args, stage_cfg, save_path):
    """Build the callback list for training."""
    callbacks = []

    # 1. EvalCallback (place_mode eval, for early stopping signal)
    eval_env = make_env(
        reward_type='place_safe',
        grasp_states=None,  # eval env doesn't need grasp states
        vision_mode=True,
        domain_randomize=False,
        release_threshold=0.10,
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize.load(V59_VECNORM_PATH, eval_env)
    eval_env.norm_reward = False
    eval_env.training = False
    eval_env = VecTransposeImage(eval_env)

    save_vecnorm_on_best = SaveVecNormalizeOnBest(
        os.path.join(save_path, 'best', 'vec_normalize.pkl'),
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, 'best'),
        log_path=os.path.join(save_path, 'eval_logs'),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=save_vecnorm_on_best,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # 2. SaveVecNormalizeCallback (periodic)
    vecnorm_callback = SaveVecNormalizeCallback(
        os.path.join(save_path, 'vec_normalize.pkl'),
        save_freq=stage_cfg["n_steps"],
    )
    callbacks.append(vecnorm_callback)

    # 3. RollingCheckpointCallback
    checkpoint_callback = RollingCheckpointCallback(
        save_path=os.path.join(save_path, 'checkpoints'),
        save_freq=50000,
        keep_last=3,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # 4. HierPlaceRateCallback (TRUE place_rate eval + safety stop)
    hier_callback = HierPlaceRateCallback(
        eval_freq=10000,
        n_episodes=50,
        grasp_model=GRASP_MODEL_PATH,
        grasp_vecnorm=GRASP_VECNORM_PATH,
        target_pos_range=HIER_TARGET_POS_RANGE,
        save_path=save_path,
        early_stop_threshold=5,       # 5% place rate
        early_stop_consecutive=3,     # 3 consecutive evals
        first_eval_floor=10,          # stop if first eval < 10% (BC degraded)
        decoupling_detection=True,    # detect reward-place_rate decoupling
        verbose=1,
    )
    callbacks.append(hier_callback)

    # 5. ClipRangeScheduleCallback (decay clip 0.2 → 0.05)
    clip_sched = ClipRangeScheduleCallback(
        initial_clip=0.2,
        final_clip=0.05,
        total_timesteps=stage_cfg["total_timesteps"],
        verbose=1,
    )
    callbacks.append(clip_sched)

    return callbacks


def create_ppo_model(args, stage_cfg, train_env, demo_obs, demo_actions):
    """Create or load a DAPGPPO model with BC warmstart."""
    lr = cosine_schedule(stage_cfg["learning_rate"], final_lr=stage_cfg["learning_rate"] * 0.1)

    if args.load_model and os.path.exists(args.load_model):
        # Load from previous stage or external checkpoint
        print(f"Loading model from {args.load_model}")
        model = DAPGPPO.load(
            args.load_model,
            env=train_env,
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            lambda_bc=stage_cfg["lambda_bc"],
            bc_decay=0.5,
            total_timesteps=stage_cfg["total_timesteps"],
            learning_rate=lr,
            image_augment=False,
            device=args.device,
        )
    elif args.bc_warmstart and os.path.exists(args.bc_warmstart):
        # Load BC warmstart model (22% place rate)
        print(f"Loading BC warmstart from {args.bc_warmstart}")
        model = DAPGPPO.load(
            args.bc_warmstart,
            env=train_env,
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            lambda_bc=stage_cfg["lambda_bc"],
            bc_decay=0.5,
            total_timesteps=stage_cfg["total_timesteps"],
            learning_rate=lr,
            image_augment=False,
            device=args.device,
        )
    else:
        raise FileNotFoundError(
            f"Neither --load_model nor --bc_warmstart found. "
            f"BC warmstart expected at: {args.bc_warmstart}"
        )

    # Override n_steps and batch_size for this stage
    model.n_steps = stage_cfg["n_steps"]
    model.batch_size = stage_cfg["batch_size"]
    model.target_kl = stage_cfg["target_kl"]
    model.ent_coef = 0.01
    model.max_grad_norm = 0.5

    # Reinitialize rollout buffer with new n_steps
    from gymnasium import spaces as gym_spaces
    from stable_baselines3.common.buffers import DictRolloutBuffer
    buffer_cls = DictRolloutBuffer(
        stage_cfg["n_steps"],
        model.observation_space,
        model.action_space,
        device=model.device,
        gamma=model.gamma,
        gae_lambda=model.gae_lambda,
        n_envs=model.n_envs,
    )
    model.rollout_buffer = buffer_cls

    return model


def create_sac_model(args, stage_cfg, train_env):
    """Create SAC model with encoder weights copied from BC warmstart."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    print("Creating SAC model (fallback algorithm)")
    print(f"  BC warmstart encoder weights will be copied from {args.bc_warmstart}")

    # Load BC warmstart to get encoder weights
    bc_model = DAPGPPO.load(args.bc_warmstart, env=train_env, device=args.device)
    bc_features_extractor = bc_model.policy.features_extractor

    # Create SAC with same features extractor class
    from pretrained_cnn import ResNetFeaturesExtractor
    policy_kwargs = {
        "features_extractor_class": ResNetFeaturesExtractor,
        "features_extractor_kwargs": {
            "features_dim": 512,
            "backbone": "resnet18",
        },
    }

    model = SAC(
        "MultiInputPolicy",
        train_env,
        learning_rate=stage_cfg["learning_rate"],
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=5000,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,  # soft target update
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(args.save_path, 'tb_logs'),
        seed=42,
        device=args.device,
    )

    # Copy encoder weights from BC warmstart
    sac_fe = model.policy.features_extractor
    if hasattr(sac_fe, 'load_state_dict'):
        try:
            sac_fe.load_state_dict(bc_features_extractor.state_dict())
            print("  Successfully copied encoder weights from BC warmstart to SAC")
        except RuntimeError as e:
            print(f"  Warning: could not copy encoder weights: {e}")

    return model


def save_training_results(save_path, args, stage_cfg, model, train_env,
                           safety_stop_triggered=False, decoupling_detected=False):
    """Save training_results.json for decision tree evidence extraction."""
    results = {
        "algorithm": args.algorithm,
        "stage": args.stage,
        "total_timesteps": stage_cfg["total_timesteps"],
        "bc_warmstart": args.bc_warmstart,
        "reward_type": "place_safe",
        "lambda_bc": stage_cfg["lambda_bc"],
        "learning_rate": stage_cfg["learning_rate"],
        "freeze_backbone": stage_cfg["freeze_backbone"],
        "target_kl": stage_cfg["target_kl"],
        "safety_stop_triggered": safety_stop_triggered,
        "decoupling_detected": decoupling_detected,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Try to extract best place_rate from hier eval logs
    hier_log_path = os.path.join(save_path, 'eval_logs')
    if os.path.exists(hier_log_path):
        try:
            # HierPlaceRateCallback stores results in best_hier/
            best_hier_path = os.path.join(save_path, 'best_hier')
            if os.path.exists(best_hier_path):
                results["best_model_path"] = best_hier_path
        except Exception:
            pass

    results_file = os.path.join(save_path, 'training_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RL from scratch training with BC warmstart + hack-free reward"
    )
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo',
                        help='RL algorithm (PPO primary, SAC fallback)')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], default=1,
                        help='Training stage: 1=100K smoke, 2=5M main, 3=15M final')
    parser.add_argument('--total_timesteps', type=int, default=None,
                        help='Override total timesteps for this stage')
    parser.add_argument('--bc_warmstart', type=str, default=BC_WARMSTART_PATH,
                        help='Path to BC warmstart model (PPO .zip)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to a saved model .zip to continue from (e.g. stage 1 best)')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Output directory for this training run')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--grasp_states', type=str, default=GRASP_STATES_PATH,
                        help='Path to collected grasp states pkl')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='Disable TensorBoard logging')
    args = parser.parse_args()

    # Get stage configuration
    stage_cfg = STAGE_CONFIG[args.stage].copy()
    if args.total_timesteps is not None:
        stage_cfg["total_timesteps"] = args.total_timesteps

    os.makedirs(args.save_path, exist_ok=True)

    print("=" * 70)
    print(f"RL From Scratch Training — Stage {args.stage} ({args.algorithm.upper()})")
    print("=" * 70)
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Stage: {args.stage} — {stage_cfg['description']}")
    print(f"Total timesteps: {stage_cfg['total_timesteps']:,}")
    print(f"Learning rate: {stage_cfg['learning_rate']}")
    print(f"Freeze backbone: {stage_cfg['freeze_backbone']}")
    print(f"Target KL: {stage_cfg['target_kl']}")
    print(f"n_steps: {stage_cfg['n_steps']}, batch_size: {stage_cfg['batch_size']}")
    print(f"BC warmstart: {args.bc_warmstart}")
    print(f"Save path: {args.save_path}")
    print(f"Reward type: place_safe (ALL per-step rewards <= 0)")
    print(f"Lambda BC: {stage_cfg['lambda_bc']} (DAPG guided exploration)")
    print()

    # Load Oracle IK demos for DAPG BC regularization
    demo_obs, demo_actions = load_oracle_demos(ORACLE_DEMOS_PATH)

    # Load grasp states for realistic env initialization
    grasp_states = None
    if os.path.exists(args.grasp_states):
        with open(args.grasp_states, 'rb') as f:
            grasp_states = pickle.load(f)
        print(f"Loaded {len(grasp_states)} grasp states from {args.grasp_states}")
    else:
        print(f"Warning: grasp states not found at {args.grasp_states}")

    # Create training env with place_safe reward
    print("\nCreating training environment (reward_type='place_safe')...")
    train_env = DummyVecEnv([functools.partial(
        make_env,
        reward_type='place_safe',
        grasp_states=grasp_states,
        vision_mode=True,
        domain_randomize=False,
        release_threshold=0.10,
    )])

    # Load V59 VecNormalize stats (BC warmstart uses V59's normalization)
    print(f"Loading VecNormalize stats from {V59_VECNORM_PATH}")
    train_env = VecNormalize.load(V59_VECNORM_PATH, train_env)
    train_env.norm_reward = True
    train_env.training = True
    train_env = VecTransposeImage(train_env)

    # Create or load model
    if args.algorithm == 'ppo':
        model = create_ppo_model(args, stage_cfg, train_env, demo_obs, demo_actions)
    else:
        model = create_sac_model(args, stage_cfg, train_env)

    # Freeze BN running stats (V59 root cause fix)
    if stage_cfg["freeze_bn"]:
        frozen_count = freeze_bn_running_stats(model)
        print(f"BN running stats FROZEN: {frozen_count} BatchNorm layers locked")

    # Freeze backbone for stage 1 (prevent PPO from destroying BC features)
    if stage_cfg["freeze_backbone"]:
        total, frozen = freeze_backbone_params(model)
        print(f"Backbone FROZEN: {total} total params, {frozen} frozen")

    # Build callbacks
    callbacks = build_callbacks(args, stage_cfg, args.save_path)

    # Train
    print(f"\n{'='*70}")
    print(f"Starting training: {stage_cfg['total_timesteps']:,} steps")
    print(f"{'='*70}\n")

    start_time = time.time()
    try:
        model.learn(
            total_timesteps=stage_cfg["total_timesteps"],
            callback=callbacks,
            progress_bar=False,
        )
        safety_stop_triggered = False
        decoupling_detected = False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        safety_stop_triggered = False
        decoupling_detected = False
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        safety_stop_triggered = "safety_stop" in str(e).lower()
        decoupling_detected = "decoupling" in str(e).lower()
        raise

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.1f} hours")

    # Save final model + normalization stats
    model.save(os.path.join(args.save_path, 'final_model'))
    train_env.save(os.path.join(args.save_path, 'vec_normalize.pkl'))

    # Save training_results.json for decision tree evidence extraction
    save_training_results(
        args.save_path, args, stage_cfg, model, train_env,
        safety_stop_triggered, decoupling_detected
    )

    print(f"\nTraining complete! Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
