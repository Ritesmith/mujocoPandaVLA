#!/usr/bin/env python3
"""Benchmark env sampling speed to estimate RL training wall-clock time.

Tests 1/4/8/12 envs with both DummyVecEnv (single-process) and SubprocVecEnv
(multi-process). Runs 5000 random-action steps per config, reports steps/second
and estimated wall-clock time for 100K/5M/15M step training stages.

Usage:
    python bench_env_speed.py
    python bench_env_speed.py --n_steps 2000  # quick test
    python bench_env_speed.py --env_configs 1,4  # test specific configs
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import VisionObs

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecTransposeImage

from train_place_policy import make_env


GRASP_STATES_PATH = "/home/w/vla_workspace/outputs/grasp_states_500.pkl"


def load_grasp_states():
    """Load grasp states for realistic env initialization."""
    import pickle
    try:
        with open(GRASP_STATES_PATH, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, OSError):
        print(f"Warning: {GRASP_STATES_PATH} not found, using None")
        return None


def make_env_factory(grasp_states, env_idx=0):
    """Return a callable that creates a place_safe env for VecEnv."""
    def _factory():
        env = make_env(
            reward_type='place_safe',
            grasp_states=grasp_states,
            vision_mode=True,
            domain_randomize=False,
            release_threshold=0.10,
        )
        return env
    return _factory


def make_vec_env(n_envs, vec_type, grasp_states):
    """Create a vectorized env with n_envs environments."""
    factories = [make_env_factory(grasp_states, i) for i in range(n_envs)]
    if vec_type == "dummy":
        vec_env = DummyVecEnv(factories)
    else:
        vec_env = SubprocVecEnv(factories)
    # Wrap with VecNormalize and VecTransposeImage (same as training)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, norm_obs_keys=["state"])
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def benchmark_config(n_envs, vec_type, grasp_states, n_steps):
    """Run n_steps random actions and return (steps_per_second, total_steps)."""
    print(f"\n--- {vec_type} x {n_envs} envs ({n_steps} steps) ---")
    try:
        vec_env = make_vec_env(n_envs, vec_type, grasp_states)
    except Exception as e:
        print(f"  FAILED to create env: {e}")
        return None, 0

    obs = vec_env.reset()
    start_time = time.time()
    total_steps = 0
    try:
        for _ in range(n_steps):
            actions = np.random.uniform(-1, 1, size=(n_envs, 8)).astype(np.float32)
            obs, rewards, dones, infos = vec_env.step(actions)
            total_steps += n_envs
            # Auto-reset on done (VecEnv handles this internally)
    except Exception as e:
        print(f"  FAILED during stepping: {e}")
        vec_env.close()
        return None, total_steps
    elapsed = time.time() - start_time

    vec_env.close()
    sps = total_steps / elapsed if elapsed > 0 else 0
    print(f"  {total_steps} steps in {elapsed:.1f}s = {sps:.1f} steps/s")
    return sps, total_steps


def format_time(seconds):
    """Format seconds as human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def main():
    parser = argparse.ArgumentParser(description="Benchmark env sampling speed")
    parser.add_argument("--n_steps", type=int, default=5000,
                        help="Number of steps per config (default 5000)")
    parser.add_argument("--env_configs", type=str, default="1,4,8,12",
                        help="Comma-separated n_envs to test (default 1,4,8,12)")
    parser.add_argument("--vec_types", type=str, default="dummy,subproc",
                        help="Comma-separated vec env types (default dummy,subproc)")
    args = parser.parse_args()

    env_configs = [int(x) for x in args.env_configs.split(",")]
    vec_types = args.vec_types.split(",")

    print("=" * 70)
    print("RL From Scratch — Env Speed Benchmark")
    print("=" * 70)
    print(f"Reward type: place_safe")
    print(f"Steps per config: {args.n_steps}")
    print(f"Env configs: {env_configs}")
    print(f"Vec types: {vec_types}")

    grasp_states = load_grasp_states()
    print(f"Grasp states: {len(grasp_states) if grasp_states else 'None'}")

    results = {}
    for vec_type in vec_types:
        for n_envs in env_configs:
            sps, _ = benchmark_config(n_envs, vec_type, grasp_states, args.n_steps)
            results[(vec_type, n_envs)] = sps

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Steps/second")
    print("=" * 70)
    header = f"{'vec_type':<10} {'n_envs':<8} {'steps/s':<12} {'100K':<10} {'5M':<10} {'15M':<10}"
    print(header)
    print("-" * 70)

    for vec_type in vec_types:
        for n_envs in env_configs:
            sps = results.get((vec_type, n_envs))
            if sps and sps > 0:
                t_100k = format_time(100000 / sps)
                t_5m = format_time(5000000 / sps)
                t_15m = format_time(15000000 / sps)
                print(f"{vec_type:<10} {n_envs:<8} {sps:<12.1f} {t_100k:<10} {t_5m:<10} {t_15m:<10}")
            else:
                print(f"{vec_type:<10} {n_envs:<8} {'FAILED':<12} {'-':<10} {'-':<10} {'-':<10}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    best_sps = max((sps for sps in results.values() if sps), default=0)
    if best_sps > 0:
        best_config = [k for k, v in results.items() if v == best_sps][0]
        print(f"Best config: {best_config[0]} x {best_config[1]} envs = {best_sps:.1f} steps/s")
        t_15m = 15000000 / best_sps
        print(f"15M steps estimated time: {format_time(t_15m)}")
        if best_sps < 500:
            print("WARNING: <500 steps/s — SubprocVecEnv with more envs recommended")
        if t_15m > 86400:
            print(f"WARNING: 15M steps would take >1 day — consider reducing to 5M or using more envs")
        else:
            print(f"15M steps is feasible ({format_time(t_15m)})")


if __name__ == "__main__":
    main()
