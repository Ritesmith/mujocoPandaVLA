#!/usr/bin/env python3
"""Debug place-only eval: run place model in place_mode and track behavior.

This script mimics the EvalCallback's place-only eval but adds detailed
logging to understand why place-only eval succeeds but hierarchical eval
fails.
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import pickle
import numpy as np
import gymnasium
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

GRASP_STATES_PATH = "/home/w/vla_workspace/outputs/grasp_states_500.pkl"
PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v10/best/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v10/vec_normalize.pkl"

TABLE_Z = 0.22
PLACE_THRESHOLD = 0.05
MAX_STEPS = 500
N_EPISODES = 10


def make_env(grasp_states=None):
    env = gymnasium.make(
        "PandaVLA-v0",
        reward_type="place_only",
        place_mode=True,
        gravity_comp=True,
        target_pos=np.array([0.5, 0.3, 0.2]),
        grasp_states=grasp_states,
    )
    return FlattenObs(env)


def main():
    # Load grasp states
    with open(GRASP_STATES_PATH, 'rb') as f:
        grasp_states = pickle.load(f)
    print(f"Loaded {len(grasp_states)} grasp states")

    # Create env (same as train_place_policy.py eval_env)
    raw_env = DummyVecEnv([lambda: make_env(grasp_states)])
    eval_env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Load place model with its VecNormalize stats
    place_vec_env = DummyVecEnv([lambda: make_env(grasp_states)])
    place_vec_env = VecNormalize.load(PLACE_VECNORM_PATH, place_vec_env)
    place_vec_env.norm_reward = False
    place_vec_env.training = False
    model = PPO.load(PLACE_MODEL_PATH, env=place_vec_env, device="cpu")

    # Sync eval_env's VecNormalize stats with the loaded stats
    # (This is what EvalCallback should do but might not)
    eval_env.obs_rms = place_vec_env.obs_rms
    eval_env.ret_rms = place_vec_env.ret_rms
    eval_env.training = False

    np.random.seed(42)
    raw_env.seed(42)

    successes = 0
    for ep in range(N_EPISODES):
        obs = eval_env.reset()
        ep_reward = 0.0
        min_dist = float("inf")
        final_dist = float("inf")
        ep_len = 0
        placed = False

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            ep_len += 1

            i = info[0]
            block_pos = i.get("block_position", None)
            if block_pos is not None:
                dist = float(np.linalg.norm(np.array(block_pos) - np.array([0.5, 0.3, 0.2])))
                min_dist = min(min_dist, dist)
                final_dist = dist

            if i.get("place_success", False):
                placed = True

            if done[0]:
                break

        if placed or final_dist < PLACE_THRESHOLD:
            successes += 1

        print(f"Ep {ep:2d}: reward={ep_reward:8.2f}  ep_len={ep_len:3d}  "
              f"min_dist={min_dist*100:5.1f}cm  final_dist={final_dist*100:5.1f}cm  "
              f"placed={'Y' if placed else 'N'}")

    print(f"\nPlace success: {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.0f}%)")


if __name__ == "__main__":
    main()
