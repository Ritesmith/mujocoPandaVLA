#!/usr/bin/env python3
"""Debug hierarchical eval: simulate place phase with grasp_states init.

This script mimics the hierarchical eval's place phase but uses
grasp_states for initialization (instead of running the grasp model).
It prints detailed observations to understand why the place model
fails in hierarchical eval but succeeds in place-only eval.

Key difference from debug_place_eval.py:
- Uses place_mode=False initially (like hierarchical eval)
- Manually activates place_mode + snap_block_to_hand (like hierarchical eval)
- Uses place_vec_env.normalize_obs (like hierarchical eval)
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
MAX_STEPS = 200
N_EPISODES = 5


def make_env():
    """Create env like eval_hierarchical.py (place_mode=False, reward_type=dense)."""
    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    return FlattenObs(env)


def main():
    # Load grasp states
    with open(GRASP_STATES_PATH, 'rb') as f:
        grasp_states = pickle.load(f)
    print(f"Loaded {len(grasp_states)} grasp states")

    # Create env like eval_hierarchical.py
    raw_env = DummyVecEnv([make_env])
    _inner_env = raw_env.envs[0].env.unwrapped

    # Load place model with its VecNormalize stats
    place_vec_env = DummyVecEnv([make_env])
    place_vec_env = VecNormalize.load(PLACE_VECNORM_PATH, place_vec_env)
    place_vec_env.norm_reward = False
    place_vec_env.training = False
    model = PPO.load(PLACE_MODEL_PATH, env=place_vec_env, device="cpu")

    np.random.seed(42)
    raw_env.seed(42)

    successes = 0
    for ep in range(N_EPISODES):
        # Reset env (place_mode=False, like hierarchical eval)
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()

        # Load a grasp state
        state = grasp_states[ep * 30]  # spread out
        _inner_env.data.qpos[_inner_env._arm_qpos_adrs] = state['arm_qpos']
        _inner_env.data.ctrl[_inner_env._arm_actuator_ids] = state['arm_qpos']
        _inner_env.data.qpos[_inner_env._finger_qpos_adrs] = state['finger_qpos']
        _inner_env.data.ctrl[_inner_env._gripper_actuator_id] = 0.0  # closed

        # CRITICAL: update _arm_target and _gripper_target to match
        # the manually set qpos. Without this, step() adds arm_delta
        # to the old _arm_target (from reset), causing the arm to jump.
        _inner_env._arm_target = state['arm_qpos'].copy()
        _inner_env._gripper_target = float(np.array(state['finger_qpos']).mean())

        # Set block position (like snap_block_to_hand)
        hand_pos = state['hand_pos']
        _inner_env.data.qpos[_inner_env._red_block_qpos_adr + 0] = hand_pos[0]
        _inner_env.data.qpos[_inner_env._red_block_qpos_adr + 1] = hand_pos[1]
        _inner_env.data.qpos[_inner_env._red_block_qpos_adr + 2] = hand_pos[2] - 0.05

        import mujoco
        mujoco.mj_forward(_inner_env.model, _inner_env.data)

        # Activate place_mode (like hierarchical eval)
        _inner_env.place_mode = True
        _inner_env._place_gravcomp_active = True
        _inner_env.snap_block_to_hand()

        # Get observation like hierarchical eval
        flatten_wrapper = raw_env.envs[0]
        inner_obs = _inner_env._get_obs()
        new_flat = flatten_wrapper.observation(inner_obs)
        raw_obs = new_flat[np.newaxis, :].astype(np.float32)

        # Print initial state
        block_pos = _inner_env.data.xpos[_inner_env._red_block_id].copy()
        target_pos = np.array([0.5, 0.3, 0.2])
        init_dist = float(np.linalg.norm(block_pos - target_pos))
        print(f"\nEp {ep}: init_dist={init_dist*100:.1f}cm, hand_pos={hand_pos}")
        print(f"  raw_obs={raw_obs[0]}")

        # Normalize obs like hierarchical eval
        obs = place_vec_env.normalize_obs(raw_obs)
        print(f"  norm_obs={obs[0]}")

        # Run place model
        min_dist = init_dist
        final_dist = init_dist
        ep_len = 0

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            if step < 3:
                print(f"  step {step}: action={action[0]}")

            raw_obs, reward, done, info = raw_env.step(action)
            ep_len += 1

            i = info[0]
            block_pos = i.get("block_position", None)
            if block_pos is not None:
                dist = float(np.linalg.norm(block_pos - target_pos))
                min_dist = min(min_dist, dist)
                final_dist = dist

            obs = place_vec_env.normalize_obs(raw_obs)

            if done[0]:
                break

        success = final_dist < PLACE_THRESHOLD
        if success:
            successes += 1

        print(f"  result: ep_len={ep_len}, min_dist={min_dist*100:.1f}cm, "
              f"final_dist={final_dist*100:.1f}cm, success={'Y' if success else 'N'}")

    print(f"\nPlace success: {successes}/{N_EPISODES} ({100*successes/N_EPISODES:.0f}%)")


if __name__ == "__main__":
    main()
