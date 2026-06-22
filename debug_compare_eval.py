#!/usr/bin/env python3
"""Compare place-only vs hierarchical eval in detail.

Runs both eval modes with detailed logging to find why the place model
succeeds in place-only eval (60%) but fails in hierarchical eval (0%).

Logs for each mode:
  - Initial state when place phase starts (arm_qpos, hand_pos, block_pos,
    block_target_dist, _arm_target, _gripper_target)
  - Raw and normalized observations
  - First 10 place-model actions and their effects
  - Final result (min_dist, final_dist, ep_len)
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import pickle
import numpy as np
import gymnasium
import gym_env
from gym_env.wrappers import FlattenObs
from hierarchical_policy import HierarchicalPickPlacePolicy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/vec_normalize.pkl"
PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v10/best/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v10/vec_normalize.pkl"
GRASP_STATES_PATH = "/home/w/vla_workspace/outputs/grasp_states_500.pkl"

TABLE_Z = 0.22
TARGET = np.array([0.5, 0.3, 0.2])
PLACE_THRESHOLD = 0.05
MAX_STEPS = 200
N_EPISODES = 5


def make_env(place_mode=False, grasp_states=None):
    kwargs = dict(reward_type="dense", gravity_comp=True)
    if place_mode:
        kwargs["place_mode"] = True
    if grasp_states is not None:
        kwargs["grasp_states"] = grasp_states
        kwargs["target_pos"] = TARGET
    return FlattenObs(gymnasium.make("PandaVLA-v0", **kwargs))


def load_model(model_path, vecnorm_path, env_factory):
    vec_env = DummyVecEnv([env_factory])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0)
        vec_env.training = False
    model = PPO.load(model_path, env=vec_env, device="cpu")
    return model, vec_env


def print_state(inner_env, label):
    arm_qpos = inner_env.data.qpos[inner_env._arm_qpos_adrs].copy()
    hand_pos = inner_env.data.xpos[inner_env._hand_id].copy()
    block_pos = inner_env.data.xpos[inner_env._red_block_id].copy()
    block_target_dist = float(np.linalg.norm(block_pos - TARGET))
    print(f"\n  [{label}]")
    print(f"    arm_qpos:       {arm_qpos}")
    print(f"    hand_pos:       {hand_pos}")
    print(f"    block_pos:      {block_pos}")
    print(f"    block_tgt_dist: {block_target_dist*100:.1f}cm")
    print(f"    _arm_target:    {inner_env._arm_target}")
    print(f"    _gripper_tgt:   {inner_env._gripper_target:.4f}")
    print(f"    arm_target_diff: {np.linalg.norm(inner_env._arm_target - arm_qpos):.4f}")
    return block_target_dist


def run_place_only(grasp_states, n_episodes):
    """Run place-only eval (place_mode=True from reset)."""
    print("\n" + "=" * 64)
    print("PLACE-ONLY EVAL (place_mode=True from reset)")
    print("=" * 64)

    raw_env = DummyVecEnv([lambda: make_env(place_mode=True, grasp_states=grasp_states)])
    eval_env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    place_vec_env = DummyVecEnv([lambda: make_env(place_mode=True, grasp_states=grasp_states)])
    place_vec_env = VecNormalize.load(PLACE_VECNORM_PATH, place_vec_env)
    place_vec_env.norm_reward = False
    place_vec_env.training = False
    model = PPO.load(PLACE_MODEL_PATH, env=place_vec_env, device="cpu")

    eval_env.obs_rms = place_vec_env.obs_rms
    eval_env.ret_rms = place_vec_env.ret_rms
    eval_env.training = False

    inner_env = raw_env.envs[0].env.unwrapped
    np.random.seed(42)
    raw_env.seed(42)

    successes = 0
    for ep in range(n_episodes):
        obs = eval_env.reset()
        init_dist = print_state(inner_env, f"Place-only Ep {ep} init")
        print(f"    norm_obs:       {obs[0]}")

        min_dist = init_dist
        final_dist = init_dist
        ep_len = 0

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            if step < 5:
                arm_delta = action[0][:7] * 1.0 * inner_env.control_dt
                gripper_cmd = action[0][7]
                block_pos = inner_env.data.xpos[inner_env._red_block_id].copy()
                dist = float(np.linalg.norm(block_pos - TARGET))
                print(f"    step {step}: action={action[0]}, arm_delta={arm_delta}, "
                      f"gripper={gripper_cmd:.2f}, dist={dist*100:.1f}cm")

            obs, reward, done, info = eval_env.step(action)
            ep_len += 1
            i = info[0]
            block_pos = i.get("block_position", None)
            if block_pos is not None:
                dist = float(np.linalg.norm(np.array(block_pos) - TARGET))
                min_dist = min(min_dist, dist)
                final_dist = dist

            if done[0]:
                break

        success = final_dist < PLACE_THRESHOLD
        if success:
            successes += 1
        print(f"  Result: ep_len={ep_len}, min_dist={min_dist*100:.1f}cm, "
              f"final_dist={final_dist*100:.1f}cm, success={'Y' if success else 'N'}")

    print(f"\nPlace-only success: {successes}/{n_episodes} "
          f"({100*successes/n_episodes:.0f}%)")
    raw_env.close()
    return successes


def run_hierarchical(grasp_model, grasp_vec_env, place_model, place_vec_env,
                     n_episodes):
    """Run hierarchical eval (grasp model -> place model)."""
    print("\n" + "=" * 64)
    print("HIERARCHICAL EVAL (grasp model -> place model)")
    print("=" * 64)

    raw_env = DummyVecEnv([make_env])
    inner_env = raw_env.envs[0].env.unwrapped
    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    np.random.seed(42)
    try:
        raw_env.seed(42)
    except Exception:
        pass

    successes = 0
    for ep in range(n_episodes):
        inner_env.place_mode = False
        inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        min_dist = float("inf")
        final_dist = float("inf")
        prev_info = None

        for step in range(MAX_STEPS):
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                inner_env.place_mode = True
                inner_env._place_gravcomp_active = True
                inner_env.snap_block_to_hand()

                inner_env._arm_target = inner_env.data.qpos[
                    inner_env._arm_qpos_adrs].copy()
                inner_env._gripper_target = float(
                    inner_env.data.qpos[inner_env._finger_qpos_adrs].mean())

                flatten_wrapper = raw_env.envs[0]
                inner_obs = inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

                init_dist = print_state(inner_env, f"Hierarchical Ep {ep} place_phase_start (step={step})")
                print(f"    raw_obs:        {raw_obs[0]}")

            if phase == "place":
                obs = place_vec_env.normalize_obs(raw_obs)
            else:
                obs = grasp_vec_env.normalize_obs(raw_obs)

            if phase == "place" and first_place_step is not None:
                place_step = step - first_place_step
                if place_step < 5:
                    print(f"    norm_obs:       {obs[0]}")

            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            if phase == "place" and first_place_step is not None:
                place_step = step - first_place_step
                if place_step < 5:
                    arm_delta = action[0][:7] * 1.0 * inner_env.control_dt
                    gripper_cmd = action[0][7]
                    block_pos = inner_env.data.xpos[inner_env._red_block_id].copy()
                    dist = float(np.linalg.norm(block_pos - TARGET))
                    print(f"    place_step {place_step}: action={action[0]}, "
                          f"arm_delta={arm_delta}, gripper={gripper_cmd:.2f}, "
                          f"dist={dist*100:.1f}cm")

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            i = info[0]

            block_pos = i.get("block_position", None)
            if block_pos is not None:
                dist = float(np.linalg.norm(np.array(block_pos) - TARGET))
                min_dist = min(min_dist, dist)
                final_dist = dist

            if done[0]:
                break

        success = final_dist < PLACE_THRESHOLD
        if success:
            successes += 1
        print(f"  Result: place_step={first_place_step}, ep_len={step+1}, "
              f"min_dist={min_dist*100:.1f}cm, final_dist={final_dist*100:.1f}cm, "
              f"success={'Y' if success else 'N'}")

    print(f"\nHierarchical success: {successes}/{n_episodes} "
          f"({100*successes/n_episodes:.0f}%)")
    raw_env.close()
    return successes


def main():
    with open(GRASP_STATES_PATH, 'rb') as f:
        grasp_states = pickle.load(f)
    print(f"Loaded {len(grasp_states)} grasp states")

    # Place-only eval
    run_place_only(grasp_states, N_EPISODES)

    # Hierarchical eval
    print("\nLoading grasp model...")
    grasp_model, grasp_vec_env = load_model(
        GRASP_MODEL_PATH, GRASP_VECNORM_PATH, make_env)
    print("Loading place model...")
    place_model, place_vec_env = load_model(
        PLACE_MODEL_PATH, PLACE_VECNORM_PATH, make_env)
    run_hierarchical(grasp_model, grasp_vec_env, place_model, place_vec_env,
                     N_EPISODES)


if __name__ == "__main__":
    main()
