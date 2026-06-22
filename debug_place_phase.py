#!/usr/bin/env python3
"""Debug hierarchical eval place phase: track all conditions for place_success.

Logs for each place step:
  - block_target_dist (need < 5cm)
  - block_z / block_on_table (need z < 0.25)
  - gripper_opening / gripper_open (need > 0.02)
  - _place_gravcomp_active
  - _place_success
  - action (arm_delta, gripper_cmd)
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

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

TABLE_Z = 0.22
TARGET = np.array([0.5, 0.3, 0.2])
PLACE_THRESHOLD = 0.05
MAX_STEPS = 500
N_EPISODES = 20


def make_env():
    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    return FlattenObs(env)


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


def main():
    print("Loading grasp model...")
    grasp_model, grasp_vec_env = load_model(
        GRASP_MODEL_PATH, GRASP_VECNORM_PATH, make_env)
    print("Loading place model...")
    place_model, place_vec_env = load_model(
        PLACE_MODEL_PATH, PLACE_VECNORM_PATH, make_env)

    raw_env = DummyVecEnv([make_env])
    inner_env = raw_env.envs[0].env.unwrapped
    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    np.random.seed(42)
    try:
        raw_env.seed(42)
    except Exception:
        pass

    grab_count = 0
    place_count = 0
    phase_switch_count = 0

    for ep in range(N_EPISODES):
        inner_env.place_mode = False
        inner_env._place_gravcomp_active = False
        inner_env.reward_type = "dense"
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        min_dist = float("inf")
        final_dist = float("inf")
        max_lift = 0.0
        prev_info = None
        place_debug_lines = []

        for step in range(MAX_STEPS):
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                phase_switch_count += 1
                inner_env.place_mode = True
                inner_env._place_gravcomp_active = True
                inner_env.snap_block_to_hand()
                inner_env._arm_target = inner_env.data.qpos[
                    inner_env._arm_qpos_adrs].copy()
                inner_env._gripper_target = float(
                    inner_env.data.qpos[inner_env._finger_qpos_adrs].mean())
                inner_env.reward_type = "place_only"
                inner_env._place_approach_bonus_given = False
                inner_env._place_proximity_15_given = False
                inner_env._place_proximity_10_given = False
                inner_env._place_success = False
                inner_env._prev_block_target_dist = None
                inner_env._prev_block_height = None

                flatten_wrapper = raw_env.envs[0]
                inner_obs = inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

            if phase == "place":
                obs = place_vec_env.normalize_obs(raw_obs)
            else:
                obs = grasp_vec_env.normalize_obs(raw_obs)

            action, _ = policy.predict(obs, info=prev_info, deterministic=True)
            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            i = info[0]

            block_pos = i.get("block_position", None)
            if block_pos is not None:
                dist = float(np.linalg.norm(np.array(block_pos) - TARGET))
                min_dist = min(min_dist, dist)
                final_dist = dist

            block_h = float(i.get("block_height", 0.0))
            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift

            # Log place phase details
            if phase == "place" and first_place_step is not None:
                place_step = step - first_place_step
                block_z = float(block_pos[2]) if block_pos is not None else 0
                gripper_opening = float(i.get("gripper_opening", 0))
                gripper_open = gripper_opening > 0.02
                block_on_table = block_z < TABLE_Z + 0.03
                gravcomp = inner_env._place_gravcomp_active
                place_success = inner_env._place_success
                gripper_cmd = action[0][7]

                # Print every 10 steps or when close to target
                if place_step % 10 == 0 or dist < 0.10 or place_success:
                    line = (f"  ps={place_step:3d} dist={dist*100:5.1f}cm "
                            f"z={block_z:.3f} on_table={'Y' if block_on_table else 'N'} "
                            f"grip={gripper_opening:.3f} open={'Y' if gripper_open else 'N'} "
                            f"grav={'Y' if gravcomp else 'N'} "
                            f"gcmd={gripper_cmd:+.2f} "
                            f"success={'Y' if place_success else 'N'}")
                    place_debug_lines.append(line)

            if done[0]:
                break

        grabbed = max_lift > 0.03
        placed = final_dist < PLACE_THRESHOLD
        if grabbed:
            grab_count += 1
        if placed:
            place_count += 1

        status = f"Ep {ep:2d}: lift={max_lift*100:5.1f}cm dist={final_dist*100:5.1f}cm "
        status += f"min={min_dist*100:5.1f}cm ps={'  -' if first_place_step is None else f'{first_place_step:3d}'} "
        status += f"grab={'Y' if grabbed else 'N'} place={'Y' if placed else 'N'}"
        print(status)

        # Print place phase debug for episodes that switched
        if first_place_step is not None and place_debug_lines:
            for line in place_debug_lines[-10:]:  # last 10 lines
                print(line)

    raw_env.close()

    print(f"\n{'='*64}")
    print(f"Grab:  {grab_count}/{N_EPISODES} ({100*grab_count/N_EPISODES:.0f}%)")
    print(f"Place: {place_count}/{N_EPISODES} ({100*place_count/N_EPISODES:.0f}%)")
    print(f"Phase switches: {phase_switch_count}/{N_EPISODES}")


if __name__ == "__main__":
    main()
