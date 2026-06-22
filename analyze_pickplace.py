#!/usr/bin/env python3
"""Analyze pick-and-place capability of the best DAPG-500K model.

For each episode we track:
  - Grab  : max lift height > 0.03 m  (block was picked up)
  - Place : final block-target dist < 0.05 m  (block was placed)
  - Pick+Place : both conditions in the same episode (in order)

We also record the 3D geometry at the moment of closest hand-block
approach to diagnose WHY grabbing fails (approach from side vs top,
gripper opening, block lateral offset relative to fingers).
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium
import gym_env  # noqa: F401  registers PandaVLA-v0
from gym_env.wrappers import FlattenObs

DAPG_500K_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/best/best_model.zip"
DAPG_500K_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/vec_normalize.pkl"

LIFT_THRESHOLD = 0.03   # m above table (table_z=0.22)
PLACE_THRESHOLD = 0.05  # m block-target distance
TABLE_Z = 0.22
MAX_STEPS = 500
N_EPISODES = 20
SEED = 42


def make_env():
    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    return FlattenObs(env)


def main():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(DAPG_500K_MODEL_PATH, device="auto")

    def _env_fn():
        return make_env()

    eval_env = DummyVecEnv([_env_fn])
    eval_env = VecNormalize.load(DAPG_500K_VECNORM_PATH, eval_env)
    eval_env.norm_reward = False
    eval_env.training = False

    np.random.seed(SEED)
    try:
        eval_env.seed(SEED)
    except Exception:
        pass

    grab_flags, place_flags, pickplace_flags = [], [], []
    max_lifts, final_dists = [], []
    approach_records = []  # 3D geometry at closest approach per episode

    for ep in range(N_EPISODES):
        obs = eval_env.reset()
        ep_reward = 0.0
        max_lift = 0.0
        min_hand_dist = float("inf")
        approach_snapshot = None
        block_target_dist = float("inf")
        block_grabbed_at = None  # step index when first grabbed

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            i = info[0]

            block_h = float(i.get("block_height", 0.0))
            hand_dist = float(i.get("hand_block_distance", float("inf")))
            block_target_dist = float(i.get("block_target_distance", block_target_dist))

            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            if lift > LIFT_THRESHOLD and block_grabbed_at is None:
                block_grabbed_at = step

            # record closest approach geometry
            if hand_dist < min_hand_dist:
                min_hand_dist = hand_dist
                approach_snapshot = {
                    "step": step,
                    "hand_dist": hand_dist,
                    "block_h": block_h,
                    "lift": lift,
                    "hand_pos": np.array(i.get("hand_position", [0, 0, 0])),
                    "block_pos": np.array(i.get("block_position", [0, 0, 0])),
                    "gripper_opening": float(i.get("gripper_opening", 0.0)),
                }

            if done[0]:
                break

        grabbed = max_lift > LIFT_THRESHOLD
        placed = block_target_dist < PLACE_THRESHOLD
        # pick+place: grabbed at some point AND ended placed
        pickplace = grabbed and placed

        grab_flags.append(grabbed)
        place_flags.append(placed)
        pickplace_flags.append(pickplace)
        max_lifts.append(max_lift)
        final_dists.append(block_target_dist)
        if approach_snapshot is not None:
            approach_snapshot["ep_reward"] = ep_reward
            approach_snapshot["grabbed"] = grabbed
            approach_snapshot["placed"] = placed
            approach_snapshot["max_lift"] = max_lift
            approach_snapshot["final_dist"] = block_target_dist
            approach_records.append(approach_snapshot)

        print(f"Ep {ep:2d}: max_lift={max_lift*100:5.1f}cm  "
              f"final_dist={block_target_dist*100:5.1f}cm  "
              f"min_hand={min_hand_dist*100:5.1f}cm  "
              f"grab={'Y' if grabbed else 'N'} place={'Y' if placed else 'N'}")

    eval_env.close()

    # ---- Summary ----
    print()
    print("=" * 64)
    print("Pick-and-Place Summary  (DAPG-500K v4, best_model)")
    print("=" * 64)
    print(f"Episodes              : {N_EPISODES}")
    print(f"Grab  (lift>{LIFT_THRESHOLD*100:.0f}cm)   : "
          f"{sum(grab_flags)}/{N_EPISODES} "
          f"({100*sum(grab_flags)/N_EPISODES:.0f}%)")
    print(f"Place (dist<{PLACE_THRESHOLD*100:.0f}cm)  : "
          f"{sum(place_flags)}/{N_EPISODES} "
          f"({100*sum(place_flags)/N_EPISODES:.0f}%)")
    print(f"Pick+Place (both)     : "
          f"{sum(pickplace_flags)}/{N_EPISODES} "
          f"({100*sum(pickplace_flags)/N_EPISODES:.0f}%)")
    print(f"Mean max lift         : {np.mean(max_lifts)*100:.1f} cm")
    print(f"Best max lift         : {np.max(max_lifts)*100:.1f} cm")
    print(f"Mean final dist       : {np.mean(final_dists)*100:.1f} cm")
    print(f"Best final dist       : {np.min(final_dists)*100:.1f} cm")

    # ---- Approach geometry diagnostics ----
    print()
    print("-" * 64)
    print("Approach geometry at closest hand-block contact")
    print("-" * 64)
    print(f"{'ep':>3} {'step':>4} {'hand_dist':>9} {'lift':>6} "
          f"{'dx':>6} {'dy':>6} {'dz':>6} {'grip':>6} grab")
    for ep, a in enumerate(approach_records):
        h = a["hand_pos"]
        b = a["block_pos"]
        dx, dy, dz = h - b
        print(f"{ep:3d} {a['step']:4d} {a['hand_dist']*100:8.1f}cm "
              f"{a['lift']*100:5.1f}cm "
              f"{dx*100:+5.1f} {dy*100:+5.1f} {dz*100:+5.1f} "
              f"{a['gripper_opening']*1000:5.1f}mm "
              f"{'Y' if a['grabbed'] else 'N'}")

    # Average approach direction
    if approach_records:
        dzs = np.array([(a['hand_pos'][2] - a['block_pos'][2]) for a in approach_records])
        print()
        print(f"Mean hand-block dz at closest approach: {dzs.mean()*100:+.1f} cm")
        print(f"  (dz>0 means hand ABOVE block -> top grasp)")
        print(f"  (dz<0 means hand BELOW block -> side/under grasp)")
        from_above = np.mean(dzs > 0.02)
        print(f"Approaches from above (dz>2cm): "
              f"{int(from_above*N_EPISODES)}/{N_EPISODES}")


if __name__ == "__main__":
    main()
