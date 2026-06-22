#!/usr/bin/env python3
"""Quick test for place_mode_realistic.

Verifies:
1. Block stays held when gripper is closed (no fall)
2. Block falls when gripper opens
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import sys
sys.path.insert(0, "/home/w/vla_workspace")

import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs

TABLE_Z = 0.22


def test_block_held():
    """Block should stay held when gripper remains closed."""
    env = gymnasium.make(
        "PandaVLA-v0",
        reward_type="place_only",
        place_mode_realistic=True,
        gravity_comp=True,
        target_pos=np.array([0.5, 0.3, 0.2]),
    )
    env = FlattenObs(env)

    obs, info = env.reset()
    initial_block_z = info["block_height"]
    initial_lift = initial_block_z - TABLE_Z
    print(f"Initial block height: {initial_block_z:.4f}m (lift={initial_lift:.4f}m)")

    # Run 100 steps with gripper held (action[7] = 0.0 = no change)
    action = np.zeros(8, dtype=np.float32)
    action[7] = 0.0  # hold gripper position

    min_lift = initial_lift
    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(action)
        lift = info["block_height"] - TABLE_Z
        if lift < min_lift:
            min_lift = lift
        if terminated:
            print(f"  Episode terminated at step {step}")
            break

    print(f"After 100 steps (gripper closed): min_lift={min_lift:.4f}m")
    held = min_lift > 0.02  # block should stay above 2cm lift
    print(f"  Block held: {'YES' if held else 'NO (block fell)'}")
    env.close()
    return held


def test_block_falls():
    """Block should fall when gripper opens."""
    env = gymnasium.make(
        "PandaVLA-v0",
        reward_type="place_only",
        place_mode_realistic=True,
        gravity_comp=True,
        target_pos=np.array([0.5, 0.3, 0.2]),
    )
    env = FlattenObs(env)

    obs, info = env.reset()
    initial_block_z = info["block_height"]
    initial_lift = initial_block_z - TABLE_Z
    print(f"Initial block height: {initial_block_z:.4f}m (lift={initial_lift:.4f}m)")

    # Run 50 steps with gripper held to settle
    action_close = np.zeros(8, dtype=np.float32)
    action_close[7] = 0.0  # hold
    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(action_close)

    settled_z = info["block_height"]
    print(f"After 50 steps (closed): block_z={settled_z:.4f}m (lift={settled_z - TABLE_Z:.4f}m)")

    # Open gripper (action[7] = -1.0 = open)
    action_open = np.zeros(8, dtype=np.float32)
    action_open[7] = -1.0  # open

    for step in range(100):
        obs, reward, terminated, truncated, info = env.step(action_open)
        if step % 20 == 0:
            lift = info["block_height"] - TABLE_Z
            print(f"  Step {step}: block_z={info['block_height']:.4f}m (lift={lift:.4f}m)")

    final_z = info["block_height"]
    final_lift = final_z - TABLE_Z
    print(f"After 100 steps (gripper open): block_z={final_z:.4f}m (lift={final_lift:.4f}m)")
    fell = final_lift < 0.01  # block should be on or near table
    print(f"  Block fell: {'YES' if fell else 'NO (block still held)'}")
    env.close()
    return fell


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Block held when gripper closed")
    print("=" * 60)
    held = test_block_held()

    print()
    print("=" * 60)
    print("Test 2: Block falls when gripper opens")
    print("=" * 60)
    fell = test_block_falls()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Block held when closed: {'PASS' if held else 'FAIL'}")
    print(f"Block falls when open:  {'PASS' if fell else 'FAIL'}")
