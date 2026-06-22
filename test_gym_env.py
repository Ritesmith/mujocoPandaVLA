"""Test script for PandaVLAEnv gymnasium environment.

Creates the environment, resets it, runs 100 random steps,
and verifies observation shapes and no crashes.
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import gymnasium as gym


def test_env():
    print("=" * 60)
    print("PandaVLAEnv Test")
    print("=" * 60)

    # Register the environment
    gym.register(
        id="PandaVLA-v0",
        entry_point="gym_env.panda_vla_env:PandaVLAEnv",
    )
    print("[OK] Environment registered")

    # Create environment
    env = gym.make("PandaVLA-v0", render_mode="rgb_array", image_size=256)
    print(f"[OK] Environment created: {env}")

    # Reset
    obs, info = env.reset(seed=42)
    print(f"\n[OK] Environment reset")
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Image shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
    print(f"  Joint positions shape: {obs['joint_positions'].shape}, dtype: {obs['joint_positions'].dtype}")
    print(f"  Joint positions: {obs['joint_positions']}")
    print(f"  Gripper shape: {obs['gripper'].shape}, dtype: {obs['gripper'].dtype}")
    print(f"  Gripper value: {obs['gripper']}")
    print(f"  Info keys: {list(info.keys())}")

    # Verify observation shapes
    assert obs["image"].shape == (256, 256, 3), f"Unexpected image shape: {obs['image'].shape}"
    assert obs["image"].dtype == np.uint8, f"Unexpected image dtype: {obs['image'].dtype}"
    assert obs["joint_positions"].shape == (7,), f"Unexpected joint_positions shape: {obs['joint_positions'].shape}"
    assert obs["gripper"].shape == (1,), f"Unexpected gripper shape: {obs['gripper'].shape}"
    print("\n[OK] Observation shapes verified")

    # Verify observation spaces
    assert env.observation_space.contains(obs), "Observation not in observation space!"
    print("[OK] Observation within observation space")

    # Run 100 random steps
    print(f"\nRunning 100 random steps...")
    total_reward = 0.0
    n_terminated = 0
    n_truncated = 0

    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            n_terminated += 1
            print(f"  Step {step}: terminated (joint limit violation)")
            obs, info = env.reset()
        if truncated:
            n_truncated += 1
            print(f"  Step {step}: truncated (max steps reached)")
            obs, info = env.reset()

        if step % 25 == 0:
            print(f"  Step {step}: reward={reward:.4f}, joint_pos={obs['joint_positions'][:3]}, gripper={obs['gripper'][0]:.4f}")

    print(f"\n[OK] 100 random steps completed")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Terminations: {n_terminated}")
    print(f"  Truncations: {n_truncated}")

    # Final observation check
    print(f"\nFinal observation:")
    print(f"  Image shape: {obs['image'].shape}")
    print(f"  Image range: [{obs['image'].min()}, {obs['image'].max()}]")
    print(f"  Joint positions: {obs['joint_positions']}")
    print(f"  Gripper: {obs['gripper']}")

    # Verify render
    img = env.render()
    assert img is not None, "render() returned None"
    assert img.shape == (256, 256, 3), f"Unexpected render shape: {img.shape}"
    print(f"\n[OK] render() works, shape: {img.shape}")

    # Close
    env.close()
    print("[OK] Environment closed")

    # Test direct instantiation
    print("\n--- Testing direct instantiation ---")
    from gym_env.panda_vla_env import PandaVLAEnv
    env2 = PandaVLAEnv(render_mode="rgb_array", image_size=128)
    obs2, info2 = env2.reset(seed=123)
    assert obs2["image"].shape == (128, 128, 3), f"Unexpected image shape for 128px: {obs2['image'].shape}"
    print(f"[OK] Direct instantiation with image_size=128 works, image shape: {obs2['image'].shape}")
    env2.close()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_env()
