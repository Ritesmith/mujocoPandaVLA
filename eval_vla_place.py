#!/usr/bin/env python3
"""Evaluate fine-tuned SmolVLA on Panda pick-and-place task.

Loads the fine-tuned VLA model and runs N episodes in the MuJoCo
environment, tracking grab rate and place rate.

Usage:
    python eval_vla_place.py
    python eval_vla_place.py --model_path outputs/smolvla_place_finetuned/final
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import argparse
import numpy as np
import torch
import gymnasium

# LeRobot source
sys.path.insert(0, "/home/w/vla_workspace/lerobot/src")

import gym_env  # noqa: F401  registers PandaVLA-v0
from gym_env.wrappers import FlattenObs


MODEL_PATH = "/home/w/vla_workspace/outputs/smolvla_place_finetuned_v3/final"
TASK_INSTRUCTION = "pick up the red block and place it on the target"
TABLE_Z = 0.22
LIFT_THRESHOLD = 0.03   # m
PLACE_THRESHOLD = 0.05  # m
MAX_STEPS = 500
N_EPISODES = 10
SEED = 42


def load_vla_model(model_path):
    """Load fine-tuned SmolVLA model with pre/post processors."""
    from lerobot.policies.smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors

    print(f"Loading VLA model from: {model_path}")

    # Determine device: use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    policy = SmolVLAPolicy.from_pretrained(model_path)
    # Override device in config if CUDA is not available
    if device == "cpu":
        policy.config.device = "cpu"
    policy.eval()
    policy = policy.to(device)

    # Build processor overrides for CPU fallback: the saved preprocessor
    # config hardcodes device='cuda' for the device_processor step, which
    # fails when CUDA is unavailable. Override it to match the runtime device.
    preprocessor_overrides = None
    postprocessor_overrides = None
    if device == "cpu":
        preprocessor_overrides = {"device_processor": {"device": "cpu"}}
        postprocessor_overrides = {"device_processor": {"device": "cpu"}}

    preprocess, postprocess = make_pre_post_processors(
        policy.config, model_path,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )

    img_key = list(policy.config.image_features.keys())[0]
    state_dim = policy.config.input_features["observation.state"].shape[0]
    action_dim = policy.config.output_features["action"].shape[0]

    print(f"  Image key: {img_key}")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")

    return policy, preprocess, postprocess, img_key, state_dim, action_dim


def vla_inference(policy, preprocess, postprocess, img_key, state_dim,
                  image, state_np, task):
    """Run VLA inference and return full action chunk.

    Returns:
        actions: np.ndarray of shape (n_action_steps, action_dim)
                 containing the full action chunk for sequential execution.
    """
    from torchvision.transforms import ToTensor
    from PIL import Image

    pil_image = Image.fromarray(image)
    img_tensor = ToTensor()(pil_image)  # [C, H, W] in [0, 1]

    # Pad/truncate state to state_dim
    state = np.zeros(state_dim, dtype=np.float32)
    n_copy = min(len(state_np), state_dim)
    state[:n_copy] = state_np[:n_copy]
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # Build obs dict — replicate single camera to all image keys
    obs = {}
    for key in policy.config.image_features.keys():
        obs[key] = img_tensor
    obs["observation.state"] = state_tensor
    obs["task"] = task

    # Preprocess
    processed = preprocess(obs)

    # Inference
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(processed)

    # Postprocess (denormalize)
    action_final = postprocess(action_chunk)

    # Return full chunk: (batch, n_action_steps, action_dim) -> (n_action_steps, action_dim)
    actions = action_final[0].cpu().numpy()
    return actions


def vla_action_to_env_action(vla_action, current_joint_pos, control_dt=0.05):
    """Convert VLA action (absolute joint positions) to env action (velocity deltas).

    The VLA now outputs absolute target joint positions (7D) + gripper cmd (1D).
    The environment expects velocity deltas (rad/s, clipped to [-1, 1]) + gripper.

    Gripper convention mismatch:
    - Training data: 1.0 = open, -1.0 = close
    - Environment:    >0  = close, <0  = open
    We invert the gripper action to bridge this convention gap.

    Args:
        vla_action: VLA output array (8D: 7 joint positions + 1 gripper).
        current_joint_pos: Current 7 joint positions (arm_target) from the env.
        control_dt: Environment control timestep (for velocity conversion).
    """
    action = np.zeros(8, dtype=np.float32)
    n_arm = min(len(vla_action), 7)

    # Convert absolute target positions to velocity deltas
    target_pos = vla_action[:n_arm]
    delta = (target_pos - current_joint_pos[:n_arm]) / control_dt
    action[:n_arm] = np.clip(delta, -1.0, 1.0)

    # Gripper command: invert convention (training: 1=open, env: 1=close)
    if len(vla_action) > 7:
        action[7] = -vla_action[7]
    elif len(vla_action) > 0:
        action[7] = -vla_action[-1]

    return action


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLA pick-place")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES)
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--task', type=str, default=TASK_INSTRUCTION)
    args = parser.parse_args()

    # Load VLA model
    policy, preprocess, postprocess, img_key, state_dim, action_dim = \
        load_vla_model(args.model_path)

    # Create environment (with rendering for VLA)
    env = gymnasium.make(
        "PandaVLA-v0",
        render_mode="rgb_array",
        reward_type="dense",
        gravity_comp=True,
        image_size=256,
    )

    np.random.seed(SEED)

    grab_flags, place_flags, pickplace_flags = [], [], []
    max_lifts, final_dists = [], []

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=SEED + ep)
        max_lift = 0.0
        block_target_dist = float("inf")
        ep_reward = 0.0

        # Action chunking: run VLA inference once, execute multiple
        # actions from the chunk before re-inferring. This dramatically
        # speeds up evaluation on CPU (VLA outputs ~16 actions/chunk).
        action_chunk = None
        chunk_idx = 0

        for step in range(args.max_steps):
            # Re-run inference only when chunk is exhausted
            if action_chunk is None or chunk_idx >= len(action_chunk):
                # Get current image and state from env
                image = env.render()
                if image is None:
                    image = env.unwrapped._render_image()

                # Extract state: 7 joints + 1 gripper
                joint_pos = env.unwrapped.data.qpos[
                    env.unwrapped._arm_qpos_adrs
                ].copy()
                gripper = env.unwrapped.data.qpos[
                    env.unwrapped._finger_qpos_adrs
                ].mean()
                state_np = np.concatenate(
                    [joint_pos, [gripper]]
                ).astype(np.float32)

                # VLA inference -> full action chunk
                try:
                    action_chunk = vla_inference(
                        policy, preprocess, postprocess, img_key, state_dim,
                        image, state_np, args.task
                    )
                    chunk_idx = 0
                except Exception as e:
                    print(f"  Ep {ep} step {step}: VLA inference error: {e}")
                    break

            # Use current action from chunk
            vla_action = action_chunk[chunk_idx]
            chunk_idx += 1

            # Directly set arm_target to VLA's predicted position.
            # The training data was generated with direct position control
            # (data.ctrl = ctrl_target), so we match that by setting the
            # env's internal _arm_target directly. We then pass a zero
            # velocity action so step() doesn't add any delta, and only
            # the gripper command is forwarded.
            env.unwrapped._arm_target[:7] = np.clip(
                vla_action[:7],
                env.unwrapped._arm_joint_ranges[:, 0],
                env.unwrapped._arm_joint_ranges[:, 1],
            )

            # Build env action: zero arm velocity + gripper command
            # Gripper convention: training 1=open, env 1=close → invert
            env_action = np.zeros(8, dtype=np.float32)
            if len(vla_action) > 7:
                env_action[7] = -vla_action[7]
            else:
                env_action[7] = 0.0

            # Step environment
            obs, reward, terminated, truncated, info = env.step(env_action)
            ep_reward += float(reward)

            # Track metrics
            block_h = float(info.get("block_height", 0.0))
            block_target_dist = float(
                info.get("block_target_distance", block_target_dist)
            )
            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift

            if terminated or truncated:
                break

        grabbed = max_lift > LIFT_THRESHOLD
        placed = block_target_dist < PLACE_THRESHOLD
        pickplace = grabbed and placed

        grab_flags.append(grabbed)
        place_flags.append(placed)
        pickplace_flags.append(pickplace)
        max_lifts.append(max_lift)
        final_dists.append(block_target_dist)

        print(f"Ep {ep:2d}: max_lift={max_lift*100:5.1f}cm  "
              f"final_dist={block_target_dist*100:5.1f}cm  "
              f"reward={ep_reward:.1f}  "
              f"grab={'Y' if grabbed else 'N'} place={'Y' if placed else 'N'}")

    env.close()

    # Summary
    print()
    print("=" * 64)
    print("VLA Pick-and-Place Summary")
    print("=" * 64)
    print(f"Model: {args.model_path}")
    print(f"Task:  {args.task}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Grab  (lift>{LIFT_THRESHOLD*100:.0f}cm)   : "
          f"{sum(grab_flags)}/{args.n_episodes} "
          f"({100*sum(grab_flags)/args.n_episodes:.0f}%)")
    print(f"Place (dist<{PLACE_THRESHOLD*100:.0f}cm)  : "
          f"{sum(place_flags)}/{args.n_episodes} "
          f"({100*sum(place_flags)/args.n_episodes:.0f}%)")
    print(f"Pick+Place (both)     : "
          f"{sum(pickplace_flags)}/{args.n_episodes} "
          f"({100*sum(pickplace_flags)/args.n_episodes:.0f}%)")
    print(f"Mean max lift         : {np.mean(max_lifts)*100:.1f} cm")
    print(f"Best max lift         : {np.max(max_lifts)*100:.1f} cm")
    print(f"Mean final dist       : {np.mean(final_dists)*100:.1f} cm")
    print(f"Best final dist       : {np.min(final_dists)*100:.1f} cm")


if __name__ == "__main__":
    main()
