#!/usr/bin/env python3
"""Generate LeRobot-format dataset for VLA fine-tuning on Panda pick-and-place.

Uses the gravity-compensated scripted policy (generate_pick_place_trajectory) to
produce pick-place trajectories, replays them in a MuJoCo simulation with
rendering, and stores (state, image, action) frames in LeRobot format suitable
for SmolVLA fine-tuning.

Usage:
    conda run -n vla python generate_vla_place_data.py --n_demos 20
"""
import os
os.environ["MUJOCO_GL"] = "egl"

import sys
import argparse
import shutil
from contextlib import contextmanager

import numpy as np
import mujoco
from mujoco import MjModel, MjData

# Make lerobot importable
sys.path.insert(0, "/home/w/vla_workspace/lerobot/src")
sys.path.insert(0, "/home/w/vla_workspace")

from lerobot.datasets import LeRobotDataset
from scripted_policy import generate_pick_place_trajectory


SCENE_XML = "/home/w/mujoco/ros2_ws/src/panda_mujoco_ros2/mjcf/franka_emika_panda/scene.xml"
TASK_DESCRIPTION = "pick up the red block and place it on the target"
CONTROL_DT = 0.05  # 20 Hz
FPS = 20


@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout to hide verbose scripted-policy output."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def get_features():
    """Define LeRobot feature specifications for SmolVLA."""
    joint_names = [f"joint{i}" for i in range(1, 8)]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": joint_names + ["gripper"],
        },
        "observation.images.camera1": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera2": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera3": {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": joint_names + ["gripper"],
        },
    }


def generate_dataset(n_demos, output_dir, stride=1):
    """Generate a LeRobot dataset with pick-place demonstrations.

    Args:
        n_demos: Number of successful episodes to generate.
        output_dir: Directory where the dataset will be stored.
        stride: Record every Nth trajectory step (1 = all steps). When > 1 the
            effective fps is reduced to FPS // stride so delta-timestamps stay
            consistent with the recorded frame spacing.
    """
    # LeRobotDataset.create requires a non-existing root directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        # Fallback for stubborn directories (e.g. leftover from async image writers)
        if os.path.exists(output_dir):
            import subprocess
            subprocess.run(["rm", "-rf", output_dir], check=False)

    # Create MuJoCo model, data, and renderer
    model = MjModel.from_xml_path(SCENE_XML)
    data = MjData(model)
    renderer = mujoco.Renderer(model, height=256, width=256)

    # Joint qpos address lookups
    arm_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}")
        for j in range(1, 8)
    ]
    arm_qpos_adrs = [model.jnt_qposadr[jid] for jid in arm_joint_ids]
    finger_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2"),
    ]
    finger_qpos_adrs = [model.jnt_qposadr[jid] for jid in finger_joint_ids]
    block_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "red_block_joint"
    )
    block_qpos_adr = model.jnt_qposadr[block_joint_id]

    # Effective fps accounts for stride so delta-timestamps remain valid
    effective_fps = max(1, FPS // stride) if stride > 1 else FPS

    # Create LeRobot dataset (images stored on disk, not encoded as video)
    dataset = LeRobotDataset.create(
        repo_id="panda_pickplace",
        fps=effective_fps,
        features=get_features(),
        root=output_dir,
        use_videos=False,
        image_writer_threads=4,
    )

    successful = 0
    total_frames = 0
    attempts = 0
    max_attempts = n_demos * 5

    while successful < n_demos and attempts < max_attempts:
        attempts += 1
        print(f"\n{'='*60}")
        print(f"Episode {successful + 1}/{n_demos} (attempt {attempts})")
        print(f"{'='*60}")

        # Generate trajectory (suppress verbose scripted-policy debug output)
        try:
            # Randomize initial block position (reduced range for IK success)
            block_x = np.random.uniform(0.42, 0.58)
            block_y = np.random.uniform(-0.08, 0.08)
            init_block_pos = [block_x, block_y, 0.24]
            with suppress_stdout():
                trajectory, traj_success, final_block_pos = (
                    generate_pick_place_trajectory(SCENE_XML, block_pos=init_block_pos)
                )
        except Exception as e:
            print(f"  ERROR generating trajectory: {e}")
            continue

        if not traj_success:
            print(f"  FAILED (scripted policy did not reach target), skipping")
            continue

        print(f"  Trajectory: {len(trajectory)} steps, success=True")

        # Replay trajectory in simulation with rendering
        try:
            # Reset to canonical initial state
            mujoco.mj_resetData(model, data)
            data.qpos[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
            data.qpos[7:9] = [0.04, 0.04]
            data.ctrl[:7] = [0, 0, 0, -1.57079, 0, 1.57079, 0.7854]
            data.ctrl[7] = 0.04
            # Use the same randomized block position as trajectory generation
            data.qpos[block_qpos_adr : block_qpos_adr + 7] = [
                block_x, block_y, 0.24, 1.0, 0.0, 0.0, 0.0
            ]
            mujoco.mj_forward(model, data)

            episode_frames = 0

            for step_idx, (qpos_target, ctrl_target) in enumerate(trajectory):
                # Record frame at stride intervals
                if step_idx % stride == 0:
                    # Render image (single render, replicated to 3 cameras)
                    # camera=-1 uses the default free camera (scene has no named cameras)
                    renderer.update_scene(data, camera=-1)
                    img = renderer.render().copy()

                    # State: [7 joint positions, 1 gripper position (mean of fingers)]
                    joint_pos = data.qpos[arm_qpos_adrs].copy()
                    gripper = np.array([data.qpos[finger_qpos_adrs].mean()])
                    state = np.concatenate([joint_pos, gripper]).astype(np.float32)

                    # Action: [7 absolute joint target positions, 1 gripper cmd]
                    # Using absolute positions instead of velocity deltas because
                    # the scripted policy's smooth IK interpolation produces
                    # near-zero deltas (>90% of frames < 0.01), causing the VLA
                    # to learn near-zero actions. Absolute positions have a much
                    # wider distribution, making the VLA learning signal stronger.
                    action = np.zeros(8, dtype=np.float32)
                    action[:7] = ctrl_target[:7].astype(np.float32)
                    action[7] = -1.0 if ctrl_target[7] < 0.02 else 1.0

                    dataset.add_frame({
                        "observation.state": state,
                        "observation.images.camera1": img,
                        "observation.images.camera2": img,
                        "observation.images.camera3": img,
                        "action": action,
                        "task": TASK_DESCRIPTION,
                    })
                    episode_frames += 1

                # Step simulation with gravity compensation
                data.ctrl[:7] = ctrl_target[:7]
                data.ctrl[7] = ctrl_target[7]
                mujoco.mj_forward(model, data)
                data.qfrc_applied[:7] = data.qfrc_bias[:7]
                mujoco.mj_step(model, data)
                data.qfrc_applied[:] = 0

            dataset.save_episode()
            successful += 1
            total_frames += episode_frames
            print(f"  OK: {episode_frames} frames recorded (total: {total_frames})")

        except Exception as e:
            print(f"  ERROR during replay: {e}")
            dataset.clear_episode_buffer()
            continue

    # Finalize dataset (writes parquet footers, flushes image writers)
    dataset.finalize()

    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"  Successful episodes: {successful}")
    print(f"  Total frames: {total_frames}")
    print(f"  Dataset path: {output_dir}")
    print(f"  Effective fps: {effective_fps}")
    print(f"{'='*60}")

    return successful, total_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LeRobot dataset for VLA fine-tuning on Panda pick-and-place"
    )
    parser.add_argument(
        "--n_demos", type=int, default=50,
        help="Number of successful demos to generate (default: 50)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/w/vla_workspace/data/panda_pickplace_lerobot_v3",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Record every Nth trajectory step (1=all steps, default: 1). "
             "Values > 1 reduce effective fps to FPS//stride.",
    )
    args = parser.parse_args()

    generate_dataset(args.n_demos, args.output_dir, args.stride)
