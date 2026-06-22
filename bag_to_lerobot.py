#!/usr/bin/env python3
"""Convert ros2bag recordings to LeRobot training data format.

Usage: python3 bag_to_lerobot.py --bag_dir /path/to/bag --output_dir /path/to/output
"""
import argparse
import os
import json
import numpy as np
from PIL import Image


def parse_bag(bag_dir):
    """Parse ros2bag and extract obs-action pairs."""
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    except ImportError:
        print("ERROR: rosbag2_py not available. Install with: sudo apt install ros-jazzy-rosbag2-storage-default-plugins")
        return None

    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    data = {
        'images': [],
        'joint_positions': [],
        'actions': [],
        'timestamps': [],
        'gripper': [],
    }

    while reader.has_next():
        topic, msg, timestamp = reader.read_next()
        # Parse based on topic
        # This is a simplified version - actual implementation needs
        # proper deserialization of ROS2 messages

    return data


def save_lerobot_format(data, output_dir, task_name="pick_place"):
    """Save data in LeRobot format."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    meta = {
        "task": task_name,
        "robot_type": "panda",
        "fps": 30,
        "n_episodes": 1,
        "n_frames": len(data['images']),
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save frames as individual images
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    for i, img in enumerate(data['images']):
        Image.fromarray(img).save(os.path.join(img_dir, f'frame_{i:06d}.png'))

    # Save actions and states
    np.savez(
        os.path.join(output_dir, 'data.npz'),
        actions=np.array(data['actions']),
        joint_positions=np.array(data['joint_positions']),
        gripper=np.array(data['gripper']),
        timestamps=np.array(data['timestamps']),
    )

    print(f"Saved {len(data['images'])} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--task', default='pick_place')
    args = parser.parse_args()

    data = parse_bag(args.bag_dir)
    if data:
        save_lerobot_format(data, args.output_dir, args.task)


if __name__ == '__main__':
    main()
