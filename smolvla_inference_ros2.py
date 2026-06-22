#!/usr/bin/env python3
"""Standalone SmolVLA inference script for ROS2 bridge.

Called by vla_bridge_node via subprocess to avoid conda/ROS2 Python conflicts.

Usage: python3 smolvla_inference_ros2.py --image path.png --task "pick up block" --model path --output out.json
"""
import argparse
import json
import time
import sys
import os

# Unset conflicting PYTHONPATH from ROS2 environment
os.environ.pop('PYTHONPATH', None)

# Must use conda vla environment's Python
sys.path.insert(0, '/home/w/vla_workspace/lerobot/src')

import torch
from PIL import Image
import torchvision.transforms as T

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--task', required=True, help='Task instruction')
    parser.add_argument('--model', required=True, help='Model path')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    # Load model
    policy = SmolVLAPolicy.from_pretrained(args.model)
    policy.eval()
    policy.to("cuda")

    # Get preprocessors
    preprocess, postprocess = make_pre_post_processors(policy.config, args.model)

    # Load and preprocess image
    img = Image.open(args.image).convert("RGB")
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(img)

    # Get the image key from config
    img_key = list(policy.config.image_features.keys())[0]

    # Build observation
    state_key = None
    for k, v in policy.config.input_features.items():
        if 'state' in k:
            state_key = k
            break

    obs = {
        img_key: img_tensor,
        "task": args.task + "\n",
    }
    if state_key:
        obs[state_key] = torch.zeros(policy.config.max_state_dim)

    # Preprocess
    processed = preprocess(obs)

    # Run inference
    t0 = time.time()
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(processed)
    t_infer = time.time() - t0

    # Postprocess
    action_chunk = postprocess(action_chunk)

    # Convert to list
    actions = action_chunk.squeeze(0).cpu().tolist()  # [chunk_size, action_dim]

    # Save results
    results = {
        "actions": actions,
        "inference_time_s": t_infer,
        "action_shape": list(action_chunk.shape),
        "task": args.task,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Inference done: {len(actions)} actions, {t_infer:.3f}s")


if __name__ == '__main__':
    main()
