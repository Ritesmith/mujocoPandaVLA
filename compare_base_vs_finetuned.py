#!/usr/bin/env python3
"""Compare base vs finetuned SmolVLA model inference outputs."""
import os
os.environ.pop('PYTHONPATH', None)

import sys
import json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.policies import make_pre_post_processors

MODEL_BASE = "/home/w/vla_workspace/models/smolvla_base"
MODEL_FINETUNED = "/home/w/vla_workspace/outputs/smolvla_finetuned/checkpoint_5000/pretrained_model"
IMAGE_PATH = "/home/w/mujoco/panda_render.png"
TASK_INSTRUCTION = "pick up the red block"
OUTPUT_JSON = "/home/w/vla_workspace/finetune_comparison.json"


def load_model_and_processors(model_path):
    """Load SmolVLA policy and its pre/post processors."""
    print(f"Loading model from {model_path}...")
    policy = SmolVLAPolicy.from_pretrained(model_path)
    policy.eval()

    preprocess, postprocess = make_pre_post_processors(policy.config, model_path)
    print(f"  Model and processors loaded.")
    return policy, preprocess, postprocess


def run_inference(policy, preprocess, postprocess, image_path, task):
    """Run inference and return the final (postprocessed) action tensor."""
    img = Image.open(image_path).convert("RGB")
    to_tensor = ToTensor()
    img_tensor = to_tensor(img)  # [C, H, W] in [0, 1]

    # Get image key from config
    img_key = list(policy.config.image_features.keys())[0]
    state_dim = policy.config.input_features["observation.state"].shape[0]

    # Build observation dict (batch format, no batch dimension - preprocessor adds it)
    obs = {
        img_key: img_tensor,
        "observation.state": torch.zeros(state_dim),
        "task": task,
    }

    # Preprocess
    processed = preprocess(obs)

    # Inference
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(processed)

    # Postprocess (unnormalize)
    action_final = postprocess(action_chunk)

    return action_final


def main():
    print("=" * 60)
    print("SmolVLA Base vs Finetuned Comparison")
    print("=" * 60)

    # Load base model
    base_policy, base_pre, base_post = load_model_and_processors(MODEL_BASE)

    # Run base inference
    print(f"\nRunning base model inference...")
    base_action = run_inference(base_policy, base_pre, base_post, IMAGE_PATH, TASK_INSTRUCTION)
    print(f"  Base action shape: {base_action.shape}")
    print(f"  Base action[0,0,:]: {base_action[0, 0, :].tolist()}")

    # Free base model to save VRAM
    del base_policy, base_pre, base_post
    torch.cuda.empty_cache()

    # Load finetuned model
    ft_policy, ft_pre, ft_post = load_model_and_processors(MODEL_FINETUNED)

    # Run finetuned inference
    print(f"\nRunning finetuned model inference...")
    ft_action = run_inference(ft_policy, ft_pre, ft_post, IMAGE_PATH, TASK_INSTRUCTION)
    print(f"  Finetuned action shape: {ft_action.shape}")
    print(f"  Finetuned action[0,0,:]: {ft_action[0, 0, :].tolist()}")

    # Compare actions - use the first action step
    base_np = base_action[0, 0, :].cpu().numpy().flatten()
    ft_np = ft_action[0, 0, :].cpu().numpy().flatten()

    # Also compare full action chunk
    base_full = base_action[0, :, :].cpu().numpy().flatten()
    ft_full = ft_action[0, :, :].cpu().numpy().flatten()

    # Metrics for first action step
    l2 = np.linalg.norm(base_np - ft_np)
    cos_sim = np.dot(base_np, ft_np) / (np.linalg.norm(base_np) * np.linalg.norm(ft_np) + 1e-8)
    per_dim_diff = np.abs(base_np - ft_np)
    max_dim_diff = np.max(per_dim_diff)
    mean_dim_diff = np.mean(per_dim_diff)

    # Metrics for full action chunk
    l2_full = np.linalg.norm(base_full - ft_full)
    cos_sim_full = np.dot(base_full, ft_full) / (np.linalg.norm(base_full) * np.linalg.norm(ft_full) + 1e-8)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Base model:      {MODEL_BASE}")
    print(f"Finetuned model: {MODEL_FINETUNED}")
    print(f"Image:           {IMAGE_PATH}")
    print(f"Task:            {TASK_INSTRUCTION}")
    print()
    print("--- First Action Step ---")
    print(f"Base action:      {base_np}")
    print(f"Finetuned action: {ft_np}")
    print(f"L2 distance:       {l2:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Per-dim diff:      {per_dim_diff}")
    print(f"Max dim diff:      {max_dim_diff:.6f}")
    print(f"Mean dim diff:     {mean_dim_diff:.6f}")
    print()
    print("--- Full Action Chunk ---")
    print(f"Chunk shape:       {base_action.shape}")
    print(f"L2 distance:       {l2_full:.6f}")
    print(f"Cosine similarity: {cos_sim_full:.6f}")
    print("=" * 60)

    # Save results
    results = {
        "base_model": MODEL_BASE,
        "finetuned_model": MODEL_FINETUNED,
        "image_path": IMAGE_PATH,
        "task": TASK_INSTRUCTION,
        "first_action_step": {
            "base_action": base_np.tolist(),
            "finetuned_action": ft_np.tolist(),
            "l2_distance": float(l2),
            "cosine_similarity": float(cos_sim),
            "per_dim_diff": per_dim_diff.tolist(),
            "max_dim_diff": float(max_dim_diff),
            "mean_dim_diff": float(mean_dim_diff),
        },
        "full_action_chunk": {
            "l2_distance": float(l2_full),
            "cosine_similarity": float(cos_sim_full),
            "action_shape": list(base_action.shape),
        },
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
