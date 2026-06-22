#!/usr/bin/env python
"""SmolVLA inference script.

Loads the SmolVLA model, runs inference on a test image with a task instruction,
and outputs the action vector.
"""

import json
import sys
import time

import torch
from PIL import Image
from torchvision.transforms import ToTensor

MODEL_PATH = "/home/w/vla_workspace/models/smolvla_base"
IMAGE_PATH = "/home/w/mujoco/panda_render.png"
TASK_INSTRUCTION = "pick up the red block"
OUTPUT_PATH = "/home/w/vla_workspace/smolvla_inference_output.json"


def main():
    print("=" * 60)
    print("SmolVLA Inference Script")
    print("=" * 60)

    # --- VRAM before loading ---
    if torch.cuda.is_available():
        vram_before = torch.cuda.memory_allocated() / 1024**3
        print(f"[VRAM] Before model loading: {vram_before:.2f} GB")
    else:
        print("[VRAM] CUDA not available, using CPU")
        vram_before = 0.0

    # --- Load model ---
    print(f"\n[1/4] Loading SmolVLA model from: {MODEL_PATH}")
    t0 = time.time()
    from lerobot.policies.smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(MODEL_PATH)
    policy.eval()
    t_load = time.time() - t0
    print(f"      Model loaded in {t_load:.2f}s")

    # --- VRAM after loading ---
    if torch.cuda.is_available():
        vram_after_load = torch.cuda.memory_allocated() / 1024**3
        print(f"[VRAM] After model loading:  {vram_after_load:.2f} GB")
    else:
        vram_after_load = 0.0

    # --- Load pre/post processors ---
    print(f"\n[2/4] Loading pre/post processors")
    from lerobot.policies import make_pre_post_processors

    preprocess, postprocess = make_pre_post_processors(policy.config, MODEL_PATH)
    print("      Processors loaded")

    # --- Prepare observation ---
    print(f"\n[3/4] Preparing observation")
    print(f"      Image: {IMAGE_PATH}")
    print(f"      Task:  '{TASK_INSTRUCTION}'")

    img = Image.open(IMAGE_PATH).convert("RGB")
    to_tensor = ToTensor()
    img_tensor = to_tensor(img)  # [C, H, W] in [0, 1]

    # Get image key from config
    img_key = list(policy.config.image_features.keys())[0]
    print(f"      Image key: {img_key}")
    print(f"      Image tensor shape: {img_tensor.shape}")
    print(f"      All image keys in config: {list(policy.config.image_features.keys())}")

    # State dimension from config
    state_dim = policy.config.input_features["observation.state"].shape[0]
    print(f"      State dim: {state_dim}")

    # Build observation dict (batch format, no batch dimension - preprocessor adds it)
    obs = {
        img_key: img_tensor,  # [C, H, W]
        "observation.state": torch.zeros(state_dim),  # dummy state
        "task": TASK_INSTRUCTION,
    }

    # --- Preprocess ---
    t_pre = time.time()
    processed = preprocess(obs)
    t_pre_time = time.time() - t_pre
    print(f"      Preprocessing done in {t_pre_time:.3f}s")
    print(f"      Processed keys: {list(processed.keys())}")
    for k, v in processed.items():
        if isinstance(v, torch.Tensor):
            print(f"        {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

    # --- Run inference ---
    print(f"\n[4/4] Running inference")
    if torch.cuda.is_available():
        vram_before_infer = torch.cuda.memory_allocated() / 1024**3
        print(f"[VRAM] Before inference: {vram_before_infer:.2f} GB")

    t_infer_start = time.time()
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(processed)
    t_infer = time.time() - t_infer_start
    print(f"      Inference done in {t_infer:.3f}s")

    if torch.cuda.is_available():
        vram_after_infer = torch.cuda.memory_allocated() / 1024**3
        print(f"[VRAM] After inference:  {vram_after_infer:.2f} GB")

    # --- Postprocess ---
    # The postprocessor expects a PolicyAction (torch.Tensor) and returns a
    # PolicyAction (torch.Tensor) after unnormalization and moving to CPU
    action_final = postprocess(action_chunk)

    # --- Print results ---
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Action chunk shape: {action_chunk.shape}")
    print(f"Action final shape: {action_final.shape}")
    print(f"First action (step 0): {action_final[0, 0, :].tolist()}")
    print(f"Last action  (step -1): {action_final[0, -1, :].tolist()}")
    print(f"Action range: [{action_final.min().item():.4f}, {action_final.max().item():.4f}]")
    print(f"\nInference time: {t_infer:.3f}s")
    print(f"Model load time: {t_load:.2f}s")
    if torch.cuda.is_available():
        print(f"VRAM (model): {vram_after_load:.2f} GB")
        print(f"VRAM (peak):  {vram_after_infer:.2f} GB")

    # --- Save output ---
    output = {
        "model_path": MODEL_PATH,
        "image_path": IMAGE_PATH,
        "task": TASK_INSTRUCTION,
        "action_shape": list(action_final.shape),
        "first_action": action_final[0, 0, :].tolist(),
        "last_action": action_final[0, -1, :].tolist(),
        "action_min": action_final.min().item(),
        "action_max": action_final.max().item(),
        "inference_time_s": t_infer,
        "model_load_time_s": t_load,
        "vram_after_load_gb": vram_after_load,
        "vram_after_infer_gb": vram_after_infer if torch.cuda.is_available() else None,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
