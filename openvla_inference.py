#!/usr/bin/env python3
"""OpenVLA 4-bit quantized inference on GTX 1660 Ti 6GB.

Based on 地问.md analysis:
- Must use 4-bit NF4 quantization (FP16 needs 14GB)
- Must use float16 (no bfloat16 on GTX 1660 Ti)
- Must use eager attention (no flash_attention_2)
- Expected VRAM: ~5-6GB
- Expected inference speed: ~1-3 seconds/step
"""
import torch
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import json
import time
import os

MODEL_PATH = "/home/w/vla_workspace/models/openvla-7b"
DEVICE = "cuda:0"
UNNORM_KEY = "bridge_orig"

def main():
    print("=" * 60)
    print("  OpenVLA 4-bit Inference on GTX 1660 Ti 6GB")
    print("=" * 60)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("Please download first:")
        print("  huggingface-cli download openvla/openvla-7b --local-dir /home/w/vla_workspace/models/openvla-7b")
        return

    # VRAM before loading
    vram_before = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM before loading: {vram_before:.2f} GB")

    # 4-bit quantization config (REQUIRED for GTX 1660 Ti 6GB)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    print("Loading model (4-bit quantized)...")
    t0 = time.time()
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="eager",  # NOT flash_attention_2!
        torch_dtype=torch.float16,    # NOT bfloat16!
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": 0},
    )
    t_load = time.time() - t0
    print(f"Model loaded in {t_load:.1f}s")

    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after loading: {vram_after_load:.2f} GB")

    # Load test image
    img_path = "/home/w/mujoco/panda_render.png"
    if not os.path.exists(img_path):
        print(f"Test image not found: {img_path}")
        return

    image = Image.open(img_path).convert("RGB").resize((256, 256))

    # Prepare input
    prompt = "In: What action should the robot take to pick up the red block?\nOut:"
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.float16)

    # Fix: predict_action appends token 29871 to input_ids but doesn't update attention_mask.
    # We handle this manually to avoid the size mismatch error.
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    # Append special empty token if not already present (matching training-time inputs)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        )
        attention_mask = torch.cat(
            (attention_mask, torch.ones(1, 1, dtype=attention_mask.dtype, device=attention_mask.device)), dim=1
        )

    # Get action dimension for this dataset
    action_dim = vla.get_action_dim(UNNORM_KEY)
    print(f"Action dimension for '{UNNORM_KEY}': {action_dim}")

    # Run inference
    print("Running inference...")
    t0 = time.time()
    with torch.no_grad():
        generated_ids = vla.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=action_dim,
        )
    t_infer = time.time() - t0
    print(f"Inference time: {t_infer:.3f}s")

    vram_after_infer = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after inference: {vram_after_infer:.2f} GB")

    # Decode action tokens
    predicted_action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
    print(f"\nPredicted action token IDs: {predicted_action_token_ids}")
    print(f"Decoded tokens: {processor.tokenizer.decode(predicted_action_token_ids)}")

    # Convert token IDs to continuous actions (same logic as predict_action)
    vocab_size = vla.config.text_config.vocab_size - vla.config.pad_to_multiple_of
    bins = np.linspace(-1, 1, vla.config.n_action_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    discretized_actions = vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=bin_centers.shape[0] - 1)
    normalized_actions = bin_centers[discretized_actions]

    # Unnormalize actions
    action_norm_stats = vla.get_action_stats(UNNORM_KEY)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    print(f"\nNormalized actions: {normalized_actions}")
    print(f"Unnormalized actions: {actions}")

    # Save results
    results = {
        "model": "openvla-7b-4bit",
        "unnorm_key": UNNORM_KEY,
        "action_token_ids": predicted_action_token_ids.tolist(),
        "normalized_actions": normalized_actions.tolist(),
        "actions": actions.tolist(),
        "inference_time_s": t_infer,
        "load_time_s": t_load,
        "vram_after_load_gb": vram_after_load,
        "vram_after_infer_gb": vram_after_infer,
    }
    with open("/home/w/vla_workspace/openvla_inference_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /home/w/vla_workspace/openvla_inference_output.json")

if __name__ == "__main__":
    main()
