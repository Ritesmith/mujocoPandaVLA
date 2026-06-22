#!/usr/bin/env python3
"""Evaluate SmolVLA checkpoints to get loss curve data."""
import os
os.environ.pop('PYTHONPATH', None)

import sys
import torch
from pathlib import Path

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.utils.collate import lerobot_collate_fn

MODEL_PATH = "/home/w/vla_workspace/models/smolvla_base"
DATASET_REPO = "lerobot/svla_so101_pickplace"
OUTPUT_DIR = "/home/w/vla_workspace/outputs/smolvla_finetuned"
IMAGE_RESIZE = (256, 256)
SEED = 1000

def main():
    # Create policy config
    policy_cfg = SmolVLAConfig()
    policy_cfg.device = "cuda"
    policy_cfg.resize_imgs_with_padding = IMAGE_RESIZE

    # Load dataset
    print("Loading dataset metadata...", flush=True)
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)

    print("Loading dataset...", flush=True)
    dataset = LeRobotDataset(
        DATASET_REPO,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )

    # Create a fixed eval batch (first 4 samples)
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    eval_indices = list(range(4))
    eval_samples = [dataset[i] for i in eval_indices]
    eval_batch = collate_fn(eval_samples) if collate_fn else torch.utils.data.default_collate(eval_samples)

    # Convert uint8 images to float32
    for cam_key in dataset.meta.camera_keys:
        if cam_key in eval_batch and eval_batch[cam_key].dtype == torch.uint8:
            eval_batch[cam_key] = eval_batch[cam_key].to(dtype=torch.float32) / 255.0

    # Evaluate each checkpoint
    checkpoint_dirs = sorted(
        [d for d in Path(OUTPUT_DIR).iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda x: int(x.name.split("_")[1])
    )
    # Also evaluate the base model
    all_checkpoints = [("base", MODEL_PATH)] + [(d.name, str(d / "pretrained_model")) for d in checkpoint_dirs]

    results = []
    print(f"\nEvaluating {len(all_checkpoints)} checkpoints...", flush=True)
    print("-" * 70, flush=True)

    for name, ckpt_path in all_checkpoints:
        print(f"Loading {name} from {ckpt_path}...", flush=True)

        policy_cfg_eval = SmolVLAConfig()
        policy_cfg_eval.pretrained_path = ckpt_path
        policy_cfg_eval.device = "cuda"
        policy_cfg_eval.resize_imgs_with_padding = IMAGE_RESIZE

        policy = make_policy(cfg=policy_cfg_eval, ds_meta=ds_meta)
        policy.eval()

        # Create preprocessor for this checkpoint
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=ckpt_path,
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "normalizer_processor": {
                    "stats": dataset.meta.stats,
                    "features": {**policy.config.input_features, **policy.config.output_features},
                    "norm_map": policy.config.normalization_mapping,
                },
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={},
        )

        # Preprocess batch
        batch = preprocessor(eval_batch)

        # Forward pass
        with torch.no_grad(), torch.cuda.amp.autocast():
            loss, loss_dict = policy.forward(batch)

        vram = torch.cuda.memory_allocated() / (1024**3)
        step_num = int(name.split("_")[1]) if name != "base" else 0
        results.append((step_num, name, loss.item(), vram))

        print(f"  {name}: Loss={loss.item():.4f}, VRAM={vram:.2f}GB", flush=True)

        # Free memory
        del policy, preprocessor
        torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 70, flush=True)
    print("LOSS CURVE SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Step':>6s} | {'Checkpoint':>15s} | {'Loss':>8s} | {'VRAM (GB)':>10s}", flush=True)
    print("-" * 70, flush=True)
    for step_num, name, loss_val, vram in results:
        print(f"{step_num:>6d} | {name:>15s} | {loss_val:>8.4f} | {vram:>10.2f}", flush=True)

    # Loss curve data for plotting
    print("\nLoss curve data (step, loss):", flush=True)
    for step_num, name, loss_val, vram in results:
        print(f"  {step_num}, {loss_val:.4f}", flush=True)

    if len(results) >= 2:
        initial_loss = results[0][2]
        final_loss = results[-1][2]
        min_loss = min(r[2] for r in results)
        min_step = [r for r in results if r[2] == min_loss][0][0]
        print(f"\nInitial loss (base): {initial_loss:.4f}", flush=True)
        print(f"Final loss: {final_loss:.4f}", flush=True)
        print(f"Min loss: {min_loss:.4f} at step {min_step}", flush=True)
        print(f"Loss reduction: {initial_loss - final_loss:.4f} ({(initial_loss - final_loss)/initial_loss*100:.1f}%)", flush=True)

    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
