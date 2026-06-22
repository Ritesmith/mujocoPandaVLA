#!/usr/bin/env python3
"""SmolVLA Fine-tuning on GTX 1660 Ti 6GB.

Uses LeRobot's training pipeline directly via Python API
to avoid draccus CLI compatibility issues.

Training approach:
- Load pretrained SmolVLA base model
- Freeze VLM backbone (only train action expert + state_proj)
- Use mixed precision (AMP) via accelerate
- Use EpisodeAwareSampler for proper episode-based sampling
- Apply preprocessor pipeline (tokenize language, normalize, etc.)
- Use delta_timestamps for action chunking (chunk_size=50)
"""
import os
os.environ.pop('PYTHONPATH', None)

import logging
import math
import sys
import time
import csv
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

# LeRobot imports
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.datasets import LeRobotDataset, EpisodeAwareSampler
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.utils.collate import lerobot_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_PATH = "/home/w/vla_workspace/models/smolvla_base"
DATASET_REPO = "lerobot/svla_so101_pickplace"
OUTPUT_DIR = "/home/w/vla_workspace/outputs/smolvla_finetuned"

BATCH_SIZE = 4
NUM_STEPS = 5000
LR = 1e-4
WARMUP_STEPS = 500
DECAY_STEPS = 5000
DECAY_LR = 2.5e-6
GRAD_CLIP_NORM = 10.0
SAVE_EVERY = 500
LOG_EVERY = 100
SEED = 1000
NUM_WORKERS = 2
IMAGE_RESIZE = (256, 256)  # Smaller than default 512x512 to save VRAM


def create_lr_scheduler(optimizer, num_training_steps):
    """Cosine decay with linear warmup (matches SmolVLA default schedule)."""
    actual_warmup = WARMUP_STEPS
    actual_decay = DECAY_STEPS

    if num_training_steps < DECAY_STEPS:
        scale = num_training_steps / DECAY_STEPS
        actual_warmup = int(WARMUP_STEPS * scale)
        actual_decay = num_training_steps
        logger.info(
            f"Auto-scaling LR scheduler: warmup {WARMUP_STEPS}->{actual_warmup}, "
            f"decay {DECAY_STEPS}->{actual_decay}"
        )

    def lr_lambda(current_step):
        if current_step < actual_warmup:
            if current_step <= 0:
                return 1.0 / (actual_warmup + 1)
            frac = 1.0 - current_step / actual_warmup
            return (1.0 / (actual_warmup + 1) - 1.0) * frac + 1.0

        step = min(current_step, actual_decay)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay))
        alpha = DECAY_LR / LR
        return (1 - alpha) * cosine_decay + alpha

    return LambdaLR(optimizer, lr_lambda, -1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Create policy config ──────────────────────────────────────────────────
    policy_cfg = SmolVLAConfig()
    policy_cfg.pretrained_path = MODEL_PATH
    policy_cfg.device = "cuda"
    policy_cfg.resize_imgs_with_padding = IMAGE_RESIZE

    # ── Load dataset with delta_timestamps for action chunking ────────────────
    logger.info("Loading dataset metadata...")
    from lerobot.datasets import LeRobotDatasetMetadata
    ds_meta = LeRobotDatasetMetadata(DATASET_REPO)

    # Resolve delta_timestamps from policy config (action chunking)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    logger.info(f"delta_timestamps: {delta_timestamps}")

    logger.info("Loading dataset with action chunking...")
    dataset = LeRobotDataset(
        DATASET_REPO,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    logger.info(
        f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames"
    )
    logger.info(f"Camera keys: {dataset.meta.camera_keys}")

    # ── Create policy using factory (handles feature mapping) ─────────────────
    logger.info(f"Creating policy from {MODEL_PATH}...")
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.train()

    logger.info(f"Policy image_features: {list(policy.config.image_features.keys())}")
    logger.info(f"Policy input_features: {list(policy.config.input_features.keys())}")
    logger.info(f"Policy output_features: {list(policy.config.output_features.keys())}")

    # ── Create preprocessor / postprocessor ───────────────────────────────────
    logger.info("Creating preprocessor and postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=MODEL_PATH,
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
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )

    # ── Count parameters ──────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")

    # ── Create dataloader ─────────────────────────────────────────────────────
    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        shuffle=True,
        seed=SEED,
    )

    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
        persistent_workers=NUM_WORKERS > 0,
    )

    # ── Create optimizer and scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    scheduler = create_lr_scheduler(optimizer, NUM_STEPS)

    # ── Use accelerate for mixed precision ────────────────────────────────────
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )

    policy, optimizer, dataloader, scheduler = accelerator.prepare(
        policy, optimizer, dataloader, scheduler
    )

    # Simple cycle over dataloader
    def cycle(dl):
        while True:
            for batch in dl:
                yield batch

    dl_iter = cycle(dataloader)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"Starting training for {NUM_STEPS} steps (batch_size={BATCH_SIZE})...")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    running_loss = 0.0
    running_grad_norm = 0.0
    running_data_time = 0.0
    running_update_time = 0.0
    loss_history = []

    # Open CSV file for loss logging (avoids stdout buffering issues)
    csv_path = Path(OUTPUT_DIR) / "training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "loss", "grad_norm", "lr", "vram_gb", "vram_peak_gb", "data_time_s", "update_time_s"])
    csv_file.flush()

    for step in range(1, NUM_STEPS + 1):
        # ── Data loading ──────────────────────────────────────────────────
        data_start = time.perf_counter()
        batch = next(dl_iter)

        # Convert uint8 images to float32 [0,1]
        for cam_key in dataset.meta.camera_keys:
            if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

        # Apply preprocessor (tokenize, normalize, move to device)
        batch = preprocessor(batch)
        data_time = time.perf_counter() - data_start

        # ── Forward + backward ────────────────────────────────────────────
        update_start = time.perf_counter()
        policy.train()

        with accelerator.autocast():
            loss, loss_dict = policy.forward(batch)

        accelerator.backward(loss)

        # Gradient clipping
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), GRAD_CLIP_NORM)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        update_time = time.perf_counter() - update_start

        # ── Track metrics ─────────────────────────────────────────────────
        loss_val = loss.item()
        running_loss += loss_val
        running_grad_norm += grad_norm.item()
        running_data_time += data_time
        running_update_time += update_time
        loss_history.append(loss_val)

        # ── Logging ───────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            avg_grad = running_grad_norm / LOG_EVERY
            avg_data_s = running_data_time / LOG_EVERY
            avg_update_s = running_update_time / LOG_EVERY
            lr = optimizer.param_groups[0]["lr"]
            vram = torch.cuda.memory_allocated() / (1024**3)
            vram_peak = torch.cuda.max_memory_allocated() / (1024**3)

            logger.info(
                f"Step {step}/{NUM_STEPS} | "
                f"Loss: {avg_loss:.4f} | "
                f"GradNorm: {avg_grad:.3f} | "
                f"LR: {lr:.6f} | "
                f"Data: {avg_data_s:.2f}s | "
                f"Update: {avg_update_s:.2f}s | "
                f"VRAM: {vram:.2f}/{vram_peak:.2f} GB"
            )

            # Write to CSV file
            csv_writer.writerow([step, f"{avg_loss:.6f}", f"{avg_grad:.6f}", f"{lr:.8f}", f"{vram:.4f}", f"{vram_peak:.4f}", f"{avg_data_s:.4f}", f"{avg_update_s:.4f}"])
            csv_file.flush()

            running_loss = 0.0
            running_grad_norm = 0.0
            running_data_time = 0.0
            running_update_time = 0.0

        # ── Checkpoint ────────────────────────────────────────────────────
        if step % SAVE_EVERY == 0:
            unwrapped = accelerator.unwrap_model(policy)
            ckpt_dir = Path(OUTPUT_DIR) / f"checkpoint_{step}"
            logger.info(f"Saving checkpoint to {ckpt_dir}...")
            unwrapped.save_pretrained(ckpt_dir / "pretrained_model")
            # Save processor configs for inference
            preprocessor.save_pretrained(ckpt_dir / "pretrained_model")
            postprocessor.save_pretrained(ckpt_dir / "pretrained_model")
            logger.info(f"Checkpoint saved at step {step}")

    # ── Save final model ──────────────────────────────────────────────────────
    csv_file.close()
    unwrapped = accelerator.unwrap_model(policy)
    final_dir = Path(OUTPUT_DIR) / "final"
    logger.info(f"Saving final model to {final_dir}...")
    unwrapped.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    # ── Print loss summary ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Final model saved to: {final_dir}")
    logger.info(f"Loss curve (sampled):")
    for i in range(0, len(loss_history), max(1, len(loss_history) // 20)):
        logger.info(f"  Step {i+1}: {loss_history[i]:.4f}")
    logger.info(f"  Step {len(loss_history)}: {loss_history[-1]:.4f}")
    logger.info(f"Initial loss: {loss_history[0]:.4f}")
    logger.info(f"Final loss: {loss_history[-1]:.4f}")
    logger.info(f"Min loss: {min(loss_history):.4f} at step {loss_history.index(min(loss_history))+1}")
    logger.info(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
