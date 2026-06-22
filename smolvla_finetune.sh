#!/bin/bash
# SmolVLA Fine-tuning Script for GTX 1660 Ti 6GB
# Dataset: lerobot/svla_so101_pickplace (50 episodes, 11939 frames, SO101 pick-and-place)
# Policy: lerobot/smolvla_base (SmolVLM2-500M backbone + action expert)
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla
unset PYTHONPATH
export HF_ENDPOINT=https://hf-mirror.com

# Output directory
OUTPUT_DIR=/home/w/vla_workspace/outputs/smolvla_finetuned
DATASET_REPO=lerobot/svla_so101_pickplace

# Remove existing output dir to avoid FileExistsError (lerobot-train refuses to overwrite)
rm -rf "${OUTPUT_DIR}"

# Fine-tuning parameters optimized for 6GB VRAM (GTX 1660 Ti)
# Key memory-saving settings:
#   - batch_size=4: small batch to fit in 6GB VRAM
#   - use_amp=true: mixed precision (FP16) to reduce memory and speed up training
#   - freeze_vision_encoder=true: freeze VLM backbone, only train action expert + state proj
#   - train_expert_only=true: only train the action expert layers
#   - resize_imgs_with_padding=(256,256): smaller image resolution to save VRAM
#     (default is 512x512; 256x256 significantly reduces memory for vision encoder)
#   - num_workers=2: reduce worker count to save system RAM
#   - grad_clip_norm=10.0: gradient clipping for stability
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id="${DATASET_REPO}" \
    --batch_size=4 \
    --steps=5000 \
    --save_freq=1000 \
    --eval_freq=500 \
    --output_dir="${OUTPUT_DIR}" \
    --log_freq=100 \
    --num_workers=2 \
    --seed=1000 \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=true \
    --policy.resize_imgs_with_padding="[256,256]" \
    --policy.optimizer_lr=1e-4 \
    --policy.optimizer_grad_clip_norm=10.0 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=5000

echo "Training complete. Checkpoint saved to: ${OUTPUT_DIR}"
