#!/usr/bin/env python3
"""V69 pipeline — Disable image augmentation (single-variable from V68).

Diagnostic context (Task 1 result):
  V68 step 1k = V59 (56%, 48/50 episodes identical) — first PPO update
  (step 2048) is what destroyed the policy (50% -> 5% at step 3k).

V69 hypothesis: Image augmentation shifts apparent object positions during
BC loss computation, which may cause the PPO update to move features in
the wrong direction. Place phase needs mm-level spatial precision; even
small augmentation-induced shifts could degrade the pretrained ResNet-18's
spatial representations. Disabling augmentation tests whether this is a
contributing factor to the first-update destruction.

Single-variable change: REMOVE --image_augment flag (all else identical to V68)
  V68: --image_augment (enabled)
  V69: (flag absent, augmentation disabled)

Usage:
    python run_v69_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline_common import make_v68_base_cmd, run_pipeline

VERSION = "V69"
SAVE_PATH = "/home/w/vla_workspace/outputs/place_policy_v69"
DESCRIPTION = "关闭图像增强 (V68 config - --image_augment)"
SINGLE_VARIABLE = "移除 --image_augment (augmentation: enabled -> disabled)"
BASELINE_NOTE = "V68 config: --image_augment enabled. V69: flag absent (disabled)."

# Generate V68 base command, then remove --image_augment
cmd = make_v68_base_cmd(SAVE_PATH)
# --image_augment is at index 5 (after --pretrained_cnn at index 4)
assert "--image_augment" in cmd, "Base command must contain --image_augment"
cmd.remove("--image_augment")

if __name__ == "__main__":
    run_pipeline(VERSION, DESCRIPTION, cmd, SAVE_PATH,
                 SINGLE_VARIABLE, BASELINE_NOTE)
