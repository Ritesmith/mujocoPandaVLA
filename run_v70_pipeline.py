#!/usr/bin/env python3
"""V70 pipeline — Freeze entire ResNet backbone (single-variable from V68).

Diagnostic context (Task 1 result):
  V68 step 1k = V59 (56%, 48/50 episodes identical) — first PPO update
  (step 2048) is what destroyed the policy (50% -> 5% at step 3k).
  V70 is FIRST PRIORITY per diagnostic conclusion.

V70 hypothesis: The first PPO update destroys pretrained ResNet-18 visual
features. By freezing the entire feature extractor (requires_grad=False for
all backbone parameters including layer4), PPO can only update the MLP head
(policy_net, value_net). If V70 survives step 3k (place_rate > 45%), the
destruction was in the feature extractor. If V70 also crashes, the
destruction is in the MLP head or the PPO update mechanism itself.

Single-variable change: ADD --freeze_backbone flag (all else identical to V68)
  V68: (no --freeze_backbone, backbone trainable)
  V70: --freeze_backbone (entire ResNet frozen, only MLP head trains)

Note: --freeze_bn is still present (redundant when --freeze_backbone is used,
since all backbone params are frozen anyway), kept for exact single-variable
isolation (only --freeze_backbone is added, nothing else removed).

Usage:
    python run_v70_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline_common import make_v68_base_cmd, run_pipeline

VERSION = "V70"
SAVE_PATH = "/home/w/vla_workspace/outputs/place_policy_v70"
DESCRIPTION = "冻结 ResNet backbone (V68 config + --freeze_backbone)"
SINGLE_VARIABLE = "添加 --freeze_backbone (backbone: trainable -> frozen)"
BASELINE_NOTE = "V68 config: backbone trainable. V70: --freeze_backbone (only MLP head trains)."

# Generate V68 base command, then add --freeze_backbone
cmd = make_v68_base_cmd(SAVE_PATH)
assert "--freeze_backbone" not in cmd, "Base command should not have --freeze_backbone"
# Insert --freeze_backbone right after --freeze_bn for clarity
bn_idx = cmd.index("--freeze_bn")
cmd.insert(bn_idx + 1, "--freeze_backbone")

if __name__ == "__main__":
    run_pipeline(VERSION, DESCRIPTION, cmd, SAVE_PATH,
                 SINGLE_VARIABLE, BASELINE_NOTE)
