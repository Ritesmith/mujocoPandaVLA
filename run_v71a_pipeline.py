#!/usr/bin/env python3
"""V71a pipeline — Smaller rollout buffer n_steps=512 (single-variable from V68).

Diagnostic context (Task 1 result):
  V68 step 1k = V59 (56%, 48/50 episodes identical) — first PPO update
  (step 2048) is what destroyed the policy (50% -> 5% at step 3k).

V71a hypothesis: With n_steps=2048, the first PPO update processes 2048
steps of experience in one shot — a large batch that can cause a big
policy shift. Reducing to n_steps=512 makes updates 4x smaller and 4x
more frequent, potentially reducing the per-update destruction magnitude.
With n_steps=512, the first update happens at step 512 (vs 2048 for V68),
so the step 1k eval will already reflect 1 PPO update's effect.

Single-variable change: ADD --n_steps 512 (all else identical to V68)
  V68: n_steps=2048 (default, hardcoded)
  V71a: --n_steps 512 (4x smaller rollouts)

Note: V71a and V71b are the split of the original V71 proposal which
changed both n_steps AND target_kl. Per single-variable isolation
principle, they are now separate experiments.

Usage:
    python run_v71a_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline_common import make_v68_base_cmd, run_pipeline

VERSION = "V71a"
SAVE_PATH = "/home/w/vla_workspace/outputs/place_policy_v71a"
DESCRIPTION = "小 rollout n_steps=512 (V68 config + --n_steps 512)"
SINGLE_VARIABLE = "添加 --n_steps 512 (n_steps: 2048 -> 512, 4x 更小更新)"
BASELINE_NOTE = "V68 config: n_steps=2048 (default). V71a: --n_steps 512 (4x smaller, 4x more frequent updates)."

# Generate V68 base command, then add --n_steps 512
cmd = make_v68_base_cmd(SAVE_PATH)
assert "--n_steps" not in cmd, "Base command should not have --n_steps (uses default 2048)"
# Add --n_steps 512 after --n_epochs
epochs_idx = cmd.index("--n_epochs")
cmd.insert(epochs_idx + 2, "--n_steps")
cmd.insert(epochs_idx + 3, "512")

if __name__ == "__main__":
    run_pipeline(VERSION, DESCRIPTION, cmd, SAVE_PATH,
                 SINGLE_VARIABLE, BASELINE_NOTE)
