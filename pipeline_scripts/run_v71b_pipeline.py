#!/usr/bin/env python3
"""V71b pipeline — Tighter KL constraint target_kl=0.01 (single-variable from V68).

Diagnostic context (Task 1 result):
  V68 step 1k = V59 (56%, 48/50 episodes identical) — first PPO update
  (step 2048) is what destroyed the policy (50% -> 5% at step 3k).

V71b hypothesis: V68 used target_kl=0.015, which allows a relatively large
policy shift per update (approx_kl=0.0047 was observed, well within the
0.015 limit). Tightening to target_kl=0.01 limits how far the policy can
move per update, potentially preventing the destructive first update from
moving too far from the V59 solution. Note: the observed approx_kl=0.0047
is already below 0.01, so this change may have NO effect (the constraint
isn't binding). This would itself be informative — if V71b crashes
identically to V68, KL constraint is not the lever.

Single-variable change: CHANGE --target_kl from 0.015 to 0.01
  V68: --target_kl 0.015
  V71b: --target_kl 0.01

Note: V71a and V71b are the split of the original V71 proposal which
changed both n_steps AND target_kl. Per single-variable isolation
principle, they are now separate experiments.

Usage:
    python run_v71b_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_scripts.run_pipeline_common import make_v68_base_cmd, run_pipeline

VERSION = "V71b"
SAVE_PATH = "/home/w/vla_workspace/outputs/place_policy_v71b"
DESCRIPTION = "紧 KL 约束 target_kl=0.01 (V68 config, target_kl 0.015->0.01)"
SINGLE_VARIABLE = "修改 --target_kl 0.015 -> 0.01 (KL constraint tightened)"
BASELINE_NOTE = "V68 config: target_kl=0.015. V71b: target_kl=0.01. Note: V68 approx_kl=0.0047 < 0.01, constraint may not be binding."

# Generate V68 base command, then change --target_kl value
cmd = make_v68_base_cmd(SAVE_PATH)
assert "--target_kl" in cmd, "Base command must contain --target_kl"
kl_idx = cmd.index("--target_kl")
assert cmd[kl_idx + 1] == "0.015", "Expected target_kl=0.015 in base command"
cmd[kl_idx + 1] = "0.01"

if __name__ == "__main__":
    run_pipeline(VERSION, DESCRIPTION, cmd, SAVE_PATH,
                 SINGLE_VARIABLE, BASELINE_NOTE)
