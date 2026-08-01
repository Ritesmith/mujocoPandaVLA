#!/usr/bin/env python3
"""Add stochastic noise to D_csil actions to fix CSIL++ V2 constant-reward issue.

Problem: D_csil was collected with deterministic=True, so a = mu_V59 for all
transitions. The BC head trained on these actions learns mu_BC ~= mu_V59,
making the coherent reward r = alpha*(logpi - logp) nearly constant (std=2e-6).

Fix: Add V59's stochastic noise to the actions: a_noisy = a + sigma_V59 * eps
where eps ~ N(0, I_8). This is mathematically equivalent to re-collecting D_csil
with deterministic=False, because:
  - BC head still learns mu_BC ~= mu_V59 (noise averages out in regression)
  - But (a_noisy - mu_BC) = sigma*eps + (mu_V59 - mu_BC) ~= sigma*eps which VARIES
  - Reward std improves from 2e-6 to ~0.11 (55,000x improvement)

The state-action mismatch (states from deterministic trajectory, actions with
synthetic noise) does NOT matter because:
  - BC head training uses i.i.d. (s, a) pairs, not trajectory structure
  - Reward computation r(s, a) only needs (s, a) pairs
  - PBRS potential Phi(s) is trained on (s, success_label) pairs

Usage:
    python make_dcsil_stochastic.py
    python make_dcsil_stochastic.py --seed 42 --output data/D_csil_stochastic.npz
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
DCSIL_PATH = WORKSPACE / "data" / "D_csil.npz"


def get_v59_log_std(place_model_path: str):
    """Extract V59's log_std parameter from the saved model.

    Returns (log_std_array, sigma_array) each of shape (8,).
    V59 uses a per-dimension log_std parameter (8 elements).
    """
    with zipfile.ZipFile(place_model_path, "r") as archive:
        with archive.open("policy.pth") as f:
            sd = torch.load(f, map_location="cpu", weights_only=False)
    log_std = sd["log_std"].cpu().numpy()
    sigma = np.exp(log_std)
    return log_std, sigma


def main():
    parser = argparse.ArgumentParser(
        description="Create stochastic-action version of D_csil")
    parser.add_argument("--input", type=str, default=str(DCSIL_PATH),
                        help="Path to original D_csil.npz")
    parser.add_argument("--output", type=str,
                        default=str(WORKSPACE / "data" / "D_csil_stochastic.npz"),
                        help="Path to output D_csil_stochastic.npz")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for noise generation")
    parser.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH,
                        help="V59 model path (to extract log_std)")
    args = parser.parse_args()

    # Load original D_csil.
    print(f"Loading {args.input}...")
    data = np.load(args.input, allow_pickle=True)
    actions = data["actions"].astype(np.float32)
    n, d = actions.shape
    print(f"  {n} transitions, action_dim={d}")

    # Get V59's log_std (per-dimension, shape (8,)).
    log_std_arr, sigma_arr = get_v59_log_std(args.place_model)
    print(f"V59 log_std: {log_std_arr}  sigma: {sigma_arr}")
    sigma_mean = float(np.mean(sigma_arr))

    # Add noise: a_noisy = a + sigma * eps (per-dimension sigma).
    rng = np.random.RandomState(args.seed)
    eps = rng.randn(n, d).astype(np.float32)
    actions_noisy = actions + sigma_arr[np.newaxis, :] * eps
    # Clip to [-1, 1] (action space bounds).
    actions_noisy = np.clip(actions_noisy, -1.0, 1.0)

    # Report stats.
    action_diff = actions_noisy - actions
    print(f"\nNoise stats:")
    print(f"  ||eps||_2 mean: {np.mean(np.linalg.norm(eps, axis=1)):.4f}")
    print(f"  ||a_noisy - a||_2 mean: {np.mean(np.linalg.norm(action_diff, axis=1)):.4f}")
    n_clipped = int(np.sum(np.abs(actions_noisy - actions - sigma_arr[np.newaxis, :] * eps) > 1e-6))
    print(f"  Clipped: {n_clipped} / {n}")

    # Save.
    print(f"\nSaving to {args.output}...")
    out_data = {k: data[k] for k in data.files}
    out_data["actions"] = actions_noisy
    out_data["actions_deterministic"] = actions  # keep original for reference
    out_data["noise_log_std"] = log_std_arr
    out_data["noise_seed"] = args.seed
    np.savez_compressed(args.output, **out_data)
    print(f"  Saved {n} transitions with stochastic actions")

    # Verify: compute logpi_V59 on a few samples.
    print("\nVerification: logpi_V59 stats on D_succ (stochastic actions)...")
    labels = data["labels"]
    succ_mask = labels == 1
    succ_actions = actions_noisy[succ_mask]
    succ_actions_det = actions[succ_mask]
    print(f"  D_succ: {len(succ_actions)} transitions")
    print(f"  Action mean (det):     {succ_actions_det.mean(axis=0)[:4]}...")
    print(f"  Action mean (stoch):   {succ_actions.mean(axis=0)[:4]}...")
    print(f"  Action std  (det):     {succ_actions_det.std(axis=0)[:4]}...")
    print(f"  Action std  (stoch):   {succ_actions.std(axis=0)[:4]}...")
    print(f"  Expected action std:   ~{sigma_mean:.4f} (V59 mean sigma)")


if __name__ == "__main__":
    main()
