#!/usr/bin/env python3
"""Cache V59 ResNet-18 features for D_expert.npz.

Extracts (N, 524) features = 512 image + 12 state using V59's frozen
features_extractor. Saves to data/D_expert_features.npz for use by the
diffusion policy training (avoids running ResNet-18 online during training).

Reuses load_v59_feature_extractor() and extract_features() from
backbone_probe.py, and normalize_states() from train_bc_expert.py.

Usage:
    python cache_resnet_features.py
    python cache_resnet_features.py --device cuda --batch_size 256
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import time
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).parent.resolve()

V59_MODEL_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/best_model.zip")
V59_VECNORM_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/vec_normalize.pkl")
EXPERT_DATA_PATH = str(WORKSPACE / "data/D_expert.npz")
OUTPUT_PATH = str(WORKSPACE / "data/D_expert_features.npz")


def normalize_states(states, vecnorm_path):
    """Normalize states using V59's VecNormalize statistics.

    Mirrors train_bc_expert.py:normalize_states() — applies the same
    (x - mean) / sqrt(var + eps) transform and clamps to [-clip_obs, clip_obs]
    so the features_extractor sees the same distribution it was trained on.
    """
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        print("Warning: no vec_normalize.pkl, states used as-is")
        return states
    import pickle
    with open(vecnorm_path, "rb") as f:
        vecnorm = pickle.load(f)
    obs_rms = vecnorm.obs_rms
    epsilon = vecnorm.epsilon
    clip_obs = vecnorm.clip_obs
    if isinstance(obs_rms, dict) and "state" in obs_rms:
        rms = obs_rms["state"]
        mean = torch.as_tensor(rms.mean, dtype=torch.float32)
        var = torch.as_tensor(rms.var, dtype=torch.float32)
        states = (states - mean) / torch.sqrt(var + epsilon)
        states = torch.clamp(states, -clip_obs, clip_obs)
    else:
        print(f"Warning: obs_rms format not recognized: {type(obs_rms)}")
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("Caching V59 ResNet-18 features for D_expert.npz")
    print("=" * 60)

    # Load expert demos
    print(f"\nLoading {EXPERT_DATA_PATH}...")
    data = np.load(EXPERT_DATA_PATH, allow_pickle=True)
    images = data["images"]  # (N, 84, 84, 3) uint8 HWC — keep as numpy for extract_features
    states = torch.as_tensor(data["states"], dtype=torch.float32)  # (N, 12) — tensor for normalize_states
    actions = data["actions"].astype(np.float32)  # (N, 8)
    episode_ids = data["episode_ids"]  # (N,) int64
    print(f"  {len(actions)} transitions, {len(np.unique(episode_ids))} episodes")
    print(f"  images: {images.shape}, states: {states.shape}, actions: {actions.shape}")

    # Normalize states using V59's VecNormalize stats
    print("\nNormalizing states with V59 VecNormalize stats...")
    states = normalize_states(states, V59_VECNORM_PATH)
    states_np = states.numpy().astype(np.float32)  # back to numpy for extract_features
    print(f"  Normalized states range: [{states_np.min():.3f}, {states_np.max():.3f}]")

    # Load V59's frozen features extractor
    print(f"\nLoading V59 features_extractor from {V59_MODEL_PATH}...")
    from backbone_probe import load_v59_feature_extractor, extract_features
    fe, _ = load_v59_feature_extractor(V59_MODEL_PATH, device=args.device)

    # Extract features
    print(f"\nExtracting features (batch_size={args.batch_size})...")
    t0 = time.time()
    features = extract_features(fe, images, states_np, device=args.device,
                                batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Features shape: {features.shape}, dtype: {features.dtype}")
    print(f"  Features range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Features mean: {features.mean():.3f}, std: {features.std():.3f}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    np.savez(OUTPUT_PATH,
             features=features.astype(np.float32),
             actions=actions,
             episode_ids=episode_ids.astype(np.int64),
             states=states_np)
    file_size = Path(OUTPUT_PATH).stat().st_size / (1024 * 1024)
    print(f"  Saved ({file_size:.1f} MB)")

    # Verify
    print("\nVerification:")
    verify = np.load(OUTPUT_PATH)
    for k in verify.keys():
        print(f"  {k}: shape={verify[k].shape}, dtype={verify[k].dtype}")

    print("\n" + "=" * 60)
    print("Feature caching complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
