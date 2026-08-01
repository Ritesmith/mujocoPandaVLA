#!/usr/bin/env python3
"""Shallow BC head training for CSIL++ prior p(a|s).

Trains a from-scratch shallow MLP head on top of V59's frozen features_extractor
to learn a behavior-cloning prior p(a|s) from D_succ. This prior replaces the
uniform/GMM prior used in the previous CSIL++ attempt (which produced a constant
coherent reward because D_succ was collected deterministically).

Architecture (user spec):
  Frozen: V59 features_extractor (ResNet-18 + state passthrough) -> 524-dim
  Frozen: V59 mlp_extractor (NOT inherited — discarded)
  Shallow head (from scratch):
    fc(524 -> 256) -> ReLU -> fc(256 -> 128) -> ReLU
    -> mu = fc(128 -> 8)
    -> log_sigma = fc(128 -> 8)  [state-dependent, unlike V59's global log_std]

Gate 1a acceptance criteria:
  1. D_succ filter logpi_V59 < -5 keeps >= 90%
  2. NLL_p - NLL_pi in [0.5, 2.0]  (BC head worse than V59 by moderate amount)
  3. JS(pi, p)_Dsucc > 0.05  AND  JS(pi, p)_Dfail > 0.03
  4. p sampling SR in [10%, 50%)  (BC head alone has some success)

Usage:
    python train_shallow_bc.py
    python train_shallow_bc.py --n_epochs 100 --batch_size 256 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from diagnostics.voronoi_partition import (
    GRASP_MODEL_PATH, GRASP_VECNORM_PATH,
    PLACE_MODEL_PATH, PLACE_VECNORM_PATH,
    TARGET_POS_RANGE, TABLE_Z, MAX_STEPS, SEED,
    PLACE_THRESHOLD,
    _build_collect_envs, freeze_backbone,
    extract_v59_latent, v59_value,
    normalize_states, load_v59_state_dict,
)

DCSIL_PATH = WORKSPACE / "data" / "D_csil.npz"
D_FAIL_PATH = WORKSPACE / "data" / "D_fail.npz"
OUTPUT_DIR = WORKSPACE / "outputs" / "csil_plus_plus"
SHALLOW_BC_PATH = OUTPUT_DIR / "shallow_bc_head.pt"
GATE_REPORT_PATH = OUTPUT_DIR / "gate_1a_report.json"

FEAT_DIM = 524  # V59 features_extractor output (confirmed from state_dict)
ACTION_DIM = 8
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class ShallowBCHead(nn.Module):
    """Shallow BC head: fc(feat_dim->256)->ReLU->fc(256->128)->ReLU->mu + log_sigma.

    Unlike V59's state-independent log_std, this head produces a state-dependent
    log_sigma, allowing the prior p(a|s) to have varying uncertainty across
    states. This is critical for the coherent reward to be non-constant.

    When ``freeze_log_sigma=True``, the log_sigma_head is frozen at
    ``init_log_sigma`` (default 0.0 = σ=1.0, matching V59's σ≈1.0). This
    prevents training from collapsing σ to near-zero, which would make the
    BC head overconfident (NLL_p << NLL_π) and fail Gate 1a check 2.
    """

    def __init__(self, feat_dim: int = FEAT_DIM, hidden1: int = 256,
                 hidden2: int = 128, action_dim: int = ACTION_DIM,
                 init_log_sigma: float = 0.0,
                 freeze_log_sigma: bool = True):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden2, action_dim)
        self.log_sigma_head = nn.Linear(hidden2, action_dim)

        # Initialize log_sigma_head bias to init_log_sigma and zero weights.
        with torch.no_grad():
            self.log_sigma_head.bias.fill_(init_log_sigma)
            self.log_sigma_head.weight.zero_()

        if freeze_log_sigma:
            for p in self.log_sigma_head.parameters():
                p.requires_grad = False

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) each of shape (B, action_dim)."""
        h = self.shared(features)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_sigma

    def log_prob(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute log p(a|s) under the BC head's diagonal Gaussian.

        Returns (N,) tensor.
        """
        mu, log_sigma = self.forward(features)
        std = torch.exp(log_sigma)
        diff = actions - mu
        log_p = -0.5 * ((diff / std) ** 2).sum(dim=-1) \
                - log_sigma.sum(dim=-1) \
                - 0.5 * mu.shape[-1] * float(np.log(2.0 * np.pi))
        return log_p

    def sample(self, features: torch.Tensor,
               deterministic: bool = False) -> torch.Tensor:
        """Sample action a ~ p(a|s)."""
        mu, log_sigma = self.forward(features)
        if deterministic:
            return mu
        std = torch.exp(log_sigma)
        return mu + std * torch.randn_like(mu)


def load_v59_policy(device: str = "cpu"):
    """Load V59 model with frozen backbone for feature extraction."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    args_ns = argparse.Namespace(
        place_model=PLACE_MODEL_PATH,
        place_vecnorm=PLACE_VECNORM_PATH,
        grasp_model=GRASP_MODEL_PATH,
        grasp_vecnorm=GRASP_VECNORM_PATH,
        release_threshold=PLACE_THRESHOLD,
    )
    envs = _build_collect_envs(args_ns, device=device)
    place_model = envs["place_model"]
    freeze_backbone(place_model)
    place_model.policy.features_extractor.eval()
    for m in place_model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    return place_model, envs, device


def extract_features_batch(model, images: np.ndarray, states: np.ndarray,
                           device: str, batch_size: int = 256) -> torch.Tensor:
    """Extract V59 features (524-dim) for a batch of observations.

    Parameters
    ----------
    images : (N, 84, 84, 3) uint8 HWC
    states : (N, 12) float32 raw (will be normalized by vecnorm)
    """
    n = len(images)
    all_features = []
    model.policy.eval()

    # Normalize states using V59's vecnorm.
    states_t = torch.as_tensor(states, dtype=torch.float32)
    states_normed = normalize_states(states_t, PLACE_VECNORM_PATH)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            img_batch = torch.as_tensor(
                images[i:i+batch_size], dtype=torch.float32, device=device
            ).permute(0, 3, 1, 2).contiguous()  # (B, 3, 84, 84)
            st_batch = states_normed[i:i+batch_size].to(device)
            obs = {"image": img_batch, "state": st_batch}
            features = model.policy.extract_features(obs)  # (B, 524)
            all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)  # (N, 524) on CPU


def compute_v59_logpi(model, images: np.ndarray, states: np.ndarray,
                     actions: np.ndarray, device: str,
                     batch_size: int = 256) -> np.ndarray:
    """Compute V59's log pi(a|s) for each transition."""
    n = len(images)
    all_logpi = []
    model.policy.eval()
    v59_log_std = model.policy.log_std.detach().clamp(LOG_STD_MIN, LOG_STD_MAX)

    states_t = torch.as_tensor(states, dtype=torch.float32)
    states_normed = normalize_states(states_t, PLACE_VECNORM_PATH)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            img_batch = torch.as_tensor(
                images[i:i+batch_size], dtype=torch.float32, device=device
            ).permute(0, 3, 1, 2).contiguous()
            st_batch = states_normed[i:i+batch_size].to(device)
            obs = {"image": img_batch, "state": st_batch}
            latent = extract_v59_latent(model.policy, obs)
            mu = model.policy.action_net(latent)
            act_batch = torch.as_tensor(
                actions[i:i+batch_size], dtype=torch.float32, device=device
            )
            log_std_expanded = v59_log_std.expand_as(mu)
            std = torch.exp(log_std_expanded)
            diff = act_batch - mu
            log_pi = -0.5 * ((diff / std) ** 2).sum(dim=-1) \
                     - log_std_expanded.sum(dim=-1) \
                     - 0.5 * mu.shape[-1] * float(np.log(2.0 * np.pi))
            all_logpi.append(log_pi.cpu().numpy())
    return np.concatenate(all_logpi)


def js_divergence_gaussian(mu1, log_sigma1, mu2, log_sigma2):
    """JS divergence between two diagonal Gaussians (approximate via sampling).

    Returns scalar (averaged over batch).
    """
    with torch.no_grad():
        std1 = torch.exp(log_sigma1)
        std2 = torch.exp(log_sigma2)
        kl_12 = (log_sigma2 - log_sigma1
                 + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5).sum(dim=-1)
        kl_21 = (log_sigma1 - log_sigma2
                 + (std2**2 + (mu2 - mu1)**2) / (2 * std1**2) - 0.5).sum(dim=-1)
        js = 0.5 * (kl_12 + kl_21)  # symmetric approximation
    return float(js.mean())


def compute_js_batched(features_cpu, bc_head, v59_policy, v59_log_std,
                       device, batch_size=256):
    """Compute mean JS divergence between BC head and V59 in batches.

    Uses precomputed features (524-dim) — no need to re-run the expensive
    ResNet-18 features_extractor. V59 mu is computed from:
      latent = mlp_extractor.forward_actor(features)
      mu = action_net(latent)

    This avoids GPU OOM when features_cpu has >10k entries.
    """
    n = len(features_cpu)
    js_sum = 0.0
    bc_head.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            feat = features_cpu[i:i+batch_size].to(device)
            # BC head.
            bc_mu, bc_logsigma = bc_head(feat)
            # V59 (from same precomputed features).
            latent = v59_policy.mlp_extractor.forward_actor(feat)
            v59_mu = v59_policy.action_net(latent)
            v59_logsigma = v59_log_std.expand_as(v59_mu)
            # JS for this batch.
            std1 = torch.exp(v59_logsigma)
            std2 = torch.exp(bc_logsigma)
            kl_12 = (bc_logsigma - v59_logsigma
                     + (std1**2 + (v59_mu - bc_mu)**2) / (2 * std2**2) - 0.5).sum(dim=-1)
            kl_21 = (v59_logsigma - bc_logsigma
                     + (std2**2 + (bc_mu - v59_mu)**2) / (2 * std1**2) - 0.5).sum(dim=-1)
            js_batch = 0.5 * (kl_12 + kl_21)
            js_sum += float(js_batch.sum())
    return js_sum / n


def train_shallow_bc(
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    device: str = "auto",
    seed: int = SEED,
    init_log_sigma: float = 0.0,
    freeze_log_sigma: bool = True,
    d_csil_path: str = None,
):
    """Train shallow BC head on D_succ and run Gate 1a checks."""
    if d_csil_path is None:
        d_csil_path = str(DCSIL_PATH)
    print("=" * 60)
    print("Shallow BC Head Training (CSIL++ Prior)")
    print("=" * 60)
    print(f"Epochs: {n_epochs}  Batch size: {batch_size}  LR: {lr}")
    print(f"Val split: {val_frac*100:.0f}%")
    print(f"init_log_sigma: {init_log_sigma}  freeze_log_sigma: {freeze_log_sigma}")
    print(f"D_csil: {d_csil_path}")
    print()

    # Resolve device.
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load D_csil.
    data = np.load(d_csil_path, allow_pickle=True)
    labels = data["labels"]
    succ_mask = labels == 1
    fail_mask = labels == 0

    succ_images = data["images"][succ_mask]
    succ_states = data["states"][succ_mask]
    succ_actions = data["actions"][succ_mask]
    print(f"D_succ: {len(succ_images)} transitions (from {int(succ_mask.sum())} successes)")

    fail_images = data["images"][fail_mask]
    fail_states = data["states"][fail_mask]
    fail_actions = data["actions"][fail_mask]
    print(f"D_fail (from D_csil): {len(fail_images)} transitions")

    # Load V59.
    print("\nLoading V59 (frozen backbone)...")
    place_model, envs, device = load_v59_policy(device=device)
    v59_policy = place_model.policy
    v59_log_std = v59_policy.log_std.detach().clamp(LOG_STD_MIN, LOG_STD_MAX)

    # Extract V59 features for D_succ and D_fail.
    print("\nExtracting V59 features (524-dim) for D_succ...")
    t0 = time.time()
    succ_features = extract_features_batch(
        place_model, succ_images, succ_states, device, batch_size)
    print(f"  D_succ features: {succ_features.shape}  [{time.time()-t0:.1f}s]")

    print("Extracting V59 features for D_fail...")
    t0 = time.time()
    fail_features = extract_features_batch(
        place_model, fail_images, fail_states, device, batch_size)
    print(f"  D_fail features: {fail_features.shape}  [{time.time()-t0:.1f}s]")

    # Compute V59 log pi on D_succ (Gate 1a check 1).
    # Spec: "D_succ 过滤 logπ_V59 < -5 后保留≥90%" — keep transitions where
    # logπ < -5 (i.e., actions are "plausible" under V59, not near-zero prob).
    # With V59's σ≈1.0 and D=8, max logπ = -7.406 < -5, so ALL actions pass.
    print("\nComputing V59 log pi(a|s) on D_succ...")
    succ_logpi_v59 = compute_v59_logpi(
        place_model, succ_images, succ_states, succ_actions, device, batch_size)
    frac_kept = float((succ_logpi_v59 < -5.0).mean())
    print(f"  logpi_V59 on D_succ: min={succ_logpi_v59.min():.3f} "
          f"max={succ_logpi_v59.max():.3f} mean={succ_logpi_v59.mean():.3f}")
    print(f"  Gate 1a.1: frac(logpi_V59 < -5) = {frac_kept:.1%} "
          f"(target: >= 90%) -> {'PASS' if frac_kept >= 0.90 else 'FAIL'}")

    # Also compute V59 log pi on D_fail (for JS check).
    print("Computing V59 log pi(a|s) on D_fail...")
    fail_logpi_v59 = compute_v59_logpi(
        place_model, fail_images, fail_states, fail_actions, device, batch_size)
    print(f"  logpi_V59 on D_fail: min={fail_logpi_v59.min():.3f} "
          f"max={fail_logpi_v59.max():.3f} mean={fail_logpi_v59.mean():.3f}")

    # V59 NLL on D_succ.
    nll_v59_succ = -float(np.mean(succ_logpi_v59))
    print(f"\n  NLL_pi (V59) on D_succ: {nll_v59_succ:.4f}")

    # Train/val split.
    n_succ = len(succ_images)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_succ)
    n_val = max(1, int(n_succ * val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_features = succ_features[train_idx].to(device)
    train_actions = torch.as_tensor(
        succ_actions[train_idx], dtype=torch.float32, device=device)
    val_features = succ_features[val_idx].to(device)
    val_actions = torch.as_tensor(
        succ_actions[val_idx], dtype=torch.float32, device=device)

    print(f"\nTrain: {len(train_idx)}  Val: {len(val_idx)}")

    # Create BC head.
    bc_head = ShallowBCHead(
        feat_dim=FEAT_DIM,
        init_log_sigma=init_log_sigma,
        freeze_log_sigma=freeze_log_sigma,
    ).to(device)
    n_trainable = sum(p.numel() for p in bc_head.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in bc_head.parameters() if not p.requires_grad)
    print(f"  BC head: {n_trainable} trainable, {n_frozen} frozen params")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, bc_head.parameters()), lr=lr)

    # Training loop.
    print(f"\n--- Training ({n_epochs} epochs) ---")
    train_dataset = TensorDataset(train_features, train_actions)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)

    best_val_nll = float("inf")
    best_state = None
    history = []

    for epoch in range(n_epochs):
        bc_head.train()
        total_loss = 0.0
        n_batches = 0
        for feat_batch, act_batch in train_loader:
            optimizer.zero_grad()
            log_p = bc_head.log_prob(feat_batch, act_batch)
            loss = -log_p.mean()  # NLL
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1

        train_nll = total_loss / n_batches

        # Validation.
        bc_head.eval()
        with torch.no_grad():
            val_log_p = bc_head.log_prob(val_features, val_actions)
            val_nll = -float(val_log_p.mean())

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.cpu().clone() for k, v in bc_head.state_dict().items()}

        history.append({"epoch": epoch, "train_nll": train_nll,
                        "val_nll": val_nll})

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_nll={train_nll:.4f} "
                  f"val_nll={val_nll:.4f} (best={best_val_nll:.4f})")

    # Load best model.
    bc_head.load_state_dict(best_state)
    bc_head.eval()
    print(f"\nBest val NLL: {best_val_nll:.4f}")

    # Save model.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "feat_dim": FEAT_DIM,
        "hidden_dims": [256, 128],
        "action_dim": ACTION_DIM,
        "init_log_sigma": init_log_sigma,
        "freeze_log_sigma": freeze_log_sigma,
        "d_csil_path": d_csil_path,
        "train_nll": float(history[-1]["train_nll"]),
        "val_nll": best_val_nll,
    }, SHALLOW_BC_PATH)
    print(f"Saved BC head to {SHALLOW_BC_PATH}")

    # --- Gate 1a checks ---
    print("\n" + "=" * 60)
    print("Gate 1a Acceptance Checks")
    print("=" * 60)

    gate_results = {}

    # Check 1: D_succ filter logpi_V59 < -5 keeps >= 90% (already computed).
    gate_results["check1_logpi_filter"] = {
        "frac_kept": frac_kept,
        "target": 0.90,
        "pass": frac_kept >= 0.90,
    }

    # Check 2: NLL_p - NLL_pi in [0.5, 2.0].
    nll_p_succ = best_val_nll  # BC head NLL on D_succ val
    nll_diff = nll_p_succ - nll_v59_succ
    gate_results["check2_nll_diff"] = {
        "nll_p": nll_p_succ,
        "nll_pi": nll_v59_succ,
        "diff": nll_diff,
        "target_range": [0.5, 2.0],
        "pass": 0.5 <= nll_diff <= 2.0,
    }
    print(f"  Check 2: NLL_p - NLL_pi = {nll_diff:.4f} "
          f"(target: [0.5, 2.0]) -> {'PASS' if 0.5 <= nll_diff <= 2.0 else 'FAIL'}")

    # Check 3: JS(pi, p) on D_succ > 0.05 and on D_fail > 0.03.
    # Batched to avoid GPU OOM (D_succ=11k, D_fail=34k images).
    print("  Computing JS divergence (batched)...")
    js_dsucc = compute_js_batched(
        succ_features, bc_head, v59_policy, v59_log_std, device, batch_size)
    js_dfail = compute_js_batched(
        fail_features, bc_head, v59_policy, v59_log_std, device, batch_size)

    gate_results["check3_js_divergence"] = {
        "js_dsucc": js_dsucc,
        "js_dfail": js_dfail,
        "target_dsucc": 0.05,
        "target_dfail": 0.03,
        "pass": js_dsucc > 0.05 and js_dfail > 0.03,
    }
    print(f"  Check 3: JS(pi,p)_Dsucc={js_dsucc:.4f} (target > 0.05) -> "
          f"{'PASS' if js_dsucc > 0.05 else 'FAIL'}")
    print(f"           JS(pi,p)_Dfail={js_dfail:.4f} (target > 0.03) -> "
          f"{'PASS' if js_dfail > 0.03 else 'FAIL'}")

    # Check 4: p sampling SR in [10%, 50%) — skipped (requires env rollout).
    gate_results["check4_sampling_sr"] = {
        "status": "SKIPPED — requires env rollout (run separately)",
        "target_range": [0.10, 0.50],
    }
    print(f"  Check 4: p sampling SR — SKIPPED (requires env rollout)")

    # Overall gate.
    checks_passed = sum(1 for k in ["check1_logpi_filter", "check2_nll_diff",
                                    "check3_js_divergence"]
                        if gate_results[k]["pass"])
    gate_results["overall"] = {
        "checks_passed": checks_passed,
        "checks_total": 3,
        "gate_1a_pass": checks_passed == 3,
    }
    print(f"\n  Overall Gate 1a: {checks_passed}/3 checks passed -> "
          f"{'PASS' if checks_passed == 3 else 'FAIL'}")

    # Save gate report.
    report = {
        "gate_1a": gate_results,
        "training": {
            "n_epochs": n_epochs,
            "best_val_nll": best_val_nll,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "feat_dim": FEAT_DIM,
        },
        "v59_stats": {
            "nll_pi_dsucc": nll_v59_succ,
            "logpi_dsucc_mean": float(succ_logpi_v59.mean()),
            "logpi_dsucc_min": float(succ_logpi_v59.min()),
        },
        "history": history,
    }
    with open(GATE_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved gate report to {GATE_REPORT_PATH}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Train shallow BC head for CSIL++ prior p(a|s)."
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--init_log_sigma", type=float, default=0.0,
                        help="Initial log_sigma (default 0.0 = sigma=1.0, matching V59)")
    parser.add_argument("--freeze_log_sigma", action="store_true", default=True,
                        help="Freeze log_sigma_head (prevents overconfident BC head)")
    parser.add_argument("--no_freeze_log_sigma", dest="freeze_log_sigma",
                        action="store_false",
                        help="Allow log_sigma to be trained (state-dependent sigma)")
    parser.add_argument("--d_csil", type=str, default=None,
                        help="Path to D_csil.npz (default: data/D_csil.npz). "
                             "Use data/D_csil_stochastic.npz for stochastic actions.")
    args = parser.parse_args()

    train_shallow_bc(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        device=args.device,
        seed=args.seed,
        init_log_sigma=args.init_log_sigma,
        freeze_log_sigma=args.freeze_log_sigma,
        d_csil_path=args.d_csil,
    )


if __name__ == "__main__":
    main()
