#!/usr/bin/env python3
"""CSIL++: Coherent-reward Self-Imitation + Ensemble strategy for V59.

Task 3 of the v59-breakthrough-csil-voronoi spec.

V59 is a PPO-trained vision-based pick-and-place policy with 56% place rate.
ALL previous gradient-based fine-tuning methods (PPO, BC, DAgger, distillation,
ensemble) failed because they all use V59 as the sole information source.
CSIL++ introduces a NEW information signal: a coherent reward function

    r(s, a) = alpha * (log pi_V59(a|s) - log p(a|s))

that identifies states where V59 behaves inconsistently. A potential function
Phi(s) is trained from success/failure labels and used as a PBRS potential for
fine-tuning a learned policy a_pi. An EnsemblePolicy then averages V59 and
a_pi with a safety_rollback() that reverts to V59 if eval place_rate < 30%.

Subtasks implemented here:
  3.1  compute_coherent_reward(policy, states, actions, alpha, prior)
  3.2  detect_inconsistency(policy, states, n_samples)
  3.3  PotentialFunction + train_potential_function
  3.4  verify_coherent_reward(policy, states, actions)

CLI subcommands:
  collect         Collect V59 trajectories (success + failure) -> data/D_csil.npz
  train-reward    Train Phi on D_csil, run verification
  train-ensemble  PBRS fine-tune a_pi using Phi (skeleton for Task 4)
  verify          Run coherent reward verification on collected data

Usage:
    python train_csil_plus_plus.py --help
    python train_csil_plus_plus.py collect --n_episodes 200
    python train_csil_plus_plus.py train-reward
    python train_csil_plus_plus.py verify
    python train_csil_plus_plus.py train-ensemble

Notes:
  - V59 backbone (ResNet-18 features_extractor) is ALWAYS frozen.
  - Image augmentation is DISABLED during data collection (project_memory
    hard constraint).
  - BN running stats are frozen via features_extractor.eval().
  - All V59 forward passes use torch.no_grad() (frozen policy).
  - log_std is clamped to [-5, 2] for numerical stability.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

# Lazy imports for case_memory / version_tree / metadata are done inside
# functions so that `python train_csil_plus_plus.py --help` works without
# the full RL env / GPU stack being available.

# ---------------------------------------------------------------------------
# Constants — mirror collect_successful_trajectories.py and eval_hierarchical.py
# ---------------------------------------------------------------------------

GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"

TARGET_RANGE = "0.35,0.15,0.22,0.65,0.45,0.22"
TARGET_POS_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]

DCSIL_PATH = WORKSPACE / "data" / "D_csil.npz"
CSIL_OUTPUT_DIR = WORKSPACE / "outputs" / "csil_plus_plus"
POTENTIAL_FN_PATH = CSIL_OUTPUT_DIR / "potential_fn.pt"
VERIFICATION_REPORT_PATH = CSIL_OUTPUT_DIR / "verification_report.json"
TRAINING_LOG_PATH = CSIL_OUTPUT_DIR / "training_log.json"
ENSEMBLE_POLICY_PATH = CSIL_OUTPUT_DIR / "ensemble_policy.pt"

LIFT_THRESHOLD = 0.03    # m, grab success
PLACE_THRESHOLD = 0.05   # m, place success
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42

LATENT_DIM = 524  # features_extractor output dim (input to mlp_extractor.policy_net)
HIDDEN_DIM = 64   # mlp_extractor.policy_net output dim (input to action_net)
ACTION_DIM = 8   # 7 arm + 1 gripper
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

DEFAULT_ALPHA = 1.0
INCONSISTENCY_TOP_FRACTION = 0.30  # top 30% by disagreement
N_INCONSISTENCY_SAMPLES = 10


# ---------------------------------------------------------------------------
# V59 model loading helpers
# ---------------------------------------------------------------------------

def load_v59_state_dict(path: str = PLACE_MODEL_PATH) -> "OrderedDict[str, torch.Tensor]":
    """Load SB3 policy state_dict directly from the V59 zip (CPU, no env).

    Mirrors diagnose_ensemble_weights.load_state_dict(). The policy weights
    live in ``policy.pth`` (~45 MB); the 5 GB bulk is the replay buffer which
    we do NOT need.

    Parameters
    ----------
    path : str
        Path to V59 ``best_model.zip``.

    Returns
    -------
    OrderedDict[str, Tensor]
        The full SB3 ActorCriticPolicy state_dict.
    """
    with zipfile.ZipFile(path, "r") as archive:
        with archive.open("policy.pth") as f:
            sd = torch.load(f, map_location="cpu", weights_only=False)
    return sd  # type: ignore[return-value]


def freeze_backbone(model) -> int:
    """Freeze ResNet-18 features_extractor; only MLP head is trainable.

    Also freezes BatchNorm running stats (calls ``.eval()`` on BN modules) so
    that V59's learned normalization statistics are not perturbed during
    fine-tuning. Mirrors train_bc_only.freeze_backbone().
    """
    fe = model.policy.features_extractor
    for p in fe.parameters():
        p.requires_grad = False
    for m in fe.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"Backbone frozen: {total} total params, {trainable} trainable")
    return trainable


def normalize_states(states: torch.Tensor, vecnorm_path: Optional[str]) -> torch.Tensor:
    """Normalize 12-dim states using V59's VecNormalize statistics.

    Mirrors train_bc_only.normalize_states(). Operates only on the ``state``
    key; images are passed through unchanged (uint8 [0,255]).
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


def extract_v59_features(policy, obs: dict) -> torch.Tensor:
    """Run V59's frozen features_extractor to get 524-dim features.

    This is the INPUT to ``mlp_extractor.policy_net``, NOT the output.
    Use :func:`extract_v59_latent` for the 64-dim policy latent (output of
    ``mlp_extractor.policy_net``, input to ``action_net``).

    Parameters
    ----------
    policy : SB3 ActorCriticPolicy
        V59's policy (frozen). Called under torch.no_grad() by the caller.
    obs : dict
        ``{"image": Tensor (B,3,84,84) float, "state": Tensor (B,12) float}``.

    Returns
    -------
    Tensor
        Features of shape (B, 524) — features_extractor output.
    """
    features = policy.extract_features(obs)
    return features  # (B, 524)


def extract_v59_latent(policy, obs: dict) -> torch.Tensor:
    """Run V59's frozen backbone + policy_net to get the 64-dim policy latent.

    This is the OUTPUT of ``mlp_extractor.policy_net`` (and the INPUT to
    ``action_net``). Use :func:`extract_v59_features` for the 524-dim
    features_extractor output that feeds ``mlp_extractor.policy_net`` (and
    the learned ``_LearnedHead``).

    Parameters
    ----------
    policy : SB3 ActorCriticPolicy
        V59's policy (frozen). Called under torch.no_grad() by the caller.
    obs : dict
        ``{"image": Tensor (B,3,84,84) float, "state": Tensor (B,12) float}``.

    Returns
    -------
    Tensor
        Policy latent of shape (B, 64) — input to ``action_net``.
    """
    features = policy.extract_features(obs)
    latent = policy.mlp_extractor.forward_actor(features)
    return latent


def v59_action_mean(policy, obs: dict) -> torch.Tensor:
    """Return V59's deterministic action mean mu = action_net(latent).

    Runs entirely under torch.no_grad(); does NOT sample from the DiagGaussian.
    """
    latent = extract_v59_latent(policy, obs)
    mean = policy.action_net(latent)
    return mean


def v59_log_std(policy) -> torch.Tensor:
    """Return V59's log_std parameter, clamped to [-5, 2] for stability."""
    return policy.log_std.detach().clamp(LOG_STD_MIN, LOG_STD_MAX)


# ---------------------------------------------------------------------------
# SubTask 3.1: Coherent Reward Function
# ---------------------------------------------------------------------------

class GaussianMixturePrior:
    """Simple 2-component diagonal GMM prior over actions, fit once.

    Used as ``p(a|s)`` in the coherent reward when ``prior='gaussian_mixture'``.
    Fit via a few EM iterations on V59's collected actions.Kept dependency-free
    (no sklearn) so the file is self-contained.
    """

    def __init__(self, n_components: int = 2, n_iter: int = 50, tol: float = 1e-4,
                 seed: int = 0):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        # Fitted parameters
        self.means: list[np.ndarray] = []      # each (D,)
        self.vars: list[np.ndarray] = []       # each (D,) diagonal variance
        self.weights: np.ndarray = np.array([])  # (K,)
        self._fitted = False

    def fit(self, actions: np.ndarray) -> "GaussianMixturePrior":
        """Fit the GMM on a (N, D) array of actions via EM."""
        rng = np.random.RandomState(self.seed)
        actions = np.asarray(actions, dtype=np.float64)
        N, D = actions.shape
        K = self.n_components

        # Initialize means by random data points, equal weights, unit var.
        idx = rng.choice(N, K, replace=False)
        self.means = [actions[i].copy() for i in idx]
        self.vars = [np.ones(D) for _ in range(K)]
        self.weights = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        for it in range(self.n_iter):
            # E-step: compute responsibilities
            log_resp = np.zeros((N, K))
            for k in range(K):
                diff = actions - self.means[k]
                log_pdf = (
                    -0.5 * np.sum((diff ** 2) / self.vars[k], axis=1)
                    - 0.5 * np.sum(np.log(2.0 * math.pi * self.vars[k]))
                )
                log_resp[:, k] = np.log(self.weights[k] + 1e-12) + log_pdf
            # Normalize
            log_norm = np.logaddexp.reduce(log_resp, axis=1, keepdims=True)
            log_resp -= log_norm
            resp = np.exp(log_resp)  # (N, K)

            # M-step
            Nk = resp.sum(axis=0) + 1e-12
            for k in range(K):
                self.means[k] = (resp[:, k:k+1] * actions).sum(axis=0) / Nk[k]
                diff = actions - self.means[k]
                self.vars[k] = (resp[:, k:k+1] * diff ** 2).sum(axis=0) / Nk[k] + 1e-3
            self.weights = Nk / N

            ll = float(log_norm.sum())
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self._fitted = True
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log p(a) under the fitted GMM. Returns (N,) tensor."""
        if not self._fitted:
            raise RuntimeError("GaussianMixturePrior not fitted; call .fit() first.")
        device = actions.device
        actions_np = actions.detach().cpu().numpy().astype(np.float64)
        N, D = actions_np.shape
        K = self.n_components
        log_prob = np.full(N, -np.inf)
        for k in range(K):
            diff = actions_np - self.means[k]
            log_pdf = (
                -0.5 * np.sum((diff ** 2) / self.vars[k], axis=1)
                - 0.5 * np.sum(np.log(2.0 * math.pi * self.vars[k]))
            )
            log_pdf += np.log(self.weights[k] + 1e-12)
            log_prob = np.logaddexp(log_prob, log_pdf)
        return torch.as_tensor(log_prob, dtype=torch.float32, device=device)


def compute_coherent_reward(
    policy,
    states: Any,
    actions: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    prior: str = "uniform",
    gmm_prior: Optional[GaussianMixturePrior] = None,
    bc_head=None,
    batch_size: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Compute the CSIL++ coherent reward r(s,a) = alpha * (log pi_V59(a|s) - log p(a|s)).

    Parameters
    ----------
    policy : SB3 ActorCriticPolicy
        Frozen V59 policy. All forward passes use torch.no_grad().
    states : dict or Tensor
        Either a dict ``{"image": (N,3,84,84), "state": (N,12)}`` (full obs
        for V59 backbone) OR a precomputed latent Tensor of shape (N, 64)
        — the policy latent (output of ``mlp_extractor.policy_net``, input
        to ``action_net``). When a latent is passed, the backbone is skipped.
    actions : Tensor
        Actions of shape (N, 8) in [-1, 1].
    alpha : float
        Reward scale.
    prior : str
        ``'uniform'`` -> p(a) = (1/2)^D on [-1,1]^D, log p = -D*log(2).
        ``'gaussian_mixture'`` -> 2-component GMM prior (requires
        ``gmm_prior`` to be fitted).
        ``'bc_head'`` -> state-dependent BC head prior (requires ``bc_head``
        and dict states for feature extraction). Uses clamp(-12) per V2 spec.
    gmm_prior : GaussianMixturePrior, optional
        Pre-fitted GMM. Required if ``prior='gaussian_mixture'``.
    bc_head : nn.Module, optional
        Trained ShallowBCHead. Required if ``prior='bc_head'``.
    batch_size : int
        Forward batch size for V59 backbone passes.
    device : str
        Torch device for V59 forward.

    Returns
    -------
    Tensor
        Reward of shape (N,). For consistent states (action ~= mean),
        log pi is high (near max) so reward is high. For inconsistent
        states (action far from mean), log pi is low so reward is low.
    """
    if prior not in ("uniform", "gaussian_mixture", "bc_head"):
        raise ValueError(f"Unknown prior: {prior!r}")
    if prior == "gaussian_mixture" and gmm_prior is None:
        raise ValueError("prior='gaussian_mixture' requires a fitted gmm_prior.")
    if prior == "bc_head" and bc_head is None:
        raise ValueError("prior='bc_head' requires a bc_head model.")

    actions = actions.to(device)
    N = actions.shape[0]
    D = actions.shape[-1]
    if D != ACTION_DIM:
        raise ValueError(f"action dim {D} != expected {ACTION_DIM}")

    # V2 BC head prior uses clamp(-12) per user spec; V1 priors use clamp(-50).
    clamp_val = 12.0 if prior == "bc_head" else 50.0

    log_pi_all = []
    log_p_bc_all = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            act_batch = actions[i:i + batch_size]
            if isinstance(states, dict):
                img = states["image"][i:i + batch_size].to(device)
                st = states["state"][i:i + batch_size].to(device)
                obs = {"image": img, "state": st}
                if prior == "bc_head":
                    features = policy.extract_features(obs)
                    latent = policy.mlp_extractor.forward_actor(features)
                    mean = policy.action_net(latent)
                    bc_mu, bc_logsigma = bc_head(features)
                    std_bc = torch.exp(bc_logsigma)
                    diff_bc = act_batch.to(device) - bc_mu
                    log_p_bc = (
                        -0.5 * ((diff_bc / std_bc) ** 2).sum(dim=-1)
                        - bc_logsigma.sum(dim=-1)
                        - 0.5 * D * math.log(2.0 * math.pi)
                    )
                    log_p_bc_all.append(log_p_bc.cpu())
                else:
                    mean = v59_action_mean(policy, obs)
            else:
                latent = states[i:i + batch_size].to(device)
                mean = policy.action_net(latent)
                if prior == "bc_head":
                    raise ValueError(
                        "prior='bc_head' requires dict states for feature extraction"
                    )
            log_std = v59_log_std(policy).to(device)  # (D,)
            std = torch.exp(log_std)
            if std.dim() == 1:
                std = std.unsqueeze(0).expand_as(mean)
                log_std_b = log_std.unsqueeze(0).expand_as(mean)
            else:
                log_std_b = log_std
            z = (act_batch - mean) / std
            log_pi = (
                -0.5 * (z ** 2).sum(dim=-1)
                - log_std_b.sum(dim=-1)
                - 0.5 * D * math.log(2.0 * math.pi)
            )
            log_pi = log_pi.clamp(min=-clamp_val, max=clamp_val)
            log_pi_all.append(log_pi.cpu())
    log_pi = torch.cat(log_pi_all, dim=0)  # (N,)

    # Prior log p(a)
    if prior == "uniform":
        log_p = -float(D) * math.log(2.0)
        log_p_tensor = torch.full((N,), log_p, dtype=torch.float32)
    elif prior == "gaussian_mixture":
        log_p_tensor = gmm_prior.log_prob(actions.cpu()).float()  # type: ignore[union-attr]
        log_p_tensor = log_p_tensor.clamp(min=-50.0, max=50.0)
    else:  # bc_head
        log_p_tensor = torch.cat(log_p_bc_all, dim=0)
        log_p_tensor = log_p_tensor.clamp(min=-clamp_val, max=clamp_val)

    reward = alpha * (log_pi - log_p_tensor)
    return reward


def gate_1b_check(
    policy,
    bc_head,
    states,
    actions,
    alpha0: float = 0.1,
    batch_size: int = 256,
    device: str = "cpu",
    n_samples: int = 1024,
) -> dict:
    """Gate 1b: verify BC head coherent reward has no NaN/inf.

    Computes r = alpha * (clamp(logpi_V59, -12) - clamp(log_p_BC, -12))
    on a sample of transitions and checks for numerical stability.

    Returns dict with pass/fail and reward statistics.
    """
    n = len(actions)
    if n > n_samples:
        idx = np.random.RandomState(42).choice(n, n_samples, replace=False)
    else:
        idx = np.arange(n)

    sample_states = {k: v[idx] for k, v in states.items()} if isinstance(states, dict) else states[idx]
    sample_actions = actions[idx]

    reward = compute_coherent_reward(
        policy=policy,
        states=sample_states,
        actions=sample_actions,
        alpha=alpha0,
        prior="bc_head",
        bc_head=bc_head,
        batch_size=batch_size,
        device=device,
    )

    has_nan = bool(torch.isnan(reward).any())
    has_inf = bool(torch.isinf(reward).any())
    reward_np = reward.numpy()
    reward_std = float(reward_np.std())
    # Non-constancy check: CSIL++ V1 failed because reward was CONSTANT
    # (std=0) due to deterministic collection + state-independent log_std.
    # With BC head prior, reward should vary because mu_BC(s) != mu_V59(s)
    # exactly (shallow MLP imperfectly approximates V59's mlp_extractor).
    reward_is_constant = reward_std < 1e-6

    result = {
        "n_samples": len(reward),
        "alpha0": alpha0,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "reward_min": float(reward_np.min()),
        "reward_max": float(reward_np.max()),
        "reward_mean": float(reward_np.mean()),
        "reward_std": reward_std,
        "reward_is_constant": reward_is_constant,
        "gate_1b_pass": not has_nan and not has_inf and not reward_is_constant,
    }
    return result

def detect_inconsistency(
    policy,
    states: Any,
    n_samples: int = N_INCONSISTENCY_SAMPLES,
    top_fraction: float = INCONSISTENCY_TOP_FRACTION,
    batch_size: int = 256,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect states where V59's mean action is unreliable.

    For each state s:
      1. Get V59's deterministic action a_det = mean.
      2. Sample ``n_samples`` stochastic actions a_i ~ pi(.|s).
      3. Compute disagreement d(s) = mean_i ||a_i - a_det||_2.

    States with high d(s) are "inconsistent" — V59's mean is unreliable there
    (the Gaussian has high entropy in that region of state space, so the
    expected action varies widely across samples).

    Parameters
    ----------
    policy : SB3 ActorCriticPolicy
        Frozen V59 policy.
    states : dict or Tensor
        Full obs dict or precomputed latent (N, 64) — the policy latent
        (output of ``mlp_extractor.policy_net``). See
        ``compute_coherent_reward``.
    n_samples : int
        Number of stochastic action samples per state.
    top_fraction : float
        Fraction of states to flag as inconsistent (default 0.30 = top 30%).
    batch_size : int
        Forward batch size.
    device : str
        Torch device.

    Returns
    -------
    disagreement : Tensor (N,)
        Per-state disagreement score d(s).
    inconsistent_mask : Tensor (N,) bool
        True for the top ``top_fraction`` states by disagreement.
    """
    N_total: int
    if isinstance(states, dict):
        N_total = states["state"].shape[0]
    else:
        N_total = states.shape[0]

    # First pass: compute mean (deterministic action) per state.
    means_all = []
    with torch.no_grad():
        for i in range(0, N_total, batch_size):
            if isinstance(states, dict):
                img = states["image"][i:i + batch_size].to(device)
                st = states["state"][i:i + batch_size].to(device)
                obs = {"image": img, "state": st}
                mean = v59_action_mean(policy, obs)
            else:
                latent = states[i:i + batch_size].to(device)
                mean = policy.action_net(latent)
            means_all.append(mean.cpu())
    means = torch.cat(means_all, dim=0)  # (N, D)

    log_std = v59_log_std(policy).cpu()  # (D,)
    std = torch.exp(log_std)  # (D,)

    # Sample n_samples stochastic actions and accumulate disagreement.
    # We loop over samples (cheap) rather than building (N, n_samples, D).
    disagreement = torch.zeros(N_total, dtype=torch.float32)
    for _ in range(n_samples):
        eps = torch.randn_like(means)
        a_sample = means + eps * std  # (N, D)
        disagreement += (a_sample - means).norm(dim=-1)  # (N,)
    disagreement /= n_samples

    # Top `top_fraction` mask
    k = max(1, int(top_fraction * N_total))
    if k >= N_total:
        inconsistent_mask = torch.ones(N_total, dtype=torch.bool)
    else:
        # Threshold = k-th largest disagreement value
        topk_vals, _ = torch.topk(disagreement, k, largest=True)
        threshold = topk_vals.min()
        inconsistent_mask = disagreement >= threshold
    return disagreement, inconsistent_mask


# ---------------------------------------------------------------------------
# SubTask 3.3: Potential Function Phi(s)
# ---------------------------------------------------------------------------

class PotentialFunction(nn.Module):
    """Small MLP potential function Phi(s) for PBRS.

    Input can be either V59's latent features (524-dim) or the raw 12-dim
    state. Architecture: Linear(in,128) -> Tanh -> Linear(128,64) -> Tanh ->
    Linear(64,1). Output is a scalar Phi(s) used as the PBRS potential:

        F(s, s', gamma) = gamma * Phi(s') - Phi(s)

    which leaves the optimal policy of an agent unchanged (PBRS theorem).
    """

    def __init__(self, in_dim: int = LATENT_DIM):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Phi(s) of shape (B,) (squeeze last dim)."""
        return self.net(x).squeeze(-1)


def train_potential_function(
    potential_fn: PotentialFunction,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 256,
    device: str = "cpu",
    log_interval: int = 10,
) -> list[float]:
    """Train Phi(s) to classify success (+1) vs failure (-1) states.

    Uses MSE loss directly on the +/-1 labels (equivalent to a regression
    formulation; BCE on (label+1)/2 would also work and gives the same
    decision boundary up to a monotonic transform).

    Parameters
    ----------
    potential_fn : PotentialFunction
        Module to train (in-place).
    optimizer : torch.optim.Optimizer
        Optimizer bound to potential_fn.parameters().
    states : Tensor (N, in_dim)
        Latent features (or raw states) for each transition.
    labels : Tensor (N,)
        +1 for states from successful trajectories, -1 for failure.
    n_epochs : int
    batch_size : int
    device : str
    log_interval : int
        Print loss every N epochs.

    Returns
    -------
    list[float]
        Per-epoch average loss history.
    """
    potential_fn.to(device)
    states = states.to(device)
    labels = labels.to(device).float()
    dataset = TensorDataset(states, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    loss_history: list[float] = []
    for epoch in range(n_epochs):
        epoch_losses: list[float] = []
        potential_fn.train()
        for batch_states, batch_labels in loader:
            optimizer.zero_grad()
            pred = potential_fn(batch_states)
            loss = F.mse_loss(pred, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(potential_fn.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        avg = float(np.mean(epoch_losses))
        loss_history.append(avg)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  Phi epoch {epoch+1}/{n_epochs}: mse_loss={avg:.6f}")
    return loss_history


# ---------------------------------------------------------------------------
# SubTask 3.4: Verification
# ---------------------------------------------------------------------------

def verify_coherent_reward(
    policy,
    states: Any,
    actions: torch.Tensor,
    alpha: float = DEFAULT_ALPHA,
    prior: str = "uniform",
    gmm_prior: Optional[GaussianMixturePrior] = None,
    bc_head=None,
    n_samples: int = N_INCONSISTENCY_SAMPLES,
    top_fraction: float = INCONSISTENCY_TOP_FRACTION,
    device: str = "cpu",
) -> dict:
    """Verify that |r(s,a)| is significantly higher on inconsistent states.

    Splits states into:
      - "consistent": bottom ``top_fraction`` by disagreement
      - "inconsistent": top ``top_fraction`` by disagreement

    Computes mean |r(s,a)| for each group and asserts the inconsistent group
    has higher mean |r| than the consistent group. This validates that the
    coherent reward indeed identifies V59's inconsistent behavior.

    Parameters
    ----------
    policy, states, actions, alpha, prior, gmm_prior, n_samples, device :
        See ``compute_coherent_reward`` and ``detect_inconsistency``.
    top_fraction : float
        Fraction for the consistent / inconsistent split (default 0.30).

    Returns
    -------
    dict
        Verification statistics, also written to
        ``outputs/csil_plus_plus/verification_report.json``.
    """
    print("\n=== Coherent Reward Verification ===")
    disagreement, _ = detect_inconsistency(
        policy, states, n_samples=n_samples, top_fraction=top_fraction,
        device=device,
    )

    N = disagreement.shape[0]
    k = max(1, int(top_fraction * N))
    if k >= N:
        consistent_mask = torch.zeros(N, dtype=torch.bool)
        inconsistent_mask = torch.ones(N, dtype=torch.bool)
    else:
        sorted_vals, _ = torch.sort(disagreement)
        consistent_thresh = sorted_vals[k - 1]
        inconsistent_thresh = sorted_vals[N - k]
        consistent_mask = disagreement <= consistent_thresh
        inconsistent_mask = disagreement >= inconsistent_thresh

    print(f"  N={N}  consistent={int(consistent_mask.sum())}  "
          f"inconsistent={int(inconsistent_mask.sum())}")

    reward = compute_coherent_reward(
        policy, states, actions, alpha=alpha, prior=prior,
        gmm_prior=gmm_prior, bc_head=bc_head, device=device,
    )
    abs_r = reward.abs()

    def _stats(t: torch.Tensor) -> dict:
        if t.numel() == 0:
            return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "n": int(t.numel()),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()) if t.numel() > 1 else 0.0,
            "min": float(t.min().item()),
            "max": float(t.max().item()),
        }

    consistent_stats = _stats(abs_r[consistent_mask])
    inconsistent_stats = _stats(abs_r[inconsistent_mask])
    ratio = (inconsistent_stats["mean"] / consistent_stats["mean"]
             if consistent_stats["mean"] > 1e-9 else float("inf"))

    passed = inconsistent_stats["mean"] > consistent_stats["mean"]

    report = {
        "alpha": alpha,
        "prior": prior,
        "n_total": int(N),
        "n_consistent": int(consistent_mask.sum()),
        "n_inconsistent": int(inconsistent_mask.sum()),
        "consistent": consistent_stats,
        "inconsistent": inconsistent_stats,
        "ratio_inconsistent_to_consistent": ratio,
        "assertion_passed": bool(passed),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print(f"  Consistent   |r|: mean={consistent_stats['mean']:.4f}  "
          f"std={consistent_stats['std']:.4f}  "
          f"min={consistent_stats['min']:.4f}  max={consistent_stats['max']:.4f}")
    print(f"  Inconsistent |r|: mean={inconsistent_stats['mean']:.4f}  "
          f"std={inconsistent_stats['std']:.4f}  "
          f"min={inconsistent_stats['min']:.4f}  max={inconsistent_stats['max']:.4f}")
    print(f"  Ratio (inconsistent / consistent): {ratio:.3f}")
    print(f"  Assertion (inconsistent > consistent): "
          f"{'PASSED' if passed else 'FAILED'}")

    if not passed:
        # Soft warning instead of hard assert for the report path; callers
        # can choose to hard-fail by inspecting report['assertion_passed'].
        print("  WARNING: coherent reward did NOT separate inconsistent states.")

    # Persist report
    CSIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(VERIFICATION_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report written to {VERIFICATION_REPORT_PATH}")

    return report


# ---------------------------------------------------------------------------
# EnsemblePolicy (skeleton for Task 4)
# ---------------------------------------------------------------------------

class EnsemblePolicy(nn.Module):
    """Ensemble of V59 (frozen) and a learned policy a_pi.

    The ensemble action is

        a_ens = 0.5 * (a_V59 + a_pi)

    When ``a_pi`` is untrained (initialized as a copy of V59's MLP head),
    ``a_ens = a_V59`` (safe degradation). ``safety_rollback()`` restores this
    behavior if eval place_rate drops below 30%.

    Note: training of ``a_pi`` via PBRS is Task 4. This class provides the
    inference / safety-rollback infrastructure only.
    """

    def __init__(self, v59_policy, learned_policy: Optional[nn.Module] = None,
                 input_dim: int = LATENT_DIM, hidden_dim: int = HIDDEN_DIM,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        # V59 is frozen — store as attribute but do NOT register its parameters
        # for gradient tracking. We keep a reference for forward passes.
        self.v59_policy = v59_policy
        for p in self.v59_policy.parameters():
            p.requires_grad = False

        # Learned policy: a small head mirroring V59's mlp_extractor.policy_net
        # + action_net, initialized from V59 so a_ens = a_V59 at start.
        if learned_policy is not None:
            self.learned_policy = learned_policy
        else:
            self.learned_policy = _LearnedHead(
                input_dim=input_dim, hidden_dim=hidden_dim, action_dim=action_dim,
            )
            self._init_learned_from_v59()

        self._rolled_back = False
        self.action_dim = action_dim

    def _init_learned_from_v59(self) -> None:
        """Initialize the learned head as a copy of V59's MLP head."""
        try:
            v59_sd = self.v59_policy.state_dict()
            target_sd = self.learned_policy.state_dict()
            copied = {}
            # Map V59 keys -> _LearnedHead keys
            mapping = {
                "mlp_extractor.policy_net.0.weight": "policy_net.0.weight",
                "mlp_extractor.policy_net.0.bias": "policy_net.0.bias",
                "mlp_extractor.policy_net.2.weight": "policy_net.2.weight",
                "mlp_extractor.policy_net.2.bias": "policy_net.2.bias",
                "action_net.weight": "action_net.weight",
                "action_net.bias": "action_net.bias",
            }
            for src, dst in mapping.items():
                if src in v59_sd and dst in target_sd:
                    if v59_sd[src].shape == target_sd[dst].shape:
                        copied[dst] = v59_sd[src].clone()
            if copied:
                target_sd.update(copied)
                self.learned_policy.load_state_dict(target_sd)
                print(f"  EnsemblePolicy: learned head initialized from V59 "
                      f"({len(copied)} tensors copied)")
            else:
                print("  EnsemblePolicy: WARNING could not copy V59 head; "
                      "learned head uses random init.")
        except Exception as e:
            print(f"  EnsemblePolicy: WARNING init_from_v59 failed: {e}")

    def forward(self, states: Any) -> torch.Tensor:
        """Return a_ens = 0.5 * (a_V59 + a_pi).

        Parameters
        ----------
        states : dict or Tensor
            Full obs dict ``{"image": ..., "state": ...}`` (both V59 and
            learned head go through V59's frozen backbone) OR a precomputed
            features Tensor (N, 524) — the features_extractor output that
            feeds both ``mlp_extractor.policy_net`` and ``learned_policy``.
        """
        with torch.no_grad():
            if isinstance(states, dict):
                features = extract_v59_features(self.v59_policy, states)  # (B, 524)
                latent = self.v59_policy.mlp_extractor.forward_actor(features)  # (B, 64)
            else:
                # Assume precomputed features (524-dim).
                features = states
                latent = self.v59_policy.mlp_extractor.forward_actor(features)  # (B, 64)
            a_v59 = self.v59_policy.action_net(latent)  # 64-dim -> 8-dim
        if self._rolled_back:
            return a_v59
        a_pi = self.learned_policy(features)  # 524-dim -> 8-dim
        return 0.5 * (a_v59 + a_pi)

    def get_action(self, states: Any, deterministic: bool = True) -> torch.Tensor:
        """Return ensemble action (matches SB3 .predict() return convention
        for the action only)."""
        return self.forward(states)

    def safety_rollback(self) -> None:
        """Roll back to V59-only behavior.

        Call this if eval place_rate < 30%. After rollback, ``forward``
        returns a_V59 directly (a_pi is bypassed).
        """
        self._rolled_back = True
        print("  EnsemblePolicy: SAFETY ROLLBACK engaged — using V59 only.")

    def restore_ensemble(self) -> None:
        """Re-enable ensemble averaging after a safety rollback."""
        self._rolled_back = False
        print("  EnsemblePolicy: ensemble re-enabled.")


class _LearnedHead(nn.Module):
    """Mirror of V59's mlp_extractor.policy_net + action_net.

    Two-layer MLP (Linear-Tanh-Linear-Tanh) -> Linear action projection.
    Architecture mirrors V59 exactly so weights can be copied verbatim:

      - policy_net.0: Linear(LATENT_DIM=524, HIDDEN_DIM=64)
      - policy_net.2: Linear(HIDDEN_DIM=64, HIDDEN_DIM=64)
      - action_net  : Linear(HIDDEN_DIM=64, ACTION_DIM=8)

    Input is the 524-dim features_extractor output (NOT the 64-dim policy
    latent). Initialized from V59's weights by
    :meth:`EnsemblePolicy._init_learned_from_v59`.
    """

    def __init__(self, input_dim: int = LATENT_DIM, hidden_dim: int = HIDDEN_DIM,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),    # Linear(524, 64) — matches V59
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),   # Linear(64, 64) — matches V59
            nn.Tanh(),
        )
        self.action_net = nn.Linear(hidden_dim, action_dim)  # Linear(64, 8) — matches V59

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = self.policy_net(features)  # features is 524-dim
        return self.action_net(h)      # 8-dim action


class ValueHead(nn.Module):
    """Small MLP value function V(s) for PPO, trained alongside a_pi.

    Takes the (frozen) V59 latent (524-dim) and outputs a scalar value
    estimate.  Used as the critic in the PPO update of ``a_pi``.  Not part
    of ``EnsemblePolicy`` (the spec requires the value head live alongside
    the training function, not inside the ensemble class).
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return V(s) of shape (B,) (squeeze last dim)."""
        return self.net(latent).squeeze(-1)


class _EnsemblePlaceAdapter:
    """Wraps :class:`EnsemblePolicy` to expose ``.predict()`` for
    :class:`HierarchicalPickPlacePolicy`.

    ``HierarchicalPickPlacePolicy`` calls ``place_model.predict(obs, deterministic=...)``
    and expects a ``(numpy_action, None)`` tuple.  This adapter converts the
    dict obs to tensors, runs the ensemble forward pass, and returns the
    ensemble action as a numpy array.

    During **rollout** (``stochastic=True``) the adapter samples
    ``a_pi ~ N(mean=a_pi(features), std=exp(v59.log_std))`` and sends the safe
    average ``a_ens = 0.5 * (a_V59 + a_pi)`` to the env.  The sampled
    ``a_pi`` is stored in ``last_a_pi_sample`` for PPO log-prob computation.

    During **eval** (``stochastic=False``) the adapter uses the deterministic
    mean ``a_pi_mean = learned_policy(features)`` and sends
    ``a_ens = 0.5 * (a_V59 + a_pi_mean)`` to the env.

    Attributes (set after each ``predict()`` call):
        last_features    : Tensor (1, 524) — V59 features_extractor output
                          (CPU, detached) — feeds ``learned_policy``.
        last_latent      : Tensor (1, 64)  — V59 policy latent (CPU, detached)
                          — feeds ``action_net``, ``phi``, ``value_head``.
        last_a_pi_sample : Tensor (1, 8)   — a_pi action stored for PPO
    """

    def __init__(self, ensemble: "EnsemblePolicy", v59_policy, device: str,
                 v59_log_std: torch.Tensor):
        self.ensemble = ensemble
        self.v59_policy = v59_policy
        self.device = device
        # v59_log_std: (D,) or (1, D) tensor, frozen
        self.v59_log_std = v59_log_std.to(device).detach()
        self.stochastic = True
        # Cached outputs from the most recent predict() call
        self.last_features: Optional[torch.Tensor] = None
        self.last_latent: Optional[torch.Tensor] = None
        self.last_a_pi_sample: Optional[torch.Tensor] = None

    def predict(self, obs, deterministic: bool = True):
        """Return ``(action_numpy, None)`` following SB3 convention.

        Parameters
        ----------
        obs : dict
            Normalized vision obs ``{"image": (1,3,84,84), "state": (1,12)}``.
        deterministic : bool
            Ignored — sampling is controlled by ``self.stochastic``.
        """
        with torch.no_grad():
            img = torch.as_tensor(
                obs["image"], dtype=torch.float32, device=self.device)
            st = torch.as_tensor(
                obs["state"], dtype=torch.float32, device=self.device)
            obs_t = {"image": img, "state": st}
            features = extract_v59_features(self.v59_policy, obs_t)  # (1, 524)
            latent = self.v59_policy.mlp_extractor.forward_actor(features)  # (1, 64)
            a_v59 = self.v59_policy.action_net(latent)               # (1, 8)

            if self.ensemble._rolled_back:
                a_ens = a_v59
                a_pi_sample = a_v59
            else:
                a_pi_mean = self.ensemble.learned_policy(features)  # (1, 8)
                if self.stochastic:
                    log_std = self.v59_log_std
                    if log_std.dim() == 1:
                        std = log_std.exp().unsqueeze(0).expand_as(a_pi_mean)
                    else:
                        std = log_std.exp().expand_as(a_pi_mean)
                    eps = torch.randn_like(a_pi_mean)
                    a_pi_sample = a_pi_mean + eps * std
                else:
                    a_pi_sample = a_pi_mean
                a_ens = 0.5 * (a_v59 + a_pi_sample)

            self.last_features = features.detach().cpu()
            self.last_latent = latent.detach().cpu()
            self.last_a_pi_sample = a_pi_sample.detach().cpu()
            action_np = a_ens.cpu().numpy().astype(np.float32)
        return action_np, None


# ---------------------------------------------------------------------------
# Data collection (collect subcommand)
# ---------------------------------------------------------------------------

def _build_collect_envs(args):
    """Build grasp + place vec envs and the raw eval env. Mirrors
    collect_successful_trajectories.py."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import functools
    import gymnasium  # noqa: F401
    import gym_env  # noqa: F401  registers PandaVLA-v0
    from gym_env.wrappers import FlattenObs, VisionObs
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_env(vision_mode=False, target_pos_range=None, domain_randomize=False):
        kwargs = dict(reward_type="dense", gravity_comp=True)
        if target_pos_range is not None:
            kwargs["target_pos_range"] = target_pos_range
        kwargs["domain_randomize"] = domain_randomize
        env = gymnasium.make("PandaVLA-v0", **kwargs)
        if vision_mode:
            env = VisionObs(env, image_size=84)
        else:
            env = FlattenObs(env)
        return env

    def load_model(model_path, vecnorm_path, vision_mode=False,
                   target_pos_range=None, domain_randomize=False):
        env_factory = lambda: make_env(
            vision_mode=vision_mode,
            target_pos_range=target_pos_range,
            domain_randomize=domain_randomize,
        )
        vec_env = DummyVecEnv([env_factory])
        if vecnorm_path and os.path.exists(vecnorm_path):
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            vec_env.norm_reward = False
            vec_env.training = False
        else:
            norm_obs_keys = ["state"] if vision_mode else None
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                                   clip_obs=10.0, norm_obs_keys=norm_obs_keys)
            vec_env.training = False
        model = PPO.load(model_path, env=vec_env, device="auto")
        return model, vec_env

    print(f"Loading grasp model: {args.grasp_model}")
    grasp_model, grasp_vec_env = load_model(
        args.grasp_model, args.grasp_vecnorm,
        vision_mode=False,
        target_pos_range=TARGET_POS_RANGE,
        domain_randomize=False,
    )
    print(f"Loading place model: {args.place_model}")
    place_model, place_vec_env = load_model(
        args.place_model, args.place_vecnorm,
        vision_mode=True,
        target_pos_range=TARGET_POS_RANGE,
        domain_randomize=False,
    )

    from hierarchical_policy import HierarchicalPickPlacePolicy
    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    raw_env = DummyVecEnv([lambda: make_env(
        target_pos_range=TARGET_POS_RANGE,
        domain_randomize=False,
    )])
    _inner_env = raw_env.envs[0].env.unwrapped
    _inner_env._release_dist_threshold = args.release_threshold
    _inner_env._release_height_threshold = float("inf")
    place_vision_wrapper = VisionObs(_inner_env, image_size=84)

    return {
        "policy": policy,
        "raw_env": raw_env,
        "inner_env": _inner_env,
        "grasp_vec_env": grasp_vec_env,
        "place_vec_env": place_vec_env,
        "place_vision_wrapper": place_vision_wrapper,
    }


def cmd_collect(args) -> None:
    """Collect V59 trajectories (success + failure) and save to D_csil.npz.

    Reuses logic from collect_successful_trajectories.py but records BOTH
    successes and failures with labels, and saves as .npz (not .pkl) with
    arrays: images, states, actions, labels, episode_ids, final_dists.
    """
    print("=" * 60)
    print("CSIL++ Data Collection (success + failure)")
    print("=" * 60)
    print(f"Place model: {args.place_model}")
    print(f"Grasp model: {args.grasp_model}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Output: {args.output}")
    print(f"Image augmentation: DISABLED (project_memory hard constraint)")
    print()

    envs = _build_collect_envs(args)
    policy = envs["policy"]
    raw_env = envs["raw_env"]
    _inner_env = envs["inner_env"]
    place_vec_env = envs["place_vec_env"]
    place_vision_wrapper = envs["place_vision_wrapper"]

    np.random.seed(args.seed)
    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    all_images: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_labels: list[int] = []
    all_ep_ids: list[int] = []
    all_final_dists: list[float] = []

    n_placed = 0
    n_grabbed = 0
    n_entered_place = 0
    t0 = time.time()

    for ep in range(args.n_episodes):
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        ep_target_pos = _inner_env._target_pos.copy()
        max_lift = 0.0
        block_target_dist = float("inf")
        first_place_step = None
        prev_info = None

        ep_images: list[np.ndarray] = []
        ep_states: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []

        for step in range(args.max_steps):
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                _inner_env.place_mode = True
                _inner_env._place_gravcomp_active = True
                _inner_env.snap_block_to_hand()
                _inner_env._arm_target = _inner_env.data.qpos[
                    _inner_env._arm_qpos_adrs
                ].copy()
                _inner_env._gripper_target = float(
                    _inner_env.data.qpos[_inner_env._finger_qpos_adrs].mean()
                )
                _inner_env.reward_type = "place_only"
                _inner_env._place_approach_bonus_given = False
                _inner_env._place_proximity_15_given = False
                _inner_env._place_proximity_10_given = False
                _inner_env._place_success = False
                _inner_env._prev_block_target_dist = None
                _inner_env._prev_block_height = None
                _inner_env._use_gripper_target_check = True

                flatten_wrapper = raw_env.envs[0]
                inner_obs = _inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

            if phase == "place":
                vision_obs = place_vision_wrapper.observation(_inner_env._get_obs())
                vision_obs_batched = {
                    "image": vision_obs["image"][np.newaxis, ...],
                    "state": vision_obs["state"][np.newaxis, ...],
                }
                obs = place_vec_env.normalize_obs(vision_obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
            else:
                raw_obs_for_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_for_grasp[0, 8:11]
                default_target = np.array([0.5, 0.3, 0.2])
                raw_obs_for_grasp[0, 15] = np.linalg.norm(block_pos - default_target)
                obs = envs["grasp_vec_env"].normalize_obs(raw_obs_for_grasp)

            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            if phase == "place":
                # Store RAW (unnormalized) data — training normalizes later.
                ep_images.append(vision_obs["image"].copy())    # (84,84,3) uint8
                ep_states.append(vision_obs["state"].copy())    # (12,) float32
                ep_actions.append(action[0].copy())             # (8,) float32

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            i = info[0]
            block_h = float(i.get("block_height", 0.0))
            block_target_dist = float(
                i.get("block_target_distance", block_target_dist)
            )
            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            if done[0]:
                break

        grabbed = max_lift > LIFT_THRESHOLD
        entered_place = first_place_step is not None
        placed = block_target_dist < PLACE_THRESHOLD

        if grabbed:
            n_grabbed += 1
        if entered_place:
            n_entered_place += 1

        # Record BOTH successes and failures (must have entered place phase
        # so we have place-phase transitions).
        if len(ep_images) > 0:
            label = 1 if placed else 0
            n_trans = len(ep_images)
            for j in range(n_trans):
                all_images.append(ep_images[j])
                all_states.append(ep_states[j])
                all_actions.append(ep_actions[j])
                all_labels.append(label)
                all_ep_ids.append(ep)
                all_final_dists.append(float(block_target_dist))
            if placed:
                n_placed += 1

        ep_status = "PLACED" if placed else ("grabbed" if grabbed else "failed")
        elapsed = time.time() - t0
        print(f"Ep {ep:3d}: {ep_status:7s}  dist={block_target_dist*100:5.1f}cm  "
              f"lift={max_lift*100:5.1f}cm  place_steps={len(ep_images):3d}  "
              f"| placed={n_placed}/{ep+1}  transitions={len(all_images)}  "
              f"[{elapsed:.0f}s]")

    # ---- Save ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        images=np.array(all_images, dtype=np.uint8) if all_images else np.zeros((0, 84, 84, 3), dtype=np.uint8),
        states=np.array(all_states, dtype=np.float32) if all_states else np.zeros((0, 12), dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32) if all_actions else np.zeros((0, 8), dtype=np.float32),
        labels=np.array(all_labels, dtype=np.int64) if all_labels else np.zeros((0,), dtype=np.int64),
        episode_ids=np.array(all_ep_ids, dtype=np.int64) if all_ep_ids else np.zeros((0,), dtype=np.int64),
        final_dists=np.array(all_final_dists, dtype=np.float32) if all_final_dists else np.zeros((0,), dtype=np.float32),
    )
    print()
    print("=" * 60)
    print("Collection Complete")
    print("=" * 60)
    print(f"Episodes run:        {ep + 1}")
    print(f"Grabbed (lift>3cm):  {n_grabbed}/{ep+1}")
    print(f"Entered place phase: {n_entered_place}/{ep+1}")
    print(f"Placed (dist<5cm):   {n_placed}/{ep+1}")
    print(f"Total transitions:   {len(all_images)}")
    n_succ = sum(1 for l in all_labels if l == 1)
    print(f"  Success transitions:   {n_succ}")
    print(f"  Failure transitions:   {len(all_labels) - n_succ}")
    print(f"Output: {out_path}")
    print(f"Elapsed: {time.time() - t0:.0f}s")

    raw_env.close()


# ---------------------------------------------------------------------------
# V59 policy loader for train-reward / verify / train-ensemble
# ---------------------------------------------------------------------------

def _load_v59_policy(args, device: str):
    """Load the full V59 PPO model (DAPGPPO) with VecNormalize env.

    Returns (model, place_vec_env). Mirrors train_bc_only.py loading.
    """
    import functools
    import pickle
    import gymnasium  # noqa: F401
    import gym_env  # noqa: F401
    from gym_env.wrappers import FlattenObs
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, VecNormalize, VecTransposeImage,
    )
    from train_dapg import DAPGPPO
    from train_place_policy import make_env

    grasp_states_path = str(WORKSPACE / "outputs/grasp_states_v5_500.pkl")
    grasp_states = None
    if os.path.exists(grasp_states_path):
        with open(grasp_states_path, "rb") as f:
            grasp_states = pickle.load(f)

    env_kwargs = dict(
        grasp_states=grasp_states,
        release_threshold=0.05,
        target_pos_range=TARGET_POS_RANGE,
        vision_mode=True,
        domain_randomize=False,
        better_reward=False,
        use_pbrs=False,
        pbrs_alpha=1.0, pbrs_beta=0.0, pbrs_scale=0.5,
    )
    train_env = DummyVecEnv([functools.partial(make_env, **env_kwargs)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        norm_obs_keys=["state"],
    )
    if args.place_vecnorm and os.path.exists(args.place_vecnorm):
        train_env = VecNormalize.load(args.place_vecnorm, train_env)
    train_env = VecTransposeImage(train_env)

    model = DAPGPPO.load(
        args.place_model,
        env=train_env,
        device=device,
        demo_obs=None,
        demo_actions=None,
    )
    print("V59 model loaded.")
    return model, train_env


def _load_d_csil(path: str) -> dict:
    """Load data/D_csil.npz -> dict of tensors / arrays."""
    data = np.load(path, allow_pickle=True)
    out = {
        "images": torch.as_tensor(data["images"], dtype=torch.float32),
        "states": torch.as_tensor(data["states"], dtype=torch.float32),
        "actions": torch.as_tensor(data["actions"], dtype=torch.float32),
        "labels": torch.as_tensor(data["labels"], dtype=torch.long),
        "episode_ids": torch.as_tensor(data["episode_ids"], dtype=torch.long),
        "final_dists": torch.as_tensor(data["final_dists"], dtype=torch.float32),
    }
    print(f"D_csil loaded: {len(out['actions'])} transitions "
          f"({int((out['labels'] == 1).sum())} success, "
          f"{int((out['labels'] == 0).sum())} failure)")
    return out


# ---------------------------------------------------------------------------
# train-reward subcommand
# ---------------------------------------------------------------------------

def cmd_train_reward(args) -> None:
    """Train the potential function Phi on D_csil and run verification."""
    print("=" * 60)
    print("CSIL++ Train Reward (Potential Function Phi)")
    print("=" * 60)

    CSIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Record metadata
    from auto_iter.metadata import record_experiment
    meta = record_experiment(
        experiment_id=args.experiment_id,
        optimization_method="CSIL++_reward",
        parent_experiment_id="V59",
        decision_reason=(
            "CSIL++ Task 3: train potential function Phi(s) from success/"
            "failure labels and verify coherent reward r(s,a)="
            "alpha*(log pi_V59 - log p)."
        ),
        random_seed=args.seed,
        training_config={
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "alpha": args.alpha,
            "prior": args.prior,
            "phi_in_dim": "latent" if args.phi_input == "latent" else 12,
        },
        eval_config={"method": "coherent_reward_verification"},
        train_cmd=" ".join(sys.argv),
        save_path=str(CSIL_OUTPUT_DIR),
        cwd=str(WORKSPACE),
    )
    print(f"Metadata recorded: git={meta.git_commit[:12]}")

    # Query case memory
    from auto_iter.case_memory import CaseMemory
    cm = CaseMemory()
    warnings = cm.query({"optimization_method": "CSIL++"}, "CSIL++")
    if warnings:
        print("\n=== Case Memory Warnings ===")
        for w in warnings:
            print(f"  WARNING: {w.evidence_id}: {w.failure_mode}")
    else:
        print("Case memory: no CSIL++-specific failure modes recorded.")

    # Load D_csil
    if not os.path.exists(args.d_csil):
        print(f"\nERROR: D_csil not found at {args.d_csil}.")
        print("  Run `python train_csil_plus_plus.py collect` first.")
        sys.exit(1)
    print(f"\n=== Loading D_csil from {args.d_csil} ===")
    data = _load_d_csil(args.d_csil)
    images = data["images"]
    states_raw = data["states"]
    actions = data["actions"]
    labels = data["labels"]

    # Load V59 model
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n=== Loading V59 from {args.place_model} ===")
    model, _ = _load_v59_policy(args, device)
    freeze_backbone(model)
    model.policy.features_extractor.eval()

    # Normalize states using V59 VecNormalize stats
    states = normalize_states(states_raw, args.place_vecnorm)

    # Precompute V59 latents on the full dataset (one pass, no grad).
    print("\n=== Precomputing V59 latents ===")
    latents = _precompute_latents(model.policy, images, states, device,
                                  batch_size=args.batch_size)
    print(f"  Latents shape: {tuple(latents.shape)}")

    # Train Phi on latents
    print(f"\n=== Training Phi (input=latent, dim={latents.shape[1]}) ===")
    phi = PotentialFunction(in_dim=latents.shape[1])
    optimizer = torch.optim.Adam(phi.parameters(), lr=args.learning_rate)
    # labels: 1=success -> +1, 0=failure -> -1
    phi_labels = (labels.float() * 2.0 - 1.0)
    loss_history = train_potential_function(
        phi, optimizer, latents, phi_labels,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        device=device,
    )
    torch.save(phi.state_dict(), str(POTENTIAL_FN_PATH))
    print(f"\nPhi saved to {POTENTIAL_FN_PATH}")

    # Optionally fit GMM prior on collected actions
    gmm_prior = None
    bc_head = None
    if args.prior == "gaussian_mixture":
        print("\n=== Fitting 2-component GMM prior on actions ===")
        gmm_prior = GaussianMixturePrior(n_components=2, n_iter=100, seed=args.seed)
        gmm_prior.fit(actions.numpy())
        print(f"  GMM weights: {gmm_prior.weights}")
        print(f"  GMM means[0]: {gmm_prior.means[0][:4]} ...")
        print(f"  GMM means[1]: {gmm_prior.means[1][:4]} ...")
    elif args.prior == "bc_head":
        print(f"\n=== Loading BC head from {args.bc_head_path} ===")
        bc_head, _ = _load_bc_head(args.bc_head_path, device)

    # Verification
    obs_for_verify = {"image": images.permute(0, 3, 1, 2).contiguous(),
                      "state": states}
    verify_coherent_reward(
        model.policy, obs_for_verify, actions,
        alpha=args.alpha, prior=args.prior, gmm_prior=gmm_prior,
        bc_head=bc_head, device=device,
    )

    # Save training log
    log = {
        "experiment_id": args.experiment_id,
        "method": "CSIL++_reward",
        "phi_input": "latent",
        "phi_in_dim": int(latents.shape[1]),
        "n_transitions": int(len(actions)),
        "n_success": int((labels == 1).sum().item()),
        "n_failure": int((labels == 0).sum().item()),
        "alpha": args.alpha,
        "prior": args.prior,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "phi_loss_history": loss_history,
        "phi_init_loss": loss_history[0] if loss_history else None,
        "phi_final_loss": loss_history[-1] if loss_history else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log written to {TRAINING_LOG_PATH}")

    # Version tree
    from auto_iter.version_tree import VersionTree, make_node
    tree = VersionTree()
    node = make_node(
        experiment_id=args.experiment_id,
        parent_id="V59",
        optimization_method="CSIL++_reward",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        decision_reason="CSIL++ Task 3: train Phi + verify coherent reward.",
        config={
            "alpha": args.alpha,
            "prior": args.prior,
            "n_epochs": args.n_epochs,
            "learning_rate": args.learning_rate,
            "phi_input": "latent",
            "phi_in_dim": int(latents.shape[1]),
            "n_transitions": int(len(actions)),
        },
    )
    node.status = "completed"
    node.verdict = "pending"
    tree.add_node(node)
    print(f"\nAdded to version tree: {args.experiment_id} (parent=V59)")


def _precompute_latents(policy, images: torch.Tensor, states: torch.Tensor,
                        device: str, batch_size: int = 256) -> torch.Tensor:
    """Run V59's frozen backbone + policy_net on all (image, state) pairs.

    Returns latents of shape (N, 64) — the policy latent (output of
    ``mlp_extractor.policy_net``, input to ``action_net``). Used to train
    :class:`PotentialFunction` (Phi), which therefore takes a 64-dim input.
    All under torch.no_grad().
    """
    policy.eval()
    N = images.shape[0]
    all_latents: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            img = images[i:i + batch_size].permute(0, 3, 1, 2).contiguous().to(device)
            st = states[i:i + batch_size].to(device)
            obs = {"image": img, "state": st}
            latent = extract_v59_latent(policy, obs)
            all_latents.append(latent.cpu())
    return torch.cat(all_latents, dim=0)


# ---------------------------------------------------------------------------
# verify subcommand
# ---------------------------------------------------------------------------

def cmd_verify(args) -> None:
    """Run coherent reward verification on collected data."""
    print("=" * 60)
    print("CSIL++ Coherent Reward Verification")
    print("=" * 60)

    if not os.path.exists(args.d_csil):
        print(f"ERROR: D_csil not found at {args.d_csil}.")
        print("  Run `python train_csil_plus_plus.py collect` first.")
        sys.exit(1)

    data = _load_d_csil(args.d_csil)
    images = data["images"]
    states_raw = data["states"]
    actions = data["actions"]

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n=== Loading V59 from {args.place_model} ===")
    model, _ = _load_v59_policy(args, device)
    freeze_backbone(model)
    model.policy.features_extractor.eval()

    states = normalize_states(states_raw, args.place_vecnorm)

    gmm_prior = None
    bc_head = None
    if args.prior == "gaussian_mixture":
        print("\n=== Fitting 2-component GMM prior on actions ===")
        gmm_prior = GaussianMixturePrior(n_components=2, n_iter=100, seed=args.seed)
        gmm_prior.fit(actions.numpy())
    elif args.prior == "bc_head":
        print(f"\n=== Loading BC head from {args.bc_head_path} ===")
        bc_head, _ = _load_bc_head(args.bc_head_path, device)

    obs = {"image": images.permute(0, 3, 1, 2).contiguous(), "state": states}
    report = verify_coherent_reward(
        model.policy, obs, actions,
        alpha=args.alpha, prior=args.prior, gmm_prior=gmm_prior,
        bc_head=bc_head, device=device,
    )
    print(f"\nAssertion passed: {report['assertion_passed']}")
    print(f"Ratio (inconsistent / consistent): "
          f"{report['ratio_inconsistent_to_consistent']:.3f}")


# ---------------------------------------------------------------------------
# train-ensemble subcommand: PBRS fine-tuning of a_pi (Task 4)
# ---------------------------------------------------------------------------

def _compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                 next_values: torch.Tensor, dones: torch.Tensor,
                 gamma: float, gae_lambda: float
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards      : (T,) — environment (or shaped) rewards.
    values       : (T,) — V(s_t) from the value head.
    next_values  : (T,) — V(s_{t+1}); zeroed at terminal steps by the caller
                   or multiplied by (1 - done) here.
    dones        : (T,) — 1.0 if episode terminated at step t, else 0.0.
    gamma        : discount factor.
    gae_lambda   : GAE lambda.

    Returns
    -------
    advantages : (T,)
    returns    : (T,) — advantages + values (value-function targets).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(1, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def _compute_a_pi_log_prob(learned_policy: nn.Module, features: torch.Tensor,
                           actions: torch.Tensor,
                           v59_log_std: torch.Tensor) -> torch.Tensor:
    """Compute log pi(a|s) under a_pi's diagonal Gaussian distribution.

    ``a_pi ~ N(mean=learned_policy(features), std=exp(v59.log_std))`` where
    ``v59_log_std`` is frozen (copied from V59).  Same DiagGaussian formula
    as ``compute_coherent_reward``.

    Parameters
    ----------
    learned_policy : _LearnedHead (trainable — gradients flow through here).
    features       : (B, 524) — precomputed V59 features_extractor outputs
                     (no grad needed); these feed ``learned_policy``.
    actions        : (B, 8).
    v59_log_std    : (D,) or (1, D) frozen log-std from V59.

    Returns
    -------
    Tensor (B,) — log probabilities (with grad through ``learned_policy``).
    """
    mean = learned_policy(features)  # (B, 8)
    log_std = v59_log_std
    if log_std.dim() == 1:
        log_std_b = log_std.unsqueeze(0).expand_as(mean)
    else:
        log_std_b = log_std.expand_as(mean)
    std = log_std_b.exp()
    z = (actions - mean) / std
    log_prob = (
        -0.5 * (z ** 2).sum(dim=-1)
        - log_std_b.sum(dim=-1)
        - 0.5 * ACTION_DIM * math.log(2.0 * math.pi)
    )
    return log_prob.clamp(min=-50.0, max=50.0)


def _compute_coherent_reward_from_features(
    v59_policy,
    bc_head: Optional[nn.Module],
    features: torch.Tensor,
    actions: torch.Tensor,
    v59_log_std: torch.Tensor,
    device: str,
    batch_size: int = 512,
) -> Tuple[torch.Tensor, dict]:
    """CSIL++ V2: coherent reward from precomputed features.

    Computes ``r(s,a) = log_pi_V59(a|s) - log_p_BC(a|s)`` using the 524-dim
    features_extractor output already cached in the rollout (avoids re-running
    ResNet-18). Both V59 and BC head are frozen — no gradients flow here.

    Parameters
    ----------
    v59_policy    : frozen V59 ActorCriticPolicy (used for action_net + log_std).
    bc_head       : frozen ShallowBCHead (provides state-dependent mu, log_sigma).
    features      : (T, 524) — V59 features_extractor outputs.
    actions       : (T, 8) — a_pi samples stored in the rollout.
    v59_log_std   : (D,) frozen log_std from V59 (per-dim).
    device        : torch device.
    batch_size    : chunk size for the forward passes.

    Returns
    -------
    (coherent_reward, stats) where coherent_reward is (T,) on CPU and stats
    holds mean/std/min/max for logging.
    """
    if bc_head is None:
        return torch.zeros(features.shape[0]), {"mean": 0.0, "std": 0.0,
                                                 "min": 0.0, "max": 0.0}

    D = ACTION_DIM
    clamp_val = 12.0  # V2 spec: clamp(-12, 12)
    log_2pi = math.log(2.0 * math.pi)

    rewards_all: list[torch.Tensor] = []
    v59_policy.eval()
    bc_head.eval()
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            feat = features[i:i + batch_size].to(device)
            act = actions[i:i + batch_size].to(device)

            # V59 mean: action_net(mlp_extractor.forward_actor(features))
            latent = v59_policy.mlp_extractor.forward_actor(feat)
            v59_mu = v59_policy.action_net(latent)            # (B, 8)
            log_std = v59_log_std.to(device)
            if log_std.dim() == 1:
                log_std_b = log_std.unsqueeze(0).expand_as(v59_mu)
            else:
                log_std_b = log_std.expand_as(v59_mu)
            v59_std = log_std_b.exp()
            z_v59 = (act - v59_mu) / v59_std
            log_pi = (
                -0.5 * (z_v59 ** 2).sum(dim=-1)
                - log_std_b.sum(dim=-1)
                - 0.5 * D * log_2pi
            ).clamp(min=-clamp_val, max=clamp_val)

            # BC head mean + state-dependent log_sigma
            bc_mu, bc_logsigma = bc_head(feat)                # (B, 8), (B, 8)
            bc_std = bc_logsigma.exp()
            z_bc = (act - bc_mu) / bc_std
            log_p_bc = (
                -0.5 * (z_bc ** 2).sum(dim=-1)
                - bc_logsigma.sum(dim=-1)
                - 0.5 * D * log_2pi
            ).clamp(min=-clamp_val, max=clamp_val)

            rewards_all.append((log_pi - log_p_bc).cpu())

    coherent = torch.cat(rewards_all, dim=0)
    stats = {
        "mean": float(coherent.mean().item()),
        "std": float(coherent.std().item()),
        "min": float(coherent.min().item()),
        "max": float(coherent.max().item()),
    }
    return coherent, stats


def _build_train_envs(args) -> dict:
    """Build env components for PBRS training without loading the place model.

    Loads the grasp model (for the grasp phase) and V59's VecNormalize
    statistics (for place-phase obs normalization).  The place policy is the
    :class:`EnsemblePolicy` (plugged in by the caller), NOT a model loaded
    from disk.

    Returns a dict with keys: ``grasp_model``, ``grasp_vec_env``,
    ``place_vec_env``, ``raw_env``, ``inner_env``, ``place_vision_wrapper``.
    Mirrors ``_build_collect_envs`` but skips the place-model load.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    import functools  # noqa: F401
    import gymnasium  # noqa: F401
    import gym_env  # noqa: F401  registers PandaVLA-v0
    from gym_env.wrappers import FlattenObs, VisionObs
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_env(vision_mode=False, target_pos_range=None,
                 domain_randomize=False):
        kwargs = dict(reward_type="dense", gravity_comp=True)
        if target_pos_range is not None:
            kwargs["target_pos_range"] = target_pos_range
        kwargs["domain_randomize"] = domain_randomize
        env = gymnasium.make("PandaVLA-v0", **kwargs)
        if vision_mode:
            env = VisionObs(env, image_size=84)
        else:
            env = FlattenObs(env)
        return env

    # ---- Grasp model (state-only, 16-dim) ----
    print(f"Loading grasp model: {args.grasp_model}")
    grasp_env_factory = lambda: make_env(
        vision_mode=False, target_pos_range=TARGET_POS_RANGE,
        domain_randomize=False,
    )
    grasp_vec_env = DummyVecEnv([grasp_env_factory])
    if args.grasp_vecnorm and os.path.exists(args.grasp_vecnorm):
        grasp_vec_env = VecNormalize.load(args.grasp_vecnorm, grasp_vec_env)
        grasp_vec_env.norm_reward = False
        grasp_vec_env.training = False
    else:
        grasp_vec_env = VecNormalize(
            grasp_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        grasp_vec_env.training = False
    grasp_model = PPO.load(args.grasp_model, env=grasp_vec_env, device="auto")

    # ---- Place VecNormalize stats (for vision obs normalization) ----
    place_env_factory = lambda: make_env(
        vision_mode=True, target_pos_range=TARGET_POS_RANGE,
        domain_randomize=False,
    )
    place_vec_env = DummyVecEnv([place_env_factory])
    if args.place_vecnorm and os.path.exists(args.place_vecnorm):
        place_vec_env = VecNormalize.load(args.place_vecnorm, place_vec_env)
        place_vec_env.norm_reward = False
        place_vec_env.training = False
    else:
        place_vec_env = VecNormalize(
            place_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
            norm_obs_keys=["state"])
        place_vec_env.training = False

    # ---- Raw eval env (FlattenObs 16-dim, for grasp-phase stepping) ----
    raw_env = DummyVecEnv([lambda: make_env(
        target_pos_range=TARGET_POS_RANGE, domain_randomize=False,
    )])
    _inner_env = raw_env.envs[0].env.unwrapped
    _inner_env._release_dist_threshold = 0.05
    _inner_env._release_height_threshold = float("inf")
    place_vision_wrapper = VisionObs(_inner_env, image_size=84)

    return {
        "grasp_model": grasp_model,
        "grasp_vec_env": grasp_vec_env,
        "place_vec_env": place_vec_env,
        "raw_env": raw_env,
        "inner_env": _inner_env,
        "place_vision_wrapper": place_vision_wrapper,
    }


def _activate_place_mode(env_components) -> None:
    """Snap block to hand + switch reward to place_only (mirrors eval pattern)."""
    inner_env = env_components["inner_env"]
    raw_env = env_components["raw_env"]
    inner_env.place_mode = True
    inner_env._place_gravcomp_active = True
    inner_env.snap_block_to_hand()
    inner_env._arm_target = inner_env.data.qpos[inner_env._arm_qpos_adrs].copy()
    inner_env._gripper_target = float(
        inner_env.data.qpos[inner_env._finger_qpos_adrs].mean())
    inner_env.reward_type = "place_only"
    inner_env._place_approach_bonus_given = False
    inner_env._place_proximity_15_given = False
    inner_env._place_proximity_10_given = False
    inner_env._place_success = False
    inner_env._prev_block_target_dist = None
    inner_env._prev_block_height = None
    inner_env._use_gripper_target_check = True
    # Re-flatten raw_obs so block_target_distance reflects snapped block
    flatten_wrapper = raw_env.envs[0]
    inner_obs = inner_env._get_obs()
    new_flat = flatten_wrapper.observation(inner_obs)
    raw_obs = new_flat[np.newaxis, :].astype(np.float32)
    env_components["_last_raw_obs"] = raw_obs


def _build_place_obs(env_components, device: str) -> dict:
    """Construct a normalized vision obs dict for the place phase."""
    inner_env = env_components["inner_env"]
    place_vec_env = env_components["place_vec_env"]
    place_vision_wrapper = env_components["place_vision_wrapper"]
    vision_obs = place_vision_wrapper.observation(inner_env._get_obs())
    batched = {
        "image": vision_obs["image"][np.newaxis, ...],
        "state": vision_obs["state"][np.newaxis, ...],
    }
    obs = place_vec_env.normalize_obs(batched)
    obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))  # HWC -> CHW
    return obs


def _build_grasp_obs(env_components) -> np.ndarray:
    """Construct a normalized 16-dim grasp obs from raw_env obs."""
    raw_env = env_components["raw_env"]
    grasp_vec_env = env_components["grasp_vec_env"]
    raw_obs = env_components.get("_last_raw_obs")
    if raw_obs is None:
        raw_obs = raw_env.reset()
    raw_obs_for_grasp = raw_obs[:, :16].copy()
    block_pos = raw_obs_for_grasp[0, 8:11]
    default_target = np.array([0.5, 0.3, 0.2])
    raw_obs_for_grasp[0, 15] = np.linalg.norm(block_pos - default_target)
    return grasp_vec_env.normalize_obs(raw_obs_for_grasp)


def _compute_v59_latent_from_obs(v59_policy, obs: dict, device: str
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run V59 backbone to get (features, latent) on CPU, under no_grad.

    Returns
    -------
    (features, latent) : Tuple[Tensor (1, 524), Tensor (1, 64)]
        ``features`` is the features_extractor output (feeds
        ``mlp_extractor.policy_net`` and ``learned_policy``);
        ``latent`` is the mlp_extractor.policy_net output (feeds
        ``action_net``, ``phi``, ``value_head``).
    """
    with torch.no_grad():
        obs_t = {
            "image": torch.as_tensor(obs["image"], dtype=torch.float32,
                                     device=device),
            "state": torch.as_tensor(obs["state"], dtype=torch.float32,
                                     device=device),
        }
        features = extract_v59_features(v59_policy, obs_t).cpu()
        latent = v59_policy.mlp_extractor.forward_actor(features.to(device)).cpu()
    return features, latent


def _collect_pbrs_rollout(ensemble: "EnsemblePolicy", v59_policy,
                          env_components: dict, grasp_model, device: str,
                          v59_log_std: torch.Tensor, phi: PotentialFunction,
                          gamma: float, n_steps: int,
                          max_steps: int = MAX_STEPS) -> Optional[dict]:
    """Collect place-phase transitions for one PPO rollout.

    Uses :class:`_EnsemblePlaceAdapter` (stochastic) wrapped in
    :class:`HierarchicalPickPlacePolicy` so the grasp phase is handled by
    the standard grasp model.  Only place-phase transitions are recorded.

    Returns a dict with tensors ``features``, ``next_features`` (524-dim,
    for ``learned_policy``), ``latents``, ``next_latents`` (64-dim, for
    ``action_net``/``phi``/``value_head``), ``actions`` (a_pi samples),
    ``rewards``, ``dones``, ``phi_s``, ``phi_s_next``, or ``None`` if no
    place-phase transitions were collected.
    """
    from hierarchical_policy import HierarchicalPickPlacePolicy

    place_adapter = _EnsemblePlaceAdapter(ensemble, v59_policy, device,
                                          v59_log_std)
    place_adapter.stochastic = True
    hier_policy = HierarchicalPickPlacePolicy(grasp_model, place_adapter)

    raw_env = env_components["raw_env"]
    inner_env = env_components["inner_env"]

    features_list: list[torch.Tensor] = []
    next_features_list: list[torch.Tensor] = []
    latents_list: list[torch.Tensor] = []
    next_latents_list: list[torch.Tensor] = []
    actions_list: list[torch.Tensor] = []
    rewards_list: list[float] = []
    dones_list: list[float] = []

    steps_collected = 0
    episode_count = 0
    max_episodes = max(20, n_steps // 10)  # safety bound

    while steps_collected < n_steps and episode_count < max_episodes:
        inner_env.place_mode = False
        inner_env._place_gravcomp_active = False
        env_components["_last_raw_obs"] = raw_env.reset()
        hier_policy.reset()
        prev_info = None
        first_place_step = None

        for step in range(max_steps):
            phase = hier_policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                _activate_place_mode(env_components)

            if phase == "place":
                obs = _build_place_obs(env_components, device)
            else:
                obs = _build_grasp_obs(env_components)

            action, _ = hier_policy.predict(obs, info=prev_info,
                                            deterministic=True)

            # Capture pre-step place-phase data
            place_step_data = None
            if phase == "place":
                place_step_data = {
                    "features": place_adapter.last_features.clone(),  # (1,524)
                    "latent": place_adapter.last_latent.clone(),      # (1,64)
                    "a_pi": place_adapter.last_a_pi_sample.clone(),   # (1,8)
                }

            raw_obs, reward, done, info = raw_env.step(action)
            env_components["_last_raw_obs"] = raw_obs
            prev_info = info[0]
            i = info[0]

            if place_step_data is not None:
                # Compute next-state features + latent
                if not done[0]:
                    next_obs = _build_place_obs(env_components, device)
                    next_features, next_latent = _compute_v59_latent_from_obs(
                        v59_policy, next_obs, device)
                else:
                    next_features = torch.zeros((1, LATENT_DIM),
                                                dtype=torch.float32)
                    next_latent = torch.zeros((1, HIDDEN_DIM),
                                              dtype=torch.float32)
                features_list.append(place_step_data["features"].squeeze(0))
                next_features_list.append(next_features.squeeze(0))
                latents_list.append(place_step_data["latent"].squeeze(0))
                next_latents_list.append(next_latent.squeeze(0))
                actions_list.append(place_step_data["a_pi"].squeeze(0))
                rewards_list.append(float(reward[0]))
                dones_list.append(1.0 if done[0] else 0.0)
                steps_collected += 1
                if steps_collected >= n_steps:
                    break

            if done[0]:
                break

        episode_count += 1

    if not latents_list:
        print("  WARNING: no place-phase transitions collected in rollout.")
        return None

    features = torch.stack(features_list, dim=0)          # (T, 524)
    next_features = torch.stack(next_features_list, dim=0)
    latents = torch.stack(latents_list, dim=0)            # (T, 64)
    next_latents = torch.stack(next_latents_list, dim=0)
    actions = torch.stack(actions_list, dim=0)            # (T, 8)
    rewards = torch.tensor(rewards_list, dtype=torch.float32)
    dones = torch.tensor(dones_list, dtype=torch.float32)

    # Compute Phi(s) and Phi(s') under no_grad (phi is frozen; takes 64-dim latent)
    with torch.no_grad():
        phi_s = phi(latents.to(device)).cpu()
        phi_s_next = phi(next_latents.to(device)).cpu()

    print(f"  Rollout: {steps_collected} place-phase transitions "
          f"over {episode_count} episodes.")
    return {
        "features": features,
        "next_features": next_features,
        "latents": latents,
        "next_latents": next_latents,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "phi_s": phi_s,
        "phi_s_next": phi_s_next,
    }


def _run_pbrs_eval(ensemble: "EnsemblePolicy", v59_policy,
                   env_components: dict, grasp_model, device: str,
                   v59_log_std: torch.Tensor, n_episodes: int,
                   max_steps: int = MAX_STEPS) -> Tuple[float, float, int]:
    """Run deterministic eval episodes and return metrics.

    Returns
    -------
    (place_rate, mean_final_dist_cm, n_placed)
    """
    from hierarchical_policy import HierarchicalPickPlacePolicy

    place_adapter = _EnsemblePlaceAdapter(ensemble, v59_policy, device,
                                          v59_log_std)
    place_adapter.stochastic = False  # deterministic for eval
    hier_policy = HierarchicalPickPlacePolicy(grasp_model, place_adapter)

    raw_env = env_components["raw_env"]
    inner_env = env_components["inner_env"]

    n_placed = 0
    final_dists: list[float] = []

    for ep in range(n_episodes):
        inner_env.place_mode = False
        inner_env._place_gravcomp_active = False
        env_components["_last_raw_obs"] = raw_env.reset()
        hier_policy.reset()
        prev_info = None
        first_place_step = None
        block_target_dist = float("inf")

        for step in range(max_steps):
            phase = hier_policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                _activate_place_mode(env_components)

            if phase == "place":
                obs = _build_place_obs(env_components, device)
            else:
                obs = _build_grasp_obs(env_components)

            action, _ = hier_policy.predict(obs, info=prev_info,
                                            deterministic=True)
            raw_obs, reward, done, info = raw_env.step(action)
            env_components["_last_raw_obs"] = raw_obs
            prev_info = info[0]
            i = info[0]
            block_target_dist = float(
                i.get("block_target_distance", block_target_dist))
            if done[0]:
                break

        placed = block_target_dist < PLACE_THRESHOLD
        if placed:
            n_placed += 1
        final_dists.append(block_target_dist)

    place_rate = n_placed / n_episodes if n_episodes > 0 else 0.0
    mean_dist_cm = float(np.mean(final_dists) * 100) if final_dists else 999.0
    return place_rate, mean_dist_cm, n_placed


def _ppo_update(ensemble: "EnsemblePolicy", value_head: ValueHead,
                optimizer: torch.optim.Optimizer, rollout: dict,
                v59_log_std: torch.Tensor, args, gamma: float,
                gae_lambda: float, device: str,
                bc_head: Optional[nn.Module] = None,
                alpha: float = 0.0,
                v59_policy_for_reward=None) -> dict:
    """Perform one PPO update of a_pi + value_head on a rollout.

    Implements the clipped PPO objective with shaped rewards (PBRS), GAE
    advantages, KL early-stopping, and gradient clipping.  Only
    ``ensemble.learned_policy`` (a_pi) and ``value_head`` are updated — V59
    is frozen.

    CSIL++ V2 extension: when ``bc_head`` is provided and ``alpha > 0``, the
    shaped reward becomes::

        r_shaped = r_env + alpha * r_coherent(s,a) + gamma * Phi(s') - Phi(s)

    where ``r_coherent = log_pi_V59(a|s) - log_p_BC(a|s)``.  ``alpha`` is
    annealed by the caller (linear decay from ``alpha0`` to 0 over training).
    When ``bc_head is None`` or ``alpha == 0``, this reduces to V1 PBRS.
    """
    latents = rollout["latents"].to(device)            # (T, 64) — for value_head/phi/action_net
    next_latents = rollout["next_latents"].to(device)
    features = rollout["features"].to(device)         # (T, 524) — for learned_policy
    actions = rollout["actions"].to(device)
    rewards = rollout["rewards"].to(device)
    dones = rollout["dones"].to(device)
    phi_s = rollout["phi_s"].to(device)
    phi_s_next = rollout["phi_s_next"].to(device)
    v59_log_std = v59_log_std.to(device)

    # 1. Shaped rewards (PBRS):
    #    V2: r_shaped = r + alpha * r_coherent(s,a) + gamma * Phi(s') - Phi(s)
    #    V1: r_shaped = r + gamma * Phi(s') - Phi(s)   (bc_head=None or alpha=0)
    coherent_reward = torch.zeros(features.shape[0],
                                  dtype=torch.float32, device=device)
    coherent_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if bc_head is not None and alpha > 0.0 and v59_policy_for_reward is not None:
        # Compute on CPU (frozen models), then move to device.
        coherent_reward_cpu, coherent_stats = _compute_coherent_reward_from_features(
            v59_policy_for_reward, bc_head, rollout["features"],
            rollout["actions"], v59_log_std, device,
        )
        coherent_reward = coherent_reward_cpu.to(device)

    shaped_rewards = (rewards
                      + alpha * coherent_reward
                      + gamma * phi_s_next * (1.0 - dones) - phi_s)

    # 2. Old values (no grad — these are the pre-update targets)
    value_head.eval()
    with torch.no_grad():
        old_values = value_head(latents).squeeze(-1)
        next_values = value_head(next_latents).squeeze(-1)
    value_head.train()

    # 3. GAE advantages + returns
    advantages, returns = _compute_gae(
        shaped_rewards, old_values, next_values, dones, gamma, gae_lambda)
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. Old log probs (no grad) — learned_policy takes 524-dim features
    with torch.no_grad():
        old_log_probs = _compute_a_pi_log_prob(
            ensemble.learned_policy, features, actions, v59_log_std)

    # 5. PPO epochs (mini-batch updates)
    T = latents.shape[0]
    batch_size = min(args.batch_size, T)
    clip_range = args.clip_range
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.3
    indices = torch.arange(T, device=device)

    policy_losses: list[float] = []
    value_losses: list[float] = []
    kls: list[float] = []
    clip_fractions: list[float] = []
    entropy_losses: list[float] = []
    early_stopped = False

    for epoch in range(args.n_epochs):
        perm = indices[torch.randperm(T, device=device)]
        for start in range(0, T, batch_size):
            idx = perm[start:start + batch_size]
            feat_b = features[idx]   # 524-dim -> learned_policy
            lat_b = latents[idx]     # 64-dim  -> value_head
            act_b = actions[idx]
            ret_b = returns[idx]
            adv_b = advantages[idx]
            old_lp_b = old_log_probs[idx]

            # New log probs (with grad through learned_policy — uses features)
            new_log_probs = _compute_a_pi_log_prob(
                ensemble.learned_policy, feat_b, act_b, v59_log_std)
            values_pred = value_head(lat_b).squeeze(-1)

            ratio = torch.exp(new_log_probs - old_lp_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, ret_b)
            # Entropy: for a fixed-std Gaussian, entropy is constant; set to 0
            entropy_loss = torch.tensor(0.0, device=device)
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ensemble.learned_policy.parameters())
                + list(value_head.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                kl = (old_lp_b - new_log_probs).mean().item()
                clip_frac = (torch.abs(ratio - 1.0) > clip_range
                             ).float().mean().item()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            kls.append(kl)
            clip_fractions.append(clip_frac)

        # KL early stop (use mean KL of this epoch's batches)
        epoch_mean_kl = float(np.mean(kls[-max(1, T // batch_size):])) if kls else 0.0
        if epoch_mean_kl > args.max_kl:
            print(f"    KL early stop at epoch {epoch+1}/{args.n_epochs}: "
                  f"mean_kl={epoch_mean_kl:.6f} > max_kl={args.max_kl}")
            early_stopped = True
            break

    return {
        "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
        "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "mean_shaped_reward": float(shaped_rewards.mean().item()),
        "mean_env_reward": float(rewards.mean().item()),
        "alpha": float(alpha),
        "coherent_reward_mean": float(coherent_stats["mean"]),
        "coherent_reward_std": float(coherent_stats["std"]),
        "coherent_reward_min": float(coherent_stats["min"]),
        "coherent_reward_max": float(coherent_stats["max"]),
        "early_stopped": early_stopped,
    }


def _verify_policy_invariance(ensemble: "EnsemblePolicy", v59_policy,
                              sample_features: torch.Tensor,
                              sample_latents: torch.Tensor,
                              v59_log_std: torch.Tensor, device: str
                              ) -> dict:
    """Verify PBRS didn't shift a_pi dramatically from V59 (SubTask 4.3).

    Computes KL(a_pi || a_V59) on sample states.  Since both distributions
    share the same (frozen) log_std, KL reduces to:

        KL = 0.5 * sum_i [(mu_pi_i - mu_V59_i) / sigma_i]^2

    Note: ``a_V59`` and ``a_pi`` are computed from DIFFERENT tensors because
    their heads take different inputs after the Task 4 architecture fix:

      - ``a_V59 = action_net(latents)``      — 64-dim policy latent
      - ``a_pi   = learned_policy(features)`` — 524-dim features_extractor output

    Returns a dict with ``kl`` and ``max_weight_delta`` (max |a_pi - V59|
    parameter difference across the learned head).
    """
    with torch.no_grad():
        features = sample_features.to(device)   # (N, 524) -> learned_policy
        latents = sample_latents.to(device)     # (N, 64)  -> action_net
        a_v59_mean = v59_policy.action_net(latents)
        a_pi_mean = ensemble.learned_policy(features)
        log_std = v59_log_std.to(device)
        if log_std.dim() == 1:
            std = log_std.exp().unsqueeze(0).expand_as(a_v59_mean)
        else:
            std = log_std.exp().expand_as(a_v59_mean)
        kl = 0.5 * (((a_pi_mean - a_v59_mean) / std) ** 2).sum(dim=-1).mean()

    # Compare weights directly
    max_delta = 0.0
    try:
        v59_sd = v59_policy.state_dict()
        pi_sd = ensemble.learned_policy.state_dict()
        mapping = {
            "mlp_extractor.policy_net.0.weight": "policy_net.0.weight",
            "mlp_extractor.policy_net.0.bias": "policy_net.0.bias",
            "mlp_extractor.policy_net.2.weight": "policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias": "policy_net.2.bias",
            "action_net.weight": "action_net.weight",
            "action_net.bias": "action_net.bias",
        }
        for v59_key, pi_key in mapping.items():
            if v59_key in v59_sd and pi_key in pi_sd:
                delta = (v59_sd[v59_key].float() - pi_sd[pi_key].float()
                         ).abs().max().item()
                max_delta = max(max_delta, delta)
    except Exception as e:
        print(f"  (weight comparison skipped: {e})")

    return {
        "kl": float(kl.item()),
        "max_weight_delta": float(max_delta),
        "policy_invariant": float(kl.item()) < 0.1,
    }


def _save_ensemble_checkpoint(ensemble: "EnsemblePolicy",
                              value_head: ValueHead, path: Path,
                              iteration: int, place_rate: float,
                              rolled_back: bool, extra: Optional[dict] = None
                              ) -> None:
    """Save ensemble + value head state_dict to ``path``."""
    payload = {
        "learned_policy_state_dict": ensemble.learned_policy.state_dict(),
        "value_head_state_dict": value_head.state_dict(),
        "rolled_back": rolled_back,
        "iteration": iteration,
        "place_rate": place_rate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def cmd_train_ensemble(args) -> None:
    """PBRS fine-tune a_pi using Phi as the potential function (Task 4).

    Implements a custom PPO loop with shaped reward
    ``r_shaped = r_env + gamma * Phi(s') - Phi(s)``.  V59 backbone is frozen;
    only ``ensemble.learned_policy`` (a_pi) and a separate ``ValueHead`` are
    trained.  Safety rollback triggers if eval place_rate < safety_threshold
    or KL divergence from V59 exceeds 0.2.
    """
    print("=" * 60)
    print("CSIL++ Train Ensemble (PBRS) — Task 4")
    print("=" * 60)
    print(f"  max_iterations={args.max_iterations}  "
          f"n_steps_per_rollout={args.n_steps_per_rollout}  "
          f"lr={args.learning_rate}  clip={args.clip_range}  "
          f"max_kl={args.max_kl}")
    print(f"  eval_every={args.eval_every}  eval_episodes={args.eval_episodes}  "
          f"safety_threshold={args.safety_threshold}")
    print(f"  gamma={args.gamma}  gae_lambda={args.gae_lambda}  "
          f"batch_size={args.batch_size}  n_epochs={args.n_epochs}")
    print("  Image augmentation: DISABLED (eval mode)")
    print("  BN running stats: FROZEN (features_extractor.eval())")
    print("  V59 backbone: FROZEN — only a_pi MLP head + value head trained")

    CSIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_model_dir = CSIL_OUTPUT_DIR / "best_model"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(POTENTIAL_FN_PATH):
        print(f"\nERROR: Phi not found at {POTENTIAL_FN_PATH}.")
        print("  Run `python train_csil_plus_plus.py train-reward` first.")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n=== Loading V59 from {args.place_model} ===")
    model, _ = _load_v49_policy_safe(args, device)
    freeze_backbone(model)
    model.policy.features_extractor.eval()
    # Freeze BN running stats on ALL modules in the policy (not just features_extractor)
    for m in model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

    print(f"\n=== Loading Phi from {POTENTIAL_FN_PATH} ===")
    phi_state_dict = torch.load(str(POTENTIAL_FN_PATH), map_location="cpu")
    # Infer in_dim from the first layer weight (Phi was trained on the 64-dim
    # policy latent from _precompute_latents, NOT the 524-dim features).
    phi_in_dim = phi_state_dict["net.0.weight"].shape[1]
    print(f"  Phi in_dim inferred from checkpoint: {phi_in_dim}")
    phi = PotentialFunction(in_dim=phi_in_dim)
    phi.load_state_dict(phi_state_dict)
    phi.to(device).eval()
    for p in phi.parameters():
        p.requires_grad = False
    print("  Phi loaded (frozen, eval mode).")

    print("\n=== Building EnsemblePolicy ===")
    ensemble = EnsemblePolicy(v59_policy=model.policy)
    ensemble.to(device) if hasattr(ensemble, "to") else None
    print("  EnsemblePolicy ready (learned head initialized from V59).")

    # Value head (trained alongside a_pi). Uses the SAME input dim as Phi —
    # the 64-dim policy latent (output of mlp_extractor.policy_net), since
    # value_head consumes the same `latents` tensor that Phi does in _ppo_update.
    value_head = ValueHead(latent_dim=phi_in_dim).to(device)
    print(f"  ValueHead created (latent_dim={phi_in_dim}, "
          f"{sum(p.numel() for p in value_head.parameters())} params).")

    # Frozen V59 log_std
    v59_log_std_tensor = v59_log_std(model.policy).squeeze().clone()
    if v59_log_std_tensor.dim() == 0:
        v59_log_std_tensor = v59_log_std_tensor.unsqueeze(0)
    print(f"  V59 log_std shape: {tuple(v59_log_std_tensor.shape)}  "
          f"range=[{v59_log_std_tensor.min().item():.3f}, "
          f"{v59_log_std_tensor.max().item():.3f}]")

    # Optimizer: only a_pi + value_head (V59 is frozen)
    trainable_params = list(ensemble.learned_policy.parameters()) + \
                       list(value_head.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Optimizer: Adam lr={args.learning_rate}  "
          f"trainable params={n_trainable}")

    # CSIL++ V2: load BC head prior for the coherent reward term.
    # When prior=='bc_head', the shaped reward becomes
    #   r_shaped = r_env + alpha * (logpi_V59 - logp_BC) + gamma*Phi(s') - Phi(s)
    # with alpha annealed linearly from alpha0 to 0 over max_iterations.
    bc_head = None
    if args.prior == "bc_head":
        if not args.bc_head_path or not os.path.exists(args.bc_head_path):
            print(f"\nERROR: --prior bc_head requires --bc_head_path "
                  f"(looked for: {args.bc_head_path})")
            sys.exit(1)
        print(f"\n=== Loading BC head prior from {args.bc_head_path} ===")
        bc_head, bc_feat_dim = _load_bc_head(args.bc_head_path, device)
        print(f"  BC head feat_dim={bc_feat_dim}, "
              f"params={sum(p.numel() for p in bc_head.parameters())}, "
              f"frozen (eval mode).")
        print(f"  Coherent reward: alpha0={args.alpha0}, "
              f"anneal linearly to 0 over {args.max_iterations} iters.")
    else:
        print(f"\n  Prior: {args.prior} (no BC head — V1 PBRS shaping only).")

    # Sanity check: with untrained a_pi, ensemble action == V59 action.
    print("\n=== Sanity check: a_ens == a_V59 when a_pi untrained ===")
    if os.path.exists(args.d_csil):
        data = _load_d_csil(args.d_csil)
        images = data["images"][:8]
        states_raw = data["states"][:8]
        states = normalize_states(states_raw, args.place_vecnorm)
        obs = {"image": images.permute(0, 3, 1, 2).contiguous().to(device),
               "state": states.to(device)}
        with torch.no_grad():
            a_v59 = v59_action_mean(model.policy, obs)
            a_ens = ensemble.get_action(obs)
        delta = (a_v59 - a_ens).abs().max().item()
        print(f"  max|a_V59 - a_ens| = {delta:.6f} (should be ~0)")
    else:
        print(f"  (skipped — D_csil not found at {args.d_csil})")

    # Build training envs (grasp model + raw env + VecNormalize stats)
    print("\n=== Building training envs ===")
    env_components = _build_train_envs(args)
    grasp_model = env_components["grasp_model"]
    print("  Training envs ready.")

    # ---- Training loop ----
    print("\n=== Starting PBRS training loop ===")
    best_place_rate = 0.0
    training_log: list[dict] = []
    np.random.seed(args.seed)
    try:
        raw_env = env_components["raw_env"]
        raw_env.seed(args.seed)
    except Exception:
        pass

    rollback_triggered = False
    t_train_start = time.time()

    for iteration in range(args.max_iterations):
        iter_t0 = time.time()
        # CSIL++ V2: linear alpha anneal from alpha0 -> 0 over max_iterations.
        # alpha(iter) = alpha0 * (1 - iter / max_iterations).
        # At iter=0: alpha=alpha0; at iter=max_iterations-1: alpha ≈ alpha0/max.
        if bc_head is not None:
            alpha = args.alpha0 * (1.0 - iteration / max(1, args.max_iterations - 1))
        else:
            alpha = 0.0
        print(f"\n--- Iteration {iteration+1}/{args.max_iterations} ---"
              f"  alpha={alpha:.4f}")
        try:
            # 1. Collect rollout (stochastic, place-phase only)
            print("  [1/4] Collecting rollout...")
            rollout = _collect_pbrs_rollout(
                ensemble, model.policy, env_components, grasp_model,
                device, v59_log_std_tensor, phi, args.gamma,
                args.n_steps_per_rollout,
            )
            if rollout is None:
                print("  No rollout data — skipping update.")
                continue

            # 2-4. PPO update with shaped rewards + GAE + KL early stop
            print("  [2/4] PPO update...")
            update_stats = _ppo_update(
                ensemble, value_head, optimizer, rollout,
                v59_log_std_tensor, args, args.gamma, args.gae_lambda, device,
                bc_head=bc_head, alpha=alpha,
                v59_policy_for_reward=model.policy,
            )
            print(f"    policy_loss={update_stats['policy_loss']:.6f}  "
                  f"value_loss={update_stats['value_loss']:.6f}  "
                  f"kl={update_stats['kl']:.6f}  "
                  f"clip_frac={update_stats['clip_fraction']:.4f}  "
                  f"mean_shaped_r={update_stats['mean_shaped_reward']:.4f}  "
                  f"early_stopped={update_stats['early_stopped']}")
            if bc_head is not None and alpha > 0.0:
                print(f"    coherent_reward: "
                      f"mean={update_stats['coherent_reward_mean']:+.4f}  "
                      f"std={update_stats['coherent_reward_std']:.4f}  "
                      f"range=[{update_stats['coherent_reward_min']:+.4f}, "
                      f"{update_stats['coherent_reward_max']:+.4f}]  "
                      f"alpha*mean={alpha * update_stats['coherent_reward_mean']:+.5f}")

            # KL-based safety: if KL from V59 is catastrophic, rollback
            if update_stats["kl"] > 0.2:
                print(f"  CATASTROPHIC KL drift ({update_stats['kl']:.4f} > 0.2) "
                      f"— triggering safety rollback.")
                ensemble.safety_rollback()
                rollback_triggered = True
                _save_ensemble_checkpoint(
                    ensemble, value_head, ENSEMBLE_POLICY_PATH,
                    iteration, 0.0, True,
                    extra={"reason": "catastrophic_kl",
                           "kl": update_stats["kl"]})
                training_log.append({
                    "iteration": iteration,
                    "alpha": alpha,
                    "policy_loss": update_stats["policy_loss"],
                    "value_loss": update_stats["value_loss"],
                    "kl": update_stats["kl"],
                    "clip_fraction": update_stats["clip_fraction"],
                    "mean_shaped_reward": update_stats["mean_shaped_reward"],
                    "mean_env_reward": update_stats["mean_env_reward"],
                    "coherent_reward_mean": update_stats["coherent_reward_mean"],
                    "coherent_reward_std": update_stats["coherent_reward_std"],
                    "place_rate": None,
                    "rollback": True,
                    "rollback_reason": "catastrophic_kl",
                })
                break

            # 5. Periodic eval + safety rollback check
            place_rate = None
            mean_dist_cm = None
            do_eval = ((iteration + 1) % args.eval_every == 0
                       or iteration == args.max_iterations - 1)
            if do_eval:
                print(f"  [3/4] Evaluating ({args.eval_episodes} episodes)...")
                place_rate, mean_dist_cm, n_placed = _run_pbrs_eval(
                    ensemble, model.policy, env_components, grasp_model,
                    device, v59_log_std_tensor, args.eval_episodes,
                )
                print(f"    place_rate={place_rate:.2%}  "
                      f"mean_final_dist={mean_dist_cm:.1f}cm  "
                      f"placed={n_placed}/{args.eval_episodes}")

                if place_rate < args.safety_threshold:
                    print(f"  SAFETY ROLLBACK at iteration {iteration+1}: "
                          f"place_rate={place_rate:.2%} < "
                          f"threshold={args.safety_threshold:.2%}")
                    ensemble.safety_rollback()
                    rollback_triggered = True
                    _save_ensemble_checkpoint(
                        ensemble, value_head, ENSEMBLE_POLICY_PATH,
                        iteration, place_rate, True,
                        extra={"reason": "low_place_rate",
                               "place_rate": place_rate})
                    training_log.append({
                        "iteration": iteration,
                        "alpha": alpha,
                        "policy_loss": update_stats["policy_loss"],
                        "value_loss": update_stats["value_loss"],
                        "kl": update_stats["kl"],
                        "clip_fraction": update_stats["clip_fraction"],
                        "mean_shaped_reward": update_stats["mean_shaped_reward"],
                        "mean_env_reward": update_stats["mean_env_reward"],
                        "coherent_reward_mean": update_stats["coherent_reward_mean"],
                        "coherent_reward_std": update_stats["coherent_reward_std"],
                        "place_rate": place_rate,
                        "mean_final_dist_cm": mean_dist_cm,
                        "rollback": True,
                        "rollback_reason": "low_place_rate",
                    })
                    break

                if place_rate > best_place_rate:
                    best_place_rate = place_rate
                    _save_ensemble_checkpoint(
                        ensemble, value_head,
                        best_model_dir / "ensemble_policy.pt",
                        iteration, place_rate, False,
                        extra={"best": True})
                    print(f"    New best place_rate={best_place_rate:.2%} "
                          f"— saved to {best_model_dir / 'ensemble_policy.pt'}")
            else:
                print("  [3/4] Eval skipped (not eval_every interval).")

            # 6. Log metrics
            entry = {
                "iteration": iteration,
                "alpha": alpha,
                "policy_loss": update_stats["policy_loss"],
                "value_loss": update_stats["value_loss"],
                "kl": update_stats["kl"],
                "clip_fraction": update_stats["clip_fraction"],
                "mean_shaped_reward": update_stats["mean_shaped_reward"],
                "mean_env_reward": update_stats["mean_env_reward"],
                "coherent_reward_mean": update_stats["coherent_reward_mean"],
                "coherent_reward_std": update_stats["coherent_reward_std"],
                "place_rate": place_rate,
                "mean_final_dist_cm": mean_dist_cm,
                "best_place_rate": best_place_rate,
                "rollback": False,
                "elapsed_s": time.time() - iter_t0,
            }
            training_log.append(entry)

            # Save current ensemble after each iteration (robustness)
            _save_ensemble_checkpoint(
                ensemble, value_head, ENSEMBLE_POLICY_PATH,
                iteration, place_rate if place_rate is not None else -1.0,
                ensemble._rolled_back)

            # Write training log incrementally
            with open(TRAINING_LOG_PATH, "w") as f:
                json.dump({
                    "experiment_id": args.experiment_id,
                    "method": "CSIL++_ensemble_PBRS",
                    "config": {
                        "max_iterations": args.max_iterations,
                        "n_steps_per_rollout": args.n_steps_per_rollout,
                        "learning_rate": args.learning_rate,
                        "clip_range": args.clip_range,
                        "max_kl": args.max_kl,
                        "gamma": args.gamma,
                        "gae_lambda": args.gae_lambda,
                        "batch_size": args.batch_size,
                        "n_epochs": args.n_epochs,
                        "eval_every": args.eval_every,
                        "eval_episodes": args.eval_episodes,
                        "safety_threshold": args.safety_threshold,
                        "prior": args.prior,
                        "bc_head_path": args.bc_head_path,
                        "alpha0": args.alpha0,
                    },
                    "iterations": training_log,
                }, f, indent=2)

        except Exception as e:
            print(f"\n  ERROR at iteration {iteration+1}: {e}")
            import traceback
            traceback.print_exc()
            print("  Saving current state and exiting gracefully...")
            try:
                _save_ensemble_checkpoint(
                    ensemble, value_head, ENSEMBLE_POLICY_PATH,
                    iteration, -1.0, ensemble._rolled_back,
                    extra={"error": str(e), "emergency_save": True})
                with open(TRAINING_LOG_PATH, "w") as f:
                    json.dump({
                        "experiment_id": args.experiment_id,
                        "method": "CSIL++_ensemble_PBRS",
                        "error": str(e),
                        "iterations": training_log,
                    }, f, indent=2)
            except Exception as save_err:
                print(f"  FATAL: could not save checkpoint: {save_err}")
            break

    # ---- Post-training: policy invariance verification (SubTask 4.3) ----
    print("\n=== Policy Invariance Verification (SubTask 4.3) ===")
    invariance_report = {"kl": None, "max_weight_delta": None,
                         "policy_invariant": None}
    try:
        # Use D_csil features+latents or rollout features+latents as test states
        test_features = None
        test_latents = None
        if os.path.exists(args.d_csil):
            data = _load_d_csil(args.d_csil)
            images = data["images"][:64]
            states_raw = data["states"][:64]
            states = normalize_states(states_raw, args.place_vecnorm)
            obs = {"image": images.permute(0, 3, 1, 2).contiguous(),
                   "state": states}
            with torch.no_grad():
                obs_t = {
                    "image": torch.as_tensor(obs["image"], dtype=torch.float32,
                                             device=device),
                    "state": torch.as_tensor(obs["state"], dtype=torch.float32,
                                             device=device),
                }
                test_features = extract_v59_features(model.policy, obs_t).cpu()
                test_latents = extract_v59_latent(model.policy, obs_t).cpu()
        elif training_log and "latents" in locals():
            test_features = rollout["features"][:64]
            test_latents = rollout["latents"][:64]

        if test_features is not None and test_latents is not None:
            inv = _verify_policy_invariance(
                ensemble, model.policy, test_features, test_latents,
                v59_log_std_tensor, device)
            invariance_report = inv
            print(f"  KL(a_pi || a_V59) = {inv['kl']:.6f}")
            print(f"  Max |weight_delta| = {inv['max_weight_delta']:.6f}")
            if inv["kl"] > 0.1:
                print(f"  WARNING: KL={inv['kl']:.4f} > 0.1 — policy invariance "
                      f"may be violated.")
            else:
                print("  Policy invariance: OK (KL < 0.1)")
        else:
            print("  (skipped — no test states available)")
    except Exception as e:
        print(f"  (verification failed: {e})")

    # ---- Final save ----
    if not rollback_triggered:
        _save_ensemble_checkpoint(
            ensemble, value_head, ENSEMBLE_POLICY_PATH,
            len(training_log), best_place_rate, ensemble._rolled_back,
            extra={"invariance": invariance_report})
    print(f"\nFinal ensemble saved to {ENSEMBLE_POLICY_PATH}")
    print(f"Best place_rate: {best_place_rate:.2%}")
    print(f"Rollback triggered: {rollback_triggered}")
    print(f"Training log: {TRAINING_LOG_PATH}")

    # Write final training log
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump({
            "experiment_id": args.experiment_id,
            "method": "CSIL++_ensemble_PBRS",
            "config": {
                "max_iterations": args.max_iterations,
                "n_steps_per_rollout": args.n_steps_per_rollout,
                "learning_rate": args.learning_rate,
                "clip_range": args.clip_range,
                "max_kl": args.max_kl,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "safety_threshold": args.safety_threshold,
                "prior": args.prior,
                "bc_head_path": args.bc_head_path,
                "alpha0": args.alpha0,
            },
            "best_place_rate": best_place_rate,
            "rollback_triggered": rollback_triggered,
            "invariance_report": invariance_report,
            "total_elapsed_s": time.time() - t_train_start,
            "iterations": training_log,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)

    # ---- Cleanup ----
    try:
        env_components["raw_env"].close()
    except Exception:
        pass

    # ---- Metadata + version tree ----
    try:
        from auto_iter.metadata import record_experiment
        from auto_iter.version_tree import VersionTree, make_node
        record_experiment(
            experiment_id=args.experiment_id,
            optimization_method="CSIL++_ensemble",
            parent_experiment_id="V59",
            decision_reason=(
                "CSIL++ Task 4: PBRS fine-tune a_pi using Phi as potential. "
                "Custom PPO loop with shaped rewards, GAE, KL early stop, "
                "and safety rollback."
            ),
            random_seed=args.seed,
            training_config={
                "max_iterations": args.max_iterations,
                "n_steps_per_rollout": args.n_steps_per_rollout,
                "learning_rate": args.learning_rate,
                "clip_range": args.clip_range,
                "max_kl": args.max_kl,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "eval_every": args.eval_every,
                "eval_episodes": args.eval_episodes,
                "safety_threshold": args.safety_threshold,
                "prior": args.prior,
                "bc_head_path": args.bc_head_path,
                "alpha0": args.alpha0,
            },
            eval_config={
                "best_place_rate": best_place_rate,
                "rollback_triggered": rollback_triggered,
                "invariance_kl": invariance_report.get("kl"),
            },
            train_cmd=" ".join(sys.argv),
            save_path=str(CSIL_OUTPUT_DIR),
            cwd=str(WORKSPACE),
        )
        tree = VersionTree()
        node = make_node(
            experiment_id=args.experiment_id,
            parent_id="V59",
            optimization_method="CSIL++_ensemble",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            decision_reason="CSIL++ Task 4: PBRS fine-tune a_pi.",
            config={
                "lr": args.learning_rate,
                "clip": args.clip_range,
                "max_kl": args.max_kl,
                "best_place_rate": best_place_rate,
                "rollback": rollback_triggered,
            },
        )
        node.status = "completed"
        node.verdict = "pending"
        tree.add_node(node)
        print(f"Added to version tree: {args.experiment_id} (parent=V59)")
    except Exception as e:
        print(f"(metadata/version_tree skipped: {e})")

    print("\n" + "=" * 60)
    print("CSIL++ Train Ensemble Complete")
    print("=" * 60)
    print(f"  Iterations completed: {len(training_log)}")
    print(f"  Best place_rate:      {best_place_rate:.2%}")
    print(f"  Rollback triggered:   {rollback_triggered}")
    print(f"  Invariance KL:        {invariance_report.get('kl', 'N/A')}")
    print(f"  Elapsed:              {time.time() - t_train_start:.0f}s")


def cmd_eval_ensemble(args) -> None:
    """Load a saved EnsemblePolicy and run N-episode eval (SubTask 5.4).

    Loads the ensemble checkpoint from ``--ensemble_path`` (default:
    ``outputs/csil_plus_plus/best_model/ensemble_policy.pt``), rebuilds the
    EnsemblePolicy around a frozen V59, and runs ``--n_episodes`` deterministic
    eval episodes via :func:`_run_pbrs_eval`.
    """
    print("=" * 60)
    print("CSIL++ Ensemble Eval — SubTask 5.4")
    print("=" * 60)
    print(f"  ensemble_path: {args.ensemble_path}")
    print(f"  n_episodes:    {args.n_episodes}")

    if not os.path.exists(args.ensemble_path):
        print(f"\nERROR: ensemble checkpoint not found at {args.ensemble_path}")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    # ---- Load V59 (frozen backbone) ----
    print(f"\n=== Loading V59 from {args.place_model} ===")
    model, _ = _load_v59_policy(args, device)
    freeze_backbone(model)
    model.policy.features_extractor.eval()
    for m in model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

    # ---- Load Phi (needed for the potential, but not for eval) ----
    if os.path.exists(POTENTIAL_FN_PATH):
        phi_state_dict = torch.load(str(POTENTIAL_FN_PATH), map_location="cpu")
        phi_in_dim = phi_state_dict["net.0.weight"].shape[1]
        phi = PotentialFunction(in_dim=phi_in_dim)
        phi.load_state_dict(phi_state_dict)
        phi.to(device).eval()
        for p in phi.parameters():
            p.requires_grad = False
        print(f"  Phi loaded (in_dim={phi_in_dim}, frozen).")
    else:
        phi = None
        print("  Phi not found — eval will run without potential (OK for eval).")

    # ---- Build EnsemblePolicy and load checkpoint ----
    print("\n=== Building EnsemblePolicy ===")
    ensemble = EnsemblePolicy(v59_policy=model.policy)
    ensemble.to(device) if hasattr(ensemble, "to") else None

    checkpoint = torch.load(args.ensemble_path, map_location="cpu")
    ensemble.learned_policy.load_state_dict(checkpoint["learned_policy_state_dict"])
    rolled_back = checkpoint.get("rolled_back", False)
    ckpt_place_rate = checkpoint.get("place_rate", "N/A")
    ckpt_iter = checkpoint.get("iteration", "N/A")
    print(f"  Checkpoint loaded: iteration={ckpt_iter}  "
          f"place_rate={ckpt_place_rate}  rolled_back={rolled_back}")
    if rolled_back:
        print("  NOTE: ensemble was rolled back to V59-only during training.")
        ensemble.safety_rollback()

    # V59 log_std (frozen)
    v59_log_std_tensor = v59_log_std(model.policy).squeeze().clone()
    if v59_log_std_tensor.dim() == 0:
        v59_log_std_tensor = v59_log_std_tensor.unsqueeze(0)

    # ---- Build eval envs ----
    print("\n=== Building eval envs ===")
    env_components = _build_train_envs(args)
    grasp_model = env_components["grasp_model"]
    print("  Eval envs ready.")

    # ---- Run eval ----
    print(f"\n=== Running {args.n_episodes}-episode eval ===")
    np.random.seed(args.seed)
    try:
        raw_env = env_components["raw_env"]
        raw_env.seed(args.seed)
    except Exception:
        pass

    t0 = time.time()
    place_rate, mean_dist_cm, n_placed = _run_pbrs_eval(
        ensemble, model.policy, env_components, grasp_model,
        device, v59_log_std_tensor, args.n_episodes,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"Eval Complete: CSIL_PLUS_PLUS_V1 Ensemble")
    print(f"{'=' * 60}")
    print(f"  Episodes:        {args.n_episodes}")
    print(f"  Placed:          {n_placed}/{args.n_episodes}")
    print(f"  Place rate:      {place_rate:.2%}")
    print(f"  Mean final dist: {mean_dist_cm:.1f} cm")
    print(f"  V59 baseline:    56% (28/50)")
    print(f"  Rolled back:     {rolled_back}")
    print(f"  Elapsed:         {elapsed:.0f}s")

    # Save eval report
    eval_report = {
        "experiment_id": "CSIL_PLUS_PLUS_V1",
        "n_episodes": args.n_episodes,
        "n_placed": n_placed,
        "place_rate": place_rate,
        "mean_final_dist_cm": mean_dist_cm,
        "v59_baseline_place_rate": 0.56,
        "rolled_back": rolled_back,
        "checkpoint_iteration": ckpt_iter,
        "checkpoint_place_rate": ckpt_place_rate,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    eval_report_path = CSIL_OUTPUT_DIR / "eval_report_50ep.json"
    with open(eval_report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\n  Report saved to {eval_report_path}")

    try:
        env_components["raw_env"].close()
    except Exception:
        pass


def _load_v49_policy_safe(args, device: str):
    """Wrapper around _load_v59_policy for the ensemble subcommand.

    (Named with a typo-safe alias to avoid clashing with the train-reward
    loader; both call into _load_v59_policy.)
    """
    return _load_v59_policy(args, device)


def _load_bc_head(bc_head_path: str, device: str = "cpu"):
    """Load a ShallowBCHead from a checkpoint saved by train_shallow_bc.py.

    Returns (bc_head, feat_dim). The BC head is moved to ``device`` and set
    to eval mode.
    """
    from train_shallow_bc import ShallowBCHead, FEAT_DIM
    ckpt = torch.load(bc_head_path, map_location="cpu", weights_only=False)
    feat_dim = ckpt.get("feat_dim", FEAT_DIM)
    init_log_sigma = ckpt.get("init_log_sigma", 0.0)
    freeze_log_sigma = ckpt.get("freeze_log_sigma", True)
    bc_head = ShallowBCHead(
        feat_dim=feat_dim,
        init_log_sigma=init_log_sigma,
        freeze_log_sigma=freeze_log_sigma,
    )
    bc_head.load_state_dict(ckpt["state_dict"])
    bc_head.to(device).eval()
    for p in bc_head.parameters():
        p.requires_grad = False
    return bc_head, feat_dim


def cmd_gate_1b(args) -> None:
    """Gate 1b: verify BC head coherent reward has no NaN/inf.

    Loads V59 + BC head, samples transitions from D_csil, computes
    r = alpha * (clamp(logpi_V59, -12) - clamp(log_p_BC, -12)), and checks
    for numerical stability. Pass condition: no NaN and no inf.
    """
    print("=" * 60)
    print("Gate 1b: BC Head Coherent Reward Stability Check")
    print("=" * 60)
    print(f"  BC head: {args.bc_head_path}")
    print(f"  D_csil:  {args.d_csil}")
    print(f"  alpha0:  {args.alpha0}")
    print(f"  n_samples: {args.n_samples}")

    if not os.path.exists(args.bc_head_path):
        print(f"\nERROR: BC head not found at {args.bc_head_path}")
        print("  Run `python train_shallow_bc.py` first (Gate 1a).")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"  Device:  {device}")

    # Load V59.
    print("\nLoading V59 (frozen)...")
    model, _ = _load_v59_policy(args, device)
    freeze_backbone(model)
    model.policy.features_extractor.eval()
    for m in model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

    # Load BC head.
    print("Loading BC head...")
    bc_head, feat_dim = _load_bc_head(args.bc_head_path, device)
    print(f"  feat_dim={feat_dim}, params={sum(p.numel() for p in bc_head.parameters())}")

    # Load D_csil and prepare states/actions.
    print("Loading D_csil...")
    data = _load_d_csil(args.d_csil)
    n = len(data["images"])
    print(f"  {n} transitions")

    # Subsample for speed.
    rng = np.random.RandomState(42)
    idx = rng.choice(n, min(args.n_samples, n), replace=False)
    images = data["images"][idx].to(device)
    states_raw = data["states"][idx]
    states = normalize_states(states_raw, args.place_vecnorm).to(device)
    actions = data["actions"][idx].to(device)
    obs = {"image": images.permute(0, 3, 1, 2).contiguous(),
           "state": states}

    # Run Gate 1b check.
    print("\nComputing r = alpha * (clamp(logpi_V59, -12) - clamp(log_p_BC, -12))...")
    result = gate_1b_check(
        policy=model.policy,
        bc_head=bc_head,
        states=obs,
        actions=actions,
        alpha0=args.alpha0,
        batch_size=256,
        device=device,
        n_samples=len(idx),
    )

    print("\n--- Gate 1b Results ---")
    print(f"  n_samples:         {result['n_samples']}")
    print(f"  alpha0:            {result['alpha0']}")
    print(f"  has_nan:           {result['has_nan']}")
    print(f"  has_inf:           {result['has_inf']}")
    print(f"  reward_min:        {result['reward_min']:.6f}")
    print(f"  reward_max:        {result['reward_max']:.6f}")
    print(f"  reward_mean:       {result['reward_mean']:.6f}")
    print(f"  reward_std:        {result['reward_std']:.6f}")
    print(f"  reward_is_constant:{result['reward_is_constant']} "
          f"(std<1e-6 -> CSIL++ V1 failure mode)")
    passed = result["gate_1b_pass"]
    print(f"\n  Gate 1b: {'PASS' if passed else 'FAIL'} "
          f"(no NaN/inf AND reward non-constant required)")

    # Save report.
    report_path = CSIL_OUTPUT_DIR / "gate_1b_report.json"
    CSIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Report saved to {report_path}")

    if not passed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="train_csil_plus_plus.py",
        description=(
            "CSIL++: Coherent-reward Self-Imitation + Ensemble strategy for "
            "V59. Implements Task 3 of the v59-breakthrough-csil-voronoi spec."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- common V59 paths (shared by train-reward / verify / train-ensemble) ----
    def add_v59_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH,
                       help="Path to V59 place policy .zip")
        p.add_argument("--place_vecnorm", type=str, default=PLACE_VECNORM_PATH,
                       help="Path to V59 vec_normalize.pkl")
        p.add_argument("--device", type=str, default="cuda",
                       help="torch device (default: cuda)")
        p.add_argument("--seed", type=int, default=SEED)

    # ---- collect ----
    p_collect = sub.add_parser(
        "collect",
        help="Collect V59 trajectories (success + failure) -> data/D_csil.npz",
    )
    p_collect.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH)
    p_collect.add_argument("--place_vecnorm", type=str, default=PLACE_VECNORM_PATH)
    p_collect.add_argument("--grasp_model", type=str, default=GRASP_MODEL_PATH)
    p_collect.add_argument("--grasp_vecnorm", type=str, default=GRASP_VECNORM_PATH)
    p_collect.add_argument("--n_episodes", type=int, default=200,
                           help="Total episodes to run (default 200)")
    p_collect.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p_collect.add_argument("--release_threshold", type=float, default=0.05,
                           help="Release distance threshold (m)")
    p_collect.add_argument("--output", type=str, default=str(DCSIL_PATH),
                           help=f"Output .npz path (default: {DCSIL_PATH})")
    p_collect.add_argument("--seed", type=int, default=SEED)
    p_collect.set_defaults(func=cmd_collect)

    # ---- train-reward ----
    p_tr = sub.add_parser(
        "train-reward",
        help="Train potential function Phi on D_csil + run verification",
    )
    add_v59_args(p_tr)
    p_tr.add_argument("--d_csil", type=str, default=str(DCSIL_PATH),
                      help="Path to D_csil.npz")
    p_tr.add_argument("--experiment_id", type=str, default="CSILPP_PHI_V1")
    p_tr.add_argument("--n_epochs", type=int, default=100,
                      help="Phi training epochs (default 100)")
    p_tr.add_argument("--batch_size", type=int, default=256)
    p_tr.add_argument("--learning_rate", type=float, default=1e-3)
    p_tr.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                      help="Coherent reward scale alpha")
    p_tr.add_argument("--prior", type=str, default="uniform",
                      choices=["uniform", "gaussian_mixture", "bc_head"],
                      help="Prior p(a|s) for coherent reward")
    p_tr.add_argument("--bc_head_path", type=str,
                      default=str(CSIL_OUTPUT_DIR / "shallow_bc_head.pt"),
                      help="Path to BC head checkpoint (required if prior=bc_head)")
    p_tr.add_argument("--phi_input", type=str, default="latent",
                      choices=["latent", "state"],
                      help="Phi input: 'latent' (524-dim) or 'state' (12-dim)")
    p_tr.set_defaults(func=cmd_train_reward)

    # ---- verify ----
    p_v = sub.add_parser(
        "verify",
        help="Run coherent reward verification on collected data",
    )
    add_v59_args(p_v)
    p_v.add_argument("--d_csil", type=str, default=str(DCSIL_PATH))
    p_v.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p_v.add_argument("--prior", type=str, default="uniform",
                     choices=["uniform", "gaussian_mixture", "bc_head"])
    p_v.add_argument("--bc_head_path", type=str,
                     default=str(CSIL_OUTPUT_DIR / "shallow_bc_head.pt"),
                     help="Path to BC head checkpoint (required if prior=bc_head)")
    p_v.set_defaults(func=cmd_verify)

    # ---- train-ensemble ----
    p_te = sub.add_parser(
        "train-ensemble",
        help="PBRS fine-tune a_pi using Phi (Task 4)",
    )
    add_v59_args(p_te)
    p_te.add_argument("--d_csil", type=str, default=str(DCSIL_PATH),
                      help="Path to D_csil.npz (for sanity check + invariance test)")
    p_te.add_argument("--experiment_id", type=str, default="CSILPP_ENS_V1")
    p_te.add_argument("--grasp_model", type=str, default=GRASP_MODEL_PATH,
                      help="Path to grasp policy .zip (for grasp phase)")
    p_te.add_argument("--grasp_vecnorm", type=str, default=GRASP_VECNORM_PATH,
                      help="Path to grasp vec_normalize.pkl")
    # PBRS training hyperparameters (conservative — V59 is at a sharp optimum)
    p_te.add_argument("--max_iterations", type=int, default=20,
                      help="Max PPO iterations (~n_steps_per_rollout each, "
                           "default 20 = ~10k steps total)")
    p_te.add_argument("--n_steps_per_rollout", type=int, default=512,
                      help="Place-phase transitions per rollout (default 512)")
    p_te.add_argument("--learning_rate", type=float, default=1e-7,
                      help="Adam LR (1e-7 = 100x lower than typical PPO)")
    p_te.add_argument("--clip_range", type=float, default=0.1,
                      help="PPO clip range (0.1 = very tight)")
    p_te.add_argument("--max_kl", type=float, default=0.005,
                      help="Early-stop if KL exceeds this (V70 crashed at KL=0.003)")
    p_te.add_argument("--batch_size", type=int, default=64,
                      help="Mini-batch size for PPO update")
    p_te.add_argument("--n_epochs", type=int, default=2,
                      help="PPO epochs per update (few = conservative)")
    p_te.add_argument("--eval_every", type=int, default=5,
                      help="Eval place_rate every N iterations (~2500 steps)")
    p_te.add_argument("--eval_episodes", type=int, default=15,
                      help="Episodes per eval")
    p_te.add_argument("--safety_threshold", type=float, default=0.30,
                      help="place_rate below this triggers safety rollback")
    p_te.add_argument("--gamma", type=float, default=0.99,
                      help="Discount factor for GAE + PBRS shaping")
    p_te.add_argument("--gae_lambda", type=float, default=0.95,
                      help="GAE lambda")
    # CSIL++ V2: BC head prior + coherent reward with alpha annealing
    p_te.add_argument("--prior", type=str, default="uniform",
                      choices=["uniform", "gaussian_mixture", "bc_head"],
                      help="Prior for coherent reward. 'bc_head' enables V2 "
                           "shaped reward r + alpha*(logpi_V59 - logp_BC) + "
                           "gamma*Phi(s') - Phi(s). 'uniform' = V1 PBRS.")
    p_te.add_argument("--bc_head_path", type=str,
                      default=str(CSIL_OUTPUT_DIR / "shallow_bc_head.pt"),
                      help="Path to BC head checkpoint (required if --prior bc_head)")
    p_te.add_argument("--alpha0", type=float, default=0.1,
                      help="Initial coherent reward scale alpha (annealed "
                           "linearly to 0 over max_iterations). Default 0.1.")
    p_te.set_defaults(func=cmd_train_ensemble)

    # ---- eval-ensemble subcommand (SubTask 5.4: 50-ep eval) ----
    p_ee = sub.add_parser(
        "eval-ensemble",
        help="Load saved ensemble and run N-episode eval (SubTask 5.4)",
    )
    add_v59_args(p_ee)
    p_ee.add_argument("--ensemble_path", type=str,
                      default=str(CSIL_OUTPUT_DIR / "best_model" / "ensemble_policy.pt"),
                      help="Path to saved ensemble_policy.pt")
    p_ee.add_argument("--grasp_model", type=str, default=GRASP_MODEL_PATH,
                      help="Path to grasp policy .zip (for grasp phase)")
    p_ee.add_argument("--grasp_vecnorm", type=str, default=GRASP_VECNORM_PATH,
                      help="Path to grasp vec_normalize.pkl")
    p_ee.add_argument("--n_episodes", type=int, default=50,
                      help="Number of eval episodes (default 50 for reliable estimate)")
    p_ee.set_defaults(func=cmd_eval_ensemble)

    # ---- gate-1b subcommand (CSIL++ V2: BC head prior stability check) ----
    p_g1b = sub.add_parser(
        "gate-1b",
        help="Gate 1b: verify BC head coherent reward has no NaN/inf (V2)",
    )
    add_v59_args(p_g1b)
    p_g1b.add_argument("--bc_head_path", type=str,
                       default=str(CSIL_OUTPUT_DIR / "shallow_bc_head.pt"),
                       help="Path to BC head checkpoint from train_shallow_bc.py")
    p_g1b.add_argument("--d_csil", type=str, default=str(DCSIL_PATH),
                       help="Path to D_csil.npz for sampling test transitions")
    p_g1b.add_argument("--alpha0", type=float, default=0.1,
                       help="Initial alpha for reward scaling (default 0.1)")
    p_g1b.add_argument("--n_samples", type=int, default=1024,
                       help="Number of transitions to sample for check")
    p_g1b.set_defaults(func=cmd_gate_1b)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(0)

    # Dispatch
    args.func(args)


if __name__ == "__main__":
    main()
