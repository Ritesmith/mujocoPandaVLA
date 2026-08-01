"""Implicit Q-Learning (IQL) agent for offline RL.

Architecture (per user spec):
  - V Network:     MLP(state_dim → 256 → 256 → 1)
  - Q1, Q2:        MLP(state_dim + action_dim*h → 256 → 256 → 1)  (twin Q)
  - Q1_target, Q2_target: EMA copies of Q1, Q2 (τ_polyak=0.005)
  - Policy:        Gaussian with tanh squashing, state-dependent std

When chunk_size > 1 (v4 true action chunking):
  - Q networks take (state, action_chunk) where action_chunk = action_dim * h
  - Policy outputs action_dim * h (a full action sequence)
  - This gives temporal consistency: h consecutive actions from one decision

Training (three-step):
  1. V update:  Expectile loss L_τ(Q_target(s,a_chunk) - V(s)), τ=0.7
  2. Q update:  MSE(Q(s,a_chunk), r + γ^h(1-done)V(s'))
  3. Policy:    AWR loss -E[exp(β·A(s,a_chunk)) · log π(a_chunk|s)], β=3.0

Key design decisions (based on reward density diagnostic):
  - τ=0.7 (not 0.9): 68.5% success rate, don't need aggressive upper-tail
  - β=3.0 (Kostrikov default): amplifies small advantages via exp()
  - Twin Q + min: counteracts overestimation
  - tanh squashing: ensures actions in [-1, 1]
  - Advantage clip at 100.0: prevents numerical overflow in exp()
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Network definitions
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Simple MLP with ReLU activations."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256,
                 n_hidden: int = 2):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy with state-dependent std and tanh squashing.

    Outputs μ(s) and log_std(s), then applies tanh to (μ + σ·ε).
    Log-probability includes the tanh Jacobian correction.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_hidden: int = 2, log_std_min: float = -20.0,
                 log_std_max: float = 2.0):
        super().__init__()
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        layers = []
        prev = state_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        """Sample action and compute log-prob with tanh correction."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)

        # Log-prob with tanh Jacobian correction
        log_prob = normal.log_prob(x)
        # Enforcing Action Bound (Appendix C of SAC paper)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def log_prob(self, state, action):
        """Compute log π(a|s) for a given action (for AWR loss).

        Since the action was squashed with tanh, we need to inverse-transform:
        x = atanh(a), then compute log N(x; μ, σ) - log|d/dx tanh(x)|
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Clamp action to avoid atanh(±1) = ±inf
        action_clamped = action.clamp(-1 + 1e-6, 1 - 1e-6)
        x = 0.5 * torch.log((1 + action_clamped) / (1 - action_clamped))  # atanh

        log_prob = -0.5 * ((x - mean) / std).pow(2) - log_std - 0.5 * math.log(2 * math.pi)
        log_prob -= torch.log(1 - action_clamped.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return log_prob

    def get_action(self, state, deterministic=True):
        """Get action for evaluation (numpy interface)."""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if deterministic:
                action = torch.tanh(mean)
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x = normal.sample()
                action = torch.tanh(x)
        return action.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# IQL Agent
# ---------------------------------------------------------------------------

class IQLAgent:
    """Implicit Q-Learning agent.

    Implements the three-step training:
      1. V update via Expectile regression
      2. Q update via Bellman backup with V target
      3. Policy update via Advantage-Weighted Regression (AWR)
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 8,
        hidden_dim: int = 256,
        tau: float = 0.7,           # Expectile coefficient
        beta: float = 3.0,          # AWR temperature
        gamma: float = 0.99,        # Discount factor
        polyak: float = 0.005,      # Target network soft update rate
        lr_v: float = 3e-4,
        lr_q: float = 3e-4,
        lr_policy: float = 3e-4,
        advantage_clip: float = 100.0,
        n_step: int = 1,            # Q-chunking: h-step bootstrap (1=standard IQL)
        chunk_size: int = 1,        # v4: true action chunking (Actor outputs h actions)
        device: str = "cpu",
        # Phase 7 Round 1 A/B stability options (消融式, each independently testable)
        ema_v: bool = False,        # Maintain EMA copy of V for Q-targets & advantage
        ema_tau: float = 0.005,     # EMA soft-update rate for V
        huber_loss: bool = False,   # Use smooth_L1 (Huber) instead of MSE on Q update
        huber_delta: float = 10.0,  # Huber beta: |err|<delta → MSE-like, else L1-like
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim        # single action dimension
        self.chunk_size = chunk_size        # h: number of actions per chunk
        self.chunk_action_dim = action_dim * chunk_size  # total chunk output dim
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_n = gamma ** n_step  # Precompute γ^h for n-step bootstrap
        self.polyak = polyak
        self.advantage_clip = advantage_clip
        self.device = torch.device(device)

        # Phase 7 Round 1 A/B stability options
        self.ema_v = ema_v
        self.ema_tau = ema_tau
        self.huber_loss = huber_loss
        self.huber_delta = huber_delta

        # Networks
        # V network: state-only input (unchanged)
        self.v_net = MLP(state_dim, 1, hidden_dim).to(self.device)
        # EMA copy of V (only created when ema_v=True). Used as a stable target
        # for Q bootstrap and advantage computation, decoupled from the V that
        # is being actively trained via expectile regression. This mitigates the
        # V multi-solution oscillation identified as the root cause of training
        # instability (Phase 7 P-0: v_mean CV=51.7% → 77.1% under reward norm).
        self.v_net_ema = copy.deepcopy(self.v_net) if ema_v else None
        # Q networks: take (state, action_chunk) — chunk_action_dim = action_dim * h
        self.q1_net = MLP(state_dim + self.chunk_action_dim, 1, hidden_dim).to(self.device)
        self.q2_net = MLP(state_dim + self.chunk_action_dim, 1, hidden_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1_net)
        self.q2_target = copy.deepcopy(self.q2_net)
        # Policy: outputs action_chunk (action_dim * h values)
        self.policy = GaussianPolicy(state_dim, self.chunk_action_dim, hidden_dim).to(self.device)

        # Optimizers
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=lr_v)
        self.q1_optimizer = torch.optim.Adam(self.q1_net.parameters(), lr=lr_q)
        self.q2_optimizer = torch.optim.Adam(self.q2_net.parameters(), lr=lr_q)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)

        # Freeze target networks (and EMA V)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        if self.v_net_ema is not None:
            for p in self.v_net_ema.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _expectile_loss(self, diff: torch.Tensor, tau: float) -> torch.Tensor:
        """Asymmetric L2 loss for Expectile regression.

        L_τ(diff) = τ * max(diff, 0)^2 + (1-τ) * min(diff, 0)^2
        When diff > 0 (Q > V): weight = τ (upward pull)
        When diff < 0 (Q < V): weight = 1-τ (downward pull, weaker)
        """
        weight = torch.where(diff > 0, tau, 1.0 - tau)
        return (weight * diff.pow(2)).mean()

    def update_v(self, states, actions):
        """Step 1: Update V network via Expectile regression."""
        with torch.no_grad():
            sa = torch.cat([states, actions], dim=-1)
            q1 = self.q1_target(sa)
            q2 = self.q2_target(sa)
            q_min = torch.min(q1, q2)

        v = self.v_net(states)
        v_loss = self._expectile_loss(q_min - v, self.tau)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return {
            "v_loss": v_loss.item(),
            "v_mean": v.mean().item(),
            "q_target_mean": q_min.mean().item(),
            "expectile_gap": (q_min.mean() - v.mean()).item(),
        }

    def update_q(self, states, actions, rewards, next_states, dones):
        """Step 2: Update Q networks via Bellman backup.

        When n_step > 1 (Q-chunking), rewards contains the n-step
        discounted return Σ γ^i r_{t+i}, next_states is s_{t+h},
        and the bootstrap uses γ^h (self.gamma_n).

        Phase 7 Round 1 A/B options:
          - ema_v=True: bootstrap V target from v_net_ema (stable copy) instead
            of v_net (actively trained, oscillates between expectile solutions).
          - huber_loss=True: replace MSE with smooth_L1 (Huber) for robustness
            to heavy-tailed Q targets ( Towards Robust Offline RL, ICLR 2024).
        """
        with torch.no_grad():
            # Use EMA V for bootstrap target if available (stability A/B option)
            v_source = self.v_net_ema if self.v_net_ema is not None else self.v_net
            v_next = v_source(next_states)
            q_target = rewards + (1.0 - dones) * self.gamma_n * v_next

        sa = torch.cat([states, actions], dim=-1)
        q1_pred = self.q1_net(sa)
        q2_pred = self.q2_net(sa)

        if self.huber_loss:
            # smooth_L1: 0.5*err^2/delta if |err|<delta else |err|-0.5*delta
            # Behaves like MSE for in-distribution errors, L1 for outliers.
            q1_loss = F.smooth_l1_loss(q1_pred, q_target, beta=self.huber_delta)
            q2_loss = F.smooth_l1_loss(q2_pred, q_target, beta=self.huber_delta)
        else:
            q1_loss = F.mse_loss(q1_pred, q_target)
            q2_loss = F.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
            "q_target_mean": q_target.mean().item(),
        }

    def update_policy(self, states, actions):
        """Step 3: Update policy via Advantage-Weighted Regression (AWR).

        Phase 7 Round 1 A/B: when ema_v=True, the advantage baseline V is
        taken from v_net_ema (stable) rather than v_net (oscillating). This
        stabilizes AWR weights — advantage = Q - V_ema is less sensitive to
        per-step V jumps caused by expectile-regression non-convexity.
        """
        with torch.no_grad():
            sa = torch.cat([states, actions], dim=-1)
            q1 = self.q1_net(sa)
            q2 = self.q2_net(sa)
            q_min = torch.min(q1, q2)
            # Use EMA V as advantage baseline if available (stability A/B option)
            v_source = self.v_net_ema if self.v_net_ema is not None else self.v_net
            v = v_source(states)

            advantage = q_min - v
            # Clip advantage to prevent numerical overflow in exp()
            weight = torch.exp(self.beta * advantage).clamp(max=self.advantage_clip)

        log_prob = self.policy.log_prob(states, actions)
        policy_loss = -(weight * log_prob).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Compute monitoring metrics
        with torch.no_grad():
            awr_weights = weight.squeeze(-1)  # (batch_size,) — don't squeeze batch dim
            max_weight = awr_weights.max().item()
            mean_weight = awr_weights.mean().item()
            # Effective sample size (ESS)
            ess = (awr_weights.sum() ** 2) / (awr_weights ** 2).sum()
            # Weight entropy (high entropy = weights too flat = no signal)
            p = awr_weights / awr_weights.sum()
            entropy = -(p * torch.log(p + 1e-8)).sum().item()
            max_entropy = math.log(awr_weights.shape[0])

        return {
            "policy_loss": policy_loss.item(),
            "advantage_mean": advantage.mean().item(),
            "advantage_std": advantage.std().item() if advantage.numel() > 1 else 0.0,
            "advantage_max": advantage.max().item(),
            "advantage_min": advantage.min().item(),
            "awr_weight_max": max_weight,
            "awr_weight_mean": mean_weight,
            "ess": ess.item(),
            "weight_entropy": entropy,
            "weight_entropy_ratio": entropy / max_entropy if max_entropy > 0 else 0,
            "log_prob_mean": log_prob.mean().item(),
        }

    def soft_update_targets(self):
        """Soft-update target Q networks: θ_target ← (1-τ)θ_target + τθ"""
        for target_param, param in zip(self.q1_target.parameters(), self.q1_net.parameters()):
            target_param.data.mul_(1.0 - self.polyak)
            target_param.data.add_(self.polyak * param.data)
        for target_param, param in zip(self.q2_target.parameters(), self.q2_net.parameters()):
            target_param.data.mul_(1.0 - self.polyak)
            target_param.data.add_(self.polyak * param.data)

    def soft_update_v_ema(self):
        """Soft-update EMA V network: v_ema ← (1-τ_ema)v_ema + τ_ema·v_net.

        Called after update_v() so the EMA tracks the just-updated V. With
        τ_ema=0.005 (default, same as polyak), the EMA lags ~200 steps behind,
        providing a smoothed V for Q-targets and advantage computation. This
        is the Phase 7 Round 1 A/B stability mechanism for V multi-solution
        oscillation.
        """
        if self.v_net_ema is None:
            return
        for ema_p, p in zip(self.v_net_ema.parameters(), self.v_net.parameters()):
            ema_p.data.mul_(1.0 - self.ema_tau)
            ema_p.data.add_(self.ema_tau * p.data)

    def train_step(self, batch):
        """Full three-step IQL training update."""
        states = batch["state"].to(self.device)
        actions = batch["action"].to(self.device)
        rewards = batch["reward"].to(self.device)
        next_states = batch["next_state"].to(self.device)
        dones = batch["done"].to(self.device)

        v_info = self.update_v(states, actions)
        # EMA V must be updated AFTER update_v so it tracks the new V params.
        # Done before update_q/update_policy so this step's Q-target & advantage
        # use a V_ema that already reflects the latest V (one-step lag remains
        # because τ_ema is small, but the update is sequenced correctly).
        self.soft_update_v_ema()
        q_info = self.update_q(states, actions, rewards, next_states, dones)
        policy_info = self.update_policy(states, actions)
        self.soft_update_targets()

        return {**v_info, **q_info, **policy_info}

    def train_step_full(self, data):
        """Training step using full dataset tensors (no DataLoader)."""
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        rewards = data["rewards"].to(self.device)
        next_states = data["next_states"].to(self.device)
        dones = data["dones"].to(self.device)

        v_info = self.update_v(states, actions)
        self.soft_update_v_ema()
        q_info = self.update_q(states, actions, rewards, next_states, dones)
        policy_info = self.update_policy(states, actions)
        self.soft_update_targets()

        return {**v_info, **q_info, **policy_info}

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_q_separation(self, data, success_mask=None):
        """Check if Q values separate success vs failure trajectories.

        This is the key metric from the user's risk analysis:
        Q_success - Q_failure should be > 100 for IQL to work.
        """
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)
        sa = torch.cat([states, actions], dim=-1)

        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        q = torch.min(q1, q2)
        v = self.v_net(states)
        advantage = q - v

        if success_mask is not None:
            success_mask_t = torch.BoolTensor(success_mask).to(self.device)
            q_success = q[success_mask_t].mean().item() if success_mask_t.any() else 0
            q_failure = q[~success_mask_t].mean().item() if (~success_mask_t).any() else 0
            a_success = advantage[success_mask_t].mean().item() if success_mask_t.any() else 0
            a_failure = advantage[~success_mask_t].mean().item() if (~success_mask_t).any() else 0
        else:
            q_success = q_failure = 0
            a_success = a_failure = 0

        return {
            "q_success": q_success,
            "q_failure": q_failure,
            "q_gap": q_success - q_failure,
            "advantage_success": a_success,
            "advantage_failure": a_failure,
            "advantage_gap": a_success - a_failure,
        }

    @torch.no_grad()
    def compute_awr_weight_distribution(self, data):
        """Compute AWR weight distribution for histogram analysis (P-1c sanity check).

        Returns raw advantage and AWR weights on the full dataset to verify
        that reward normalization fixed the weight binarization problem
        (weights should be continuous, not bimodal at 0 and clamp ceiling).
        """
        states = data["states"].to(self.device)
        actions = data["actions"].to(self.device)

        sa = torch.cat([states, actions], dim=-1)
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        q_min = torch.min(q1, q2)
        v = self.v_net(states)

        advantage = (q_min - v).squeeze(-1)
        weight = torch.exp(self.beta * advantage).clamp(max=self.advantage_clip)

        adv_np = advantage.cpu().numpy()
        weight_np = weight.cpu().numpy()

        # Binarization diagnostic: fraction at clamp ceiling and near zero
        n_total = len(weight_np)
        n_at_clamp = int((weight_np >= self.advantage_clip - 1e-4).sum())
        n_near_zero = int((weight_np < 1e-2).sum())
        n_continuous = n_total - n_at_clamp - n_near_zero

        # Histogram bins (0 to clamp ceiling)
        n_bins = 50
        hist_counts, hist_edges = np.histogram(weight_np, bins=n_bins, range=(0, self.advantage_clip))

        return {
            "n_total": n_total,
            "advantage_mean": float(adv_np.mean()),
            "advantage_std": float(adv_np.std()),
            "advantage_min": float(adv_np.min()),
            "advantage_max": float(adv_np.max()),
            "weight_mean": float(weight_np.mean()),
            "weight_std": float(weight_np.std()),
            "weight_min": float(weight_np.min()),
            "weight_max": float(weight_np.max()),
            "n_at_clamp_ceiling": n_at_clamp,
            "frac_at_clamp_ceiling": n_at_clamp / n_total,
            "n_near_zero": n_near_zero,
            "frac_near_zero": n_near_zero / n_total,
            "n_continuous_middle": n_continuous,
            "frac_continuous_middle": n_continuous / n_total,
            "histogram_counts": hist_counts.tolist(),
            "histogram_edges": hist_edges.tolist(),
            "beta": self.beta,
            "advantage_clip": self.advantage_clip,
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        ckpt = {
            "v_net": self.v_net.state_dict(),
            "q1_net": self.q1_net.state_dict(),
            "q2_net": self.q2_net.state_dict(),
            "policy": self.policy.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "q1_optimizer": self.q1_optimizer.state_dict(),
            "q2_optimizer": self.q2_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "tau": self.tau,
            "beta": self.beta,
            "gamma": self.gamma,
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "n_step": self.n_step,
            # Phase 7 Round 1 A/B config (for evaluation-side consistency)
            "ema_v": self.ema_v,
            "ema_tau": self.ema_tau,
            "huber_loss": self.huber_loss,
            "huber_delta": self.huber_delta,
        }
        # EMA V state (only if trained with ema_v=True). Saved so evaluation
        # can reconstruct the same V_ema used for advantage if needed, and so
        # resumed training starts from the correct EMA state.
        if self.v_net_ema is not None:
            ckpt["v_net_ema"] = self.v_net_ema.state_dict()
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.v_net.load_state_dict(ckpt["v_net"])
        self.q1_net.load_state_dict(ckpt["q1_net"])
        self.q2_net.load_state_dict(ckpt["q2_net"])
        self.policy.load_state_dict(ckpt["policy"])
