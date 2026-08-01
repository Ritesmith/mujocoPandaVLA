"""Offline dataset for IQL training on D_expert.npz.

Loads the Oracle expert demonstrations and constructs (s, a, r, s', done)
tuples for offline RL. Rewards are computed from the state vector using
the place_safe reward structure (dominant components only).

State layout (VisionObs, 12-dim):
  [0:7]   joint_positions
  [7]     gripper opening (finger position, ~0=closed, ~0.04=open)
  [8]     block_target_distance (meters)
  [9:12]  target_position (x, y, z)

Reward computation (place_safe structure):
  Per-step (all <= 0):
    - distance: -5.0 * dist (while holding)
    - hover:    -0.5 * (0.05 - dist)/0.05 (while holding & dist < 0.05)
    - jerk:     -0.001 * ||a_t - 2*a_{t-1} + a_{t-2}||^2
    - action_diff: -0.005 * ||a_t - a_{t-1}||^2
    - time:     -0.01
  Terminal (one-time):
    - success: +200 (at end of successful episode)
    - release: +50  (at end of successful episode)
    - early_release: -5.0 (if released away from target)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

GRIPPER_OPEN_THRESHOLD = 0.02
TABLE_Z = 0.22


def compute_rewards_from_states(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    success_flags: np.ndarray,
    reward_shaping: bool = False,
) -> np.ndarray:
    """Compute per-step rewards from state observations.

    Approximates the place_safe reward function using the available state
    information (block_target_distance and gripper opening). Drops the
    height penalty (block_z not in state) which is a secondary component.

    When reward_shaping=True (Step 2c), adds direction-aware dense rewards:
      - Overshoot penalty: approaching target too fast (|Δdist| > threshold)
      - Stay bonus: close to target AND stable (low velocity)
      - Leave penalty: was close, now moving away from target
    These address the "reach 3cm then drift to 20cm" pattern by rewarding
    stable proximity rather than just distance reduction.
    """
    # Step 2c shaping parameters
    OVERSHOOT_SPEED_THRESHOLD = 0.015   # 1.5cm/step — penalize faster approach
    OVERSHOOT_PENALTY = 10.0
    STAY_DIST_THRESHOLD = 0.08          # 8cm — within this range, staying is rewarded
    STAY_VELOCITY_THRESHOLD = 0.005     # 0.5cm/step — "stable" if moving less than this
    STAY_BONUS = 0.5                    # positive reward per step for staying close & stable
    LEAVE_PENALTY = 2.0                 # penalty for leaving the close zone after being in it

    n = len(states)
    rewards = np.zeros(n, dtype=np.float32)

    unique_eps = np.unique(episode_ids)

    for ep_id in unique_eps:
        mask = episode_ids == ep_id
        ep_indices = np.where(mask)[0]
        ep_len = len(ep_indices)
        ep_success = int(success_flags[ep_id]) if ep_id < len(success_flags) else 0

        for i, idx in enumerate(ep_indices):
            s = states[idx]
            dist = float(s[8])
            gripper = float(s[7])
            is_holding = gripper < GRIPPER_OPEN_THRESHOLD

            reward = 0.0

            # Distance penalty (while holding)
            if is_holding:
                reward -= 5.0 * dist

                # Hover penalty (when very close to target)
                if dist < 0.05:
                    hover_intensity = (0.05 - dist) / 0.05
                    reward -= 0.5 * hover_intensity

            # Action diff penalty
            if i > 0:
                prev_idx = ep_indices[i - 1]
                action_diff = actions[idx] - actions[prev_idx]
                reward -= 0.005 * float(np.sum(action_diff ** 2))

            # Jerk penalty
            if i > 1:
                prev_idx = ep_indices[i - 1]
                prev_prev_idx = ep_indices[i - 2]
                jerk = actions[idx] - 2.0 * actions[prev_idx] + actions[prev_prev_idx]
                reward -= 0.001 * float(np.sum(jerk ** 2))

            # Time penalty
            reward -= 0.01

            # Step 2c: Direction-aware dense reward shaping
            if reward_shaping and i > 0 and is_holding:
                prev_dist = float(states[ep_indices[i - 1]][8])
                delta_dist = dist - prev_dist  # negative=approaching, positive=receding

                # 1. Overshoot penalty: approaching too fast
                if delta_dist < -OVERSHOOT_SPEED_THRESHOLD:
                    excess_speed = abs(delta_dist) - OVERSHOOT_SPEED_THRESHOLD
                    reward -= OVERSHOOT_PENALTY * excess_speed

                # 2. Stay bonus: close AND stable
                if dist < STAY_DIST_THRESHOLD and abs(delta_dist) < STAY_VELOCITY_THRESHOLD:
                    reward += STAY_BONUS

                # 3. Leave penalty: was close, now moving away
                if prev_dist < STAY_DIST_THRESHOLD and delta_dist > STAY_VELOCITY_THRESHOLD:
                    reward -= LEAVE_PENALTY * delta_dist

            # Terminal rewards at last step of episode
            if i == ep_len - 1:
                if ep_success:
                    # Success: +200 terminal + +50 release bonus
                    reward += 200.0 + 50.0
                else:
                    # Failure: check for early release penalty
                    # If was holding but not at target and released
                    if is_holding and dist > 0.05:
                        reward -= 5.0  # early release penalty

            rewards[idx] = np.clip(reward, -20.0, 400.0)

    return rewards


def compute_next_states_and_dones(
    states: np.ndarray,
    episode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute next_states and done flags from consecutive transitions.

    next_state = states[i+1] if same episode, else zeros (terminal)
    done = True if next transition is different episode or last transition
    """
    n = len(states)
    next_states = np.zeros_like(states)
    dones = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if i + 1 < n and episode_ids[i + 1] == episode_ids[i]:
            next_states[i] = states[i + 1]
            dones[i] = 0.0
        else:
            # Last transition in episode
            next_states[i] = np.zeros_like(states[i])
            dones[i] = 1.0

    return next_states, dones


def compute_n_step_data(
    states: np.ndarray,
    rewards: np.ndarray,
    episode_ids: np.ndarray,
    gamma: float = 0.99,
    n_step: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute n-step returns, n-step next states, and n-step dones.

    For transition at position t within an episode of length L:
      n_step_return = Σ_{i=0}^{h-1} γ^i * r_{t+i}  (truncated at episode end)
      n_step_next_state = states[t+h] if t+h < L, else zeros (terminal)
      n_step_done = 0.0 if t+h < L, else 1.0

    This enables h-step bootstrap in IQL's Q update:
      Q(s_t, a_t) ← n_step_return + γ^h * (1 - n_step_done) * V(s_{t+h})

    Key property: no off-policy bias because we use dataset actions for
    the h-step lookahead (not policy-sampled actions). This is the
    in-sample n-step return, consistent with IQL's design principle.
    """
    n = len(states)
    n_step_returns = np.zeros(n, dtype=np.float32)
    n_step_next_states = np.zeros_like(states)
    n_step_dones = np.zeros(n, dtype=np.float32)

    unique_eps = np.unique(episode_ids)
    for ep_id in unique_eps:
        ep_mask = episode_ids == ep_id
        ep_indices = np.where(ep_mask)[0]
        ep_len = len(ep_indices)

        for i, t in enumerate(ep_indices):
            # Compute n-step discounted return
            cum_reward = 0.0
            discount = 1.0
            actual_h = 0
            for j in range(n_step):
                if i + j >= ep_len:
                    break
                actual_h += 1
                cum_reward += discount * rewards[ep_indices[i + j]]
                discount *= gamma

            n_step_returns[t] = cum_reward

            # n-step next state and done
            if i + actual_h < ep_len:
                next_idx = ep_indices[i + actual_h]
                n_step_next_states[t] = states[next_idx]
                n_step_dones[t] = 0.0
            else:
                # Episode ended within the window
                n_step_next_states[t] = np.zeros_like(states[t])
                n_step_dones[t] = 1.0

    return n_step_returns, n_step_next_states, n_step_dones


def compute_action_chunks(
    actions: np.ndarray,
    episode_ids: np.ndarray,
    chunk_size: int = 4,
) -> np.ndarray:
    """Compute action chunks a_{t:t+h-1} for each transition (v4 true action chunking).

    For transition t in episode of length L:
      chunk = [a_t, a_{t+1}, ..., a_{min(t+h-1, L-1)}] padded with zeros to length h

    Returns: (N, action_dim * chunk_size) array of flattened action chunks.
    Each h-step sub-block is one action; zeros padding at episode boundaries.
    """
    n = len(actions)
    action_dim = actions.shape[1]
    chunks = np.zeros((n, action_dim * chunk_size), dtype=np.float32)

    unique_eps = np.unique(episode_ids)
    for ep_id in unique_eps:
        ep_mask = episode_ids == ep_id
        ep_indices = np.where(ep_mask)[0]
        ep_len = len(ep_indices)

        for i, t in enumerate(ep_indices):
            for j in range(chunk_size):
                if i + j < ep_len:
                    src_idx = ep_indices[i + j]
                    chunks[t, j * action_dim:(j + 1) * action_dim] = actions[src_idx]
                # else: leave as zeros (padding for episode boundary)

    return chunks


class OfflineDataset(Dataset):
    """PyTorch Dataset for offline RL on D_expert.npz.

    Loads (s, a, r, s', done) tuples with state normalization.
    Supports n-step returns (Q-chunking bootstrap) and distant state
    oversampling for drift mitigation.
    """

    def __init__(
        self,
        data_path: str = "/home/w/vla_workspace/data/D_expert.npz",
        normalize_states: bool = True,
        normalize_actions: bool = True,
        n_step: int = 1,
        gamma: float = 0.99,
        oversample_dist_range: tuple = None,
        oversample_factor: int = 3,
        reward_shaping: bool = False,
        chunk_size: int = 1,
        normalize_rewards: bool = False,
        max_episode_steps: int = 480,
    ):
        data = np.load(data_path, allow_pickle=True)

        self.states = data["states"].astype(np.float32)        # (N, 12)
        self.actions = data["actions"].astype(np.float32)       # (N, 8)
        self.episode_ids = data["episode_ids"]                   # (N,)
        self.success_flags = data["success_flags"]               # (n_episodes,)
        self.n_step = n_step
        self.gamma = gamma
        self.reward_shaping = reward_shaping
        self.chunk_size = chunk_size
        self.normalize_rewards = normalize_rewards
        self.max_episode_steps = max_episode_steps

        n_transitions = len(self.states)
        n_episodes = int(data["n_episodes"])
        n_placed = int(data["n_placed"])
        print(f"[OfflineDataset] Loaded {n_transitions} transitions from {n_episodes} episodes")
        print(f"  Place rate: {n_placed}/{n_episodes} = {n_placed/n_episodes:.1%}")
        print(f"  n_step: {n_step}, gamma: {gamma}, reward_shaping: {reward_shaping}")
        print(f"  chunk_size: {chunk_size} (true action chunking)")
        print(f"  normalize_rewards: {normalize_rewards}")

        # Compute rewards, next_states, dones
        print("  Computing rewards from states...")
        self.rewards = compute_rewards_from_states(
            self.states, self.actions, self.episode_ids, self.success_flags,
            reward_shaping=reward_shaping,
        )

        # --- Reward normalization (std-based, Phase 7 P-1b BUG fix) ---
        # Root cause: unnormalized rewards (std≈17, range=[-6,250]) → large Q targets
        # → AWR weight binarization (exp(β·adv) overflows, clamped to 100) →
        # gradient signal loss → V multi-solution → v_mean CV=51.7%
        #
        # CORL trajectory-range formula (reward /= (max_ret-min_ret) * max_ep_len)
        # is INEFFECTIVE for this dataset: norm factor ≈ 1.10 because
        # return_range (435.75) ≈ max_episode_steps (480). Using std-based
        # normalization instead (reward /= reward_std → std=1.0).
        # Trajectory return statistics logged for CORL-style comparability.
        self.reward_mean_raw = float(self.rewards.mean())
        self.reward_std_raw = float(self.rewards.std())
        self.reward_min_raw = float(self.rewards.min())
        self.reward_max_raw = float(self.rewards.max())

        # Trajectory returns (per-episode sum) for CORL reference
        unique_eps = np.unique(self.episode_ids)
        traj_returns = np.array([
            self.rewards[self.episode_ids == ep].sum() for ep in unique_eps
        ])
        self.traj_return_min = float(traj_returns.min())
        self.traj_return_max = float(traj_returns.max())
        self.traj_return_range = self.traj_return_max - self.traj_return_min
        self.traj_return_mean = float(traj_returns.mean())
        self.traj_return_std = float(traj_returns.std())
        # CORL norm factor (reference only — ≈1.10, ineffective)
        self.corl_norm_factor = (
            float(self.max_episode_steps / self.traj_return_range)
            if self.traj_return_range > 0 else 1.0
        )

        if normalize_rewards:
            self.reward_norm_std = self.reward_std_raw + 1e-8
            self.rewards = (self.rewards / self.reward_norm_std).astype(np.float32)
            print(f"  Reward normalization (std-based):")
            print(f"    Raw: mean={self.reward_mean_raw:.4f}, std={self.reward_std_raw:.4f}, "
                  f"range=[{self.reward_min_raw:.4f}, {self.reward_max_raw:.4f}]")
            print(f"    Trajectory returns: min={self.traj_return_min:.2f}, "
                  f"max={self.traj_return_max:.2f}, range={self.traj_return_range:.2f}")
            print(f"    CORL norm factor (reference): {self.corl_norm_factor:.4f} "
                  f"(≈1, ineffective for this dataset)")
            print(f"    Normed: mean={self.rewards.mean():.4f}, "
                  f"std={self.rewards.std():.4f} (target std=1.0)")
        else:
            self.reward_norm_std = 1.0
            print(f"  Reward normalization DISABLED (raw std={self.reward_std_raw:.4f})")

        print("  Computing next_states and dones...")
        self.next_states, self.dones = compute_next_states_and_dones(
            self.states, self.episode_ids
        )

        # Compute n-step data if n_step > 1
        if n_step > 1:
            print(f"  Computing {n_step}-step returns (Q-chunking bootstrap)...")
            self.n_step_returns, self.n_step_next_states, self.n_step_dones = \
                compute_n_step_data(
                    self.states, self.rewards, self.episode_ids,
                    gamma=gamma, n_step=n_step,
                )
            print(f"    n_step_return: mean={self.n_step_returns.mean():.2f}, "
                  f"std={self.n_step_returns.std():.2f}")
            print(f"    n_step_done rate: {self.n_step_dones.mean():.1%}")
        else:
            self.n_step_returns = self.rewards.copy()
            self.n_step_next_states = self.next_states.copy()
            self.n_step_dones = self.dones.copy()

        # State normalization
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0) + 1e-6
        if normalize_states:
            self.states_norm = ((self.states - self.state_mean) / self.state_std).astype(np.float32)
            self.next_states_norm = ((self.next_states - self.state_mean) / self.state_std).astype(np.float32)
            self.n_step_next_states_norm = (
                (self.n_step_next_states - self.state_mean) / self.state_std
            ).astype(np.float32)
        else:
            self.states_norm = self.states.copy()
            self.next_states_norm = self.next_states.copy()
            self.n_step_next_states_norm = self.n_step_next_states.copy()

        # Action normalization (verify range, clip to [-1, 1])
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0) + 1e-6
        if normalize_actions:
            self.actions_norm = ((self.actions - self.action_mean) / self.action_std).astype(np.float32)
        else:
            self.actions_norm = np.clip(self.actions, -1.0, 1.0).copy()

        # v4: Compute action chunks for true action chunking
        # When chunk_size > 1, each transition's "action" becomes a flattened
        # sequence of h consecutive actions. The Actor outputs action_dim*h,
        # and the Critic takes (state, action_chunk) as input.
        if chunk_size > 1:
            print(f"  Computing action chunks (chunk_size={chunk_size})...")
            self.action_chunks = compute_action_chunks(
                self.actions_norm, self.episode_ids, chunk_size=chunk_size,
            )
            print(f"    Action chunk shape: {self.action_chunks.shape} "
                  f"(action_dim={self.actions.shape[1]} × h={chunk_size})")
        else:
            # chunk_size=1: action chunk is just the single action (backward compatible)
            self.action_chunks = self.actions_norm.copy()

        # Distant state oversampling: duplicate transitions with dist in range
        self.sample_indices = np.arange(len(self.states))
        if oversample_dist_range is not None:
            dist_lo, dist_hi = oversample_dist_range
            dists = self.states[:, 8]  # block_target_distance
            distant_mask = (dists >= dist_lo) & (dists <= dist_hi)
            n_distant = distant_mask.sum()
            distant_indices = np.where(distant_mask)[0]
            # Duplicate distant transitions oversample_factor times
            extra_indices = np.tile(distant_indices, oversample_factor - 1)
            self.sample_indices = np.concatenate([self.sample_indices, extra_indices])
            print(f"  Distant state oversampling: dist=[{dist_lo},{dist_hi}]m, "
                  f"{n_distant} transitions × {oversample_factor}x = "
                  f"{n_distant * oversample_factor} (added {n_distant * (oversample_factor - 1)})")
            print(f"  Effective dataset size: {len(self.sample_indices)} (was {len(self.states)})")

        # Build normalize_dict for persistence (save with model checkpoint)
        # Critical: evaluation must reuse these stats to avoid train/eval
        # distribution mismatch (sony/oil apply_norm_state pattern)
        self.normalize_dict = {
            "reward": {
                "method": "std" if normalize_rewards else "none",
                "norm_std": float(self.reward_norm_std),
                "mean_raw": self.reward_mean_raw,
                "std_raw": self.reward_std_raw,
                "min_raw": self.reward_min_raw,
                "max_raw": self.reward_max_raw,
                "traj_return_min": self.traj_return_min,
                "traj_return_max": self.traj_return_max,
                "traj_return_range": self.traj_return_range,
                "traj_return_mean": self.traj_return_mean,
                "traj_return_std": self.traj_return_std,
                "corl_norm_factor": self.corl_norm_factor,
                "max_episode_steps": max_episode_steps,
            },
            "state": {
                "mean": self.state_mean.tolist(),
                "std": self.state_std.tolist(),
                "normalize_states": normalize_states,
            },
            "action": {
                "mean": self.action_mean.tolist(),
                "std": self.action_std.tolist(),
                "normalize_actions": normalize_actions,
            },
        }

        # Print statistics
        self._print_stats()

    def get_normalize_dict(self) -> dict:
        """Return normalization statistics for persistence.

        Save this dict alongside the model checkpoint. Evaluation code
        must apply the same state/reward normalization to avoid
        train/eval distribution mismatch.
        """
        return self.normalize_dict

    def _print_stats(self):
        norm_tag = f" (normalized, raw std={self.reward_std_raw:.4f})" if self.normalize_rewards else ""
        print(f"\n  Reward statistics{norm_tag}:")
        print(f"    mean: {self.rewards.mean():.4f}")
        print(f"    std:  {self.rewards.std():.4f}")
        print(f"    min:  {self.rewards.min():.4f}")
        print(f"    max:  {self.rewards.max():.4f}")
        non_zero = (np.abs(self.rewards) > 0.001).mean()
        print(f"    non-zero ratio: {non_zero:.4f}")

        # Per-episode return
        unique_eps = np.unique(self.episode_ids)
        ep_returns = []
        for ep_id in unique_eps:
            mask = self.episode_ids == ep_id
            ep_returns.append(self.rewards[mask].sum())
        ep_returns = np.array(ep_returns)
        success_mask = self.success_flags[:len(ep_returns)] == 1
        print(f"\n  Episode returns:")
        print(f"    Success episodes: mean={ep_returns[success_mask].mean():.2f}, "
              f"std={ep_returns[success_mask].std():.2f}")
        print(f"    Failure episodes: mean={ep_returns[~success_mask].mean():.2f}, "
              f"std={ep_returns[~success_mask].std():.2f}")
        print(f"    Q-value gap: {ep_returns[success_mask].mean() - ep_returns[~success_mask].mean():.2f}")

        print(f"\n  State normalization:")
        print(f"    mean: {self.state_mean}")
        print(f"    std:  {self.state_std}")

        print(f"\n  Action statistics:")
        print(f"    mean: {self.action_mean}")
        print(f"    std:  {self.action_std}")
        print(f"    min:  {self.actions.min(axis=0)}")
        print(f"    max:  {self.actions.max(axis=0)}")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        """Return n-step data. When n_step=1, this is equivalent to 1-step IQL.

        When chunk_size > 1 (v4), 'action' is a flattened action chunk of
        action_dim * chunk_size values. Otherwise it's a single action.
        """
        real_idx = self.sample_indices[idx]
        return {
            "state": torch.as_tensor(self.states_norm[real_idx], dtype=torch.float32),
            "action": torch.as_tensor(self.action_chunks[real_idx], dtype=torch.float32),
            "reward": torch.as_tensor([float(self.n_step_returns[real_idx])], dtype=torch.float32),
            "next_state": torch.as_tensor(self.n_step_next_states_norm[real_idx], dtype=torch.float32),
            "done": torch.as_tensor([float(self.n_step_dones[real_idx])], dtype=torch.float32),
        }

    def get_dataloader(self, batch_size: int = 256, shuffle: bool = True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def get_all_data(self):
        """Return all data as tensors (for batch Q-separation evaluation).

        When chunk_size > 1, 'actions' are flattened action chunks.
        """
        return {
            "states": torch.FloatTensor(self.states_norm),
            "actions": torch.FloatTensor(self.action_chunks),
            "rewards": torch.FloatTensor(self.n_step_returns).unsqueeze(1),
            "next_states": torch.FloatTensor(self.n_step_next_states_norm),
            "dones": torch.FloatTensor(self.n_step_dones).unsqueeze(1),
        }


if __name__ == "__main__":
    dataset = OfflineDataset()
    print(f"\nDataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    for k, v in sample.items():
        print(f"  {k}: shape={v.shape}, value={v}")
