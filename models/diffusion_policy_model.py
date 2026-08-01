#!/usr/bin/env python3
"""Diffusion Policy model: ConditionalUNet1D + DiffusionPlacePolicy wrapper.

ConditionalUNet1D implements a 1D U-Net with FiLM (Feature-wise Linear
Modulation) conditioning, following the community-standard architecture
from Chi et al. (diffusion_policy repo). The diffusers library's UNet1DModel
lacks FiLM conditioning for arbitrary condition vectors, so a custom
implementation is required.

DiffusionPlacePolicy wraps the U-Net to match the place_model.predict()
interface expected by HierarchicalPickPlacePolicy. It maintains an action
chunk buffer and uses receding-horizon execution: predict 16 steps, execute
8, then re-plan.

Architecture:
    down_dims = [512, 1024, 2048]  (community standard)
    diffusion_step_embed_dim = 128
    kernel_size = 5, GroupNorm groups = 8
    FiLM: condition -> MLP -> per-channel scale/shift after GroupNorm
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (standard DDPM timestep encoding)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, num_train_timesteps)
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ---------------------------------------------------------------------------
# FiLM conditioning
# ---------------------------------------------------------------------------

class FiLMModulation(nn.Module):
    """Feature-wise Linear Modulation: condition -> per-channel scale & shift.

    Given a global condition vector, produces scale (gamma) and shift (beta)
    of shape (B, C, 1) that modulate a (B, C, T) feature map via:
        out = gamma * x + beta

    Uses a bottleneck MLP (cond_dim -> 256 -> 2*channels) to keep parameter
    count manageable at high channel dimensions (2048). Without the bottleneck,
    a single FiLM layer at 2048 channels would add ~21M params; with the
    bottleneck it's ~1.3M.
    """

    def __init__(self, cond_dim: int, channels: int, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        gamma_beta = self.mlp(cond)  # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, C)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# U-Net blocks
# ---------------------------------------------------------------------------

class DownBlock1D(nn.Module):
    """Down-sampling block: 2x (Conv1D + GroupNorm + FiLM + SiLU) + downsample."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(n_groups, out_channels), out_channels)
        self.film1 = FiLMModulation(cond_dim, out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(min(n_groups, out_channels), out_channels)
        self.film2 = FiLMModulation(cond_dim, out_channels)

        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                                     stride=2, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        # x: (B, C_in, T), cond: (B, cond_dim)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film1(h, cond)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, cond)
        h = F.silu(h)

        skip = h  # save for up-block skip connection (before downsample)
        h = self.downsample(h)
        return h, skip


class MidBlock1D(nn.Module):
    """Mid block: 2x (Conv1D + GroupNorm + FiLM + SiLU), no resampling."""

    def __init__(self, channels: int, cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(n_groups, channels), channels)
        self.film1 = FiLMModulation(cond_dim, channels)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(min(n_groups, channels), channels)
        self.film2 = FiLMModulation(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film1(h, cond)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, cond)
        h = F.silu(h)
        return h


class UpBlock1D(nn.Module):
    """Up-sampling block: upsample + concat skip + 2x (Conv1D + GroupNorm + FiLM + SiLU).

    Parameters
    ----------
    in_channels : int
        Channels of the input tensor x (before upsampling, before skip concat).
    skip_channels : int
        Channels of the skip connection from the corresponding down block.
    out_channels : int
        Output channels after this block.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 cond_dim: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2

        # Upsample: keeps in_channels
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4,
                                            stride=2, padding=1)

        # After concat: in_channels + skip_channels
        concat_channels = in_channels + skip_channels
        self.conv1 = nn.Conv1d(concat_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(min(n_groups, out_channels), out_channels)
        self.film1 = FiLMModulation(cond_dim, out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(min(n_groups, out_channels), out_channels)
        self.film2 = FiLMModulation(cond_dim, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor):
        # x: (B, in_channels, T/2), skip: (B, skip_channels, T)
        h = self.upsample(x)  # (B, in_channels, T)
        # Ensure h and skip have the same length (may differ by 1 due to rounding)
        if h.shape[-1] != skip.shape[-1]:
            h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
        h = torch.cat([h, skip], dim=1)  # (B, in_channels + skip_channels, T)

        h = self.conv1(h)
        h = self.norm1(h)
        h = self.film1(h, cond)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, cond)
        h = F.silu(h)
        return h


# ---------------------------------------------------------------------------
# Full Conditional U-Net 1D
# ---------------------------------------------------------------------------

class ConditionalUNet1D(nn.Module):
    """Conditional 1D U-Net for diffusion policy action generation.

    Parameters
    ----------
    input_dim : int
        Action dimension (8 for this task).
    down_dims : list[int]
        Channel dimensions for each down/up block. Community standard:
        [512, 1024, 2048].
    T_pred : int
        Action chunk length (16).
    cond_dim : int
        Condition vector dimension (524 = 512 image + 12 state features).
    diffusion_step_embed_dim : int
        Timestep embedding dimension (128, explicit per spec).
    kernel_size : int
        Conv1D kernel size (5).
    n_groups : int
        GroupNorm groups (8).

    Forward
    -------
    sample : (B, T_pred, input_dim)  — noisy action chunk
    timestep : (B,)                   — diffusion timestep
    condition : (B, T_obs, cond_dim)  — observation feature window

    Returns
    -------
    (B, T_pred, input_dim) — predicted epsilon (noise)
    """

    def __init__(self,
                 input_dim: int = 8,
                 down_dims: list = [512, 1024, 2048],
                 T_pred: int = 16,
                 cond_dim: int = 524,
                 diffusion_step_embed_dim: int = 128,
                 kernel_size: int = 5,
                 n_groups: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.T_pred = T_pred
        self.cond_dim = cond_dim

        # --- Timestep embedding ---
        self.time_embed = SinusoidalPosEmb(diffusion_step_embed_dim)
        time_embed_dim = diffusion_step_embed_dim * 4  # 128 -> 512 after MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # --- Condition embedding ---
        # Pool (B, T_obs, cond_dim) -> (B, cond_dim), then embed to time_embed_dim
        cond_embed_dim = time_embed_dim  # match time embedding for symmetry
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )

        # Global condition = time_embed + cond_embed
        global_cond_dim = time_embed_dim + cond_embed_dim  # 1024

        # --- Input projection ---
        # (B, T_pred, input_dim) -> (B, down_dims[0], T_pred)
        self.input_proj = nn.Conv1d(input_dim, down_dims[0], kernel_size=1)

        # --- Down blocks ---
        self.down_blocks = nn.ModuleList()
        in_ch = down_dims[0]
        for out_ch in down_dims:
            self.down_blocks.append(
                DownBlock1D(in_ch, out_ch, global_cond_dim, kernel_size, n_groups)
            )
            in_ch = out_ch

        # --- Mid block ---
        self.mid_block = MidBlock1D(down_dims[-1], global_cond_dim, kernel_size, n_groups)

        # --- Up blocks (reverse order) ---
        self.up_blocks = nn.ModuleList()
        # For each up block:
        #   in_channels  = current_ch (output of mid block or previous up block)
        #   skip_channels = output channels of the corresponding down block
        #   out_channels  = new output channels
        reversed_dims = list(reversed(down_dims))  # [2048, 1024, 512]
        current_ch = down_dims[-1]  # 2048 (mid block output channels)
        for i, out_ch in enumerate(reversed_dims):
            skip_ch = reversed_dims[i]  # skip from corresponding down block
            self.up_blocks.append(
                UpBlock1D(current_ch, skip_ch, out_ch, global_cond_dim,
                          kernel_size, n_groups)
            )
            current_ch = out_ch

        # --- Output projection ---
        # (B, down_dims[0], T_pred) -> (B, input_dim, T_pred)
        self.output_proj = nn.Sequential(
            nn.Conv1d(down_dims[0], down_dims[0], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(down_dims[0], input_dim, kernel_size=1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        # sample: (B, T_pred, input_dim)
        # timestep: (B,)
        # condition: (B, T_obs, cond_dim)

        B = sample.shape[0]

        # --- Embed timestep ---
        t_emb = self.time_embed(timestep)  # (B, diffusion_step_embed_dim)
        t_emb = self.time_mlp(t_emb)       # (B, time_embed_dim=512)

        # --- Embed condition (pool over T_obs) ---
        cond = condition.mean(dim=1)  # (B, cond_dim=524) — mean pool over T_obs=2
        cond = self.cond_mlp(cond)    # (B, cond_embed_dim=512)

        # --- Global condition ---
        global_cond = torch.cat([t_emb, cond], dim=-1)  # (B, 1024)

        # --- Input projection ---
        # (B, T_pred, input_dim) -> (B, input_dim, T_pred) -> (B, base_dim, T_pred)
        h = sample.permute(0, 2, 1)  # (B, input_dim, T_pred)
        h = self.input_proj(h)       # (B, down_dims[0], T_pred)

        # --- Down blocks ---
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, global_cond)
            skips.append(skip)

        # --- Mid block ---
        h = self.mid_block(h, global_cond)

        # --- Up blocks ---
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            h = up_block(h, skip, global_cond)

        # --- Output projection ---
        h = self.output_proj(h)  # (B, input_dim, T_pred)
        h = h.permute(0, 2, 1)   # (B, T_pred, input_dim)

        return h


# ---------------------------------------------------------------------------
# Policy wrapper for hierarchical evaluation
# ---------------------------------------------------------------------------

class DiffusionPlacePolicy:
    """Wraps ConditionalUNet1D to match the place_model.predict() interface.

    Maintains an action chunk buffer. When empty, runs DDIM sampling to
    produce a T_pred-step action chunk, then returns one action at a time.
    Re-plans every T_exec steps (receding horizon).

    This class is plug-in compatible with HierarchicalPickPlacePolicy:
    policy = HierarchicalPickPlacePolicy(grasp_model, diffusion_policy)
    """

    def __init__(self,
                 unet: ConditionalUNet1D,
                 ddim_scheduler,
                 feature_extractor,
                 vecnorm,
                 action_min: np.ndarray,
                 action_max: np.ndarray,
                 T_obs: int = 2,
                 T_pred: int = 16,
                 T_exec: int = 8,
                 num_inference_steps: int = 20,
                 device: str = "cuda"):
        self.unet = unet
        self.unet.eval()
        self.ddim = ddim_scheduler
        self.feat_extractor = feature_extractor
        self.vecnorm = vecnorm
        self.action_min = action_min
        self.action_max = action_max
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.T_exec = T_exec
        self.num_inference_steps = num_inference_steps
        self.device = device

        # Set DDIM timesteps
        self.ddim.set_timesteps(num_inference_steps)

        # Action buffer + observation history
        self._buffer = deque()
        self._obs_history = deque(maxlen=T_obs)
        self._steps_since_replan = 0

    def reset(self):
        """Reset at the start of a new episode."""
        self._buffer.clear()
        self._obs_history.clear()
        self._steps_since_replan = 0

    @torch.no_grad()
    def _sample_action_chunk(self, obs_cond: torch.Tensor) -> np.ndarray:
        """Run DDIM sampling to produce a T_pred-step action chunk.

        Parameters
        ----------
        obs_cond : (1, T_obs, cond_dim) tensor
            Observation feature window.

        Returns
        -------
        (T_pred, action_dim) numpy array, denormalized to original range.
        """
        B = 1
        shape = (B, self.T_pred, self.unet.input_dim)
        # Start from pure noise
        sample = torch.randn(shape, device=self.device, dtype=torch.float32)

        obs_cond = obs_cond.to(self.device)

        # DDIM denoising loop
        for t in self.ddim.timesteps:
            timestep = torch.full((B,), t, device=self.device, dtype=torch.long)
            noise_pred = self.unet(sample, timestep.float(), obs_cond)
            sample = self.ddim.step(noise_pred, t, sample).prev_sample

        # sample: (1, T_pred, action_dim) in [-1, 1] (clip_sample=True enforces this)
        actions = sample[0].cpu().numpy()  # (T_pred, action_dim)

        # Denormalize from [-1, 1] to original range
        # action = (normalized + 1) / 2 * (max - min) + min
        actions = (actions + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min
        return actions

    def _extract_features(self, obs: dict) -> torch.Tensor:
        """Extract (1, 524) features from raw obs dict.

        obs contains normalized image (1, C, H, W) and normalized state (1, 12).
        """
        # obs is already VecNormalize-normalized
        image = torch.as_tensor(obs["image"], dtype=torch.float32, device=self.device)
        state = torch.as_tensor(obs["state"], dtype=torch.float32, device=self.device)

        # Features extractor expects {"image": (B, C, H, W), "state": (B, 12)}
        with torch.no_grad():
            features = self.feat_extractor({"image": image, "state": state})  # (1, 524)
        return features

    def predict(self, obs, deterministic=True, info=None):
        """Predict next action. Matches SB3 model.predict() interface.

        Parameters
        ----------
        obs : dict
            VecNormalize-normalized observation with "image" (1, C, H, W)
            and "state" (1, 12).
        deterministic : bool
            Ignored (diffusion sampling is inherently stochastic, but with
            fixed seed it's deterministic).
        info : dict, optional
            Unused, for interface compatibility.

        Returns
        -------
        (action, None) : tuple
            action is (1, action_dim) numpy array in original action range.
        """
        # Extract features from current observation
        features = self._extract_features(obs)  # (1, 524)
        self._obs_history.append(features)  # deque(maxlen=T_obs)

        # If buffer empty, sample a new action chunk
        if len(self._buffer) == 0:
            # Build condition: (1, T_obs, 524)
            # Pad with zeros if history < T_obs
            while len(self._obs_history) < self.T_obs:
                self._obs_history.appendleft(torch.zeros_like(features))

            obs_cond = torch.stack(list(self._obs_history), dim=1)  # (1, T_obs, 524)

            # Sample action chunk
            action_chunk = self._sample_action_chunk(obs_cond)  # (T_pred, action_dim)

            # Push T_pred actions to buffer
            for a in action_chunk:
                self._buffer.append(a)

            self._steps_since_replan = 0

        # Pop next action
        action = self._buffer.popleft()
        self._steps_since_replan += 1

        # Force replan after T_exec steps (receding horizon)
        if self._steps_since_replan >= self.T_exec:
            self._buffer.clear()

        return action[np.newaxis, :].astype(np.float32), None


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Smoke test: ConditionalUNet1D forward pass")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConditionalUNet1D(
        input_dim=8,
        down_dims=[512, 1024, 2048],
        T_pred=16,
        cond_dim=524,
        diffusion_step_embed_dim=128,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Dummy forward pass
    B = 4
    sample = torch.randn(B, 16, 8, device=device)
    timestep = torch.randint(0, 100, (B,), device=device).float()
    condition = torch.randn(B, 2, 524, device=device)

    import time
    t0 = time.time()
    output = model(sample, timestep, condition)
    elapsed = time.time() - t0
    print(f"Input:    sample={sample.shape}, timestep={timestep.shape}, condition={condition.shape}")
    print(f"Output:   {output.shape}")
    print(f"Forward time: {elapsed*1000:.1f}ms")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Check for NaN
    if torch.isnan(output).any():
        print("ERROR: NaN detected in output!")
    else:
        print("No NaN detected. Smoke test PASSED.")

    # Memory check
    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"Peak GPU memory: {mem_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
