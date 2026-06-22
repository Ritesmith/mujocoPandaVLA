#!/usr/bin/env python3
"""VLA Policy Adapter: wraps SmolVLA as a GRPO-compatible policy interface.

This adapter bridges the gap between SmolVLA (a VLA model that takes
image+state+task and outputs action chunks via flow matching) and the
GRPO trainer (which expects a policy with Gaussian action distributions
that support sampling, log-prob computation, and gradient updates).

Strategy:
  1. Run VLA inference to get a mean action mu
  2. Define a learnable std sigma (start at 0.1)
  3. Sample K actions: a_i = mu + sigma * epsilon_i, epsilon_i ~ N(0, I)
  4. Compute log_prob: log N(a_i | mu, sigma^2 I)
  5. For gradient updates, only update LoRA parameters (affect mu) and sigma

LoRA is applied to the last linear layer of SmolVLA's action head
(action_out_proj) so that gradient updates are parameter-efficient.
"""

import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around an existing nn.Linear layer.

    The original layer is frozen; a low-rank residual (A @ B) * scaling
    is added so that only lora_A and lora_B receive gradients.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8):
        super().__init__()
        self.original = original_linear
        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        d_in = original_linear.in_features
        d_out = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.scaling = 1.0 / rank

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


# ---------------------------------------------------------------------------
# VLA Policy Adapter
# ---------------------------------------------------------------------------

class VLAPolicyAdapter(nn.Module):
    """Wraps SmolVLA as a GRPO-compatible policy.

    Key challenge: SmolVLA is a VLA model that takes (image, state, task)
    and outputs action chunks via flow matching.  We need to:
    1. Sample K action candidates from the VLA's output distribution
    2. Compute log_prob for each action
    3. Support gradient updates via LoRA
    """

    def __init__(self, vla_model_path, obs_dim=16, act_dim=8,
                 lora_rank=8, device='cuda'):
        """
        Args:
            vla_model_path: path to the SmolVLA pretrained model directory.
            obs_dim: flat observation dimension (default 16).
            act_dim: environment action dimension (default 8).
            lora_rank: rank for LoRA adapters.
            device: torch device.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.vla_model_path = vla_model_path

        # ---- Load SmolVLA via LeRobot ----
        self._load_vla(vla_model_path, device)

        # ---- Apply LoRA to the action head's last linear layer ----
        self._apply_lora(lora_rank)

        # ---- Learnable log-std for the Gaussian exploration noise ----
        # Start at sigma = 0.1  =>  log_std = log(0.1) ≈ -2.3026
        self.log_std = nn.Parameter(torch.full((act_dim,), -2.3026))

        # Move everything to device
        self.to(device)

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    def _load_vla(self, model_path, device):
        """Load SmolVLA model, preprocessor and postprocessor."""
        # Ensure lerobot is importable
        lerobot_src = os.path.join(
            os.path.dirname(__file__), 'lerobot', 'src'
        )
        if lerobot_src not in sys.path:
            sys.path.insert(0, lerobot_src)

        # Verify model path exists before attempting to load
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"VLA model directory not found: {model_path}\n"
                "Please check the --vla_model_path argument or download the "
                "SmolVLA base model to the expected location."
            )

        try:
            from lerobot.policies.smolvla import SmolVLAPolicy
            from lerobot.policies import make_pre_post_processors
            from lerobot.policies.utils import populate_queues
            from lerobot.utils.constants import ACTION
        except ImportError as exc:
            raise ImportError(
                f"Cannot import LeRobot SmolVLA: {exc}. "
                "Make sure lerobot is installed or its src is on sys.path."
            ) from exc

        # Store references needed by _vla_infer (gradient-enabled path)
        self._populate_queues = populate_queues
        self._ACTION = ACTION

        logger.info("Loading SmolVLA from %s ...", model_path)
        self.vla_policy = SmolVLAPolicy.from_pretrained(model_path)
        self.vla_policy.eval()

        # Pre/post processors
        self.vla_preprocess, self.vla_postprocess = make_pre_post_processors(
            self.vla_policy.config, model_path
        )

        # Discover image key and state dim from config
        self._vla_img_key = list(self.vla_policy.config.image_features.keys())[0]
        self._vla_state_dim = self.vla_policy.config.input_features[
            "observation.state"
        ].shape[0]

        logger.info(
            "SmolVLA loaded. img_key=%s  state_dim=%d",
            self._vla_img_key, self._vla_state_dim,
        )

    # ------------------------------------------------------------------
    # Internal: LoRA injection
    # ------------------------------------------------------------------

    def _apply_lora(self, rank):
        """Replace action_out_proj with a LoRA-wrapped version."""
        # The action head is at  self.vla_policy.model.action_out_proj
        model = self.vla_policy.model

        target_layer = getattr(model, 'action_out_proj', None)
        if target_layer is None:
            logger.warning(
                "Could not find action_out_proj on VLA model; "
                "LoRA will not be applied."
            )
            return

        if not isinstance(target_layer, nn.Linear):
            logger.warning(
                "action_out_proj is not nn.Linear (type=%s); "
                "LoRA will not be applied.",
                type(target_layer).__name__,
            )
            return

        lora_layer = LoRALinear(target_layer, rank=rank)
        # Replace in-place so the rest of the model sees it
        model.action_out_proj = lora_layer
        logger.info(
            "LoRA (rank=%d) applied to action_out_proj (%d -> %d).",
            rank, target_layer.in_features, target_layer.out_features,
        )

    # ------------------------------------------------------------------
    # Internal: VLA inference helpers
    # ------------------------------------------------------------------

    def _build_vla_input(self, obs, image=None, task="pick up the red block"):
        """Build the observation dict expected by SmolVLA's preprocessor.

        Args:
            obs: flat observation tensor (obs_dim,) — we extract joint
                 positions (7) + gripper (1) from the first 8 dims.
            image: RGB uint8 numpy array (H, W, 3). If None, a black
                   image is created (the VLA will still produce an
                   action based on state + language).
            task: task instruction string.

        Returns:
            processed: dict ready for predict_action_chunk
        """
        from torchvision.transforms import ToTensor

        # --- Image ---
        if image is None:
            # Fallback: 256x256 black image
            image = np.zeros((256, 256, 3), dtype=np.uint8)

        pil_image = __import__("PIL").Image.fromarray(image)
        img_tensor = ToTensor()(pil_image)  # [C, H, W] in [0, 1]

        # --- State ---
        # obs layout: joints(7) + gripper(1) + block_pos(3) + hand_pos(3)
        #             + hand_block_dist(1) + block_target_dist(1) = 16
        # VLA expects state_dim values; we take the first 8 (joints+gripper)
        # and pad/truncate to _vla_state_dim.
        if isinstance(obs, torch.Tensor):
            obs_np = obs.detach().cpu().numpy()
        else:
            obs_np = np.asarray(obs, dtype=np.float32)

        full_state = obs_np[:8].copy()  # 7 joints + 1 gripper
        state = np.zeros(self._vla_state_dim, dtype=np.float32)
        n_copy = min(len(full_state), self._vla_state_dim)
        state[:n_copy] = full_state[:n_copy]
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # --- Build dict ---
        obs_dict = {}
        for key in self.vla_policy.config.image_features.keys():
            obs_dict[key] = img_tensor
        obs_dict["observation.state"] = state_tensor
        obs_dict["task"] = task

        # Preprocess
        processed = self.vla_preprocess(obs_dict)
        return processed

    def _vla_infer(self, obs, image=None, task="pick up the red block"):
        """Run VLA inference and return the mean action (act_dim,).

        Returns:
            mu: (act_dim,) tensor on self.device — the VLA's predicted
                action mapped to the environment's action space.

        Note: This method intentionally does NOT use ``torch.no_grad()`` so
        that gradients can flow through the LoRA-adapted
        ``action_out_proj`` during the GRPO update step.  Callers that only
        need inference (``collect_group``, ``evaluate``) already wrap the
        call in a ``torch.no_grad()`` context.
        """
        processed = self._build_vla_input(obs, image=image, task=task)

        # Call _get_action_chunk directly instead of predict_action_chunk,
        # because the latter is decorated with @torch.no_grad() which would
        # block gradient flow to LoRA params during training.
        self.vla_policy.eval()
        batch = self.vla_policy._prepare_batch(processed)
        self.vla_policy._queues = self._populate_queues(
            self.vla_policy._queues, batch, exclude_keys=[self._ACTION]
        )
        action_chunk = self.vla_policy._get_action_chunk(batch)
        # Postprocess (denormalize)
        action_final = self.vla_postprocess(action_chunk)

        # Extract first action from chunk: (batch, n_action_steps, action_dim)
        vla_action = action_final[0, 0, :].to(device=self.device, dtype=torch.float32)
        # Map 6D VLA action -> 8D env action
        mu = self._vla_action_to_env_action(vla_action)
        return mu

    def _vla_action_to_env_action(self, vla_action):
        """Convert VLA action (6D) to env action (8D).

        SmolVLA outputs 6D actions. We map the first 6 dims to the first
        6 joint velocity deltas, leave joint 7 unchanged (pad 0), and set
        gripper based on the last action dimension.
        """
        n_arm_joints = 7
        action = torch.zeros(self.act_dim, dtype=vla_action.dtype, device=vla_action.device)
        vla_np = vla_action.detach().cpu().numpy() if vla_action.is_cuda else vla_action.detach().numpy()
        n_dims = min(len(vla_np), n_arm_joints)
        action[:n_dims] = vla_action[:n_dims]
        # Use last VLA dim as gripper command if available
        if len(vla_np) > n_arm_joints:
            action[n_arm_joints] = vla_action[n_arm_joints - 1]
        # Clip to [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        return action

    # ------------------------------------------------------------------
    # Public API  (GRPO-compatible)
    # ------------------------------------------------------------------

    def forward(self, obs_tensor):
        """Not used directly — VLA uses image+state, not flat obs."""
        raise NotImplementedError("Use get_action_group instead")

    def get_action(self, obs, image=None, task="pick up the red block",
                   deterministic=False):
        """Get a single action from VLA inference.

        Args:
            obs: flat observation tensor (obs_dim,) or numpy array.
            image: RGB image (H, W, 3) uint8 numpy array.
            task: task instruction string.
            deterministic: if True, return mean action; if False, sample.

        Returns:
            If deterministic: action (act_dim,) tensor.
            If not deterministic: (action, log_prob) tuple.
        """
        mu = self._vla_infer(obs, image=image, task=task)

        if deterministic:
            return mu

        std = torch.exp(self.log_std.clamp(-5, 2))
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def get_action_group(self, obs, K, image=None,
                         task="pick up the red block"):
        """Generate K action candidates with log probabilities.

        Strategy: Run VLA inference once to get mean action mu, then add
        Gaussian noise to create K candidates.  Compute log_prob under a
        Gaussian centered at mu with learnable std.

        Args:
            obs: flat observation tensor (obs_dim,) or numpy array.
            K: number of candidates.
            image: RGB image (H, W, 3) uint8 numpy array.
            task: task instruction string.

        Returns:
            actions: (K, act_dim) tensor
            log_probs: (K,) tensor
        """
        mu = self._vla_infer(obs, image=image, task=task)  # (act_dim,)

        std = torch.exp(self.log_std.clamp(-5, 2))  # (act_dim,)

        # Expand mu to (K, act_dim)
        mu_expanded = mu.unsqueeze(0).expand(K, -1)   # (K, act_dim)
        std_expanded = std.unsqueeze(0).expand(K, -1)  # (K, act_dim)

        # Sample K actions
        dist = torch.distributions.Normal(mu_expanded, std_expanded)
        actions = dist.rsample()                        # (K, act_dim)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # (K,)

        return actions, log_probs

    def get_log_prob(self, obs, actions, image=None,
                     task="pick up the red block"):
        """Compute log probability of actions under current policy.

        Args:
            obs: flat observation tensor (obs_dim,) or numpy array.
            actions: (batch, act_dim) tensor.
            image: RGB image (H, W, 3) uint8 numpy array.
            task: task instruction string.

        Returns:
            log_probs: (batch,) tensor
        """
        mu = self._vla_infer(obs, image=image, task=task)  # (act_dim,)

        std = torch.exp(self.log_std.clamp(-5, 2))  # (act_dim,)

        # Expand mu/std to match batch
        batch_size = actions.shape[0]
        mu_expanded = mu.unsqueeze(0).expand(batch_size, -1)
        std_expanded = std.unsqueeze(0).expand(batch_size, -1)

        dist = torch.distributions.Normal(mu_expanded, std_expanded)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # (batch,)

        return log_probs

    def get_lora_params(self):
        """Return LoRA + log_std parameters for optimizer.

        Only LoRA adapters and the learnable log_std should be updated;
        the frozen VLA backbone is excluded to keep the optimizer memory
        footprint small and training parameter-efficient.
        """
        return [
            p for n, p in self.named_parameters()
            if 'lora' in n.lower() or n == 'log_std'
        ]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def save_lora(self, path):
        """Save only the LoRA + log_std parameters."""
        state = {
            name: param.data
            for name, param in self.named_parameters()
            if 'lora' in name.lower() or name == 'log_std'
        }
        torch.save(state, path)
        logger.info("LoRA params saved to %s", path)

    def load_lora(self, path):
        """Load LoRA + log_std parameters from a checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        for name, param in self.named_parameters():
            if name in state:
                param.data.copy_(state[name])
        logger.info("LoRA params loaded from %s", path)
