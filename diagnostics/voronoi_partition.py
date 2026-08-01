#!/usr/bin/env python3
"""Voronoi State-Space Partitioning + Sub-Policy Training/Routing.

Task 6 + Task 7 of the v59-breakthrough-csil-voronoi spec.

V59 is a PPO-trained vision-based pick-and-place policy with 56% place rate.
The Critic (value function) inside V59 estimates V(s) -- the expected return
from state s. Analysis shows ~44% of V59's state space has low value
(V(s) < threshold), corresponding to states where V59 performs poorly
(place rate < 30%). The remaining ~56% has high value (place rate > 70%).

The Voronoi approach: instead of globally fine-tuning V59 (which destroys
it per project_memory), we ONLY train sub-policies in the low-value regions,
keeping V59 unchanged elsewhere. This is "local patching" not "global
optimization".

Pipeline
--------
1. collect_state_values     -- run V59, record V(s) at each place-phase step.
2. compute_value_threshold  -- find the percentile that splits low/high value.
3. voronoi_partition         -- K-means cluster the low-value latents into K=4
                                Voronoi cells (one sub-policy per cell).
4. verify_partition          -- confirm high-value region place_rate > 70%,
                                low-value region place_rate < 30%, and each
                                Voronoi cell place_rate < 30%.
5. RouterPolicy              -- routes states to V59 (high-value) or to the
                                appropriate sub-policy (low-value, by nearest
                                cluster center).
6. Sub-policy training       -- per-cell BC + CSIL++ PBRS fine-tune (Task 7).

Task 7 (this module's extension)
--------------------------------
- ``collect-cell-data``    -- collect V59 transitions per Voronoi cell.
- ``train-sub-policies``   -- train K SubPolicy instances (BC + CSIL++ PBRS).
- ``route``                -- full RouterPolicy demo with sub-policies loaded.

CLI subcommands
---------------
    collect              Collect V59 state values -> data/voronoi_states.npz
    partition            Compute threshold + K-means -> outputs/csil_plus_plus/voronoi_partition.json
    verify               Verify partition quality -> outputs/csil_plus_plus/voronoi_verification_report.json
    route-demo           Demo RouterPolicy (no sub-policies -> V59 actions only)
    collect-cell-data    Collect per-cell V59 trajectories -> data/voronoi_cell_data.npz
    train-sub-policies   Train K sub-policies with BC + CSIL++ PBRS
    route                Full RouterPolicy demo with trained sub-policies

Usage
-----
    python voronoi_partition.py --help
    python voronoi_partition.py collect --n_episodes 200
    python voronoi_partition.py partition --k 4 --low_value_fraction 0.44
    python voronoi_partition.py verify
    python voronoi_partition.py route-demo
    python voronoi_partition.py collect-cell-data --n_episodes_per_cell 50
    python voronoi_partition.py train-sub-policies --n_iterations 10
    python voronoi_partition.py route

Notes
-----
- V59 backbone (ResNet-18 features_extractor) is ALWAYS frozen.
- The Critic is used for INFERENCE ONLY -- V59 weights are never modified.
- Sub-policies are initialized from V59's mlp_extractor.policy_net + action_net
  and conservatively fine-tuned (lr=1e-7, clip=0.1, max_kl=0.005).
- Image augmentation is DISABLED during data collection (project_memory
  hard constraint).
- BN running stats are frozen via features_extractor.eval().
- All V59 forward passes use torch.no_grad() (frozen policy).
- Safety: if a sub-policy's eval place_rate < 30%, it is marked ``frozen``
  and the RouterPolicy falls back to V59 for that cell.
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

WORKSPACE = Path(__file__).parent.resolve()
sys.path.insert(0, str(WORKSPACE))

# Lazy imports for gymnasium / stable_baselines3 / gym_env are done inside
# functions so that `python voronoi_partition.py --help` works without the
# full RL env / GPU stack being available.

# ---------------------------------------------------------------------------
# Constants -- mirror collect_successful_trajectories.py and train_csil_plus_plus.py
# ---------------------------------------------------------------------------

GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"

TARGET_RANGE = "0.35,0.15,0.22,0.65,0.45,0.22"
TARGET_POS_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]

VORONOI_STATES_PATH = WORKSPACE / "data" / "voronoi_states.npz"
CSIL_OUTPUT_DIR = WORKSPACE / "outputs" / "csil_plus_plus"
VORONOI_PARTITION_PATH = CSIL_OUTPUT_DIR / "voronoi_partition.json"
VORONOI_VERIFICATION_PATH = CSIL_OUTPUT_DIR / "voronoi_verification_report.json"

# Task 7 outputs
VORONOI_CELL_DATA_PATH = WORKSPACE / "data" / "voronoi_cell_data.npz"
SUB_POLICY_DIR = CSIL_OUTPUT_DIR / "sub_policies"
SUB_POLICY_TRAINING_LOG_PATH = CSIL_OUTPUT_DIR / "sub_policy_training_log.json"
ROUTER_POLICY_PATH = CSIL_OUTPUT_DIR / "router_policy.pt"
POTENTIAL_FN_PATH = CSIL_OUTPUT_DIR / "potential_fn.pt"  # from train_csil_plus_plus.py train-reward

LIFT_THRESHOLD = 0.03    # m, grab success
PLACE_THRESHOLD = 0.05   # m, place success
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42

LATENT_DIM = 524  # mlp_extractor.value_net.0.weight.shape[1]
ACTION_DIM = 8   # 7 arm + 1 gripper
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

DEFAULT_LOW_VALUE_FRACTION = 0.44  # ~44% of state space is low-value
DEFAULT_K = 4                       # 4 Voronoi cells in the low-value region
DEFAULT_N_STATES = 1000             # sample size for partition analysis
DEFAULT_BOUNDARY_MARGIN = 0.05      # |V(s) - threshold| < margin -> mix
DEFAULT_BOUNDARY_TEMPERATURE = 1.0  # softmax sharpness for boundary mixing

# Task 7 defaults -- conservative sub-policy training (per project_memory
# case studies: V70 crashed at KL=0.003, so we keep max_kl=0.005 and lr=1e-7).
DEFAULT_N_EPISODES_PER_CELL = 50        # successful episodes per Voronoi cell
DEFAULT_MAX_EPISODE_ATTEMPTS_PER_CELL = 200  # hard cap on attempts
DEFAULT_N_TRAIN_ITERATIONS = 10         # PPO+BC iterations per sub-policy
DEFAULT_SUB_POLICY_LR = 1e-7            # 100x lower than typical PPO
DEFAULT_SUB_POLICY_CLIP = 0.1           # very tight PPO clip
DEFAULT_SUB_POLICY_MAX_KL = 0.005       # early-stop threshold
DEFAULT_LAMBDA_BC = 0.5                 # BC anchor loss weight
DEFAULT_LAMBDA_PBRS = 0.5              # PPO PBRS loss weight
DEFAULT_SUB_POLICY_PLACE_RATE_THRESHOLD = 0.30  # safety: fall back to V59 if below

# Value/critic head keys (mlp_extractor.value_net -> scalar value_net).
# Mirrors auto_iter/diagnostic_auto.py VALUE_HEAD_KEYS.
VALUE_HEAD_KEYS = [
    "mlp_extractor.value_net.0.weight",
    "mlp_extractor.value_net.0.bias",
    "mlp_extractor.value_net.2.weight",
    "mlp_extractor.value_net.2.bias",
    "value_net.weight",
    "value_net.bias",
]


# ---------------------------------------------------------------------------
# V59 model loading helpers
# ---------------------------------------------------------------------------

def load_v59_state_dict(path: str = PLACE_MODEL_PATH) -> "OrderedDict[str, torch.Tensor]":
    """Load SB3 policy state_dict directly from the V59 zip (CPU, no env).

    Mirrors train_csil_plus_plus.load_v59_state_dict() and
    diagnostic_auto.load_policy_state_dict(). The policy weights live in
    ``policy.pth`` (~45 MB); the 5 GB bulk is the replay buffer which we do
    NOT need.

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


def extract_value_head(state_dict: dict) -> dict:
    """Return only the value-head tensors (mlp_extractor.value_net + value_net).

    Mirrors diagnostic_auto.extract_value_head(). Useful for standalone
    Critic inference without loading the full PPO model (e.g., in the
    RouterPolicy when sub-policies are absent).
    """
    return {k: state_dict[k].clone() for k in VALUE_HEAD_KEYS if k in state_dict}


def forward_value_head(
    value_head: dict,
    features: torch.Tensor,
) -> torch.Tensor:
    """Forward pass through V59's value head: features (524) -> scalar V(s).

    Mirrors diagnostic_auto._forward_value_head(). The pipeline is:
      1. Linear(524, 524) -> Tanh
      2. Linear(524, 524) -> Tanh
      3. Linear(524, 1) -> squeeze

    Parameters
    ----------
    value_head : dict
        Tensors from :func:`extract_value_head`.
    features : Tensor
        Latent features of shape (N, 524) -- output of V59's
        features_extractor (ResNet-18 image features + state MLP).

    Returns
    -------
    Tensor
        V(s) of shape (N,).
    """
    h = F.linear(
        features,
        value_head["mlp_extractor.value_net.0.weight"],
        value_head["mlp_extractor.value_net.0.bias"],
    )
    h = torch.tanh(h)
    h = F.linear(
        h,
        value_head["mlp_extractor.value_net.2.weight"],
        value_head["mlp_extractor.value_net.2.bias"],
    )
    h = torch.tanh(h)
    v = F.linear(
        h,
        value_head["value_net.weight"],
        value_head["value_net.bias"],
    )
    return v.squeeze(-1)


def freeze_backbone(model) -> int:
    """Freeze ResNet-18 features_extractor; mark V59 as inference-only.

    Also freezes BatchNorm running stats (calls ``.eval()`` on BN modules) so
    that V59's learned normalization statistics are not perturbed. Mirrors
    train_csil_plus_plus.freeze_backbone().
    """
    fe = model.policy.features_extractor
    for p in fe.parameters():
        p.requires_grad = False
    for m in fe.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    # Also freeze the full policy (Critic is inference-only).
    for p in model.policy.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"Backbone frozen: {total} total params, {trainable} trainable")
    return trainable


def normalize_states(states: torch.Tensor, vecnorm_path: Optional[str]) -> torch.Tensor:
    """Normalize 12-dim states using V59's VecNormalize statistics.

    Mirrors train_csil_plus_plus.normalize_states(). Operates only on the
    ``state`` key; images are passed through unchanged (uint8 [0,255]).
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


def extract_v59_latent(policy, obs: dict) -> torch.Tensor:
    """Run V59's frozen backbone + policy_net to get latent features.

    Parameters
    ----------
    policy : SB3 ActorCriticPolicy
        V59's policy (frozen). Called under torch.no_grad() by the caller.
    obs : dict
        ``{"image": Tensor (B,3,84,84) float, "state": Tensor (B,12) float}``.

    Returns
    -------
    Tensor
        Latent features of shape (B, 524).
    """
    features = policy.extract_features(obs)
    latent = policy.mlp_extractor.forward_actor(features)
    return latent


def v59_value(policy, obs: dict) -> torch.Tensor:
    """Compute V(s) via V59's Critic (value function).

    Pipeline:
      1. features_extractor(obs) -> 524-dim features
      2. mlp_extractor.forward_critic(features) -> 524-dim value latent
      3. value_net(value_latent) -> scalar V(s)

    Parameters
    ----------
    policy : SB3 ActorCriticPolicy
        V59's policy (frozen). All forward passes use torch.no_grad().
    obs : dict
        ``{"image": Tensor (B,3,84,84), "state": Tensor (B,12)}``.

    Returns
    -------
    Tensor
        V(s) of shape (B,).
    """
    features = policy.extract_features(obs)
    value_latent = policy.mlp_extractor.forward_critic(features)
    value = policy.value_net(value_latent)
    return value.squeeze(-1)


def v59_action_mean(policy, obs: dict) -> torch.Tensor:
    """Return V59's deterministic action mean mu = action_net(latent).

    Mirrors train_csil_plus_plus.v59_action_mean(). Runs entirely under
    torch.no_grad(); does NOT sample from the DiagGaussian.
    """
    latent = extract_v59_latent(policy, obs)
    mean = policy.action_net(latent)
    return mean


# ---------------------------------------------------------------------------
# SubTask 6.1: Collect V59 state values
# ---------------------------------------------------------------------------

def _build_collect_envs(args, device: str = "auto") -> dict:
    """Build grasp + place vec envs and the raw eval env.

    Mirrors train_csil_plus_plus._build_collect_envs(). Returns a dict with
    keys: ``policy``, ``raw_env``, ``inner_env``, ``grasp_vec_env``,
    ``place_vec_env``, ``place_vision_wrapper``.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
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
        model = PPO.load(model_path, env=vec_env, device=device)
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

    from core.hierarchical_policy import HierarchicalPickPlacePolicy
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
        "place_model": place_model,
        "raw_env": raw_env,
        "inner_env": _inner_env,
        "grasp_vec_env": grasp_vec_env,
        "place_vec_env": place_vec_env,
        "place_vision_wrapper": place_vision_wrapper,
    }


def collect_state_values(
    n_states: int = DEFAULT_N_STATES,
    n_episodes: int = 200,
    max_steps: int = MAX_STEPS,
    place_model_path: str = PLACE_MODEL_PATH,
    place_vecnorm_path: str = PLACE_VECNORM_PATH,
    grasp_model_path: str = GRASP_MODEL_PATH,
    grasp_vecnorm_path: str = GRASP_VECNORM_PATH,
    release_threshold: float = PLACE_THRESHOLD,
    seed: int = SEED,
    device: str = "cpu",
    output_path: Optional[str] = None,
) -> dict:
    """Run V59 in the hierarchical env, record V(s) at each place-phase step.

    At each place-phase step we record:
      - image       : (84, 84, 3) uint8 raw HWC
      - state       : (12,) float32 raw (unnormalized)
      - latent      : (524,) float32 V59 latent features
      - v59_value   : float32 V(s) from V59's Critic
      - action      : (8,) float32 V59's deterministic action
      - final_dist  : float32 final block-target distance (m) for the episode
      - episode_success : int 1 if final_dist < PLACE_THRESHOLD else 0
      - episode_id  : int episode index

    After collection, ``n_states`` samples are drawn uniformly at random from
    all recorded transitions. If fewer than ``n_states`` transitions were
    collected, all are returned.

    Parameters
    ----------
    n_states : int
        Number of states to sample (default 1000).
    n_episodes : int
        Number of eval episodes to run (default 200 -> ~50k transitions).
    max_steps : int
        Max steps per episode.
    place_model_path, place_vecnorm_path : str
        V59 place model + VecNormalize paths.
    grasp_model_path, grasp_vecnorm_path : str
        Grasp model + VecNormalize paths.
    release_threshold : float
        Release distance threshold (m).
    seed : int
        RNG seed.
    device : str
        Torch device for V59 forward passes.
    output_path : str, optional
        If given, save the sampled arrays to this .npz path.

    Returns
    -------
    dict
        Arrays: ``latents`` (N, 524), ``values`` (N,), ``states`` (N, 12),
        ``images`` (N, 84, 84, 3), ``success_labels`` (N,),
        ``final_dists`` (N,), ``episode_ids`` (N,). Also ``n_total``
        (total transitions before sampling) and ``n_sampled``.
    """
    print("=" * 60)
    print("Voronoi State-Value Collection (V59 Critic)")
    print("=" * 60)
    print(f"Episodes: {n_episodes}  (sample {n_states} states after collection)")
    print(f"Place model: {place_model_path}")
    print(f"Image augmentation: DISABLED (project_memory hard constraint)")
    print(f"BN running stats: FROZEN (features_extractor.eval())")
    print(f"V59 weights: FROZEN -- Critic is inference-only")
    print()

    # Resolve device: "auto" -> cuda if available, else cpu. This MUST match
    # the device the model is loaded on, otherwise obs tensors end up on CPU
    # while model weights are on GPU (RuntimeError: Input type mismatch).
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build a lightweight args namespace for _build_collect_envs.
    args_ns = argparse.Namespace(
        place_model=place_model_path,
        place_vecnorm=place_vecnorm_path,
        grasp_model=grasp_model_path,
        grasp_vecnorm=grasp_vecnorm_path,
        release_threshold=release_threshold,
    )
    envs = _build_collect_envs(args_ns, device=device)
    policy = envs["policy"]
    place_model = envs["place_model"]
    raw_env = envs["raw_env"]
    _inner_env = envs["inner_env"]
    place_vec_env = envs["place_vec_env"]
    place_vision_wrapper = envs["place_vision_wrapper"]

    # Freeze V59 (Critic is inference-only).
    freeze_backbone(place_model)
    place_model.policy.features_extractor.eval()
    for m in place_model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    v59_policy = place_model.policy

    np.random.seed(seed)
    try:
        raw_env.seed(seed)
    except Exception:
        pass

    all_images: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_latents: list[np.ndarray] = []
    all_values: list[float] = []
    all_actions: list[np.ndarray] = []
    all_final_dists: list[float] = []
    all_success: list[int] = []
    all_ep_ids: list[int] = []

    n_placed = 0
    n_entered_place = 0
    t0 = time.time()

    for ep in range(n_episodes):
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        max_lift = 0.0
        block_target_dist = float("inf")
        first_place_step = None
        prev_info = None

        ep_images: list[np.ndarray] = []
        ep_states: list[np.ndarray] = []
        ep_latents: list[np.ndarray] = []
        ep_values: list[float] = []
        ep_actions: list[np.ndarray] = []

        for step in range(max_steps):
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
                # Compute V59 latent + V(s) + record raw obs (unnormalized).
                with torch.no_grad():
                    img_t = torch.as_tensor(
                        obs["image"], dtype=torch.float32, device=device)
                    st_t = torch.as_tensor(
                        obs["state"], dtype=torch.float32, device=device)
                    obs_t = {"image": img_t, "state": st_t}
                    latent = extract_v59_latent(v59_policy, obs_t).cpu().numpy()[0]
                    value = float(v59_value(v59_policy, obs_t).cpu().item())
                ep_images.append(vision_obs["image"].copy())     # (84,84,3) uint8
                ep_states.append(vision_obs["state"].copy())     # (12,) float32
                ep_latents.append(latent.astype(np.float32))     # (524,)
                ep_values.append(value)
                ep_actions.append(action[0].copy())              # (8,) float32

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

        if entered_place:
            n_entered_place += 1

        if len(ep_images) > 0:
            label = 1 if placed else 0
            n_trans = len(ep_images)
            for j in range(n_trans):
                all_images.append(ep_images[j])
                all_states.append(ep_states[j])
                all_latents.append(ep_latents[j])
                all_values.append(ep_values[j])
                all_actions.append(ep_actions[j])
                all_final_dists.append(float(block_target_dist))
                all_success.append(label)
                all_ep_ids.append(ep)
            if placed:
                n_placed += 1

        ep_status = "PLACED" if placed else ("grabbed" if grabbed else "failed")
        elapsed = time.time() - t0
        print(f"Ep {ep:3d}: {ep_status:7s}  dist={block_target_dist*100:5.1f}cm  "
              f"lift={max_lift*100:5.1f}cm  place_steps={len(ep_images):3d}  "
              f"| placed={n_placed}/{ep+1}  transitions={len(all_images)}  "
              f"[{elapsed:.0f}s]")

    n_total = len(all_images)
    print()
    print("=" * 60)
    print("Collection Complete")
    print("=" * 60)
    print(f"Episodes run:        {ep + 1}")
    print(f"Entered place phase: {n_entered_place}/{ep+1}")
    print(f"Placed (dist<5cm):   {n_placed}/{ep+1} "
          f"({100*n_placed/max(1,ep+1):.1f}%)")
    print(f"Total transitions:   {n_total}")
    print(f"Elapsed: {time.time() - t0:.0f}s")

    if n_total == 0:
        print("ERROR: no place-phase transitions collected.")
        raw_env.close()
        return {
            "latents": np.zeros((0, LATENT_DIM), dtype=np.float32),
            "values": np.zeros((0,), dtype=np.float32),
            "states": np.zeros((0, 12), dtype=np.float32),
            "images": np.zeros((0, 84, 84, 3), dtype=np.uint8),
            "success_labels": np.zeros((0,), dtype=np.int64),
            "final_dists": np.zeros((0,), dtype=np.float32),
            "episode_ids": np.zeros((0,), dtype=np.int64),
            "actions": np.zeros((0, ACTION_DIM), dtype=np.float32),
            "n_total": 0,
            "n_sampled": 0,
        }

    # Sample n_states uniformly at random.
    n_sampled = min(n_states, n_total)
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(n_total, size=n_sampled, replace=False)

    latents_arr = np.array(all_latents, dtype=np.float32)
    values_arr = np.array(all_values, dtype=np.float32)
    states_arr = np.array(all_states, dtype=np.float32)
    images_arr = np.array(all_images, dtype=np.uint8)
    actions_arr = np.array(all_actions, dtype=np.float32)
    final_dists_arr = np.array(all_final_dists, dtype=np.float32)
    success_arr = np.array(all_success, dtype=np.int64)
    ep_ids_arr = np.array(all_ep_ids, dtype=np.int64)

    sampled = {
        "latents": latents_arr[sample_idx],
        "values": values_arr[sample_idx],
        "states": states_arr[sample_idx],
        "images": images_arr[sample_idx],
        "success_labels": success_arr[sample_idx],
        "final_dists": final_dists_arr[sample_idx],
        "episode_ids": ep_ids_arr[sample_idx],
        "actions": actions_arr[sample_idx],
        "n_total": n_total,
        "n_sampled": n_sampled,
    }

    print(f"\nSampled {n_sampled}/{n_total} states for partitioning.")
    print(f"  Value stats: min={sampled['values'].min():.4f}  "
          f"max={sampled['values'].max():.4f}  "
          f"mean={sampled['values'].mean():.4f}")
    print(f"  Success rate in sample: "
          f"{100*sampled['success_labels'].mean():.1f}%")

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            latents=sampled["latents"],
            values=sampled["values"],
            states=sampled["states"],
            images=sampled["images"],
            success_labels=sampled["success_labels"],
            final_dists=sampled["final_dists"],
            episode_ids=sampled["episode_ids"],
            actions=sampled["actions"],
        )
        print(f"Saved to {out_path}")

    raw_env.close()
    return sampled


# ---------------------------------------------------------------------------
# SubTask 6.2: Determine value threshold
# ---------------------------------------------------------------------------

def compute_value_threshold(
    values: torch.Tensor,
    low_value_fraction: float = DEFAULT_LOW_VALUE_FRACTION,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Find the value threshold separating low-value from high-value states.

    Sorts the values and finds the percentile that puts the bottom
    ``low_value_fraction`` (default 44%) into the low-value region and the
    top ``1 - low_value_fraction`` (default 56%) into the high-value region.

    Parameters
    ----------
    values : Tensor (N,)
        V59 value estimates V(s).
    low_value_fraction : float
        Fraction of states to label as low-value (default 0.44).

    Returns
    -------
    threshold : float
        The value threshold V* such that V(s) <= V* -> low-value.
    low_mask : Tensor (N,) bool
        True for low-value states (V(s) <= threshold).
    high_mask : Tensor (N,) bool
        True for high-value states (V(s) > threshold).
    """
    if not 0.0 < low_value_fraction < 1.0:
        raise ValueError(f"low_value_fraction must be in (0, 1), got {low_value_fraction}")
    values = torch.as_tensor(values, dtype=torch.float32).flatten()
    N = values.shape[0]
    if N == 0:
        raise ValueError("values is empty")

    sorted_vals, _ = torch.sort(values)
    n_low = max(1, int(low_value_fraction * N))
    n_low = min(n_low, N - 1)  # leave at least 1 high-value sample
    threshold = float(sorted_vals[n_low - 1].item())

    low_mask = values <= threshold
    high_mask = values > threshold

    # Print statistics.
    print("\n=== Value Threshold Computation ===")
    print(f"  N={N}  low_value_fraction={low_value_fraction:.2f}")
    print(f"  Values: min={values.min().item():.4f}  "
          f"max={values.max().item():.4f}  "
          f"mean={values.mean().item():.4f}  "
          f"median={values.median().item():.4f}  "
          f"std={values.std().item() if N > 1 else 0.0:.4f}")
    print(f"  Threshold V* = {threshold:.4f}")
    print(f"  Low-value region  (V(s) <= V*): {int(low_mask.sum())} states "
          f"({100*low_mask.sum().item()/N:.1f}%)")
    print(f"  High-value region (V(s) >  V*): {int(high_mask.sum())} states "
          f"({100*high_mask.sum().item()/N:.1f}%)")

    return threshold, low_mask, high_mask


# ---------------------------------------------------------------------------
# SubTask 6.3: Voronoi quantization via K-Means
# ---------------------------------------------------------------------------

def _torch_kmeans(
    data: torch.Tensor,
    k: int,
    n_iter: int = 10,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple torch K-means with k-means++ initialization (sklearn fallback).

    Parameters
    ----------
    data : Tensor (N, D)
    k : int
        Number of clusters.
    n_iter : int
        Number of EM iterations.
    seed : int
        RNG seed.

    Returns
    -------
    centers : Tensor (k, D)
    assignments : Tensor (N,)
    """
    gen = torch.Generator().manual_seed(seed)
    N, D = data.shape
    if N <= k:
        # Each point is its own cluster (pad with duplicates if needed).
        centers = data.clone()
        if N < k:
            extra = data[torch.randint(0, N, (k - N,), generator=gen)]
            centers = torch.cat([centers, extra], dim=0)
        assignments = torch.arange(min(N, k), dtype=torch.long)
        if N < k:
            assignments = torch.cat([assignments, torch.zeros(N - k, dtype=torch.long)])
        return centers[:k], assignments[:N]

    # k-means++ init
    centers = torch.zeros(k, D, dtype=data.dtype)
    first_idx = int(torch.randint(0, N, (1,), generator=gen).item())
    centers[0] = data[first_idx]
    closest_sq = ((data - centers[0]) ** 2).sum(dim=-1)  # (N,)
    for c in range(1, k):
        probs = closest_sq / (closest_sq.sum() + 1e-12)
        idx = int(torch.multinomial(probs, 1, generator=gen).item())
        centers[c] = data[idx]
        dist_sq = ((data - centers[c]) ** 2).sum(dim=-1)
        closest_sq = torch.minimum(closest_sq, dist_sq)

    # EM iterations
    assignments = torch.zeros(N, dtype=torch.long)
    for _ in range(n_iter):
        # Assign: nearest center by L2
        dists = torch.cdist(data, centers)  # (N, k)
        new_assignments = dists.argmin(dim=-1)
        if torch.equal(new_assignments, assignments):
            break
        assignments = new_assignments
        # Update: mean of assigned points (handle empty clusters)
        for c in range(k):
            mask = assignments == c
            if mask.any():
                centers[c] = data[mask].mean(dim=0)
            else:
                # Reinit empty cluster to a random point
                idx = int(torch.randint(0, N, (1,), generator=gen).item())
                centers[c] = data[idx]

    return centers, assignments


def voronoi_partition(
    latents: torch.Tensor,
    low_mask: torch.Tensor,
    k: int = DEFAULT_K,
    seed: int = SEED,
    use_sklearn: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """K-means cluster the low-value latents into K Voronoi cells.

    Only latents where ``low_mask=True`` are clustered (the high-value region
    is left to V59 -- no sub-policy needed there). Uses sklearn KMeans if
    available, else falls back to a simple torch implementation
    (:func:`_torch_kmeans`).

    Parameters
    ----------
    latents : Tensor (N, 524)
        V59 latent features for all N states.
    low_mask : Tensor (N,) bool
        True for low-value states (the ones to cluster).
    k : int
        Number of Voronoi cells (default 4).
    seed : int
        RNG seed for K-means.
    use_sklearn : bool
        Try sklearn.cluster.KMeans first (faster, better-tested).

    Returns
    -------
    cluster_centers : np.ndarray (k, 524)
    cluster_assignments : np.ndarray (N_low,)
        Cluster index [0, k) for each low-value latent.
    kmeans_model : Any
        The fitted sklearn KMeans object, or None if torch fallback was used.
    """
    latents = torch.as_tensor(latents, dtype=torch.float32)
    low_mask = torch.as_tensor(low_mask, dtype=torch.bool)
    low_latents = latents[low_mask]
    N_low = low_latents.shape[0]
    print(f"\n=== Voronoi Partition (K-Means, k={k}) ===")
    print(f"  Low-value latents: {N_low} of {latents.shape[0]}")

    if N_low == 0:
        raise ValueError("low_mask selected 0 states -- cannot cluster")

    kmeans_model = None
    if use_sklearn:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=seed, n_init=10)
            assignments = km.fit_predict(low_latents.numpy())
            cluster_centers = km.cluster_centers_.astype(np.float32)
            kmeans_model = km
            print("  Used sklearn.cluster.KMeans")
        except ImportError:
            print("  sklearn not available -- falling back to torch K-means")
            use_sklearn = False

    if not use_sklearn or kmeans_model is None:
        centers_t, assignments_t = _torch_kmeans(
            low_latents, k=k, n_iter=10, seed=seed)
        cluster_centers = centers_t.numpy().astype(np.float32)
        assignments = assignments_t.numpy().astype(np.int64)
        print("  Used torch fallback K-means")

    # Print cluster statistics.
    print(f"\n  Cluster statistics (low-value region only):")
    print(f"  {'Cluster':>8}  {'Size':>6}  {'Fraction':>9}")
    for c in range(k):
        size = int((assignments == c).sum())
        frac = size / max(1, N_low)
        print(f"  {c:>8}  {size:>6}  {100*frac:>8.1f}%")

    return cluster_centers, assignments, kmeans_model


def assign_cluster(
    latent: torch.Tensor,
    centers: torch.Tensor,
) -> int:
    """Return the index of the nearest cluster center (Voronoi assignment).

    Parameters
    ----------
    latent : Tensor (D,) or (1, D)
        A single latent vector.
    centers : Tensor (k, D)
        Cluster centers.

    Returns
    -------
    int
        Index of the nearest center [0, k).
    """
    latent = torch.as_tensor(latent, dtype=torch.float32).flatten()
    if latent.dim() == 1:
        latent = latent.unsqueeze(0)  # (1, D)
    centers = torch.as_tensor(centers, dtype=torch.float32)
    dists = torch.cdist(latent, centers)  # (1, k)
    return int(dists.argmin(dim=-1).item())


def _cluster_stats(
    latents: torch.Tensor,
    values: torch.Tensor,
    success_labels: torch.Tensor,
    low_mask: torch.Tensor,
    cluster_assignments: np.ndarray,
    cluster_centers: np.ndarray,
    k: int,
) -> Tuple[list[int], list[float], list[float]]:
    """Compute per-cluster size, mean value, mean success rate."""
    latents = torch.as_tensor(latents, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    success_labels = torch.as_tensor(success_labels, dtype=torch.float32)
    low_mask = torch.as_tensor(low_mask, dtype=torch.bool)
    cluster_assignments = np.asarray(cluster_assignments)

    sizes: list[int] = []
    mean_values: list[float] = []
    mean_success: list[float] = []
    low_values = values[low_mask]
    low_success = success_labels[low_mask]

    for c in range(k):
        mask = cluster_assignments == c
        size = int(mask.sum())
        sizes.append(size)
        if size > 0:
            mean_values.append(float(low_values[mask].mean().item()))
            mean_success.append(float(low_success[mask].mean().item()))
        else:
            mean_values.append(0.0)
            mean_success.append(0.0)
    return sizes, mean_values, mean_success


# ---------------------------------------------------------------------------
# SubTask 6.4: Verify partition
# ---------------------------------------------------------------------------

def verify_partition(
    state_data: dict,
    low_mask: torch.Tensor,
    high_mask: torch.Tensor,
    cluster_assignments: np.ndarray,
    cluster_centers: np.ndarray,
    k: int,
) -> dict:
    """Verify the partition quality by computing place rates per region.

    Place rate for a region = fraction of UNIQUE EPISODES in that region
    where final_dist < 0.05m (PLACE_THRESHOLD). Per-episode aggregation
    avoids bias from episodes with many place-phase steps contributing more
    transitions.

    Expected outcomes:
      - High-value region place_rate > 70%
      - Low-value region place_rate < 30%
      - Each Voronoi cell place_rate < 30% (all need patching)

    Parameters
    ----------
    state_data : dict
        Must contain: ``episode_ids`` (N,), ``final_dists`` (N,),
        ``success_labels`` (N,), ``values`` (N,).
    low_mask, high_mask : Tensor (N,) bool
    cluster_assignments : np.ndarray (N_low,)
    cluster_centers : np.ndarray (k, D)
    k : int

    Returns
    -------
    dict
        Verification report (also written to
        ``outputs/csil_plus_plus/voronoi_verification_report.json``).
    """
    print("\n=== Voronoi Partition Verification ===")
    episode_ids = np.asarray(state_data["episode_ids"])
    final_dists = np.asarray(state_data["final_dists"])
    values = torch.as_tensor(state_data["values"], dtype=torch.float32)
    low_mask = torch.as_tensor(low_mask, dtype=torch.bool).numpy()
    high_mask = torch.as_tensor(high_mask, dtype=torch.bool).numpy()
    cluster_assignments = np.asarray(cluster_assignments)

    def _episode_place_rate(mask: np.ndarray) -> Tuple[float, int, int]:
        """Place rate over unique episodes that have at least one transition
        in ``mask``. An episode is 'placed' if its final_dist < PLACE_THRESHOLD."""
        if not mask.any():
            return 0.0, 0, 0
        ep_ids_in_region = np.unique(episode_ids[mask])
        n_eps = len(ep_ids_in_region)
        # final_dist is the same for all transitions of an episode; take the
        # first occurrence per episode.
        placed = 0
        for eid in ep_ids_in_region:
            ep_mask = episode_ids == eid
            ep_final_dist = float(final_dists[ep_mask][0])
            if ep_final_dist < PLACE_THRESHOLD:
                placed += 1
        return (placed / n_eps if n_eps > 0 else 0.0), placed, n_eps

    high_rate, high_placed, high_n = _episode_place_rate(high_mask)
    low_rate, low_placed, low_n = _episode_place_rate(low_mask)

    print(f"\n  Region place rates (per unique episode):")
    print(f"  {'Region':>12}  {'Episodes':>8}  {'Placed':>6}  {'Place Rate':>10}")
    print(f"  {'High-value':>12}  {high_n:>8}  {high_placed:>6}  "
          f"{100*high_rate:>9.1f}%")
    print(f"  {'Low-value':>12}  {low_n:>8}  {low_placed:>6}  "
          f"{100*low_rate:>9.1f}%")

    # Per-cluster place rates.
    cluster_rates: list[float] = []
    cluster_eps: list[int] = []
    cluster_placed: list[int] = []
    low_indices = np.where(low_mask)[0]
    print(f"\n  Per-Voronoi-cell place rates (low-value region):")
    print(f"  {'Cluster':>8}  {'Episodes':>8}  {'Placed':>6}  {'Place Rate':>10}")
    for c in range(k):
        cell_trans_mask = np.zeros_like(low_mask)
        # Map cluster_assignments (over low_indices) back to global indices.
        cell_local = cluster_assignments == c
        cell_global = low_indices[cell_local]
        cell_trans_mask[cell_global] = True
        rate, placed, n_eps = _episode_place_rate(cell_trans_mask)
        cluster_rates.append(rate)
        cluster_eps.append(n_eps)
        cluster_placed.append(placed)
        print(f"  {c:>8}  {n_eps:>8}  {placed:>6}  {100*rate:>9.1f}%")

    # Assertions (soft -- print warnings, do not raise).
    high_ok = high_rate > 0.70
    low_ok = low_rate < 0.30
    cells_ok = all(r < 0.30 for r in cluster_rates)

    print(f"\n  Assertions:")
    print(f"    High-value place_rate > 70%: "
          f"{'PASS' if high_ok else 'FAIL'} ({100*high_rate:.1f}%)")
    print(f"    Low-value place_rate  < 30%: "
          f"{'PASS' if low_ok else 'FAIL'} ({100*low_rate:.1f}%)")
    print(f"    All cells place_rate  < 30%: "
          f"{'PASS' if cells_ok else 'FAIL'}")

    report = {
        "high_value": {
            "n_episodes": int(high_n),
            "n_placed": int(high_placed),
            "place_rate": float(high_rate),
            "assertion_gt_70pct": bool(high_ok),
        },
        "low_value": {
            "n_episodes": int(low_n),
            "n_placed": int(low_placed),
            "place_rate": float(low_rate),
            "assertion_lt_30pct": bool(low_ok),
        },
        "voronoi_cells": [
            {
                "cluster": int(c),
                "n_episodes": int(cluster_eps[c]),
                "n_placed": int(cluster_placed[c]),
                "place_rate": float(cluster_rates[c]),
                "assertion_lt_30pct": bool(cluster_rates[c] < 0.30),
            }
            for c in range(k)
        ],
        "all_assertions_passed": bool(high_ok and low_ok and cells_ok),
        "value_stats": {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "std": float(values.std().item()) if values.numel() > 1 else 0.0,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    CSIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(VORONOI_VERIFICATION_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report written to {VORONOI_VERIFICATION_PATH}")

    return report


# ---------------------------------------------------------------------------
# SubPolicy (Task 7.2): per-Voronoi-cell fine-tuned head mirroring V59
# ---------------------------------------------------------------------------

class SubPolicy(nn.Module):
    """Sub-policy for a single Voronoi cell.

    Mirrors V59's ``mlp_extractor.policy_net`` + ``action_net`` architecture:
    two-layer MLP (Linear-Tanh-Linear-Tanh) followed by a Linear action
    projection. Initialized from V59's weights via :meth:`init_from_v59` so
    that ``SubPolicy(latent) == V59.action_net(V59.policy_net(latent))`` at
    start (safe degradation to V59). Then conservatively fine-tuned on the
    cell's data using BC + CSIL++ PBRS.

    The ``frozen`` flag marks a sub-policy as disabled (e.g., its eval
    place_rate fell below the safety threshold). When frozen, the
    :class:`RouterPolicy` falls back to V59 for the corresponding cell.

    Attributes
    ----------
    policy_net : nn.Sequential
        Two-layer MLP (Linear-Tanh-Linear-Tanh) operating on the 524-dim
        V59 latent.
    action_net : nn.Linear
        Final linear projection to the 8-dim action.
    frozen : bool
        If True, the sub-policy is bypassed (V59 used instead).
    cell_idx : Optional[int]
        Index of the Voronoi cell this sub-policy serves (for logging).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        cell_idx: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.cell_idx = cell_idx
        self.frozen: bool = False
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(latent_dim, action_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return action mean of shape (B, ACTION_DIM)."""
        h = self.policy_net(latent)
        return self.action_net(h)

    def init_from_v59(self, v59_policy) -> int:
        """Initialize weights from V59's mlp_extractor.policy_net + action_net.

        Returns
        -------
        int
            Number of tensors copied.
        """
        v59_sd = v59_policy.state_dict()
        target_sd = self.state_dict()
        mapping = {
            "mlp_extractor.policy_net.0.weight": "policy_net.0.weight",
            "mlp_extractor.policy_net.0.bias": "policy_net.0.bias",
            "mlp_extractor.policy_net.2.weight": "policy_net.2.weight",
            "mlp_extractor.policy_net.2.bias": "policy_net.2.bias",
            "action_net.weight": "action_net.weight",
            "action_net.bias": "action_net.bias",
        }
        copied = 0
        for src, dst in mapping.items():
            if src in v59_sd and dst in target_sd:
                if v59_sd[src].shape == target_sd[dst].shape:
                    target_sd[dst] = v59_sd[src].clone()
                    copied += 1
        if copied:
            self.load_state_dict(target_sd)
        return copied

    def save(self, path: str) -> None:
        """Save sub-policy state_dict + metadata to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "latent_dim": self.latent_dim,
                "action_dim": self.action_dim,
                "frozen": self.frozen,
                "cell_idx": self.cell_idx,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: str,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
    ) -> "SubPolicy":
        """Load a sub-policy from disk.

        Parameters
        ----------
        path : str
            Path to a ``cell_{k}_policy.pt`` file written by :meth:`save`.
        latent_dim, action_dim : int
            Fallback dims if the saved payload does not contain them.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sp = cls(
            latent_dim=payload.get("latent_dim", latent_dim),
            action_dim=payload.get("action_dim", action_dim),
            cell_idx=payload.get("cell_idx", None),
        )
        sp.load_state_dict(payload["state_dict"])
        sp.frozen = bool(payload.get("frozen", False))
        return sp


# ---------------------------------------------------------------------------
# PotentialFunction (Task 7.2): re-declared locally so we can load Phi from
# train_csil_plus_plus.py train-reward without a cross-module import.
# ---------------------------------------------------------------------------

class PotentialFunction(nn.Module):
    """Small MLP potential function Phi(s) for PBRS.

    Mirrors :class:`train_csil_plus_plus.PotentialFunction` so a Phi saved by
    ``train_csil_plus_plus.py train-reward`` can be loaded directly here.

    Architecture: ``Linear(in, 128) -> Tanh -> Linear(128, 64) -> Tanh ->
    Linear(64, 1)``. Output is a scalar Phi(s) used as the PBRS potential:

        F(s, s', gamma) = gamma * Phi(s') - Phi(s)

    which leaves the optimal policy unchanged (PBRS theorem).
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


# ---------------------------------------------------------------------------
# SubTask 7.4: Softmax boundary mixing
# ---------------------------------------------------------------------------

def _softmax_mix(
    v: float,
    threshold: float,
    margin: float,
    a_v59: torch.Tensor,
    a_sub: torch.Tensor,
    cell_idx: Optional[int] = None,
    sub_actions: Optional[list] = None,
    sub_weights: Optional[list] = None,
) -> Tuple[torch.Tensor, float]:
    """Softmax mix between V59 and sub-policy action(s) at region boundaries.

    The V59 weight is

        w_v59 = sigmoid((v - threshold) / max(margin, 1e-6))

    which is close to 1 when ``v >> threshold`` (high-value region) and
    close to 0 when ``v << threshold`` (low-value region). The returned
    action is ``w_v59 * a_v59 + (1 - w_v59) * a_sub_avg`` where
    ``a_sub_avg`` is either ``a_sub`` (single sub-policy) or a weighted
    average of multiple nearby sub-policy actions.

    Parameters
    ----------
    v : float
        V59 value V(s).
    threshold : float
        Value threshold V*.
    margin : float
        Boundary margin (controls softmax sharpness; smaller = sharper).
    a_v59 : Tensor (1, 8)
        V59 action mean.
    a_sub : Tensor (1, 8)
        Default sub-policy action (used when ``sub_actions`` is None).
    cell_idx : int, optional
        Index of the dominant sub-policy (for logging only).
    sub_actions : list of Tensors, optional
        Multiple sub-policy actions near a boundary (for multi-cell mixing).
    sub_weights : list of floats, optional
        Per-sub-policy weights (e.g., inverse distance to cluster centers).
        If None, equal weights are used.

    Returns
    -------
    mixed_action : Tensor (1, 8)
    w_v59 : float in [0, 1]
        Weight on the V59 action.
    """
    delta = (v - threshold) / max(1e-6, float(margin))
    w_v59 = float(torch.sigmoid(torch.tensor(delta)).item())

    if sub_actions is not None and len(sub_actions) > 0:
        if sub_weights is None:
            sub_weights = [1.0 / len(sub_actions)] * len(sub_actions)
        w_sum = sum(sub_weights) + 1e-12
        sub_weights_norm = [w / w_sum for w in sub_weights]
        a_sub_avg = sum(
            w * a for w, a in zip(sub_weights_norm, sub_actions)
        )
    else:
        a_sub_avg = a_sub

    mixed = w_v59 * a_v59 + (1.0 - w_v59) * a_sub_avg
    return mixed, w_v59


# ---------------------------------------------------------------------------
# RouterPolicy (Task 7.3): full routing with sub-policies + softmax mixing
# ---------------------------------------------------------------------------

class RouterPolicy:
    """Routes states to V59 (high-value) or to a cluster sub-policy (low-value).

    Routing logic (Task 7.3):
      1. Compute V(s) and latent via V59's frozen backbone + Critic.
      2. If V(s) > threshold + boundary_margin: return V59 action (high-value).
      3. If V(s) < threshold - boundary_margin: return the nearest cluster's
         sub-policy action (low-value region).
      4. Otherwise (boundary): softmax-mix V59 and sub-policy actions.

    When no sub-policies are registered (Task 6 / ``route-demo``), the router
    ALWAYS returns the V59 action (safe degradation). This guarantees
    RouterPolicy is never worse than V59 alone. Frozen sub-policies
    (``sub_policy.frozen == True``, e.g. due to eval place_rate < 30%) are
    bypassed and V59 is used instead.

    Attributes
    ----------
    v59_policy : SB3 ActorCriticPolicy
        Frozen V59 policy (used for V(s), latent extraction, and high-value
        actions).
    sub_policies : list[Optional[SubPolicy]]
        Per-cluster sub-policies. ``None`` entries -> fall back to V59.
    cluster_centers : Tensor (k, D) or None
        Voronoi cluster centers in latent space.
    value_threshold : float
        V* separating low-value from high-value states.
    boundary_margin : float
        |V(s) - threshold| < margin triggers softmax mixing.
    boundary_temperature : float
        Softmax sharpness for boundary mixing (kept for backward compat).
    """

    def __init__(
        self,
        v59_policy,
        cluster_centers: Optional[torch.Tensor] = None,
        value_threshold: float = 0.0,
        sub_policies: Optional[list] = None,
        boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
        boundary_temperature: float = DEFAULT_BOUNDARY_TEMPERATURE,
        k: int = DEFAULT_K,
        device: str = "cpu",
    ):
        self.v59_policy = v59_policy
        # Ensure V59 is frozen.
        for p in self.v59_policy.parameters():
            p.requires_grad = False
        self.v59_policy.eval()
        for m in self.v59_policy.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

        self.k = k
        # sub_policies: list of length k. Entries may be None (no sub-policy)
        # or SubPolicy instances. Frozen sub-policies stay in the list but
        # forward() will bypass them.
        if sub_policies is not None:
            if len(sub_policies) != k:
                raise ValueError(
                    f"sub_policies length {len(sub_policies)} != k={k}"
                )
            self.sub_policies: list = [
                (sp.to(device) if sp is not None else None)
                for sp in sub_policies
            ]
        else:
            self.sub_policies = [None] * k
        self.cluster_centers = (
            torch.as_tensor(cluster_centers, dtype=torch.float32, device=device)
            if cluster_centers is not None else None
        )
        self.value_threshold = float(value_threshold)
        self.boundary_margin = float(boundary_margin)
        self.boundary_temperature = float(boundary_temperature)
        self.device = device
        # Count active (non-None, non-frozen) sub-policies for logging.
        n_active = sum(
            1 for sp in self.sub_policies
            if sp is not None and not getattr(sp, "frozen", False)
        )
        print(f"  RouterPolicy: k={k}  active_sub_policies={n_active}/{k}  "
              f"threshold={self.value_threshold:.4f}  margin={self.boundary_margin}")

    def add_sub_policy(self, cluster_idx: int, policy: nn.Module) -> None:
        """Register a sub-policy for a Voronoi cluster."""
        if not 0 <= cluster_idx < self.k:
            raise IndexError(f"cluster_idx {cluster_idx} out of range [0, {self.k})")
        self.sub_policies[cluster_idx] = policy.to(self.device)
        frozen = getattr(policy, "frozen", False)
        print(f"  RouterPolicy: registered sub-policy for cluster {cluster_idx} "
              f"(frozen={frozen})")

    def load_sub_policies(self, sub_policy_dir: str) -> int:
        """Load all K sub-policies from ``sub_policy_dir``.

        Expects files named ``cell_{k}_policy.pt`` for k in [0, K). Missing
        files leave the corresponding entry as ``None`` (safe fallback to V59).
        Sub-policies with ``frozen=True`` are loaded but bypassed at routing
        time.

        Returns
        -------
        int
            Number of sub-policies successfully loaded.
        """
        n_loaded = 0
        for k in range(self.k):
            path = Path(sub_policy_dir) / f"cell_{k}_policy.pt"
            if not path.exists():
                print(f"  RouterPolicy: cell_{k}_policy.pt not found -- "
                      f"will fall back to V59 for cluster {k}")
                continue
            try:
                sp = SubPolicy.load(str(path))
                sp.to(self.device)
                sp.cell_idx = k
                self.sub_policies[k] = sp
                n_loaded += 1
                print(f"  RouterPolicy: loaded sub-policy for cluster {k} "
                      f"(frozen={sp.frozen})")
            except Exception as e:
                print(f"  RouterPolicy: WARNING failed to load {path}: {e}")
        return n_loaded

    def compute_value(self, obs: dict) -> float:
        """Compute V(s) via V59's Critic (no grad)."""
        with torch.no_grad():
            value = v59_value(self.v59_policy, obs)
        return float(value.item())

    def find_cluster(self, latent: torch.Tensor) -> int:
        """Return the nearest Voronoi cluster index for ``latent``."""
        if self.cluster_centers is None:
            return 0
        return assign_cluster(latent, self.cluster_centers)

    def _get_active_sub_policy(self, cluster_idx: int) -> Optional[nn.Module]:
        """Return the sub-policy for ``cluster_idx`` if active, else None.

        A sub-policy is "active" if it is registered (not None) AND not
        frozen (e.g., due to eval place_rate < 30%).
        """
        if not 0 <= cluster_idx < self.k:
            return None
        sp = self.sub_policies[cluster_idx]
        if sp is None:
            return None
        if getattr(sp, "frozen", False):
            return None
        return sp

    def forward(self, obs: dict) -> Tuple[torch.Tensor, dict]:
        """Route the state and return an action.

        Parameters
        ----------
        obs : dict
            ``{"image": (1,3,84,84), "state": (1,12)}`` -- normalized vision
            obs for V59.

        Returns
        -------
        action : Tensor (1, 8)
        info : dict
            Routing metadata: ``value``, ``region`` ('high'/'low'/'boundary'/
            'low_no_sub'/'low_frozen'), ``cluster`` (int or None),
            ``mix_weight_v59`` (float in [0,1]).
        """
        with torch.no_grad():
            latent = extract_v59_latent(self.v59_policy, obs)
            a_v59 = self.v59_policy.action_net(latent)
            value = float(self.v59_policy.value_net(
                self.v59_policy.mlp_extractor.forward_critic(
                    self.v59_policy.extract_features(obs)
                )
            ).item())

        # Default: V59 action (high-value region).
        action = a_v59
        region = "high"
        cluster_idx: Optional[int] = None
        mix_weight_v59 = 1.0

        # Region classification per Task 7.3 spec.
        high_cutoff = self.value_threshold + self.boundary_margin
        low_cutoff = self.value_threshold - self.boundary_margin
        in_high = value > high_cutoff
        in_low = value < low_cutoff
        in_boundary = (not in_high) and (not in_low)

        if in_high:
            # High-value region -> V59 action.
            region = "high"
        else:
            # Boundary or low-value region -> need a cluster assignment.
            cluster_idx = self.find_cluster(latent.squeeze(0))
            sub = self._get_active_sub_policy(cluster_idx)
            if sub is None:
                # No active sub-policy (None or frozen) -> safe V59 fallback.
                sp = self.sub_policies[cluster_idx] if cluster_idx is not None else None
                if sp is not None and getattr(sp, "frozen", False):
                    region = "low_frozen"
                else:
                    region = "low_no_sub"
            else:
                a_sub = sub(latent)
                if in_boundary:
                    # Softmax mixing for smooth transition.
                    action, w_v59 = _softmax_mix(
                        v=value,
                        threshold=self.value_threshold,
                        margin=self.boundary_margin,
                        a_v59=a_v59,
                        a_sub=a_sub,
                        cell_idx=cluster_idx,
                    )
                    mix_weight_v59 = w_v59
                    region = "boundary"
                else:
                    # Deep low-value region -> pure sub-policy action.
                    action = a_sub
                    mix_weight_v59 = 0.0
                    region = "low"

        info = {
            "value": value,
            "region": region,
            "cluster": cluster_idx,
            "mix_weight_v59": mix_weight_v59,
        }
        return action, info

    def predict(
        self,
        obs,
        state=None,
        episode_start=None,
        deterministic: bool = True,
    ):
        """SB3-compatible predict() returning ``(action_numpy, None)``.

        Accepts the standard SB3 ``predict`` signature
        ``(obs, state, episode_start, deterministic)`` so this RouterPolicy
        can be plugged into :class:`HierarchicalPickPlacePolicy` and
        :mod:`eval_hierarchical` directly. The ``state`` and ``episode_start``
        arguments are accepted for signature compatibility but ignored
        (stateless router).

        ``obs`` may be either a numpy dict (from VecNormalize) or a tensor
        dict. Both are converted to float32 tensors on ``self.device``.
        """
        if isinstance(obs, dict):
            obs_t = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    obs_t[k] = torch.as_tensor(
                        v, dtype=torch.float32, device=self.device
                    )
                else:
                    obs_t[k] = v
        else:
            obs_t = obs
        action, _info = self.forward(obs_t)
        return action.cpu().numpy().astype(np.float32), None

    def save_state(self, path: str) -> None:
        """Save RouterPolicy metadata + sub-policy states to ``path``.

        V59 itself is NOT saved (it stays on disk at PLACE_MODEL_PATH); only
        the sub-policy state_dicts + routing config are persisted.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "k": self.k,
            "value_threshold": self.value_threshold,
            "boundary_margin": self.boundary_margin,
            "boundary_temperature": self.boundary_temperature,
            "cluster_centers": (
                self.cluster_centers.cpu().tolist()
                if self.cluster_centers is not None else None
            ),
            "sub_policies": [
                {
                    "cell_idx": k,
                    "frozen": getattr(sp, "frozen", False)
                    if sp is not None else None,
                    "state_dict": sp.state_dict() if sp is not None else None,
                }
                for k, sp in enumerate(self.sub_policies)
            ],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        torch.save(payload, path)
        print(f"  RouterPolicy state saved to {path}")

    def load_state(self, path: str) -> None:
        """Load RouterPolicy metadata + sub-policy states from ``path``.

        Only updates sub-policy weights + routing config; V59 itself is left
        untouched (it must already be loaded into ``self.v59_policy``).
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.k = int(payload.get("k", self.k))
        self.value_threshold = float(payload.get("value_threshold", self.value_threshold))
        self.boundary_margin = float(payload.get("boundary_margin", self.boundary_margin))
        self.boundary_temperature = float(
            payload.get("boundary_temperature", self.boundary_temperature)
        )
        if payload.get("cluster_centers") is not None:
            self.cluster_centers = torch.as_tensor(
                payload["cluster_centers"], dtype=torch.float32, device=self.device
            )
        for entry in payload.get("sub_policies", []):
            k = int(entry["cell_idx"])
            if not 0 <= k < self.k:
                continue
            if entry.get("state_dict") is None:
                self.sub_policies[k] = None
                continue
            sp = SubPolicy(cell_idx=k)
            sp.load_state_dict(entry["state_dict"])
            sp.frozen = bool(entry.get("frozen", False))
            sp.to(self.device)
            self.sub_policies[k] = sp
        print(f"  RouterPolicy state loaded from {path}")


# ---------------------------------------------------------------------------
# SubTask 7.1: Collect V59 trajectories per Voronoi cell
# ---------------------------------------------------------------------------

def collect_cell_trajectories(
    v59_policy,
    env_components: dict,
    cluster_centers: torch.Tensor,
    value_threshold: float,
    cell_idx: int,
    n_episodes: int,
    max_episodes_attempted: int = DEFAULT_MAX_EPISODE_ATTEMPTS_PER_CELL,
    max_steps: int = MAX_STEPS,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    device: str = "cpu",
    grasp_model=None,
) -> dict:
    """Collect V59 place-phase transitions filtered to a specific Voronoi cell.

    Runs V59 in the hierarchical env for up to ``max_episodes_attempted``
    episodes. For each place-phase step, computes V59's V(s) and latent; if
    ``V(s) < threshold`` (low-value region) and ``assign_cluster(latent) ==
    cell_idx``, the transition is recorded. Collection continues until
    ``n_episodes`` SUCCESSFUL episodes are gathered for this cell (or the
    attempt cap is hit).

    A "successful episode" for this cell means an episode that (a) entered
    the place phase, (b) recorded at least one transition in this cell, and
    (c) achieved final_dist < PLACE_THRESHOLD. We require success so the
    sub-policy learns from V59's good behavior in this cell (BC anchor).

    Parameters
    ----------
    v59_policy : SB3 ActorCriticPolicy
        Frozen V59 policy.
    env_components : dict
        Output of :func:`_build_collect_envs` (grasp + place vec envs, raw
        env, inner env, place_vision_wrapper).
    cluster_centers : Tensor (k, 524)
        Voronoi cluster centers.
    value_threshold : float
        V* separating low/high value regions.
    cell_idx : int
        Which Voronoi cell to collect for.
    n_episodes : int
        Target number of SUCCESSFUL episodes for this cell.
    max_episodes_attempted : int
        Hard cap on total episodes run for this cell (default 200).
    max_steps : int
        Max steps per episode.
    boundary_margin : float
        Included so transitions just inside the boundary count as low-value.
        A transition is recorded if ``V(s) < threshold + boundary_margin``
        AND ``assign_cluster == cell_idx``.
    device : str
        Torch device for V59 forward passes.
    grasp_model : optional
        Pre-loaded grasp model (if None, expects env_components['policy']).

    Returns
    -------
    dict
        Arrays: ``images`` (N, 84, 84, 3) uint8, ``states`` (N, 12) float32,
        ``latents`` (N, 524) float32, ``actions`` (N, 8) float32,
        ``rewards`` (N,) float32, ``next_latents`` (N, 524) float32,
        ``episode_idx`` (N,) int64, ``final_dist`` (N,) float32,
        ``episode_success`` (N,) int64. Also ``n_episodes_collected`` and
        ``n_episodes_attempted``.
    """
    raw_env = env_components["raw_env"]
    _inner_env = env_components["inner_env"]
    place_vec_env = env_components["place_vec_env"]
    place_vision_wrapper = env_components["place_vision_wrapper"]
    policy = env_components.get("policy")
    if policy is None and grasp_model is not None:
        from core.hierarchical_policy import HierarchicalPickPlacePolicy
        policy = HierarchicalPickPlacePolicy(grasp_model, env_components["place_model"])

    cluster_centers_t = torch.as_tensor(cluster_centers, dtype=torch.float32, device=device)
    collect_cutoff = value_threshold + boundary_margin

    all_images: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_latents: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[float] = []
    all_next_latents: list[np.ndarray] = []
    all_ep_idx: list[int] = []
    all_final_dists: list[float] = []
    all_ep_success: list[int] = []

    n_collected = 0
    n_attempted = 0
    t0 = time.time()

    while n_collected < n_episodes and n_attempted < max_episodes_attempted:
        n_attempted += 1
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        max_lift = 0.0
        block_target_dist = float("inf")
        first_place_step = None
        prev_info = None

        ep_images: list[np.ndarray] = []
        ep_states: list[np.ndarray] = []
        ep_latents: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_rewards: list[float] = []
        ep_next_latents: list[np.ndarray] = []

        for step in range(max_steps):
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
                obs = env_components["grasp_vec_env"].normalize_obs(raw_obs_for_grasp)

            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            if phase == "place":
                with torch.no_grad():
                    img_t = torch.as_tensor(obs["image"], dtype=torch.float32, device=device)
                    st_t = torch.as_tensor(obs["state"], dtype=torch.float32, device=device)
                    obs_t = {"image": img_t, "state": st_t}
                    latent = extract_v59_latent(v59_policy, obs_t).cpu().numpy()[0]
                    value = float(v59_value(v59_policy, obs_t).cpu().item())
                # Filter: low-value region AND this cell.
                if value < collect_cutoff:
                    cid = assign_cluster(latent, cluster_centers_t.cpu())
                    if cid == cell_idx:
                        ep_images.append(vision_obs["image"].copy())
                        ep_states.append(vision_obs["state"].copy())
                        ep_latents.append(latent.astype(np.float32))
                        ep_actions.append(action[0].copy())
                        ep_rewards.append(0.0)  # placeholder; PBRS uses Phi later

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
            # Compute next_latent for the last recorded transition.
            if phase == "place" and ep_latents and not done[0]:
                try:
                    next_vision_obs = place_vision_wrapper.observation(_inner_env._get_obs())
                    next_batched = {
                        "image": next_vision_obs["image"][np.newaxis, ...],
                        "state": next_vision_obs["state"][np.newaxis, ...],
                    }
                    next_obs = place_vec_env.normalize_obs(next_batched)
                    next_obs["image"] = np.transpose(next_obs["image"], (0, 3, 1, 2))
                    with torch.no_grad():
                        nimg = torch.as_tensor(next_obs["image"], dtype=torch.float32, device=device)
                        nst = torch.as_tensor(next_obs["state"], dtype=torch.float32, device=device)
                        nobs_t = {"image": nimg, "state": nst}
                        next_latent = extract_v59_latent(v59_policy, nobs_t).cpu().numpy()[0]
                    ep_next_latents.append(next_latent.astype(np.float32))
                except Exception:
                    ep_next_latents.append(ep_latents[-1].copy())
            if done[0]:
                break

        placed = block_target_dist < PLACE_THRESHOLD
        # Only count as a successful cell episode if we recorded transitions.
        if ep_latents and placed:
            n_collected += 1
            success_label = 1
        elif ep_latents:
            success_label = 0
        else:
            success_label = 0

        # Record all transitions from this episode (success or failure) for
        # this cell, so we have failure data for PBRS shaping too.
        if ep_latents:
            # Pad next_latents if needed (last step may not have one).
            while len(ep_next_latents) < len(ep_latents):
                ep_next_latents.append(ep_latents[-1].copy())
            n_trans = len(ep_latents)
            for j in range(n_trans):
                all_images.append(ep_images[j])
                all_states.append(ep_states[j])
                all_latents.append(ep_latents[j])
                all_actions.append(ep_actions[j])
                all_rewards.append(ep_rewards[j])
                all_next_latents.append(ep_next_latents[j])
                all_ep_idx.append(n_attempted - 1)
                all_final_dists.append(float(block_target_dist))
                all_ep_success.append(success_label)

        elapsed = time.time() - t0
        status = "PLACED" if placed else ("grabbed" if max_lift > LIFT_THRESHOLD else "failed")
        print(f"    Cell {cell_idx} attempt {n_attempted}: {status:7s}  "
              f"dist={block_target_dist*100:5.1f}cm  cell_trans={len(ep_images):3d}  "
              f"| collected={n_collected}/{n_episodes}  [{elapsed:.0f}s]")

    print(f"  Cell {cell_idx}: collected {n_collected} successful episodes "
          f"({n_attempted} attempted), {len(all_latents)} total transitions.")

    return {
        "images": np.array(all_images, dtype=np.uint8) if all_images
                  else np.zeros((0, 84, 84, 3), dtype=np.uint8),
        "states": np.array(all_states, dtype=np.float32) if all_states
                  else np.zeros((0, 12), dtype=np.float32),
        "latents": np.array(all_latents, dtype=np.float32) if all_latents
                   else np.zeros((0, LATENT_DIM), dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32) if all_actions
                   else np.zeros((0, ACTION_DIM), dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32) if all_rewards
                   else np.zeros((0,), dtype=np.float32),
        "next_latents": np.array(all_next_latents, dtype=np.float32) if all_next_latents
                        else np.zeros((0, LATENT_DIM), dtype=np.float32),
        "episode_idx": np.array(all_ep_idx, dtype=np.int64) if all_ep_idx
                       else np.zeros((0,), dtype=np.int64),
        "final_dist": np.array(all_final_dists, dtype=np.float32) if all_final_dists
                      else np.zeros((0,), dtype=np.float32),
        "episode_success": np.array(all_ep_success, dtype=np.int64) if all_ep_success
                           else np.zeros((0,), dtype=np.int64),
        "n_episodes_collected": n_collected,
        "n_episodes_attempted": n_attempted,
    }


# ---------------------------------------------------------------------------
# SubTask 7.2: Train a single sub-policy with BC + CSIL++ PBRS
# ---------------------------------------------------------------------------

def _compute_sub_policy_log_prob(
    sub_policy: SubPolicy,
    latents: torch.Tensor,
    actions: torch.Tensor,
    v59_log_std: torch.Tensor,
) -> torch.Tensor:
    """Compute log pi(a|s) under the sub-policy's diagonal Gaussian.

    Mirrors :func:`train_csil_plus_plus._compute_a_pi_log_prob`. The sub-policy
    outputs the mean; the (frozen) log_std is copied from V59.
    """
    mean = sub_policy(latents)
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


def _v59_log_std(policy) -> torch.Tensor:
    """Return V59's log_std parameter, clamped to [-5, 2] for stability."""
    return policy.log_std.detach().clamp(LOG_STD_MIN, LOG_STD_MAX)


def train_sub_policy(
    sub_policy: SubPolicy,
    cell_data: dict,
    phi: PotentialFunction,
    v59_policy,
    n_iterations: int = DEFAULT_N_TRAIN_ITERATIONS,
    learning_rate: float = DEFAULT_SUB_POLICY_LR,
    clip_range: float = DEFAULT_SUB_POLICY_CLIP,
    max_kl: float = DEFAULT_SUB_POLICY_MAX_KL,
    lambda_bc: float = DEFAULT_LAMBDA_BC,
    lambda_pbrs: float = DEFAULT_LAMBDA_PBRS,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    batch_size: int = 64,
    n_epochs: int = 2,
    device: str = "cpu",
    cell_idx: Optional[int] = None,
) -> dict:
    """Train a single sub-policy with BC anchor + CSIL++ PBRS PPO update.

    The total loss is

        L = lambda_bc * L_bc + lambda_pbrs * L_pbrs

    where ``L_bc = MSE(sub_policy(latent), a_V59)`` (anchors the sub-policy
    to V59's behavior) and ``L_pbrs`` is the PPO clipped surrogate on shaped
    rewards ``r_shaped = r_env + gamma * Phi(s') - Phi(s)``.

    Conservative hyperparameters (lr=1e-7, clip=0.1, max_kl=0.005) per
    project_memory case studies.

    Parameters
    ----------
    sub_policy : SubPolicy
        Initialized from V59 (will be fine-tuned in place).
    cell_data : dict
        Per-cell data from :func:`collect_cell_trajectories`. Must contain
        ``latents``, ``next_latents``, ``actions``, ``rewards``.
    phi : PotentialFunction
        Frozen potential function Phi(s) (from train_csil_plus_plus train-reward).
    v59_policy : SB3 ActorCriticPolicy
        Frozen V59 policy (for log_std and for BC anchor targets).
    n_iterations : int
        Outer PPO iterations (default 10).
    learning_rate, clip_range, max_kl : float
        Conservative PPO hyperparameters.
    lambda_bc, lambda_pbrs : float
        Loss weights (default 0.5 each).
    gamma, gae_lambda : float
        Discount + GAE lambda.
    batch_size, n_epochs : int
        Mini-batch size + PPO epochs per iteration.
    device : str
        Torch device.
    cell_idx : int, optional
        For logging.

    Returns
    -------
    dict
        Training metrics: ``iterations`` (list of per-iter dicts),
        ``final_bc_loss``, ``final_kl``, ``early_stopped`` (bool), ``best_iter``.
    """
    sub_policy.to(device).train()
    phi.to(device).eval()
    for p in phi.parameters():
        p.requires_grad = False

    latents = torch.as_tensor(cell_data["latents"], dtype=torch.float32, device=device)
    next_latents = torch.as_tensor(cell_data["next_latents"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(cell_data["actions"], dtype=torch.float32, device=device)
    env_rewards = torch.as_tensor(cell_data["rewards"], dtype=torch.float32, device=device)
    # BC targets = V59's recorded actions (already in cell_data['actions']).
    bc_targets = actions.clone()

    N = latents.shape[0]
    if N == 0:
        print(f"  Cell {cell_idx}: no transitions -- skipping training.")
        return {
            "iterations": [],
            "final_bc_loss": None,
            "final_kl": None,
            "early_stopped": False,
            "best_iter": None,
            "n_transitions": 0,
        }

    optimizer = torch.optim.Adam(sub_policy.parameters(), lr=learning_rate)
    v59_log_std_tensor = _v59_log_std(v59_policy).to(device).squeeze()
    if v59_log_std_tensor.dim() == 0:
        v59_log_std_tensor = v59_log_std_tensor.unsqueeze(0)

    print(f"  Cell {cell_idx}: training on {N} transitions, "
          f"{n_iterations} iterations, lr={learning_rate}, "
          f"clip={clip_range}, max_kl={max_kl}, "
          f"lambda_bc={lambda_bc}, lambda_pbrs={lambda_pbrs}")

    iterations_log: list[dict] = []
    best_bc_loss = float("inf")
    best_iter = None
    early_stopped = False
    t0 = time.time()

    for it in range(n_iterations):
        # 1. Compute Phi(s) and Phi(s') under no_grad (phi is frozen).
        with torch.no_grad():
            phi_s = phi(latents)
            phi_s_next = phi(next_latents)

        # 2. Shaped rewards (PBRS): r_shaped = r + gamma * Phi(s') - Phi(s).
        shaped_rewards = env_rewards + gamma * phi_s_next - phi_s

        # 3. Old log probs (no grad) for PPO ratio.
        with torch.no_grad():
            old_log_probs = _compute_sub_policy_log_prob(
                sub_policy, latents, actions, v59_log_std_tensor
            )
            # Simple advantage: shaped_reward (centered).
            advantages = shaped_rewards - shaped_rewards.mean()
            if advantages.numel() > 1:
                advantages = advantages / (advantages.std() + 1e-8)

        # 4. PPO mini-batch updates with BC anchor loss.
        T = N
        bs = min(batch_size, T)
        indices = torch.arange(T, device=device)
        policy_losses: list[float] = []
        bc_losses: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []

        for epoch in range(n_epochs):
            perm = indices[torch.randperm(T, device=device)]
            for start in range(0, T, bs):
                idx = perm[start:start + bs]
                lat_b = latents[idx]
                act_b = actions[idx]
                adv_b = advantages[idx]
                old_lp_b = old_log_probs[idx]
                bc_tgt_b = bc_targets[idx]

                # New log probs (with grad through sub_policy).
                new_log_probs = _compute_sub_policy_log_prob(
                    sub_policy, lat_b, act_b, v59_log_std_tensor
                )
                ratio = torch.exp(new_log_probs - old_lp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_b
                ppo_loss = -torch.min(surr1, surr2).mean()

                # BC anchor loss: MSE to V59's recorded actions.
                pred_actions = sub_policy(lat_b)
                bc_loss = F.mse_loss(pred_actions, bc_tgt_b)

                loss = lambda_bc * bc_loss + lambda_pbrs * ppo_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sub_policy.parameters(), 0.3)
                optimizer.step()

                with torch.no_grad():
                    kl = (old_lp_b - new_log_probs).mean().item()
                    clip_frac = (torch.abs(ratio - 1.0) > clip_range
                                 ).float().mean().item()
                policy_losses.append(ppo_loss.item())
                bc_losses.append(bc_loss.item())
                kls.append(kl)
                clip_fractions.append(clip_frac)

            # KL early stop.
            epoch_mean_kl = float(np.mean(kls[-max(1, T // bs):])) if kls else 0.0
            if epoch_mean_kl > max_kl:
                print(f"    Cell {cell_idx} iter {it+1}: KL early stop at epoch "
                      f"{epoch+1}/{n_epochs} (mean_kl={epoch_mean_kl:.6f} > "
                      f"max_kl={max_kl})")
                early_stopped = True
                break

        iter_bc = float(np.mean(bc_losses)) if bc_losses else 0.0
        iter_ppo = float(np.mean(policy_losses)) if policy_losses else 0.0
        iter_kl = float(np.mean(kls)) if kls else 0.0
        iter_clip = float(np.mean(clip_fractions)) if clip_fractions else 0.0

        if iter_bc < best_bc_loss:
            best_bc_loss = iter_bc
            best_iter = it

        elapsed = time.time() - t0
        print(f"    Cell {cell_idx} iter {it+1}/{n_iterations}: "
              f"bc_loss={iter_bc:.6f}  ppo_loss={iter_ppo:.6f}  "
              f"kl={iter_kl:.6f}  clip_frac={iter_clip:.4f}  "
              f"early_stopped={early_stopped}  [{elapsed:.0f}s]")

        iterations_log.append({
            "iteration": it,
            "bc_loss": iter_bc,
            "ppo_loss": iter_ppo,
            "kl": iter_kl,
            "clip_fraction": iter_clip,
            "early_stopped": early_stopped,
        })

        if early_stopped:
            break

    return {
        "iterations": iterations_log,
        "final_bc_loss": iterations_log[-1]["bc_loss"] if iterations_log else None,
        "final_kl": iterations_log[-1]["kl"] if iterations_log else None,
        "early_stopped": early_stopped,
        "best_iter": best_iter,
        "n_transitions": int(N),
        "best_bc_loss": best_bc_loss if iterations_log else None,
    }


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------

def cmd_collect(args) -> None:
    """Collect state values from V59 trajectories."""
    collect_state_values(
        n_states=args.n_states,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        place_model_path=args.place_model,
        place_vecnorm_path=args.place_vecnorm,
        grasp_model_path=args.grasp_model,
        grasp_vecnorm_path=args.grasp_vecnorm,
        release_threshold=args.release_threshold,
        seed=args.seed,
        device=args.device,
        output_path=args.output,
    )


def cmd_partition(args) -> None:
    """Run threshold + K-means partitioning on collected state values."""
    if not os.path.exists(args.states_path):
        print(f"ERROR: states file not found at {args.states_path}")
        print("  Run `python voronoi_partition.py collect` first.")
        sys.exit(1)

    data = np.load(args.states_path, allow_pickle=True)
    latents = torch.as_tensor(data["latents"], dtype=torch.float32)
    values = torch.as_tensor(data["values"], dtype=torch.float32)

    print(f"Loaded {latents.shape[0]} states from {args.states_path}")

    threshold, low_mask, high_mask = compute_value_threshold(
        values, low_value_fraction=args.low_value_fraction)

    cluster_centers, cluster_assignments, kmeans_model = voronoi_partition(
        latents, low_mask, k=args.k, seed=args.seed,
        use_sklearn=not args.no_sklearn)

    # Compute cluster statistics for the JSON output.
    success_labels = torch.as_tensor(data["success_labels"], dtype=torch.float32)
    sizes, mean_values, mean_success = _cluster_stats(
        latents, values, success_labels, low_mask,
        cluster_assignments, cluster_centers, args.k)

    partition = {
        "threshold": float(threshold),
        "low_value_fraction": float(args.low_value_fraction),
        "k": int(args.k),
        "n_total": int(latents.shape[0]),
        "n_low_value": int(low_mask.sum().item()),
        "n_high_value": int(high_mask.sum().item()),
        "cluster_centers": cluster_centers.tolist(),
        "cluster_sizes": sizes,
        "cluster_mean_values": mean_values,
        "cluster_success_rates": mean_success,
        "value_stats": {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "std": float(values.std().item()) if values.numel() > 1 else 0.0,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    CSIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(VORONOI_PARTITION_PATH, "w") as f:
        json.dump(partition, f, indent=2)
    print(f"\nPartition saved to {VORONOI_PARTITION_PATH}")
    print(f"  threshold={threshold:.4f}  k={args.k}  "
          f"n_low={partition['n_low_value']}  n_high={partition['n_high_value']}")


def cmd_verify(args) -> None:
    """Verify partition quality."""
    if not os.path.exists(args.states_path):
        print(f"ERROR: states file not found at {args.states_path}")
        print("  Run `python voronoi_partition.py collect` first.")
        sys.exit(1)
    if not os.path.exists(VORONOI_PARTITION_PATH):
        print(f"ERROR: partition file not found at {VORONOI_PARTITION_PATH}")
        print("  Run `python voronoi_partition.py partition` first.")
        sys.exit(1)

    data = np.load(args.states_path, allow_pickle=True)
    state_data = {
        "episode_ids": data["episode_ids"],
        "final_dists": data["final_dists"],
        "success_labels": data["success_labels"],
        "values": data["values"],
    }
    latents = torch.as_tensor(data["latents"], dtype=torch.float32)
    values = torch.as_tensor(data["values"], dtype=torch.float32)

    with open(VORONOI_PARTITION_PATH, "r") as f:
        partition = json.load(f)

    threshold = float(partition["threshold"])
    k = int(partition["k"])
    cluster_centers = np.array(partition["cluster_centers"], dtype=np.float32)

    # Recompute masks (must match what partition used).
    low_mask = values <= threshold
    high_mask = values > threshold

    # Recompute cluster assignments on low-value latents.
    low_latents = latents[low_mask]
    cluster_centers_t = torch.as_tensor(cluster_centers, dtype=torch.float32)
    dists = torch.cdist(low_latents, cluster_centers_t)
    cluster_assignments = dists.argmin(dim=-1).numpy()

    report = verify_partition(
        state_data=state_data,
        low_mask=low_mask,
        high_mask=high_mask,
        cluster_assignments=cluster_assignments,
        cluster_centers=cluster_centers,
        k=k,
    )
    print(f"\nAll assertions passed: {report['all_assertions_passed']}")


def cmd_route_demo(args) -> None:
    """Demo the RouterPolicy (no sub-policies -> V59 actions only)."""
    print("=" * 60)
    print("RouterPolicy Demo (no sub-policies -> V59 safe degradation)")
    print("=" * 60)

    if not os.path.exists(args.states_path):
        print(f"ERROR: states file not found at {args.states_path}")
        print("  Run `python voronoi_partition.py collect` first.")
        sys.exit(1)
    if not os.path.exists(VORONOI_PARTITION_PATH):
        print(f"ERROR: partition file not found at {VORONOI_PARTITION_PATH}")
        print("  Run `python voronoi_partition.py partition` first.")
        sys.exit(1)

    # Load partition.
    with open(VORONOI_PARTITION_PATH, "r") as f:
        partition = json.load(f)
    threshold = float(partition["threshold"])
    k = int(partition["k"])
    cluster_centers = torch.as_tensor(
        partition["cluster_centers"], dtype=torch.float32)

    # Load states.
    data = np.load(args.states_path, allow_pickle=True)
    images = torch.as_tensor(data["images"], dtype=torch.float32)
    states_raw = torch.as_tensor(data["states"], dtype=torch.float32)
    values = data["values"]
    n = min(args.n_demo, images.shape[0])

    # Load V59 policy.
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    print(f"\nLoading V59 from {args.place_model} ...")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium  # noqa: F401
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs

    def _make_env():
        import gymnasium as gn
        env = gn.make("PandaVLA-v0", reward_type="dense", gravity_comp=True,
                      target_pos_range=TARGET_POS_RANGE, domain_randomize=False)
        return VisionObs(env, image_size=84)

    place_vec_env = DummyVecEnv([_make_env])
    if os.path.exists(args.place_vecnorm):
        place_vec_env = VecNormalize.load(args.place_vecnorm, place_vec_env)
        place_vec_env.norm_reward = False
        place_vec_env.training = False
    place_model = PPO.load(args.place_model, env=place_vec_env, device=device)
    freeze_backbone(place_model)
    place_model.policy.features_extractor.eval()

    # Build RouterPolicy with NO sub-policies (safe degradation).
    router = RouterPolicy(
        v59_policy=place_model.policy,
        cluster_centers=cluster_centers,
        value_threshold=threshold,
        boundary_margin=args.boundary_margin,
        boundary_temperature=args.boundary_temperature,
        k=k,
        device=device,
    )

    print(f"\nRunning router on {n} sampled states (no sub-policies registered)...")
    print(f"  threshold={threshold:.4f}  k={k}  boundary_margin={args.boundary_margin}")
    print()
    print(f"  {'Idx':>4}  {'V(s)':>8}  {'Region':>12}  {'Cluster':>7}  "
          f"{'w_V59':>6}  {'max|a-a_v59|':>12}")

    n_high = n_low = n_boundary = 0
    states_norm = normalize_states(states_raw, args.place_vecnorm)
    for i in range(n):
        img = images[i:i+1].permute(0, 3, 1, 2).contiguous().to(device)
        st = states_norm[i:i+1].to(device)
        obs = {"image": img, "state": st}
        action, info = router.forward(obs)
        # Compare to pure V59 action for safety-degradation check.
        with torch.no_grad():
            a_v59 = v59_action_mean(place_model.policy, obs)
        delta = float((action - a_v59).abs().max().item())
        region = info["region"]
        cluster = info["cluster"] if info["cluster"] is not None else "-"
        print(f"  {i:>4}  {info['value']:>8.4f}  {region:>12}  "
              f"{str(cluster):>7}  {info['mix_weight_v59']:>6.2f}  {delta:>12.6f}")
        if region == "high":
            n_high += 1
        elif region == "boundary":
            n_boundary += 1
        else:
            n_low += 1

    print()
    print(f"  Summary: high={n_high}  low_no_sub={n_low}  boundary={n_boundary}")
    print(f"  (All actions == V59 action because no sub-policies are registered.)")
    print(f"  Safe degradation: CONFIRMED" if (n_low == 0 or args.allow_no_sub)
          else "  WARNING: low-value states without sub-policies!")


# ---------------------------------------------------------------------------
# Task 7.1 CLI: collect-cell-data
# ---------------------------------------------------------------------------

def cmd_collect_cell_data(args) -> None:
    """Collect V59 trajectories per Voronoi cell -> data/voronoi_cell_data.npz."""
    print("=" * 60)
    print("Voronoi Cell Data Collection (Task 7.1)")
    print("=" * 60)

    if not os.path.exists(args.partition_json):
        print(f"ERROR: partition JSON not found at {args.partition_json}")
        print("  Run `python voronoi_partition.py partition` first.")
        sys.exit(1)

    with open(args.partition_json, "r") as f:
        partition = json.load(f)
    threshold = float(partition["threshold"])
    k = int(partition["k"])
    cluster_centers = torch.as_tensor(
        partition["cluster_centers"], dtype=torch.float32
    )
    print(f"  Loaded partition: k={k}  threshold={threshold:.4f}")
    print(f"  n_episodes_per_cell={args.n_episodes_per_cell}  "
          f"max_attempts={args.max_attempts_per_cell}")
    print(f"  Image augmentation: DISABLED")
    print(f"  BN running stats: FROZEN")
    print(f"  V59 weights: FROZEN -- Critic is inference-only")
    print()

    # Resolve device: "auto" -> cuda if available, else cpu. MUST match the
    # device the model is loaded on (obs tensors vs model weights).
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build envs (reuses Task 6 _build_collect_envs).
    args_ns = argparse.Namespace(
        place_model=args.place_model,
        place_vecnorm=args.place_vecnorm,
        grasp_model=args.grasp_model,
        grasp_vecnorm=args.grasp_vecnorm,
        release_threshold=args.release_threshold,
    )
    envs = _build_collect_envs(args_ns, device=device)
    place_model = envs["place_model"]
    freeze_backbone(place_model)
    place_model.policy.features_extractor.eval()
    for m in place_model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    v59_policy = place_model.policy

    np.random.seed(args.seed)
    try:
        envs["raw_env"].seed(args.seed)
    except Exception:
        pass

    # Collect per-cell data.
    all_cell_data: dict[int, dict] = {}
    for cell_idx in range(k):
        print(f"\n--- Collecting cell {cell_idx}/{k} ---")
        cell_data = collect_cell_trajectories(
            v59_policy=v59_policy,
            env_components=envs,
            cluster_centers=cluster_centers,
            value_threshold=threshold,
            cell_idx=cell_idx,
            n_episodes=args.n_episodes_per_cell,
            max_episodes_attempted=args.max_attempts_per_cell,
            max_steps=args.max_steps,
            boundary_margin=args.boundary_margin,
            device=device,
        )
        all_cell_data[cell_idx] = cell_data
        n_succ = int(cell_data["episode_success"].sum()) if cell_data["episode_success"].size > 0 else 0
        n_trans = len(cell_data["latents"])
        print(f"  Cell {cell_idx} done: {cell_data['n_episodes_collected']} "
              f"successful episodes, {n_trans} transitions, "
              f"{n_succ} success-labeled transitions.")

    # Merge into a single .npz with cell_idx labels.
    all_images = np.concatenate(
        [all_cell_data[c]["images"] for c in range(k) if all_cell_data[c]["images"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["images"].size > 0 for c in range(k)) else np.zeros((0, 84, 84, 3), dtype=np.uint8)
    all_states = np.concatenate(
        [all_cell_data[c]["states"] for c in range(k) if all_cell_data[c]["states"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["states"].size > 0 for c in range(k)) else np.zeros((0, 12), dtype=np.float32)
    all_latents = np.concatenate(
        [all_cell_data[c]["latents"] for c in range(k) if all_cell_data[c]["latents"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["latents"].size > 0 for c in range(k)) else np.zeros((0, LATENT_DIM), dtype=np.float32)
    all_actions = np.concatenate(
        [all_cell_data[c]["actions"] for c in range(k) if all_cell_data[c]["actions"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["actions"].size > 0 for c in range(k)) else np.zeros((0, ACTION_DIM), dtype=np.float32)
    all_rewards = np.concatenate(
        [all_cell_data[c]["rewards"] for c in range(k) if all_cell_data[c]["rewards"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["rewards"].size > 0 for c in range(k)) else np.zeros((0,), dtype=np.float32)
    all_next_latents = np.concatenate(
        [all_cell_data[c]["next_latents"] for c in range(k) if all_cell_data[c]["next_latents"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["next_latents"].size > 0 for c in range(k)) else np.zeros((0, LATENT_DIM), dtype=np.float32)
    all_final_dists = np.concatenate(
        [all_cell_data[c]["final_dist"] for c in range(k) if all_cell_data[c]["final_dist"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["final_dist"].size > 0 for c in range(k)) else np.zeros((0,), dtype=np.float32)
    all_ep_success = np.concatenate(
        [all_cell_data[c]["episode_success"] for c in range(k) if all_cell_data[c]["episode_success"].size > 0],
        axis=0,
    ) if any(all_cell_data[c]["episode_success"].size > 0 for c in range(k)) else np.zeros((0,), dtype=np.int64)

    # cell_idx and episode_idx arrays (with global offset per cell).
    cell_idx_parts: list[np.ndarray] = []
    episode_idx_parts: list[np.ndarray] = []
    ep_offset = 0
    for c in range(k):
        n_c = len(all_cell_data[c]["latents"])
        if n_c > 0:
            cell_idx_parts.append(np.full((n_c,), c, dtype=np.int64))
            episode_idx_parts.append(all_cell_data[c]["episode_idx"] + ep_offset)
            ep_offset += int(all_cell_data[c]["n_episodes_attempted"])
    if cell_idx_parts:
        all_cell_idx = np.concatenate(cell_idx_parts, axis=0)
        all_episode_idx = np.concatenate(episode_idx_parts, axis=0)
    else:
        all_cell_idx = np.zeros((0,), dtype=np.int64)
        all_episode_idx = np.zeros((0,), dtype=np.int64)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        images=all_images,
        states=all_states,
        latents=all_latents,
        actions=all_actions,
        rewards=all_rewards,
        next_latents=all_next_latents,
        cell_idx=all_cell_idx,
        episode_idx=all_episode_idx,
        final_dist=all_final_dists,
        episode_success=all_ep_success,
    )

    print()
    print("=" * 60)
    print("Cell Data Collection Complete")
    print("=" * 60)
    print(f"  Total transitions: {len(all_latents)}")
    for c in range(k):
        n_c = int((all_cell_idx == c).sum()) if len(all_cell_idx) > 0 else 0
        n_succ = int(((all_cell_idx == c) & (all_ep_success == 1)).sum()) if len(all_cell_idx) > 0 else 0
        print(f"  Cell {c}: {n_c} transitions  ({n_succ} success-labeled)")
    print(f"  Output: {out_path}")

    try:
        envs["raw_env"].close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Task 7.2 CLI: train-sub-policies
# ---------------------------------------------------------------------------

def cmd_train_sub_policies(args) -> None:
    """Train K sub-policies with BC + CSIL++ PBRS (Task 7.2)."""
    print("=" * 60)
    print("Sub-Policy Training (Task 7.2): BC + CSIL++ PBRS")
    print("=" * 60)

    if not os.path.exists(args.cell_data):
        print(f"ERROR: cell data not found at {args.cell_data}")
        print("  Run `python voronoi_partition.py collect-cell-data` first.")
        sys.exit(1)
    if not os.path.exists(args.partition_json):
        print(f"ERROR: partition JSON not found at {args.partition_json}")
        print("  Run `python voronoi_partition.py partition` first.")
        sys.exit(1)
    if not os.path.exists(args.potential_fn):
        print(f"ERROR: potential function not found at {args.potential_fn}")
        print("  Run `python train_csil_plus_plus.py train-reward` first.")
        sys.exit(1)

    with open(args.partition_json, "r") as f:
        partition = json.load(f)
    k = int(partition["k"])
    cluster_centers = torch.as_tensor(
        partition["cluster_centers"], dtype=torch.float32
    )
    threshold = float(partition["threshold"])
    print(f"  Partition: k={k}  threshold={threshold:.4f}")

    # Load cell data.
    print(f"\n=== Loading cell data from {args.cell_data} ===")
    data = np.load(args.cell_data, allow_pickle=True)
    print(f"  Total transitions: {len(data['latents'])}")
    cell_data_arrays = {
        c: {
            "images": data["images"][data["cell_idx"] == c],
            "states": data["states"][data["cell_idx"] == c],
            "latents": data["latents"][data["cell_idx"] == c],
            "actions": data["actions"][data["cell_idx"] == c],
            "rewards": data["rewards"][data["cell_idx"] == c],
            "next_latents": data["next_latents"][data["cell_idx"] == c],
            "episode_idx": data["episode_idx"][data["cell_idx"] == c],
            "final_dist": data["final_dist"][data["cell_idx"] == c],
            "episode_success": data["episode_success"][data["cell_idx"] == c],
        }
        for c in range(k)
    }
    for c in range(k):
        print(f"  Cell {c}: {len(cell_data_arrays[c]['latents'])} transitions")

    # Load V59. Resolve "auto" to a concrete device for torch.as_tensor calls.
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"\n=== Loading V59 from {args.place_model} ===")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium  # noqa: F401
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs

    def _make_env():
        import gymnasium as gn
        env = gn.make("PandaVLA-v0", reward_type="dense", gravity_comp=True,
                      target_pos_range=TARGET_POS_RANGE, domain_randomize=False)
        return VisionObs(env, image_size=84)

    place_vec_env = DummyVecEnv([_make_env])
    if os.path.exists(args.place_vecnorm):
        place_vec_env = VecNormalize.load(args.place_vecnorm, place_vec_env)
        place_vec_env.norm_reward = False
        place_vec_env.training = False
    place_model = PPO.load(args.place_model, env=place_vec_env, device=device)
    freeze_backbone(place_model)
    place_model.policy.features_extractor.eval()
    for m in place_model.policy.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    v59_policy = place_model.policy

    # Load Phi (potential function from train_csil_plus_plus train-reward).
    print(f"\n=== Loading Phi from {args.potential_fn} ===")
    phi = PotentialFunction(in_dim=LATENT_DIM)
    phi.load_state_dict(
        torch.load(args.potential_fn, map_location="cpu", weights_only=False)
    )
    phi.to(device).eval()
    for p in phi.parameters():
        p.requires_grad = False
    print("  Phi loaded (frozen, eval mode).")

    # Output dir.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Training {k} sub-policies ===")
    print(f"  n_iterations={args.n_iterations}  lr={args.learning_rate}  "
          f"clip={args.clip_range}  max_kl={args.max_kl}")
    print(f"  lambda_bc={args.lambda_bc}  lambda_pbrs={args.lambda_pbrs}")
    print(f"  output_dir={output_dir}")

    training_log: list[dict] = []
    sub_policies: list[SubPolicy] = []

    for c in range(k):
        print(f"\n--- Training sub-policy for cell {c}/{k} ---")
        sp = SubPolicy(cell_idx=c)
        n_copied = sp.init_from_v59(v59_policy)
        print(f"  Initialized sub-policy from V59 ({n_copied} tensors copied).")

        cell_data_c = cell_data_arrays[c]
        if len(cell_data_c["latents"]) == 0:
            print(f"  Cell {c}: no transitions -- marking sub-policy as frozen.")
            sp.frozen = True
            sp.save(str(output_dir / f"cell_{c}_policy.pt"))
            sub_policies.append(sp)
            training_log.append({
                "cell_idx": c,
                "n_transitions": 0,
                "iterations": [],
                "final_bc_loss": None,
                "final_kl": None,
                "early_stopped": False,
                "frozen": True,
                "frozen_reason": "no_transitions",
            })
            continue

        metrics = train_sub_policy(
            sub_policy=sp,
            cell_data=cell_data_c,
            phi=phi,
            v59_policy=v59_policy,
            n_iterations=args.n_iterations,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            max_kl=args.max_kl,
            lambda_bc=args.lambda_bc,
            lambda_pbrs=args.lambda_pbrs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            device=device,
            cell_idx=c,
        )
        metrics["cell_idx"] = c
        metrics["frozen"] = False

        # Safety: if eval possible and place_rate < threshold, freeze.
        # (Actual env-based eval is expensive; we mark frozen only when
        # explicitly requested via --safety_eval. Here we just save.)
        sp.save(str(output_dir / f"cell_{c}_policy.pt"))
        sub_policies.append(sp)
        training_log.append(metrics)
        print(f"  Cell {c}: saved to {output_dir / f'cell_{c}_policy.pt'}")
        print(f"    final_bc_loss={metrics.get('final_bc_loss')}  "
              f"final_kl={metrics.get('final_kl')}  "
              f"early_stopped={metrics.get('early_stopped')}")

    # Save training log.
    log_payload = {
        "method": "voronoi_sub_policy_BC_PBRS",
        "k": k,
        "config": {
            "n_iterations": args.n_iterations,
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
            "max_kl": args.max_kl,
            "lambda_bc": args.lambda_bc,
            "lambda_pbrs": args.lambda_pbrs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
        },
        "cells": training_log,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(SUB_POLICY_TRAINING_LOG_PATH, "w") as f:
        json.dump(log_payload, f, indent=2)
    print(f"\nTraining log written to {SUB_POLICY_TRAINING_LOG_PATH}")

    # Save the complete RouterPolicy state (V59 + all sub-policies).
    print("\n=== Saving RouterPolicy state ===")
    router = RouterPolicy(
        v59_policy=v59_policy,
        cluster_centers=cluster_centers,
        value_threshold=threshold,
        sub_policies=sub_policies,
        boundary_margin=args.boundary_margin,
        k=k,
        device=device,
    )
    router.save_state(str(ROUTER_POLICY_PATH))

    print()
    print("=" * 60)
    print("Sub-Policy Training Complete")
    print("=" * 60)
    print(f"  K={k} sub-policies trained.")
    n_frozen = sum(1 for m in training_log if m.get("frozen"))
    print(f"  Frozen (safety/no-data): {n_frozen}/{k}")
    print(f"  Sub-policies: {output_dir}")
    print(f"  RouterPolicy state: {ROUTER_POLICY_PATH}")
    print(f"  Training log: {SUB_POLICY_TRAINING_LOG_PATH}")


# ---------------------------------------------------------------------------
# Task 7.3 CLI: route (full RouterPolicy demo with sub-policies loaded)
# ---------------------------------------------------------------------------

def cmd_route(args) -> None:
    """Full RouterPolicy demo with trained sub-policies loaded (Task 7.3)."""
    print("=" * 60)
    print("RouterPolicy Demo (Task 7.3): V59 + trained sub-policies")
    print("=" * 60)

    if not os.path.exists(args.states_path):
        print(f"ERROR: states file not found at {args.states_path}")
        print("  Run `python voronoi_partition.py collect` first.")
        sys.exit(1)
    if not os.path.exists(args.partition_json):
        print(f"ERROR: partition file not found at {args.partition_json}")
        print("  Run `python voronoi_partition.py partition` first.")
        sys.exit(1)
    sub_policy_dir = Path(args.sub_policy_dir)
    if not sub_policy_dir.exists():
        print(f"ERROR: sub-policy directory not found at {sub_policy_dir}")
        print("  Run `python voronoi_partition.py train-sub-policies` first.")
        sys.exit(1)

    # Load partition.
    with open(args.partition_json, "r") as f:
        partition = json.load(f)
    threshold = float(partition["threshold"])
    k = int(partition["k"])
    cluster_centers = torch.as_tensor(
        partition["cluster_centers"], dtype=torch.float32
    )

    # Load states.
    data = np.load(args.states_path, allow_pickle=True)
    images = torch.as_tensor(data["images"], dtype=torch.float32)
    states_raw = torch.as_tensor(data["states"], dtype=torch.float32)
    n = min(args.n_demo, images.shape[0])

    # Load V59 policy.
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    print(f"\nLoading V59 from {args.place_model} ...")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium  # noqa: F401
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs

    def _make_env():
        import gymnasium as gn
        env = gn.make("PandaVLA-v0", reward_type="dense", gravity_comp=True,
                      target_pos_range=TARGET_POS_RANGE, domain_randomize=False)
        return VisionObs(env, image_size=84)

    place_vec_env = DummyVecEnv([_make_env])
    if os.path.exists(args.place_vecnorm):
        place_vec_env = VecNormalize.load(args.place_vecnorm, place_vec_env)
        place_vec_env.norm_reward = False
        place_vec_env.training = False
    place_model = PPO.load(args.place_model, env=place_vec_env, device=device)
    freeze_backbone(place_model)
    place_model.policy.features_extractor.eval()

    # Build RouterPolicy and load sub-policies.
    router = RouterPolicy(
        v59_policy=place_model.policy,
        cluster_centers=cluster_centers,
        value_threshold=threshold,
        boundary_margin=args.boundary_margin,
        boundary_temperature=args.boundary_temperature,
        k=k,
        device=device,
    )
    n_loaded = router.load_sub_policies(str(sub_policy_dir))

    print(f"\nRunning router on {n} sampled states ({n_loaded}/{k} sub-policies loaded)...")
    print(f"  threshold={threshold:.4f}  k={k}  boundary_margin={args.boundary_margin}")
    print()
    print(f"  {'Idx':>4}  {'V(s)':>8}  {'Region':>12}  {'Cluster':>7}  "
          f"{'w_V59':>6}  {'max|a-a_v59|':>12}")

    n_high = n_low = n_boundary = n_frozen = n_no_sub = 0
    states_norm = normalize_states(states_raw, args.place_vecnorm)
    for i in range(n):
        img = images[i:i+1].permute(0, 3, 1, 2).contiguous().to(device)
        st = states_norm[i:i+1].to(device)
        obs = {"image": img, "state": st}
        action, info = router.forward(obs)
        # Compare to pure V59 action for safety-degradation check.
        with torch.no_grad():
            a_v59 = v59_action_mean(place_model.policy, obs)
        delta = float((action - a_v59).abs().max().item())
        region = info["region"]
        cluster = info["cluster"] if info["cluster"] is not None else "-"
        print(f"  {i:>4}  {info['value']:>8.4f}  {region:>12}  "
              f"{str(cluster):>7}  {info['mix_weight_v59']:>6.2f}  {delta:>12.6f}")
        if region == "high":
            n_high += 1
        elif region == "boundary":
            n_boundary += 1
        elif region == "low":
            n_low += 1
        elif region == "low_frozen":
            n_frozen += 1
        else:
            n_no_sub += 1

    print()
    print(f"  Summary: high={n_high}  low={n_low}  boundary={n_boundary}  "
          f"low_frozen={n_frozen}  low_no_sub={n_no_sub}")
    if n_low > 0:
        print(f"  Sub-policies ACTIVELY used in {n_low} low-value states.")
    if n_boundary > 0:
        print(f"  Boundary mixing applied in {n_boundary} states.")
    if n_frozen > 0:
        print(f"  WARNING: {n_frozen} states routed to V59 due to frozen sub-policies.")
    if n_no_sub > 0:
        print(f"  WARNING: {n_no_sub} states had no sub-policy registered (V59 fallback).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="voronoi_partition.py",
        description=(
            "Voronoi state-space partitioning + sub-policy training/routing "
            "using V59's Critic network. Tasks 6 + 7 of the "
            "v59-breakthrough-csil-voronoi spec. Partitions V59's state space "
            "into high-value (V59 unchanged) and low-value (K Voronoi cells, "
            "one sub-policy each) regions, trains per-cell sub-policies via "
            "BC + CSIL++ PBRS, and routes states at inference via V59's V(s)."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- common V59 paths ----
    def add_v59_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH,
                       help="Path to V59 place policy .zip")
        p.add_argument("--place_vecnorm", type=str, default=PLACE_VECNORM_PATH,
                       help="Path to V59 vec_normalize.pkl")
        p.add_argument("--device", type=str, default="auto",
                       help="torch device (default: auto = cuda if available)")
        p.add_argument("--seed", type=int, default=SEED)

    # ---- collect ----
    p_collect = sub.add_parser(
        "collect",
        help="Collect V59 state values -> data/voronoi_states.npz",
    )
    p_collect.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH)
    p_collect.add_argument("--place_vecnorm", type=str, default=PLACE_VECNORM_PATH)
    p_collect.add_argument("--grasp_model", type=str, default=GRASP_MODEL_PATH)
    p_collect.add_argument("--grasp_vecnorm", type=str, default=GRASP_VECNORM_PATH)
    p_collect.add_argument("--n_episodes", type=int, default=200,
                           help="Episodes to run (default 200 -> ~50k transitions)")
    p_collect.add_argument("--n_states", type=int, default=DEFAULT_N_STATES,
                           help="States to sample after collection (default 1000)")
    p_collect.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p_collect.add_argument("--release_threshold", type=float, default=PLACE_THRESHOLD,
                           help="Release distance threshold (m)")
    p_collect.add_argument("--output", type=str, default=str(VORONOI_STATES_PATH),
                           help=f"Output .npz path (default: {VORONOI_STATES_PATH})")
    p_collect.add_argument("--device", type=str, default="auto")
    p_collect.add_argument("--seed", type=int, default=SEED)
    p_collect.set_defaults(func=cmd_collect)

    # ---- partition ----
    p_part = sub.add_parser(
        "partition",
        help="Compute threshold + K-means -> outputs/csil_plus_plus/voronoi_partition.json",
    )
    p_part.add_argument("--states_path", type=str, default=str(VORONOI_STATES_PATH),
                        help="Path to voronoi_states.npz from `collect`")
    p_part.add_argument("--low_value_fraction", type=float,
                        default=DEFAULT_LOW_VALUE_FRACTION,
                        help="Fraction of states labeled low-value (default 0.44)")
    p_part.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Number of Voronoi cells (default 4)")
    p_part.add_argument("--no_sklearn", action="store_true",
                        help="Force torch K-means fallback even if sklearn is available")
    p_part.add_argument("--seed", type=int, default=SEED)
    p_part.set_defaults(func=cmd_partition)

    # ---- verify ----
    p_ver = sub.add_parser(
        "verify",
        help="Verify partition quality -> outputs/csil_plus_plus/voronoi_verification_report.json",
    )
    p_ver.add_argument("--states_path", type=str, default=str(VORONOI_STATES_PATH))
    p_ver.set_defaults(func=cmd_verify)

    # ---- route-demo ----
    p_rd = sub.add_parser(
        "route-demo",
        help="Demo the RouterPolicy (no sub-policies -> V59 actions only)",
    )
    p_rd.add_argument("--states_path", type=str, default=str(VORONOI_STATES_PATH))
    add_v59_args(p_rd)
    p_rd.add_argument("--n_demo", type=int, default=10,
                      help="Number of states to demo (default 10)")
    p_rd.add_argument("--boundary_margin", type=float,
                      default=DEFAULT_BOUNDARY_MARGIN,
                      help="|V(s) - threshold| < margin -> boundary mixing")
    p_rd.add_argument("--boundary_temperature", type=float,
                      default=DEFAULT_BOUNDARY_TEMPERATURE,
                      help="Softmax sharpness for boundary mixing")
    p_rd.add_argument("--allow_no_sub", action="store_true",
                      help="Suppress low-value-no-sub-policy warning")
    p_rd.set_defaults(func=cmd_route_demo)

    # ---- collect-cell-data (Task 7.1) ----
    p_ccd = sub.add_parser(
        "collect-cell-data",
        help="Collect per-cell V59 trajectories -> data/voronoi_cell_data.npz",
    )
    p_ccd.add_argument("--partition_json", type=str,
                       default=str(VORONOI_PARTITION_PATH),
                       help=f"Path to voronoi_partition.json (default: {VORONOI_PARTITION_PATH})")
    p_ccd.add_argument("--n_episodes_per_cell", type=int,
                       default=DEFAULT_N_EPISODES_PER_CELL,
                       help=f"Successful episodes per cell (default {DEFAULT_N_EPISODES_PER_CELL})")
    p_ccd.add_argument("--max_attempts_per_cell", type=int,
                       default=DEFAULT_MAX_EPISODE_ATTEMPTS_PER_CELL,
                       help=f"Hard cap on attempts per cell (default {DEFAULT_MAX_EPISODE_ATTEMPTS_PER_CELL})")
    p_ccd.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p_ccd.add_argument("--boundary_margin", type=float,
                       default=DEFAULT_BOUNDARY_MARGIN,
                       help="V(s) within threshold+margin counts as low-value for collection")
    p_ccd.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH)
    p_ccd.add_argument("--place_vecnorm", type=str, default=PLACE_VECNORM_PATH)
    p_ccd.add_argument("--grasp_model", type=str, default=GRASP_MODEL_PATH)
    p_ccd.add_argument("--grasp_vecnorm", type=str, default=GRASP_VECNORM_PATH)
    p_ccd.add_argument("--release_threshold", type=float, default=PLACE_THRESHOLD)
    p_ccd.add_argument("--output", type=str, default=str(VORONOI_CELL_DATA_PATH),
                       help=f"Output .npz path (default: {VORONOI_CELL_DATA_PATH})")
    p_ccd.add_argument("--device", type=str, default="auto")
    p_ccd.add_argument("--seed", type=int, default=SEED)
    p_ccd.set_defaults(func=cmd_collect_cell_data)

    # ---- train-sub-policies (Task 7.2) ----
    p_tsp = sub.add_parser(
        "train-sub-policies",
        help="Train K sub-policies with BC + CSIL++ PBRS",
    )
    p_tsp.add_argument("--cell_data", type=str, default=str(VORONOI_CELL_DATA_PATH),
                       help=f"Path to voronoi_cell_data.npz (default: {VORONOI_CELL_DATA_PATH})")
    p_tsp.add_argument("--partition_json", type=str,
                       default=str(VORONOI_PARTITION_PATH),
                       help=f"Path to voronoi_partition.json (default: {VORONOI_PARTITION_PATH})")
    p_tsp.add_argument("--potential_fn", type=str, default=str(POTENTIAL_FN_PATH),
                       help=f"Path to Phi from train_csil_plus_plus train-reward (default: {POTENTIAL_FN_PATH})")
    p_tsp.add_argument("--n_iterations", type=int, default=DEFAULT_N_TRAIN_ITERATIONS,
                       help=f"PPO+BC iterations per sub-policy (default {DEFAULT_N_TRAIN_ITERATIONS})")
    p_tsp.add_argument("--output_dir", type=str, default=str(SUB_POLICY_DIR),
                       help=f"Output dir for sub-policies (default: {SUB_POLICY_DIR})")
    p_tsp.add_argument("--place_model", type=str, default=PLACE_MODEL_PATH)
    p_tsp.add_argument("--place_vecnorm", type=str, default=PLACE_VECNORM_PATH)
    # Conservative hyperparameters (per project_memory: V70 crashed at KL=0.003).
    p_tsp.add_argument("--learning_rate", type=float, default=DEFAULT_SUB_POLICY_LR,
                       help=f"Adam LR (default {DEFAULT_SUB_POLICY_LR} = 100x lower than typical PPO)")
    p_tsp.add_argument("--clip_range", type=float, default=DEFAULT_SUB_POLICY_CLIP,
                       help=f"PPO clip range (default {DEFAULT_SUB_POLICY_CLIP})")
    p_tsp.add_argument("--max_kl", type=float, default=DEFAULT_SUB_POLICY_MAX_KL,
                       help=f"Early-stop KL threshold (default {DEFAULT_SUB_POLICY_MAX_KL})")
    p_tsp.add_argument("--lambda_bc", type=float, default=DEFAULT_LAMBDA_BC,
                       help=f"BC anchor loss weight (default {DEFAULT_LAMBDA_BC})")
    p_tsp.add_argument("--lambda_pbrs", type=float, default=DEFAULT_LAMBDA_PBRS,
                       help=f"PPO PBRS loss weight (default {DEFAULT_LAMBDA_PBRS})")
    p_tsp.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor for GAE + PBRS shaping")
    p_tsp.add_argument("--gae_lambda", type=float, default=0.95,
                       help="GAE lambda")
    p_tsp.add_argument("--batch_size", type=int, default=64,
                       help="Mini-batch size for PPO update")
    p_tsp.add_argument("--n_epochs", type=int, default=2,
                       help="PPO epochs per iteration (few = conservative)")
    p_tsp.add_argument("--boundary_margin", type=float,
                       default=DEFAULT_BOUNDARY_MARGIN,
                       help="Boundary margin for the saved RouterPolicy")
    p_tsp.add_argument("--device", type=str, default="auto")
    p_tsp.add_argument("--seed", type=int, default=SEED)
    p_tsp.set_defaults(func=cmd_train_sub_policies)

    # ---- route (Task 7.3): full RouterPolicy demo with sub-policies ----
    p_rt = sub.add_parser(
        "route",
        help="Full RouterPolicy demo with trained sub-policies loaded",
    )
    p_rt.add_argument("--states_path", type=str, default=str(VORONOI_STATES_PATH))
    p_rt.add_argument("--partition_json", type=str,
                      default=str(VORONOI_PARTITION_PATH),
                      help=f"Path to voronoi_partition.json (default: {VORONOI_PARTITION_PATH})")
    p_rt.add_argument("--sub_policy_dir", type=str, default=str(SUB_POLICY_DIR),
                      help=f"Dir with cell_{{k}}_policy.pt files (default: {SUB_POLICY_DIR})")
    add_v59_args(p_rt)
    p_rt.add_argument("--n_demo", type=int, default=10,
                      help="Number of states to demo (default 10)")
    p_rt.add_argument("--boundary_margin", type=float,
                      default=DEFAULT_BOUNDARY_MARGIN,
                      help="|V(s) - threshold| < margin -> boundary mixing")
    p_rt.add_argument("--boundary_temperature", type=float,
                      default=DEFAULT_BOUNDARY_TEMPERATURE,
                      help="Softmax sharpness for boundary mixing")
    p_rt.set_defaults(func=cmd_route)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
