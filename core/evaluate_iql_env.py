#!/usr/bin/env python3
"""P1: Environment evaluation of IQL place policy in MuJoCo.

Uses V59's grasp model for the grasp phase and the IQL policy for the
place phase. The HierarchicalPickPlacePolicy is used only for phase
detection; during the place phase, the IQL policy's action overrides
V59's place model action.

Comparison baselines:
  - V59 place policy: 56% place rate
  - BC warmstart:     22% place rate
  - IQL (this eval):  ???

Usage:
    # Smoke test (N=20)
    python evaluate_iql_env.py --n_episodes 20

    # Full evaluation (N=200)
    python evaluate_iql_env.py --n_episodes 200
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pickle
import time
import json
from pathlib import Path

import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs, VisionObs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from core.hierarchical_policy import HierarchicalPickPlacePolicy

import torch
from core.iql_dataset import OfflineDataset
from core.iql_agent import IQLAgent

# ---- Paths (same as collect_expert_demos.py / eval_hierarchical.py) ----
GRASP_MODEL = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"
IQL_CHECKPOINT = "/home/w/vla_workspace/outputs/iql_v1/final_model.pt"
EXPERT_DATA = "/home/w/vla_workspace/data/D_expert.npz"
OUTPUT_DIR = Path("/home/w/vla_workspace/outputs/iql_v1")

LIFT_THRESHOLD = 0.03
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42
TARGET_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]


def make_env(vision_mode=False, target_pos_range=None):
    kwargs = dict(reward_type="dense", gravity_comp=True)
    if target_pos_range:
        kwargs["target_pos_range"] = target_pos_range
    kwargs["domain_randomize"] = False
    env = gymnasium.make("PandaVLA-v0", **kwargs)
    if vision_mode:
        env = VisionObs(env, image_size=84)
    else:
        env = FlattenObs(env)
    return env


def load_sb3_model(path, vecnorm_path, vision_mode=False, target_pos_range=None):
    factory = lambda: make_env(vision_mode=vision_mode, target_pos_range=target_pos_range)
    vec_env = DummyVecEnv([factory])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False
    else:
        norm_keys = ["state"] if vision_mode else None
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, norm_obs_keys=norm_keys)
        vec_env.training = False
    model = PPO.load(path, env=vec_env, device="auto")
    return model, vec_env


def load_iql_policy(checkpoint_path, state_mean, state_std, device="cpu",
                    chunk_size=1):
    """Load IQL agent and return closures for action prediction + Q-value eval.

    When chunk_size > 1 (v4), the policy outputs action_dim * chunk_size values
    (a full action chunk). The caller is responsible for executing one action
    per step from the chunk.

    Returns:
        get_action: closure (state_np, deterministic) -> action_np
        get_q_values: closure (state_np, action_chunk_np) -> (q1, q2)
    """
    agent = IQLAgent(state_dim=12, action_dim=8, hidden_dim=256,
                     tau=0.7, beta=3.0, gamma=0.99, polyak=0.005,
                     chunk_size=chunk_size,
                     device=device)
    agent.load(checkpoint_path)
    agent.policy.eval()
    agent.q1_net.eval()
    agent.q2_net.eval()
    state_mean_t = torch.FloatTensor(state_mean).to(device)
    state_std_t = torch.FloatTensor(state_std).to(device)

    @torch.no_grad()
    def get_action(state_np, deterministic=True):
        """state_np: (12,) numpy → action numpy in [-1, 1]

        When chunk_size=1: returns (8,) single action
        When chunk_size>1: returns (action_dim*chunk_size,) flattened chunk
        """
        state = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        state_norm = (state - state_mean_t) / state_std_t
        mean, log_std = agent.policy(state_norm)
        if deterministic:
            action = torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x = normal.sample()
            action = torch.tanh(x)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def get_q_values(state_np, action_chunk_np):
        """Compute Q1, Q2 for (state, action_chunk).

        action_chunk_np: (action_dim * chunk_size,) = (32,) for v4.
        Returns (q1_float, q2_float).
        """
        state = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        action_chunk = torch.FloatTensor(action_chunk_np).unsqueeze(0).to(device)
        state_norm = (state - state_mean_t) / state_std_t
        sa = torch.cat([state_norm, action_chunk], dim=-1)
        q1 = agent.q1_net(sa)
        q2 = agent.q2_net(sa)
        return float(q1.item()), float(q2.item())

    return get_action, get_q_values


def adaptive_chunk_size(dist, far_thresh=0.12, mid_thresh=0.06):
    """v4.1: Adaptive chunk size based on block-target distance.

    Args:
        dist: block-target distance in meters (state[8])
        far_thresh: dist > this → chunk_size=4 (fast approach, temporal consistency)
        mid_thresh: dist > this → chunk_size=2 (transition, partial feedback)
    Returns:
        chunk_size: 4, 2, or 1 (1 = full feedback, fine contact)

    Rationale (from P1 physics investigation):
        - All drift episodes had best_dist in [3.5, 9.5]cm
        - 6/10 drift: terminal jump ~16cm in single step (open-loop amplification)
        - 12cm threshold covers all drift upstream with 2cm buffer
        - 6cm enters fine-contact zone where block is geometrically sensitive
    """
    if dist > far_thresh:
        return 4
    elif dist > mid_thresh:
        return 2
    else:
        return 1


# 7-dim feature names (must match dt_feature_extractor.py / dt_codebook.py)
DT_FEATURE_NAMES = [
    "dist_at_step20", "dist_change_rate", "dist_variance_early",
    "early_drift_signal", "q1_at_step20", "best_dist_early", "has_q_value",
]


def compute_dt_features(early_dists_m, q1_at_step20):
    """Compute the 7-dim DT feature vector from collected early distances.

    Args:
        early_dists_m: list of block-target distances in METERS (first 20
                       place steps). May be shorter if episode placed early.
        q1_at_step20: float or None — Q1 at step 20 (from --log_q_values).

    Returns:
        list of 7 floats in DT_FEATURE_NAMES order, or None if empty.
    """
    if not early_dists_m:
        return None
    dists_cm = np.array(early_dists_m, dtype=float) * 100.0  # m → cm
    n = len(dists_cm)
    dist_at_step20 = float(dists_cm[-1])
    dist_change_rate = float((dists_cm[0] - dists_cm[-1]) / n)
    dist_variance_early = float(np.var(dists_cm))
    early_drift_signal = int(dist_change_rate < 0)
    if q1_at_step20 is not None:
        q1 = float(q1_at_step20)
        has_q = 1
    else:
        q1 = 0.0
        has_q = 0
    best_dist_early = float(np.min(dists_cm))
    return [dist_at_step20, dist_change_rate, dist_variance_early,
            early_drift_signal, q1, best_dist_early, has_q]


def main():
    parser = argparse.ArgumentParser(description="P1: IQL environment evaluation")
    parser.add_argument("--n_episodes", type=int, default=200)
    parser.add_argument("--checkpoint", type=str, default=IQL_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--release_threshold", type=float, default=0.05)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true", default=False)
    # Early abort (P2 Step 1): terminate wandering episodes early
    parser.add_argument("--early_abort", action="store_true", default=False,
                        help="Abort episode if dist doesn't improve for --abort_patience steps")
    parser.add_argument("--abort_patience", type=int, default=30,
                        help="Consecutive place-phase steps without dist improvement before abort")
    parser.add_argument("--abort_drift", type=float, default=0.50,
                        help="Absolute dist (m) above which to abort immediately (catastrophic drift)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path. Default: env_eval_results.json (or env_eval_n{N}.json if N!=200)")
    # Step 2c-ii: Hybrid policy — switch to a different (more reactive) policy near target
    parser.add_argument("--hybrid_checkpoint", type=str, default=None,
                        help="Secondary IQL checkpoint for near-target zone (e.g. v1 n_step=1). "
                             "When dist < hybrid_dist_threshold, uses this policy instead of the main one.")
    parser.add_argument("--hybrid_dist_threshold", type=float, default=0.10,
                        help="Distance threshold (m) below which to switch to hybrid_checkpoint policy")
    # v4: True action chunking — policy outputs h actions, execute open-loop
    parser.add_argument("--chunk_size", type=int, default=1,
                        help="Action chunk size h (v4). When >1, policy outputs h actions "
                             "and executes them open-loop. Every h steps, re-plan.")
    # v4.1: Adaptive chunk — distance-based chunk_size switching (4→2→1)
    parser.add_argument("--adaptive_chunk", action="store_true", default=False,
                        help="v4.1: Enable adaptive chunk sizing. dist>12cm→4, "
                             "dist>6cm→2, dist<=6cm→1. Reuses --chunk_size model output.")
    parser.add_argument("--adaptive_far", type=float, default=0.12,
                        help="Adaptive chunk: dist threshold for chunk_size=4→2 transition")
    parser.add_argument("--adaptive_mid", type=float, default=0.06,
                        help="Adaptive chunk: dist threshold for chunk_size=2→1 transition")
    # Q-value logging at switch points (for v4.2 critic bias diagnosis)
    parser.add_argument("--log_q_values", action="store_true", default=False,
                        help="Log Q1/Q2 values at chunk_size switch points and best_dist "
                             "(for v4.2 critic temporal bias diagnosis)")
    # DT orchestrator: warmup_switch (reachability pre-experiment, Task 3)
    parser.add_argument("--warmup_switch", action="store_true", default=False,
                        help="Reachability pre-experiment mode: first --dt_warmup_steps "
                             "place steps use chunk_size=4 (v4), then switch to "
                             "adaptive_chunk (v4.1). One-time transition per episode.")
    parser.add_argument("--dt_warmup_steps", type=int, default=20,
                        help="Number of warmup steps before switching to adaptive mode "
                             "(used by --warmup_switch)")
    parser.add_argument("--dt_features_path", type=str, default=None,
                        help="Output path for per-episode early features (first 20 place "
                             "steps dist + Q1 at step 20). Written as JSON after eval. "
                             "Orthogonal to execution mode -- works for ALL episodes.")
    # DT orchestrator: online routing (Task 6)
    parser.add_argument("--dt_router", type=str, default=None,
                        help="DT model path (.pkl) for online routing mode. First "
                             "--dt_warmup_steps use v4 (chunk_size=4), then DT predicts "
                             "optimal config and switches to v4.1 if confident.")
    parser.add_argument("--dt_confidence", type=float, default=0.65,
                        help="DT routing confidence threshold (default 0.65). If "
                             "predict_proba_max < this, fallback to v4.")
    parser.add_argument("--dt_leaf_min", type=int, default=0,
                        help="DT leaf node min samples for fallback (default 0 = disabled). "
                             "If leaf_samples < this, fallback to v4.")
    args = parser.parse_args()

    deterministic = not args.stochastic
    is_smoke = args.n_episodes <= 20

    # Near-miss threshold for failure mode classification
    NEAR_MISS_DIST = 0.15  # 15cm: close but not placed

    print("=" * 70)
    print(f"P1: IQL Environment Evaluation ({'SMOKE TEST' if is_smoke else 'FULL'})")
    print("=" * 70)
    print(f"  Episodes:     {args.n_episodes}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Target range: {TARGET_RANGE}")
    print(f"  Seed:         {args.seed}")
    print(f"  Early abort:  {args.early_abort} (patience={args.abort_patience}, drift={args.abort_drift}m)")
    if args.hybrid_checkpoint:
        print(f"  Hybrid policy: {args.hybrid_checkpoint} (dist < {args.hybrid_dist_threshold}m)")
    if args.chunk_size > 1:
        print(f"  Action chunking: chunk_size={args.chunk_size} (v4 true action chunking)")
    if args.adaptive_chunk:
        print(f"  Adaptive chunk: far>{args.adaptive_far}m→4, "
              f"mid>{args.adaptive_mid}m→2, near→1 (v4.1)")
    if args.log_q_values:
        print(f"  Q-value logging: enabled (switch points + best_dist)")
    if args.warmup_switch:
        print(f"  Warmup switch: first {args.dt_warmup_steps} steps v4 (chunk_size=4) "
              f"→ then adaptive_chunk (v4.1)")
    if args.dt_features_path:
        print(f"  DT features output: {args.dt_features_path} (per-episode early dists)")
    if args.dt_router:
        print(f"  DT router: {args.dt_router} (warmup={args.dt_warmup_steps} steps, "
              f"confidence={args.dt_confidence})")
    print()

    # ---- Load normalization stats from training dataset ----
    print("Loading normalization stats from D_expert.npz...")
    dataset = OfflineDataset(data_path=EXPERT_DATA, normalize_states=True,
                             normalize_actions=False)
    state_mean = dataset.state_mean
    state_std = dataset.state_std
    print(f"  State mean: {state_mean}")
    print(f"  State std:  {state_std}")
    print()

    # ---- Load IQL policy ----
    print("Loading IQL policy...")
    iql_get_action, iql_get_q_values = load_iql_policy(
        args.checkpoint, state_mean, state_std, chunk_size=args.chunk_size)
    print(f"  IQL policy loaded (chunk_size={args.chunk_size}).")

    # ---- Load DT router model (Task 6) ----
    dt_model = None
    dt_metadata = None
    dt_classes = None
    # Feature importance dict {feature_name: float} for runtime completeness
    # checks (Task: assert critical features are not zero at step 20).
    dt_feature_importances = None
    if args.dt_router:
        print(f"Loading DT router model: {args.dt_router}")
        try:
            with open(args.dt_router, "rb") as f:
                payload = pickle.load(f)
            dt_model = payload["model"]
            dt_metadata = payload["metadata"]
            dt_classes = list(dt_model.classes_)
            print(f"  DT model loaded. codebook_version={dt_metadata.get('codebook_version')}")
            print(f"  Classes: {dt_classes}")
            print(f"  Feature names: {dt_metadata.get('feature_names')}")
            # Validate feature names match
            if dt_metadata.get("feature_names") != DT_FEATURE_NAMES:
                print(f"  WARNING: feature name mismatch! Expected {DT_FEATURE_NAMES}")
            # Build feature_importance dict. Prefer explicit metadata field if
            # present; otherwise fall back to sklearn's feature_importances_
            # mapped via feature_names (the canonical source for a fitted tree).
            fns = dt_metadata.get("feature_names") or DT_FEATURE_NAMES
            if "feature_importance" in dt_metadata:
                dt_feature_importances = dict(dt_metadata["feature_importance"])
            elif "feature_importances" in dt_metadata:
                dt_feature_importances = dict(dt_metadata["feature_importances"])
            else:
                imps = getattr(dt_model, "feature_importances_", None)
                if imps is not None and len(imps) == len(fns):
                    dt_feature_importances = {n: float(v) for n, v in zip(fns, imps)}
            if dt_feature_importances:
                top = sorted(dt_feature_importances.items(),
                             key=lambda kv: -kv[1])[:3]
                print(f"  Feature importances (top 3): "
                      + ", ".join(f"{n}={v:.4f}" for n, v in top))
        except Exception as e:
            print(f"  WARNING: DT model load failed ({e}), falling back to pure v4 mode")
            print(f"  Continuing with --chunk_size {args.chunk_size} for all episodes")
            args.dt_router = None  # disable routing, use plain v4
    print()

    # ---- Load hybrid policy (Step 2c-ii) ----
    hybrid_get_action = None
    if args.hybrid_checkpoint:
        print(f"Loading hybrid policy: {args.hybrid_checkpoint}")
        hybrid_get_action, _ = load_iql_policy(args.hybrid_checkpoint, state_mean, state_std)
        print("  Hybrid policy loaded.")
    print()

    # ---- Load V59 grasp + place models (for HierarchicalPickPlacePolicy) ----
    print("Loading V59 grasp model...")
    grasp_model, grasp_vec_env = load_sb3_model(
        GRASP_MODEL, GRASP_VECNORM, vision_mode=False,
        target_pos_range=TARGET_RANGE)

    print("Loading V59 place model (for phase detection only)...")
    place_model, place_vec_env = load_sb3_model(
        PLACE_MODEL, PLACE_VECNORM, vision_mode=True,
        target_pos_range=TARGET_RANGE)

    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)
    print("  HierarchicalPickPlacePolicy loaded.")
    print()

    # ---- Create evaluation environment ----
    print("Creating evaluation environment...")
    raw_env = DummyVecEnv([lambda: make_env(vision_mode=False,
                                             target_pos_range=TARGET_RANGE)])
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = args.release_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)
    print(f"  Release threshold: {args.release_threshold}m")
    print()

    np.random.seed(args.seed)
    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    # ---- Run episodes ----
    n_grabbed = 0
    n_entered_place = 0
    n_placed = 0
    all_place_steps = []
    all_final_dists = []
    all_gripper_actions = []  # dim 7 actions during place phase
    all_max_lifts = []
    failed_eps_diagnostics = []
    # Failure mode counters: near_miss, drift, timeout, aborted
    failure_modes = {"near_miss": 0, "drift": 0, "timeout": 0, "aborted": 0}
    n_aborted = 0  # episodes terminated by early abort
    n_timeout = 0  # episodes that ran full MAX_STEPS
    # v4.1: Adaptive chunk statistics
    adaptive_chunk_steps = {"cs4": 0, "cs2": 0, "cs1": 0}  # total steps per chunk_size
    adaptive_switch_events = []  # list of {ep, step, dist, old_cs, new_cs, q1, q2}
    q_at_best = []  # Q values at best_dist for each episode
    n_success_degraded = 0  # v4 success → v4.1 failure (placeholder, needs v4 cross-ref)
    # DT orchestrator: per-episode early features (Task 3 --dt_features_path)
    dt_features_entries = []
    # DT orchestrator: per-episode routing decisions (Task 6 --dt_router)
    dt_routing_log = []
    dt_fallback_count = 0
    # One-shot guard: after the first episode's DT features are computed,
    # verify no high-importance feature is zero (catches the q1_at_step20
    # blind-spot bug where Q-values weren't computed during warmup).
    dt_feature_assertion_done = False

    t0 = time.time()

    for ep in range(args.n_episodes):
        inner.place_mode = False
        inner._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        prev_info = None
        max_lift = 0.0
        block_target_dist = float("inf")
        ep_gripper_actions = []
        ep_place_steps = 0
        ep_hybrid_steps = 0  # Steps that used hybrid policy (Step 2c-ii)
        action_chunk_buffer = []  # v4: buffer of actions from current chunk
        # Early abort tracking (place phase only)
        best_dist = float("inf")
        steps_since_best = 0
        ep_aborted = False
        abort_reason = None
        dist_trajectory = []  # sampled dist for failure diagnostics
        # v4.1: Adaptive chunk per-episode tracking
        ep_chunk_steps = {"cs4": 0, "cs2": 0, "cs1": 0}
        ep_switch_events = []
        # Runtime execution-mode flag. In warmup_switch / dt_router mode we
        # start in v4 mode (adaptive_enabled=False) and flip to True after
        # warmup_steps. In plain v4.1 mode it is True from the start.
        if args.warmup_switch or args.dt_router:
            current_chunk_size = args.chunk_size  # start in v4 fixed-chunk mode
            adaptive_enabled = False
        elif args.adaptive_chunk:
            current_chunk_size = 4
            adaptive_enabled = True
        else:
            current_chunk_size = args.chunk_size
            adaptive_enabled = False
        ep_switched = False  # warmup_switch/dt_router: has v4→v4.1 transition happened
        ep_q_at_best = None  # Q values at closest approach (updated on new best)
        last_chunk_flat = None  # store last chunk for Q-value computation
        # DT orchestrator: early-distance + Q1 collection for --dt_features_path
        # and --dt_router (both need first 20 place-step dists for features)
        ep_early_dists = []  # [dist_step1, ..., dist_step20] (meters)
        ep_early_q1 = None   # Q1 at step 20 (if --log_q_values and chunk available)
        ep_q_value_warning = False  # set True if Q-value computation failed/NaN
        ep_dt_routed = False  # dt_router: has the DT prediction been made for this ep

        for step in range(MAX_STEPS):
            phase = policy._detect_phase(prev_info)

            # Enter place phase
            if phase == "place" and first_place_step is None:
                first_place_step = step
                inner.place_mode = True
                inner._place_gravcomp_active = True
                inner.snap_block_to_hand()
                inner._arm_target = inner.data.qpos[inner._arm_qpos_adrs].copy()
                inner._gripper_target = float(
                    inner.data.qpos[inner._finger_qpos_adrs].mean())
                inner.reward_type = "place_only"
                inner._place_approach_bonus_given = False
                inner._place_proximity_15_given = False
                inner._place_proximity_10_given = False
                inner._place_success = False
                inner._prev_block_target_dist = None
                inner._prev_block_height = None
                inner._use_gripper_target_check = True
                flatten_wrapper = raw_env.envs[0]
                inner_obs = inner._get_obs()
                raw_obs = flatten_wrapper.observation(inner_obs)[np.newaxis, :].astype(np.float32)

            if phase == "place":
                # --- IQL POLICY (overrides V59 place model) ---
                vision_obs = place_vision.observation(inner._get_obs())
                state_12 = vision_obs["state"]  # (12,)

                # DT orchestrator (--dt_features_path / --dt_router): collect
                # per-step distance for the first 20 place steps of EVERY
                # episode. Also grab Q1 at step 20 when --log_q_values OR
                # --dt_router (q1_at_step20 is the DT's most important feature;
                # without it the router is blind during online evaluation).
                if (args.dt_features_path or args.dt_router) and len(ep_early_dists) < 20:
                    ep_early_dists.append(block_target_dist)
                    if (len(ep_early_dists) == 20
                            and (args.log_q_values or args.dt_router)
                            and last_chunk_flat is not None):
                        # Reuse the IQL critic forward pass (same logic as
                        # --log_q_values). Impute to 0.0 + warning on failure.
                        try:
                            q1_step20, _ = iql_get_q_values(state_12, last_chunk_flat)
                            if q1_step20 is None or np.isnan(q1_step20) or np.isinf(q1_step20):
                                q1_step20 = 0.0
                                ep_q_value_warning = True
                            ep_early_q1 = q1_step20
                        except Exception as qe:
                            ep_early_q1 = 0.0
                            ep_q_value_warning = True
                            print(f"  [ep {ep}] Q-value computation failed at "
                                  f"step 20: {qe}")
                    elif (len(ep_early_dists) == 20
                            and (args.log_q_values or args.dt_router)
                            and last_chunk_flat is None):
                        # No chunk available yet (shouldn't normally happen in
                        # v4 warmup since re-plan fires every chunk_size steps).
                        ep_early_q1 = 0.0
                        ep_q_value_warning = True

                # warmup_switch: one-time v4→v4.1 transition per episode.
                # Triggered when we have executed --dt_warmup_steps place steps
                # (ep_place_steps counts steps already done). Switch semantics
                # per spec table: CLEAR action_chunk_buffer + current_chunk_size;
                # PRESERVE best_dist, steps_since_best, last_chunk_flat,
                # ep_chunk_steps, ep_switch_events; no agent hidden state (IQL
                # is stateless feedforward).
                if (args.warmup_switch and not ep_switched
                        and ep_place_steps >= args.dt_warmup_steps):
                    action_chunk_buffer = []  # CLEAR: force re-plan with new mode
                    # Set current_chunk_size to what adaptive would pick now, so
                    # the adaptive branch does not log a spurious chunk switch.
                    current_chunk_size = adaptive_chunk_size(
                        block_target_dist, args.adaptive_far, args.adaptive_mid)
                    adaptive_enabled = True
                    ep_switched = True
                    ep_switch_events.append({
                        "ep": ep, "place_step": ep_place_steps,
                        "type": "warmup_switch",
                        "dist_cm": round(block_target_dist * 100, 2),
                    })

                # DT router (--dt_router): per-episode online routing. After
                # warmup_steps, compute 7-dim features from ep_early_dists,
                # query DT model for predicted config + confidence. Dual-signal
                # fallback: confidence < threshold OR leaf node samples < 10 →
                # stay v4. If predicted v4.1 and confident → switch to adaptive.
                if (args.dt_router and dt_model is not None and not ep_dt_routed
                        and ep_place_steps >= args.dt_warmup_steps):
                    ep_dt_routed = True
                    feat_vec = compute_dt_features(ep_early_dists, ep_early_q1)
                    if feat_vec is not None and not np.any(np.isnan(feat_vec)):
                        feat_arr = np.array([feat_vec])
                        predicted = dt_model.predict(feat_arr)[0]
                        proba = dt_model.predict_proba(feat_arr)[0]
                        conf_max = float(np.max(proba))
                        # Leaf node sample count (sparse leaf fallback)
                        leaf_id = dt_model.apply(feat_arr)[0]
                        leaf_samples = int(dt_model.tree_.n_node_samples[leaf_id])

                        # Dual-signal fallback decision
                        fallback_reason = None
                        if conf_max < args.dt_confidence:
                            fallback_reason = "low_confidence"
                        elif args.dt_leaf_min > 0 and leaf_samples < args.dt_leaf_min:
                            fallback_reason = "sparse_leaf"

                        actual_config = "v4"
                        switch_happened = False
                        # Switch when model predicts anything other than v4
                        # (baseline). Handles both v1 model (classes ["v4",
                        # "v4.1"]) and v2 model (classes ["v4",
                        # "warmup_switch"]) — switch trigger is class-agnostic.
                        if fallback_reason is None and predicted != "v4":
                            # Switch to adaptive (same semantics as warmup_switch)
                            action_chunk_buffer = []
                            current_chunk_size = adaptive_chunk_size(
                                block_target_dist, args.adaptive_far,
                                args.adaptive_mid)
                            adaptive_enabled = True
                            ep_switched = True
                            actual_config = predicted
                            switch_happened = True
                            ep_switch_events.append({
                                "ep": ep, "place_step": ep_place_steps,
                                "type": "dt_router_switch",
                                "dist_cm": round(block_target_dist * 100, 2),
                                "predicted": predicted,
                                "confidence": round(conf_max, 4),
                            })
                        else:
                            if fallback_reason:
                                dt_fallback_count += 1

                        dt_routing_log.append({
                            "ep": ep,
                            "predicted_config": predicted,
                            "predict_proba": {str(c): round(float(p), 4)
                                              for c, p in zip(dt_classes, proba)},
                            "confidence": round(conf_max, 4),
                            "leaf_samples": leaf_samples,
                            "actual_config_used": actual_config,
                            "switch_happened": switch_happened,
                            "fallback_reason": fallback_reason,
                            "q_value_warning": ep_q_value_warning,
                            "features_snapshot": dict(zip(DT_FEATURE_NAMES,
                                                          [round(float(v), 4) for v in feat_vec])),
                        })
                        # Feature completeness assertion (one-shot, after the
                        # first routed episode): if any feature with importance
                        # > 0.1 is zero, the online feature pipeline has a gap.
                        if not dt_feature_assertion_done:
                            dt_feature_assertion_done = True
                            if dt_feature_importances:
                                for name, imp in dt_feature_importances.items():
                                    if imp > 0.1 and name in DT_FEATURE_NAMES:
                                        fidx = DT_FEATURE_NAMES.index(name)
                                        if float(feat_vec[fidx]) == 0.0:
                                            print(f"WARNING: Critical feature "
                                                  f"{name} is zero — check "
                                                  f"feature pipeline")
                    else:
                        # NaN features → fallback to v4
                        dt_fallback_count += 1
                        dt_routing_log.append({
                            "ep": ep,
                            "predicted_config": None,
                            "predict_proba": {},
                            "confidence": 0.0,
                            "leaf_samples": 0,
                            "actual_config_used": "v4",
                            "switch_happened": False,
                            "fallback_reason": "nan_feature",
                            "q_value_warning": ep_q_value_warning,
                            "features_snapshot": None,
                        })

                use_hybrid = (hybrid_get_action is not None and
                              block_target_dist < args.hybrid_dist_threshold)

                if use_hybrid:
                    # Hybrid policy: always single-step (hybrid is chunk_size=1)
                    action = hybrid_get_action(state_12, deterministic=deterministic)
                    ep_hybrid_steps += 1
                    action_chunk_buffer = []  # Clear buffer when switching to hybrid
                elif adaptive_enabled:
                    # v4.1: Adaptive chunk — model always outputs 4 actions,
                    # but we execute only `desired_chunk` of them before re-planning.
                    # chunk_size=1 → re-plan every step (full feedback)
                    # chunk_size=2 → execute 2 open-loop, then re-plan
                    # chunk_size=4 → execute 4 open-loop, then re-plan
                    desired_chunk = adaptive_chunk_size(
                        block_target_dist, args.adaptive_far, args.adaptive_mid)

                    # Detect chunk_size switch for logging
                    if desired_chunk != current_chunk_size:
                        switch_event = {
                            "ep": ep, "place_step": ep_place_steps,
                            "dist_cm": round(block_target_dist * 100, 2),
                            "old_cs": current_chunk_size, "new_cs": desired_chunk,
                        }
                        # Log Q values at switch point (for v4.2 critic bias diagnosis)
                        if args.log_q_values and last_chunk_flat is not None:
                            q1, q2 = iql_get_q_values(state_12, last_chunk_flat)
                            switch_event["q1"] = round(q1, 4)
                            switch_event["q2"] = round(q2, 4)
                        ep_switch_events.append(switch_event)
                        current_chunk_size = desired_chunk

                    # Re-plan when: buffer empty, OR chunk_size=1 (always re-plan),
                    # OR desired_chunk < remaining buffer (want finer control)
                    need_replan = (not action_chunk_buffer
                                   or desired_chunk == 1
                                   or desired_chunk < len(action_chunk_buffer))
                    if need_replan:
                        chunk_flat = iql_get_action(state_12, deterministic=deterministic)
                        last_chunk_flat = chunk_flat  # store for Q-value logging
                        actions_all = chunk_flat.reshape(args.chunk_size, -1)  # (4, 8)
                        action_chunk_buffer = actions_all[:desired_chunk].tolist()

                    action = np.array(action_chunk_buffer.pop(0), dtype=np.float32)
                    # Track chunk_size usage
                    ep_chunk_steps[f"cs{desired_chunk}"] += 1

                    # Log Q at best_dist (update every time new best is found)
                    if (args.log_q_values and block_target_dist < best_dist
                            and last_chunk_flat is not None):
                        q1, q2 = iql_get_q_values(state_12, last_chunk_flat)
                        ep_q_at_best = {
                            "ep": ep, "dist_cm": round(block_target_dist * 100, 2),
                            "chunk_size": desired_chunk,
                            "q1": round(q1, 4), "q2": round(q2, 4),
                        }
                elif args.chunk_size > 1:
                    # v4: True action chunking — policy outputs h actions,
                    # execute them open-loop one per step, re-plan every h steps
                    if not action_chunk_buffer:
                        # Buffer empty: call policy to generate new chunk
                        chunk_flat = iql_get_action(state_12, deterministic=deterministic)
                        last_chunk_flat = chunk_flat
                        # Reshape (action_dim*h,) → list of h actions, each (action_dim,)
                        action_chunk_buffer = chunk_flat.reshape(args.chunk_size, -1).tolist()
                    # Execute next action from buffer (open-loop within chunk)
                    action = np.array(action_chunk_buffer.pop(0), dtype=np.float32)
                    # In warmup_switch / dt_router mode, track chunk_size usage
                    # during the v4 warmup phase so ep_chunk_steps accumulates
                    # across modes.
                    if args.warmup_switch or args.dt_router:
                        ep_chunk_steps[f"cs{args.chunk_size}"] += 1
                else:
                    # Original single-step execution (v1/v2/v3)
                    action = iql_get_action(state_12, deterministic=deterministic)

                action = action[np.newaxis, :]  # (1, 8)
                ep_gripper_actions.append(float(action[0, 7]))
                ep_place_steps += 1

                # Track dist improvement for early abort
                if block_target_dist < best_dist:
                    best_dist = block_target_dist
                    steps_since_best = 0
                else:
                    steps_since_best += 1
                # Sample dist trajectory (every 10 steps) for diagnostics
                if ep_place_steps % 10 == 0:
                    dist_trajectory.append(round(block_target_dist * 100, 1))

                # Early abort check
                if args.early_abort:
                    if steps_since_best >= args.abort_patience:
                        ep_aborted = True
                        abort_reason = f"no_improvement_{args.abort_patience}steps"
                        break
                    if block_target_dist > args.abort_drift:
                        ep_aborted = True
                        abort_reason = f"drift>{args.abort_drift}m"
                        break
            else:
                # --- Grasp phase: V59 grasp model ---
                raw_obs_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_grasp[0, 8:11]
                raw_obs_grasp[0, 15] = np.linalg.norm(
                    block_pos - np.array([0.5, 0.3, 0.2]))
                obs = grasp_vec_env.normalize_obs(raw_obs_grasp)
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(info[0].get("block_target_distance", block_target_dist))
            lift = max(0.0, float(info[0].get("block_height", 0.0)) - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            if done[0]:
                break

        if ep_aborted:
            n_aborted += 1
        elif first_place_step is not None and ep_place_steps >= MAX_STEPS - first_place_step:
            n_timeout += 1

        # v4.1 / warmup_switch: Accumulate adaptive chunk stats (covers both
        # pure v4.1 and the post-switch adaptive portion of warmup_switch)
        if args.adaptive_chunk or args.warmup_switch or args.dt_router:
            for k in adaptive_chunk_steps:
                adaptive_chunk_steps[k] += ep_chunk_steps[k]
            adaptive_switch_events.extend(ep_switch_events)
        if args.log_q_values and ep_q_at_best is not None:
            q_at_best.append(ep_q_at_best)

        # Episode finished
        if first_place_step is not None and max_lift > LIFT_THRESHOLD:
            n_entered_place += 1
            n_grabbed += 1
            placed = block_target_dist < args.release_threshold
            if placed:
                n_placed += 1
            all_place_steps.append(ep_place_steps)
            all_final_dists.append(block_target_dist)
            all_max_lifts.append(max_lift)
            all_gripper_actions.extend(ep_gripper_actions)

            # Classify failure mode
            if not placed:
                if ep_aborted:
                    if block_target_dist < NEAR_MISS_DIST:
                        failure_modes["near_miss"] += 1
                    else:
                        failure_modes["drift"] += 1
                elif ep_place_steps >= MAX_STEPS - (first_place_step or 0):
                    failure_modes["timeout"] += 1
                    if block_target_dist < NEAR_MISS_DIST:
                        failure_modes["near_miss"] += 1
                        failure_modes["timeout"] -= 1
                else:
                    if block_target_dist < NEAR_MISS_DIST:
                        failure_modes["near_miss"] += 1
                    else:
                        failure_modes["drift"] += 1

            status = "PLACE" if placed else "FAIL"
            abort_tag = f" [ABORT:{abort_reason}]" if ep_aborted else ""
            hybrid_tag = f" [hybrid:{ep_hybrid_steps}st]" if hybrid_get_action is not None else ""
            print(f"Ep {ep:3d}: place_steps={ep_place_steps:3d}, "
                  f"lift={max_lift*100:.1f}cm, dist={block_target_dist*100:.1f}cm, "
                  f"{status}{abort_tag}{hybrid_tag}")

            # Save diagnostics for failed episodes
            if not placed and len(failed_eps_diagnostics) < 10:
                diag = {
                    "ep": ep,
                    "place_steps": ep_place_steps,
                    "hybrid_steps": ep_hybrid_steps,
                    "max_lift_cm": max_lift * 100,
                    "final_dist_cm": block_target_dist * 100,
                    "best_dist_cm": best_dist * 100 if best_dist != float("inf") else None,
                    "aborted": ep_aborted,
                    "abort_reason": abort_reason,
                    "dist_trajectory_cm": dist_trajectory,
                    "gripper_action_mean": float(np.mean(ep_gripper_actions)) if ep_gripper_actions else 0,
                    "gripper_action_std": float(np.std(ep_gripper_actions)) if ep_gripper_actions else 0,
                    "gripper_action_min": float(np.min(ep_gripper_actions)) if ep_gripper_actions else 0,
                    "gripper_action_max": float(np.max(ep_gripper_actions)) if ep_gripper_actions else 0,
                }
                if args.adaptive_chunk or args.warmup_switch:
                    diag["chunk_steps"] = ep_chunk_steps
                    diag["switch_events"] = ep_switch_events
                failed_eps_diagnostics.append(diag)

            # DT orchestrator (--dt_features_path): record per-episode early
            # features for EVERY episode that entered place (placed + failed).
            # Orthogonal to execution mode; outcome is 3-class for the DT router.
            if args.dt_features_path:
                if block_target_dist < args.release_threshold:
                    outcome = "placed"
                elif block_target_dist < NEAR_MISS_DIST:
                    outcome = "near_miss"
                else:
                    outcome = "drift"
                dt_features_entries.append({
                    "ep": ep,
                    "early_dists": [round(d * 100, 2) for d in ep_early_dists],
                    "q1_at_step20": (round(ep_early_q1, 4)
                                     if ep_early_q1 is not None else None),
                    "outcome": outcome,
                    "place_steps": ep_place_steps,
                    "final_dist_cm": round(block_target_dist * 100, 2),
                    "best_dist_cm": (round(best_dist * 100, 2)
                                     if best_dist != float("inf") else None),
                })
        else:
            print(f"Ep {ep:3d}: GRASP FAIL (lift={max_lift*100:.1f}cm)")

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{ep+1}/{args.n_episodes}] {elapsed:.0f}s, "
                  f"place_rate={100*n_placed/max(1,n_entered_place):.1f}%, "
                  f"aborted={n_aborted}")

    elapsed = time.time() - t0

    # ---- Report results ----
    place_rate = n_placed / max(1, n_entered_place)
    n_failed = n_entered_place - n_placed

    print(f"\n{'='*70}")
    print(f"IQL Environment Evaluation Complete ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"Total episodes:      {args.n_episodes}")
    print(f"Grasp success:       {n_grabbed}/{args.n_episodes} "
          f"({100*n_grabbed/args.n_episodes:.1f}%)")
    print(f"Entered place:       {n_entered_place}/{args.n_episodes}")
    print(f"IQL place rate:      {n_placed}/{n_entered_place} "
          f"({100*place_rate:.1f}%)")
    if args.early_abort:
        print(f"Early aborts:        {n_aborted}/{n_entered_place} "
              f"(patience={args.abort_patience}, drift={args.abort_drift}m)")
    if args.warmup_switch:
        reachability_passed = place_rate >= 0.719
        print(f"Warmup switch:       mode=warmup_switch, warmup_steps={args.dt_warmup_steps}")
        print(f"Reachability gate:   place_rate={100*place_rate:.1f}% "
              f"(>=71.9% required) → "
              f"{'PASSED' if reachability_passed else 'FAILED'}")
    if args.dt_router:
        n_routed = len(dt_routing_log)
        n_switched = sum(1 for r in dt_routing_log if r.get("switch_happened"))
        fallback_rate = (dt_fallback_count / max(1, n_routed))
        print(f"DT router:           mode=dt_router, warmup={args.dt_warmup_steps}, "
              f"confidence={args.dt_confidence}, leaf_min={args.dt_leaf_min}")
        print(f"  Routed:            {n_routed} episodes")
        print(f"  Switched to v4.1:  {n_switched}")
        print(f"  Fallbacks:         {dt_fallback_count} ({100*fallback_rate:.1f}%)")
        if fallback_rate > 0.30:
            print(f"  ⚠ WARNING: fallback rate > 30%, blind spot coverage issue")
    print()

    # Comparison
    print("Comparison:")
    print(f"  BC warmstart:  22%")
    print(f"  V59:           56%")
    print(f"  IQL:           {100*place_rate:.1f}%")
    print()

    # Failure mode distribution
    if n_failed > 0:
        print(f"Failure mode distribution ({n_failed} failures):")
        for mode, count in failure_modes.items():
            if count > 0:
                print(f"  {mode:12s}: {count:3d} ({100*count/n_failed:.1f}%)")
        print()

    if all_place_steps:
        print("Place phase statistics:")
        steps_arr = np.array(all_place_steps)
        dists_arr = np.array(all_final_dists)
        print(f"  Place steps:   mean={steps_arr.mean():.1f}, "
              f"std={steps_arr.std():.1f}, "
              f"min={steps_arr.min()}, max={steps_arr.max()}")
        print(f"  Final dist:    mean={dists_arr.mean()*100:.1f}cm, "
              f"std={dists_arr.std()*100:.1f}cm, "
              f"min={dists_arr.min()*100:.1f}cm, max={dists_arr.max()*100:.1f}cm")

    if all_gripper_actions:
        gripper_arr = np.array(all_gripper_actions)
        print(f"\nGripper action (dim 7) statistics:")
        print(f"  Mean:   {gripper_arr.mean():.4f}")
        print(f"  Std:    {gripper_arr.std():.4f}")
        print(f"  Min:    {gripper_arr.min():.4f}")
        print(f"  Max:    {gripper_arr.max():.4f}")
        # Histogram
        hist, edges = np.histogram(gripper_arr, bins=10, range=(-1, 1))
        print(f"  Histogram (-1 to 1, 10 bins):")
        for i, (h, e) in enumerate(zip(hist, edges)):
            print(f"    [{e:.1f}, {edges[i+1]:.1f}): {h} ({100*h/len(gripper_arr):.1f}%)")

    # Diagnostic dump if place rate < 30%
    if place_rate < 0.30 and failed_eps_diagnostics:
        print(f"\n{'='*70}")
        print(f"⚠ DIAGNOSTIC: Place rate {place_rate*100:.1f}% < 30%")
        print(f"{'='*70}")
        print(f"Failed episode diagnostics (first {len(failed_eps_diagnostics)}):")
        for d in failed_eps_diagnostics:
            print(f"  Ep {d['ep']:3d}: steps={d['place_steps']}, "
                  f"dist={d['final_dist_cm']:.1f}cm, "
                  f"gripper=[{d['gripper_action_min']:.2f}, {d['gripper_action_max']:.2f}], "
                  f"gripper_mean={d['gripper_action_mean']:.3f}")
        print(f"\nLikely causes (by probability):")
        print(f"  1. Observation distribution drift (normalization mismatch)")
        print(f"  2. Action execution delay/damping mismatch")
        print(f"  3. Contact dynamics differences")
        print(f"  4. Policy overfitting to offline distribution")

    # Save results
    results = {
        "n_episodes": args.n_episodes,
        "n_grabbed": n_grabbed,
        "n_entered_place": n_entered_place,
        "n_placed": n_placed,
        "place_rate": place_rate,
        "grasp_rate": n_grabbed / args.n_episodes,
        "n_aborted": n_aborted,
        "n_timeout": n_timeout,
        "failure_modes": failure_modes,
        "place_steps": {
            "mean": float(np.mean(all_place_steps)) if all_place_steps else 0,
            "std": float(np.std(all_place_steps)) if all_place_steps else 0,
            "min": int(np.min(all_place_steps)) if all_place_steps else 0,
            "max": int(np.max(all_place_steps)) if all_place_steps else 0,
        },
        "final_dists": {
            "mean_cm": float(np.mean(all_final_dists) * 100) if all_final_dists else 0,
            "std_cm": float(np.std(all_final_dists) * 100) if all_final_dists else 0,
            "min_cm": float(np.min(all_final_dists) * 100) if all_final_dists else 0,
            "max_cm": float(np.max(all_final_dists) * 100) if all_final_dists else 0,
        },
        "gripper_actions": {
            "mean": float(np.mean(all_gripper_actions)) if all_gripper_actions else 0,
            "std": float(np.std(all_gripper_actions)) if all_gripper_actions else 0,
            "min": float(np.min(all_gripper_actions)) if all_gripper_actions else 0,
            "max": float(np.max(all_gripper_actions)) if all_gripper_actions else 0,
        },
        "failed_diagnostics": failed_eps_diagnostics,
        "config": {
            "checkpoint": args.checkpoint,
            "deterministic": deterministic,
            "seed": args.seed,
            "release_threshold": args.release_threshold,
            "target_range": TARGET_RANGE,
            "early_abort": args.early_abort,
            "abort_patience": args.abort_patience,
            "abort_drift": args.abort_drift,
            "hybrid_checkpoint": args.hybrid_checkpoint,
            "hybrid_dist_threshold": args.hybrid_dist_threshold,
            "chunk_size": args.chunk_size,
            "adaptive_chunk": args.adaptive_chunk,
            "adaptive_far": args.adaptive_far,
            "adaptive_mid": args.adaptive_mid,
            "log_q_values": args.log_q_values,
            "warmup_switch": args.warmup_switch,
            "dt_warmup_steps": args.dt_warmup_steps,
            "dt_features_path": args.dt_features_path,
            "dt_router": args.dt_router,
            "dt_confidence": args.dt_confidence,
        },
    }

    # warmup_switch: tag the output with mode + reachability gate result
    if args.warmup_switch:
        results["mode"] = "warmup_switch"
        results["reachability_passed"] = place_rate >= 0.719
        results["dt_warmup_steps"] = args.dt_warmup_steps

    # dt_router: tag the output with mode + routing stats
    if args.dt_router:
        results["mode"] = "dt_router"
        n_routed = len(dt_routing_log)
        n_switched = sum(1 for r in dt_routing_log if r.get("switch_happened"))
        fallback_rate = dt_fallback_count / max(1, n_routed)
        results["dt_routing_stats"] = {
            "n_routed": n_routed,
            "n_switched_to_v4_1": n_switched,
            "n_fallback": dt_fallback_count,
            "fallback_rate": round(fallback_rate, 4),
            "blind_spot_warning": fallback_rate > 0.30,
            "routing_log": dt_routing_log,
        }

    # v4.1 / warmup_switch / dt_router: Add adaptive chunk diagnostics
    if args.adaptive_chunk or args.warmup_switch or args.dt_router:
        results["adaptive_chunk_stats"] = {
            "total_steps_per_chunk_size": adaptive_chunk_steps,
            "n_switch_events": len(adaptive_switch_events),
            "switch_events_sample": adaptive_switch_events[:50],  # cap for file size
        }
    if args.log_q_values:
        results["q_value_diagnostics"] = {
            "q_at_best_dist": q_at_best[:100],  # cap for file size
            "n_q_logged": len(q_at_best),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else (
        OUTPUT_DIR / f"env_eval_n{args.n_episodes}.json"
        if args.n_episodes != 200 else OUTPUT_DIR / "env_eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # DT orchestrator: write per-episode early features for --dt_features_path.
    # Orthogonal to execution mode — records first 20 place-step dists + Q1 at
    # step 20 for every episode that entered place. Used by dt_feature_extractor
    # (Task 1) to build the training codebook from v4 warmup trajectories.
    if args.dt_features_path:
        dt_path = Path(args.dt_features_path)
        dt_path.parent.mkdir(parents=True, exist_ok=True)
        dt_payload = {
            "n_episodes": len(dt_features_entries),
            "warmup_steps": args.dt_warmup_steps if args.warmup_switch else None,
            "log_q_values": args.log_q_values,
            "chunk_size": args.chunk_size,
            "adaptive_chunk": args.adaptive_chunk,
            "warmup_switch": args.warmup_switch,
            "entries": dt_features_entries,
        }
        with open(dt_path, "w") as f:
            json.dump(dt_payload, f, indent=2, default=str)
        print(f"DT features saved to {dt_path} "
              f"({len(dt_features_entries)} episodes)")


if __name__ == "__main__":
    main()
