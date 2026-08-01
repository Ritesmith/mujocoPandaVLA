#!/usr/bin/env python3
"""Evaluate the hierarchical pick-and-place policy.

Loads the v3 grasp model and a place sub-policy, wraps them in
HierarchicalPickPlacePolicy, and runs N episodes tracking:
  - Grab  : max lift height > 3cm  (block was picked up)
  - Place : final block-target dist < 5cm  (block was placed)
  - Pick+Place : both conditions in the same episode (in order)

Output format mirrors analyze_pickplace.py.

Usage:
    python eval_hierarchical.py
    python eval_hierarchical.py --place_model outputs/place_policy/best/best_model.zip
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import gymnasium
import gym_env  # noqa: F401  registers PandaVLA-v0
from gym_env.wrappers import FlattenObs
from hierarchical_policy import HierarchicalPickPlacePolicy

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# v3 grasp model (65% grab rate, 0% place rate)
GRASP_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/best/best_model.zip"
GRASP_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v3/vec_normalize.pkl"

# Default place model path (may not exist yet -> falls back to grasp model)
PLACE_MODEL_PATH = "/home/w/vla_workspace/outputs/place_policy_v17/best/best_model.zip"
PLACE_VECNORM_PATH = "/home/w/vla_workspace/outputs/place_policy_v17/best/vec_normalize.pkl"

LIFT_THRESHOLD = 0.03   # m above table (table_z=0.22)
PLACE_THRESHOLD = 0.05  # m block-target distance
TABLE_Z = 0.22
MAX_STEPS = 500
N_EPISODES = 20
SEED = 42
# Match hierarchical_policy.GRASP_TO_PLACE_LIFT (0.02): an episode "enters
# the place phase" when lift exceeds this. LIFT_THRESHOLD (0.03) is a
# stricter grab-success metric. The gap [0.02, 0.03] creates "gray zone"
# episodes that enter place but are not counted as grabbed.
PHASE_SWITCH_LIFT = 0.02  # m: matches hierarchical_policy threshold


def make_env(vision_mode=False, include_target_pos=False, target_pos=None,
             target_pos_range=None, domain_randomize=True):
    kwargs = dict(reward_type="dense", gravity_comp=True)
    if target_pos is not None:
        kwargs["target_pos"] = target_pos
    if target_pos_range is not None:
        kwargs["target_pos_range"] = target_pos_range
    kwargs["domain_randomize"] = domain_randomize
    env = gymnasium.make("PandaVLA-v0", **kwargs)
    if vision_mode:
        from gym_env.wrappers import VisionObs
        env = VisionObs(env, image_size=84)
    else:
        env = FlattenObs(env, include_target_pos=include_target_pos)
    return env


def load_model(model_path, vecnorm_path, vision_mode=False, include_target_pos=False, target_pos=None,
               target_pos_range=None, domain_randomize=True):
    """Load an SB3 PPO model with its VecNormalize stats."""
    env_factory = lambda: make_env(vision_mode=vision_mode,
                                    include_target_pos=include_target_pos,
                                    target_pos=target_pos,
                                    target_pos_range=target_pos_range,
                                    domain_randomize=domain_randomize)
    vec_env = DummyVecEnv([env_factory])
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False
    else:
        # For vision (Dict) obs, only normalize the "state" key; pass image
        # through unchanged (uint8 [0,255] should not be running-mean scaled).
        norm_obs_keys = ["state"] if vision_mode else None
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False,
                               clip_obs=10.0, norm_obs_keys=norm_obs_keys)
        vec_env.training = False
    # PPO.load infers policy type (MultiInputPolicy for vision, MlpPolicy
    # otherwise) from the saved model file.
    model = PPO.load(model_path, env=vec_env, device="auto")
    return model, vec_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate hierarchical pick-place")
    parser.add_argument('--place_model', type=str, default=PLACE_MODEL_PATH,
                        help='Path to place policy .zip')
    parser.add_argument('--place_vecnorm', type=str, default=PLACE_VECNORM_PATH,
                        help='Path to place policy vec_normalize.pkl')
    parser.add_argument('--grasp_model', type=str, default=None,
                        help='Path to grasp policy .zip (default: v3 hardcoded)')
    parser.add_argument('--grasp_vecnorm', type=str, default=None,
                        help='Path to grasp vec_normalize.pkl')
    parser.add_argument('--n_episodes', type=int, default=N_EPISODES)
    parser.add_argument('--num_episodes', type=int, default=None,
                        help='Number of episodes (alias for --n_episodes)')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS)
    parser.add_argument('--release_threshold', type=float, default=0.05,
                        help='Release distance threshold (m) for place phase')
    parser.add_argument('--release_height', type=float, default=0.0,
                        help='Max block height above table (m) to allow release. 0=no gate')
    parser.add_argument('--freeze_arm_on_release', action='store_true',
                        help='Freeze arm movement after gripper opens to prevent pushing block')
    parser.add_argument('--include_target_pos', action='store_true',
                        help='Include target_pos in observation (19-dim). '
                             'Required for v15+ models. Default: False (16-dim, for v14 and earlier)')
    parser.add_argument('--target_pos', type=str, default=None,
                        help='Override target position as "x,y,z". '
                             'Default: use env default [0.5, 0.3, 0.2]')
    parser.add_argument('--target_pos_range', type=str, default=None,
                        help='Target position range as "x_low,y_low,z_low,x_high,y_high,z_high". '
                             'E.g. "0.35,-0.15,0.22,0.65,0.15,0.22".')
    parser.add_argument('--vision_mode', action='store_true',
                        help='Use vision observation (image + state Dict) for '
                             'the place model. Grasp model always uses state '
                             '(16-dim FlattenObs).')
    parser.add_argument('--domain_randomize', action='store_true',
                        help='Enable domain randomization')
    parser.add_argument('--no_domain_randomize', action='store_true',
                        help='Disable domain randomization')
    parser.add_argument('--stochastic_place', action='store_true',
                        help='Use stochastic (deterministic=False) actions for '
                             'the place model. Default: deterministic=True.')
    parser.add_argument('--action_scale', type=float, default=1.0,
                        help='Scale factor for place model actions. '
                             'Default: 1.0 (no scaling).')
    parser.add_argument('--place_log_std', type=float, default=None,
                        help='Override place model log_std (e.g. -2.3 for '
                             'std=0.1). Implies --stochastic_place.')
    args = parser.parse_args()

    if args.num_episodes is not None:
        args.n_episodes = args.num_episodes

    domain_randomize = not args.no_domain_randomize and args.domain_randomize
    if args.no_domain_randomize:
        domain_randomize = False
    elif args.domain_randomize:
        domain_randomize = True
    else:
        domain_randomize = False

    # Parse target position
    target_pos = None
    if args.target_pos:
        target_pos = np.array([float(v) for v in args.target_pos.split(',')])
        print(f"Target position override: {target_pos}")

    # Parse target position range
    target_pos_range = None
    if args.target_pos_range:
        vals = [float(v) for v in args.target_pos_range.split(',')]
        assert len(vals) == 6, "target_pos_range must be 6 values: x_low,y_low,z_low,x_high,y_high,z_high"
        target_pos_range = [[vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]]
        print(f"Target position range: x=[{vals[0]:.2f},{vals[3]:.2f}], "
              f"y=[{vals[1]:.2f},{vals[4]:.2f}], z=[{vals[2]:.2f},{vals[5]:.2f}]")

    print(f"Domain randomization: {'enabled' if domain_randomize else 'disabled'}")

    # ---- Load grasp model ----
    gm_path = args.grasp_model or GRASP_MODEL_PATH
    gv_path = args.grasp_vecnorm or GRASP_VECNORM_PATH
    print(f"Loading grasp model: {gm_path}")
    grasp_model, grasp_vec_env = load_model(
        gm_path, gv_path, include_target_pos=False,
        target_pos_range=target_pos_range, domain_randomize=domain_randomize
    )

    # ---- Load place model (or fall back to grasp model) ----
    # The grasp model ALWAYS uses state-mode (16-dim FlattenObs). Only the
    # place model can use vision_mode (Dict {image, state} -> MultiInputPolicy).
    place_fallback = False
    if os.path.exists(args.place_model):
        print(f"Loading place model: {args.place_model}")
        place_model, place_vec_env = load_model(
            args.place_model, args.place_vecnorm,
            vision_mode=args.vision_mode,
            include_target_pos=args.include_target_pos,
            target_pos=target_pos,
            target_pos_range=target_pos_range,
            domain_randomize=domain_randomize
        )
    else:
        print(f"WARNING: place model not found at {args.place_model}")
        print("  Falling back to grasp model for the place phase.")
        place_model = grasp_model
        place_vec_env = grasp_vec_env
        place_fallback = True

    # ---- Override place model log_std if requested ----
    if args.place_log_std is not None:
        import torch
        with torch.no_grad():
            place_model.policy.log_std.data.fill_(args.place_log_std)
        new_std = float(place_model.policy.log_std.data.exp().mean().item())
        print(f"Override place log_std -> {args.place_log_std:.4f} "
              f"(std={new_std:.4f})")
        args.stochastic_place = True  # log_std override requires stochastic

    # ---- Build hierarchical policy ----
    policy = HierarchicalPickPlacePolicy(
        grasp_model, place_model,
        place_deterministic=not args.stochastic_place,
        action_scale=args.action_scale)

    # ---- Eval environment (raw, unwrapped) ----
    # We normalize observations manually per-phase using grasp_vec_env /
    # place_vec_env stats, so the eval env must NOT wrap VecNormalize. This
    # fixes the VecNormalize mismatch bug where the place model was receiving
    # observations normalized with the grasp model's stats.
    raw_env = DummyVecEnv([lambda: make_env(include_target_pos=args.include_target_pos,
                                              target_pos=target_pos,
                                              target_pos_range=target_pos_range,
                                              domain_randomize=domain_randomize)])

    # Access the unwrapped PandaVLAEnv to toggle place_mode mid-episode.
    # The place policy was trained in place_mode (block hard-attached to
    # hand). Switching the env to place_mode during the place phase ensures
    # the block follows the hand as the policy expects, fixing the
    # train-eval environment mismatch that caused 0% place rate.
    _inner_env = raw_env.envs[0].env.unwrapped
    # Set configurable release threshold for the place phase
    _inner_env._release_dist_threshold = args.release_threshold
    # Set height gate: only allow release when block is near the table.
    # 0 means no height gate (use infinity). Otherwise set to the value.
    if args.release_height > 0:
        _inner_env._release_height_threshold = args.release_height
    else:
        _inner_env._release_height_threshold = float('inf')
    print(f"Release threshold: {args.release_threshold}m, height gate: "
          f"{args.release_height if args.release_height > 0 else 'off'}m")

    # For vision_mode: build a VisionObs wrapper around the SAME underlying
    # PandaVLAEnv so we can construct Dict obs {image, state} for the place
    # model on demand. The eval raw_env stays FlattenObs-wrapped (16-dim) so
    # the grasp phase and env stepping are unchanged; we only swap the obs
    # representation for the place phase by calling
    # place_vision_wrapper.observation(_inner_env._get_obs()).
    place_vision_wrapper = None
    if args.vision_mode and not place_fallback:
        from gym_env.wrappers import VisionObs
        place_vision_wrapper = VisionObs(_inner_env, image_size=84)
        print("Vision mode enabled: place model uses Dict obs {image, state}")
    elif args.vision_mode and place_fallback:
        print("Vision mode requested but place model fell back to grasp "
              "(state-only). Place phase will use 16-dim state obs.")

    np.random.seed(SEED)
    try:
        raw_env.seed(SEED)
    except Exception:
        pass

    # ---- Run episodes ----
    grab_flags, place_flags, pickplace_flags = [], [], []
    max_lifts, final_dists = [], []
    grasp_max_lifts, place_max_lifts = [], []
    phase_switch_steps = []

    for ep in range(args.n_episodes):
        # Reset place_mode to False for the grasp phase (block on table)
        _inner_env.place_mode = False
        _inner_env._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        ep_target_pos = _inner_env._target_pos.copy()
        ep_reward = 0.0
        max_lift = 0.0
        grasp_phase_max_lift = 0.0  # max lift during grasp phase only
        place_phase_max_lift = 0.0  # max lift during place phase only
        block_target_dist = float("inf")
        block_grabbed_at = None
        first_place_step = None
        prev_info = None  # info from previous step (None on first step)
        arm_frozen = False  # freeze arm after gripper opens

        for step in range(args.max_steps):
            # Detect phase from previous step's info. This call is idempotent
            # with the _detect_phase call inside policy.predict() below, so
            # calling it twice with the same info is safe (the second call
            # sees the phase already updated and makes no further change).
            phase = policy._detect_phase(prev_info)

            # When place_mode first activates, snap the block to the hand
            # immediately so the first place-policy observation reflects
            # the block at its trained position (hand_pos - 5cm). Without
            # this, the first place action is based on the grasp-phase obs
            # where the block is still at its old position.
            if phase == "place" and first_place_step is None:
                first_place_step = step
                _inner_env.place_mode = True
                _inner_env._place_gravcomp_active = True
                _inner_env.snap_block_to_hand()

                # CRITICAL: sync _arm_target and _gripper_target with the
                # current qpos. During the grasp phase, _arm_target drifts
                # from arm_qpos (due to action clipping, joint limits, etc).
                # Without this sync, the first place-model action (arm_delta)
                # is added to the stale _arm_target, causing the arm to jump
                # to a wrong position. This was the root cause of 0% place
                # rate in hierarchical eval despite place-only eval succeeding.
                _inner_env._arm_target = _inner_env.data.qpos[
                    _inner_env._arm_qpos_adrs
                ].copy()
                _inner_env._gripper_target = float(
                    _inner_env.data.qpos[_inner_env._finger_qpos_adrs].mean()
                )

                # CRITICAL: switch reward_type to "place_only" so that
                # _place_success is set when the block is placed (dist<5cm,
                # on table, gripper open). Without this, the episode never
                # terminates on place success, the place model keeps running,
                # and the block drifts away from the target — causing 0%
                # place rate despite the place model reaching <5cm.
                # Also reset the place reward state variables so the
                # one-time bonuses and progress tracking start fresh.
                _inner_env.reward_type = "place_only"
                _inner_env._place_approach_bonus_given = False
                _inner_env._place_proximity_15_given = False
                _inner_env._place_proximity_10_given = False
                _inner_env._place_success = False
                _inner_env._prev_block_target_dist = None
                _inner_env._prev_block_height = None
                # Use _gripper_target for the gripper_open check in eval
                # (immediate termination on success). Training uses
                # data.qpos (allows multi-step release bonus for stronger
                # gradient). See panda_vla_env.py _compute_reward_place
                # for details.
                _inner_env._use_gripper_target_check = True

                # Re-get raw_obs through the FlattenObs wrapper so the
                # block_position / hand_block_distance / block_target_distance
                # dims reflect the snapped block.
                flatten_wrapper = raw_env.envs[0]
                inner_obs = _inner_env._get_obs()
                new_flat = flatten_wrapper.observation(inner_obs)
                raw_obs = new_flat[np.newaxis, :].astype(np.float32)

            # Normalize obs with the VecNormalize stats matching the active
            # sub-policy. grasp_vec_env holds the grasp model's stats (16-dim);
            # place_vec_env holds the place model's stats (16/19-dim state, or
            # Dict {image, state} for vision_mode). The grasp model always
            # uses 16-dim obs (no target_pos).
            if phase == "place":
                if place_vision_wrapper is not None:
                    # Vision place model: build Dict obs {image, state} from
                    # the underlying env's raw obs (reflects snapped block /
                    # current physics state), batch to (1, ...) and normalize.
                    # Only "state" is normalized; "image" passes through.
                    vision_obs = place_vision_wrapper.observation(
                        _inner_env._get_obs()
                    )
                    vision_obs_batched = {
                        "image": vision_obs["image"][np.newaxis, ...],
                        "state": vision_obs["state"][np.newaxis, ...],
                    }
                    obs = place_vec_env.normalize_obs(vision_obs_batched)
                    # FIX: transpose image from HWC (1,84,84,3) to CHW
                    # (1,3,84,84). Training wraps the env with
                    # VecTransposeImage which converts HWC->CHW before the
                    # policy sees it; eval constructs obs manually and must
                    # match. Without this, the CNN receives HWC images but
                    # was trained on CHW, causing 0% place rate.
                    obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
                else:
                    obs = place_vec_env.normalize_obs(raw_obs)
            else:
                # Grasp model uses 16-dim obs; strip target_pos if present.
                # Also replace block_target_distance (dim 15) with the value
                # for the default target [0.5, 0.3, 0.2], since the grasp
                # model was trained with that fixed target. Without this,
                # changing the target position shifts block_target_distance,
                # which destabilizes the grasp model.
                raw_obs_for_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_for_grasp[0, 8:11]  # dims 8-10
                default_target = np.array([0.5, 0.3, 0.2])
                raw_obs_for_grasp[0, 15] = np.linalg.norm(block_pos - default_target)
                obs = grasp_vec_env.normalize_obs(raw_obs_for_grasp)

            # Predict (calls _detect_phase again — idempotent, see above).
            action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            # Freeze arm after gripper opens to prevent pushing the
            # falling/landed block. Only active during place phase.
            if (args.freeze_arm_on_release and not arm_frozen
                    and first_place_step is not None):
                if _inner_env._gripper_target > 0.02:
                    arm_frozen = True
            if arm_frozen:
                action[0][:] = 0.0  # zero arm deltas + gripper cmd

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]  # for next prediction
            ep_reward += float(reward[0])
            i = info[0]

            block_h = float(i.get("block_height", 0.0))
            block_target_dist = float(
                i.get("block_target_distance", block_target_dist)
            )

            lift = max(0.0, block_h - TABLE_Z)
            if lift > max_lift:
                max_lift = lift
            # Track lift per phase for diagnosis
            if first_place_step is not None:
                if lift > place_phase_max_lift:
                    place_phase_max_lift = lift
            else:
                if lift > grasp_phase_max_lift:
                    grasp_phase_max_lift = lift
            if lift > LIFT_THRESHOLD and block_grabbed_at is None:
                block_grabbed_at = step

            if done[0]:
                break

        grabbed = max_lift > LIFT_THRESHOLD
        entered_place = first_place_step is not None  # phase switched at least once
        placed = block_target_dist < PLACE_THRESHOLD
        pickplace = grabbed and placed

        grab_flags.append(grabbed)
        place_flags.append(placed)
        pickplace_flags.append(pickplace)
        max_lifts.append(max_lift)
        final_dists.append(block_target_dist)
        phase_switch_steps.append(first_place_step)
        grasp_max_lifts.append(grasp_phase_max_lift)
        place_max_lifts.append(place_phase_max_lift)

        print(f"Ep {ep:2d}: max_lift={max_lift*100:5.1f}cm  "
              f"final_dist={block_target_dist*100:5.1f}cm  "
              f"place_step={first_place_step if first_place_step is not None else '  -'}  "
              f"grab={'Y' if grabbed else 'N'} place={'Y' if placed else 'N'}"
              f"  [grasp={grasp_phase_max_lift*100:.1f}cm place={place_phase_max_lift*100:.1f}cm]"
              f"  target=[{ep_target_pos[0]:.3f},{ep_target_pos[1]:.3f},{ep_target_pos[2]:.3f}]")

    raw_env.close()

    # ---- Summary ----
    print()
    print("=" * 64)
    title = "Hierarchical Pick-and-Place Summary"
    if place_fallback:
        title += "  (place model = grasp fallback)"
    print(title)
    print("=" * 64)
    print(f"Episodes              : {args.n_episodes}")
    print(f"Grab  (lift>{LIFT_THRESHOLD*100:.0f}cm)   : "
          f"{sum(grab_flags)}/{args.n_episodes} "
          f"({100*sum(grab_flags)/args.n_episodes:.0f}%)")
    print(f"Place (dist<{PLACE_THRESHOLD*100:.0f}cm)  : "
          f"{sum(place_flags)}/{args.n_episodes} "
          f"({100*sum(place_flags)/args.n_episodes:.0f}%)")
    print(f"Pick+Place (both)     : "
          f"{sum(pickplace_flags)}/{args.n_episodes} "
          f"({100*sum(pickplace_flags)/args.n_episodes:.0f}%)")
    print(f"Mean max lift         : {np.mean(max_lifts)*100:.1f} cm")
    print(f"Best max lift         : {np.max(max_lifts)*100:.1f} cm")
    print(f"Mean final dist       : {np.mean(final_dists)*100:.1f} cm")
    print(f"Best final dist       : {np.min(final_dists)*100:.1f} cm")
    # Per-phase lift diagnosis: high grasp-phase lift with low place-phase
    # success suggests the grasp model lifts the arm too high, starting
    # place from an out-of-distribution arm configuration.
    print(f"Mean grasp-phase lift : {np.mean(grasp_max_lifts)*100:.1f} cm")
    print(f"Mean place-phase lift : {np.mean(place_max_lifts)*100:.1f} cm")

    valid_switches = [s for s in phase_switch_steps if s is not None]
    if valid_switches:
        print(f"Phase switches (grasp->place): {len(valid_switches)}/"
              f"{args.n_episodes} episodes")
        print(f"  Mean switch step : {np.mean(valid_switches):.1f}")
    else:
        print("Phase switches (grasp->place): 0 (never reached place phase)")


if __name__ == "__main__":
    main()
