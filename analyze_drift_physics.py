#!/usr/bin/env python3
"""P1: Physics root cause investigation for drift episodes.

Runs v4 model (chunk_size=4) with detailed physics logging to determine
whether drift is caused by:
  (a) Physics/contact dynamics (block slipping at close range)
  (b) Policy decision errors (bad chunks near target)

Key diagnostics:
  1. Block velocity at best_dist moment — non-zero = momentum, zero = static instability
  2. Target position distribution — drift concentrated at specific geometry?
  3. Post-best_dist trajectory — sudden jump or gradual drift?

Usage:
    python analyze_drift_physics.py --n_episodes 100 \
        --checkpoint outputs/iql_v4_chunking/final_model.pt --chunk_size 4
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import FlattenObs, VisionObs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hierarchical_policy import HierarchicalPickPlacePolicy

import torch
from iql_dataset import OfflineDataset
from iql_agent import IQLAgent

GRASP_MODEL = "/home/w/vla_workspace/outputs/dapg_800k_v5/best/best_model.zip"
GRASP_VECNORM = "/home/w/vla_workspace/outputs/dapg_800k_v5/vec_normalize.pkl"
PLACE_MODEL = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/best_model.zip"
PLACE_VECNORM = "/home/w/vla_workspace/outputs/place_policy_v59/best_hier/vec_normalize.pkl"
EXPERT_DATA = "/home/w/vla_workspace/data/D_expert.npz"

LIFT_THRESHOLD = 0.03
TABLE_Z = 0.22
MAX_STEPS = 500
SEED = 42
TARGET_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]
NEAR_MISS_DIST = 0.15  # 15cm threshold


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


def load_iql_policy(checkpoint_path, state_mean, state_std, chunk_size=1, device="cpu"):
    agent = IQLAgent(state_dim=12, action_dim=8, hidden_dim=256,
                     tau=0.7, beta=3.0, gamma=0.99, polyak=0.005,
                     chunk_size=chunk_size, device=device)
    agent.load(checkpoint_path)
    agent.policy.eval()
    state_mean_t = torch.FloatTensor(state_mean).to(device)
    state_std_t = torch.FloatTensor(state_std).to(device)

    @torch.no_grad()
    def get_action(state_np, deterministic=True):
        state = torch.FloatTensor(state_np).unsqueeze(0).to(device)
        state_norm = (state - state_mean_t) / state_std_t
        mean, log_std = agent.policy(state_norm)
        action = torch.tanh(mean) if deterministic else torch.tanh(
            torch.distributions.Normal(mean, log_std.exp()).sample())
        return action.squeeze(0).cpu().numpy()

    return get_action


def main():
    parser = argparse.ArgumentParser(description="P1: Drift physics analysis")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--release_threshold", type=float, default=0.05)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"outputs/iql_v4_chunking/drift_physics_n{args.n_episodes}.json"

    print("=" * 70)
    print(f"P1: Drift Physics Analysis (N={args.n_episodes}, chunk_size={args.chunk_size})")
    print("=" * 70)

    # Load normalization stats
    dataset = OfflineDataset(data_path=EXPERT_DATA, normalize_states=True,
                             normalize_actions=False)
    state_mean = dataset.state_mean
    state_std = dataset.state_std

    # Load IQL policy
    iql_get_action = load_iql_policy(args.checkpoint, state_mean, state_std,
                                     chunk_size=args.chunk_size)

    # Load V59 models for phase detection
    grasp_model, grasp_vec_env = load_sb3_model(
        GRASP_MODEL, GRASP_VECNORM, vision_mode=False, target_pos_range=TARGET_RANGE)
    place_model, place_vec_env = load_sb3_model(
        PLACE_MODEL, PLACE_VECNORM, vision_mode=True, target_pos_range=TARGET_RANGE)
    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    # Create env
    raw_env = DummyVecEnv([lambda: make_env(vision_mode=False, target_pos_range=TARGET_RANGE)])
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = args.release_threshold
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    np.random.seed(args.seed)
    try:
        raw_env.seed(args.seed)
    except Exception:
        pass

    # Episode data collection
    all_episodes = []
    n_placed = 0
    n_drift = 0
    n_near_miss = 0

    t0 = time.time()

    for ep in range(args.n_episodes):
        inner.place_mode = False
        inner._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        prev_info = None
        block_target_dist = float("inf")
        ep_place_steps = 0
        action_chunk_buffer = []
        best_dist = float("inf")
        best_step = 0

        # Physics log: list of dicts, one per place-phase step
        physics_log = []

        for step in range(MAX_STEPS):
            phase = policy._detect_phase(prev_info)

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
                vision_obs = place_vision.observation(inner._get_obs())
                state_12 = vision_obs["state"]

                # --- Physics logging BEFORE action ---
                block_pos = inner.data.xpos[inner._red_block_id].copy()
                block_vel = inner.data.qvel[inner._red_block_dof_adr:inner._red_block_dof_adr + 3].copy()
                target_pos = inner._target_pos.copy()
                dist = float(np.linalg.norm(block_pos - target_pos))

                # Contact info
                n_contacts = inner.data.ncon
                contact_geoms = []
                for ci in range(n_contacts):
                    c = inner.data.contact[ci]
                    contact_geoms.append([int(c.geom1), int(c.geom2)])

                physics_log.append({
                    "place_step": ep_place_steps,
                    "block_pos": block_pos.tolist(),
                    "block_vel": block_vel.tolist(),
                    "block_speed": float(np.linalg.norm(block_vel)),
                    "target_pos": target_pos.tolist(),
                    "dist": dist,
                    "n_contacts": n_contacts,
                    "contact_geoms": contact_geoms,
                })

                if dist < best_dist:
                    best_dist = dist
                    best_step = ep_place_steps

                # Chunk-based action
                if args.chunk_size > 1:
                    if not action_chunk_buffer:
                        chunk_flat = iql_get_action(state_12, deterministic=True)
                        action_chunk_buffer = chunk_flat.reshape(args.chunk_size, -1).tolist()
                    action = np.array(action_chunk_buffer.pop(0), dtype=np.float32)
                else:
                    action = iql_get_action(state_12, deterministic=True)

                action = action[np.newaxis, :]
                ep_place_steps += 1

                # Early abort
                steps_since_best = ep_place_steps - best_step
                if steps_since_best > 30 or dist > 0.50:
                    break
            else:
                raw_obs_grasp = raw_obs[:, :16].copy()
                block_pos_raw = raw_obs_grasp[0, 8:11]
                raw_obs_grasp[0, 15] = np.linalg.norm(
                    block_pos_raw - np.array([0.5, 0.3, 0.2]))
                obs = grasp_vec_env.normalize_obs(raw_obs_grasp)
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(info[0].get("block_target_distance", block_target_dist))

            if done or (phase == "place" and info[0].get("place_success", False)):
                break

        # Classify episode
        final_dist = block_target_dist
        is_placed = info[0].get("place_success", False) if prev_info else False
        is_drift = (not is_placed) and final_dist >= NEAR_MISS_DIST
        is_near_miss = (not is_placed) and (final_dist < NEAR_MISS_DIST) and (final_dist > 0.01)

        if is_placed:
            n_placed += 1
        elif is_drift:
            n_drift += 1
        elif is_near_miss:
            n_near_miss += 1

        # Extract best_dist physics
        best_physics = None
        if physics_log:
            best_physics = physics_log[min(best_step, len(physics_log) - 1)]

        # Extract post-best trajectory (dist for 20 steps after best)
        post_best_traj = []
        if physics_log:
            for i in range(best_step, min(best_step + 20, len(physics_log))):
                post_best_traj.append(round(physics_log[i]["dist"] * 100, 1))

        all_episodes.append({
            "ep": ep,
            "placed": is_placed,
            "drift": is_drift,
            "near_miss": is_near_miss,
            "final_dist_cm": round(final_dist * 100, 1),
            "best_dist_cm": round(best_dist * 100, 1),
            "best_step": best_step,
            "place_steps": ep_place_steps,
            "target_pos": inner._target_pos.tolist() if hasattr(inner, '_target_pos') else None,
            "best_physics": best_physics,
            "post_best_traj_cm": post_best_traj,
        })

        if (ep + 1) % 20 == 0:
            print(f"  [{ep+1}/{args.n_episodes}] {time.time()-t0:.0f}s, "
                  f"placed={n_placed}, drift={n_drift}, near_miss={n_near_miss}")

    elapsed = time.time() - t0
    print(f"\nAnalysis complete in {elapsed:.0f}s")
    print(f"  Placed:    {n_placed}/{args.n_episodes} ({n_placed/args.n_episodes:.1%})")
    print(f"  Drift:     {n_drift}/{args.n_episodes} ({n_drift/args.n_episodes:.1%})")
    print(f"  Near_miss: {n_near_miss}/{args.n_episodes} ({n_near_miss/args.n_episodes:.1%})")

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("DRIFT PHYSICS ANALYSIS")
    print("=" * 70)

    drift_eps = [e for e in all_episodes if e["drift"]]
    placed_eps = [e for e in all_episodes if e["placed"]]

    # 1. Block velocity at best_dist moment
    print("\n1. Block velocity at best_dist moment (drift episodes):")
    if drift_eps:
        speeds = []
        for e in drift_eps:
            bp = e.get("best_physics")
            if bp:
                speed = bp["block_speed"]
                speeds.append(speed)
                vel = bp["block_vel"]
                print(f"   Ep {e['ep']:3d}: best={e['best_dist_cm']:.1f}cm, "
                      f"speed={speed*100:.2f}cm/s, vel=[{vel[0]*100:.1f},{vel[1]*100:.1f},{vel[2]*100:.1f}]cm/s, "
                      f"target=[{bp['target_pos'][0]:.2f},{bp['target_pos'][1]:.2f}]")
        if speeds:
            speeds = np.array(speeds)
            print(f"\n   Mean speed at best_dist: {speeds.mean()*100:.2f} cm/s")
            print(f"   Max speed at best_dist:  {speeds.max()*100:.2f} cm/s")
            print(f"   Non-zero speed (>1cm/s): {(speeds > 0.01).sum()}/{len(speeds)}")
            if speeds.mean() > 0.01:
                print("   → Block has NON-ZERO velocity at best_dist → MOMENTUM-driven drift")
            else:
                print("   → Block has ~ZERO velocity at best_dist → STATIC instability drift")
    else:
        print("   No drift episodes found.")

    # 2. Target position distribution: drift vs placed
    print("\n2. Target position distribution:")
    if drift_eps:
        drift_targets = np.array([e["target_pos"] for e in drift_eps if e["target_pos"]])
        print(f"   Drift episodes ({len(drift_targets)}):")
        print(f"     target_x: mean={drift_targets[:,0].mean():.3f}, std={drift_targets[:,0].std():.3f}")
        print(f"     target_y: mean={drift_targets[:,1].mean():.3f}, std={drift_targets[:,1].std():.3f}")
    if placed_eps:
        placed_targets = np.array([e["target_pos"] for e in placed_eps if e["target_pos"]])
        print(f"   Placed episodes ({len(placed_targets)}):")
        print(f"     target_x: mean={placed_targets[:,0].mean():.3f}, std={placed_targets[:,0].std():.3f}")
        print(f"     target_y: mean={placed_targets[:,1].mean():.3f}, std={placed_targets[:,1].std():.3f}")
    if drift_eps and placed_eps:
        # Check if drift concentrates at edge targets
        drift_dist_from_center = np.sqrt(
            (drift_targets[:,0] - 0.5)**2 + (drift_targets[:,1] - 0.3)**2)
        placed_dist_from_center = np.sqrt(
            (placed_targets[:,0] - 0.5)**2 + (placed_targets[:,1] - 0.3)**2)
        print(f"\n   Distance from target center (0.5, 0.3):")
        print(f"     Drift:  mean={drift_dist_from_center.mean():.3f}m")
        print(f"     Placed: mean={placed_dist_from_center.mean():.3f}m")
        if drift_dist_from_center.mean() > placed_dist_from_center.mean() + 0.02:
            print("   → Drift CONCENTRATED at edge targets → GEOMETRY-driven drift")
        else:
            print("   → Drift UNIFORMLY distributed → NOT geometry-driven")

    # 3. Post-best trajectory analysis
    print("\n3. Post-best_dist trajectory (drift episodes):")
    for e in drift_eps[:10]:
        traj = e.get("post_best_traj_cm", [])
        print(f"   Ep {e['ep']:3d}: best={e['best_dist_cm']:.1f}cm → "
              f"traj_after={traj}")

    # 4. Sudden jump detection
    print("\n4. Sudden jump detection (dist increase > 5cm in one sample):")
    jump_count = 0
    for e in drift_eps:
        traj = e.get("post_best_traj_cm", [])
        for i in range(1, len(traj)):
            if traj[i] - traj[i-1] > 5.0:  # >5cm jump in one sample (10 steps)
                jump_count += 1
                print(f"   Ep {e['ep']:3d}: jump {traj[i-1]:.1f}→{traj[i]:.1f}cm "
                      f"(+{traj[i]-traj[i-1]:.1f}cm in 10 steps)")
                break
    print(f"   Total sudden jumps: {jump_count}/{len(drift_eps)} drift episodes")

    # Save results
    results = {
        "n_episodes": args.n_episodes,
        "checkpoint": args.checkpoint,
        "chunk_size": args.chunk_size,
        "summary": {
            "placed": n_placed,
            "drift": n_drift,
            "near_miss": n_near_miss,
            "place_rate": n_placed / args.n_episodes,
        },
        "episodes": all_episodes,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
