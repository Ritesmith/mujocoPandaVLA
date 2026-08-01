#!/usr/bin/env python3
"""50-ep eval comparison: V59 baseline vs BC-trained (epoch-5) model.

Both models evaluated with deterministic policy (place_deterministic=True),
50 episodes each, same env config. Removes 15-ep eval variance that
confounded the BC training run.

The BC training (train_bc_expert.py) was cut short at epoch 5 by safety
rollback (33.3% < 40% threshold on 15-ep eval). The saved final_model.zip
contains epoch-5 BC weights + log_std=-2.0, but log_std is irrelevant for
deterministic eval (returns mean action only).

Usage:
    python eval_bc_vs_v59.py
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import time
from pathlib import Path

import numpy as np
import gymnasium
import gym_env  # noqa: F401
from gym_env.wrappers import VisionObs, FlattenObs
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
from hierarchical_policy import HierarchicalPickPlacePolicy

WORKSPACE = Path(__file__).parent.resolve()

V59_MODEL = str(WORKSPACE / "outputs/place_policy_v59/best_hier/best_model.zip")
V59_VECNORM = str(WORKSPACE / "outputs/place_policy_v59/best_hier/vec_normalize.pkl")
BC_MODEL = str(WORKSPACE / "outputs/bc_expert_v1/final_model.zip")
# BC model uses V59's vec_normalize (copied during training)
BC_VECNORM = V59_VECNORM

GRASP_MODEL = str(WORKSPACE / "outputs/dapg_800k_v5/best/best_model.zip")
GRASP_VECNORM = str(WORKSPACE / "outputs/dapg_800k_v5/vec_normalize.pkl")

TARGET_RANGE = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]
RELEASE_THRESHOLD = 0.05
N_EPISODES = 50
SEED = 42


def make_raw_env():
    factory = lambda: FlattenObs(gymnasium.make(
        "PandaVLA-v0", reward_type="dense", gravity_comp=True,
        target_pos_range=TARGET_RANGE, domain_randomize=False))
    return DummyVecEnv([factory])


def load_grasp_model():
    factory = lambda: FlattenObs(gymnasium.make(
        "PandaVLA-v0", reward_type="dense", gravity_comp=True,
        target_pos_range=TARGET_RANGE, domain_randomize=False))
    vec = DummyVecEnv([factory])
    vec = VecNormalize.load(GRASP_VECNORM, vec)
    vec.norm_reward = False
    vec.training = False
    return PPO.load(GRASP_MODEL, env=vec, device="auto"), vec


def load_place_model(model_path, vecnorm_path):
    """Load place model with VisionObs-compatible vec_env."""
    factory = lambda: VisionObs(
        gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True,
                       target_pos_range=TARGET_RANGE, domain_randomize=False),
        image_size=84)
    vec = DummyVecEnv([factory])
    vec = VecNormalize.load(vecnorm_path, vec)
    vec.norm_reward = False
    vec.training = False
    vec = VecTransposeImage(vec)
    model = PPO.load(model_path, env=vec, device="cuda")
    return model, vec


def eval_model(place_model, place_vec, grasp_model, grasp_vec, n_episodes=50,
               seed=42, label=""):
    """Run hierarchical eval for n_episodes. Returns (place_rate, mean_dist, n_placed, n_grabbed)."""
    policy = HierarchicalPickPlacePolicy(grasp_model, place_model)

    raw_env = make_raw_env()
    inner = raw_env.envs[0].env.unwrapped
    inner._release_dist_threshold = RELEASE_THRESHOLD
    inner._release_height_threshold = float('inf')
    place_vision = VisionObs(inner, image_size=84)

    np.random.seed(seed)
    try:
        raw_env.seed(seed)
    except Exception:
        pass

    n_placed = 0
    n_grabbed = 0
    final_dists = []
    ep_lens = []
    t0 = time.time()

    for ep in range(n_episodes):
        inner.place_mode = False
        inner._place_gravcomp_active = False
        raw_obs = raw_env.reset()
        policy.reset()
        first_place_step = None
        prev_info = None
        max_lift = 0.0
        block_target_dist = float("inf")
        ep_steps = 0

        for step in range(500):
            ep_steps += 1
            phase = policy._detect_phase(prev_info)

            if phase == "place" and first_place_step is None:
                first_place_step = step
                inner.place_mode = True
                inner._place_gravcomp_active = True
                inner.snap_block_to_hand()
                inner._arm_target = inner.data.qpos[inner._arm_qpos_adrs].copy()
                inner._gripper_target = float(inner.data.qpos[inner._finger_qpos_adrs].mean())
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
                obs_batched = {
                    "image": vision_obs["image"][np.newaxis, ...],
                    "state": vision_obs["state"][np.newaxis, ...],
                }
                obs = place_vec.normalize_obs(obs_batched)
                obs["image"] = np.transpose(obs["image"], (0, 3, 1, 2))
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)
            else:
                raw_obs_grasp = raw_obs[:, :16].copy()
                block_pos = raw_obs_grasp[0, 8:11]
                raw_obs_grasp[0, 15] = np.linalg.norm(block_pos - np.array([0.5, 0.3, 0.2]))
                obs = grasp_vec.normalize_obs(raw_obs_grasp)
                action, _ = policy.predict(obs, info=prev_info, deterministic=True)

            raw_obs, reward, done, info = raw_env.step(action)
            prev_info = info[0]
            block_target_dist = float(info[0].get("block_target_distance", block_target_dist))
            lift = max(0.0, float(info[0].get("block_height", 0.0)) - 0.22)
            if lift > max_lift:
                max_lift = lift
            if done[0]:
                break

        if first_place_step is not None and max_lift > 0.03:
            n_grabbed += 1
            if block_target_dist < RELEASE_THRESHOLD:
                n_placed += 1

        final_dists.append(block_target_dist)
        ep_lens.append(ep_steps)

        status = "PLACE" if (first_place_step is not None and max_lift > 0.03
                             and block_target_dist < RELEASE_THRESHOLD) else "FAIL"
        if (ep + 1) % 10 == 0 or ep < 5:
            elapsed = time.time() - t0
            print(f"  [{label}] Ep {ep+1:3d}/{n_episodes}: {status}  "
                  f"dist={block_target_dist*100:5.1f}cm  steps={ep_steps:3d}  "
                  f"| running: {n_placed}/{n_grabbed} placed [{elapsed:.0f}s]")

    place_rate = n_placed / max(1, n_grabbed)
    mean_dist = float(np.mean(final_dists))
    elapsed = time.time() - t0
    print(f"\n  [{label}] FINAL: {n_placed}/{n_grabbed} placed "
          f"({100*place_rate:.1f}%), mean_dist={mean_dist*100:.1f}cm, "
          f"mean_ep_len={np.mean(ep_lens):.0f}, elapsed={elapsed:.0f}s")
    return place_rate, mean_dist, n_placed, n_grabbed


def main():
    print("=" * 70)
    print(f"50-ep Eval Comparison: V59 baseline vs BC epoch-5")
    print(f"  V59 model: {V59_MODEL}")
    print(f"  BC model:  {BC_MODEL}")
    print(f"  Episodes:  {N_EPISODES} each, deterministic eval, seed={SEED}")
    print("=" * 70)

    # Load grasp model (shared)
    print("\nLoading grasp model...")
    grasp_model, grasp_vec = load_grasp_model()

    # Eval V59 baseline
    print("\n" + "=" * 70)
    print("Evaluating V59 baseline (50 episodes)...")
    print("=" * 70)
    v59_place, v59_dist, v59_placed, v59_grabbed = eval_model(
        *load_place_model(V59_MODEL, V59_VECNORM),
        grasp_model, grasp_vec,
        n_episodes=N_EPISODES, seed=SEED, label="V59")

    # Eval BC epoch-5
    print("\n" + "=" * 70)
    print("Evaluating BC epoch-5 model (50 episodes)...")
    print("=" * 70)
    bc_place, bc_dist, bc_placed, bc_grabbed = eval_model(
        *load_place_model(BC_MODEL, BC_VECNORM),
        grasp_model, grasp_vec,
        n_episodes=N_EPISODES, seed=SEED, label="BC")

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  V59 baseline:  {v59_placed}/{v59_grabbed} placed ({100*v59_place:.1f}%), "
          f"mean_dist={v59_dist*100:.1f}cm")
    print(f"  BC epoch-5:    {bc_placed}/{bc_grabbed} placed ({100*bc_place:.1f}%), "
          f"mean_dist={bc_dist*100:.1f}cm")
    diff = bc_place - v59_place
    print(f"  Difference:    {100*diff:+.1f}% ({'BC BETTER' if diff > 0.05 else 'V59 BETTER' if diff < -0.05 else 'WITHIN NOISE'})")

    # Decision
    print("\n  VERDICT:")
    if bc_place > v59_place + 0.10:
        print("  → BC IMPROVED V59 by >10%. Breakthrough! Investigate further.")
    elif bc_place < v59_place - 0.10:
        print("  → BC DEGRADED V59 by >10%. 8th method family failure. Close V59.")
    elif abs(bc_place - v59_place) < 0.05:
        print("  → BC within ±5% of V59 (NO-OP / within noise). Close V59 — "
              "bc_loss decrease did not translate to place_rate improvement.")
    else:
        print(f"  → BC changed place_rate by {100*diff:+.1f}% (ambiguous). "
              "Likely within 50-ep eval variance (Wilson CI ±14%).")

    # Save results
    results = {
        "v59": {"place_rate": v59_place, "mean_dist": v59_dist,
                "n_placed": v59_placed, "n_grabbed": v59_grabbed},
        "bc_epoch5": {"place_rate": bc_place, "mean_dist": bc_dist,
                      "n_placed": bc_placed, "n_grabbed": bc_grabbed},
        "difference": diff,
        "n_episodes": N_EPISODES,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = WORKSPACE / "outputs/bc_expert_v1/eval_50ep_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
