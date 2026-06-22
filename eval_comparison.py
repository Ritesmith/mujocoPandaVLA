#!/usr/bin/env python3
"""Compare DAPG vs GRPO trained models on Panda pick-place task.

Loads a DAPG (SB3 PPO + VecNormalize) model and a GRPO (PyTorch state_dict)
model, evaluates both on the PandaVLA-v0 environment (gravity_comp=True,
dense reward), and reports success rate / mean reward / mean steps alongside
a random-action baseline.

Usage:
    python eval_comparison.py --n_episodes 20 --seed 42
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import numpy as np
import torch
import gymnasium
import gym_env  # noqa: F401  registers PandaVLA-v0
from gym_env.wrappers import FlattenObs

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DAPG_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg/dapg_final.zip"
DAPG_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg/vec_normalize.pkl"
DAPG_RAND_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_rand/dapg_final.zip"
DAPG_RAND_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_rand/vec_normalize.pkl"
DAPG_500K_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k/best/best_model.zip"
DAPG_500K_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k/vec_normalize.pkl"
DAPG_500K_V2_MODEL_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v2/best/best_model.zip"
DAPG_500K_V2_VECNORM_PATH = "/home/w/vla_workspace/outputs/dapg_500k_v2/vec_normalize.pkl"
GRPO_MODEL_PATH = "/home/w/vla_workspace/outputs/grpo/grpo_best.pt"

SUCCESS_THRESHOLD = 0.05      # block-target distance < 0.05  -> success
MAX_STEPS = 500               # matches env max_episode_steps
OBS_DIM = 16
ACT_DIM = 8
HIDDEN_DIM = 256


def _make_raw_env():
    """Create a single PandaVLA-v0 env with FlattenObs (gravity_comp=True)."""
    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    return FlattenObs(env)


# ---------------------------------------------------------------------------
# DAPG evaluation (SB3 PPO + VecNormalize)
# ---------------------------------------------------------------------------
def evaluate_dapg(n_episodes=20, seed=42):
    """Load DAPG (SB3 PPO) model and evaluate.

    - Load SB3 PPO model from outputs/dapg/dapg_final.zip
    - Load VecNormalize from outputs/dapg/vec_normalize.pkl
    - Create eval env with VecNormalize (stats frozen, reward un-normalized)
    - Run n_episodes, collect: success_rate, mean_reward, mean_steps
    """
    if not os.path.exists(DAPG_MODEL_PATH):
        print(f"[DAPG] Model not found: {DAPG_MODEL_PATH}  (skipping)")
        return None

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # --- Load model --------------------------------------------------------
    # DAPG was trained with a DAPGPPO(PPO) subclass; for inference a plain PPO
    # is sufficient (the BC logic only runs during train()).  Try PPO.load()
    # first, then fall back to the DAPGPPO class if SB3 insists on the saved
    # model_class.
    model = None
    try:
        model = PPO.load(DAPG_MODEL_PATH, device="auto")
    except Exception as e:  # noqa: BLE001
        print(f"[DAPG] PPO.load() failed ({e}); trying DAPGPPO.load() ...")
        try:
            from train_dapg import DAPGPPO
            model = DAPGPPO.load(DAPG_MODEL_PATH, device="auto")
        except Exception as e2:  # noqa: BLE001
            print(f"[DAPG] DAPGPPO.load() also failed: {e2}  (skipping)")
            return None

    # --- Build eval env with VecNormalize ---------------------------------
    def _env_fn():
        return _make_raw_env()

    eval_env = DummyVecEnv([_env_fn])
    if os.path.exists(DAPG_VECNORM_PATH):
        eval_env = VecNormalize.load(DAPG_VECNORM_PATH, eval_env)
        eval_env.norm_reward = False   # report raw rewards
        eval_env.training = False       # freeze running stats
    else:
        print("[DAPG] vec_normalize.pkl not found; using raw observations "
              "(results may be poor).")

    # --- Evaluation loop ---------------------------------------------------
    np.random.seed(seed)
    try:
        eval_env.seed(seed)
    except Exception:
        pass  # gymnasium envs may not implement seed() via DummyVecEnv

    successes, rewards, steps_list = [], [], []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        block_target_dist = float("inf")
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            block_target_dist = float(
                info[0].get("block_target_distance", block_target_dist)
            )
            if done[0]:
                break
        successes.append(block_target_dist < SUCCESS_THRESHOLD)
        rewards.append(ep_reward)
        steps_list.append(step + 1)

    eval_env.close()
    return _summarize("DAPG", successes, rewards, steps_list)


# ---------------------------------------------------------------------------
# DAPG-Rand evaluation (domain-randomized DAPG, SB3 PPO + VecNormalize)
# ---------------------------------------------------------------------------
def evaluate_dapg_rand(n_episodes=20, seed=42):
    """Load domain-randomized DAPG model and evaluate on fixed env."""
    if not os.path.exists(DAPG_RAND_MODEL_PATH):
        print(f"[DAPG-Rand] Model not found: {DAPG_RAND_MODEL_PATH}  (skipping)")
        return None

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = None
    try:
        model = PPO.load(DAPG_RAND_MODEL_PATH, device="auto")
    except Exception as e:  # noqa: BLE001
        print(f"[DAPG-Rand] PPO.load() failed ({e}); trying DAPGPPO.load() ...")
        try:
            from train_dapg import DAPGPPO
            model = DAPGPPO.load(DAPG_RAND_MODEL_PATH, device="auto")
        except Exception as e2:  # noqa: BLE001
            print(f"[DAPG-Rand] DAPGPPO.load() also failed: {e2}  (skipping)")
            return None

    def _env_fn():
        return _make_raw_env()

    eval_env = DummyVecEnv([_env_fn])
    if os.path.exists(DAPG_RAND_VECNORM_PATH):
        eval_env = VecNormalize.load(DAPG_RAND_VECNORM_PATH, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
    else:
        print("[DAPG-Rand] vec_normalize.pkl not found; using raw observations.")

    np.random.seed(seed)
    try:
        eval_env.seed(seed)
    except Exception:
        pass

    successes, rewards, steps_list = [], [], []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        block_target_dist = float("inf")
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            block_target_dist = float(
                info[0].get("block_target_distance", block_target_dist)
            )
            if done[0]:
                break
        successes.append(block_target_dist < SUCCESS_THRESHOLD)
        rewards.append(ep_reward)
        steps_list.append(step + 1)

    eval_env.close()
    return _summarize("DAPG-Rand", successes, rewards, steps_list)


# ---------------------------------------------------------------------------
# DAPG-500K evaluation (500K-step DAPG, best checkpoint)
# ---------------------------------------------------------------------------
def evaluate_dapg_500k(n_episodes=20, seed=42):
    """Load DAPG 500K best model and evaluate."""
    if not os.path.exists(DAPG_500K_MODEL_PATH):
        print(f"[DAPG-500K] Model not found: {DAPG_500K_MODEL_PATH}  (skipping)")
        return None

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = None
    try:
        model = PPO.load(DAPG_500K_MODEL_PATH, device="auto")
    except Exception as e:  # noqa: BLE001
        print(f"[DAPG-500K] PPO.load() failed ({e}); trying DAPGPPO.load() ...")
        try:
            from train_dapg import DAPGPPO
            model = DAPGPPO.load(DAPG_500K_MODEL_PATH, device="auto")
        except Exception as e2:  # noqa: BLE001
            print(f"[DAPG-500K] DAPGPPO.load() also failed: {e2}  (skipping)")
            return None

    def _env_fn():
        return _make_raw_env()

    eval_env = DummyVecEnv([_env_fn])
    if os.path.exists(DAPG_500K_VECNORM_PATH):
        eval_env = VecNormalize.load(DAPG_500K_VECNORM_PATH, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
    else:
        print("[DAPG-500K] vec_normalize.pkl not found; using raw observations "
              "(results may be poor).")

    np.random.seed(seed)
    try:
        eval_env.seed(seed)
    except Exception:
        pass

    successes, rewards, steps_list = [], [], []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        block_target_dist = float("inf")
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            block_target_dist = float(
                info[0].get("block_target_distance", block_target_dist)
            )
            if done[0]:
                break
        successes.append(block_target_dist < SUCCESS_THRESHOLD)
        rewards.append(ep_reward)
        steps_list.append(step + 1)

    eval_env.close()
    return _summarize("DAPG-500K", successes, rewards, steps_list)


# ---------------------------------------------------------------------------
# DAPG-500K-v2 evaluation (improved reward function with placing gradient)
# ---------------------------------------------------------------------------
def evaluate_dapg_500k_v2(n_episodes=20, seed=42):
    """Load DAPG 500K v2 (improved reward) best model and evaluate."""
    if not os.path.exists(DAPG_500K_V2_MODEL_PATH):
        print(f"[DAPG-500K-v2] Model not found: {DAPG_500K_V2_MODEL_PATH}  (skipping)")
        return None

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = None
    try:
        model = PPO.load(DAPG_500K_V2_MODEL_PATH, device="auto")
    except Exception as e:  # noqa: BLE001
        print(f"[DAPG-500K-v2] PPO.load() failed ({e}); trying DAPGPPO.load() ...")
        try:
            from train_dapg import DAPGPPO
            model = DAPGPPO.load(DAPG_500K_V2_MODEL_PATH, device="auto")
        except Exception as e2:  # noqa: BLE001
            print(f"[DAPG-500K-v2] DAPGPPO.load() also failed: {e2}  (skipping)")
            return None

    def _env_fn():
        return _make_raw_env()

    eval_env = DummyVecEnv([_env_fn])
    if os.path.exists(DAPG_500K_V2_VECNORM_PATH):
        eval_env = VecNormalize.load(DAPG_500K_V2_VECNORM_PATH, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
    else:
        print("[DAPG-500K-v2] vec_normalize.pkl not found; using raw observations "
              "(results may be poor).")

    np.random.seed(seed)
    try:
        eval_env.seed(seed)
    except Exception:
        pass

    successes, rewards, steps_list = [], [], []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        ep_reward = 0.0
        block_target_dist = float("inf")
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += float(reward[0])
            block_target_dist = float(
                info[0].get("block_target_distance", block_target_dist)
            )
            if done[0]:
                break
        successes.append(block_target_dist < SUCCESS_THRESHOLD)
        rewards.append(ep_reward)
        steps_list.append(step + 1)

    eval_env.close()
    return _summarize("DAPG-500K-v2", successes, rewards, steps_list)


# ---------------------------------------------------------------------------
# GRPO evaluation (PyTorch state_dict)
# ---------------------------------------------------------------------------
def evaluate_grpo(n_episodes=20, seed=42):
    """Load GRPO model and evaluate.

    - Load GRPOPolicy state_dict from outputs/grpo/grpo_final.pt
    - Create env (PandaVLA-v0, gravity_comp=True) with FlattenObs
    - Run n_episodes with deterministic policy
    - Collect: success_rate, mean_reward, mean_steps
    """
    if not os.path.exists(GRPO_MODEL_PATH):
        print(f"[GRPO] Model not found: {GRPO_MODEL_PATH}  (skipping)")
        return None

    from train_grpo import GRPOPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = GRPOPolicy(OBS_DIM, ACT_DIM, hidden_dim=HIDDEN_DIM)
    policy.load_state_dict(
        torch.load(GRPO_MODEL_PATH, map_location=device)
    )
    policy.to(device)
    policy.eval()

    env = _make_raw_env()
    np.random.seed(seed)
    torch.manual_seed(seed)

    successes, rewards, steps_list = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        block_target_dist = float("inf")
        for step in range(MAX_STEPS):
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                action = policy.get_action(obs_tensor, deterministic=True)
            action_np = action[0].cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_reward += float(reward)
            block_target_dist = float(
                info.get("block_target_distance", block_target_dist)
            )
            if terminated or truncated:
                break
        successes.append(block_target_dist < SUCCESS_THRESHOLD)
        rewards.append(ep_reward)
        steps_list.append(step + 1)

    env.close()
    return _summarize("GRPO", successes, rewards, steps_list)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------
def evaluate_random(n_episodes=20, seed=42):
    """Random-action baseline for comparison."""
    env = _make_raw_env()
    np.random.seed(seed)

    successes, rewards, steps_list = [], [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        block_target_dist = float("inf")
        for step in range(MAX_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            block_target_dist = float(
                info.get("block_target_distance", block_target_dist)
            )
            if terminated or truncated:
                break
        successes.append(block_target_dist < SUCCESS_THRESHOLD)
        rewards.append(ep_reward)
        steps_list.append(step + 1)

    env.close()
    return _summarize("Random", successes, rewards, steps_list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarize(name, successes, rewards, steps_list):
    """Aggregate episode metrics into a result dict."""
    return {
        "name": name,
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "n_episodes": len(successes),
    }


def _print_table(results):
    """Print a formatted comparison table."""
    print()
    print("=" * 64)
    print(f"Evaluation Results  (success = block-target dist < {SUCCESS_THRESHOLD})")
    print("=" * 64)
    header = f"| {'Method':<8} | {'Success Rate':>12} | {'Mean Reward':>12} | {'Mean Steps':>10} |"
    sep = f"|{'-'*10}|{'-'*14}|{'-'*14}|{'-'*12}|"
    print(header)
    print(sep)
    for r in results:
        print(
            f"| {r['name']:<8} | {r['success_rate']*100:>11.1f}% | "
            f"{r['mean_reward']:>12.2f} | {r['mean_steps']:>10.1f} |"
        )
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare DAPG vs GRPO models")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of eval episodes per method")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Running comparison: n_episodes={args.n_episodes}, seed={args.seed}")

    results = []
    for fn in (evaluate_random, evaluate_dapg, evaluate_dapg_rand,
               evaluate_dapg_500k, evaluate_dapg_500k_v2, evaluate_grpo):
        res = fn(n_episodes=args.n_episodes, seed=args.seed)
        if res is not None:
            results.append(res)

    if not results:
        print("No models could be evaluated.")
        return

    _print_table(results)


if __name__ == "__main__":
    main()
