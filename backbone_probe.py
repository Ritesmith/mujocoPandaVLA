#!/usr/bin/env python3
"""Backbone Probe: Test V59's ResNet-18 representation quality.

Question: Is V59's 56% place rate limited by the representation (ResNet-18
features) or by the policy head (mlp_extractor + action_net)?

Method: Freeze V59's features_extractor, extract 524-dim features for all
D_csil transitions, train simple linear/MLP probes on top. If probes achieve
high accuracy, the representation is sufficient and the bottleneck is in the
policy head. If probes fail, the representation itself is the bottleneck and
architectural changes (diffusion, transformer) are justified.

Three probe tasks:
  1. Success prediction (binary): Can features predict episode success?
  2. Action prediction (regression): Can features predict V59's action?
  3. Distance prediction (regression): Can features predict final distance?

Also compares V59-fine-tuned ResNet-18 vs ImageNet-only ResNet-18 to measure
how much V59's fine-tuning improved the representation.

Usage:
    python backbone_probe.py
    python backbone_probe.py --batch_size 256 --device cuda
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, roc_auc_score, mean_squared_error,
                              r2_score)
from sklearn.preprocessing import StandardScaler

WORKSPACE = Path(__file__).parent.resolve()

V59_MODEL_PATH = str(WORKSPACE / "outputs/place_policy_v59/best_hier/best_model.zip")
D_CSIL_PATH = str(WORKSPACE / "data/D_csil.npz")
OUTPUT_DIR = WORKSPACE / "outputs" / "backbone_probe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_DIM = 524  # 512 (ResNet-18) + 12 (state)


def load_v59_feature_extractor(model_path, device):
    """Load V59's trained ResNet-18 features_extractor (frozen)."""
    from stable_baselines3 import PPO
    import gymnasium
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    env = VisionObs(env, image_size=84)
    vec_env = DummyVecEnv([lambda: env])
    vecnorm_path = model_path.replace("best_model.zip", "vec_normalize.pkl")
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False

    model = PPO.load(model_path, env=vec_env, device=device)
    fe = model.policy.features_extractor.to(device)
    fe.eval()
    for p in fe.parameters():
        p.requires_grad = False
    print(f"Loaded V59 features_extractor: {type(fe).__name__}, "
          f"features_dim={fe.features_dim}")
    return fe, model


def load_imagenet_feature_extractor(device):
    """Load ImageNet-only ResNet-18 (no V59 fine-tuning) for comparison."""
    import gymnasium
    import gym_env  # noqa: F401
    from gym_env.wrappers import VisionObs
    from pretrained_cnn import ResNetFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = gymnasium.make("PandaVLA-v0", reward_type="dense", gravity_comp=True)
    env = VisionObs(env, image_size=84)
    vec_env = DummyVecEnv([lambda: env])
    vecnorm_path = V59_MODEL_PATH.replace("best_model.zip", "vec_normalize.pkl")
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.norm_reward = False
        vec_env.training = False

    obs_space = vec_env.observation_space
    fe = ResNetFeaturesExtractor(obs_space, features_dim=512, backbone="resnet18")
    fe = fe.to(device)
    fe.eval()
    for p in fe.parameters():
        p.requires_grad = False
    print(f"Loaded ImageNet-only ResNet-18: features_dim={fe.features_dim}")
    return fe


@torch.no_grad()
def extract_features(fe, images, states, device, batch_size=256):
    """Extract features for all transitions.

    images: (N, 84, 84, 3) uint8 HWC
    states: (N, 12) float32
    Returns: (N, 524) float32 on CPU
    """
    n = len(images)
    all_features = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        img_batch = images[start:end]  # (B, 84, 84, 3) uint8 HWC
        state_batch = states[start:end]  # (B, 12)

        # Convert to CHW float [0, 1]
        img_tensor = torch.from_numpy(img_batch).float().permute(0, 3, 1, 2) / 255.0
        state_tensor = torch.from_numpy(state_batch).float()

        img_tensor = img_tensor.to(device)
        state_tensor = state_tensor.to(device)

        obs = {"image": img_tensor, "state": state_tensor}
        features = fe(obs)  # (B, 524)
        all_features.append(features.cpu().numpy())

        if start % (batch_size * 20) == 0:
            print(f"  Extracted {start}/{n} ({100*start/n:.1f}%)")

    return np.concatenate(all_features, axis=0)


def split_by_episode(episode_ids, train_frac=0.8):
    """Split data by episode to avoid leakage. Returns train_idx, test_idx."""
    unique_eps = np.unique(episode_ids)
    n_train = int(len(unique_eps) * train_frac)
    train_eps = set(unique_eps[:n_train])
    test_eps = set(unique_eps[n_train:])

    train_idx = np.array([i for i, ep in enumerate(episode_ids) if ep in train_eps])
    test_idx = np.array([i for i, ep in enumerate(episode_ids) if ep in test_eps])
    return train_idx, test_idx


def probe_success_classification(features, labels, train_idx, test_idx):
    """Task 1: Predict episode success from state features."""
    print("\n" + "="*60)
    print("Task 1: Success Prediction (Binary Classification)")
    print("="*60)

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"Train: {len(X_train)} samples ({100*y_train.mean():.1f}% positive)")
    print(f"Test:  {len(X_test)} samples ({100*y_test.mean():.1f}% positive)")

    # --- Linear probe (Logistic Regression) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nLinear Probe (Logistic Regression):")
    print(f"  Accuracy: {acc:.4f} ({100*acc:.1f}%)")
    print(f"  AUC:      {auc:.4f}")
    print(f"  Time:     {time.time()-t0:.1f}s")

    # --- MLP probe (1 hidden layer, 128 units) ---
    t0 = time.time()
    mlp = MLPProbe(input_dim=features.shape[1], hidden_dim=128, output_dim=1)
    mlp_results = train_mlp_probe(mlp, X_train_s, y_train, X_test_s, y_test,
                                  task='classification', epochs=50, lr=1e-3)
    print(f"\nMLP Probe (128 hidden):")
    print(f"  Accuracy: {mlp_results['acc']:.4f} ({100*mlp_results['acc']:.1f}%)")
    print(f"  AUC:      {mlp_results['auc']:.4f}")
    print(f"  Time:     {time.time()-t0:.1f}s")

    return {'linear_acc': acc, 'linear_auc': auc,
            'mlp_acc': mlp_results['acc'], 'mlp_auc': mlp_results['auc']}


def probe_action_regression(features, actions, train_idx, test_idx):
    """Task 2: Predict V59's action from state features."""
    print("\n" + "="*60)
    print("Task 2: Action Prediction (Regression)")
    print("="*60)

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = actions[train_idx], actions[test_idx]

    print(f"Train: {len(X_train)} samples, action_dim={y_train.shape[1]}")
    print(f"Test:  {len(X_test)} samples")

    # --- Linear probe (Ridge Regression) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_s, y_train)
    y_pred = reg.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Per-dimension R²
    per_dim_r2 = []
    for d in range(y_test.shape[1]):
        dim_r2 = r2_score(y_test[:, d], y_pred[:, d])
        per_dim_r2.append(dim_r2)
    print(f"\nLinear Probe (Ridge Regression):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Per-dim R²: {['%.3f' % r for r in per_dim_r2]}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # --- MLP probe ---
    t0 = time.time()
    mlp = MLPProbe(input_dim=features.shape[1], hidden_dim=128, output_dim=y_train.shape[1])
    mlp_results = train_mlp_probe(mlp, X_train_s, y_train, X_test_s, y_test,
                                  task='regression', epochs=50, lr=1e-3)
    print(f"\nMLP Probe (128 hidden):")
    print(f"  MSE:  {mlp_results['mse']:.6f}")
    print(f"  R²:   {mlp_results['r2']:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    return {'linear_mse': float(mse), 'linear_r2': float(r2),
            'mlp_mse': mlp_results['mse'], 'mlp_r2': mlp_results['r2']}


def probe_distance_regression(features, final_dists, train_idx, test_idx):
    """Task 3: Predict final distance to target from state features."""
    print("\n" + "="*60)
    print("Task 3: Distance Prediction (Regression)")
    print("="*60)

    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = final_dists[train_idx], final_dists[test_idx]

    print(f"Train: {len(X_train)} samples, dist range=[{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Test:  {len(X_test)} samples, dist range=[{y_test.min():.3f}, {y_test.max():.3f}]")

    # --- Linear probe ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_s, y_train)
    y_pred = reg.predict(X_test_s)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nLinear Probe (Ridge Regression):")
    print(f"  MSE:  {mse:.6f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # --- MLP probe ---
    t0 = time.time()
    mlp = MLPProbe(input_dim=features.shape[1], hidden_dim=128, output_dim=1)
    mlp_results = train_mlp_probe(mlp, X_train_s, y_train, X_test_s, y_test,
                                  task='regression', epochs=50, lr=1e-3)
    print(f"\nMLP Probe (128 hidden):")
    print(f"  MSE:  {mlp_results['mse']:.6f}")
    print(f"  R²:   {mlp_results['r2']:.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    return {'linear_mse': float(mse), 'linear_r2': float(r2),
            'mlp_mse': mlp_results['mse'], 'mlp_r2': mlp_results['r2']}


def probe_success_by_timestep(features, labels, episode_ids, train_idx, test_idx):
    """Bonus: Can features predict success at different points in the episode?

    This tells us if the representation captures "initial conditions" (early
    predictability) or just current progress (late predictability).
    """
    print("\n" + "="*60)
    print("Bonus: Success Prediction by Episode Timestep")
    print("="*60)

    # Compute per-episode timestep (0, 1, 2, ... within each episode)
    ep_steps = np.zeros(len(episode_ids), dtype=np.int32)
    ep_counts = {}
    for i, ep in enumerate(episode_ids):
        ep_counts[ep] = ep_counts.get(ep, 0)
        ep_steps[i] = ep_counts[ep]
        ep_counts[ep] += 1

    # Bin timesteps: early (0-20%), mid (20-60%), late (60-100%)
    max_step = 500
    bins = [(0, int(0.2*max_step)), (int(0.2*max_step), int(0.6*max_step)),
            (int(0.6*max_step), max_step)]
    bin_names = ['Early (0-20%)', 'Mid (20-60%)', 'Late (60-100%)']

    scaler = StandardScaler()
    train_mask = np.zeros(len(features), dtype=bool)
    train_mask[train_idx] = True
    test_mask = np.zeros(len(features), dtype=bool)
    test_mask[test_idx] = True

    X_train_all = scaler.fit_transform(features[train_mask])
    X_test_all = scaler.transform(features[test_mask])

    # Train on all train data, evaluate per-bin on test data
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    ep_steps_test = ep_steps[test_mask]

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train_all, y_train)

    results = {}
    for (lo, hi), name in zip(bins, bin_names):
        mask = (ep_steps_test >= lo) & (ep_steps_test < hi)
        if mask.sum() < 10:
            continue
        y_pred = clf.predict(X_test_all[mask])
        y_prob = clf.predict_proba(X_test_all[mask])[:, 1]
        acc = accuracy_score(y_test[mask], y_pred)
        try:
            auc = roc_auc_score(y_test[mask], y_prob)
        except ValueError:
            auc = 0.5
        n = mask.sum()
        pos_rate = y_test[mask].mean()
        print(f"  {name}: n={n}, pos_rate={100*pos_rate:.1f}%, "
              f"acc={100*acc:.1f}%, auc={auc:.4f}")
        results[name] = {'n': int(n), 'pos_rate': float(pos_rate),
                         'acc': float(acc), 'auc': float(auc)}

    return results


class MLPProbe(nn.Module):
    """Simple 2-layer MLP probe: input -> hidden -> output."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp_probe(model, X_train, y_train, X_test, y_test, task, epochs=50, lr=1e-3):
    """Train MLP probe with PyTorch."""
    device = next(model.parameters()).device
    X_train_t = torch.from_numpy(X_train).float().to(device)
    X_test_t = torch.from_numpy(X_test).float().to(device)

    if task == 'classification':
        y_train_t = torch.from_numpy(y_train).float().to(device).unsqueeze(1)
        y_test_t = torch.from_numpy(y_test).float().to(device).unsqueeze(1)
        criterion = nn.BCEWithLogitsLoss()
    else:
        y_train_t = torch.from_numpy(y_train).float().to(device)
        if y_train_t.dim() == 1:
            y_train_t = y_train_t.unsqueeze(1)
        y_test_t = torch.from_numpy(y_test).float().to(device)
        if y_test_t.dim() == 1:
            y_test_t = y_test_t.unsqueeze(1)
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 256
    n = len(X_train_t)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).cpu().numpy()

    if task == 'classification':
        pred_labels = (pred_test.squeeze() > 0).astype(int)
        pred_prob = 1 / (1 + np.exp(-pred_test.squeeze()))
        acc = accuracy_score(y_test, pred_labels)
        try:
            auc = roc_auc_score(y_test, pred_prob)
        except ValueError:
            auc = 0.5
        return {'acc': float(acc), 'auc': float(auc)}
    else:
        mse = mean_squared_error(y_test, pred_test.squeeze())
        r2 = r2_score(y_test, pred_test.squeeze())
        return {'mse': float(mse), 'r2': float(r2)}


def main():
    parser = argparse.ArgumentParser(description="Backbone Probe for V59 ResNet-18")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--compare_imagenet', action='store_true',
                        help='Also probe ImageNet-only ResNet-18 for comparison.')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # ---- Load data ----
    print(f"\nLoading D_csil from {D_CSIL_PATH}...")
    data = np.load(D_CSIL_PATH)
    images = data['images']       # (45244, 84, 84, 3) uint8
    states = data['states']       # (45244, 12) float32
    actions = data['actions']     # (45244, 8) float32
    labels = data['labels']       # (45244,) int64
    episode_ids = data['episode_ids']  # (45244,) int64
    final_dists = data['final_dists']  # (45244,) float32
    print(f"Loaded {len(images)} transitions from {len(np.unique(episode_ids))} episodes")

    # ---- Split by episode ----
    train_idx, test_idx = split_by_episode(episode_ids, train_frac=0.8)
    print(f"Train: {len(train_idx)} transitions from {len(set(episode_ids[train_idx]))} episodes")
    print(f"Test:  {len(test_idx)} transitions from {len(set(episode_ids[test_idx]))} episodes")

    all_results = {}

    # ---- Extract V59-fine-tuned features ----
    print(f"\n{'='*60}")
    print("Extracting V59-fine-tuned ResNet-18 features...")
    print(f"{'='*60}")
    fe_v59, _ = load_v59_feature_extractor(V59_MODEL_PATH, device)
    t0 = time.time()
    features_v59 = extract_features(fe_v59, images, states, device, args.batch_size)
    print(f"V59 features: shape={features_v59.shape}, time={time.time()-t0:.1f}s")

    # Free GPU memory
    del fe_v59
    torch.cuda.empty_cache() if device == 'cuda' else None

    # ---- Run probes on V59 features ----
    print(f"\n{'#'*60}")
    print("# V59-FINE-TUNED ResNet-18 FEATURES")
    print(f"{'#'*60}")

    all_results['v59'] = {}
    all_results['v59']['success'] = probe_success_classification(
        features_v59, labels, train_idx, test_idx)
    all_results['v59']['action'] = probe_action_regression(
        features_v59, actions, train_idx, test_idx)
    all_results['v59']['distance'] = probe_distance_regression(
        features_v59, final_dists, train_idx, test_idx)
    all_results['v59']['by_timestep'] = probe_success_by_timestep(
        features_v59, labels, episode_ids, train_idx, test_idx)

    # ---- Optional: Compare with ImageNet-only ResNet-18 ----
    if args.compare_imagenet:
        print(f"\n{'='*60}")
        print("Extracting ImageNet-only ResNet-18 features...")
        print(f"{'='*60}")
        fe_imagenet = load_imagenet_feature_extractor(device)
        t0 = time.time()
        features_imagenet = extract_features(fe_imagenet, images, states, device, args.batch_size)
        print(f"ImageNet features: shape={features_imagenet.shape}, time={time.time()-t0:.1f}s")

        del fe_imagenet
        torch.cuda.empty_cache() if device == 'cuda' else None

        print(f"\n{'#'*60}")
        print("# IMAGENET-ONLY ResNet-18 FEATURES (no V59 fine-tuning)")
        print(f"{'#'*60}")

        all_results['imagenet'] = {}
        all_results['imagenet']['success'] = probe_success_classification(
            features_imagenet, labels, train_idx, test_idx)
        all_results['imagenet']['action'] = probe_action_regression(
            features_imagenet, actions, train_idx, test_idx)
        all_results['imagenet']['distance'] = probe_distance_regression(
            features_imagenet, final_dists, train_idx, test_idx)

    # ---- Save results ----
    output_path = OUTPUT_DIR / "probe_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    v59_success = all_results['v59']['success']
    v59_action = all_results['v59']['action']
    v59_distance = all_results['v59']['distance']

    print(f"\nV59 ResNet-18 Features (524-dim):")
    print(f"  Success Prediction:")
    print(f"    Linear: acc={100*v59_success['linear_acc']:.1f}%, auc={v59_success['linear_auc']:.4f}")
    print(f"    MLP:    acc={100*v59_success['mlp_acc']:.1f}%, auc={v59_success['mlp_auc']:.4f}")
    print(f"  Action Prediction:")
    print(f"    Linear: R²={v59_action['linear_r2']:.4f}")
    print(f"    MLP:    R²={v59_action['mlp_r2']:.4f}")
    print(f"  Distance Prediction:")
    print(f"    Linear: R²={v59_distance['linear_r2']:.4f}")
    print(f"    MLP:    R²={v59_distance['mlp_r2']:.4f}")

    if 'imagenet' in all_results:
        img_success = all_results['imagenet']['success']
        img_action = all_results['imagenet']['action']
        img_distance = all_results['imagenet']['distance']
        print(f"\nImageNet-only ResNet-18 Features:")
        print(f"  Success Prediction:")
        print(f"    Linear: acc={100*img_success['linear_acc']:.1f}%, auc={img_success['linear_auc']:.4f}")
        print(f"    MLP:    acc={100*img_success['mlp_acc']:.1f}%, auc={img_success['mlp_auc']:.4f}")
        print(f"  Action Prediction:")
        print(f"    Linear: R²={img_action['linear_r2']:.4f}")
        print(f"    MLP:    R²={img_action['mlp_r2']:.4f}")
        print(f"  Distance Prediction:")
        print(f"    Linear: R²={img_distance['linear_r2']:.4f}")
        print(f"    MLP:    R²={img_distance['mlp_r2']:.4f}")

    # ---- Interpretation ----
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    best_acc = max(v59_success['linear_acc'], v59_success['mlp_acc'])
    if best_acc > 0.70:
        print(f"  Success prediction acc={100*best_acc:.1f}% > 70% threshold")
        print(f"  => Representation is SUFFICIENT. Bottleneck is in policy head.")
        print(f"  => Diffusion/transformer may NOT help. Focus on policy head / distribution shift.")
    elif best_acc > 0.60:
        print(f"  Success prediction acc={100*best_acc:.1f}% (60-70%, marginal)")
        print(f"  => Representation is BORDERLINE. Some information is captured but not enough.")
        print(f"  => Consider both policy head improvements and representation improvements.")
    else:
        print(f"  Success prediction acc={100*best_acc:.1f}% < 60% threshold")
        print(f"  => Representation is INSUFFICIENT. Bottleneck is in the encoder.")
        print(f"  => Architectural changes (diffusion, transformer) ARE justified.")

    best_action_r2 = max(v59_action['linear_r2'], v59_action['mlp_r2'])
    print(f"\n  Action prediction R²={best_action_r2:.4f}")
    if best_action_r2 > 0.5:
        print(f"  => Features contain strong action information (R²>0.5)")
    elif best_action_r2 > 0.2:
        print(f"  => Features contain moderate action information (0.2<R²<0.5)")
    else:
        print(f"  => Features contain weak action information (R²<0.2)")

    best_dist_r2 = max(v59_distance['linear_r2'], v59_distance['mlp_r2'])
    print(f"\n  Distance prediction R²={best_dist_r2:.4f}")
    if best_dist_r2 > 0.5:
        print(f"  => Features strongly predict task progress (R²>0.5)")
    elif best_dist_r2 > 0.2:
        print(f"  => Features moderately predict task progress (0.2<R²<0.5)")
    else:
        print(f"  => Features weakly predict task progress (R²<0.2)")


if __name__ == "__main__":
    main()
