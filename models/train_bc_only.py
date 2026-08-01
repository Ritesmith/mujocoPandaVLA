"""BC-only fine-tuning: supervised learning on D_succ without PPO.

Loads a pretrained place policy (V59), freezes the ResNet-18 backbone, and
fine-tunes only the MLP head using behavioral cloning on the self-imitation
buffer (D_succ). No PPO policy gradient, no value function, no rollouts.

Usage:
    python train_bc_only.py \
        --load_model outputs/place_policy_v59/best_hier/best_model.zip \
        --load_vecnorm outputs/place_policy_v59/best_hier/vec_normalize.pkl \
        --d_succ data/D_succ.npz \
        --save_path outputs/place_policy_bc_v1 \
        --experiment_id BC_V1 \
        --n_epochs 50 --batch_size 320 --learning_rate 1e-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from stable_baselines3.common.vec_env import VecNormalize

from auto_iter.case_memory import CaseMemory
from auto_iter.metadata import record_experiment
from auto_iter.version_tree import EvalPoint, VersionTree, make_node


def freeze_backbone(model):
    """Freeze ResNet-18 features_extractor, only train MLP head."""
    fe = model.policy.features_extractor
    for p in fe.parameters():
        p.requires_grad = False
    # Also freeze BN running stats
    for m in fe.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"Backbone frozen: {total} total params, {trainable} trainable")
    return trainable


def freeze_gripper_dim(model):
    """Freeze the gripper output dimension (index 7) in action_net.

    Only arm joints (dimensions 0-6) will be trained; the gripper
    dimension retains V59's initial weights.
    """
    action_net = model.policy.action_net
    action_net.weight.data[7].requires_grad = False
    action_net.bias.data[7].requires_grad = False

    # Verify
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    gripper_frozen = (action_net.weight.data.shape[0] - 1) * action_net.weight.data.shape[1] + (action_net.bias.data.shape[0] - 1)
    print(f"Gripper dim frozen: {total} total params, {trainable} trainable (froze {gripper_frozen} gripper params)")
    return trainable


def load_d_succ(path: str):
    """Load D_succ.npz → (images, states, actions) tensors."""
    data = np.load(path, allow_pickle=True)
    images = torch.as_tensor(data["images"], dtype=torch.float32)  # (N, 84, 84, 3)
    states = torch.as_tensor(data["states"], dtype=torch.float32)  # (N, 12)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)  # (N, 8)
    print(f"D_succ loaded: {len(actions)} transitions")
    return images, states, actions


def normalize_states(states, vecnorm_path):
    """Normalize states using V59's VecNormalize statistics (loaded from pickle)."""
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


def compute_bc_loss(model, images, states, actions, device):
    """Forward pass through policy → MSE between predicted and demo actions."""
    obs = {"image": images.to(device), "state": states.to(device)}
    features = model.policy.extract_features(obs)
    latent = model.policy.mlp_extractor.forward_actor(features)
    pred_actions = model.policy.action_net(latent)
    return nn.functional.mse_loss(pred_actions, actions.to(device))


def train_bc(
    model,
    images,
    states,
    actions,
    n_epochs=50,
    batch_size=320,
    learning_rate=1e-5,
    device="cuda",
    log_interval=10,
):
    """Supervised BC training loop. Returns loss history."""
    dataset = TensorDataset(images, states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    trainable_params = [p for p in model.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    model.policy.set_training_mode(True)
    # Keep backbone in eval mode (frozen BN)
    fe = model.policy.features_extractor
    fe.eval()

    loss_history = []
    for epoch in range(n_epochs):
        epoch_losses = []
        for batch_imgs, batch_states, batch_acts in loader:
            # Images: (B, 84, 84, 3) → permute to (B, 3, 84, 84)
            batch_imgs = batch_imgs.permute(0, 3, 1, 2).contiguous()

            loss = compute_bc_loss(model, batch_imgs, batch_states, batch_acts, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.3)
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: bc_loss={avg_loss:.6f}")

    return loss_history


def main():
    parser = argparse.ArgumentParser(description="BC-only fine-tuning (no PPO)")
    parser.add_argument("--load_model", type=str, required=True)
    parser.add_argument("--load_vecnorm", type=str, default=None)
    parser.add_argument("--d_succ", type=str, default="data/D_succ.npz")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, default="BC_V1")
    parser.add_argument("--parent_id", type=str, default="V59")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=320)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_gripper", action="store_true", help="Freeze gripper output dimension (dim 7) during training")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Query case memory for known BC failure modes
    cm = CaseMemory()
    bc_config = {"optimization_method": "BC", "lambda_bc": 1.0}
    warnings = cm.query(bc_config, "BC")
    if warnings:
        print("\n=== Case Memory Warnings ===")
        for w in warnings:
            print(f"  WARNING: {w.evidence_id}: {w.failure_mode}")
            print(f"    {w.recommendation[:100]}")
    else:
        print("Case memory: no BC-specific failure modes recorded (expected)")

    # Record metadata
    print(f"\n=== Experiment: {args.experiment_id} ===")
    meta = record_experiment(
        experiment_id=args.experiment_id,
        optimization_method="BC",
        parent_experiment_id=args.parent_id,
        decision_reason="BC-only fine-tuning on D_succ (self-imitation). No PPO gradient. Test if non-PPO path can improve V59.",
        random_seed=args.seed,
        training_config={
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "backbone": "frozen",
            "optimizer": "Adam",
            "max_grad_norm": 0.3,
            "freeze_gripper": args.freeze_gripper,
        },
        eval_config={"episodes": 50, "method": "hier_eval"},
        train_cmd=" ".join(sys.argv),
        save_path=args.save_path,
        cwd=str(WORKSPACE),
    )
    print(f"Metadata recorded: git={meta.git_commit[:12]}, env={meta.env_versions.get('mujoco', '?')}")

    # Load D_succ
    print(f"\n=== Loading D_succ from {args.d_succ} ===")
    images, states, actions = load_d_succ(args.d_succ)
    states = normalize_states(states, args.load_vecnorm)

    # Load pretrained model
    print(f"\n=== Loading model from {args.load_model} ===")
    from train_dapg import DAPGPPO
    from train_place_policy import make_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage
    import functools
    import pickle

    grasp_states_path = str(WORKSPACE / "outputs/grasp_states_v5_500.pkl")
    grasp_states = None
    if os.path.exists(grasp_states_path):
        with open(grasp_states_path, "rb") as f:
            grasp_states = pickle.load(f)

    target_pos_range = [[0.35, 0.15, 0.22], [0.65, 0.45, 0.22]]
    env_kwargs = dict(
        grasp_states=grasp_states,
        release_threshold=0.05,
        target_pos_range=target_pos_range,
        vision_mode=True,
        domain_randomize=False,
        better_reward=False,
        use_pbrs=False,
        pbrs_alpha=1.0, pbrs_beta=0.0, pbrs_scale=0.5,
    )
    train_env = DummyVecEnv([functools.partial(make_env, **env_kwargs)])
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        norm_obs_keys=["state"],
    )
    if args.load_vecnorm and os.path.exists(args.load_vecnorm):
        train_env = VecNormalize.load(args.load_vecnorm, train_env)
    train_env = VecTransposeImage(train_env)

    model = DAPGPPO.load(
        args.load_model,
        env=train_env,
        device=args.device,
        demo_obs=None,
        demo_actions=None,
    )
    print("Model loaded.")

    # Freeze backbone
    freeze_backbone(model)

    # Optionally freeze gripper output dimension (dim 7) to retain V59's gripper behavior
    if args.freeze_gripper:
        freeze_gripper_dim(model)

    # Initial BC loss (before training)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test_imgs = images[:320].permute(0, 3, 1, 2).contiguous()
    test_states = states[:320]
    test_acts = actions[:320]
    with torch.no_grad():
        init_loss = compute_bc_loss(model, test_imgs, test_states, test_acts, device)
    print(f"\nInitial BC loss (before training): {init_loss.item():.6f}")

    # Train
    print(f"\n=== BC Training ({args.n_epochs} epochs, lr={args.learning_rate}) ===")
    start_time = time.time()
    loss_history = train_bc(
        model, images, states, actions,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
    )
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.0f}s")
    print(f"  Initial loss: {loss_history[0]:.6f}")
    print(f"  Final loss:   {loss_history[-1]:.6f}")
    print(f"  Loss change:  {loss_history[0] - loss_history[-1]:.6f}")

    # Save model
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / "best_model.zip"
    model.save(str(model_save_path))
    print(f"\nModel saved to {model_save_path}")

    # Save vec_normalize
    if args.load_vecnorm and os.path.exists(args.load_vecnorm):
        import shutil
        shutil.copy(args.load_vecnorm, save_dir / "vec_normalize.pkl")
        print(f"vec_normalize.pkl copied to {save_dir}")

    # Save training log
    log = {
        "experiment_id": args.experiment_id,
        "method": "BC-only",
        "init_bc_loss": loss_history[0],
        "final_bc_loss": loss_history[-1],
        "loss_history": loss_history,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(save_dir / "bc_train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Add to version tree
    tree = VersionTree()
    node = make_node(
        experiment_id=args.experiment_id,
        parent_id=args.parent_id,
        optimization_method="BC",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        decision_reason="BC-only fine-tuning on D_succ. No PPO gradient. Test non-PPO path.",
        config={
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "backbone": "frozen",
            "freeze_gripper": args.freeze_gripper,
            "d_succ_size": len(actions),
            "init_bc_loss": loss_history[0],
            "final_bc_loss": loss_history[-1],
        },
    )
    node.status = "completed"
    node.verdict = "pending"  # Pending eval
    tree.add_node(node)
    print(f"\nAdded to version tree: {args.experiment_id} (parent={args.parent_id})")

    print(f"\n=== Next: Run hier eval on {model_save_path} ===")
    print(f"  python eval_place_policy.py --model {model_save_path} ...")


if __name__ == "__main__":
    main()
