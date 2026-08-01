#!/bin/bash
# CSIL++ end-to-end pipeline (Task 5).
# Assumes D_csil.npz already collected by `train_csil_plus_plus.py collect`.
set -e

PYTHON=/home/w/miniconda3/envs/vla/bin/python
WORKSPACE=/home/w/vla_workspace
cd "$WORKSPACE"

echo "============================================================"
echo "CSIL++ Pipeline — Task 5"
echo "============================================================"
date

# Step 1: Train potential function Φ on D_csil
echo ""
echo "=== Step 1: Train potential function Φ ==="
$PYTHON train_csil_plus_plus.py train-reward \
    --n_epochs 100 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    2>&1 | tee outputs/csil_plus_plus/train_reward_log.txt

# Step 2: Verify coherent reward
echo ""
echo "=== Step 2: Verify coherent reward ==="
$PYTHON train_csil_plus_plus.py verify 2>&1 | tee outputs/csil_plus_plus/verify_log.txt

# Step 3: PBRS fine-tune a_pi (conservative)
echo ""
echo "=== Step 3: PBRS fine-tune a_pi (20 iterations, ~10k steps) ==="
$PYTHON train_csil_plus_plus.py train-ensemble \
    --max_iterations 20 \
    --n_steps_per_rollout 512 \
    --learning_rate 1e-7 \
    --clip_range 0.1 \
    --max_kl 0.005 \
    --eval_every 5 \
    --eval_episodes 15 \
    --safety_threshold 0.30 \
    --experiment_id CSIL_PLUS_PLUS_V1 \
    2>&1 | tee outputs/csil_plus_plus/train_ensemble_log.txt

# Step 4: 50-ep eval with ensemble policy
# Note: This requires integrating EnsemblePolicy with eval_hierarchical.py.
# For now, the train-ensemble command already runs periodic evals.
# A full 50-ep eval can be run separately if the ensemble survives training.

echo ""
echo "============================================================"
echo "CSIL++ Pipeline Complete"
echo "============================================================"
date
echo ""
echo "Results:"
echo "  - Potential function: outputs/csil_plus_plus/potential_fn.pt"
echo "  - Verification report: outputs/csil_plus_plus/verification_report.json"
echo "  - Ensemble policy: outputs/csil_plus_plus/ensemble_policy.pt"
echo "  - Training log: outputs/csil_plus_plus/training_log.json"
echo ""
echo "Check training_log.json for per-iteration metrics and safety_rollback status."
