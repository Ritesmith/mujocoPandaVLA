#!/bin/bash
# Voronoi partition end-to-end pipeline (Tasks 6-8).
# Partitions V59's state space, trains sub-policies, evaluates RouterPolicy.
set -eo pipefail

PYTHON=/home/w/miniconda3/envs/vla/bin/python
WORKSPACE=/home/w/vla_workspace
cd "$WORKSPACE"
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "Voronoi Pipeline — Tasks 6-8"
echo "============================================================"
date

# Step 1: Collect V59 state values (200 episodes)
echo ""
echo "=== Step 1: Collect V59 state values (200 episodes) ==="
if [ ! -f data/voronoi_states.npz ]; then
    $PYTHON voronoi_partition.py collect --n_episodes 200 2>&1 | tee outputs/csil_plus_plus/voronoi_collect_log.txt
else
    echo "  voronoi_states.npz already exists, skipping collection."
fi

# Step 2: Compute threshold + K-means partition
echo ""
echo "=== Step 2: Compute threshold + K-means partition (K=4) ==="
$PYTHON voronoi_partition.py partition --k 4 2>&1 | tee outputs/csil_plus_plus/voronoi_partition_log.txt

# Step 3: Verify partition quality
echo ""
echo "=== Step 3: Verify partition quality ==="
$PYTHON voronoi_partition.py verify 2>&1 | tee outputs/csil_plus_plus/voronoi_verify_log.txt

# Step 4: Collect per-cell trajectory data (50 episodes per cell)
echo ""
echo "=== Step 4: Collect per-cell trajectory data (50 eps/cell) ==="
$PYTHON voronoi_partition.py collect-cell-data --n_episodes_per_cell 50 2>&1 | tee outputs/csil_plus_plus/voronoi_cell_collect_log.txt

# Step 5: Train K sub-policies
echo ""
echo "=== Step 5: Train K sub-policies (BC + CSIL++ PBRS) ==="
$PYTHON voronoi_partition.py train-sub-policies \
    --n_iterations 10 \
    --learning_rate 1e-7 \
    --max_kl 0.005 \
    --experiment_id VORONOI_V1 \
    2>&1 | tee outputs/csil_plus_plus/voronoi_train_sub_log.txt

# Step 6: Evaluate RouterPolicy (50 episodes)
echo ""
echo "=== Step 6: Evaluate RouterPolicy (50 episodes) ==="
$PYTHON voronoi_partition.py route --n_episodes 50 2>&1 | tee outputs/csil_plus_plus/voronoi_route_eval_log.txt

echo ""
echo "============================================================"
echo "Voronoi Pipeline Complete"
echo "============================================================"
date
echo ""
echo "Results:"
echo "  - Partition: outputs/csil_plus_plus/voronoi_partition.json"
echo "  - Verification: outputs/csil_plus_plus/voronoi_verification_report.json"
echo "  - Sub-policies: outputs/csil_plus_plus/sub_policies/cell_{k}_policy.pt"
echo "  - RouterPolicy: outputs/csil_plus_plus/router_policy.pt"
