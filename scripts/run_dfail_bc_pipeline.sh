#!/bin/bash
# D_fail collection + GMM clustering + Shallow BC training pipeline.
# Runs after Voronoi pipeline completes (shares GPU/EGL).
set -eo pipefail

PYTHON=/home/w/miniconda3/envs/vla/bin/python
WORKSPACE=/home/w/vla_workspace
cd "$WORKSPACE"

echo "============================================================"
echo "D_fail + Clustering + Shallow BC Pipeline"
echo "============================================================"
date

# Step 1: Collect D_fail (100 episodes, ~20 min)
echo ""
echo "=== Step 1: Collect D_fail (100 episodes) ==="
if [ ! -f data/D_fail.npz ]; then
    $PYTHON -u -m data.collect_d_fail --n_episodes 100 \
        2>&1 | tee outputs/csil_plus_plus/dfail_collect_log.txt
else
    echo "  D_fail.npz already exists, skipping collection."
fi

# Step 2: GMM clustering (CPU-only, ~1 min)
echo ""
echo "=== Step 2: GMM Failure Mode Clustering ==="
$PYTHON -u -m diagnostics.failure_mode_clustering \
    2>&1 | tee outputs/csil_plus_plus/gmm_cluster_log.txt

# Step 3: Shallow BC head training (GPU, ~5-10 min)
echo ""
echo "=== Step 3: Shallow BC Head Training + Gate 1a ==="
$PYTHON -u -m models.train_shallow_bc \
    --n_epochs 100 \
    --batch_size 256 \
    --lr 1e-3 \
    2>&1 | tee outputs/csil_plus_plus/shallow_bc_log.txt

echo ""
echo "============================================================"
echo "Pipeline Complete"
echo "============================================================"
date
echo ""
echo "Outputs:"
echo "  - D_fail: data/D_fail.npz"
echo "  - GMM clusters: outputs/csil_plus_plus/failure_clusters.npz"
echo "  - Cluster table: outputs/csil_plus_plus/cluster_table.json"
echo "  - Shallow BC head: outputs/csil_plus_plus/shallow_bc_head.pt"
echo "  - Gate 1a report: outputs/csil_plus_plus/gate_1a_report.json"
