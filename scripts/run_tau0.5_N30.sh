#!/bin/bash

# 强制加载 conda 环境
source ~/miniconda3/bin/activate vla

WORKSPACE="/home/w/vla_workspace"
EXP_DIR="${WORKSPACE}/outputs/phase7_round2a_tau0.5_N30"
TRAIN_SCRIPT="core.train_iql"
cd "${WORKSPACE}"

TAU=0.5
MAX_PARALLEL=2  # 保守设置，确保显存不炸

mkdir -p ${EXP_DIR}

echo "=========================================="
echo "Conda Env: $CONDA_DEFAULT_ENV"
echo "Python Path: $(which python)"
echo "Experiment Dir: ${EXP_DIR}"
echo "Max Parallel Jobs: ${MAX_PARALLEL}"
echo "=========================================="

count=0
for SEED in $(seq 20 49); do
    if [ $count -ge $MAX_PARALLEL ]; then
        wait -n
        count=$((count - 1))
    fi
    
    LOG_FILE="${EXP_DIR}/train_seed${SEED}.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Seed: ${SEED}"
    
    # 使用 vla 环境下的 python 运行
    nohup python -m ${TRAIN_SCRIPT} \
        --seed ${SEED} \
        --tau ${TAU} \
        > ${LOG_FILE} 2>&1 &
    
    count=$((count + 1))
done

wait
echo "=========================================="
echo "All 30 experiments finished at $(date)!"
echo "=========================================="
