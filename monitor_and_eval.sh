#!/bin/bash
# Monitor DAPG 500K and VLA-GRPO training, auto-eval when done
# Usage: bash monitor_and_eval.sh

DAPG_PID=3064871
VLA_PID=3119963
WS=/home/w/vla_workspace

echo "=== Training Monitor Started $(date) ==="
echo "DAPG PID: $DAPG_PID"
echo "VLA  PID: $VLA_PID"

dapg_done=false
vla_done=false

while [ "$dapg_done" = false ] || [ "$vla_done" = false ]; do
    now=$(date '+%H:%M:%S')

    # Check DAPG
    if [ "$dapg_done" = false ]; then
        if kill -0 $DAPG_PID 2>/dev/null; then
            # Read latest eval progress
            progress=$(cd $WS && python -c "
import numpy as np
try:
    d = np.load('outputs/dapg_500k/eval_logs/evaluations.npz')
    ts = d['timesteps']
    r = d['results']
    print(f'Step {ts[-1]}/500K ({100*ts[-1]/500000:.0f}%) | last_mean={r[-1].mean():.1f} | best={r.mean(axis=1).max():.1f}')
except: print('reading...')
" 2>/dev/null)
            echo "[$now] DAPG: $progress"
        else
            echo "[$now] *** DAPG TRAINING COMPLETE! ***"
            dapg_done=true
        fi
    fi

    # Check VLA-GRPO
    if [ "$vla_done" = false ]; then
        if kill -0 $VLA_PID 2>/dev/null; then
            elapsed=$(ps -p $VLA_PID -o etime= 2>/dev/null | tr -d ' ')
            echo "[$now] VLA-GRPO: running (elapsed=$elapsed)"
        else
            echo "[$now] *** VLA-GRPO TRAINING COMPLETE! ***"
            vla_done=true
        fi
    fi

    # If DAPG just completed, run eval immediately
    if [ "$dapg_done" = true ] && [ ! -f /tmp/dapg_eval_done ]; then
        touch /tmp/dapg_eval_done
        echo "[$now] >>> Running DAPG 500K evaluation..."
        cd $WS && python eval_comparison.py --n_episodes 20 --seed 42 > /tmp/dapg_eval_result.txt 2>&1
        echo "[$now] >>> DAPG eval complete! See /tmp/dapg_eval_result.txt"
        cat /tmp/dapg_eval_result.txt
    fi

    # If VLA-GRPO just completed, check results
    if [ "$vla_done" = true ] && [ ! -f /tmp/vla_check_done ]; then
        touch /tmp/vla_check_done
        echo "[$now] >>> Checking VLA-GRPO outputs..."
        ls -la $WS/outputs/grpo_vla/ 2>/dev/null
    fi

    sleep 60
done

echo ""
echo "=== ALL TRAINING COMPLETE $(date) ==="
echo "DAPG eval results:"
cat /tmp/dapg_eval_result.txt 2>/dev/null
echo ""
echo "VLA-GRPO outputs:"
ls -la $WS/outputs/grpo_vla/ 2>/dev/null
