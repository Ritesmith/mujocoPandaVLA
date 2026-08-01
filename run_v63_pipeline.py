#!/usr/bin/env python3
"""V63 pipeline — Reward fix validation (v12 gating + early release penalty).

Root cause recap (V59-V62):
  All versions degraded at step 2500-3000 (50% -> 25-35%) despite different
  BC/OPR strategies. V62 (OPR self-imitation, bc_loss=0.002) confirmed the BC
  direction is correct but still degraded — proving the root cause is NOT in
  the policy regularization side but in the REWARD side.

  The raw "place_only" reward has a hackable distance penalty: -10*dist is
  applied EVERY step, accumulating over episode length. The policy can reduce
  total penalty by ending episodes early (drop block -> shorter episode ->
  less accumulated penalty). This is classic reward hacking (Goodhart's law).

V63 core change (v12 reward fix, in gym_env/panda_vla_env.py):
  1. GATE all continuous/progress rewards on is_holding (gripper closed).
     Distance penalty, height penalty, block progress, lowering bonus now
     ONLY apply while the block is held. Once released, they stop — the
     policy can no longer "save" penalty by dropping the block.
  2. ADD one-time -5 early release penalty: if the block is released away
     from the target (dist>=0.05 or off table), apply -5 once. This closes
     the "release and do nothing" escape path (0 > negative holding penalty).

  Unchanged: one-time bonuses (proximity, approach) and success/release
  bonuses — they were never hackable.

Single-variable isolation:
  V63 = V62 config + v12 reward fix. Same hyperparameters, same D_succ.npz
  OPR buffer, same V59 best_hier starting point. The ONLY difference is the
  reward function. If V63 survives step 3000 (where V60/V61/V62 all degraded),
  the reward fix is confirmed as the root cause solution.

Critical validation: step 3000 eval
  V60 (external BC λ=0.3):  15% at step 3000 (crashed)
  V61 (external BC λ=0.01): 25% at step 3000 (crashed)
  V62 (OPR λ=0.1):          35% at step 3000 (degraded, early stop)
  V63 (OPR + reward fix):   target >= 50% at step 3000 (survival)

Pipeline:
  1. Convert D_succ.pkl -> D_succ.npz (reuse V62's buffer)
  2. Train from V59 best_hier with v12 reward fix + OPR (lambda_si=0.1)
  3. 5000 steps, hier eval every 1000, early stop + decoupling detection
  4. Final 50-ep eval of best_hier

Usage:
    python run_v63_pipeline.py
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import subprocess
import time
import re
import pickle
import shutil
import numpy as np
from pathlib import Path

PYTHON = "/home/w/miniconda3/envs/vla/bin/python"
WORKSPACE = "/home/w/vla_workspace"
SAVE_PATH = os.path.join(WORKSPACE, "outputs/place_policy_v63")
TRAIN_LOG = os.path.join(WORKSPACE, "outputs/v63_train.log")
PIPELINE_LOG = os.path.join(WORKSPACE, "outputs/v63_pipeline.log")

V59_BEST_HIER_MODEL = os.path.join(WORKSPACE, "outputs/place_policy_v59/best_hier/best_model.zip")
V59_BEST_HIER_VECNORM = os.path.join(WORKSPACE, "outputs/place_policy_v59/best_hier/vec_normalize.pkl")

GRASP_STATES_V5_500 = os.path.join(WORKSPACE, "outputs/grasp_states_v5_500.pkl")
GRASP_MODEL = os.path.join(WORKSPACE, "outputs/dapg_800k_v5/best/best_model.zip")
GRASP_VECNORM = os.path.join(WORKSPACE, "outputs/dapg_800k_v5/vec_normalize.pkl")
TARGET_RANGE = "0.35,0.15,0.22,0.65,0.45,0.22"

D_SUCC_PKL = os.path.join(WORKSPACE, "data/D_succ.pkl")
D_SUCC_NPZ = os.path.join(WORKSPACE, "data/D_succ.npz")


def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    line = "[%s] %s" % (timestamp, msg)
    print(line, flush=True)
    with open(PIPELINE_LOG, 'a') as f:
        f.write(line + "\n")


def convert_pkl_to_npz():
    """Convert D_succ.pkl (list of trajectory dicts) to D_succ.npz."""
    if not os.path.exists(D_SUCC_NPZ):
        if not os.path.exists(D_SUCC_PKL):
            log("ERROR: D_succ.pkl not found at %s" % D_SUCC_PKL)
            return False
        with open(D_SUCC_PKL, 'rb') as f:
            trajectories = pickle.load(f)
        all_images, all_states, all_actions = [], [], []
        for traj in trajectories:
            all_images.append(traj["image"])
            all_states.append(traj["state"])
            all_actions.append(traj["action"])
        images = np.concatenate(all_images, axis=0)
        states = np.concatenate(all_states, axis=0)
        actions = np.concatenate(all_actions, axis=0)
        np.savez_compressed(D_SUCC_NPZ,
                            images=images.astype(np.uint8),
                            states=states.astype(np.float32),
                            actions=actions.astype(np.float32))
        log("Converted D_succ.pkl -> D_succ.npz: %d transitions" % len(actions))
    else:
        data = np.load(D_SUCC_NPZ)
        log("D_succ.npz exists: %d transitions" % len(data["actions"]))
    return True


TRAIN_CMD = [
    PYTHON, "-u", "train_place_policy.py",
    "--vision_mode", "--pretrained_cnn", "--image_augment", "--freeze_bn",
    # V63: same OPR self-imitation buffer as V62 (single-variable: only reward changed)
    "--vision_demos", D_SUCC_NPZ,
    "--lambda_bc", "0.1",
    "--bc_decay", "1.0",
    "--learning_rate", "5e-6",
    "--lr_schedule", "cosine",
    "--lr_final", "1e-6",
    "--weight_decay", "1e-4",
    "--clip_range", "0.15",
    "--total_timesteps", "5000",
    "--target_pos_range", TARGET_RANGE,
    "--release_threshold", "0.10",
    "--n_epochs", "10",
    "--grasp_states", GRASP_STATES_V5_500,
    "--use_pbrs", "--pbrs_alpha", "1.0", "--pbrs_beta", "0.0",
    "--pbrs_scale", "0.5",
    "--ent_coef", "0.005",
    "--max_grad_norm", "0.3",
    "--target_kl", "0.015",
    "--eval_episodes", "20",
    "--early_stop_patience", "8",
    "--place_eval_freq", "1000",
    "--hier_eval_freq", "1000",
    "--hier_eval_episodes", "20",
    "--grasp_model", GRASP_MODEL,
    "--grasp_vecnorm", GRASP_VECNORM,
    "--hier_target_pos_range", TARGET_RANGE,
    "--hier_early_stop_threshold", "45",
    "--hier_early_stop_consecutive", "2",
    "--first_eval_floor", "45",
    "--decoupling_detection",
    "--checkpoint_freq", "1000",
    "--checkpoint_keep_last", "3",
    "--save_path", SAVE_PATH,
    "--load_model", V59_BEST_HIER_MODEL,
    "--load_vecnorm", V59_BEST_HIER_VECNORM,
    "--no_tensorboard", "--no_domain_randomize",
]


def run_training():
    log("=" * 60)
    log("V63 启动 — v12 reward fix 验证 (gating + early release penalty)")
    log("=" * 60)
    log("起点：V59 best_hier (56% at 2.5k)")
    log("核心改进 (v12 reward fix in panda_vla_env.py):")
    log("  1. 连续奖励门控: 距离/高度/进度/下降奖励仅在持物时生效")
    log("  2. 提前释放惩罚: 释放方块远离目标时 -5 (一次性)")
    log("  3. 一次性奖励 + 成功/释放奖励保持不变 (本不可hack)")
    log("单一变量: V63 = V62 config + reward fix (其余完全相同)")
    log("对比：")
    log("  V60: 外部demo λ=0.3  + old reward → 15% at step3000 (崩溃)")
    log("  V61: 外部demo λ=0.01 + old reward → 25% at step3000 (崩溃)")
    log("  V62: 自模仿  λ=0.1  + old reward → 35% at step3000 (退化)")
    log("  V63: 自模仿  λ=0.1  + v12 reward → 目标 >=50% at step3000 (存活)")
    log("PBRS: scale=0.5, alpha=1.0, beta=0.0 (不变, 已确认非根因)")
    log("训练: 5k 步, hier eval 每1k, 早停 2×<45%, 解耦检测")

    for f in [V59_BEST_HIER_MODEL, V59_BEST_HIER_VECNORM,
              GRASP_STATES_V5_500, GRASP_MODEL, GRASP_VECNORM, D_SUCC_NPZ]:
        if not os.path.exists(f):
            log("ERROR: Missing file: %s" % f)
            return None

    if os.path.exists(SAVE_PATH) and os.listdir(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        log("清除旧 V63 输出")

    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, "train_cmd.txt"), 'w') as f:
        f.write(" ".join(TRAIN_CMD))

    with open(TRAIN_LOG, 'w') as f:
        proc = subprocess.Popen(
            TRAIN_CMD,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=WORKSPACE,
        )
    log("训练 PID: %d" % proc.pid)
    return proc


def parse_hier_evals(log_path):
    if not os.path.exists(log_path):
        return []
    evals = []
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(
                r'\[HIER_EVAL\] step=(\d+) place_rate=(\d+)%.*best=(-?\d+)% at (\d+)',
                line)
            if m:
                evals.append({
                    'step': int(m.group(1)),
                    'place_rate': int(m.group(2)),
                    'best_rate': max(0, int(m.group(3))),
                    'best_step': int(m.group(4)),
                })
    return evals


def check_train_progress():
    if not os.path.exists(TRAIN_LOG):
        return 0
    with open(TRAIN_LOG, 'r') as f:
        text = f.read()
    matches = re.findall(r'total_timesteps\s*\|\s*(\d+)\s*\|', text)
    return int(matches[-1]) if matches else 0


def check_train_done():
    if not os.path.exists(TRAIN_LOG):
        return False
    with open(TRAIN_LOG, 'r') as f:
        text = f.read()
    if "Training complete" in text:
        return True
    if "EARLY STOP triggered" in text:
        return True
    if "FIRST EVAL FLOOR" in text:
        return True
    if "DECOUPLING DETECTED" in text:
        return True
    try:
        result = subprocess.run(["pgrep", "-f", "train_place_policy.py"],
                              capture_output=True, text=True)
        return result.returncode != 0
    except Exception:
        return False


def get_stop_reason():
    if not os.path.exists(TRAIN_LOG):
        return "unknown"
    with open(TRAIN_LOG, 'r') as f:
        text = f.read()
    if "Training complete" in text:
        return "completed (5k steps)"
    if "DECOUPLING DETECTED" in text:
        return "DECOUPLING DETECTED"
    if "FIRST EVAL FLOOR" in text:
        return "FIRST EVAL FLOOR (<45%)"
    if "EARLY STOP triggered" in text:
        return "EARLY STOP (2x<45%)"
    return "process exited"


def monitor_training(train_proc):
    last_report_step = 0
    while True:
        time.sleep(90)
        step = check_train_progress()
        done = check_train_done()

        if step > last_report_step + 1000 or done:
            evals = parse_hier_evals(TRAIN_LOG)
            bc_match = re.findall(r'bc_loss\s*\|\s*([\d.]+)', open(TRAIN_LOG).read())
            bc_info = ""
            if bc_match:
                bc_info = ", bc_loss=%s" % bc_match[-1]
            if evals:
                latest = evals[-1]
                log("进度: step=%dk, hier place_rate=%d%%, best=%d%% at %dk%s" % (
                    step // 1000, latest['place_rate'],
                    latest['best_rate'], latest['best_step'] // 1000, bc_info))
                # Highlight step 3000 (critical validation point)
                if latest['step'] >= 3000 and last_report_step < 3000:
                    if latest['place_rate'] >= 50:
                        log("*** 关键验证: step3000 place_rate=%d%% >= 50%% — reward fix 确认有效! ***" % latest['place_rate'])
                    else:
                        log("*** 关键验证: step3000 place_rate=%d%% < 50%% — reward fix 未完全消除解耦 ***" % latest['place_rate'])
            else:
                log("进度: step=%dk (等待首次 hier eval)%s" % (step // 1000, bc_info))
            last_report_step = step

        if done:
            reason = get_stop_reason()
            log("训练结束: %s" % reason)
            break

        if train_proc.poll() is not None:
            log("训练进程退出, code=%d" % train_proc.returncode)
            break


def run_final_eval(model_path, vecnorm_path, episodes=50):
    eval_log = os.path.join(SAVE_PATH, "final_eval.log")
    cmd = [
        PYTHON, "-u", "eval_hierarchical.py",
        "--place_model", model_path,
        "--place_vecnorm", vecnorm_path,
        "--grasp_model", GRASP_MODEL,
        "--grasp_vecnorm", GRASP_VECNORM,
        "--vision_mode", "--no_domain_randomize",
        "--target_pos_range", TARGET_RANGE,
        "--n_episodes", str(episodes),
    ]
    log("最终评估 (best_hier model, %d episodes)..." % episodes)
    with open(eval_log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=WORKSPACE)

    with open(eval_log, 'r') as f:
        text = f.read()
    place_match = re.search(r'Place \(dist<\d+cm\)\s*:\s*\d+/(\d+)\s*\((\d+)%\)', text)
    lift_match = re.search(r'Mean max lift\s*:\s*([\d.]+)\s*cm', text)
    dist_match = re.search(r'Mean final dist\s*:\s*([\d.]+)\s*cm', text)
    if place_match:
        result = {
            'place_rate': int(place_match.group(2)),
            'mean_lift': float(lift_match.group(1)) if lift_match else 0,
            'mean_dist': float(dist_match.group(1)) if dist_match else 0,
        }
        log("  最终结果: place=%d%%, lift=%.1fcm, dist=%.1fcm" % (
            result['place_rate'], result['mean_lift'], result['mean_dist']))
        return result
    log("  最终评估解析失败，查看 %s" % eval_log)
    return None


def main():
    for f in [TRAIN_LOG, PIPELINE_LOG]:
        if os.path.exists(f):
            os.remove(f)

    # Step 1: Ensure D_succ.npz exists
    log("=" * 60)
    log("Step 1: 确认 D_succ.npz (复用 V62 的自模仿缓冲)")
    log("=" * 60)
    if not convert_pkl_to_npz():
        log("D_succ.npz 准备失败，退出。")
        return

    # Step 2: Run training
    log("")
    log("=" * 60)
    log("Step 2: V63 训练 (v12 reward fix + OPR)")
    log("=" * 60)
    train_proc = run_training()
    if train_proc:
        monitor_training(train_proc)

    # Step 3: Summary + final eval
    log("")
    log("=" * 60)
    log("V63 Pipeline 总结")
    log("=" * 60)

    reason = get_stop_reason()
    log("训练终止原因: %s" % reason)

    evals = parse_hier_evals(TRAIN_LOG)
    log("Hier eval 记录: %d 次" % len(evals))
    for e in evals:
        log("  %dk: place_rate=%d%%, best=%d%% at %dk" % (
            e['step'] // 1000, e['place_rate'],
            e['best_rate'], e['best_step'] // 1000))

    with open(TRAIN_LOG, 'r') as f:
        train_text = f.read()
    bc_losses = re.findall(r'bc_loss\s*\|\s*([\d.]+)', train_text)
    if bc_losses:
        log("BC loss 记录: %d 次, 首次=%s, 末次=%s" % (
            len(bc_losses), bc_losses[0], bc_losses[-1]))

    best_hier_model = os.path.join(SAVE_PATH, "best_hier", "best_model.zip")
    best_hier_vecnorm = os.path.join(SAVE_PATH, "best_hier", "vec_normalize.pkl")

    if os.path.exists(best_hier_model) and os.path.exists(best_hier_vecnorm):
        log("\n--- 最终评估 (best_hier, 50 episodes) ---")
        result = run_final_eval(best_hier_model, best_hier_vecnorm, episodes=50)
        if result:
            log("\n结论：")
            v63 = result['place_rate']
            # Check if step 3000 survived
            step3000_survived = any(e['step'] >= 3000 and e['place_rate'] >= 50 for e in evals)
            if step3000_survived and v63 >= 56:
                log("  V63 成功! place_rate=%d%%, step3000 存活" % v63)
                log("  → v12 reward fix 消除了解耦根因")
                log("  → 策略稳定, 可继续训练或部署")
            elif step3000_survived:
                log("  V63 step3000 存活但 best_hier=%d%% < 56%%" % v63)
                log("  → reward fix 防止了崩溃, 但策略未提升")
                log("  → 考虑增加训练步数或调整 OPR 权重")
            else:
                log("  V63 step3000 仍退化, place_rate=%d%%" % v63)
                log("  → reward fix 未完全消除解耦")
                log("  → 需进一步排查 (可能需要在线 D_succ 扩展)")
    else:
        log("ERROR: best_hier/ model not found")

    log("\n基线对比：")
    log("  V59 best_hier:  56% (无BC, 2500步解耦, old reward)")
    log("  V60 best_hier:  50% (外部BC λ=0.3, step3000崩溃, old reward)")
    log("  V61 best_hier:  56% (外部BC λ=0.01, step3000崩溃, old reward)")
    log("  V62 best_hier:  56% (自模仿 λ=0.1, step3000退化35%%, old reward)")
    log("  V63 best_hier:  %s (自模仿 λ=0.1, v12 reward fix)" % (
        "见上方" if os.path.exists(best_hier_model) else "N/A"))


if __name__ == "__main__":
    main()
