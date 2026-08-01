#!/usr/bin/env python3
"""V67 pipeline — Reward structure overhaul (v16).

Root cause recap (V59-V66):
  V66 (v15: gate lowering bonus on xy_dist<0.10) CRASHED to 5% at step 3000 —
  IDENTICAL to V64 (k=0.02) and V65 (k=1.0). v15 gating did NOT fix the
  problem. The lowering bonus Z-descent bias was NOT the root cause (or not
  the only cause).

  V66 produced the KEY DIAGNOSTIC EVIDENCE:
    [HIER_EVAL] DECOUPLING DETECTED: place_mode reward 9.3 -> 65.3
                 but hier place_rate 50% -> 5%
  This is textbook reward hacking: PPO optimizes place_mode reward (7x
  improvement) while the true hierarchical place_rate crashes. The policy
  "farms" hackable rewards without completing the task.

  ROOT CAUSE IDENTIFIED — the reward STRUCTURE, not the coefficient:
    One-time proximity bonuses are collectible WITHOUT actual placement:
      +20  at dist<0.15  (proximity_15)
      +50  at dist<0.10  (proximity_10)
      +100 at dist<0.05  (approach)
      ─────────────────────
      +170 total         (all collectible by approaching then hovering)

    At k<=1.0 (V64/V65/V66), the policy earns +170 by reaching proximity
    thresholds, then hovers — never releasing. The holding penalty is too
    weak to force actual placement. Only at k=10 (V63, 40%) did the strong
    penalty (-1.5/step at 15cm) create enough urgency to partially override
    this farming behavior.

V67 core changes (v16, in gym_env/panda_vla_env.py):
  1. k=5.0 distance penalty (was 1.0 in V65/V66).
     At k<=1.0, per-step positive rewards (+0.25) exceed the penalty.
     k=5.0 is the minimum where penalty dominates down to 5cm:
       at 15cm: -0.75/step, at 5cm: -0.25/step.
  2. REMOVE one-time proximity bonuses (+20 at 15cm, +50 at 10cm).
     These were hackable — collectible without placement.
  3. REMOVE one-time approach bonus (+100 at 5cm).
     Same hack — collectible by reaching 5cm without releasing.
  4. ADD progressive proximity reward: 0.1 / (1.0 + block_target_dist) per step.
     Smooth, no discrete thresholds to exploit. Coefficient 0.1 (not 1.0)
     keeps total ~50 (25% of +200 success) so it doesn't dominate the k=5.0
     penalty. Net per-step reward is NEGATIVE at all distances (urgency to
     place remains): at 15cm -0.66/step, at 5cm -0.16/step.

  Correctly gated rewards (UNCHANGED — were never hackable):
    +50  release: requires dist<0.05 AND on_table AND gripper_open
    +200 success: requires dist<0.05 AND on_table AND gripper_open
    -5   early release: one-time, when releasing away from target

  Retained from V66:
    - v15 lowering bonus gate (xy_dist<0.10) — auxiliary constraint, no cost
    - v12 gating (all continuous rewards on is_holding)
    - OPR self-imitation (lambda_si=0.1, D_succ.npz)
    - DECOUPLING DETECTION early stop (validated in V66, saves compute)
    - All hyperparameters unchanged

Single-variable isolation:
  V67 = V66 config + v16 reward restructure (k=5.0, remove +20/+50/+100,
  add progressive 0.1/(1.0+dist)). ONLY the reward function changed.

Critical validation: step 3000 eval
  V63 (OPR + v12, k=10):              40% at step 3000 (degraded, +5%)
  V64 (OPR + v12, k=0.02):             5% at step 3000 (crashed, no gradient)
  V65 (OPR + v12, k=1.0):              5% at step 3000 (crashed, Z-bias)
  V66 (OPR + v12, k=1.0 + v15 gate):   5% at step 3000 (crashed, structure)
  V67 (OPR + v12, k=5.0 + v16 restructure): target > 45% at step 3000

  Health checks: bc_loss stable (~0.002), approx_kl stable (<0.01),
  explained_variance > 0.9.

Pipeline:
  1. Confirm D_succ.npz (reuse V62-V66 buffer)
  2. Train from V59 best_hier with v16 reward + OPR (lambda_si=0.1)
  3. 5000 steps, hier eval every 1000, early stop + decoupling detection
  4. Final 50-ep eval of best_hier

Usage:
    python run_v67_pipeline.py
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
SAVE_PATH = os.path.join(WORKSPACE, "outputs/place_policy_v67")
TRAIN_LOG = os.path.join(WORKSPACE, "outputs/v67_train.log")
PIPELINE_LOG = os.path.join(WORKSPACE, "outputs/v67_pipeline.log")

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
    # V67: OPR self-imitation buffer (single-variable: only reward function changed)
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
    log("V67 启动 — v16 奖励结构重构 (k=5.0 + 移除分档bonus + 渐进奖励)")
    log("=" * 60)
    log("起点：V59 best_hier (56% at 2.5k)")
    log("根因诊断 (V66 DECOUPLING 证据):")
    log("  reward 9.3→65.3 (优化中) 但 place_rate 50%%→5%% (崩溃中)")
    log("  = 教科书式奖励黑客: 策略在'骗'奖励函数而非完成任务")
    log("  根因: 分档 proximity bonus (+20/+50/+100=+170) 可在不放置时收集")
    log("  v15 门控只把'任意位置下降'hack 改成'目标上方悬停'hack — 未解决结构问题")
    log("核心改进 (v16 in panda_vla_env.py):")
    log("  1. k=5.0 距离惩罚 (was 1.0) — 最小值使惩罚主导正奖励")
    log("  2. 移除 +20 (dist<0.15) 和 +50 (dist<0.10) 一次性 proximity bonus")
    log("  3. 移除 +100 (dist<0.05) 一次性 approach bonus")
    log("  4. 添加渐进奖励 0.1/(1.0+dist) per step — 无离散阈值可exploit")
    log("  系数 0.1 (非1.0): 总计~50 (25% of +200), 不主导 k=5.0 惩罚")
    log("  净 per-step: 15cm=-0.66, 5cm=-0.16 (负=保持放置紧迫感)")
    log("  保留: v15门控 + v12 gating + early release + OPR + 全部超参")
    log("  保留: DECOUPLING DETECTION 早停 (V66验证有效, 节省算力)")
    log("单一变量: V67 = V66 config + v16 奖励重构 (仅奖励函数改变)")
    log("对比：")
    log("  V63: 自模仿 λ=0.1 + v12 k=10            → 40% at step3000 (退化+5%)")
    log("  V64: 自模仿 λ=0.1 + v12 k=0.02          →  5% at step3000 (梯度消失)")
    log("  V65: 自模仿 λ=0.1 + v12 k=1.0           →  5% at step3000 (Z-下降偏置)")
    log("  V66: 自模仿 λ=0.1 + v12 k=1.0 + v15门控 →  5% at step3000 (结构问题)")
    log("  V67: 自模仿 λ=0.1 + v12 k=5.0 + v16重构 → 目标 >45% at step3000")
    log("PBRS: scale=0.5, alpha=1.0, beta=0.0 (不变, 已确认非根因)")
    log("训练: 5k 步, hier eval 每1k, 早停 2×<45%, 解耦检测")

    for f in [V59_BEST_HIER_MODEL, V59_BEST_HIER_VECNORM,
              GRASP_STATES_V5_500, GRASP_MODEL, GRASP_VECNORM, D_SUCC_NPZ]:
        if not os.path.exists(f):
            log("ERROR: Missing file: %s" % f)
            return None

    if os.path.exists(SAVE_PATH) and os.listdir(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
        log("清除旧 V67 输出")

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
                        log("*** 关键验证: step3000 place_rate=%d%% >= 50%% — v16 结构重构确认有效! ***" % latest['place_rate'])
                    elif latest['place_rate'] > 45:
                        log("*** 关键验证: step3000 place_rate=%d%% > 45%% — v16 改善 (vs V64/V65/V66=5%%) ***" % latest['place_rate'])
                    else:
                        log("*** 关键验证: step3000 place_rate=%d%% <= 45%% — v16 未达预期 ***" % latest['place_rate'])
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
    log("Step 1: 确认 D_succ.npz (复用 V62-V66 的自模仿缓冲)")
    log("=" * 60)
    if not convert_pkl_to_npz():
        log("D_succ.npz 准备失败，退出。")
        return

    # Step 2: Run training
    log("")
    log("=" * 60)
    log("Step 2: V67 训练 (v16 奖励重构 + k=5.0 + v12 gating + OPR)")
    log("=" * 60)
    train_proc = run_training()
    if train_proc:
        monitor_training(train_proc)

    # Step 3: Summary + final eval
    log("")
    log("=" * 60)
    log("V67 Pipeline 总结")
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
    kl_matches = re.findall(r'approx_kl\s*\|\s*([\d.]+)', train_text)
    if kl_matches:
        log("approx_kl: 首次=%s, 末次=%s" % (kl_matches[0], kl_matches[-1]))
    ev_matches = re.findall(r'explained_variance\s*\|\s*([\d.]+)', train_text)
    if ev_matches:
        log("explained_variance: 首次=%s, 末次=%s" % (ev_matches[0], ev_matches[-1]))

    best_hier_model = os.path.join(SAVE_PATH, "best_hier", "best_model.zip")
    best_hier_vecnorm = os.path.join(SAVE_PATH, "best_hier", "vec_normalize.pkl")

    if os.path.exists(best_hier_model) and os.path.exists(best_hier_vecnorm):
        log("\n--- 最终评估 (best_hier, 50 episodes) ---")
        result = run_final_eval(best_hier_model, best_hier_vecnorm, episodes=50)
        if result:
            log("\n结论：")
            v67 = result['place_rate']
            step3000_rate = next((e['place_rate'] for e in evals if e['step'] >= 3000), 0)
            if step3000_rate > 45 and v67 >= 56:
                log("  V67 成功! step3000=%d%% (>45%%), best_hier=%d%%" % (step3000_rate, v67))
                log("  → v16 奖励结构重构消除了解耦根因 (分档bonus hackable)")
                log("  → 策略稳定, 可继续训练或部署")
            elif step3000_rate > 45:
                log("  V67 step3000=%d%% (>45%%, vs V64/V65/V66=5%%) 但 best_hier=%d%%" % (step3000_rate, v67))
                log("  → v16 改善了解耦, 但策略未提升")
                log("  → 考虑增加训练步数或调整 OPR 权重")
            else:
                log("  V67 step3000=%d%% (<=45%%), best_hier=%d%%" % (step3000_rate, v67))
                log("  → v16 未达预期, 可能需要 k=10.0 (V63值) 或进一步重构")
                log("  → 考虑: k=10.0 + v16, 或 在线 D_succ 扩展, 或 value网络预训练")
    else:
        log("ERROR: best_hier/ model not found")

    log("\n基线对比：")
    log("  V59 best_hier:  56% (无BC, 2500步解耦, old reward)")
    log("  V62 best_hier:  56% (自模仿 λ=0.1, step3000=35%%, old reward)")
    log("  V63 best_hier:  50% (自模仿 λ=0.1, step3000=40%%, v12 k=10)")
    log("  V64 best_hier:  56% (自模仿 λ=0.1, step3000=5%%, v12 k=0.02 梯度消失)")
    log("  V65 best_hier:  56% (自模仿 λ=0.1, step3000=5%%, v12 k=1.0 Z-下降偏置)")
    log("  V66 best_hier:  56% (自模仿 λ=0.1, step3000=5%%, v12 k=1.0+v15门控 结构问题)")
    log("  V67 best_hier:  %s (自模仿 λ=0.1, v12 k=5.0+v16重构 移除分档bonus)" % (
        "见上方" if os.path.exists(best_hier_model) else "N/A"))


if __name__ == "__main__":
    main()
