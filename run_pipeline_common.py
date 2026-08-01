#!/usr/bin/env python3
"""Common pipeline runner for V69/V70/V71a/V71b experiments.

Extracted from run_v68_pipeline.py to avoid code duplication. Each
experiment script defines its config and calls run_pipeline().

Single-variable isolation principle: each experiment changes exactly
ONE variable from the V68 baseline config. The common module ensures
all other parameters are identical across experiments.
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

V59_BEST_HIER_MODEL = os.path.join(WORKSPACE, "outputs/place_policy_v59/best_hier/best_model.zip")
V59_BEST_HIER_VECNORM = os.path.join(WORKSPACE, "outputs/place_policy_v59/best_hier/vec_normalize.pkl")

GRASP_STATES_V5_500 = os.path.join(WORKSPACE, "outputs/grasp_states_v5_500.pkl")
GRASP_MODEL = os.path.join(WORKSPACE, "outputs/dapg_800k_v5/best/best_model.zip")
GRASP_VECNORM = os.path.join(WORKSPACE, "outputs/dapg_800k_v5/vec_normalize.pkl")
TARGET_RANGE = "0.35,0.15,0.22,0.65,0.45,0.22"

D_SUCC_PKL = os.path.join(WORKSPACE, "data/D_succ.pkl")
D_SUCC_NPZ = os.path.join(WORKSPACE, "data/D_succ.npz")


def log(msg, pipeline_log):
    timestamp = time.strftime("%H:%M:%S")
    line = "[%s] %s" % (timestamp, msg)
    print(line, flush=True)
    with open(pipeline_log, 'a') as f:
        f.write(line + "\n")


def convert_pkl_to_npz(pipeline_log):
    if not os.path.exists(D_SUCC_NPZ):
        if not os.path.exists(D_SUCC_PKL):
            log("ERROR: D_succ.pkl not found at %s" % D_SUCC_PKL, pipeline_log)
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
        log("Converted D_succ.pkl -> D_succ.npz: %d transitions" % len(actions), pipeline_log)
    else:
        data = np.load(D_SUCC_NPZ)
        log("D_succ.npz exists: %d transitions" % len(data["actions"]), pipeline_log)
    return True


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


def check_train_progress(log_path):
    if not os.path.exists(log_path):
        return 0
    with open(log_path, 'r') as f:
        text = f.read()
    matches = re.findall(r'total_timesteps\s*\|\s*(\d+)\s*\|', text)
    return int(matches[-1]) if matches else 0


def check_train_done(log_path):
    if not os.path.exists(log_path):
        return False
    with open(log_path, 'r') as f:
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


def get_stop_reason(log_path):
    if not os.path.exists(log_path):
        return "unknown"
    with open(log_path, 'r') as f:
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


def monitor_training(train_proc, train_log, pipeline_log):
    last_report_step = 0
    while True:
        time.sleep(90)
        step = check_train_progress(train_log)
        done = check_train_done(train_log)

        if step > last_report_step + 1000 or done:
            evals = parse_hier_evals(train_log)
            bc_match = re.findall(r'bc_loss\s*\|\s*([\d.]+)', open(train_log).read())
            bc_info = ""
            if bc_match:
                bc_info = ", bc_loss=%s" % bc_match[-1]
            if evals:
                latest = evals[-1]
                log("进度: step=%dk, hier place_rate=%d%%, best=%d%% at %dk%s" % (
                    step // 1000, latest['place_rate'],
                    latest['best_rate'], latest['best_step'] // 1000, bc_info), pipeline_log)
                if latest['step'] >= 3000 and last_report_step < 3000:
                    if latest['place_rate'] >= 50:
                        log("*** 关键验证: step3000 place_rate=%d%% >= 50%% — 单变量改动有效! ***" % latest['place_rate'], pipeline_log)
                    elif latest['place_rate'] > 45:
                        log("*** 关键验证: step3000 place_rate=%d%% > 45%% — 改善 (vs V68=5%%) ***" % latest['place_rate'], pipeline_log)
                    else:
                        log("*** 关键验证: step3000 place_rate=%d%% <= 45%% — 单变量改动无效, 与V68相同模式 ***" % latest['place_rate'], pipeline_log)
            else:
                log("进度: step=%dk (等待首次 hier eval)%s" % (step // 1000, bc_info), pipeline_log)
            last_report_step = step

        if done:
            reason = get_stop_reason(train_log)
            log("训练结束: %s" % reason, pipeline_log)
            break

        if train_proc.poll() is not None:
            log("训练进程退出, code=%d" % train_proc.returncode, pipeline_log)
            break


def run_final_eval(model_path, vecnorm_path, save_path, pipeline_log, episodes=50):
    eval_log = os.path.join(save_path, "final_eval.log")
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
    log("最终评估 (best_hier model, %d episodes)..." % episodes, pipeline_log)
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
            result['place_rate'], result['mean_lift'], result['mean_dist']), pipeline_log)
        return result
    log("  最终评估解析失败，查看 %s" % eval_log, pipeline_log)
    return None


def run_pipeline(version, description, train_cmd, save_path,
                 single_variable_change, baseline_note):
    """Run a complete training pipeline for a single-variable experiment.

    Args:
        version: e.g. "V69"
        description: brief description of the single-variable change
        train_cmd: list of strings, the full training command
        save_path: output directory for this experiment
        single_variable_change: human-readable description of what changed
        baseline_note: comparison note for logging
    """
    train_log = os.path.join(WORKSPACE, "outputs/%s_train.log" % version.lower())
    pipeline_log = os.path.join(WORKSPACE, "outputs/%s_pipeline.log" % version.lower())

    for f in [train_log, pipeline_log]:
        if os.path.exists(f):
            os.remove(f)

    # Step 1: Ensure D_succ.npz exists
    log("=" * 60, pipeline_log)
    log("Step 1: 确认 D_succ.npz", pipeline_log)
    log("=" * 60, pipeline_log)
    if not convert_pkl_to_npz(pipeline_log):
        log("D_succ.npz 准备失败，退出。", pipeline_log)
        return

    # Step 2: Run training
    log("", pipeline_log)
    log("=" * 60, pipeline_log)
    log("Step 2: %s 训练 (%s)" % (version, description), pipeline_log)
    log("=" * 60, pipeline_log)
    log("起点：V59 best_hier (56%)", pipeline_log)
    log("单变量改动: %s" % single_variable_change, pipeline_log)
    log("基线对比: V68 step3000=5% (PPO 第一次更新即崩溃)", pipeline_log)
    log("诊断结论: V70(freeze backbone)为第一优先", pipeline_log)
    if baseline_note:
        log(baseline_note, pipeline_log)

    for f in [V59_BEST_HIER_MODEL, V59_BEST_HIER_VECNORM,
              GRASP_STATES_V5_500, GRASP_MODEL, GRASP_VECNORM, D_SUCC_NPZ]:
        if not os.path.exists(f):
            log("ERROR: Missing file: %s" % f, pipeline_log)
            return

    if os.path.exists(save_path) and os.listdir(save_path):
        shutil.rmtree(save_path)
        log("清除旧 %s 输出" % version, pipeline_log)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "train_cmd.txt"), 'w') as f:
        f.write(" ".join(train_cmd))

    with open(train_log, 'w') as f:
        proc = subprocess.Popen(
            train_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=WORKSPACE,
        )
    log("训练 PID: %d" % proc.pid, pipeline_log)
    monitor_training(proc, train_log, pipeline_log)

    # Step 3: Summary + final eval
    log("", pipeline_log)
    log("=" * 60, pipeline_log)
    log("%s Pipeline 总结" % version, pipeline_log)
    log("=" * 60, pipeline_log)

    reason = get_stop_reason(train_log)
    log("训练终止原因: %s" % reason, pipeline_log)

    evals = parse_hier_evals(train_log)
    log("Hier eval 记录: %d 次" % len(evals), pipeline_log)
    for e in evals:
        log("  %dk: place_rate=%d%%, best=%d%% at %dk" % (
            e['step'] // 1000, e['place_rate'],
            e['best_rate'], e['best_step'] // 1000), pipeline_log)

    with open(train_log, 'r') as f:
        train_text = f.read()
    bc_losses = re.findall(r'bc_loss\s*\|\s*([\d.]+)', train_text)
    if bc_losses:
        log("BC loss 记录: %d 次, 首次=%s, 末次=%s" % (
            len(bc_losses), bc_losses[0], bc_losses[-1]), pipeline_log)
    kl_matches = re.findall(r'approx_kl\s*\|\s*([\d.]+)', train_text)
    if kl_matches:
        log("approx_kl: 首次=%s, 末次=%s" % (kl_matches[0], kl_matches[-1]), pipeline_log)
    ev_matches = re.findall(r'explained_variance\s*\|\s*([\d.]+)', train_text)
    if ev_matches:
        log("explained_variance: 首次=%s, 末次=%s" % (ev_matches[0], ev_matches[-1]), pipeline_log)

    best_hier_model = os.path.join(save_path, "best_hier", "best_model.zip")
    best_hier_vecnorm = os.path.join(save_path, "best_hier", "vec_normalize.pkl")

    if os.path.exists(best_hier_model) and os.path.exists(best_hier_vecnorm):
        log("\n--- 最终评估 (best_hier, 50 episodes) ---", pipeline_log)
        result = run_final_eval(best_hier_model, best_hier_vecnorm, save_path, pipeline_log, episodes=50)
        if result:
            log("\n结论：", pipeline_log)
            step3000_rate = next((e['place_rate'] for e in evals if e['step'] >= 3000), 0)
            if step3000_rate > 45 and result['place_rate'] >= 56:
                log("  %s 成功! step3000=%d%% (>45%%), best_hier=%d%%" % (
                    version, step3000_rate, result['place_rate']), pipeline_log)
                log("  → 单变量改动有效: %s" % single_variable_change, pipeline_log)
            elif step3000_rate > 45:
                log("  %s step3000=%d%% (>45%%, vs V68=5%%) 但 best_hier=%d%%" % (
                    version, step3000_rate, result['place_rate']), pipeline_log)
                log("  → 改善了崩溃, 但策略未提升", pipeline_log)
            else:
                log("  %s step3000=%d%% (<=45%%), best_hier=%d%%" % (
                    version, step3000_rate, result['place_rate']), pipeline_log)
                log("  → 单变量改动无效, 与V68相同崩溃模式", pipeline_log)
                log("  → %s 不是根因" % single_variable_change, pipeline_log)
    else:
        log("ERROR: best_hier/ model not found", pipeline_log)

    log("\n基线对比：", pipeline_log)
    log("  V59 best_hier:  56% (基线)", pipeline_log)
    log("  V68 step3000:    5% (PPO第一次更新即崩溃)", pipeline_log)
    log("  %s step3000:  %s" % (version,
        "%d%%" % next((e['place_rate'] for e in evals if e['step'] >= 3000), -1)
        if evals else "N/A"), pipeline_log)


def make_v68_base_cmd(save_path):
    """Generate the V68 baseline training command.

    V69/V70/V71a/V71b each modify exactly ONE parameter from this baseline.
    This function ensures all other parameters are identical across experiments.
    """
    return [
        PYTHON, "-u", "train_place_policy.py",
        "--vision_mode", "--pretrained_cnn", "--image_augment", "--freeze_bn",
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
        "--save_path", save_path,
        "--load_model", V59_BEST_HIER_MODEL,
        "--load_vecnorm", V59_BEST_HIER_VECNORM,
        "--no_tensorboard", "--no_domain_randomize",
    ]
