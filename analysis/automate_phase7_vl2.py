#!/usr/bin/env python3
"""automate_phase7_vl2.py — Phase 7 Round 2b V L2 正则化自动化脚本.

功能:
  1. 自动备份 iql_agent.py, 修改第 251 行 V L2 正则项系数
  2. 在 tmux 会话中启动训练 (validate=5seeds / full=30seeds)
  3. 训练完成后自动提取指标, 对比 baseline, 输出决策建议
  4. 支持 --rollback 一键回滚, --status 查看进度

用法:
    # 验证模式 (5 seeds, ~1h)
    python automate_phase7_vl2.py --mode validate --coeff 3e-6 --base_seed 200

    # 全量模式 (30 seeds, ~6h)
    python automate_phase7_vl2.py --mode full --coeff 1e-6

    # 查看状态
    python automate_phase7_vl2.py --status

    # 分析已完成实验
    python automate_phase7_vl2.py --analyze --coeff 3e-6

    # 回滚代码
    python automate_phase7_vl2.py --rollback
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

WORKSPACE = Path("/home/w/vla_workspace")
IQL_AGENT_PATH = WORKSPACE / "iql_agent.py"
PHASE7_SCRIPT = WORKSPACE / "phase7_train_cv.py"
BACKUP_PREFIX = "iql_agent.py.bak_"
CONDA_ACTIVATE = "source /home/w/miniconda3/etc/profile.d/conda.sh && conda activate vla"

# Baseline: τ=0.5 无正则, 30 seeds
BASELINE = {
    "config": "tau0.5_no_reg",
    "n_seeds": 30,
    "place_rate_cv": 6.78,
    "v_mean_cv": 30.5,
    "place_rate_mean": 60.0,
}

# 决策阈值
V_CV_TARGET_MIN = 20.0
V_CV_TARGET_MAX = 25.0
PLACE_RATE_CV_THRESHOLD = 8.0
PLACE_RATE_MEAN_THRESHOLD = 59.0

# 原始 v_loss 行 (无正则)
ORIGINAL_V_LOSS = "        v_loss = self._expectile_loss(q_min - v, self.tau)"

# 匹配 v_loss 行 (含可选正则项)
V_LOSS_REGEX = re.compile(
    r'(        v_loss = self\._expectile_loss\(q_min - v, self\.tau\))'
    r'( \+ [\d.eE\-]+ \* torch\.norm\(torch\.cat\(\[p\.flatten\(\) for p in self\.v_net\.parameters\(\)\]\), p=2\))?'
)


# ---------------------------------------------------------------------------
# iql_agent.py 修改与备份
# ---------------------------------------------------------------------------

def backup_iql_agent():
    """备份 iql_agent.py, 文件名带时间戳. 返回备份路径."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = WORKSPACE / f"{BACKUP_PREFIX}{ts}"
    shutil.copy2(IQL_AGENT_PATH, backup)
    return backup


def get_current_coeff():
    """检测当前 iql_agent.py 中的 V L2 系数.
    返回 (coeff_str, is_regularized). None 表示无法识别.
    """
    content = IQL_AGENT_PATH.read_text()
    m = V_LOSS_REGEX.search(content)
    if not m:
        return None, None  # 找不到 v_loss 行
    if m.group(2):  # 有正则项
        cm = re.search(r'\+ ([\d.eE\-]+) \* torch\.norm', m.group(2))
        if cm:
            return cm.group(1), True
    return None, False


def modify_iql_agent(coeff):
    """修改 iql_agent.py 第 251 行, 设置 V L2 系数为 coeff.
    返回 (success, old_coeff, backup_path, error).
    """
    old_coeff, was_reg = get_current_coeff()
    backup_path = backup_iql_agent()

    content = IQL_AGENT_PATH.read_text()

    if coeff > 0:
        new_line = (
            "        v_loss = self._expectile_loss(q_min - v, self.tau)"
            f" + {coeff} * torch.norm(torch.cat([p.flatten() for p in self.v_net.parameters()]), p=2)"
        )
    else:
        new_line = ORIGINAL_V_LOSS

    new_content, n = V_LOSS_REGEX.subn(new_line, content)
    if n == 0:
        return False, old_coeff, backup_path, "未找到 v_loss 行"
    if n > 1:
        return False, old_coeff, backup_path, f"匹配到 {n} 处 (应为 1)"

    IQL_AGENT_PATH.write_text(new_content)
    return True, old_coeff, backup_path, None


def rollback_iql_agent():
    """从最近的备份回滚 iql_agent.py.
    返回 (success, backup_path, error).
    """
    backups = sorted(WORKSPACE.glob(f"{BACKUP_PREFIX}*"))
    if not backups:
        return False, None, "无备份可用"
    backup = backups[-1]
    shutil.copy2(backup, IQL_AGENT_PATH)
    return True, backup, None


# ---------------------------------------------------------------------------
# tmux 会话管理
# ---------------------------------------------------------------------------

def coeff_to_tag(coeff):
    """将 coeff 转为路径/tmux 安全的标签. 3e-6 -> '3e-6'."""
    return f"{coeff:g}"


def tmux_session_name(coeff, n_seeds):
    """生成 tmux 会话名: phase7_vl2_3e-6_5."""
    return f"phase7_vl2_{coeff_to_tag(coeff)}_{n_seeds}"


def output_dir_for(coeff, mode="validate"):
    """生成输出目录路径."""
    suffix = "_full" if mode == "full" else ""
    return WORKSPACE / f"outputs/phase7_round2b_tau0.5_vl2_{coeff_to_tag(coeff)}{suffix}"


def create_tmux_session(session_name, command):
    """创建 tmux 会话并运行命令. 返回 (success, error)."""
    subprocess.run(["tmux", "kill-session", "-t", session_name],
                   capture_output=True)
    full = f"{CONDA_ACTIVATE} && cd {WORKSPACE} && {command}"
    r = subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name,
         "bash", "-c", full],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return False, r.stderr
    return True, None


def check_tmux_alive(session_name):
    """检查 tmux 会话是否存活."""
    r = subprocess.run(["tmux", "has-session", "-t", session_name],
                       capture_output=True)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# 指标提取与决策
# ---------------------------------------------------------------------------

def extract_metrics(output_dir):
    """从 training_cv_analysis.json 提取指标. 返回 (metrics, error)."""
    output_dir = Path(output_dir)
    path = output_dir / "training_cv_analysis.json"
    if not path.exists():
        return None, f"未找到分析文件: {path}"
    try:
        data = json.load(open(path))
        return data, None
    except Exception as e:
        return None, str(e)


def compare_to_baseline(metrics, coeff):
    """对比 baseline, 生成决策建议. 返回 summary dict."""
    pr_cv = metrics.get("place_rate_cv", 999)
    pr_mean = metrics.get("place_rate_mean", 0)
    qv_cv = metrics.get("qv_cv", {})
    v_cv = qv_cv.get("v_mean", 999)
    n = len(metrics.get("place_rates", []))

    b = BASELINE
    summary = {
        "coeff": coeff,
        "n_seeds": n,
        "metrics": {
            "place_rate_mean": pr_mean,
            "place_rate_cv": pr_cv,
            "v_mean_cv": v_cv,
        },
        "baseline": {
            "place_rate_mean": b["place_rate_mean"],
            "place_rate_cv": b["place_rate_cv"],
            "v_mean_cv": b["v_mean_cv"],
            "n_seeds": b["n_seeds"],
        },
        "deltas": {
            "place_rate_mean_delta": round(pr_mean - b["place_rate_mean"], 2),
            "place_rate_cv_delta": round(pr_cv - b["place_rate_cv"], 2),
            "v_mean_cv_delta": round(v_cv - b["v_mean_cv"], 2),
        },
    }

    # 三项检查
    reasons = []
    passes = 0

    # 1. V_CV 在 [20%, 25%]
    v_cv_delta = v_cv - b["v_mean_cv"]  # 正: V_CV 变差, 负: V_CV 改善
    if V_CV_TARGET_MIN <= v_cv <= V_CV_TARGET_MAX:
        reasons.append(f"✓ V_CV={v_cv:.1f}% 在目标区间 [{V_CV_TARGET_MIN}%, {V_CV_TARGET_MAX}%] "
                       f"(baseline={b['v_mean_cv']:.1f}%, delta={v_cv_delta:+.1f}pp)")
        passes += 1
    elif v_cv < V_CV_TARGET_MIN:
        reasons.append(f"✗ V_CV={v_cv:.1f}% < {V_CV_TARGET_MIN}% (正则过强, V 被过度压制, "
                       f"delta={v_cv_delta:+.1f}pp)")
    elif v_cv > b["v_mean_cv"]:
        # V_CV 比基线还差 — 正则适得其反
        reasons.append(f"✗ V_CV={v_cv:.1f}% > baseline {b['v_mean_cv']:.1f}% (正则适得其反, "
                       f"V_CV 反而上升 {v_cv_delta:+.1f}pp, 建议调小系数)")
    else:
        # V_CV 降了但仍 > 25% — 方向对但不够
        reasons.append(f"△ V_CV={v_cv:.1f}% > {V_CV_TARGET_MAX}% 但已降 {v_cv_delta:+.1f}pp "
                       f"(方向对, 可尝试略大系数)")

    # 2. place_rate_cv ≤ 8%
    if pr_cv <= PLACE_RATE_CV_THRESHOLD:
        reasons.append(f"✓ place_rate_CV={pr_cv:.2f}% ≤ {PLACE_RATE_CV_THRESHOLD}%")
        passes += 1
    else:
        reasons.append(f"✗ place_rate_CV={pr_cv:.2f}% > {PLACE_RATE_CV_THRESHOLD}%")

    # 3. place_rate_mean ≥ 59%
    if pr_mean >= PLACE_RATE_MEAN_THRESHOLD:
        reasons.append(f"✓ place_rate_mean={pr_mean:.2f}% ≥ {PLACE_RATE_MEAN_THRESHOLD}%")
        passes += 1
    else:
        reasons.append(f"✗ place_rate_mean={pr_mean:.2f}% < {PLACE_RATE_MEAN_THRESHOLD}%")

    # 决策
    if passes == 3:
        verdict = "PASS — 系数合适, 可进全量"
    elif v_cv > b["v_mean_cv"]:
        # V_CV 比基线还差 — 正则适得其反, 必须调小
        verdict = "COUNTERPRODUCTIVE — 正则适得其反 (V_CV 反升), 建议调小系数"
    elif passes == 2 and v_cv < V_CV_TARGET_MIN:
        verdict = "REGULARIZATION_TOO_STRONG — 正则过强, 建议调小系数"
    elif passes == 2 and v_cv > V_CV_TARGET_MAX:
        verdict = "REGULARIZATION_INSUFFICIENT — 方向对但不够, 可尝试略大系数"
    elif passes == 0 and v_cv > V_CV_TARGET_MAX:
        verdict = "NO_EFFECT — 正则无效, V_CV 未降, 建议换方向"
    else:
        verdict = "MIXED — 需结合具体指标判断"

    summary["reasons"] = reasons
    summary["verdict"] = verdict
    summary["passes"] = passes
    return summary


# ---------------------------------------------------------------------------
# 运行模式
# ---------------------------------------------------------------------------

def run_validate(coeff, base_seed):
    """验证模式: 5 seeds, ~1h."""
    tag = coeff_to_tag(coeff)
    out_dir = output_dir_for(coeff)
    session = tmux_session_name(coeff, 5)

    print("=" * 65)
    print(f"  Phase 7 Round 2b — V L2 验证")
    print(f"  Coeff: {tag}  |  5 seeds  |  base_seed={base_seed}")
    print("=" * 65)

    # 检查是否已存在
    if (out_dir / "training_cv_analysis.json").exists():
        print(f"\n  [WARN] 已有完成结果: {out_dir}")
        print(f"  用 --analyze --coeff {tag} 查看, 或删除目录后重跑")
        return False

    # 1. 修改代码
    print(f"\n--- 1. 修改 iql_agent.py ---")
    cur_coeff, is_reg = get_current_coeff()
    print(f"  当前 coeff: {cur_coeff or '无'} (regularized={is_reg})")
    ok, old, backup, err = modify_iql_agent(coeff)
    if not ok:
        print(f"  [ERROR] 修改失败: {err}")
        return False
    print(f"  备份: {backup.name}")
    print(f"  已设置 V L2 coeff = {tag}")
    new_c, _ = get_current_coeff()
    print(f"  验证: 当前 coeff = {new_c}")

    # 2. 启动 tmux 训练 + 自动分析
    print(f"\n--- 2. 启动训练 (tmux: {session}) ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 训练完成后自动运行分析
    train_cmd = (
        f"python -u {PHASE7_SCRIPT} "
        f"--n_seeds 5 --base_seed {base_seed} "
        f"--output_dir {out_dir} "
        f"--extra_train_args '--tau 0.5' "
        f"--skip_dry_run"
    )
    analyze_cmd = (
        f"python -u {__file__} --analyze --coeff {tag}"
    )
    full_cmd = f"{train_cmd} && echo '=== TRAIN DONE, RUNNING ANALYSIS ===' && {analyze_cmd}"

    # 用 { ...; } 包裹整条命令链, 确保 source/conda/train/analyze 输出全进 master.log
    wrapped_cmd = f"{{ {full_cmd}; }} > {out_dir}/master.log 2>&1"
    ok, err = create_tmux_session(session, wrapped_cmd)
    if not ok:
        print(f"  [ERROR] tmux 创建失败: {err}")
        return False

    print(f"  会话: {session}")
    print(f"  输出: {out_dir}")
    print(f"  日志: {out_dir}/master.log")
    print(f"\n  预计 ~1 小时完成. 完成后自动生成 summary.json")
    print(f"  查看: tmux attach -t {session}")
    print(f"  状态: python {__file__} --status")
    return True


def run_full(coeff):
    """全量模式: 30 seeds, ~6h."""
    tag = coeff_to_tag(coeff)
    out_dir = output_dir_for(coeff, mode="full")
    session = tmux_session_name(coeff, 30)

    print("=" * 65)
    print(f"  Phase 7 Round 2b — V L2 全量")
    print(f"  Coeff: {tag}  |  30 seeds  |  base_seed=20")
    print("=" * 65)

    if (out_dir / "training_cv_analysis.json").exists():
        print(f"\n  [ERROR] 已有完成结果, 拒绝覆盖: {out_dir}")
        return False

    # 1. 修改代码
    print(f"\n--- 1. 修改 iql_agent.py ---")
    ok, old, backup, err = modify_iql_agent(coeff)
    if not ok:
        print(f"  [ERROR] 修改失败: {err}")
        return False
    print(f"  备份: {backup.name}")
    new_c, _ = get_current_coeff()
    print(f"  当前 coeff = {new_c}")

    # 2. 启动
    print(f"\n--- 2. 启动训练 (tmux: {session}) ---")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = (
        f"python -u {PHASE7_SCRIPT} "
        f"--n_seeds 30 --base_seed 20 "
        f"--output_dir {out_dir} "
        f"--extra_train_args '--tau 0.5' "
        f"--skip_dry_run"
    )
    analyze_cmd = f"python -u {__file__} --analyze --coeff {tag}"
    full_cmd = f"{train_cmd} && echo '=== TRAIN DONE, RUNNING ANALYSIS ===' && {analyze_cmd}"

    # 用 { ...; } 包裹整条命令链, 确保 source/conda/train/analyze 输出全进 master.log
    wrapped_cmd = f"{{ {full_cmd}; }} > {out_dir}/master.log 2>&1"
    ok, err = create_tmux_session(session, wrapped_cmd)
    if not ok:
        print(f"  [ERROR] tmux 创建失败: {err}")
        return False

    print(f"  会话: {session}")
    print(f"  输出: {out_dir}")
    print(f"  日志: {out_dir}/master.log")
    print(f"\n  预计 ~6 小时完成. 完成后自动生成 summary.json")
    return True


def run_analyze(coeff, output_dir_override=None):
    """分析已完成实验, 生成 summary.json."""
    tag = coeff_to_tag(coeff) if coeff is not None else "unknown"
    if output_dir_override:
        out_dir = Path(output_dir_override)
    else:
        out_dir = output_dir_for(coeff)

    print("=" * 65)
    print(f"  Phase 7 Round 2b — 分析 (coeff={tag})")
    print(f"  目录: {out_dir.name}")
    print("=" * 65)

    metrics, err = extract_metrics(out_dir)
    if err:
        print(f"\n  [ERROR] {err}")
        return False

    summary = compare_to_baseline(metrics, coeff)

    # 保存 summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # 打印结果
    print(f"\n  实验结果 ({summary['n_seeds']} seeds):")
    print(f"    place_rate: mean={summary['metrics']['place_rate_mean']:.2f}%  "
          f"CV={summary['metrics']['place_rate_cv']:.2f}%")
    print(f"    v_mean CV:  {summary['metrics']['v_mean_cv']:.1f}%")

    print(f"\n  Baseline (τ=0.5 无正则, 30 seeds):")
    print(f"    place_rate: mean={summary['baseline']['place_rate_mean']:.2f}%  "
          f"CV={summary['baseline']['place_rate_cv']:.2f}%")
    print(f"    v_mean CV:  {summary['baseline']['v_mean_cv']:.1f}%")

    print(f"\n  Delta (实验 - baseline):")
    d = summary["deltas"]
    print(f"    place_rate_mean: {d['place_rate_mean_delta']:+.2f}pp")
    print(f"    place_rate_cv:   {d['place_rate_cv_delta']:+.2f}pp")
    print(f"    v_mean_cv:       {d['v_mean_cv_delta']:+.2f}pp")

    print(f"\n  检查项 ({summary['passes']}/3 通过):")
    for r in summary["reasons"]:
        print(f"    {r}")

    print(f"\n  >>> 判定: {summary['verdict']}")
    print(f"\n  summary.json 已保存: {summary_path}")
    return True


def run_status():
    """查看所有实验状态."""
    print("=" * 65)
    print("  Phase 7 Round 2b — 实验状态")
    print("=" * 65)

    # 1. 当前 iql_agent.py 状态
    print(f"\n--- 当前 iql_agent.py ---")
    coeff, is_reg = get_current_coeff()
    print(f"  V L2 coeff: {coeff or '无'} (regularized={is_reg})")

    # 2. 备份列表
    print(f"\n--- 可用备份 ---")
    backups = sorted(WORKSPACE.glob(f"{BACKUP_PREFIX}*"))
    if backups:
        for b in backups[-5:]:
            print(f"  {b.name}")
    else:
        print("  (无)")

    # 3. tmux 会话
    print(f"\n--- tmux 会话 ---")
    r = subprocess.run(["tmux", "list-sessions"], capture_output=True, text=True)
    if r.returncode == 0:
        for line in r.stdout.strip().split("\n"):
            if "phase7" in line:
                print(f"  {line}")
    else:
        print("  (无 phase7 会话)")

    # 4. 实验目录
    print(f"\n--- 实验结果 ---")
    dirs = sorted(WORKSPACE.glob("outputs/phase7_round2b_tau0.5_vl2_*"))
    if not dirs:
        print("  (无)")
    else:
        for d in dirs:
            analysis = d / "training_cv_analysis.json"
            results = d / "training_cv_results.json"
            summary = d / "summary.json"

            if summary.exists():
                s = json.load(open(summary))
                print(f"  {d.name}: DONE (n={s['n_seeds']}, "
                      f"pr={s['metrics']['place_rate_mean']:.1f}% "
                      f"CV={s['metrics']['place_rate_cv']:.2f}%, "
                      f"v_cv={s['metrics']['v_mean_cv']:.1f}%, "
                      f"verdict={s['verdict'][:30]})")
            elif analysis.exists():
                a = json.load(open(analysis))
                n = len(a.get("place_rates", []))
                print(f"  {d.name}: ANALYZED (n={n}, "
                      f"pr={a.get('place_rate_mean', 0):.1f}%, "
                      f"CV={a.get('place_rate_cv', 0):.2f}%)  "
                      f"[run --analyze to generate summary]")
            elif results.exists():
                r_data = json.load(open(results))
                n_t = len(r_data.get("training_runs", {}))
                n_e = len(r_data.get("eval_runs", {}))
                print(f"  {d.name}: IN PROGRESS (train={n_t}, eval={n_e})")
            else:
                print(f"  {d.name}: STARTED (no results yet)")

    return True


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7 Round 2b V L2 正则化自动化脚本")
    parser.add_argument("--mode", choices=["validate", "full", "status", "analyze"],
                        default="status")
    parser.add_argument("--coeff", type=float, default=None,
                        help="V L2 系数, 如 3e-6, 1e-6")
    parser.add_argument("--base_seed", type=int, default=None,
                        help="验证模式 base_seed (默认 200, 轮换 100/200/300)")
    parser.add_argument("--rollback", action="store_true",
                        help="回滚 iql_agent.py 到最近备份")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="指定输出目录 (用于 analyze 已有实验, 覆盖 coeff 推导)")
    args = parser.parse_args()

    if args.rollback:
        print("=" * 65)
        print("  回滚 iql_agent.py")
        print("=" * 65)
        ok, backup, err = rollback_iql_agent()
        if ok:
            print(f"  已从 {backup.name} 恢复")
            c, r = get_current_coeff()
            print(f"  当前 coeff: {c or '无'} (regularized={r})")
        else:
            print(f"  [ERROR] {err}")
        return

    if args.mode == "status":
        run_status()
        return

    if args.mode == "analyze":
        if args.coeff is None and not args.output_dir:
            print("[ERROR] --analyze 需要 --coeff 或 --output_dir")
            return
        run_analyze(args.coeff, args.output_dir)
        return

    if args.mode in ("validate", "full"):
        if args.coeff is None:
            print("[ERROR] --mode validate/full 需要 --coeff")
            return
        if args.mode == "validate":
            bs = args.base_seed if args.base_seed is not None else 200
            run_validate(args.coeff, bs)
        else:
            run_full(args.coeff)


if __name__ == "__main__":
    main()
