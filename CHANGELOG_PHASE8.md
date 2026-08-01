# Phase 8 — DT Router 重训 + V_CV=30% 下游验证 Changelog

## 阶段总览

目标: 在新 baseline τ=0.5 (V_CV=30%) 上重训 DT Router，验证 V_CV=30% 是否影响下游业务指标。

最终结果: **三节点全 PASS → V_CV=30% 是"纸老虎"，被系统鲁棒性吸收**。DT Router v3 效果不显著归档。τ=0.5 v4 确立为最终交付/部署 Baseline。

---

## Step 1-4: DT Router v3 训练 (2026-07-21)

### 特征提取 checkpoint 选择
- 中位数 seed: **seed1822509288** (spawn_idx=0, place_rate=60.0%，代表典型分布)
- 来自 `outputs/phase7_round2a_tau0.5_N30/train_seed1822509288/final_model.pt`

### v4 / v4.1 评估 (eval_seed=42, 200 episodes)
| Config | place_rate | drift | near_miss | 说明 |
|--------|-----------|-------|-----------|------|
| v4 (τ=0.5, chunk_size=4) | 60.0% (120/200) | 31 | 49 | 与 N30 评估一致 (可重复) |
| v4.1 (τ=0.5, adaptive_chunk) | 62.5% (125/200) | 13 | 62 | +2.5pp, drift -18 |

- v4.1 在新 τ=0.5 baseline 上有效 (旧 τ=0.7 上 FAILED)
- 可达上界 = (120+125-85)/200 = **80.0%** (vs v4 单独 60.0%)

### Codebook v3.0
- 195 trainable entries (excluded 5 all-drift)
- v4-only-win=35, **v4.1-only-win=40** (切换信号), both-win=85, both-lose=40, weak-label=22
- Optimal: v4=43.1%, v4.1=56.9%

### DT Router v3 训练
- DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, gini, balanced)
- Training acc: 76.41%
- **CV acc: 54.36% ± 6.15%** (仅略高于随机 50%，切换信号弱)
- Top features: dist_change_rate (0.41), q1_at_step20 (0.27), best_dist_early (0.20)
- v4: P=0.67 R=0.89 F1=0.77 | v4.1: P=0.89 R=0.67 F1=0.76

---

## Step 5: PILOT 5 seeds (2026-07-21)

### 结果 (N=5, seeds=[123,456,789,2024,314], 排除 42 避免泄露)
| Config | mean | CV | drift_mean |
|--------|------|-----|-----------|
| v4_tau05 | 57.70% ± 4.40pp | 7.62% | 30.6 |
| dt_router_v3 | 61.40% ± 2.10pp | 3.43% | 27.6 |

- **diff = +3.70pp**, σ_d=3.47pp, r≈0.69
- diffs 全为正: [+7.5, +7.5, +1.0, +1.5, +1.0]
- 功效: +2pp@N30=0.883 ✓ → 决定扩到 N=30

### ⚠ 乐观偏差预警 (用户提出)
- 5 seeds +3.70pp 可能因小样本偏高
- diffs 范围 +1.0~+7.5 暗示长尾分布
- N=30 均值可能回缩到 +2.5~3.0pp

---

## Step 6: N=30 扩量 (2026-07-21)

### 结果 (N=30, 30 eval seeds)
| Config | mean | std | CV | drift_mean |
|--------|------|-----|-----|-----------|
| v4_tau05 | 59.82% | 3.45pp | 5.77% | 28.5 |
| dt_router_v3 | 60.57% | 3.04pp | 5.01% | 28.7 |

- **diff = +0.75pp ± 3.17pp** (t=1.295, p>0.05 不显著)
- r ≈ 0.58
- diffs: 18 正 / 9 负 / 3 零
- **PILOT 乐观偏差证实**: +3.70pp → +0.75pp (回缩 2.95pp，用户预警完全正确)

### 功效分析 (基于实测 σ_d=3.17pp)
| 效应 | N=10 | N=30 |
|------|------|------|
| +1pp | 0.168 ✗ | 0.408 ✗ |
| +2pp | 0.513 ✗ | **0.932 ✓** |

---

## Step 7: 三节点监测 (N=30)

| 节点 | 指标 | N=30 值 | 阈值 | 状态 |
|------|------|---------|------|------|
| 1. Router 置信度分布 | fallback_rate CV | 10.2% | <30% | ✓ PASS |
| 2. 路由决策一致性 | n_switched CV | 13.0% | <40% | ✓ PASS |
| 3. 端到端 place_rate CV | place_rate CV | 5.01% | ≤8% | ✓ PASS |

- fallback_rate mean=39.8% (偏高但跨 seeds 稳定)
- n_switched mean=56.1/200 (28.2% switch rate)
- **三节点全 PASS → V_CV=30% 被系统鲁棒性吸收**

---

## Step 8: TOST + 决策门

### TOST 分析
- diff mean=+0.75pp, std=3.17pp, N=30
- TOST delta=±1pp
- t1=3.02 > 1.697 (reject), **t2=-0.43 > -1.697 (fail)**
- TOST FAIL: 无法声明 ±1pp 等价 (std 太大)

### 决策门 R1-R6
- R1 (cv_threshold=3.0%): ⚠ TRIGGERED (CV=5.77% > 3.0%)
- R2 (TOST not equivalent): ⚠ TRIGGERED
- R3 (declare equivalent): pass (inconclusive)
- R6 (anchor_ci_lower=0.68): ⚠ TRIGGERED (58.6% < 68%)
- FINAL VERDICT: BLOCKED

### ⚠ 阈值不适用说明 (用户签收 2026-07-21)
旧决策门阈值基于 τ=0.7 baseline 设置，不适用于 τ=0.5 新 baseline:
- **R1 cv_threshold=3.0%**: 是训练 CV 阈值误用于评估 CV。环境固有评估方差底线 5%~6% (200 eps × 30 seeds)，τ=0.5 评估 CV=5.77% 已触碰底线，无法进一步降低
- **R6 min_anchor_ci_lower=0.68**: 基于 τ=0.7 v4 的 68.2% place_rate。τ=0.5 牺牲 mean 换稳定性 (59.82% vs 68.2%)，58.6% CI 下界对新 baseline 合理
- **新 baseline 标准下: PASS**

---

## 最终结论

### 1. Phase 7 悬念闭环 (核心成果)
**V_CV=30% 是"纸老虎"**: IQL 策略提取 (AWR) + 动作执行有足够容错空间，V 网络数值波动未传导到机器人行为。τ=0.5 换训练稳定性的战略完全正确，**无需回 Round 2c 折腾 V 预训练**。

### 2. DT Router v3 归档 (方向 A)
- 效果不显著: +0.75pp (p>0.05)
- 信号太弱: CV accuracy 54.4% 仅略高于随机
- 徒增复杂性: 推理延迟 + 维护成本 > 收益
- 归档路径: `outputs/phase8_dt_router_v3/`

### 3. τ=0.5 v4 确立为最终交付/部署 Baseline
- **最终配置**: `IQLAgent(tau=0.5, beta=3.0, gamma=0.99, n_step=5, chunk_size=4)` 无正则
- **性能**: mean=59.82% ± 3.45pp, 评估 CV=5.77%
- **checkpoint**: `outputs/phase7_round2a_tau0.5_N30/train_seed1822509288/final_model.pt` (中位数 seed，代表典型部署)

### 4. 未来探索方向 (留档)
- (a) **回溯数据集**: RL 上限由数据决定，60% 瓶颈可能需检查成功轨迹质量/多样性
- (b) **微调 expectile**: 在 τ=0.5 稳定基础上极微小提升 τ (0.52/0.55) 寻找"mean 略回升但 CV<8%"甜点，需极谨慎 N=30 功效分析

---

## 产物清单

| 文件 | 说明 |
|------|------|
| `outputs/phase8_dt_router_v3/v4_features_tau05_seed42.json` | v4 特征提取 (200 eps) |
| `outputs/phase8_dt_router_v3/v4_1_features_tau05_seed42.json` | v4.1 特征提取 (200 eps) |
| `outputs/phase8_dt_router_v3/v4_extracted_features.json` | 7-dim 特征向量 |
| `outputs/phase8_dt_router_v3/codebook_v3.json` | Codebook v3.0 (195 trainable) |
| `outputs/phase8_dt_router_v3/dt_model_v3.pkl` | DT Router v3 模型 |
| `outputs/phase8_dt_router_v3/feature_importance_v3.json` | 特征重要性 |
| `outputs/phase8_dt_router_v3/training_report_v3.json` | 训练报告 |
| `outputs/phase8_dt_router_v3/pilot_results.json` | PILOT 5 seeds 结果 |
| `outputs/phase8_dt_router_v3/n30_results.json` | N=30 结果 |
| `outputs/phase8_dt_router_v3/3node_monitor_n30_results.json` | 三节点监测报告 |
| `outputs/dt_orchestrator/experiment_manifest_dt_router_v3_vs_v4_tau05.json` | 预注册 manifest |

## 工具脚本
| 文件 | 说明 |
|------|------|
| `phase8_pilot_eval.py` | PILOT/N30 评估脚本 (v4_tau05 vs dt_router_v3 配对) |
| `phase8_3node_monitor.py` | 三节点监测分析 (置信度/决策一致性/端到端 CV) |
