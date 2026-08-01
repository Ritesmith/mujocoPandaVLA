# Phase 7 — IQL 训练稳定性优化 Changelog

## 阶段总览

目标: 解决 IQL 训练方差过大 (CV=17.83%) 导致 DT Router 增益无法统计检测的问题。

最终结果: **place_rate CV 17.83% → 6.78% (-62%)**, 达标 8% 阈值。

---

## Round 1: EMA / Huber 消融式 A/B (2026-07-17 → 2026-07-18)

### 实施
- **EMA on V network** (`--ema_v`, τ_ema=0.005): v_net_ema 平滑副本用于 Q-target bootstrap 和 advantage baseline
- **Huber Loss on Q** (`--huber_loss`, δ=10.0): smooth_L1 替代 MSE, 对重尾 Q target 鲁棒

### 结果 (10 seeds 配对)
| Config | mean | CV | v_mean CV | r (vs baseline) | TOST |
|--------|------|----|-----------|-----------------|------|
| baseline | 58.9% | 17.83% | 51.7% | — | — |
| EMA-only | 60.4% (+1.5pp) | 14.94% (-2.89pp) | 49.7% | 0.90 (强正相关) | INCONCLUSIVE (p=0.0699) |
| Huber-only | 58.25% (-0.65pp) | 19.50% (+1.67pp) | 27.1% | -0.02 (零相关!) | INCONCLUSIVE |

### 关键洞察
- EMA 方向对但平滑力度不够 (r=0.90 保持 landscape 一致性)
- Huber 把 V 锁进不同盆地 (r≈0), 证实 "Q/V 稳定 ≠ 策略稳定"
- 两者都没直击 V 多解根因 → 转 Round 2

---

## Round 2a: τ 扫描 (2026-07-19 → 2026-07-20)

### 假设
IQL expectile regression 在 τ>0.5 时非凸, 导致 V 多解。降 τ 到 0.5 (退化为 MSE) 可消除多解性。

### 结果
| Config | N | mean | CV | v_mean CV | 关键事件 |
|--------|---|------|----|-----------|----------|
| baseline (τ=0.7) | 10 | 58.9% | 17.83% | 51.7% | — |
| τ=0.5 (10 seeds) | 10 | 58.85% | 11.57% | — | 救回灾难性 seed 2976135721 (35%→47.5%) |
| τ=0.6 (10 seeds) | 10 | 57.65% | 13.00% | — | 不如 τ=0.5 |
| **τ=0.5 (30 seeds)** | **30** | **60.00%** | **6.78%** | **30.5%** | **PASS 8% 阈值** |

### 决策
τ=0.5 确立为新 baseline。place_rate CV 达标, 但 V_mean CV=30.5% 仍高于目标 25%。

---

## Round 2b: V L2 正则化 (2026-07-20) — NEGATIVE RESULT

### 假设
V 网络权重 L2 正则化可约束 V 的解空间, 降低 V_mean CV。

### 结果
| Config | N | place_rate mean | place_rate CV | V_mean CV |
|--------|---|-----------------|---------------|-----------|
| baseline (τ=0.5, 无正则) | 30 | 60.00% | 6.78% | 30.5% |
| τ=0.5 + 1e-5 V-L2 | 5 | 59.7% | 7.64% | 38.6% (+8.1pp 反升) |
| τ=0.5 + 3e-6 V-L2 (验证) | 5 | 58.5% | 5.57% | 22.6% (看似达标) |
| τ=0.5 + 3e-6 V-L2 (全量) | 19 | 59.97% | 6.60% | 33.45% (回升, 小样本噪声) |

### 失败原因
1. **5-seed V_CV=22.6% 是抽样偏差** — 19 seeds 后回升到 33.45%
2. **L2 对权重尺度偏移不具不变性** — 模型可找到等价参数使 L2 项更小但不改变行为
3. **多解根源是损失曲面非凸性**, L2 作用于权重范数而非优化曲面凸性

### 处置
- 已回滚 iql_agent.py L251 到无正则版本
- 实验归档: `outputs/archive/phase7_round2b_*`
- 判定: L2 是"错误的工具", 停止此方向

---

## 最终配置 (2026-07-20)

```python
# τ=0.5, 无正则, 无 EMA, 无 Huber
IQLAgent(tau=0.5, beta=3.0, gamma=0.99, n_step=5, chunk_size=4, ...)
```

| 指标 | Phase 7 起点 | Phase 7 终点 | 改善 |
|------|-------------|-------------|------|
| place_rate mean | 58.9% | 60.00% | +1.1pp |
| place_rate CV | 17.83% | 6.78% | **-11.05pp (-62%)** |
| V_mean CV | 51.7% | 30.5% | -21.2pp |
| 灾难性 seed (2976135721) | 35.0% | 47.5% | +12.5pp |

**达标**: place_rate CV ≤ 8% ✓
**未达标**: V_mean CV ≤ 25% (当前 30.5%, 留待后续)

---

## Round 2c 候选方向 (待决策)

1. **接受现状进 DT Router 重训** (推荐): 验证 V_CV=30% 是否真的影响下游业务指标
2. **V 预训练**: 先 τ=0.5 纯 MSE 预训练 100 epoch 给 V 良好初始化, 再切 τ=0.7 微调
3. **V 网络架构加深** (256→512): 风险高需重调参

用户决策: 优先 (1) 接受现状

---

## 工具与产物

| 工具 | 用途 |
|------|------|
| `phase7_train_cv.py` | 10/30-seed 训练编排 |
| `phase7_round1_tost_analysis.py` | TOST 配对分析 |
| `automate_phase7_vl2.py` | V L2 正则化自动化 (含 tmux/备份/回滚) |
| `outputs/phase7_round1_*` | Round 1 EMA/Huber 数据 |
| `outputs/phase7_round2a_*` | Round 2a τ 扫描数据 |
| `outputs/archive/phase7_round2b_*` | Round 2b V L2 (归档) |
| `outputs/phase7_variance_decomposition/` | Round 2a 30-seed τ=0.5 baseline |
