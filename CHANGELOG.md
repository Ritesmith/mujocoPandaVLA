# VLA Workspace — Master Changelog

> **任务**: PandaVLA-v0 抓放（place 阶段）策略迭代
> **数据集**: `data/D_expert.npz`（29467 transitions / 200 episodes / 68.5% expert place rate）
> **工作空间**: `/home/w/vla_workspace`
> **生成时间**: 2026-07-15
> **覆盖范围**: 从项目初始化到 DT Router v2 非确定性根因分析的完整版本史

本 changelog 按时间顺序记录所有版本/实验/架构决策，每条含：动机、配置、结果、决策。索引表见末尾。

---

## Phase 0 — 项目初始化与基础设施（2026-06-14 ~ 06-22）

### [Setup] ROS2 ↔ MuJoCo 迁移 & 仿真器选型
- **决策**: 选定 MuJoCo 为主力仿真器（放弃 ManiSkill3 / Isaac Lab）
- **依据** (`SIMULATOR_EVAL.md`, 2026-06-18): 当前硬件 GTX 1660 Ti (6GB, 无 RT Cores) 不满足 Isaac Lab 最低要求；ManiSkill3 GPU 并行需额外 PhysX 库且动作空间 9D 与现有 8D 不一致，迁移成本中等。MuJoCo 已与 VLA 流程完全集成，零迁移成本。
- **长期预案**: 硬件升级至 RTX 3070+ 后重新评估 Isaac Lab（4096 并行环境）。

### [Setup] 抓取物理修复（solution_summary.md, 2026-06-26）
- **根因**: Panda 手 45° z 旋转导致 IK 定位误差 >100mm，手指 pad 与方块不对齐，闭合时推开方块。
- **修复**: 修改 `scene.xml` 将 hand quat 从 `-0.3826834 0 0 0.9238795` 改为 `1 0 0 0`（无旋转）；增加摩擦 `friction="2 1 0.01"`；校准 hand-to-finger-mid 偏移量。
- **影响**: 解锁后续所有 place 策略训练（抓取阶段稳定）。

---

## Phase 1 — Place 策略迭代 V5 → V58（2026-06-21 ~ 07-09）

> 此阶段为 PPO 在线训练迭代，目标 place_rate。绝大多数版本日志在 `outputs/place_policy_v{N}_train.log`。V5–V58 均未稳定突破 54%。

### V5 — 首个 place 策略基线
- **配置**: BC warmstart + PPO 微调，初始 place 策略
- **结果**: ~22% place rate（仅 BC warmstart 水平）
- **日志**: `outputs/v5_pipeline.log`, `outputs/place_policy_bc_1_0_train.log`

### V25–V38 — 奖励函数调参阶段
- 多轮迭代调整 proximity bonus、distance penalty、lift threshold
- V31–V34 探索不同 reward shaping 组合，未突破 50%
- V35a/V35b 双分支实验，V36–V38 收敛性优化
- **日志**: `outputs/place_policy_v{31..38}_train.log`

### V40–V45 — 学习率与固定化修复
- V40–V44 调整 lr (1e-4 / 1e-5)、BN running stats 固定
- V45 引入 cosine LR scheduler
- **日志**: `outputs/place_policy_v{40..45}_train.log`

### V46–V49e — Pipeline 多实验编排
- V46: 3 子实验（exp1/2/3），探索不同 OPR (Offline Policy Regularization) 系数
- V47–V48: 双子实验对比
- V49–V49e: 迭代式训练（iter2/iter3/iter4/iter5），`v49d_pipeline_full.log`
- **日志**: `outputs/v4{6..9}*_*.log`

### V50a/b/c — 备份与目标范围实验
- V50a: 基线，V50b: 备份对比 + finetune + 随机目标评估，V50c: 消融
- **日志**: `outputs/v50{a,b,c}_*.log`, `v50_ablation.log`
- **关键发现**: `v50b_failure_analysis.txt` 首次系统记录失败模式

### V51–V58 — 诊断与修复迭代
- V51: 失败诊断 + restored env 评估
- V52–V55: pipeline 调优
- V56: postmortem 分析（`v56_postmortem_run.log`）
- V57–V58: 训练收敛性改进
- **结果**: V58 达到 ~54% place rate，成为 V59 的直接前驱
- **日志**: `outputs/v5{1..8}_*.log`

---

## Phase 2 — V59 突破与方法族穷举（2026-07-10 ~ 07-12）

### V59 — 生产基线确立（56% place rate）
- **配置**: 164 v5 grasp states，L2 正则化，cosine LR，early stopping
- **关键修复** (`20260710/topics.md`): 
  - 图像增强仅在训练时启用（评估时关闭，避免物体定位扰动）
  - `freeze_bn_running_stats()` 锁定 BN running stats（防止特征提取退化）
  - 停止于 2500 steps first_eval_floor 后重启，2.5k 评估验证 place_rate 回到 ~54%
- **最终结果**: **56% place rate (N=200)** — 成为后续所有对比的生产基线
- **日志**: `outputs/v59_train.log`, `outputs/v59_pipeline.log`
- **检查点**: `outputs/place_policy_v59/best_hier/`

### V60 — BC 梯度修复尝试（CRASHED）
- **配置**: `lambda_bc=0.3`（BC 正则化系数）
- **结果**: 单次 PPO 更新即崩溃 50%→15%。BC contribution (0.145) 是 PPO policy_loss (0.007) 的 21 倍
- **教训**: `lambda_bc=0.3` 适用于 from-scratch 训练，不适用于微调
- **日志**: `outputs/v60_train.log`

### V61 — 温和 BC 锚定
- **配置**: `lambda_bc=0.01`（BC contribution ~0.005，~0.7x PPO）
- **结果**: 仍无法超越 V59
- **关键洞察**: V59 策略与 demo actions 差异显著（RMS 误差 0.7 on [-1,1]），demo 可能不代表最优策略

### V62–V66 — 奖励结构探索（全部 CRASHED to 5%）
- V62–V65: 不同 reward shaping，均在 step 3000 评估跌至 5%
- V65 根因分析（`20260711/topics.md`）推翻两个假设:
  - ❌ block_target_dist 是 3D（已确认 `np.linalg.norm(block_pos - target_pos)`）
  - ❌ height penalty 激活（excess_height=0 at 6-9cm lift）
  - ✅ 真因: 无条件 lowering bonus `+5*height_progress` 奖励 Z 下降而不论 XY 位置
- V65 fix (v15): gating lowering bonus on `xy_dist < 0.10`
- V66 (v15 fix 应用): **仍 CRASHED to 5%**
  - 关键诊断证据 `[HIER_EVAL] DECOUPLING DETECTED`: place_mode reward 9.3→65.3（改善）但 hier place_rate 50%→5%（崩溃）
  - **根因**: reward STRUCTURE 而非 coefficient — 一次性 proximity bonuses (+20@15cm, +50@10cm, +100@5cm = +170) 可在不实际放置情况下被"刷取"，policy 学会接近后悬停
  - V66 50-ep eval = 56%（与 V59 相同），证明 V59-V66 best_hier 都在 50-56%，PPO 更新是唯一退化原因
- **日志**: `outputs/v6{2..6}_*.log`

### V67 — v16 奖励重构
- **配置**: 
  - k=5.0 distance penalty（原 1.0）
  - 移除一次性 +20/+50 proximity bonuses
  - 移除一次性 +100 approach bonus
  - 新增 progressive proximity reward `0.1/(1.0+block_target_dist)` per step
  - 保留 v15 gate + DECOUPLING DETECTION early stop
- **结果**: 仍未突破 V59
- **日志**: `outputs/v67_train.log`, `outputs/v67_nohup.log`

### V68–V71b — 收尾实验
- V68: 最终 reward 调参，V69–V70: 并行分支，V71a/V71b: 消融
- **结论**: PPO 在线训练路径彻底穷尽，V59 (56%) 为最终生产策略

### [Method Family Exhaustion] 7 大方法族全部失败（2026-07-12）

> Spec: `v59-breakthrough-csil-voronoi` — 信息论瓶颈诊断

**核心洞察** (`20260712/topics.md`): V59 处于信息论根本瓶颈 — 所有方法都依赖 V59 自身作为唯一信息源，"自己教自己"必然退化。突破需引入 V59 无法自我生成的新信息。

| 方法族 | 实验 | 结果 | 失败原因 |
|--------|------|------|----------|
| 1. PPO 微调 | V60–V71b | 全部 ≤56% | PPO 梯度对预训练解破坏性 |
| 2. BC | bc_expert_v1 | 无提升 | covariate shift |
| 3. DAgger | V2A/V2B | 34% / 13% | V2A 臂分布偏移，V2B 梯度污染 |
| 4. Policy Distillation | distill_v1 | NO-OP | student=teacher → zero loss |
| 5. Ensemble | V62–V68 | NO-OP | checkpoints 与 V59 字节相同 |
| 6. CSIL++ 连贯奖励 | csil_plus_plus | 无突破 | 奖励信号信息增益不足 |
| 7. Voronoi 分区 | voronoi_partition | **STRUCTURAL NO-OP** | V59 deterministic actions 在 success/failure 间相同（diff<0.007 < within-group std 0.004-0.018），success/failure 由 STATE 决定而非 action |

**动作选择实验确认**: V59 deterministic mean 最优 — std=0.0=53%, std=0.1=27%, std=0.3=20%, std=0.5=13%, std=1.0=20%。任何随机噪声单调退化 V59。

**Voronoi 量化判定**: STRUCTURAL NO-OP — 子策略训练无差异可学。V59 在策略权重 AND 动作选择两个维度均达信息论极限。

**结论**: V59 (56%) 为最终生产策略。突破需 EXTERNAL 信息或 ARCHITECTURAL 改变。

### [Automation] 实验决策树引擎交付（2026-07-13）
- **产物**: `experiment_decision_tree.py` (668 行)，基于 8 大失败方法族的静态规则
- 5 条决策路径（self_generated / external_demos / human_corrections / different_architecture / different_paradigm）
- 负证据回填至 3 个 JSON: `nv_voronoi_quantization_noop.json`, `nv_bc_expert_demos_covariate_shift.json`, `pi_backbone_probe_sufficient.json`
- 44 tests passed，Mermaid 文档 `decision_tree.mmd`
- **日志**: `20260713/topics.md`

### [Online DAgger] 迭代 DAgger 实验（2026-07-13）
- 10 轮迭代（50-ep 收集 + 5-epoch 训练 + 50-ep 评估），PID 324262
- L2 `execute()` 方法实现，49/49 tests passed
- **结果**: 未突破 V59（仍 56%）

### [Diffusion Policy] 多模态原型尝试（2026-07-13）
- **目标**: 验证多模态动作分布建模能否突破 V59 单峰 56% 上限
- **配置**: K=20 DDIM steps, horizon=16, batch_size=64, Conditional U-Net (1D) + FiLM, channels [256,512,1024]
- **成功标准**: val success ≥60%；失败: 3-seed mean <56%
- **日志**: `outputs/diffusion_policy_smoke/`

---

## Phase 3 — RL-from-scratch 失败与 IQL 范式确立（2026-07-13 ~ 07-14）

### RL-from-scratch v1 — PPO 微调 BC warmstart（FAILED, 0%）
- **配置**: BC warmstart (22%) + PPO fine-tuning，防御措施齐全（frozen backbone, BN, target_kl=0.015, BC regularization, gradient clipping, entropy regularization）
- **结果**: 10K steps 内 place_rate → 0%，触发 first_eval_floor 安全停止
- **根因** (`20260714/topics.md`): PPO 梯度方向对预训练解根本性破坏，与 V59 微调失败同因
- **决策树更新**: `different_paradigm → rl_from_scratch` 路径标记为 falsified
- **剩余开放路径**: 仅 `human_corrections → realtime_dagger`（需 human_teleop_interface）
- **归档**: `outputs/rl_from_scratch_v1/`, `family_10_rl_from_scratch/`
- **测试**: 65/65 passed

### 奖励密度诊断（diagnose_reward_density.py, 2026-07-14）
- **关键发现**:
  - action-related rewards 比 state-related rewards 弱 3077 倍
  - 优势函数 A(s,a) 无方向性信号（V(s) 准确捕获 state rewards）
  - terminal success rewards 从未触发（0% place rate）
- **对 IQL 的启示**: IQL 用 offline oracle data（含 +200 terminal rewards）有机会，但 Q 函数仍可能被 state rewards 主导
- **决策**: 启动 IQL 离线 RL 实现

---

## Phase 4 — IQL 离线 RL 时代（2026-07-14 ~ 07-15）

> 中央索引: `outputs/IQL_EXPERIMENTS.md`
> 训练脚本: `train_iql.py`（支持 `--n_step`, `--oversample_dist`, `--reward_shaping`）
> 评估脚本: `evaluate_iql_env.py`（支持 `--hybrid_checkpoint`, `--chunk_size`, `--adaptive_chunk`, `--log_q_values`）
> Orchestrator: `orchestrator.py`（L1 后验规则匹配器，读 YAML + JSON 决策）

### IQL v1 — 标准 IQL (n_step=1) ✅ 首次超越 V59
- **配置**: n_step=1, τ=0.7, β=3.0, γ=0.99, 100 epochs
- **训练指标**: Q_gap=102.25 (>100 target), advantage_success=0.87 > advantage_failure=-0.21 (排序正确), AWR weight entropy=75.7%, ESS=65.3
- **环境评估**: **65.5% (N=200)**, 64.4% (N=1000), drift 34.6% (N=1000)
- **意义**: 首个 offline RL 击败 V59 (56%)
- **关键成功因素**: Expectile regression (τ=0.7) 偏置 V 至上分位 + offline expert data (68.5% success) + AWR 指数放大 advantage
- **目录**: `outputs/iql_v1/`

### IQL v2 — Q-Chunking (n_step=5) ⚠️ Advantage 反转
- **假设**: n-step bootstrap (h=5) 加速稀疏奖励任务的价值传播
- **配置**: n_step=5 + oversampling
- **训练指标**: Q_gap=137.22 (+34%), 但 **advantage 反转** — advantage_success=-2.23 < advantage_failure=-1.58（AWR 给 failure actions 更高权重）
- **环境评估**: 67.5% (N=200, +2pp), 但 N=1000 回退至 63.4%（与 v1 持平）
- **决策**: 引入 reward shaping 修复 advantage 反转
- **目录**: `outputs/iql_v2_qchunk/`

### IQL v2 hybrid — 混合策略（假设证伪）
- **假设**: n_step=5 导致近目标 overshoot，dist<10cm 时切换 v1 (n_step=1)
- **结果**: 66.5%（无提升 vs v2-only 67.5%）
- **结论**: drift 非由 n_step=5 overshoot 引起，根因是 stay-near-target 奖励信号不足
- **drift**: 31.3% (N=1000, +3.6pp vs v2)

### IQL v3 — Direction-Aware Reward Shaping（无 place rate 提升）
- **假设**: 密集 reward shaping（overshoot penalty + stay bonus + leave penalty）修复 advantage 反转 + 减少 "reach 3cm then drift to 20cm"
- **配置**: n_step=5 + shaping
- **训练指标**: advantage 反转 **修复** (gap: -0.65 → +2.07), Q_gap=117.58 (-14% vs v2), failure return -136→-6（stay bonus 累积）
- **环境评估**: **64.8% (N=200)** — 低于 v2 (67.5%)，回到 v1 floor (65.5%)
- **drift**: 27.1% (无改善 vs v2 27.7%), near_miss 72.9% (无改善)
- **关键教训 (#6)**: 训练指标 ≠ 环境性能。advantage 反转是症状非根因。reward shaping 路径穷尽 — drift 需架构性改变（action chunking）而非 reward design
- **目录**: `outputs/iql_v3_shaping/`

### IQL v4 — True Action Chunking (h=4) ✅ 突破 70% 屏障
- **假设**: 近距单步决策不一致导致 drift。Actor 输出 h=4 连续动作（true action chunking）提供时序一致性
- **配置**: chunk_size=4, 共用 v4 checkpoint
- **训练指标**: Q_gap=138.74（最高，44-dim Critic 输入无崩溃）
- **环境评估**: **71.9% (N=200)** — **突破 70% 屏障** (+4.4pp over v2)
- **drift**: 37.5% of failures（**上升** from 27.7%）— 4/10 采样 drift 仍现 "reach near then drift to 20cm" 灾难性跳变
- **解读**: chunking 帮助成功 episode（时序一致性提升 reach 可靠性），但 **未修复 drift** — 近目标生成 bad chunk 时，4 个连续 bad action 开环执行无 mid-chunk 修正
- **决策**: v4 达成主成功指标 (place_rate > 70%)。持续 drift 需不同方法 — adaptive chunk size 或物理接触调查
- **目录**: `outputs/iql_v4_chunking/`
- **P1 物理调查** (`analyze_drift_physics.py`, 100 episodes):
  - ✅ 所有 drift episodes 在 best_dist 时 block_speed=0.00 cm/s（静态/被夹持，非滑动）
  - ✅ drift best_dist 分布均匀 [3.5–9.5cm]（无几何不稳定）
  - ✅ 6/10 drift: terminal jump ~16cm in single step (steps_after_best=1) — chunk_size=4 开环放大直接证据
  - **结论**: drift 是 POLICY-driven（非 physics）
  - **模式**: 60% 终末跳变 / 30% 突然跳变+部分恢复 / 10% 渐进漂移

### IQL v4.1 — Adaptive Action Chunking (4→2→1) ❌ REGRESSION
- **假设**: drift 由开环放大 bad chunks 引起（proximate cause）。distance-based adaptive chunk_size (4→2→1) 在 dist<6cm 精接触区启用 per-step 反馈修正
- **配置**: 复用 v4 checkpoint（不重训），adaptive_chunk 模式，thresholds far=0.12m / mid=0.06m
- **结果** (`experiment_registry.yaml`):
  - place_rate: **64.5% (-7.4pp from v4 71.9%)** — 低于预期 [0.74, 0.82]
  - drift_abs: 22（unchanged from v4's 21）— 低于预期 [5, 12]
  - near_miss_abs: 49 (+14 from 35)
  - chunk_steps: cs4=6226 (40.7%), cs2=5127 (33.5%), cs1=3938 (25.8%), 504 switch events
- **Critic 时序偏差 CONFIRMED** (Q-value diagnostics at best_dist):
  - drift episodes: Q1 mean=177.8 (range [144.6, 220.9])
  - near_miss episodes: Q1 mean=63.1 (range [-58.9, 207.9])
  - **同一 ~5cm 距离，critic 系统性高估导致 drift 的 chunks**
- **解读** (教训 #10, #11):
  - v4 的 place rate 增益来自 **TEMPORAL CONSISTENCY**（4-step 开环 chunks 帮助 episode 可靠 reach target），**非** 开环执行速度
  - 拆小 chunks 破坏时序一致性（+14 near_miss）但 **未减少 drift**
  - 推理时 chunk_size 操作无法修复 policy/critic 问题
  - Feynman "root cause vs proximate cause": v4.1 修了 proximate cause（开环放大），但 root cause（critic 时序偏差）需重训
- **Orchestrator 匹配 branch C** (place_rate < 0.72 → v4.3 threshold tuning)，但数据表明 threshold tuning 无效
- **决策点**: (a) 接受 v4 (71.9%) 为基线进入 P3 sim-to-real；(b) v4.2 critic 重训；(c) v5 Cal-QL
- **目录**: `outputs/iql_v4_1_adaptive/`

### IQL 训练指标对比表

| Metric | v1 (n_step=1) | v2 (n_step=5) | v3 (n_step=5+shaping) | v4 (chunking h=4) |
|--------|:---:|:---:|:---:|:---:|
| Q_gap | 102.25 | 137.22 (+34%) | 117.58 | 138.74 |
| Advantage gap | 1.08 ✅ | -0.65 ⚠️ INVERTED | 2.07 ✅ FIXED | — |
| Place rate (N=200) | 65.5% | 67.5% | 64.8% | **71.9%** |
| Drift (N=1000) | 34.6% | 34.4% | 27.1% | 37.5% (N=200) |

---

## Phase 5 — DT Orchestrator 决策树编排器（2026-07-15）

> Spec: `dt-orchestrator`（审查修订 v2）+ `harden-dt-router`
> 目标: 把"人工看报告→选下一步"升级为"离线穷举 + 在线决策路由"两阶段范式

### 架构: 两阶段决策树编排器
1. **离线码本构建**: 跨配置运行评估 → 收集 per-episode features + outcomes → 标注最优配置
2. **在线 per-episode 路由**: 前 20 步 v4 warmup 采集特征 → DT 预测最优配置 → 切换

### 7 维特征向量（前 20 place-phase 步内可观测）
| 特征 | 含义 |
|------|------|
| `dist_at_step20` | 第 20 步 block-target 距离 (cm) |
| `dist_change_rate` | 前 20 步距离变化率 (cm/step) |
| `dist_variance_early` | 前 20 步距离方差 |
| `early_drift_signal` | 前 20 步内 dist_change_rate < 0（远离目标） |
| `q1_at_step20` | 第 20 步 Q1 值（critic 置信度） |
| `best_dist_early` | 前 20 步最近距离 (cm) |
| `has_q_value` | Q 值是否可用（imputation 标记） |

### Reachability 预实验（P0 前置门控）
- **配置**: `--warmup_switch` 模式（前 20 步 v4 chunk_size=4 → 第 21 步起 v4.1 adaptive），N=200, seed=42
- **结果**: warmup_switch place_rate = 65.5%（聚合指标）
- **门控修正** (教训): warmup_switch 聚合 65.5% **不适用**于评估选择性切换效果。正确门控指标 = 可达上界 = (v4_placed ∪ ws_placed) / N = **87.0%** >> 71.9% → 门控通过
- **日志**: `outputs/dt_orchestrator/reachability.log`, `reachability_n200.json`

### DT Router v1 — 概念验证（76.3% / 75.5%）
- **配置**: codebook v1.0 (`config_set=["v4", "v4.1"]`), DT 模型 v1, `--dt_confidence=0.65`, warmup=20 steps
- **结果**: **76.3% place_rate (+4.4pp vs v4 71.9%)**（首次报告）；逐 episode 对齐后 75.5%
- **切换统计**: 20 selective switches — 4 successful rescues, 3 意外损失, 3 neutral, 10 both failed
- **CRITICAL 发现**: DT 模型最重要特征 `q1_at_step20` (importance=0.6364) 在在线评估中 **完全缺失**（0/199 有值）— `--dt_router` 模式未启用 Q-value 计算
- **特征重要性** (v1): 仅用距离特征仍达 +4.4pp
- **日志**: `outputs/dt_orchestrator/dt_router.log`, `dt_router_n200.json`

### Harden DT Router Spec — P0 修复（2026-07-15）
> Spec: `harden-dt-router` — 修复 train/serve skew + 标签错配 + 消融实验

**P0 问题 1: Train/Serve Feature Distribution Skew**
- 训练用 v4.1 轨迹特征，在线路由仅有 v4 warmup 特征 → 系统性预测错误
- **修复**: codebook features 一律用 warmup 配置 (v4) 提取，v4.1 JSON 仅用于取 outcome 标签

**P0 问题 2: 标签错配**
- codebook 用 v4.1 outcomes 做标签，但路由器实际切换到 warmup_switch（v4 warmup 20 + v4.1 续跑），两者 per-episode outcome 不完全一致
- **修复**: 新增 `dt_outcomes_parser.py` 从 `reachability.log` 解析 ws outcomes → codebook 标签源切换为 warmup_switch

**Task 1: Q-value 特征修复** ✅
- `--dt_router` 模式 warmup 阶段每步计算 IQL critic Q-value，step 20 取值
- Q-value 异常降级为 imputation (q1=0.0, has_q=0) + `q_value_warning` 日志
- 特征完整性断言: importance > 0.1 的特征非全零
- N=5 冒烟测试: `q1_at_step20=23.9455`（与训练数据一致）

**Task 2: Codebook v2.0 重建** ✅
- `dt_outcomes_parser.py` 解析 200 条 per-episode outcomes
- `dt_codebook.py --outcomes ws:path` 支持
- codebook v2.0: `config_set=["v4", "warmup_switch"]`, `version="2.0"`, 197 trainable
- DT 模型 v2: CV accuracy 58.47% (vs v1 54.04%, +4.43pp), ws recall 0.787 (vs v1 0.689, +0.098)
- **产物**: `codebook_v2.json`, `dt_model_v2.pkl`, `feature_importance_v2.json`, `training_report_v2.json`, `tree_structure_v2.txt`

**Task 3.0: 路由代码 Bug 修复** ✅ (CRITICAL)
- **Bug**: 路由代码检查 `predicted == "v4.1"`，但 v2 模型 classes = `["v4", "warmup_switch"]` → 0 switches
- **修复** (`evaluate_iql_env.py` ~line 594): 改为 `predicted != "v4"`（class-agnostic），`actual_config = predicted`
- 向后兼容: v1 模型 (classes `["v4","v4.1"]`) 仍可运行

**Task 3.1-3.5: 消融实验** ✅

| 实验组 | 配置 | place_rate | vs baseline |
|--------|------|:---:|:---:|
| A. pure-v4 | 仅 v4 | 71.5% | — |
| B. pure-v4.1 | 仅 v4.1 | 64.5% | -7.0pp |
| C-v1. DT-router v1 | Q-value 缺失 | 75.5% | +4.0pp |
| C-v2. DT-router v2 | Q-value 修复 + v2 模型 | 73.0% | +1.5pp |
| D. oracle | 完美路由上界 | 87.0% | +15.5pp |

- **v2 切换统计**: 42 switches (vs v1 20), 87 fallbacks
- **切换效果分析**: v1 的 3 例意外损失 (ep 102/191/197) **全部纠正**；v2 引入 2 例新损失 (ep 109/173)
- **净贡献**: C - max(A,B) = +1.5pp
- **改进潜力**: D - C = +14.0pp
- **差距分析**: v2 实际 73.0% < 模拟预测 76.5% — 切换 episode 仅 30/42 匹配 ws outcomes（在线 Q-value 计算引入非确定性）
- **日志**: `outputs/dt_orchestrator/dt_router_v2.log`, `dt_router_v2_n200.json`

### Task 4 — 置信度阈值调优（2026-07-15）
- **目标**: place_rate ≥ 76.3%, false positives ≤ 1
- **SubTask 4.1 阈值敏感性回放** (τ ∈ [0.50, 0.95]):
  - **置信度分布双峰**: 58 个 ws 预测 conf<0.65, **0 个在 [0.65, 0.80)**, 42 个 conf≥0.80
  - τ 从 0.65→0.80 **无变化**（双峰分布使中间区间无效）
  - 最佳 τ=0.95 给 73.4%（1 loss）
- **SubTask 4.3 非确定性根因分析** — **重大发现**:
  - **v1 place_rate 分解**: 143 baseline + 4 rescued - 3 lost + **12 non-det gain - 5 non-det loss** = 151
  - **v2 place_rate 分解**: 143 baseline + 4 rescued - 2 lost + **1 non-det gain - 0 non-det loss** = 146
  - v1 的 **+7pp 噪声贡献**来自运行间非确定性（17 个非切换 episode 不匹配 v4），**非 DT 路由贡献**
  - **v2 DT 路由真实贡献 = +2 episodes > v1 的 +1 episode**（v2 实际更优）
  - v2 非确定性极低（仅 1 个非切换不匹配），结果更可信
- **SubTask 4.4 结论**:
  - **76.3% 目标基于 v1 噪声膨胀结果，不现实**
  - v2@τ=0.65 为推荐配置（73.0%, DT 路由净贡献 +2, 3 例 v1 损失全纠正, 非确定性极低）
- **日志**: `outputs/dt_orchestrator/dt_router_v2*.log/json`

### Phase 6 — 可重复性诊断与多 seed 验证（2026-07-15，进行中）

**起因**: 用户引用 Colas et al. 《How Many Random Seeds?》指出深度 RL 的"可重复性危机"，要求对 v1 的 +7pp"噪声膨胀"做根因诊断，并按 P0→P1→统计→决策门路径推进。

#### P0 诊断: `diagnose_nondeterminism.py` ✅ 完成
- **4 层比对设计**: checkpoint 哈希 → 逐 episode outcome → dist 轨迹发散 → 源隔离
- **测试 1**: v4 N=20, seed=42, 重复 2 次 → **0 mismatch**, 两次均 85.0%
- **测试 2**: dt_router_v2 N=200, seed=42, 重复 2 次 → **0 mismatch**, 两次均 73.0%
- **关键结论**: 固定 seed 下评估**完全确定性**（evaluate_iql_env.py 仅设 `np.random.seed` + `env.seed`，未设任何 torch/cudnn 确定性标志，却仍 0 发散）
- **假设证伪**: 之前归因于"GPU 非确定性噪声膨胀"的 +7pp **实为代码版本差异**（v1 与 v2 之间的 code-path 不同），非随机噪声
- **影响**: P1 多 seed 测量的方差**仅来自 seed 方差**，排除 GPU 干扰；确定性 GPU 模式（原优先级 2）已无必要

#### P1 多 seed 评估: `multi_seed_eval.py` ✅ 完成
- **配对设计**: 同一 seed 同时跑 v4 和 dt_router_v2，降低组间方差
- **种子**: `[42, 123, 456, 789, 2024, 314, 271, 1618, 9999, 7777]`（10 个）
- **20 runs 全部完成**，总耗时 ~3.7 小时
- **日志**: `outputs/dt_orchestrator/multi_seed_{config}_seed{S}.log`

#### 统计分析 ✅ 完成（`analyze_multi_seed.py`）

**核心结果:**

| 配置 | mean ± std | median ± IQR | 95% CI (bootstrap) | CV |
|------|:---:|:---:|:---:|:---:|
| v4 | 68.2% ± 3.35pp | 68.5% ± 5.00pp | [66.2%, 70.2%] | 4.91% |
| dt_router_v2 | 68.2% ± 3.07pp | 67.5% ± 4.75pp | [66.5%, 70.1%] | 4.51% |

**配对比较 (10 common seeds):**
- DT Router v2 - v4: **-0.05pp ± 3.55pp**（中位数 +0.25pp）
- 配对 t 检验: t=-0.045, df=9, **p >> 0.05（不显著）**
- Cohen's d = -0.01（几乎为零的效应量）
- 95% CI of diff: [-2.15pp, +1.95pp]（跨越 0）
- CI 重叠: **YES**（v4 [66.2%, 70.2%] vs dt2 [66.5%, 70.1%] 完全重叠）

**三个关键结论:**
1. **v4 与 dt_router_v2 统计等价** — 之前单次 seed=42 看到的 +2 episodes (73.0% vs 71.5%) 完全落在 seed 噪声范围内，DT Router 路由**无真实统计增益**
2. **CV > 3% 阈值** — v4 CV=4.91%, dt_router_v2 CV=4.51%，seed 间方差达 10pp 跨度（v4: 62.5%~72.5%），**训练稳定性是首要瓶颈**
3. **seed=42 偏高** — 原报告的 v4 71.9% 和 dt_router_v2 73.0% 均为偏高种子，真实期望约 68.2%

#### 决策门结论 ✅
- **CV=4.91% > 3% → 优先优化训练稳定性**（触发）
- v2 vs v4 差异不显著（p>>0.05, diff≈0）→ DT Router 路由无统计增益
- v4 95% CI 下界 66.2% 远超 V59 的 56% → v4 仍为最强可重复基线
- **v4.2 Critic 重训优先级降低** — DT Router 本身无增益，Critic 偏差修复不会带来路由收益

---

### Harden DT Router Checklist 最终状态
- ✅ Task 1: Q-value 特征修复（199/199 episode 有真实值, mean=98.58）
- ✅ Task 2: Codebook v2.0 重建
- ✅ Task 3: 4 组消融 + 路由 bug 修复
- ✅ Task 4: 阈值调优 + 非确定性根因
- ✅ 向后兼容（不传 `--dt_router` 行为不变；v1 模型仍可运行）

---

## 关键技术洞察汇总（跨版本沉淀）

1. **Advantage ordering is critical** — advantage_success < advantage_failure（反转）时 AWR 给 failure actions 更高权重，退化策略（v2 教训）
2. **Q_gap alone is insufficient** — v2 最大 Q_gap (137) 但 advantage 排序最差（v3 教训）
3. **Drift is systemic** — ~34% failures across v1/v2 现 drift（"reach 3cm then drift to 20cm"），确定性开环行为非随机游走
4. **N=200 vs N=1000 discrepancy** — v2 在 N=200 +2pp 但 N=1000 回退。N=200 噪声显著，N=1000 更可靠
5. **PPO is destructive for pretrained policies** — V59 fine-tuning + RL-from-scratch 双重确认。在线 RL 无法改进此任务预训练解
6. **Training metrics ≠ environment performance** — v3 修复 advantage 反转但 place rate 反而下降。advantage 反转是症状非根因
7. **Reward shaping ceiling** — direction-aware dense shaping 达天花板，drift 模式跨架构持续
8. **Action chunking improves success but not drift** — v4 突破 70% 但 drift 占比反升（27.7%→37.5%）
9. **Drift resistant to all 3 intervention types** — reward shaping (v3) / hybrid policy / action chunking (v4) 均未减少 drift
10. **Temporal consistency is the key benefit of chunking** — v4.1 拆小 chunks 破坏时序一致性，证明 v4 增益来自 consistency 非开环速度
11. **Critic temporal bias is ROOT CAUSE of drift** — 同一 ~5cm 距离 drift Q1=177.8 vs near_miss Q1=63.1。critic 系统性高估导致 drift 的 chunks。需重训（v4.2/v5），非推理时 trick

### DT Router 专属洞察
- warmup_switch 聚合指标不适用于评估选择性切换；正确门控 = 可达上界 (v4_placed ∪ ws_placed)/N
- 缺失 Q-value 特征仍可 +4.4pp（距离特征单独贡献）
- v2 启用 Q-value 后 over-switching（42 vs 20）→ place rate 反降（在线 Q-value 计算引入非确定性）
- **非确定性噪声膨胀**是 RL 可重复性危机的核心痛点（参考 Colas et al. "How Many Random Seeds?"）

### 统计方法学洞察（Batch 1-3 沉淀）
- **"未拒绝 H₀" ≠ "证明 H₀ 为真"** — 配对 t 检验 p>>0.05 只表明"无法区分"，要声称"等价"必须做 TOST（预设 δ，两侧检验同时显著）
- **N=10 只能检测 +5pp** — RL 文献普遍 ≤5 seeds 已偏上，但对 +1–2pp 的小效应完全无分辨力（功效 0.18–0.39）。下次启动对比前**必须先做前瞻功效分析**
- **CV 阈值需绑定物理含义** — "3%"本身无意义，"3% @ 68% place_rate ≈ 2.05pp std ≈ 期望最小可检测效应的 1σ"才有工程意义
- **方差归因指导优化方向** — between-seed 50.9% / within-seed 49.1% 近乎相等，降 CV 需同时增加 episodes/seed AND 改善 env 采样多样性，单一手段不够
- **训练稳定性方案需区分两类** — "降低报告方差"（多 seed 取最佳=制度化 cherry-picking，集成=改部署形态）vs "降低真实方差"（LR 调度/batch/warmup/EMA-SWA），应优先后者
- **决策门必须代码化** — 人工判定"CV>3% 触发"易被绕过；6 条规则 (R1–R6) + 预注册 manifest 把工程门禁变成可追溯的硬约束

---

## 工程约定（写入 project_memory.md）

- **DT Router 决策树根节点**: `early_drift_signal`（前 20 步内是否出现 1→2 或 2→4 drift 切换）
- **子节点**: `q_value_skew` (drift_chunk Q1 / near_miss_chunk Q1) + `place_rate_rolling`（滑动窗口）
- **fallback 机制**: 遇未注册状态默认 v4 baseline + '未见状态计数器' 触发新离线穷举
- **特征来源约束**: codebook features 一律用 warmup 配置 (v4) 提取
- **Q-value 日志**: DT Router 在线评估必须启用 `--log_q_values`

---

## 当前决策状态（截至 2026-07-16，方法学加固后）

### 形式化决策门 VERDICT: **BLOCKED**
- **R1 TRIGGERED**: CV=4.91% > 3% → 需优化训练稳定性
- **R2 TRIGGERED**: TOST fail (t1=0.85, t2=-0.94, 均 ≤ tcrit=1.833) → 无法声明 v4 ≡ dt_router_v2 等价
- **R6 TRIGGERED**: v4 CI 下界 66.2% < 68% min → 数据不足以做决策声明
- 当前数据状态: 任何"DT Router 无增益"或"v4 ≡ dt_router_v2"的声明均**未通过形式化门控**

### 已确认事实（方法学加固后修正）
- **v4 (68.2% ± 3.35pp, 10 seeds)** = 当前最强可重复基线（true action chunking h=4），95% CI [66.2%, 70.2%]
- **v4.1 (64.5%, 单次)** = FAILED（adaptive chunk 回退）
- **DT Router v2 (68.2% ± 3.07pp, 10 seeds)** = 与 v4 **10 seed 下无法区分**（配对 t=-0.045, p>>0.05, Cohen's d=-0.01），但 **TOST 等价检验失败**（±1pp 不确定区间仍存在）
- **N=10 功效不足**: 对 +2pp 增益功效仅 0.39（<0.8 阈值），检测 +2pp 需 N=30，+1pp 需 N=150
- **Oracle 上界** = 87.0%（单次 seed=42 理论上界，改进潜力 +14pp）
- **Critic 时序偏差** = drift 根因（已确认，但 v4.2 重训优先级降低——DT Router 无增益）
- **固定 seed 完全确定性** = P0 诊断证明（v4 N=20 + dt_router_v2 N=200 均 0 mismatch）
- **训练稳定性瓶颈** = CV=4.91% > 3% 阈值（3% @ 68.2% ≈ 2.05pp std ≈ 期望最小可检测效应），**首要优化目标**
- **方差归因**: between-seed 50.9% / within-seed 49.1%（近乎相等，降 CV 需双管齐下）

### 76.3% 目标处置
- 76.3% 基于 v1 的单次运行结果，含 +7pp 代码版本差异（原误标为"噪声膨胀"）
- **P1 证伪**: v4 真实期望 68.2%（非 71.9%），dt_router_v2 真实期望 68.2%（非 73.0%）——seed=42 偏高
- 76.3% 在当前架构下**不可达**，需从根本上提升训练稳定性（降低 CV）而非调路由

### 长期方向优先级（方法学加固后最终更新）
1. ~~**多次运行取中位数**~~ — ✅ 已完成，结论: v4=dt_router_v2 10 seed 下无法区分
2. ~~**确定性 GPU 模式**~~ — ❌ 已由 P0 证伪不必要
3. **训练稳定性优化**（降低 CV<3%）— **当前最高优先级**
   - **前置**: 方差溯源实验（Part B, 45 configs, 4 维 seed 拆解）→ 精确定位 init/data/env/grad 哪一维贡献主要方差
   - **候选**: 学习率调度 / 更大 batch / warmup / 梯度裁剪 / 权重平均 (EMA/SWA)
   - **避免**: "多 seed 取最佳"（制度化 cherry-picking）、"集成"（改部署形态，非训练稳定性）
4. **稳定 CV<3% 后**: ≥30 seeds + TOST 复核 → 正式归档"DT Router 已证伪"
5. ~~**v4.2 Critic 重训**~~ — ⬇️ 优先级降低（DT Router 无增益，Critic 修复无路由收益）
6. **动态自适应阈值** — ⬇️ 优先级降低（DT Router 本身无增益）
7. **AdaStop 序贯检验**（arXiv:2306.10882）— 未来新配置对比时采用，避免"先跑完才发现功效不足"

### 建议产物
- `diagnose_nondeterminism.py` — 可复用诊断工具 ✅ 已交付
- `multi_seed_eval.py` — 多 seed 配对评估脚本 ✅ 已交付
- `analyze_multi_seed.py` — 统计分析 + 可视化 ✅ 已交付
- `outputs/dt_orchestrator/multi_seed_analysis.png` — 箱线图 + 配对折线图 ✅ 已生成

### Phase 6 续 — 方法学加固 (Batch 1-3, 2026-07-16)

**起因**: 用户提供 5 维方法学评议（引用 Colas et al. arXiv:1806.08295, AdaStop arXiv:2306.10882），指出五个缺口：
1. "统计等价"措辞过强 — 需 TOST 等价检验
2. 3% CV 阈值缺溯源 — 需物理含义论证
3. 10 seeds 功效不足 — 需前瞻功效分析
4. 训练稳定性方案混淆"报告方差" vs "真实方差" — 需方差溯源
5. 决策门人工触发 — 需代码化 + 预注册

#### Batch 1: TOST + 功效分析 + CV 溯源

**TOST 等价检验**（`analyze_multi_seed.py` 扩展 `tost_equivalence()`）：
- H0: |mean(diff)| ≥ δ（NOT equivalent）；H1: |mean(diff)| < δ（equivalent）
- δ = 1.0pp（预设等价边界）
- 结果: t1=0.85 (≤1.833, fail), t2=-0.94 (≥-1.833, fail) → **无法声明等价**
- **结论修正**: "v4 ≡ dt_router_v2" 从"统计等价"降级为"10 seed 下无法区分"，±1pp 不确定区间仍存在

**CV 阈值溯源**（写入决策门输出）：
- 3.0% CV @ 68.2% place_rate → 隐含 std ≈ 2.05pp
- 物理含义: 2.05pp ≈ DT Router 期望的 +1–2pp 增益 → 3% CV = "效应量 < 噪声 1σ"
- 此溯源把"拍脑袋阈值"升级为"与最小可检测效应量绑定的物理量"

**前瞻功效分析**（`bootstrap_power_analysis.py` 创建）：
- Monte Carlo 仿真: 用 10 个实测 diff (std≈3.55pp) 作噪声模型，2000 次重采样 + 注入效应 + 配对 t 检验

| 目标效应 | 所需 N (power=0.8) | 当前 N=10 的功效 |
|----------|:---:|:---:|
| +0.5pp | >200 | 0.13 |
| +1.0pp | 150 | 0.18 |
| +2.0pp | 30 | 0.39 |
| +3.0pp | 15 | 0.61 |
| +5.0pp | 10 | 0.98 |

- **关键发现**: N=10 只能可靠检测 +5pp；检测 +2pp 需 N=30；+1pp 需 N=150
- 当前 N=10 对 +2pp 功效仅 0.39（远低于 0.8）→ 不足以声称"DT Router 无增益"
- **产物**: `outputs/dt_orchestrator/power_analysis.json`

#### Batch 2: 方差溯源实验（`variance_attribution.py` 创建）

**Part A: 评估方差分解**（基于 10 个 v4 eval log × 200 episodes）：
- Between-seed 方差（env 初始状态分布）: **50.9%**
- Within-seed 方差（有限样本采样）: **49.1%**
- 两者近乎相等 → 降 CV 需同时增加 episodes/seed AND 改善 env 采样多样性
- N≈2158 episodes/seed 可达 within-seed std=1pp

**Part B: 训练方差实验设计**（4 维 seed 拆解）：
- 4 维度: `init_seed`（权重初始化）、`data_seed`（replay buffer 顺序）、`env_seed`（env 初始状态）、`grad_seed`（dropout/随机梯度）
- 实验矩阵: 45 configs（4 维 × 10 varied + 5 baseline）
- **前置依赖**: `train_iql.py` 需扩展支持 `--init_seed`、`--data_seed`、`--env_seed`、`--grad_seed`
- **未执行**: 需先修改训练脚本，预计运行 45 × ~30min = ~22 小时
- **产物**: `outputs/dt_orchestrator/variance_attribution.json`、`training_variance_design.json`

#### Batch 3: 预注册 + 形式化决策门（`preregister_and_validate.py` 创建）

**预注册机制**（两模式）：
- `init` 模式: 跑前生成 manifest JSON（expected_effect、equivalence_delta、target_power、anchor_config）
- `validate` 模式: 跑后自动比对结果 vs manifest，输出每条规则触发状态 + 最终 VERDICT

**形式化决策门（6 条规则）**:
```
R1: cv > cv_threshold           → TRIGGER stability_optimization
R2: not tost_equivalent         → CANNOT declare equivalence
R3: tost_equivalent & not sig   → DECLARE equivalent
R4: significant & diff > 0      → treatment better, proceed
R5: significant & diff < 0      → anchor better, accept anchor
R6: anchor_ci_lower < min       → insufficient, need more seeds
```

**回溯验证（dt_router_v2 vs v4, 10 seeds）**:
- R1: TRIGGERED（CV=4.91% > 3%）
- R2: TRIGGERED（TOST fail，无法声明等价）
- R6: TRIGGERED（v4 CI 下界 66.2% < 68% min）
- **FINAL VERDICT: BLOCKED** — 当前数据不足以做出任何决策声明，需优先优化训练稳定性
- **产物**: `outputs/dt_orchestrator/experiment_manifest_dt_router_v2_vs_v4_retro.json`

#### 方法学加固产物清单

| 工具 | 文件 | 用途 |
|------|------|------|
| TOST 等价检验 | `analyze_multi_seed.py` (扩展) | 闭环"统计等价"声明 |
| 前瞻功效分析 | `bootstrap_power_analysis.py` (新) | 估算检测 +Xpp 所需 N |
| 评估方差分解 | `variance_attribution.py` Part A | between/within-seed 归因 |
| 训练方差实验设计 | `variance_attribution.py` Part B | 4 维 seed 拆解矩阵 |
| 预注册 + 决策门 | `preregister_and_validate.py` (新) | 6 规则代码触发 + manifest |

#### 签收记录（2026-07-16）

**决策门规则集版本**: `v1.0`（6 规则 R1–R6，定义见上）。未来规则增删时版本递增，保证回溯可审计。

**形式化签收结论**:
- 五个缺口：**全部闭环**
- 当前结论"v4 与 dt_router_v2 在 10 seed 下无法区分"：**站得住**
- "DT Router 无真实增益"：统计上留 ±1pp 不确定区间，已用 TOST fail + R2 正式标注，BLOCKED 状态正确
- 解锁条件：CV<3%（稳定性优化）→ ≥30 seeds + TOST 复核 → 归档
- 形式化决策门 VERDICT=BLOCKED 已取代人工判定

**Batch 1-3 交付签收通过，进入 Phase 7（稳定性优化 + 四维方差拆解）**

**方法论评议补充洞察（用户签收时提供）**:
- TOST 降级是结论性质改变（阳性声明→阴性声明），在功效不足时阴性声明天然不可靠
- CV=4.91% 意味噪声 std≈3.34pp 已超过期望增益本身 → "噪声已淹没信号"，这也是 TOST 必然 fail 的根因
- 当噪声 σ > 效应量 δ 时，等价检验功效本身不可能高
- N=10 对 +1–3pp 区间（DT Router 声称范围）从设计阶段就不具备定论能力，"跑完才发现不够"本应在设计阶段拦截

---

## Phase 7 — 训练稳定性优化与训练 CV 测量（2026-07-16 启动）

> 目标: 降 CV<3% → 解锁 BLOCKED → ≥30 seeds + TOST 复核归档
> 前置: `train_iql.py` 加 seed 控制 + probe states Q/V 诊断

### Step 1a: Seed 控制基础设施（2026-07-16 完成）

**4D→2D 降维**: 代码审查发现 `train_iql.py` 原先无任何 seed 控制，且 IQL 离线训练无 env 交互、无 dropout、CPU 设备 → 4D（init/data/env/grad）降为 2D（init_seed × data_seed）。

**实现**:
- `set_global_seed()`: torch.manual_seed + np.random.seed + random.seed 三连调用，支持 `--init_seed` / `--data_seed` 解耦（Option B 预留接口）
- `sample_probe_states()`: 从 D_expert.npz 用 seed=0 采样 100 个 in-distribution state 作为 Q/V 诊断探针
- `compute_qv_diagnostics()`: 在 probe states 上计算 Q1/Q2/V 的 mean±std + q1_q2_gap_mean（双 Q 一致性指标）
- `compute_init_hash()`: 训练前 Q/V 输出的 MD5 hash，用于 dry-run seed 控制验证
- `phase7_train_cv.py`: 编排器，含 SeedSequence(0).spawn(10) → dry-run 验证 → 10-seed 训练+评估 → 决策门分析

**Dry-run 验证通过**:
- seed 独立性: 3757552657→`9bcbc4b4d4c7` ≠ 673228719→`c05d89921838` ✓
- seed 可复现性: 3757552657 重跑→相同 hash ✓

**文件**: [train_iql.py](file:///home/w/vla_workspace/train_iql.py), [phase7_train_cv.py](file:///home/w/vla_workspace/phase7_train_cv.py)
**日志**: `outputs/phase7_variance_decomposition/dry_run/`

### Step 1b: 10-seed 训练 CV 测量（2026-07-17 完成）

**配置**: v4 冻结配置（n_epochs=100, batch_size=256, n_step=5, chunk_size=4, tau=0.7, beta=3.0, gamma=0.99, lr=3e-4, polyak=0.005, oversample_dist=0.20,0.40, oversample_factor=3）。10 seeds via SeedSequence(0).spawn(10)。每 seed 训练后用 eval_seed=42 / N=200 episodes 评估。

**结果**:

| Seed | place_rate | drift | near_miss | Q1Q2gap |
|------|:---------:|:-----:|:---------:|:-------:|
| 3757552657 | 46.5% | 53 | 54 | 13.6 |
| 673228719 | 62.5% | 21 | 54 | 9.7 |
| 3241444873 | 68.0% | 27 | 37 | 9.5 |
| 3685993406 | 60.0% | 34 | 46 | 10.8 |
| 1216546553 | 62.5% | 39 | 36 | 12.5 |
| 2078861726 | 57.0% | 36 | 49 | 4.1 |
| 2471122328 | 66.0% | 27 | 41 | 9.9 |
| 23012616 | 68.0% | 26 | 38 | 22.0 |
| 3031610183 | 63.5% | 25 | 48 | 10.2 |
| 2976135721 | 35.0% | 83 | 47 | 23.3 |

**统计**:
- place_rate: mean=58.9%, std=10.5pp, **CV=17.8%**（远超 8% 阈值）
- Q/V CV: q1_mean=36.0%, q2_mean=39.1%, v_mean=51.7%, q1_q2_gap=46.7%
- **决策门输出: `FIX_STABILITY`**（训练方差主导 + Q/V 价值不稳定副规则也触发）

**关键发现**:
1. **训练方差是系统瓶颈**: CV=17.8% 远超评估方差 CV=4.91%，且远超 8% 阈值
2. **Q/V 价值函数极不稳定**: v_mean CV=51.7%（跨 seed V 值差异巨大），q1_q2_gap CV=46.7%
3. **原 v4 baseline 68.2% 是分布上端**: 仅 2/10 seeds 达到 68.0%，均值 58.9% 低 9.3pp
4. **存在灾难性 seed**: seed 2976135721=35.0%（83 drifts），seed 3757552657=46.5%（53 drifts）
5. **Q1Q2gap 与 place_rate 无可靠相关**: 22.0→68.0%, 4.1→57.0%, 23.3→35.0% — 双 Q 分歧不是策略质量预测因子

**文件**: `outputs/phase7_variance_decomposition/training_cv_results.json`, `training_cv_analysis.json`
**日志**: `outputs/phase7_variance_decomposition/full_run_master.log`, `train_seed*.log`, `eval_seed*.log`

### Step 2: 稳定性优化（消融式 A/B）

**设计原则**（用户补充建议 2）: 消融而非打包上线。每个手段单独 A/B（每边 ≥10 seeds），用 TOST 判定"该手段是否带来可检测的 CV 下降"。

**候选手段**（降低真实方差类）:
- LR 调度（cosine / linear warmup）
- 更大 batch
- Warmup
- 梯度裁剪
- 权重平均（EMA/SWA）

**避免**（降低报告方差类）: 多 seed 取最佳（制度化 cherry-picking）、集成（改部署形态）

### Step 3: 解锁复核

**触发条件**: CV<3% 达成
**执行**: ≥30 seeds + TOST（δ=1.0pp）复核 v4 vs dt_router_v2
**归档条件**: TOST pass (R3) → 正式归档"DT Router 已证伪"

### AdaStop 引入时机（用户补充建议 3）

- **不用于**当前 v4 vs dt_router_v2 复核（已有 10 seed 数据，直接扩到 30 即可）
- **用于**下一轮新配置对比（如稳定性优化后的 v5 vs v4）——序贯设计能在实验进行中判定是否足够，省掉无效 seed 投入

---

## 版本索引表

| 版本 | 日期 | place_rate | 状态 | 关键产物 |
|------|------|:---:|------|------|
| Setup | 06-14~06-22 | — | ✅ | MuJoCo 选定, 抓取物理修复 |
| V5–V58 | 06-21~07-09 | ≤54% | ✅ 归档 | place_policy_v{N}/ |
| V59 | 07-10 | **56%** | ✅ 生产基线 | place_policy_v59/best_hier/ |
| V60–V71b | 07-10~07-12 | ≤56% | ❌ 7 方法族穷尽 | decision_tree.mmd |
| RL-from-scratch v1 | 07-13~14 | 0% | ❌ PPO 破坏性 | rl_from_scratch_v1/ |
| IQL v1 (n_step=1) | 07-14 | 65.5% | ✅ 超越 V59 | iql_v1/ |
| IQL v2 (n_step=5) | 07-14 | 67.5% | ⚠️ adv 反转 | iql_v2_qchunk/ |
| IQL v2 hybrid | 07-14 | 66.5% | ❌ 假设证伪 | — |
| IQL v3 (shaping) | 07-14 | 64.8% | ❌ 无提升 | iql_v3_shaping/ |
| IQL v4 (chunking h=4) | 07-15 | **71.9%** | ✅ 突破 70% | iql_v4_chunking/ |
| IQL v4.1 (adaptive) | 07-15 | 64.5% | ❌ REGRESSION | iql_v4_1_adaptive/ |
| DT Router v1 | 07-15 | 75.5%* (seed=42) | ⚠️ 含代码路径差异 | dt_router_n200.json |
| DT Router v2 | 07-15 | 73.0% (seed=42) / 68.2%±3.1 (10 seeds) | ⚠️ 与 v4 10 seed 无法区分（TOST fail） | dt_router_v2_n200.json |
| Oracle 上界 | 07-15 | 87.0% | — 理论上界 | ablation_analysis.py |
| P0 确定性诊断 | 07-15 | — | ✅ 0 mismatch | nondet_*.json |
| P1 多 seed 评估 | 07-15 | v4 68.2%±3.4 / dt2 68.2%±3.1 | ✅ 10 seed 无法区分 | multi_seed_results.json |
| 方法学加固 Batch 1-3 | 07-16 | — | ✅ TOST+功效+方差溯源+决策门（已签收，决策门 v1.0）| power_analysis.json, variance_attribution.json, experiment_manifest_*.json |
| Phase 7 Step 1 | 07-16~17 | train CV=17.8% (mean 58.9%±10.5) | ✅ Step 1 完成 → FIX_STABILITY | training_cv_results.json, training_cv_analysis.json |

\* v1 的 75.5% 含 +7pp 代码版本差异（原误标为"GPU 非确定性噪声膨胀"，P0 诊断证伪），真实 DT 贡献仅 +1 episode

---

## 文件组织约定

每个 IQL 实验目录含:
- `README.md` — 配置、结果、文件清单、复现命令
- `training.log` — 训练 stdout
- `training_results.json` — 完整指标历史
- `final_model.pt` — 最终 checkpoint
- `best_q_gap.pt` — 最佳 Q-gap checkpoint
- `checkpoint_epoch_{50,100}.pt` — 中间 checkpoint
- `env_eval_*.json` / `env_eval_*.log` — 评估结果与日志
- `experiment_registry.yaml`（v4.1+）— Orchestrator 决策输入

### 关键脚本（workspace root）
| 文件 | 用途 |
|------|------|
| `train_iql.py` | IQL 训练（`--n_step`, `--oversample_dist`, `--reward_shaping`）|
| `evaluate_iql_env.py` | 环境评估（`--hybrid_checkpoint`, `--chunk_size`, `--adaptive_chunk`, `--log_q_values`, `--warmup_switch`, `--dt_router`）|
| `orchestrator.py` | L1 自动化: 读 YAML+JSON 匹配决策树 |
| `iql_agent.py` | IQL agent (V, Q1/Q2, GaussianPolicy) |
| `iql_dataset.py` | 离线数据集 + reward shaping |
| `analyze_drift_physics.py` | P1 drift 物理根因调查 |
| `dt_feature_extractor.py` | 7 维特征提取（v4 JSON）|
| `dt_codebook.py` | 跨配置码本构建 |
| `dt_trainer.py` | sklearn DecisionTreeClassifier 训练 |
| `dt_outcomes_parser.py` | reachability.log → ws_outcomes.json |
| `diagnose_nondeterminism.py` | P0 确定性诊断（4 层比对）|
| `multi_seed_eval.py` | 多 seed 配对评估（增量保存 + resume）|
| `analyze_multi_seed.py` | 统计分析（含 TOST + CV 溯源）|
| `bootstrap_power_analysis.py` | 前瞻功效分析（Monte Carlo 仿真）|
| `variance_attribution.py` | 评估方差分解 + 训练方差实验设计 |
| `preregister_and_validate.py` | 预注册 manifest + 6 规则形式化决策门 |
| `data/D_expert.npz` | 专家演示数据集 |

---

*本 changelog 基于工作空间实际文件、IQL_EXPERIMENTS.md、experiment_registry.yaml、specs/、memory/ 综合生成。后续版本应按相同格式追加。*
