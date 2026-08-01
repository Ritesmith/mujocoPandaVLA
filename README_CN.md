# MuJoCo Panda VLA — 机器人抓取-放置任务的离线强化学习（IQL）

**[English](README.md)** | 中文版

基于 MuJoCo 物理仿真环境的 Franka Panda 机械臂抓取-放置任务，采用离线强化学习（Implicit Q-Learning）+ 分层策略架构，在严格的预注册统计协议下进行评估。

---

## 动机

本项目探索**离线强化学习**能否从固定的专家演示数据集中，无需训练时的在线环境交互，生成可靠的 Panda 机械臂抓取-放置策略。

两个具体问题塑造了这项工作：

1. **Expectile 回归的训练不稳定性。** IQL 的价值函数通过 expectile 回归拟合，当 `tau > 0.5` 时损失函数非凸，存在多个 V 网络解。实践中表现为灾难性的 seed 方差（CV ≈ 18%），淹没了所有下游对比。本项目 Phase 7 诊断并解决了这个问题：通过将 `tau` 从 0.7 降到 0.5，恢复 MSE 式凸性，使放置成功率 CV 下降约 62%。

2. **强化学习评估的统计危机。** 强化学习结果对 seed 选择和小样本量极其敏感，却常被报告为单 seed 点估计。本项目将评估视为一等公民的统计问题：每个性能声明在数据采集前**预注册**，通过形式化的决策门（R1–R6）验证，并基于 N=30 seeds × 200 episodes 的 TOST 等价检验、置信区间、Cohen's d 和 bootstrap 功效分析支撑。

核心论点：**严格的统计验证不是 RL 评估的可选装饰——它是区分真实增益与噪声的唯一方式。** Phase 8 DT Router 实验（候选增益 +0.75pp）就是警世寓言：PILOT 看起来有希望，但在 N=30 时未能达到显著性，最终归档。

---

## 架构

```
┌──────────────────┐    ┌─────────────────────────┐    ┌──────────────────────────────┐    ┌──────────────────┐
│  离线数据集       │───▶│  IQL 训练 (tau=0.5)    │───▶│  分层策略                     │───▶│  MuJoCo 评估    │
│  iql_dataset.py  │    │  iql_agent.py           │    │  (抓取子策略 + 放置            │    │  evaluate_iql_   │
│  D_expert.npz    │    │  train_iql.py           │    │   子策略, chunk_size=4)       │    │  env.py          │
└──────────────────┘    └─────────────────────────┘    └──────────────────────────────┘    └──────────────────┘
   专家演示              expectile V + AWR             两阶段分解动作选择                 200 episodes ×
   (训练时无环境         策略提取,                      基于动作块的                      30 seeds, 配对
    交互)                n_step=5 returns                                                设计
```

- **离线数据集（`iql_dataset.py`）** — 加载 `data/D_expert.npz`（专家演示）。训练期间无环境交互；智能体仅从固定缓冲区学习，使用 n-step returns（`n_step=5`）。
- **IQL 训练（`iql_agent.py`, `train_iql.py`）** — Implicit Q-Learning：expectile 回归的价值网络（`tau=0.5`）、双 Q 网络、Advantage-Weighted Regression（AWR, `beta=3.0`）策略提取，`gamma=0.99`。Expectile 为 0.5 时退化为 MSE，消除了 V 网络多解病态。
- **分层策略（`gym_env/`）** — 两阶段抓取-放置分解。放置子策略发出大小为 4 的动作块（`chunk_size=4`），降低决策频率并平滑控制。
- **MuJoCo 评估（`evaluate_iql_env.py`）** — Panda MuJoCo 环境中的 rollout，每 seed 200 episodes，跨 30 seeds 配对进行方差控制对比。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 激活项目环境
conda activate vla

# 3. 训练 IQL agent（最终基线：tau=0.5，无正则）
python -m core.train_iql --tau 0.5 --beta 3.0 --gamma 0.99 --n_step 5 --chunk_size 4

# 4. 评估 checkpoint（200 episodes）
python -m core.evaluate_iql_env --checkpoint <path_to_model.pt> --n_episodes 200
```

训练 checkpoint 默认值可通过 `--output_dir`（训练）和 `--checkpoint` / `--seed`（评估）覆盖。完整的多 seed 统计流程见 [统计框架](#统计框架) 段落。

---

## 结果

| 方法 | 放置成功率 | 备注 |
|------|-----------|------|
| BC（warmstart） | ~22% | 协变量偏移限制；纯行为克隆无法从专家演示泛化 |
| IQL（tau=0.7，旧） | 58.9% ± 10.5pp | CV=17.83%，不稳定 — 灾难性 seed 方差，V 网络多解 |
| **IQL（tau=0.5，最终）** | **59.82% ± 3.45pp** | **CV=5.77%，稳定** — N=30 seeds × 200 episodes，无正则 |

**最终基线配置：** `IQLAgent(tau=0.5, beta=3.0, gamma=0.99, n_step=5, chunk_size=4)`（无正则）。

### Phase 7 — 训练稳定性（tau 调优）
- **问题：** 在 `tau=0.7` 时，放置成功率 CV 达到 17.83%，V 均值 CV 达到 51.7%，由 expectile 回归在 0.5 以上非凸性驱动。这种 seed 方差使任何下游 A/B 对比统计上不可检测。
- **解决方案：** 将 expectile `tau` 从 0.7 降到 0.5，expectile 损失退化为 MSE，恢复唯一的 V 解。首先尝试了 EMA 和 Huber loss 消融（Round 1），但未解决根因；tau 扫描（Round 2a）成功。
- **结果：** 放置成功率 CV **17.83% → 6.78%（−62%）**，V 均值 CV 51.7% → 30.5%。V L2 正则化（Round 2b）测试为阴性结果并拒绝。见 [CHANGELOG_PHASE7.md](CHANGELOG_PHASE7.md)。

### Phase 8 — DT Router 重训 & V_CV=30% 验证
- **问题：** Phase 7 后 V 均值 CV 仍在 ~30%，这种残存的价值函数方差是否会泄露到下游任务性能？
- **回答：** 否。三个监测节点（端到端放置成功率 CV、drift、near-miss）全部 **PASS** 8% 阈值 — 端到端 CV=5.01%，确认系统吸收了残存 V 方差。V_CV=30% 是"纸老虎"。
- **DT Router v3：** 在新 tau=0.5 基线上重训但归档 — +0.75pp 效应统计不显著（p > 0.05，CV accuracy 54.4% ≈ 随机）。tau=0.5 v4 配置确立为最终交付基线。见 [CHANGELOG_PHASE8.md](CHANGELOG_PHASE8.md)。

---

## 统计框架

本项目中的每个性能声明在报告前都通过形式化的预注册决策门。流程：

1. **预注册**（`preregister_and_validate.py --init`）— 在接触评估数据前锁定假设、样本量和接受准则。
2. **功效分析**（`bootstrap_power_analysis.py`）— 基于 bootstrap 的分析，确认计划的 N 对检测最小关注效应（如 +2pp @ N=30）有足够功效。
3. **多 seed 配对评估**（`multi_seed_eval.py`）— 跨 30 seeds × 200 episodes 的配对 rollout 设计，控制 seed 方差。
4. **统计分析**（`analyze_multi_seed.py`）— 报告均值 ± 标准差、置信区间、配对 t 检验、Cohen's d 和 **TOST**（双单边检验）用于等价性。
5. **决策门**（`preregister_and_validate.py --validate`）— 应用规则 **R1–R6**（如 CV ≤ 8%、TOST 在等价界内、功效达标）产生 PASS/FAIL 判定。

这个框架区分了真实的 Phase 7 胜利（tau 调优，CV −62%）和虚幻的 Phase 8 DT Router 增益（归档为不显著）。完整叙事：

- [CHANGELOG_PHASE7.md](CHANGELOG_PHASE7.md) — 训练稳定性、tau 扫描、V L2 阴性结果
- [CHANGELOG_PHASE8.md](CHANGELOG_PHASE8.md) — DT Router 重训、三节点 V_CV=30% 验证

---

## 项目结构

```
├── core/                        # 核心 IQL：agent, dataset, 训练, 评估
│   ├── iql_agent.py             # IQL agent: expectile V + twin Q + AWR policy
│   ├── iql_dataset.py           # 离线数据集加载器（n-step returns）
│   ├── train_iql.py             # 训练入口
│   ├── evaluate_iql_env.py      # MuJoCo 评估（200 episodes, 配对设计）
│   ├── evaluate_iql_policy.py   # IQL 策略评估工具
│   ├── hierarchical_policy.py   # 抓取 + 放置两阶段分解
│   ├── eval_hierarchical.py     # 分层策略评估
│   ├── train_place_policy.py    # 放置子策略训练
│   ├── train_dapg.py            # DAPG/PPO 训练
│   └── pretrained_cnn.py        # 预训练 ResNet 特征提取器
│
├── analysis/                    # 统计分析与验证
│   ├── preregister_and_validate.py  # 预注册 + R1-R6 决策门
│   ├── analyze_multi_seed.py    # 多 seed 统计：TOST, CI, Cohen's d
│   ├── bootstrap_power_analysis.py  # 实验规划的功效分析
│   ├── bootstrap_cv_ci.py / bca_bootstrap_cv.py  # Bootstrap CV 置信区间
│   ├── variance_attribution.py  # 方差归属
│   ├── multi_seed_eval.py       # 多 seed 配对评估运行器
│   ├── phase7_*.py              # Phase 7: 训练稳定性工具（tau 扫描, TOST）
│   ├── phase8_*.py              # Phase 8: DT Router 重训 + 三节点监测
│   └── automate_phase7_vl2.py   # Phase 7 自动化
│
├── data/                        # 数据收集脚本（*.npz 数据文件已 gitignore）
│   ├── collect_expert_demos.py  # 专家演示收集
│   ├── collect_dagger_data.py   # DAgger 数据收集
│   ├── collect_d_fail.py / collect_rejection_sampling.py / collect_successful_trajectories.py
│   ├── dagger_oracle.py         # DAgger oracle
│   ├── cache_resnet_features.py # 缓存 ResNet 特征
│   └── make_dcsil_stochastic.py
│
├── models/                      # 策略训练 & 模型变体
│   ├── train_bc_only.py / train_bc_expert.py / train_shallow_bc.py
│   ├── train_csil_plus_plus.py  # CSIL++ PBRS
│   ├── train_diffusion_policy.py / diffusion_policy_model.py
│   ├── train_online_dagger.py / train_policy_distillation.py / train_rl_from_scratch.py
│   ├── backbone_probe.py / ensemble_predict.py
│   └── eval_bc_vs_v59.py / orchestrator.py
│
├── diagnostics/                 # 诊断 & 失败分析
│   ├── diagnose_nondeterminism.py / diagnose_reward_density.py
│   ├── diagnose_v59_sensitivity.py / diagnose_ensemble_weights.py
│   ├── analyze_drift_physics.py / failure_mode_clustering.py
│   └── bench_env_speed.py / voronoi_partition.py
│
├── dt_router/                   # Phase 8 DT Router（已归档 — 效应不显著）
│   ├── dt_codebook.py / dt_trainer.py
│   ├── dt_feature_extractor.py / dt_outcomes_parser.py
│   └── dt_overlap_analysis.py
│
├── pipeline_scripts/            # 历史迭代脚本（非活跃代码）
│   ├── run_v63_pipeline.py … run_v71b_pipeline.py
│   └── run_pipeline_common.py
│
├── scripts/                     # Shell pipeline 运行器（.sh）
│   ├── run_tau0.5_N30.sh / run_csil_plus_plus_pipeline.sh
│   └── run_dfail_bc_pipeline.sh / run_voronoi_pipeline.sh
│
├── docs/                        # 文档
│   ├── SIMULATOR_EVAL.md
│   └── solution_summary.md
│
├── gym_env/                     # Panda MuJoCo 环境（留于根目录）
├── tests/                       # 测试套件
├── CHANGELOG.md                 # 完整项目历史（V5–V59 演化）
├── CHANGELOG_PHASE7.md          # Phase 7: tau 调优 & 稳定性
├── CHANGELOG_PHASE8.md          # Phase 8: DT Router & V_CV 验证
└── requirements.txt             # Python 依赖
```

> **注意：** 目录重组后，所有模块位于包下（`core/`、`analysis/`、`data/`、`models/`、`diagnostics/`、`dt_router/`、`pipeline_scripts/`）。从仓库根目录以模块形式运行，如 `python -m core.train_iql`。`pipeline_scripts/run_v*.py`（v63–v71b）是 PPO 时代的历史迭代脚本，保留用于复现，**不**属于活跃 IQL 代码库。`dt_router/` 文件是 Phase 8 DT Router 实验的归档，其效应未达显著性，非活跃代码。

---

## 相关工作

- **IQL** — Kostrikov, Nair, Levine. *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR 2022. 核心算法：expectile 回归的价值函数 + 优势加权回归，无需策略约束或 OOD 动作抑制。
- **RT-1 / RT-2** — Brohan et al. (Google DeepMind). 真实机器人操作视觉-语言-动作模型。RT-2 展示了 VLM 可协同微调以发出机器人动作，实现语言条件控制。
- **OpenVLA** — Kim et al. 基于 Prismatic VLM backbone 的开源视觉-语言-动作模型，旨在拓宽 VLA 研究的可及性。

> **范围说明：** 本项目聚焦于 **RL 方法论** — 离线 IQL 训练稳定性与严格的统计验证 — 而非视觉-语言对齐。见 [局限性](#局限性) 对名称中"VLA"涵盖范围的诚实声明。

---

## 局限性

我们诚实地声明以下局限：

1. **"VLA" 属性有限。** 本项目**未**实现语言指令理解或视觉-语言对齐。没有 CLIP 编码器，没有 BERT/语言头，没有 VLM。仓库名称中的"VLA"指 *视觉到动作* 管道（图像/状态特征 → 动作），**而非**语言条件策略。更准确的名称应为"视觉到动作离线 RL"。

2. **单一任务。** 仅演示抓取-放置。未评估跨任务泛化（如推动、堆叠、插入）。分层抓取 + 放置分解是任务特定的。

3. **仅仿真。** 所有训练和评估在 MuJoCo 中进行。未进行仿真到真实迁移实验；域随机化和真实硬件验证超出范围。

4. **放置成功率上限（~60%）。** 最终策略稳定在 ~59.82% 放置成功率。这主要受**离线数据集质量**（专家演示覆盖、BC 协变量偏移）限制，而非 IQL 架构 — BC warmstart 的 ~22% 和 IQL 从相同数据恢复到 ~60% 即为证据。要弥合到数据集 ~68.5% 专家放置率的差距，可能需要在线微调或数据集扩展，均超出范围。

---

## GitHub 仓库配置

托管此仓库的建议设置：

- **描述：** `基于 MuJoCo 的 Panda 抓取-放置任务离线强化学习（IQL），配严格统计验证`
- **主题：** `reinforcement-learning`, `offline-rl`, `iql`, `mujoco`, `robotics`, `panda`, `pick-and-place`

这些准确反映了项目对离线 RL 方法论和统计严谨性的聚焦，提升了搜索 IQL 或 MuJoCo 操作基线的研究者的可发现性。