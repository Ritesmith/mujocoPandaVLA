# MuJoCo Panda VLA — Offline RL (IQL) for Robotic Pick-and-Place

**[中文版](README_CN.md)** | English

Offline Reinforcement Learning (Implicit Q-Learning) with a hierarchical grasp + place policy for the Franka Panda arm in MuJoCo simulation, evaluated under a rigorous pre-registered statistical protocol.

---

## Motivation

This project investigates whether **offline RL** can produce a reliable pick-and-place policy for the Panda manipulator from a fixed dataset of demonstrations, without any online environment interaction during training.

Two concrete problems shape the work:

1. **Training instability in expectile regression.** IQL's value function is fit by expectile regression, which becomes non-convex for `tau > 0.5` and admits multiple V-network solutions. In practice this manifested as catastrophic seed variance (CV ≈ 18%) that swamped any downstream comparison. Phase 7 of this project diagnoses and resolves this by lowering `tau` from 0.7 to 0.5, restoring MSE-like convexity and cutting the place-rate CV by ~62%.

2. **The evaluation crisis in RL.** Reinforcement-learning results are notoriously sensitive to seed choice and small sample sizes, yet are frequently reported as single-seed point estimates. This project treats evaluation as a first-class statistical problem: every performance claim is **pre-registered** before data collection, validated through a formal decision gate (R1–R6), and backed by TOST equivalence tests, confidence intervals, Cohen's d, and bootstrap power analysis over N=30 seeds × 200 episodes.

The central thesis is that **rigorous statistical validation is not optional ornamentation for RL evaluation — it is the only way to distinguish a real improvement from noise.** The Phase 8 DT Router experiment (a candidate +0.75pp gain) is the cautionary tale: it looked promising in pilot, then failed to reach significance at N=30 and was archived.

---

## Architecture

```
┌──────────────────┐    ┌─────────────────────────┐    ┌──────────────────────────────┐    ┌──────────────────┐
│  Offline Dataset │───▶│  IQL Training (tau=0.5) │───▶│  Hierarchical Policy         │───▶│  MuJoCo Eval     │
│  iql_dataset.py  │    │  iql_agent.py           │    │  (Grasp sub-policy + Place    │    │  evaluate_iql_   │
│  D_expert.npz    │    │  train_iql.py           │    │   sub-policy, chunk_size=4)  │    │  env.py          │
└──────────────────┘    └─────────────────────────┘    └──────────────────────────────┘    └──────────────────┘
   expert demos            expectile V + AWR             two-stage decomposed action         200 episodes ×
   (no env interact-       policy extraction,            selection over action chunks        30 seeds, paired
    ion during training)   n_step=5 returns                                                  design
```

- **Offline Dataset (`iql_dataset.py`)** — Loads `data/D_expert.npz` (expert demonstrations). No environment interaction occurs during training; the agent learns purely from this fixed buffer with n-step returns (`n_step=5`).
- **IQL Training (`iql_agent.py`, `train_iql.py`)** — Implicit Q-Learning: an expectile-regressed value network (`tau=0.5`), twin Q-networks, and Advantage-Weighted Regression (AWR, `beta=3.0`) policy extraction with `gamma=0.99`. The expectile at 0.5 degenerates to MSE, eliminating the V-network multi-solution pathology.
- **Hierarchical Policy (`gym_env/`)** — A two-stage grasp-then-place decomposition. The place sub-policy emits action chunks of size 4 (`chunk_size=4`), reducing decision frequency and smoothing control.
- **MuJoCo Eval (`evaluate_iql_env.py`)** — Rollouts in the Panda MuJoCo environment, 200 episodes per seed, paired across 30 seeds for variance-controlled comparison.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Activate the project environment
conda activate vla

# 3. Train the IQL agent (final baseline: tau=0.5, no regularization)
python -m core.train_iql --tau 0.5 --beta 3.0 --gamma 0.99 --n_step 5 --chunk_size 4

# 4. Evaluate a checkpoint (200 episodes)
python -m core.evaluate_iql_env --checkpoint <path_to_model.pt> --n_episodes 200
```

The trained checkpoint defaults can be overridden via `--output_dir` (training) and `--checkpoint` / `--seed` (evaluation). For the full multi-seed statistical pipeline, see the [Statistical Framework](#statistical-framework) section.

---

## Results

| Method | Place Rate | Notes |
|--------|-----------|-------|
| BC (warmstart) | ~22% | Covariate-shift limitation; behavior cloning alone fails to generalize from expert demos |
| IQL (tau=0.7, old) | 58.9% ± 10.5pp | CV=17.83%, unstable — catastrophic seed variance, V-network multi-solution |
| **IQL (tau=0.5, final)** | **59.82% ± 3.45pp** | **CV=5.77%, stable** — N=30 seeds × 200 episodes, no regularization |

**Final baseline config:** `IQLAgent(tau=0.5, beta=3.0, gamma=0.99, n_step=5, chunk_size=4)` (no regularization).

### Phase 7 — Training Stability (tau tuning)
- **Problem:** At `tau=0.7`, place-rate CV reached 17.83% and V-mean CV reached 51.7%, driven by the non-convexity of expectile regression above 0.5. This seed variance made any downstream A/B comparison statistically undetectable.
- **Solution:** Lower the expectile `tau` from 0.7 to 0.5, which degenerates the expectile loss to MSE and restores a unique V solution. EMA and Huber-loss ablations (Round 1) were tried first but did not address the root cause; the tau sweep (Round 2a) did.
- **Result:** place-rate CV **17.83% → 6.78% (−62%)**, V-mean CV 51.7% → 30.5%. V-L2 regularization (Round 2b) was tested as a negative result and rejected. See [CHANGELOG_PHASE7.md](CHANGELOG_PHASE7.md).

### Phase 8 — DT Router Retraining & V_CV=30% Validation
- **Question:** With V-mean CV still at ~30% after Phase 7, does this residual value-function variance leak into downstream task performance?
- **Answer:** No. Three monitoring nodes (end-to-end place-rate CV, drift, and near-miss) all **PASS** the 8% threshold — end-to-end CV=5.01%, confirming the system absorbs the residual V-variance. V_CV=30% is a "paper tiger."
- **DT Router v3:** Retrained on the new tau=0.5 baseline but archived — the +0.75pp effect was not statistically significant (p > 0.05, CV accuracy 54.4% ≈ chance). The tau=0.5 v4 config was established as the final delivery baseline. See [CHANGELOG_PHASE8.md](CHANGELOG_PHASE8.md).

---

## Statistical Framework

Every performance claim in this project passes a formal, pre-registered decision gate before it is reported. The workflow is:

1. **Pre-register** (`preregister_and_validate.py --init`) — lock hypotheses, sample size, and acceptance criteria *before* touching evaluation data.
2. **Power analysis** (`bootstrap_power_analysis.py`) — bootstrap-based analysis to confirm the planned N has adequate power to detect the smallest effect of interest (e.g., +2pp @ N=30).
3. **Multi-seed paired eval** (`multi_seed_eval.py`) — paired rollout design across 30 seeds × 200 episodes to control for seed variance.
4. **Statistical analysis** (`analyze_multi_seed.py`) — reports mean ± std, confidence intervals, paired t-test, Cohen's d, and **TOST** (Two One-Sided Tests) for equivalence.
5. **Decision gate** (`preregister_and_validate.py --validate`) — applies rules **R1–R6** (e.g., CV ≤ 8%, TOST within equivalence bounds, power met) to produce a PASS/FAIL verdict.

This framework is what separated the real Phase 7 win (tau tuning, CV −62%) from the illusory Phase 8 DT Router gain (archived as non-significant). Full narratives:

- [CHANGELOG_PHASE7.md](CHANGELOG_PHASE7.md) — training stability, tau sweep, V-L2 negative result
- [CHANGELOG_PHASE8.md](CHANGELOG_PHASE8.md) — DT Router retraining, three-node V_CV=30% validation

---

## Project Structure

```
├── core/                        # Core IQL: agent, dataset, training, evaluation
│   ├── iql_agent.py             # IQL agent: expectile V + twin Q + AWR policy
│   ├── iql_dataset.py           # Offline dataset loader (n-step returns)
│   ├── train_iql.py             # Training entry point
│   ├── evaluate_iql_env.py      # MuJoCo evaluation (200 episodes, paired design)
│   ├── evaluate_iql_policy.py   # IQL policy evaluation utilities
│   ├── hierarchical_policy.py   # Grasp + Place two-stage decomposition
│   ├── eval_hierarchical.py     # Hierarchical policy evaluation
│   ├── train_place_policy.py    # Place sub-policy training
│   ├── train_dapg.py            # DAPG/PPO training
│   └── pretrained_cnn.py        # Pretrained ResNet feature extractor
│
├── analysis/                    # Statistical analysis & validation
│   ├── preregister_and_validate.py  # Pre-registration + R1-R6 decision gate
│   ├── analyze_multi_seed.py    # Multi-seed stats: TOST, CI, Cohen's d
│   ├── bootstrap_power_analysis.py  # Power analysis for experiment planning
│   ├── bootstrap_cv_ci.py / bca_bootstrap_cv.py  # Bootstrap CV confidence intervals
│   ├── variance_attribution.py  # Variance attribution
│   ├── multi_seed_eval.py       # Multi-seed paired evaluation runner
│   ├── phase7_*.py              # Phase 7: training-stability tooling (tau sweep, TOST)
│   ├── phase8_*.py              # Phase 8: DT Router retraining + 3-node monitoring
│   └── automate_phase7_vl2.py   # Phase 7 automation
│
├── data/                        # Data collection scripts (*.npz data files are gitignored)
│   ├── collect_expert_demos.py  # Expert demonstration collection
│   ├── collect_dagger_data.py   # DAgger data collection
│   ├── collect_d_fail.py / collect_rejection_sampling.py / collect_successful_trajectories.py
│   ├── dagger_oracle.py         # DAgger oracle
│   ├── cache_resnet_features.py # Cache ResNet features
│   └── make_dcsil_stochastic.py
│
├── models/                      # Policy training & model variants
│   ├── train_bc_only.py / train_bc_expert.py / train_shallow_bc.py
│   ├── train_csil_plus_plus.py  # CSIL++ PBRS
│   ├── train_diffusion_policy.py / diffusion_policy_model.py
│   ├── train_online_dagger.py / train_policy_distillation.py / train_rl_from_scratch.py
│   ├── backbone_probe.py / ensemble_predict.py
│   └── eval_bc_vs_v59.py / orchestrator.py
│
├── diagnostics/                 # Diagnostics & failure analysis
│   ├── diagnose_nondeterminism.py / diagnose_reward_density.py
│   ├── diagnose_v59_sensitivity.py / diagnose_ensemble_weights.py
│   ├── analyze_drift_physics.py / failure_mode_clustering.py
│   └── bench_env_speed.py / voronoi_partition.py
│
├── dt_router/                   # Phase 8 DT Router (archived — effect not significant)
│   ├── dt_codebook.py / dt_trainer.py
│   ├── dt_feature_extractor.py / dt_outcomes_parser.py
│   └── dt_overlap_analysis.py
│
├── pipeline_scripts/            # Historical iteration scripts (NOT active code)
│   ├── run_v63_pipeline.py … run_v71b_pipeline.py
│   └── run_pipeline_common.py
│
├── scripts/                     # Shell pipeline runners (.sh)
│   ├── run_tau0.5_N30.sh / run_csil_plus_plus_pipeline.sh
│   └── run_dfail_bc_pipeline.sh / run_voronoi_pipeline.sh
│
├── docs/                        # Documentation
│   ├── SIMULATOR_EVAL.md
│   └── solution_summary.md
│
├── gym_env/                     # Panda MuJoCo environment (stays in root)
├── tests/                       # Test suite
├── CHANGELOG.md                 # Full project history (V5–V59 evolution)
├── CHANGELOG_PHASE7.md          # Phase 7: tau tuning & stability
├── CHANGELOG_PHASE8.md          # Phase 8: DT Router & V_CV validation
└── requirements.txt             # Python dependencies
```

> **Note:** After the directory reorganization, all modules live under packages (`core/`, `analysis/`, `data/`, `models/`, `diagnostics/`, `dt_router/`, `pipeline_scripts/`). Run them as modules from the repo root, e.g. `python -m core.train_iql`. `pipeline_scripts/run_v*.py` (v63–v71b) are historical PPO-era iteration scripts retained for reproducibility and are **not** part of the active IQL codebase. `dt_router/` files are archived DT Router experiments from Phase 8 whose effect did not reach significance and are not active.

---

## Related Work

- **IQL** — Kostrikov, Nair, Levine. *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR 2022. The core algorithm: expectile-regressed value function + advantage-weighted regression, avoiding the need for policy constraints or OOD-action suppression.
- **RT-1 / RT-2** — Brohan et al. (Google DeepMind). Vision-Language-Action models for real-robot manipulation. RT-2 demonstrates that VLMs can be co-fine-tuned to emit robot actions, enabling language-conditioned control.
- **OpenVLA** — Kim et al. An open-source Vision-Language-Action model built on a Prismatic VLM backbone, released to broaden access to VLA research.

> **Scope note:** This project focuses on the **RL methodology** — offline IQL training stability and rigorous statistical validation — not on vision-language alignment. See [Limitations](#limitations) for an honest declaration of what the "VLA" in the name does and does not cover.

---

## Limitations

We declare the following limitations honestly:

1. **The "VLA" attribute is limited.** This project does **not** implement language-instruction understanding or vision-language alignment. There is no CLIP encoder, no BERT/language head, and no VLM. The "VLA" in the repository name refers to the *vision-to-action* pipeline (image/state features → action), **not** a language-conditioned policy. A more precise name would be "vision-to-action offline RL."

2. **Single task.** Only pick-and-place is demonstrated. No cross-task generalization (e.g., pushing, stacking, insertion) has been evaluated. The hierarchical grasp + place decomposition is task-specific.

3. **Simulation only.** All training and evaluation are in MuJoCo. No sim-to-real transfer experiments have been conducted; domain-randomization and real-hardware validation are out of scope.

4. **Place rate ceiling (~60%).** The final policy stabilizes at ~59.82% place rate. This is bounded primarily by **offline dataset quality** (expert-demonstration coverage, covariate shift from BC), not by the IQL architecture — as evidenced by the BC warmstart's ~22% and IQL's recovery to ~60% from the same data. Closing the gap to the dataset's ~68.5% expert place rate would likely require online fine-tuning or dataset expansion, neither of which is in scope.

---

## GitHub Repository Configuration

Suggested settings for hosting this repository:

- **Description:** `Offline RL (IQL) with rigorous statistical validation for Panda pick-and-place in MuJoCo`
- **Topics:** `reinforcement-learning`, `offline-rl`, `iql`, `mujoco`, `robotics`, `panda`, `pick-and-place`

These accurately reflect the project's focus on offline RL methodology and statistical rigor, and improve discoverability for researchers searching for IQL or MuJoCo manipulation baselines.
