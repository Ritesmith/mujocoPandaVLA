# MuJoCo Panda VLA Workspace

Offline RL (IQL) + Hierarchical Policy for Panda pick-and-place in MuJoCo.

## Quick Start

```bash
# Environment
conda activate vla

# Train IQL agent (final baseline: tau=0.5)
python train_iql.py --tau 0.5 --beta 3.0 --gamma 0.99 --n_step 5 --chunk_size 4

# Evaluate
python evaluate_iql_env.py --checkpoint <path_to_model.pt> --n_episodes 200
```

## Final Baseline

**Config**: `IQLAgent(tau=0.5, beta=3.0, gamma=0.99, n_step=5, chunk_size=4)` (no regularization)

**Performance**: place_rate = 59.82% ± 3.45pp (CV=5.77%, N=30 seeds × 200 episodes)

## Project Structure

```
├── iql_agent.py              # IQL agent (expectile regression + AWR)
├── iql_dataset.py            # Offline dataset loader
├── train_iql.py              # Training entry
├── evaluate_iql_env.py       # Evaluation with MuJoCo env
├── gym_env/                  # Panda MuJoCo environment
├── dt_*.py                   # DT Router (Decision Tree router, archived)
├── phase7_*.py               # Phase 7: training stability tools
├── phase8_*.py               # Phase 8: DT Router retraining + V_CV validation
├── preregister_and_validate.py  # Pre-registration + decision gate (R1-R6)
├── analyze_multi_seed.py     # Multi-seed statistical analysis (TOST, CI, Cohen's d)
├── bootstrap_power_analysis.py  # Power analysis for experiment planning
└── multi_seed_eval.py        # Multi-seed paired evaluation runner
```

## Key Findings

### Phase 7: Training Stability (tau=0.5)
- **Problem**: Training CV=17.83% (catastrophic seed variance, V-network multi-solution)
- **Solution**: Lower expectile tau 0.7→0.5 (restores MSE-like convexity)
- **Result**: CV 17.83%→6.78% (-62%), V_mean CV 51.7%→30.5%
- See [CHANGELOG_PHASE7.md](CHANGELOG_PHASE7.md)

### Phase 8: DT Router + V_CV Validation
- **Question**: Does V_CV=30% affect downstream task performance?
- **Answer**: No — three monitoring nodes all PASS (end-to-end CV=5.01% ≤ 8%)
- **DT Router v3**: Archived (effect not significant, +0.75pp p>0.05, CV accuracy 54.4%)
- See [CHANGELOG_PHASE8.md](CHANGELOG_PHASE8.md)

## Methodology

All performance claims pass the formal decision gate:
1. **Pre-register** (`preregister_and_validate.py --init`) before experiment
2. **Multi-seed eval** (`multi_seed_eval.py`) with paired design
3. **Statistical analysis** (`analyze_multi_seed.py`): mean±std, CI, t-test, Cohen's d, TOST
4. **Power analysis** (`bootstrap_power_analysis.py`) before committing to N
5. **Decision gate** (`preregister_and_validate.py --validate`): R1-R6 rules

## License

Private.
