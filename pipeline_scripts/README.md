# Pipeline Scripts (Historical Iteration Records)

These scripts are historical iteration pipeline runners from the development
process (v63–v71b). They are **NOT** active code and are kept for
reproducibility and historical reference.

Each script encapsulates one iteration of an RL/PPO reward-shaping and
training experiment, recording the diagnostic context, hypothesis, and the
single-variable change applied in that version.

## Script Inventory

| Filename | Version | Brief Description | Status |
|----------|---------|-------------------|--------|
| `run_v63_pipeline.py` | V63 | Reward fix validation (v12 gating + early release penalty) | archived |
| `run_v64_pipeline.py` | V64 | Distance penalty normalization (v13) | archived |
| `run_v65_pipeline.py` | V65 | Distance penalty coefficient tuning (v14, k=1.0) | archived |
| `run_v66_pipeline.py` | V66 | Gate lowering bonus on XY proximity (v15) | archived |
| `run_v67_pipeline.py` | V67 | Reward structure overhaul (v16) | archived |
| `run_v68_pipeline.py` | V68 | Hover penalty (v17): first truly hack-free reward | archived |
| `run_v69_pipeline.py` | V69 | Disable image augmentation (single-variable from V68) | archived |
| `run_v70_pipeline.py` | V70 | Freeze entire ResNet backbone (single-variable from V68) | archived |
| `run_v71a_pipeline.py` | V71a | Smaller rollout buffer n_steps=512 (single-variable from V68) | archived |
| `run_v71b_pipeline.py` | V71b | Tighter KL constraint target_kl=0.01 (single-variable from V68) | archived |

## Note

The final production pipeline is `train_iql.py` + `evaluate_iql_env.py`.
These `run_v*.py` scripts were used during iterative development to explore
different configurations (reward shaping, backbone freezing, augmentation,
rollout buffer size, KL constraints, etc.) and are retained only as a
record of the experimentation that led to the production setup.
