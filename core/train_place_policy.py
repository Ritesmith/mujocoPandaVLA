#!/usr/bin/env python3
"""DAPG-PPO training for an independent place policy.

The place policy starts from an already-grasped state (place_mode=True)
and learns to move the block toward the target and release it. Only
placing rewards are used (reward_type="place_only").

The place_mode (hard-attached block) is used because v2 (hard-attached)
outperformed v3 (realistic physics) in hierarchical evaluation
(best dist 10.6cm vs 17.1cm). A release constraint in the environment
gates gripper opening to only when block_target_dist < 0.10m, preventing
the premature release that caused 0% place rate.

Demo data is filtered to the place phase: transitions after the block
has been lifted more than 5 cm above the table
(lift_height = obs[10] - 0.22 > 0.05, where obs[10] = block_z).

Usage:
    python train_place_policy.py --demos demos/panda_pickplace.npz \
        --total_timesteps 200000
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import functools
import pickle
import sys
import shutil
import subprocess
import re

import gymnasium
import numpy as np
import torch as th
import gym_env
from gym_env.wrappers import FlattenObs
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# DAPGPPO is defined in train_dapg.py. A dedicated dapg_ppo.py module does
# not exist yet; importing from train_dapg runs only harmless module-level
# setup (env vars + imports) since main() is guarded by __main__.
from core.train_dapg import DAPGPPO


GRASP_STATES_PATH = "/home/w/vla_workspace/outputs/grasp_states_500.pkl"


def linear_schedule(initial_lr, final_lr=0.0):
    """Linear learning rate schedule.

    Returns a callable that decays the learning rate linearly from
    ``initial_lr`` (at training start, progress_remaining=1) to
    ``final_lr`` (at training end, progress_remaining=0). SB3 calls the
    schedule with ``progress_remaining`` (1 -> 0 over training).

    Use final_lr > 0 to avoid the late-stage LR -> 0 cliff that can
    freeze policy updates before convergence (V49b postmortem).
    """
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


def cosine_schedule(initial_lr, final_lr=0.0):
    """Cosine annealing learning rate schedule.

    Decays LR from ``initial_lr`` (progress_remaining=1) to ``final_lr``
    (progress_remaining=0) following a cosine curve. Smoother than linear
    near both endpoints: gentle early decay preserves learning speed,
    flat late decay avoids the LR cliff that can freeze policy updates.

    V59: chosen over linear for fine-tuning because the policy is already
    near-converged (V58 54%); cosine's slow start avoids premature LR
    reduction while its smooth tail prevents late-stage instability.
    """
    def schedule(progress_remaining: float) -> float:
        # progress_remaining: 1 -> 0 over training. Cosine from 0 -> pi.
        cosine = 0.5 * (1 + np.cos(np.pi * (1 - progress_remaining)))
        return final_lr + (initial_lr - final_lr) * cosine
    return schedule


def freeze_bn_running_stats(model):
    """Freeze BatchNorm running statistics during fine-tuning.

    V59 root cause: PPO's train() calls set_training_mode(True), which puts
    all BN layers in train mode. Each mini-batch forward pass then updates
    running_mean/running_var via exponential moving average. When the
    training data distribution (164 grasp states) differs from the
    pre-training distribution (65 states), the running stats drift —
    corrupting feature extraction at eval time (BN uses running stats in
    eval mode). V59 lost 19 place_rate points this way (54%->35%) in a
    single PPO update (321 mini-batches, max 21.5%% relative drift in
    running_mean).

    Fix: override each BN layer's train() to always stay in eval mode.
    This freezes running_mean/running_var while still allowing the weight
    parameters (gamma/beta) to receive gradients and be updated by the
    optimizer. Standard practice for fine-tuning pretrained feature
    extractors (He et al. 2019, "Bag of Tricks for Image Classification").
    """
    bn_types = (th.nn.BatchNorm1d, th.nn.BatchNorm2d, th.nn.BatchNorm3d)
    frozen_count = 0
    for module in model.policy.modules():
        if isinstance(module, bn_types):
            module.eval()
            original_train = module.train
            module.train = lambda mode=False, _orig=original_train: _orig(False)
            frozen_count += 1
    return frozen_count


def freeze_backbone_params(model):
    """Freeze ALL parameters in the features extractor (ResNet backbone).

    Unlike freeze_bn_running_stats (which only freezes BN running stats but
    allows weight updates), this sets requires_grad=False for the ENTIRE
    feature extractor — conv weights, BN weights (gamma/beta), and layer4.
    PPO can only update the MLP head (policy_net, value_net).

    V68 diagnostic: the first PPO update (step 2048) destroyed pretrained
    V59 visual features, crashing place_rate from 50% to 5%. Freezing the
    backbone prevents PPO from modifying the pretrained ResNet-18 features,
    forcing it to learn only the MLP head mappings. This isolates whether
    the destruction is in the feature extractor or the MLP head.
    """
    fe = model.policy.features_extractor
    total_params = sum(p.numel() for p in fe.parameters())
    trainable_before = sum(p.numel() for p in fe.parameters() if p.requires_grad)
    for param in fe.parameters():
        param.requires_grad = False
    trainable_after = sum(p.numel() for p in fe.parameters() if p.requires_grad)
    print(f"Backbone FROZEN: {total_params} total params, "
          f"{trainable_before} -> {trainable_after} trainable "
          f"(delta={trainable_before - trainable_after})")
    return total_params, trainable_before - trainable_after


class SaveVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats periodically so models can be evaluated
    mid-training (the final vec_normalize.pkl is only saved at the end)."""

    def __init__(self, save_path, save_freq=10000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self.model.env.save(self.save_path)
        return True


class SaveVecNormalizeOnBest(BaseCallback):
    """Save VecNormalize stats when a new best model is found.

    This is passed as callback_on_new_best to EvalCallback, fixing the
    vec_normalize/best_model mismatch issue: previously, best_model.zip
    could be from step 90K while vec_normalize.pkl was from step 150K+,
    causing evaluation with mismatched normalization stats.
    """

    def __init__(self, vec_normalize_path, verbose=0):
        super().__init__(verbose)
        self.vec_normalize_path = vec_normalize_path

    def _on_step(self):
        os.makedirs(os.path.dirname(self.vec_normalize_path), exist_ok=True)
        self.model.env.save(self.vec_normalize_path)
        if self.verbose > 0:
            print(f"Saved vec_normalize.pkl alongside best_model to "
                  f"{self.vec_normalize_path}")
        return True


class ClipRangeScheduleCallback(BaseCallback):
    """Linearly decay clip_range from `initial` to `final` over training.

    PPO's fixed clip_range=0.2 is appropriate early in training when the
    policy is far from optimal and ratio r_t(theta) is broadly distributed.
    As the policy converges, r_t concentrates near 1.0, and a single bad
    rollout batch can push r_t past 1.2, triggering clip saturation and
    a cascade of biased advantage estimates. This is the classic PPO
    long-training instability (Engstrom et al. 2020).

    Decaying clip_range from 0.2 to 0.05 tightens the trust region as
    the policy matures, preventing the post-convergence cascade. Paired
    with LR decay, this stabilises late-stage training.

    V49b postmortem: clip_fraction rose from 0.12 (60k) to 0.20 (130k)
    and eval reward collapsed from -863 to -4625. This callback addresses
    that failure mode directly.
    """

    def __init__(self, initial_clip=0.2, final_clip=0.05,
                 total_timesteps=200000, log_freq=10000, verbose=0):
        super().__init__(verbose)
        self.initial_clip = initial_clip
        self.final_clip = final_clip
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        self._last_log = -1

    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.total_timesteps)
        current_clip = self.initial_clip - (self.initial_clip - self.final_clip) * progress

        # SB3 PPO stores clip_range as a callable schedule (it converts
        # float inputs to a constant function in __init__). PPO.train()
        # then calls self.clip_range(progress_remaining) each update.
        # We must replace it with a callable, not a float, or PPO will
        # crash trying to call a float. We close over current_clip so
        # the next rollout uses the decayed value.
        self.model.clip_range = lambda _: float(current_clip)

        if self.verbose > 0 and self.num_timesteps - self._last_log >= self.log_freq:
            print(f"[SCHEDULE] step={self.num_timesteps} "
                  f"clip_range={current_clip:.4f} "
                  f"(progress={progress:.1%})")
            self._last_log = self.num_timesteps
        return True


class PBRSDiagnosticsCallback(BaseCallback):
    """Log raw_reward, shaping_reward and potential running means.

    Healthy PBRS shows:
      - shaping_reward -> 0 as the policy converges (potential absorbed)
      - raw_reward tracks true task progress (place_success, distance)
      - potential -> 0 (block reaches target)

    If shaping_reward stays large and positive while raw_reward stays
    flat, the potential function is mis-specified and being exploited.
    """

    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._raw_rewards = []
        self._shaping_rewards = []
        self._potentials = []

    def _on_step(self):
        # SB3 PPO collects n_steps rollout before each update. infos is
        # a list of (n_envs,) per step. We accumulate raw/shaping/potential
        # from the underlying VecEnv's info dicts.
        infos = self.locals.get("infos", [])
        for info in infos:
            # SB3 may wrap infos in a list per env during rollout
            if isinstance(info, list):
                info = info[0] if info else {}
            if not isinstance(info, dict):
                continue
            if "raw_reward" in info:
                self._raw_rewards.append(float(info["raw_reward"]))
            if "shaping_reward" in info:
                self._shaping_rewards.append(float(info["shaping_reward"]))
            if "potential" in info:
                self._potentials.append(float(info["potential"]))

        if self.n_calls % self.log_freq == 0 and (
            self._raw_rewards or self._shaping_rewards
        ):
            raw_mean = (sum(self._raw_rewards) / len(self._raw_rewards)
                        if self._raw_rewards else 0.0)
            shaping_mean = (sum(self._shaping_rewards) / len(self._shaping_rewards)
                            if self._shaping_rewards else 0.0)
            pot_mean = (sum(self._potentials) / len(self._potentials)
                        if self._potentials else 0.0)
            print(f"[PBRS] step={self.num_timesteps} "
                  f"raw_r={raw_mean:+.4f} "
                  f"shape_r={shaping_mean:+.4f} "
                  f"phi={pot_mean:+.4f} "
                  f"(n={len(self._shaping_rewards)})")
            # Reset accumulators so each log window is a fresh mean
            self._raw_rewards.clear()
            self._shaping_rewards.clear()
            self._potentials.clear()
        return True


class HierPlaceRateCallback(BaseCallback):
    """Evaluate policy with hierarchical eval (real physics) and save best.

    Fixes the V55 EvalCallback bug: EvalCallback selects 'best' based on
    place_mode eval reward (hard-attached block), which does NOT correlate
    with the true hierarchical place_rate (real physics). This caused V55's
    40% model (at 16k) to be overwritten by a 14% model (at 100k) that had
    higher place_mode reward.

    This callback runs eval_hierarchical.py as a subprocess every
    eval_freq steps, parses the TRUE place_rate, and saves the model to
    best_hier/ when a new best is found.

    Three-directory isolation:
      - best/        : EvalCallback (place_mode, for early stopping)
      - best_hier/   : This callback (hier place_rate, TRUE best)
      - checkpoints/ : RollingCheckpointCallback (keep_last=N)
    """

    def __init__(self, eval_freq, n_episodes, grasp_model, grasp_vecnorm,
                 target_pos_range, save_path,
                 early_stop_threshold=0, early_stop_consecutive=2,
                 first_eval_floor=0, decoupling_detection=False,
                 eval_log_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_episodes = n_episodes
        self.grasp_model = grasp_model
        self.grasp_vecnorm = grasp_vecnorm
        self.target_pos_range = target_pos_range
        self.best_hier_path = os.path.join(save_path, 'best_hier')
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_consecutive = early_stop_consecutive
        self.first_eval_floor = first_eval_floor
        self.decoupling_detection = decoupling_detection
        self.eval_log_path = eval_log_path or os.path.join(
            save_path, 'eval_logs', 'evaluations.npz')
        self.best_place_rate = -1
        self.best_step = 0
        self.consecutive_low = 0
        self.eval_history = []
        self._first_eval_done = False
        self._prev_place_mode_reward = None
        self._prev_hier_place_rate = None
        self._last_grab_rate = 0
        self._last_switch_rate = 0
        self._last_mean_lift = 0.0

    def _on_step(self):
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        step = self.num_timesteps
        os.makedirs(self.best_hier_path, exist_ok=True)
        temp_model = os.path.join(self.best_hier_path, 'temp_model.zip')
        temp_vecnorm = os.path.join(self.best_hier_path, 'temp_vecnorm.pkl')

        self.model.save(temp_model)
        vec_norm = self._get_vec_normalize()
        if vec_norm is not None:
            vec_norm.save(temp_vecnorm)

        place_rate = self._run_hier_eval(temp_model, temp_vecnorm)
        print(f"[HIER_EVAL] step={step} place_rate={place_rate}% "
              f"grab={self._last_grab_rate}% switch={self._last_switch_rate}% "
              f"lift={self._last_mean_lift:.1f}cm "
              f"(best={self.best_place_rate}% at {self.best_step})")
        self.eval_history.append({'step': step, 'place_rate': place_rate})

        if place_rate > self.best_place_rate:
            self.best_place_rate = place_rate
            self.best_step = step
            shutil.copy(temp_model, os.path.join(self.best_hier_path, 'best_model.zip'))
            shutil.copy(temp_vecnorm, os.path.join(self.best_hier_path, 'vec_normalize.pkl'))
            print(f"[HIER_EVAL] *** New best: {place_rate}% at {step} ***")
            self.consecutive_low = 0
        else:
            if self.early_stop_threshold > 0 and place_rate < self.early_stop_threshold:
                self.consecutive_low += 1
                print(f"[HIER_EVAL] WARNING place_rate={place_rate}% < "
                      f"{self.early_stop_threshold}% "
                      f"(consecutive_low={self.consecutive_low}/"
                      f"{self.early_stop_consecutive})")
                if self.consecutive_low >= self.early_stop_consecutive:
                    print(f"[HIER_EVAL] EARLY STOP triggered! "
                          f"Best: {self.best_place_rate}% at {self.best_step}")
                    return False
            else:
                self.consecutive_low = 0

        if not self._first_eval_done:
            self._first_eval_done = True
            if self.first_eval_floor > 0 and place_rate < self.first_eval_floor:
                print(f"[HIER_EVAL] FIRST EVAL FLOOR: {place_rate}% < "
                      f"{self.first_eval_floor}% — stopping immediately "
                      f"(start model may have degraded)")
                return False

        if self.decoupling_detection:
            current_reward = self._get_place_mode_reward()
            if (current_reward is not None
                    and self._prev_place_mode_reward is not None
                    and self._prev_hier_place_rate is not None):
                reward_improving = current_reward > self._prev_place_mode_reward
                hier_declining = (place_rate
                                  < self._prev_hier_place_rate * 0.9)
                if reward_improving and hier_declining:
                    print(f"[HIER_EVAL] DECOUPLING DETECTED: "
                          f"place_mode reward {self._prev_place_mode_reward:.1f} "
                          f"-> {current_reward:.1f} (improving) but "
                          f"hier place_rate {self._prev_hier_place_rate}% "
                          f"-> {place_rate}% (declining >10%)")
                    print(f"[HIER_EVAL] EARLY STOP triggered "
                          f"(reward-policy decoupling)! "
                          f"Best: {self.best_place_rate}% at {self.best_step}")
                    return False
            self._prev_place_mode_reward = current_reward
            self._prev_hier_place_rate = place_rate

        if os.path.exists(temp_model):
            os.remove(temp_model)
        if os.path.exists(temp_vecnorm):
            os.remove(temp_vecnorm)

        return True

    def _get_place_mode_reward(self):
        """Read the latest place_mode eval reward from evaluations.npz.

        EvalCallback writes results to this file after each eval. We use it
        to detect reward-policy decoupling: if place_mode reward improves
        while hier place_rate degrades, the policy is finding place_mode
        'shortcuts' that fail in real physics.
        """
        if not os.path.exists(self.eval_log_path):
            return None
        try:
            data = np.load(self.eval_log_path, allow_pickle=False)
            if data['results'].shape[0] == 0:
                return None
            return float(np.mean(data['results'][-1]))
        except Exception:
            return None

    def _run_hier_eval(self, model_path, vecnorm_path):
        cmd = [
            sys.executable, "-u", "eval_hierarchical.py",
            "--place_model", model_path,
            "--place_vecnorm", vecnorm_path,
            "--grasp_model", self.grasp_model,
            "--grasp_vecnorm", self.grasp_vecnorm,
            "--vision_mode", "--no_domain_randomize",
            "--target_pos_range", self.target_pos_range,
            "--n_episodes", str(self.n_episodes),
        ]
        cwd = os.path.dirname(os.path.abspath(__file__))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=cwd, timeout=900,
            )
            stdout = result.stdout
            match = re.search(
                r'Place \(dist<\d+cm\)\s*:\s*\d+/\d+\s*\((\d+)%\)',
                stdout)
            if match:
                place_rate = int(match.group(1))
                grab_match = re.search(
                    r'Grab\s+\(lift>\d+cm\)\s*:\s*\d+/\d+\s*\((\d+)%\)',
                    stdout)
                switch_match = re.search(
                    r'Phase switches \(grasp->place\):\s*(\d+)/(\d+)',
                    stdout)
                lift_match = re.search(
                    r'Mean max lift\s*:\s*([\d.]+)\s*cm',
                    stdout)
                self._last_grab_rate = (int(grab_match.group(1))
                                        if grab_match else 0)
                if switch_match:
                    self._last_switch_rate = int(int(switch_match.group(1))
                                                 / int(switch_match.group(2))
                                                         * 100)
                else:
                    self._last_switch_rate = 0
                self._last_mean_lift = (float(lift_match.group(1))
                                        if lift_match else 0.0)
                return place_rate
            print(f"[HIER_EVAL] Parse failed. tail stdout: "
                  f"{stdout[-300:]}")
            return 0
        except subprocess.TimeoutExpired:
            print(f"[HIER_EVAL] Eval timed out after 900s")
            return 0
        except Exception as e:
            print(f"[HIER_EVAL] Eval failed: {e}")
            return 0

    def _get_vec_normalize(self):
        env = self.model.env
        while env is not None:
            if isinstance(env, VecNormalize):
                return env
            env = getattr(env, "venv", None)
        return None


class RollingCheckpointCallback(BaseCallback):
    """Save checkpoint every N steps, keeping only the last K.

    Prevents disk exhaustion: each checkpoint is ~5GB (includes pretrained
    CNN), so keeping all checkpoints would fill the disk. This callback
    deletes older checkpoints beyond keep_last, ensuring bounded disk usage.
    """

    def __init__(self, save_path, save_freq, keep_last=3, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.keep_last = keep_last
        self._saved_files = []

    def _on_step(self):
        if self.n_calls % self.save_freq != 0:
            return True

        os.makedirs(self.save_path, exist_ok=True)
        step = self.num_timesteps
        filename = os.path.join(self.save_path, f'checkpoint_{step}steps.zip')
        self.model.save(filename)
        self._saved_files.append(filename)

        while len(self._saved_files) > self.keep_last:
            old_file = self._saved_files.pop(0)
            if os.path.exists(old_file):
                os.remove(old_file)
                if self.verbose > 0:
                    print(f"[CHECKPOINT] Deleted old: {old_file}")

        if self.verbose > 0:
            print(f"[CHECKPOINT] Saved: {filename} "
                  f"(kept {len(self._saved_files)}/{self.keep_last})")
        return True


def make_env(env_id='PandaVLA-v0', reward_type='place_only',
             grasp_states=None, release_threshold=0.10,
             target_pos_range=None, vision_mode=False,
             domain_randomize=None, better_reward=False,
             use_pbrs=False, pbrs_gamma=0.99,
             pbrs_alpha=1.0, pbrs_beta=2.0, pbrs_scale=1.0):
    """Create a place-mode training/eval environment.

    Uses place_mode (hard-attached block): the block position is set
    to the hand each step, so the policy learns arm motion without
    dealing with grip physics. A release constraint in the env gates
    gripper opening to only when block_target_dist < release_threshold.
    The target matches the v3 grasp model's default target
    [0.5, 0.3, 0.2] so the place policy is compatible with the
    hierarchical eval.

    If grasp_states is provided, the env initializes the arm from
    collected grasp-policy states instead of a fixed lifted pose. This
    bridges the train-eval mismatch (see collect_grasp_states.py).

    Args:
        release_threshold: Distance (m) below which the gripper is
            allowed to open. Default 0.10m. Tightening to 0.05m for
            v13 forces the model to navigate closer before releasing,
            reducing post-release drift.
        target_pos_range: [[x_low, y_low, z_low], [x_high, y_high, z_high]]
            When set, the target position is randomized within this range
            on each reset. This trains the policy to generalize to different
            target positions. Default: None (fixed [0.5, 0.3, 0.2]).
        vision_mode: If True, use VisionObs wrapper (Dict {"image", "state"})
            instead of FlattenObs. This requires MultiInputPolicy
            (CNN for image + MLP for state).
        domain_randomize: If None, defaults to vision_mode (enable DR for
            vision). If True/False, overrides the default. Set to False
            for curriculum learning (train without DR first, then fine-tune
            with DR).
        better_reward: If True, enable improved reward signals (directional
            reward, height shaping, progressive proximity bonuses).
            NOTE: better_reward uses non-PBRS shaping that can be hacked
            (see V48 postmortem). For new experiments, prefer use_pbrs.
        use_pbrs: If True, wrap with PBRSShapingWrapper using
            placement_potential. This is the policy-invariant alternative
            to better_reward's ad-hoc shaping (Ng et al. 1999).
        pbrs_gamma: Discount factor used in PBRS shaping term.
        pbrs_alpha: Weight on horizontal distance in the potential function.
        pbrs_beta: Weight on vertical lift deficit in the potential function.
        pbrs_scale: Multiplier on the shaping term. At scale=1 the shaping
            signal is typically ~1000x smaller than raw_reward and has no
            practical effect on learning. Recommended 50-200 to bring
            per-step shaping to 5-15% of |raw_reward|. Policy invariance
            is preserved for any scale (Ng et al. 1999).
    """
    if domain_randomize is None:
        domain_randomize = vision_mode
    env = gymnasium.make(
        env_id,
        reward_type=reward_type,
        place_mode=True,
        gravity_comp=True,
        target_pos=np.array([0.5, 0.3, 0.2]),
        grasp_states=grasp_states,
        target_pos_range=target_pos_range,
        domain_randomize=domain_randomize,
        better_reward=better_reward,
    )
    if vision_mode:
        from gym_env.wrappers import VisionObs
        env = VisionObs(env, image_size=84)
    else:
        env = FlattenObs(env, include_target_pos=True)
    # Set configurable release threshold on the inner PandaVLAEnv
    env.unwrapped._release_dist_threshold = release_threshold

    if use_pbrs:
        # PBRS must wrap AFTER the observation wrapper so placement_potential
        # can read block_position/target_position from info (which is
        # produced by the env's _get_info, unchanged by obs wrappers).
        # PBRS only modifies the reward signal; observation space is
        # unchanged, so VecNormalize stats remain compatible.
        from gym_env.wrappers import PBRSShapingWrapper, placement_potential
        potential_fn = lambda obs, info, alpha=pbrs_alpha, beta=pbrs_beta: \
            placement_potential(obs, info, alpha=alpha, beta=beta)
        env = PBRSShapingWrapper(env, potential_fn, gamma=pbrs_gamma,
                                 shaping_scale=pbrs_scale)
    return env


def load_place_demos(demo_path, lift_threshold=0.05):
    """Load demos and keep only the place phase.

    For each episode (split by the `dones` flag), the place phase starts
    at the first transition where the block has been lifted above
    `lift_threshold` meters from the table
    (lift_height = obs[10] - 0.22 > lift_threshold) and runs to the end
    of the episode.

    Handles both 16-dim (old) and 19-dim (with target_pos) observations.
    If demos have 16-dim obs, target_pos [0.5, 0.3, 0.2] is appended.

    Args:
        demo_path: Path to the npz file with keys observations(N,16 or 19),
            actions(N,8), rewards, next_observations, dones.
        lift_threshold: Lift height (m) above the table that marks the
            start of the place phase.

    Returns:
        place_obs (np.ndarray, (M, 19)): place-phase observations.
        place_actions (np.ndarray, (M, 8)): place-phase actions.
    """
    data = np.load(demo_path, allow_pickle=True)
    observations = data['observations'].astype(np.float32)
    actions = data['actions'].astype(np.float32)
    dones = data['dones'].astype(np.float32)

    # Pad old 16-dim observations with target_pos [0.5, 0.3, 0.2]
    if observations.shape[1] == 16:
        target_pos = np.tile(np.array([0.5, 0.3, 0.2], dtype=np.float32),
                             (len(observations), 1))
        observations = np.concatenate([observations, target_pos], axis=1)

    table_z = 0.22
    # obs layout (FlattenObs 19-dim): [joint(7), gripper(1), block_xyz(3),
    #                            hand_xyz(3), hand_block_dist(1),
    #                            block_target_dist(1), target_xyz(3)]
    # -> obs[10] = block_z
    lift_height = observations[:, 10] - table_z

    # Episode boundaries from the dones flag
    done_idx = np.where(dones > 0.5)[0]
    ep_starts = np.concatenate([[0], done_idx + 1])
    ep_ends = np.concatenate([done_idx + 1, [len(dones)]])

    place_obs, place_actions = [], []
    for start, end in zip(ep_starts, ep_ends):
        if start >= end:
            continue
        ep_lift = lift_height[start:end]
        lifted = np.where(ep_lift > lift_threshold)[0]
        if len(lifted) == 0:
            continue
        place_start = start + lifted[0]
        place_obs.append(observations[place_start:end])
        place_actions.append(actions[place_start:end])

    if not place_obs:
        # Fallback: no episode structure found -> global filter
        mask = lift_height > lift_threshold
        if mask.sum() == 0:
            raise ValueError(
                f"No place-phase transitions (lift_height > {lift_threshold}m) "
                f"found in {demo_path}"
            )
        return observations[mask], actions[mask]

    place_obs = np.concatenate(place_obs, axis=0)
    place_actions = np.concatenate(place_actions, axis=0)
    return place_obs, place_actions


def main():
    parser = argparse.ArgumentParser(description="DAPG-PPO place policy training")
    parser.add_argument('--total_timesteps', type=int, default=300000)
    parser.add_argument(
        '--save_path', type=str,
        default='/home/w/vla_workspace/outputs/place_policy_v20',
    )
    parser.add_argument(
        '--grasp_states', type=str, default=GRASP_STATES_PATH,
        help='Path to collected grasp states pkl (for realistic init)',
    )
    parser.add_argument(
        '--demos', type=str,
        default='/home/w/vla_workspace/demos/panda_pickplace.npz',
    )
    parser.add_argument('--lambda_bc', type=float, default=None,
                        help='BC regularization weight (default: 0.5 for scratch, 0.1 for fine-tune)')
    parser.add_argument('--bc_decay', type=float, default=0.8,
                        help='BC weight decay factor (per 10%% of training)')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of episodes per eval (higher = less noise, slower)')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop training if no best model improvement for N evals (0=disabled)')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='PPO epochs per rollout (reduce to 3-5 for CPU training)')
    parser.add_argument(
        '--release_threshold', type=float, default=0.10,
        help='Release distance threshold (m). 0.05 for v13 tight release.',
    )
    parser.add_argument(
        '--load_model', type=str, default=None,
        help='Path to a saved model .zip to fine-tune from (e.g. v11 best_model)',
    )
    parser.add_argument(
        '--load_vecnorm', type=str, default=None,
        help='Path to vec_normalize.pkl matching --load_model',
    )
    parser.add_argument(
        '--target_pos_range', type=str, default=None,
        help='Target position range as "x_low,y_low,z_low,x_high,y_high,z_high". '
             'E.g. "0.35,-0.15,0.22,0.65,0.15,0.22". Default: fixed [0.5,0.3,0.2]',
    )
    parser.add_argument('--vision_mode', action='store_true',
        help='Use vision observation (image + state) instead of flat state vector')
    parser.add_argument('--vision_demos_extra', type=str, default=None,
                        help='Additional vision demo .npz to merge with '
                             '--vision_demos (e.g. 5k collection).')
    parser.add_argument('--vision_demos', type=str, default=None,
        help='Path to vision demo .npz (keys: images, states, actions, '
             'episode_lengths). Required for BC in vision_mode.')
    parser.add_argument('--pretrained_cnn', action='store_true',
        help='Use ImageNet-pretrained ResNet as the image feature '
             'extractor (instead of SB3 default NatureCNN from scratch). '
             'Only effective with --vision_mode. Early layers frozen, '
             'only layer4 is fine-tuned.')
    parser.add_argument('--cnn_backbone', type=str, default='resnet18',
        choices=['resnet18', 'resnet50'],
        help='ResNet backbone type. resnet18 (512-dim) or resnet50 (2048-dim). '
             'Only effective with --pretrained_cnn.')
    parser.add_argument('--better_reward', action='store_true',
        help='Enable improved reward signals: directional reward '
             '(encourage moving toward target), height shaping '
             '(keep block near target height), and progressive '
             'proximity bonuses.')
    parser.add_argument('--lr_schedule', type=str, default='constant',
        choices=['constant', 'linear', 'cosine'],
        help='Learning rate schedule. "constant" keeps LR fixed (v21 '
             'behavior); "linear" decays LR linearly to 0 over training; '
             '"cosine" uses cosine annealing (smoother endpoints, V59).')
    parser.add_argument('--weight_decay', type=float, default=0.0,
        help='L2 weight decay for the PPO optimizer. Injected into the '
             'Adam param_groups after model creation/load (does not '
             'change network architecture, so it is safe for fine-tuning). '
             'V59: 1e-4 for mild regularization against overfitting to '
             'limited grasp states.')
    parser.add_argument('--image_augment', action='store_true',
                        help='Enable image augmentation (random crop, brightness, '
                             'contrast) on demo images during BC loss. '
                             'Only effective with --vision_mode.')
    parser.add_argument('--freeze_bn', action='store_true',
        help='Freeze BatchNorm running statistics during fine-tuning. '
             'Overrides each BN layer\'s train() to stay in eval mode, '
             'preventing running_mean/running_var drift when training data '
             'distribution differs from pre-training. V59 root cause fix: '
             '321 mini-batch updates shifted BN running_mean by up to 21.5%%, '
             'crashing place_rate from 54%% to 35%%. Weights (gamma/beta) '
             'still receive gradients. Recommended for ALL fine-tuning runs.')
    parser.add_argument('--freeze_backbone', action='store_true',
        help='Freeze ALL feature extractor parameters (requires_grad=False). '
             'Unlike --freeze_bn (only BN running stats), this freezes the '
             'entire ResNet backbone including layer4. PPO can only update '
             'the MLP head. V68 diagnostic showed first PPO update destroyed '
             'pretrained features (50%%->5%%); this isolates whether the '
             'destruction is in the feature extractor or MLP head.')
    parser.add_argument('--n_steps', type=int, default=2048,
        help='Number of steps to collect per PPO rollout before each update. '
             'Smaller values = more frequent, smaller updates (less per-update '
             'change). V71a uses 512 (4x smaller than default 2048) to test '
             'whether smaller updates reduce destruction magnitude.')
    parser.add_argument('--no_domain_randomize', action='store_true',
        help='Disable domain randomization even in vision mode. Useful for '
             'curriculum learning: train without DR first (easier task), '
             'then fine-tune with DR enabled (harder, generalized task).')
    parser.add_argument('--no_tensorboard', action='store_true',
        help='Disable TensorBoard logging to prevent cleanup race conditions.')
    parser.add_argument('--use_pbrs', action='store_true',
        help='Wrap env with PBRSShapingWrapper (policy-invariant potential-based '
             'reward shaping, Ng et al. 1999). Replaces --better_reward with a '
             'theoretically sound alternative that cannot change the optimal '
             'policy set. Recommended for V49+ experiments.')
    parser.add_argument('--pbrs_alpha', type=float, default=1.0,
        help='Weight on horizontal distance in placement_potential. '
             'Controls how strongly Phi rewards approaching the target in xy.')
    parser.add_argument('--pbrs_beta', type=float, default=2.0,
        help='Weight on vertical lift deficit in placement_potential. '
             'Controls how strongly Phi rewards lifting the block to target z.')
    parser.add_argument('--pbrs_scale', type=float, default=1.0,
        help='Scale factor multiplied to the PBRS shaping term '
             'gamma*Phi(s\')-Phi(s). At scale=1 the shaping signal is '
             'typically 1000x smaller than the raw reward and has no '
             'practical effect on learning. Recommended: 50-200, tuned '
             'so per-step shaping is 5-15%% of |raw_reward|. Policy '
             'invariance is preserved for any scale (Ng et al. 1999).')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
        help='Max gradient norm for clipping. SB3 default is 0.5. '
             'Set to a smaller value (0.3) for stronger stabilisation '
             'against shaping-induced gradient spikes.')
    parser.add_argument('--ent_coef', type=float, default=0.01,
        help='Entropy coefficient. Lower (0.005) when using PBRS, since PBRS '
             'already provides exploration guidance.')
    parser.add_argument('--target_kl', type=float, default=None,
        help='Target KL divergence for early stopping PPO updates within an '
             'epoch. When set (e.g. 0.03), stops epoch iterations if KL exceeds '
             'this value, preventing policy drift. V51 postmortem: target_kl '
             'was never enabled, KL drifted freely and eval variance exploded '
             'from ±488 to ±1660. CRITICAL for long-training stability.')
    parser.add_argument('--clip_range', type=float, default=None,
        help='Fixed clip_range for PPO updates. If set, overrides the default '
             '0.2. V52 postmortem: KL jumped from 0.017 to 0.05 at 206k, '
             'clip_fraction reached 0.5. Tighter clip_range (0.1) reduces '
             'trust region size, preventing large policy updates that cause '
             'KL explosions.')
    parser.add_argument('--clip_range_schedule', action='store_true',
        help='Enable linear clip_range decay from --clip_range_initial to '
             '--clip_range_final over training. Prevents the PPO long-'
             'training instability where clip_fraction saturates near 0.2 '
             'after the policy converges (V49b postmortem: 60k peak -863 '
             'collapsed to -4625 at 130k).')
    parser.add_argument('--clip_range_initial', type=float, default=0.2,
        help='Initial clip_range when --clip_range_schedule is enabled.')
    parser.add_argument('--clip_range_final', type=float, default=0.05,
        help='Final clip_range when --clip_range_schedule is enabled. '
        'Recommended 0.05 (Engstrom et al. 2020).')
    parser.add_argument('--lr_final', type=float, default=None,
        help='Final LR when --lr_schedule=linear. If set, LR decays linearly '
             'from --learning_rate to this value instead of to 0. '
             'Recommended 25%% of initial LR (e.g. 2e-5 -> 5e-6). '
             'If None, decays to 0 (SB3 default behavior).')
    parser.add_argument('--hier_eval_freq', type=int, default=0,
        help='Run hierarchical eval (real physics) every N steps. When >0, '
             'enables HierPlaceRateCallback which saves the best model based '
             'on TRUE place_rate to best_hier/. Fixes the V55 bug where '
             'EvalCallback overwrote the 40%% model with a 14%% model.')
    parser.add_argument('--hier_eval_episodes', type=int, default=20,
        help='Number of episodes for hierarchical eval. Lower (20) for '
             'faster in-training eval; higher (50) for final evaluation.')
    parser.add_argument('--grasp_model', type=str, default=None,
        help='Path to grasp model for hierarchical eval. Required when '
             '--hier_eval_freq > 0.')
    parser.add_argument('--grasp_vecnorm', type=str, default=None,
        help='Path to grasp vec_normalize for hierarchical eval.')
    parser.add_argument('--hier_target_pos_range', type=str, default=None,
        help='Target position range for hierarchical eval (same format as '
             '--target_pos_range).')
    parser.add_argument('--hier_early_stop_threshold', type=int, default=0,
        help='Stop training if hier place_rate < threshold for N consecutive '
             'evals. 0 = disabled. V56 uses 35 (2 consecutive < 35%%).')
    parser.add_argument('--hier_early_stop_consecutive', type=int, default=2,
        help='Number of consecutive low hier evals to trigger early stop.')
    parser.add_argument('--first_eval_floor', type=int, default=0,
        help='If the first hier eval place_rate is below this, stop training '
             'immediately (start model may have degraded). 0 = disabled. '
             'V57 uses 35 to guard the 42%% starting point.')
    parser.add_argument('--decoupling_detection', action='store_true',
        help='Enable reward-policy decoupling detection. If place_mode eval '
             'reward improves while hier place_rate declines >10%%, stop '
             'training immediately. Catches the V56-style degradation where '
             'the policy finds place_mode shortcuts that fail in real physics.')
    parser.add_argument('--place_eval_freq', type=int, default=10000,
        help='EvalCallback (place_mode) eval frequency in steps. Default '
             '10000. V57 uses 2500 to match --hier_eval_freq for '
             'decoupling detection (needs synchronized data).')
    parser.add_argument('--checkpoint_freq', type=int, default=50000,
        help='Save checkpoint every N steps. Default 50000. V56 uses 5000 '
             'for frequent checkpoints aligned with hier eval.')
    parser.add_argument('--checkpoint_keep_last', type=int, default=3,
        help='Keep only the N most recent checkpoints, deleting older ones. '
             'Prevents disk exhaustion (each checkpoint is ~5GB).')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Load demos for BC regularization.
    # - State mode (from scratch): load place-phase state demos from --demos.
    # - State mode (fine-tune): no BC needed.
    # - Vision mode with --vision_demos: load Dict {"image","state"} demos.
    # - Vision mode without --vision_demos: no BC.
    demo_obs, demo_actions = None, None
    if args.vision_mode:
        if args.vision_demos:
            if not os.path.exists(args.vision_demos):
                print(f"Vision demo file not found: {args.vision_demos}")
                print("Run `python collect_vision_demos.py` first to generate demos.")
                return
            demo_data = np.load(args.vision_demos)
            demo_obs = {
                "image": demo_data["images"],   # (N, 84, 84, 3) uint8
                "state": demo_data["states"],   # (N, 12) float32
            }
            demo_actions = demo_data["actions"]  # (N, 8) float32
            print(f"Loaded {len(demo_actions)} vision demo transitions "
                  f"from {args.vision_demos}")
            # Merge extra demos if provided
            if args.vision_demos_extra and os.path.exists(args.vision_demos_extra):
                extra = np.load(args.vision_demos_extra)
                demo_obs["image"] = np.concatenate(
                    [demo_obs["image"], extra["images"]], axis=0)
                demo_obs["state"] = np.concatenate(
                    [demo_obs["state"], extra["states"]], axis=0)
                demo_actions = np.concatenate(
                    [demo_actions, extra["actions"]], axis=0)
                print(f"Merged {len(extra['actions'])} extra transitions "
                      f"from {args.vision_demos_extra} "
                      f"(total: {len(demo_actions)})")
        else:
            print("Vision mode without --vision_demos: skipping BC")
    elif not args.load_model:
        if not os.path.exists(args.demos):
            print(f"Demo file not found: {args.demos}")
            print("Run `python collect_demos.py` first to generate demos.")
            return
        demo_obs, demo_actions = load_place_demos(args.demos)
        print(f"Loaded {len(demo_obs)} place-phase demo transitions "
              f"from {args.demos}")
    else:
        print("Fine-tuning: no BC demos loaded")

    # Load collected grasp states for realistic initialization
    grasp_states = None
    if os.path.exists(args.grasp_states):
        with open(args.grasp_states, 'rb') as f:
            grasp_states = pickle.load(f)
        print(f"Loaded {len(grasp_states)} grasp states from {args.grasp_states}")
    else:
        print(f"WARNING: grasp states not found at {args.grasp_states}")
        print("  Falling back to fixed lifted pose. Run collect_grasp_states.py first.")

    print(f"Release threshold: {args.release_threshold}m")

    # Parse target position range
    target_pos_range = None
    if args.target_pos_range:
        vals = [float(v) for v in args.target_pos_range.split(',')]
        assert len(vals) == 6, "target_pos_range must be 6 values: x_low,y_low,z_low,x_high,y_high,z_high"
        target_pos_range = [[vals[0], vals[1], vals[2]], [vals[3], vals[4], vals[5]]]
        print(f"Target position range: x=[{vals[0]:.2f},{vals[3]:.2f}], "
              f"y=[{vals[1]:.2f},{vals[4]:.2f}], z=[{vals[2]:.2f},{vals[5]:.2f}]")

    # Create environments (place_mode + place_only reward).
    # See make_env docstring: requires PandaVLAEnv place_mode/place_only support.
    # domain_randomize: disabled if --no_domain_randomize (curriculum learning).
    domain_randomize = not args.no_domain_randomize if args.vision_mode else False
    env_kwargs = dict(
        grasp_states=grasp_states,
        release_threshold=args.release_threshold,
        target_pos_range=target_pos_range,
        vision_mode=args.vision_mode,
        domain_randomize=domain_randomize,
        better_reward=args.better_reward,
        use_pbrs=args.use_pbrs,
        pbrs_alpha=args.pbrs_alpha,
        pbrs_beta=args.pbrs_beta,
        pbrs_scale=args.pbrs_scale,
    )
    train_env = DummyVecEnv([functools.partial(make_env, **env_kwargs)])
    eval_env = DummyVecEnv([functools.partial(make_env, **env_kwargs)])

    # For vision mode, only normalize the "state" key (not the image).
    norm_obs_keys = ["state"] if args.vision_mode else None

    # Load VecNormalize stats if fine-tuning, otherwise create fresh
    if args.load_vecnorm and os.path.exists(args.load_vecnorm):
        print(f"Loading VecNormalize stats from {args.load_vecnorm}")
        train_env = VecNormalize.load(args.load_vecnorm, train_env)
        train_env.norm_reward = True
        train_env.training = True
        eval_env = VecNormalize.load(args.load_vecnorm, eval_env)
        eval_env.norm_reward = False
        eval_env.training = False
    else:
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=10.0,
            norm_obs_keys=norm_obs_keys,
        )
        eval_env = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
            norm_obs_keys=norm_obs_keys,
        )

    # For vision mode, wrap eval_env with VecTransposeImage to match training env
    if args.vision_mode:
        eval_env = VecTransposeImage(eval_env)

    # Determine hyperparameters: use lower LR and BC weight for fine-tuning
    if args.load_model:
        lr = args.learning_rate if args.learning_rate != 3e-4 else 1e-4
        lambda_bc = args.lambda_bc if args.lambda_bc is not None else 0.1
    else:
        lr = args.learning_rate
        lambda_bc = args.lambda_bc if args.lambda_bc is not None else 0.5

    # Apply learning rate schedule: "linear"/"cosine" decays LR from its
    # initial value to lr_final (or 0 if lr_final is None) over training.
    if args.lr_schedule in ("linear", "cosine"):
        final_lr = args.lr_final if args.lr_final is not None else 0.0
        if args.lr_schedule == "linear":
            lr = linear_schedule(lr, final_lr=final_lr)
        else:
            lr = cosine_schedule(lr, final_lr=final_lr)

    # Build policy_kwargs for pretrained ResNet-18 feature extractor.
    # Only effective for new training in vision_mode; fine-tuning reuses
    # the saved model's features extractor (SB3 load rejects mismatched
    # policy_kwargs).
    policy_kwargs = None
    if args.pretrained_cnn and not args.load_model:
        if not args.vision_mode:
            print("WARNING: --pretrained_cnn requires --vision_mode; ignoring")
        else:
            from core.pretrained_cnn import ResNetFeaturesExtractor
            if args.cnn_backbone == "resnet50":
                features_dim = 2048
            else:
                features_dim = 512
            policy_kwargs = {
                "features_extractor_class": ResNetFeaturesExtractor,
                "features_extractor_kwargs": {
                    "features_dim": features_dim,
                    "backbone": args.cnn_backbone,
                },
            }
            print(f"Using pretrained ResNet-{args.cnn_backbone[6:]} feature extractor "
                  f"(ImageNet weights, layer4 trainable, early layers frozen)")

    # Create or load DAPG-PPO model
    if args.load_model and os.path.exists(args.load_model):
        print(f"Fine-tuning from {args.load_model}")
        print(f"  lr={lr}, lambda_bc={lambda_bc}, image_augment={args.image_augment}")
        model = DAPGPPO.load(
            args.load_model,
            env=train_env,
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            lambda_bc=lambda_bc,
            bc_decay=args.bc_decay,
            total_timesteps=args.total_timesteps,
            learning_rate=lr,
            image_augment=args.image_augment,
            device=args.device,
        )
    else:
        policy_type = "MultiInputPolicy" if args.vision_mode else "MlpPolicy"
        tb_log_path = os.path.join(args.save_path, 'tb_logs') if not args.no_tensorboard else None
        model = DAPGPPO(
            policy_type,
            train_env,
            demo_obs=demo_obs,
            demo_actions=demo_actions,
            lambda_bc=lambda_bc,
            bc_decay=args.bc_decay,
            total_timesteps=args.total_timesteps,
            image_augment=args.image_augment,
            n_steps=args.n_steps,
            batch_size=64,
            learning_rate=lr,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range if args.clip_range is not None else (
                args.clip_range_initial if args.clip_range_schedule else 0.2),
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            verbose=1,
            tensorboard_log=tb_log_path,
            seed=42,
            device=args.device,
            policy_kwargs=policy_kwargs,
        )

    # Override n_steps for loaded models (V71a: smaller rollouts)
    if args.load_model and args.n_steps != model.n_steps:
        old_n_steps = model.n_steps
        model.n_steps = args.n_steps
        from gymnasium import spaces as gym_spaces
        from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
        buffer_cls = DictRolloutBuffer if isinstance(
            model.observation_space, gym_spaces.Dict) else RolloutBuffer
        model.rollout_buffer = buffer_cls(
            args.n_steps,
            model.observation_space,
            model.action_space,
            device=model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=model.n_envs,
        )
        print(f"n_steps overridden: {old_n_steps} -> {args.n_steps} "
              f"(rollout buffer reinitialized as {buffer_cls.__name__})")

    # Inject L2 weight decay into the optimizer (safe for fine-tuning:
    # does not change network architecture, only optimizer hyperparams).
    if args.weight_decay > 0:
        for pg in model.policy.optimizer.param_groups:
            pg['weight_decay'] = args.weight_decay
        print(f"L2 weight_decay={args.weight_decay} injected into optimizer "
              f"({len(model.policy.optimizer.param_groups)} param_groups)")

    # Freeze BatchNorm running statistics (V59 root cause fix).
    # Must be called AFTER model load/creation and BEFORE training starts.
    # PPO.train() calls set_training_mode(True) each update, which would
    # re-enable BN running stat updates — the override on each BN layer's
    # train() method prevents this permanently.
    if args.freeze_bn:
        frozen_count = freeze_bn_running_stats(model)
        print(f"BN running stats FROZEN: {frozen_count} BatchNorm layers "
              f"locked to eval mode (running_mean/var will not update)")
        assert frozen_count > 0, \
            "--freeze_bn specified but no BatchNorm layers found in policy"

    # Freeze entire backbone (V70: prevent PPO from destroying pretrained features)
    if args.freeze_backbone:
        total, frozen = freeze_backbone_params(model)
        assert frozen > 0, \
            "--freeze_backbone specified but no trainable params found in features extractor"

    # Eval callback: evaluate every 10000 steps, save best model
    # and vec_normalize stats together (fixes mismatch issue from v10)
    save_vecnorm_on_best = SaveVecNormalizeOnBest(
        os.path.join(args.save_path, 'best', 'vec_normalize.pkl'),
        verbose=1,
    )
    # Early stopping: stop training if no best-model improvement for N evals
    early_stop_callback = None
    if args.early_stop_patience > 0:
        early_stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=args.early_stop_patience,
            min_evals=5,
            verbose=1,
        )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.save_path, 'best'),
        log_path=os.path.join(args.save_path, 'eval_logs'),
        eval_freq=args.place_eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        callback_on_new_best=save_vecnorm_on_best,
        callback_after_eval=early_stop_callback,
        verbose=1,
    )

    # Save VecNormalize stats periodically for mid-training evaluation
    vecnorm_callback = SaveVecNormalizeCallback(
        os.path.join(args.save_path, 'vec_normalize.pkl'),
        save_freq=args.n_steps,  # save every rollout
    )

    # Rolling checkpoint: save every checkpoint_freq steps, keep only the
    # last checkpoint_keep_last. Prevents disk exhaustion (each checkpoint
    # is ~5GB with pretrained CNN). Replaces the old CheckpointCallback
    # which saved all checkpoints without cleanup.
    checkpoint_callback = RollingCheckpointCallback(
        save_path=os.path.join(args.save_path, 'checkpoints'),
        save_freq=args.checkpoint_freq,
        keep_last=args.checkpoint_keep_last,
        verbose=1,
    )

    # PBRS diagnostics: log raw_reward, shaping_reward and potential
    # running means. Healthy PBRS should show shaping_reward -> 0 as
    # the policy converges (the potential is absorbed into V), while
    # raw_reward tracks true task progress. If shaping_reward stays
    # large and positive while raw_reward stays flat, the potential
    # function is mis-specified.
    callbacks = [eval_callback, vecnorm_callback, checkpoint_callback]

    # Hierarchical eval callback: runs eval_hierarchical.py (real physics)
    # and saves the TRUE best model to best_hier/. Fixes the V55 bug where
    # EvalCallback's place_mode reward caused the 40% model to be
    # overwritten by a 14% model.
    if args.hier_eval_freq > 0:
        if not args.grasp_model or not os.path.exists(args.grasp_model):
            print(f"ERROR: --grasp_model required when --hier_eval_freq > 0, "
                  f"got: {args.grasp_model}")
            return
        hier_target_range = args.hier_target_pos_range or args.target_pos_range
        if hier_target_range is None:
            print(f"ERROR: --hier_target_pos_range or --target_pos_range "
                  f"required when --hier_eval_freq > 0")
            return
        hier_callback = HierPlaceRateCallback(
            eval_freq=args.hier_eval_freq,
            n_episodes=args.hier_eval_episodes,
            grasp_model=args.grasp_model,
            grasp_vecnorm=args.grasp_vecnorm,
            target_pos_range=hier_target_range,
            save_path=args.save_path,
            early_stop_threshold=args.hier_early_stop_threshold,
            early_stop_consecutive=args.hier_early_stop_consecutive,
            first_eval_floor=args.first_eval_floor,
            decoupling_detection=args.decoupling_detection,
            verbose=1,
        )
        callbacks.append(hier_callback)
        extras = []
        if args.first_eval_floor > 0:
            extras.append(f"first_eval_floor={args.first_eval_floor}%")
        if args.decoupling_detection:
            extras.append("decoupling_detection=ON")
        extras_str = f", {', '.join(extras)}" if extras else ""
        print(f"HierEval: every {args.hier_eval_freq} steps, "
              f"{args.hier_eval_episodes} episodes, "
              f"early_stop={args.hier_early_stop_threshold}% x "
              f"{args.hier_early_stop_consecutive}{extras_str}")

    if args.use_pbrs:
        pbrs_diag = PBRSDiagnosticsCallback(verbose=1)
        callbacks.append(pbrs_diag)
    if args.clip_range_schedule:
        clip_sched = ClipRangeScheduleCallback(
            initial_clip=args.clip_range_initial,
            final_clip=args.clip_range_final,
            total_timesteps=args.total_timesteps,
            verbose=1,
        )
        callbacks.append(clip_sched)

    # Train
    print(f"\n{'='*60}")
    print(f"DAPG-PPO Place Training: {args.total_timesteps} steps")
    if args.load_model:
        print(f"Fine-tuning from: {args.load_model}")
    if args.vision_mode:
        cnn_desc = "pretrained ResNet-18" if args.pretrained_cnn else "NatureCNN (scratch)"
        dr_status = "disabled (--no_domain_randomize)" if args.no_domain_randomize else "enabled"
        print(f"Vision mode: MultiInputPolicy ({cnn_desc}+MLP), domain_randomize={dr_status}")
    lr_display = (f"linear(initial={args.learning_rate})"
                  if args.lr_schedule == "linear" and not args.load_model
                  else (f"linear(initial=1e-4)" if args.lr_schedule == "linear"
                        else f"{lr}"))
    print(f"LR: {lr_display} (schedule={args.lr_schedule}), "
          f"BC weight: {lambda_bc} (decay={args.bc_decay})")
    print(f"Device: {args.device}")
    print(f"Eval episodes: {args.eval_episodes} (early_stop_patience={args.early_stop_patience})")
    print(f"Release threshold: {args.release_threshold}m")
    if args.use_pbrs:
        print(f"PBRS: ENABLED (alpha={args.pbrs_alpha}, beta={args.pbrs_beta}, "
              f"gamma=0.99, scale={args.pbrs_scale}) — "
              f"policy-invariant shaping (Ng et al. 1999)")
    elif args.better_reward:
        print(f"Reward: better_reward (non-PBRS, hackable — see V48 postmortem)")
    else:
        print(f"Reward: place_only (sparse)")
    print(f"Entropy coef: {args.ent_coef}")
    if args.max_grad_norm is not None:
        print(f"Grad clip: max_norm={args.max_grad_norm}")
    if args.clip_range_schedule:
        print(f"Clip range schedule: {args.clip_range_initial} -> "
              f"{args.clip_range_final} (linear decay over "
              f"{args.total_timesteps} steps)")
    if args.lr_schedule == "linear":
        final_lr_display = args.lr_final if args.lr_final is not None else 0.0
        print(f"LR schedule: linear {args.learning_rate} -> {final_lr_display}")
    if demo_obs is None:
        n_demos = 0
    elif isinstance(demo_obs, dict):
        n_demos = len(demo_obs["image"])
    else:
        n_demos = len(demo_obs)
    print(f"Place-phase demos: {n_demos} transitions")
    if args.vision_mode:
        print(f"Image augmentation: {'enabled' if args.image_augment else 'disabled'}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    # Save final model + normalization stats
    model.save(os.path.join(args.save_path, 'place_final'))
    train_env.save(os.path.join(args.save_path, 'vec_normalize.pkl'))
    print(f"\nTraining complete! Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
