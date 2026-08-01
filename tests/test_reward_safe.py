#!/usr/bin/env python3
"""Tests for the hack-free reward function _compute_reward_safe().

Verifies the core design invariant: ALL per-step rewards are <= 0. Only
terminal rewards (+200 success, +50 release) can be positive, and they
fire one-time only.

These tests use a lightweight fixture that bypasses MuJoCo initialization
(by calling object.__new__ and manually setting the needed attributes).
This isolates the reward function logic for unit testing.
"""
import os
os.environ.pop('PYTHONPATH', None)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from gym_env.panda_vla_env import PandaVLAEnv


TABLE_Z = 0.22


def make_reward_env(block_pos, gripper_open=False, use_gripper_target_check=False,
                    prev_action=None, prev_prev_action=None, last_action=None,
                    place_success=False, place_was_holding=True,
                    early_release_penalty_given=False, release_bonus_given=False):
    """Create a minimal PandaVLAEnv instance for reward testing.

    Bypasses __init__ (which loads MuJoCo) by using object.__new__ and
    manually setting only the attributes accessed by _compute_reward_safe().
    """
    env = object.__new__(PandaVLAEnv)

    # Block state
    env._red_block_id = 0
    env._target_pos = np.array([0.5, 0.3, TABLE_Z])
    block_pos_arr = np.array(block_pos, dtype=float)
    env._finger_qpos_adrs = [0, 1]

    # Mock data object with xpos and qpos
    class _MockData:
        def __init__(self, block_pos, gripper_open):
            # xpos is indexed by body id; block_id=0
            self.xpos = np.zeros((1, 3))
            self.xpos[0] = block_pos
            # qpos for fingers: [pos0, pos1], mean > 0.02 means open
            finger_val = 0.035 if gripper_open else 0.005
            self.qpos = np.array([finger_val, finger_val])

    env.data = _MockData(block_pos_arr, gripper_open)

    # Gripper target (used when _use_gripper_target_check=True)
    env._gripper_target = 0.035 if gripper_open else 0.005
    env._use_gripper_target_check = use_gripper_target_check

    # Action history for jerk/action_diff
    env._last_action = last_action
    env._safe_prev_action = prev_action
    env._safe_prev_prev_action = prev_prev_action

    # Reward state flags
    env._place_success = place_success
    env._place_was_holding = place_was_holding
    env._place_early_release_penalty_given = early_release_penalty_given
    env._safe_release_bonus_given = release_bonus_given

    # Block tracking (updated but not used in shaping)
    env._prev_block_target_dist = None
    env._prev_block_height = None
    env._prev_block_pos = None

    return env


def block_at_dist(dist_from_target, height=TABLE_Z):
    """Return block position at given distance from target (in XY plane)."""
    target = np.array([0.5, 0.3, TABLE_Z])
    # Place block along +X from target at the given distance
    block = target.copy()
    block[0] += dist_from_target
    block[2] = height
    return block.tolist()


# ---------------------------------------------------------------------------
# Test 1: Per-step reward is negative at all distances (while holding)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist", [0.0, 0.05, 0.10, 0.15, 0.30])
def test_per_step_reward_negative_at_all_distances(dist):
    """At all distances, per-step reward (excluding terminal) must be <= 0.

    Uses identical actions to neutralize jerk/action_diff (they'd be 0).
    Block is at table height, gripper closed (holding).
    """
    action = np.zeros(8, dtype=float)
    env = make_reward_env(
        block_pos=block_at_dist(dist, height=TABLE_Z),
        gripper_open=False,  # holding
        last_action=action,
        prev_action=action,
        prev_prev_action=action,
    )
    reward = env._compute_reward_safe()
    # At dist=0 with holding, the only non-penalty is time (-0.01).
    # Terminal (+200/+50) does NOT fire because gripper is closed (holding).
    assert reward <= 0.0, f"dist={dist}: reward={reward} should be <= 0"


# ---------------------------------------------------------------------------
# Test 2: Terminal rewards are the ONLY positive rewards
# ---------------------------------------------------------------------------

def test_terminal_rewards_positive():
    """When block is at target, on table, gripper open: reward should be +250.

    +200 (success) + +50 (release) = +250. This is the ONLY scenario where
    reward can be positive.
    """
    action = np.zeros(8, dtype=float)
    env = make_reward_env(
        block_pos=block_at_dist(0.0, height=TABLE_Z),  # at target
        gripper_open=True,  # released
        last_action=action,
        prev_action=action,
        prev_prev_action=action,
    )
    reward = env._compute_reward_safe()
    # +200 (success) + +50 (release) - 0.01 (time) = +249.99
    assert reward > 0, f"Terminal reward should be positive, got {reward}"
    assert abs(reward - 249.99) < 0.1, f"Expected ~+249.99, got {reward}"


# ---------------------------------------------------------------------------
# Test 3: Isuccess gating — after success, reward = 0.0
# ---------------------------------------------------------------------------

def test_isuccess_gating():
    """After _place_success=True, reward must be 0.0 (no shaping)."""
    action = np.zeros(8, dtype=float)
    env = make_reward_env(
        block_pos=block_at_dist(0.30, height=TABLE_Z),  # far from target
        gripper_open=False,  # holding
        last_action=action,
        prev_action=action,
        prev_prev_action=action,
        place_success=True,  # ALREADY succeeded
    )
    reward = env._compute_reward_safe()
    assert reward == 0.0, f"After success, reward should be 0.0, got {reward}"


# ---------------------------------------------------------------------------
# Test 4: Release bonus fires only once per episode
# ---------------------------------------------------------------------------

def test_release_bonus_one_time():
    """+50 release bonus should fire only once per episode.

    First call: +50 + +200 - 0.01 = +249.99
    Second call (after _place_success=True): 0.0 (Isuccess gate)
    """
    action = np.zeros(8, dtype=float)
    env = make_reward_env(
        block_pos=block_at_dist(0.0, height=TABLE_Z),
        gripper_open=True,
        last_action=action,
        prev_action=action,
        prev_prev_action=action,
    )
    # First call: terminal rewards fire
    reward1 = env._compute_reward_safe()
    assert reward1 > 200, f"First call should include +200+50, got {reward1}"

    # _place_success is now True, so second call returns 0.0 (Isuccess gate)
    reward2 = env._compute_reward_safe()
    assert reward2 == 0.0, f"Second call should be 0 (Isuccess gate), got {reward2}"


# ---------------------------------------------------------------------------
# Test 5: Early release penalty fires once when releasing away from target
# ---------------------------------------------------------------------------

def test_early_release_penalty():
    """Releasing away from target should apply -5 one-time.

    Scenario: was holding, now gripper open, block far from target.
    Reward should include -5 (early release) plus per-step penalties.
    """
    action = np.zeros(8, dtype=float)
    env = make_reward_env(
        block_pos=block_at_dist(0.20, height=TABLE_Z),  # far from target
        gripper_open=True,  # just released
        last_action=action,
        prev_action=action,
        prev_prev_action=action,
        place_was_holding=True,  # was holding before
        early_release_penalty_given=False,  # not yet given
    )
    reward = env._compute_reward_safe()
    # Should be negative (no terminal, has early release -5 + time -0.01)
    assert reward < 0, f"Early release should make reward negative, got {reward}"
    # The early release penalty flag should now be set
    assert env._place_early_release_penalty_given is True


# ---------------------------------------------------------------------------
# Test 6: Jerk penalty is always <= 0
# ---------------------------------------------------------------------------

def test_jerk_penalty_negative():
    """Jerk penalty should always be <= 0 (it's -0.001 * ||jerk||^2)."""
    # Large jerk: a_t = [1,1,...], a_{t-1} = [0,0,...], a_{t-2} = [1,1,...]
    # jerk = 1 - 2*0 + 1 = 2 per dim, ||jerk||^2 = 8*4 = 32, penalty = -0.032
    action_t = np.ones(8, dtype=float)
    action_prev = np.zeros(8, dtype=float)
    action_prev_prev = np.ones(8, dtype=float)

    env = make_reward_env(
        block_pos=block_at_dist(0.10, height=TABLE_Z),
        gripper_open=False,  # holding
        last_action=action_t,
        prev_action=action_prev,
        prev_prev_action=action_prev_prev,
    )
    reward_with_jerk = env._compute_reward_safe()

    # Compare with zero-jerk case (same setup, identical actions)
    env_zero = make_reward_env(
        block_pos=block_at_dist(0.10, height=TABLE_Z),
        gripper_open=False,
        last_action=action_t,
        prev_action=action_t,
        prev_prev_action=action_t,
    )
    reward_zero_jerk = env_zero._compute_reward_safe()

    # Jerk case should be MORE negative (or equal) than zero-jerk case
    assert reward_with_jerk < reward_zero_jerk, (
        f"Jerk penalty should make reward more negative: "
        f"with_jerk={reward_with_jerk}, zero_jerk={reward_zero_jerk}"
    )


# ---------------------------------------------------------------------------
# Test 7: At dist=0 holding, reward <= -0.01 (only time penalty)
# ---------------------------------------------------------------------------

def test_no_positive_per_step_at_dist_zero():
    """At dist=0, holding, no jerk/action_diff: reward is strictly negative.

    This is the critical hack-free proof: even at the target, if the agent
    holds without releasing, reward is strictly negative. No farming possible.

    At dist=0 holding: hover penalty (-0.5) + time penalty (-0.01) = -0.51.
    The hover penalty INTENTIONALLY fires at dist<0.05 to prevent hover-farming
    at the release threshold.
    """
    action = np.zeros(8, dtype=float)
    env = make_reward_env(
        block_pos=block_at_dist(0.0, height=TABLE_Z),  # exactly at target
        gripper_open=False,  # holding (not released)
        last_action=action,
        prev_action=action,
        prev_prev_action=action,
    )
    reward = env._compute_reward_safe()
    # dist=0: no distance penalty. height=table_z: no height penalty.
    # hover penalty fires (dist<0.05): -0.5 * (0.05-0)/0.05 = -0.5
    # jerk=0, action_diff=0. Time penalty = -0.01.
    # Total: -0.5 (hover) + -0.01 (time) = -0.51
    assert reward < 0, f"At dist=0 holding, reward must be negative, got {reward}"
    assert reward == pytest.approx(-0.51, abs=1e-6), (
        f"At dist=0 holding, expected -0.51 (hover -0.5 + time -0.01), got {reward}"
    )


# ---------------------------------------------------------------------------
# Additional: distance penalty scales linearly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist,expected_penalty", [
    (0.10, -0.5),   # -5.0 * 0.10
    (0.15, -0.75),  # -5.0 * 0.15
    (0.30, -1.5),   # -5.0 * 0.30
])
def test_distance_penalty_linear(dist, expected_penalty):
    """Distance penalty should be -5.0 * block_target_dist (while holding)."""
    action = np.zeros(8, dtype=float)
    env = make_reward_env(
        block_pos=block_at_dist(dist, height=TABLE_Z),
        gripper_open=False,  # holding
        last_action=action,
        prev_action=action,
        prev_prev_action=action,
    )
    reward = env._compute_reward_safe()
    # reward = -5.0*dist (distance) + 0 (height, at table) + 0 (hover, dist>=0.05)
    #        + 0 (jerk) + 0 (action_diff) - 0.01 (time)
    expected = expected_penalty - 0.01
    assert reward == pytest.approx(expected, abs=1e-6), (
        f"dist={dist}: expected reward={expected}, got {reward}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
