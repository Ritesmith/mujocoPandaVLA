#!/usr/bin/env python3
"""Hierarchical pick-and-place policy.

Splits the pick-and-place task into two phases driven by independent
sub-policies:

  1. grasp phase  — v3 model grasps the block and lifts it.
  2. place phase  — place model moves the lifted block to the target
                    and releases it.

Phase transitions are driven by the block's lift height and gripper state
read from the env info dict, so the policy is agnostic to which sub-policy
is currently active.

FlattenObs layout (19-dim):
    [joint_pos(7), gripper(1), block_pos(3), hand_pos(3),
     hand_block_dist(1), block_target_dist(1), target_pos(3)]
"""
import numpy as np


TABLE_Z = 0.22


class HierarchicalPickPlacePolicy:
    """Hierarchical policy: v3 grasps -> place policy places.

    Parameters
    ----------
    grasp_model : stable_baselines3 PPO model
        Loaded SB3 model responsible for the grasp + lift phase.
    place_model : stable_baselines3 PPO model
        Loaded SB3 model responsible for the place phase.
    """

    # Phase transition thresholds
    GRASP_TO_PLACE_LIFT = 0.02   # m: block lifted enough to switch to place
    PLACE_TO_GRASP_LIFT = 0.01   # m: block dropped -> back to grasp
    GRIPPER_OPEN_THRESHOLD = 0.03  # m: gripper considered open above this
    MIN_GRASP_STEPS = 20          # min steps in grasp phase before switching

    def __init__(self, grasp_model, place_model):
        self.grasp_model = grasp_model
        self.place_model = place_model
        self.phase = "grasp"
        self.phase_steps = 0

    def reset(self):
        """Reset phase tracking at the start of a new episode."""
        self.phase = "grasp"
        self.phase_steps = 0

    def _detect_phase(self, info):
        """Update phase based on info dict.

        grasp -> place: lift > 0.02m, gripper closed, AND min 20 grasp steps
        place -> grasp: lift < 0.01m (block dropped)

        The MIN_GRASP_STEPS delay is critical: without it, the phase switches
        at step ~9 (when the block first reaches 2cm lift), while the arm is
        still moving fast. The place model was trained with the arm starting
        at rest, so it fails when the arm has high velocity (4% place rate).
        With the 20-step delay, the switch happens at step ~20, when the arm
        has stabilized and the block is near its peak height (~11cm). This
        matches the training distribution and gives 14%+ place rate.

        The lift threshold is 0.02 (lowered from 0.03) to catch episodes
        where the block is lifted to 2.8-2.9cm (just below the old 3cm
        threshold). Combined with the min-steps delay, this should increase
        the phase switch rate from 32% to ~60%.
        """
        if info is None:
            return self.phase

        block_height = float(info.get("block_height", TABLE_Z))
        gripper_opening = float(info.get("gripper_opening", 0.04))
        lift = max(0.0, block_height - TABLE_Z)
        gripper_open = gripper_opening > self.GRIPPER_OPEN_THRESHOLD

        if self.phase == "grasp":
            # Switch to place once block is lifted, held, and arm stabilized
            if (self.phase_steps >= self.MIN_GRASP_STEPS
                    and lift > self.GRASP_TO_PLACE_LIFT
                    and not gripper_open):
                self.phase = "place"
                self.phase_steps = 0
        elif self.phase == "place":
            # Block dropped -> back to grasp
            if lift < self.PLACE_TO_GRASP_LIFT:
                self.phase = "grasp"
                self.phase_steps = 0

        return self.phase

    def predict(self, obs, info=None, deterministic=True):
        """Predict action for the current observation.

        Parameters
        ----------
        obs : np.ndarray
            VecNormalize-normalized observation (already normalized).
        info : dict, optional
            Info dict from the env containing block_height,
            gripper_opening, etc.
        deterministic : bool
            Whether to use deterministic policy prediction.

        Returns
        -------
        (action, None) : tuple
            Action vector and None (SB3 convention).
        """
        phase = self._detect_phase(info)
        self.phase_steps += 1

        if phase == "place":
            action, _ = self.place_model.predict(obs, deterministic=deterministic)
        else:
            action, _ = self.grasp_model.predict(obs, deterministic=deterministic)

        return action, None
