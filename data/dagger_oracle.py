"""DAgger oracle: analytical controller for place-phase expert actions.

Computes "correct" actions using Jacobian-based inverse kinematics:
1. Compute desired Cartesian velocity (block → target, proportional control)
2. Convert to joint velocity via MuJoCo Jacobian pseudo-inverse
3. Normalize to [-1, 1] action space (max 1 rad/s)
4. Open gripper when block is within release threshold of target

This provides NEW information from outside V59's policy distribution —
the analytical solution is NOT a sample from the policy, so bc_loss > 0
with a meaningful directional signal (unlike rejection sampling where
the mean of best samples ≈ policy mean).
"""

import numpy as np
import mujoco


class DAggerOracle:
    """Analytical oracle for DAgger on the place phase.

    Usage:
        oracle = DAggerOracle(env_inner)
        expert_action = oracle.get_expert_action()
    """

    def __init__(self, env_inner, gain=2.0, max_speed=0.5,
                 release_threshold=0.05):
        self.env = env_inner
        self.model = env_inner.model
        self.data = env_inner.data
        self.hand_id = env_inner._hand_id
        self.block_id = env_inner._red_block_id
        self.gain = gain
        self.max_speed = max_speed
        self.release_threshold = release_threshold

    def get_expert_action(self):
        """Compute expert action using Jacobian-based proportional control.

        Returns:
            action: (8,) float32 array — 7 joint velocities + 1 gripper cmd
        """
        block_pos = self.data.xpos[self.block_id].copy()
        target_pos = self.env._target_pos.copy()

        error = target_pos - block_pos
        dist = float(np.linalg.norm(error))

        if dist < 1e-4:
            velocity = np.zeros(3)
        else:
            speed = min(self.max_speed, dist * self.gain)
            velocity = (error / dist) * speed

        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, None,
                      block_pos, self.hand_id)

        jac_arm = jacp[:, :7]
        joint_vel = np.linalg.pinv(jac_arm, rcond=0.01) @ velocity

        action = np.zeros(8, dtype=np.float32)
        max_vel = np.max(np.abs(joint_vel)) if len(joint_vel) > 0 else 0.0
        if max_vel > 1.0:
            joint_vel = joint_vel / max_vel
        action[:7] = np.clip(joint_vel, -1.0, 1.0)

        if dist < self.release_threshold:
            action[7] = -1.0
        else:
            action[7] = 1.0

        return action

    def get_debug_info(self):
        """Return diagnostic info for logging."""
        block_pos = self.data.xpos[self.block_id].copy()
        target_pos = self.env._target_pos.copy()
        dist = float(np.linalg.norm(target_pos - block_pos))
        return {
            "block_pos": block_pos.tolist(),
            "target_pos": target_pos.tolist(),
            "dist": dist,
            "will_release": dist < self.release_threshold,
        }


class DAggerOracleV2(DAggerOracle):
    """Modified oracle: arm joints use Jacobian IK, gripper uses V59-style output.

    Unlike the parent DAggerOracle which outputs binary gripper (+1.0/-1.0),
    this version outputs a fixed V59-style gripper value (-0.07) that relies
    on the environment's distance gating for release timing.
    """

    def __init__(self, env_inner, gain=2.0, max_speed=0.5,
                 release_threshold=0.05, v59_gripper_value=-0.07):
        super().__init__(env_inner, gain, max_speed, release_threshold)
        self.v59_gripper_value = v59_gripper_value

    def get_expert_action(self):
        """Compute expert action with V59-style gripper output.

        Returns:
            action: (8,) float32 — 7 joint velocities (Jacobian IK) + 1 gripper cmd (V59 style)
        """
        # Call parent to get arm joint velocities (dimensions 0-6)
        action = super().get_expert_action()
        # Override gripper dimension (7) with V59-style value
        action[7] = self.v59_gripper_value
        return action
