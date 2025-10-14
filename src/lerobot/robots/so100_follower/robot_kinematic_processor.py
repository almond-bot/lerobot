#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    EnvTransition,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    TransitionKey,
)
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("ee_reference_and_delta")
@dataclass
class EEReferenceAndDelta(RobotActionProcessorStep):
    """
    Computes a target end-effector pose from a relative delta command.

    This step takes a desired change in position and orientation (`target_*`) and applies it to a
    reference end-effector pose to calculate an absolute target pose. The reference pose is derived
    from the current robot joint positions using forward kinematics.

    The processor can operate in two modes:
    1.  `use_latched_reference=True`: The reference pose is "latched" or saved at the moment the action
        is first enabled. Subsequent commands are relative to this fixed reference.
    2.  `use_latched_reference=False`: The reference pose is updated to the robot's current pose at
        every step.

    All operations are performed in vector space for simplicity:
    - Position: 3D cartesian vector (x, y, z)
    - Rotation: 3D rotation vector / axis-angle representation

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        end_effector_step_sizes: A dictionary scaling the input delta commands. Should contain keys:
            - "x", "y", "z" for position scaling (required)
            - "wx", "wy", "wz" for rotation scaling (optional, defaults to 1.0)
        motor_names: A list of motor names required for forward kinematics.
        use_latched_reference: If True, latch the reference pose on enable; otherwise, always use the
            current pose as the reference.
        reference_position: Internal state storing the latched reference position vector.
        reference_rotation: Internal state storing the latched reference rotation vector.
        _prev_enabled: Internal state to detect the rising edge of the enable signal.
        _prev_rotvec: Internal state for rotation vector continuity (avoids ±π discontinuities).
        _command_when_disabled: Internal state to hold the last command while disabled.
    """

    kinematics: RobotKinematics
    end_effector_step_sizes: dict
    motor_names: list[str]
    use_latched_reference: bool = (
        True  # If True, latch reference on enable; if False, always use current pose
    )
    use_ik_solution: bool = False

    reference_position: np.ndarray | None = field(default=None, init=False, repr=False)
    reference_rotation: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _prev_rotvec: np.ndarray | None = field(default=None, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.use_ik_solution and "IK_solution" in self.transition.get(TransitionKey.COMPLEMENTARY_DATA):
            q_raw = self.transition.get(TransitionKey.COMPLEMENTARY_DATA)["IK_solution"]
        else:
            q_raw = np.array(
                [
                    float(v)
                    for k, v in observation.items()
                    if isinstance(k, str)
                    and k.endswith(".pos")
                    and k.removesuffix(".pos") in self.motor_names
                ],
                dtype=float,
            )

        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        # Current pose from FK on measured joints
        t_curr = self.kinematics.forward_kinematics(q_raw)
        curr_position = t_curr[:3, 3]
        curr_rotation = Rotation.from_matrix(t_curr[:3, :3]).as_rotvec(previous_rotvec=self._prev_rotvec)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        gripper_vel = float(action.pop("gripper_vel"))

        desired_position = None
        desired_rotation = None

        if enabled:
            ref_position = curr_position
            ref_rotation = curr_rotation

            if self.use_latched_reference:
                # Latched reference mode: latch reference at the rising edge
                if not self._prev_enabled or self.reference_position is None:
                    self.reference_position = curr_position.copy()
                    self.reference_rotation = curr_rotation.copy()
                ref_position = self.reference_position
                ref_rotation = self.reference_rotation

            delta_p = np.array(
                [
                    tx * self.end_effector_step_sizes["x"],
                    ty * self.end_effector_step_sizes["y"],
                    tz * self.end_effector_step_sizes["z"],
                ],
                dtype=float,
            )
            delta_r = np.array(
                [
                    wx * self.end_effector_step_sizes["wx"],
                    wy * self.end_effector_step_sizes["wy"],
                    wz * self.end_effector_step_sizes["wz"],
                ],
                dtype=float,
            )

            # Apply position delta directly
            desired_position = ref_position + delta_p

            # Apply rotation delta properly using rotation composition
            r_ref = Rotation.from_rotvec(ref_rotation)
            r_delta = Rotation.from_rotvec(delta_r)
            r_desired = r_delta * r_ref
            desired_rotation = r_desired.as_rotvec()

            self._command_when_disabled = (desired_position.copy(), desired_rotation.copy())
        else:
            # While disabled, keep sending the same command to avoid drift.
            if self._command_when_disabled is None:
                # If we've never had an enabled command yet, freeze current pose.
                self._command_when_disabled = (curr_position.copy(), curr_rotation.copy())
            desired_position, desired_rotation = self._command_when_disabled

        # Write action fields
        action["ee.x"] = float(desired_position[0])
        action["ee.y"] = float(desired_position[1])
        action["ee.z"] = float(desired_position[2])
        action["ee.wx"] = float(desired_rotation[0])
        action["ee.wy"] = float(desired_rotation[1])
        action["ee.wz"] = float(desired_rotation[2])
        action["ee.gripper_vel"] = gripper_vel

        # Update state for next iteration
        self._prev_rotvec = desired_rotation.copy()
        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the internal state of the processor."""
        self._prev_enabled = False
        self.reference_position = None
        self.reference_rotation = None
        self._prev_rotvec = None
        self._command_when_disabled = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_vel",
        ]:
            features[PipelineFeatureType.ACTION].pop(f"{feat}", None)

        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_vel"]:
            features[PipelineFeatureType.ACTION][f"ee.{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("ee_bounds_and_safety")
@dataclass
class EEBoundsAndSafety(RobotActionProcessorStep):
    """
    Clips the end-effector pose to predefined bounds and checks for unsafe jumps.

    This step ensures that the target end-effector pose remains within a safe operational workspace.
    It also moderates the command to prevent large, sudden movements between consecutive steps.

    Attributes:
        end_effector_bounds: A dictionary with "min" and "max" keys for pose clipping.
            Each should be a list/array with at least 3 elements [x, y, z] for position bounds.
            If 6 elements are provided [x, y, z, wx, wy, wz], both position and orientation are clipped.
        max_ee_step_m: The maximum allowed change in position (in meters) between steps.
        max_ee_step_rad: The maximum allowed change in orientation (in radians) between steps.
        kinematics: Optional kinematics solver for computing current EE pose from joint positions.
        motor_names: List of motor names for forward kinematics.
        _last_pos: Internal state storing the last commanded position.
        _last_rot: Internal state storing the last commanded orientation (as rotation vector).
        _initialized: Whether _last_pos/_last_rot have been initialized from actual robot state.
    """

    end_effector_bounds: dict
    max_ee_step_m: float = 0.05
    max_ee_step_rad: float = 0.2
    kinematics: RobotKinematics | None = None
    motor_names: list[str] | None = None
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_rot: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        x = action["ee.x"]
        y = action["ee.y"]
        z = action["ee.z"]
        wx = action["ee.wx"]
        wy = action["ee.wy"]
        wz = action["ee.wz"]
        # TODO(Steven): ee.gripper_vel does not need to be bounded

        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.array([x, y, z], dtype=float)
        twist = np.array([wx, wy, wz], dtype=float)

        # Clip position and rotation
        bounds_min = np.array(self.end_effector_bounds["min"], dtype=float)
        bounds_max = np.array(self.end_effector_bounds["max"], dtype=float)

        # Clip position (first 3 elements)
        pos = np.clip(pos, bounds_min[:3], bounds_max[:3])

        # Clip rotation if bounds include orientation (6 elements total)
        if len(bounds_min) >= 6 and len(bounds_max) >= 6:
            twist = np.clip(twist, bounds_min[3:6], bounds_max[3:6])

        # Initialize from actual robot state on first call (after reset)
        if (
            self._last_pos is None
            and self._last_rot is None
            and self.kinematics is not None
            and self.motor_names is not None
        ):
            # Get current robot joint positions from observation
            observation = self.transition[TransitionKey.OBSERVATION]
            try:
                current_joints = np.array(
                    [
                        float(observation[f"{name}.pos"])
                        for name in self.motor_names
                        if f"{name}.pos" in observation
                    ],
                    dtype=float,
                )
                if len(current_joints) == len(self.motor_names):
                    # Compute current EE pose from actual robot joints
                    current_ee_pose = self.kinematics.forward_kinematics(current_joints)
                    self._last_pos = current_ee_pose[:3, 3]  # position
                    self._last_rot = Rotation.from_matrix(current_ee_pose[:3, :3]).as_rotvec()  # rotation
            except (KeyError, ValueError):
                pass

        if self._last_pos is None or self._last_rot is None:
            raise ValueError("Last position and rotation are not initialized")

        # Check for jumps in position
        dpos = pos - self._last_pos
        pos_norm = float(np.linalg.norm(dpos))
        if pos_norm > self.max_ee_step_m and pos_norm > 0:
            pos = self._last_pos + dpos * (self.max_ee_step_m / pos_norm)
            raise ValueError(f"EE position jump {pos_norm:.3f}m > {self.max_ee_step_m}m")

        # Check for jumps in rotation
        # Compute the actual angular difference between rotations (not just vector difference)
        r_last = Rotation.from_rotvec(self._last_rot)
        r_current = Rotation.from_rotvec(twist)
        r_diff = r_current * r_last.inv()
        drot_vec = r_diff.as_rotvec()
        rot_norm = float(np.linalg.norm(drot_vec))
        if rot_norm > self.max_ee_step_rad and rot_norm > 0:
            # Scale the rotation difference to max allowed step
            drot_vec_scaled = drot_vec * (self.max_ee_step_rad / rot_norm)
            r_diff_scaled = Rotation.from_rotvec(drot_vec_scaled)
            r_clamped = r_diff_scaled * r_last
            twist = r_clamped.as_rotvec()
            raise ValueError(f"EE rotation jump {rot_norm:.3f}rad > {self.max_ee_step_rad}rad")

        self._last_pos = pos
        self._last_rot = twist

        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        return action

    def reset(self):
        """Resets the last known position and orientation."""
        self._last_pos = None
        self._last_rot = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("inverse_kinematics_ee_to_joints")
@dataclass
class InverseKinematicsEEToJoints(RobotActionProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This step translates a Cartesian command (position and orientation of the end-effector) into
    the corresponding joint-space commands for each motor.

    Attributes:
        kinematics: The robot's kinematic model for inverse kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        q_curr: Internal state storing the last joint positions, used as an initial guess for the IK solver.
        initial_guess_current_joints: If True, use the robot's current joint state as the IK guess.
            If False, use the solution from the previous step.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def action(self, action: RobotAction) -> RobotAction:
        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = self.transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("gripper_velocity_to_joint")
@dataclass
class GripperVelocityToJoint(RobotActionProcessorStep):
    """
    Converts a gripper velocity command into a target gripper joint position.

    This step integrates a normalized velocity command over time to produce a position command,
    taking the current gripper position as a starting point. It also supports a discrete mode
    where integer actions map to open, close, or no-op.

    Attributes:
        motor_names: A list of motor names, which must include 'gripper'.
        speed_factor: A scaling factor to convert the normalized velocity command to a position change.
        clip_min: The minimum allowed gripper joint position.
        clip_max: The maximum allowed gripper joint position.
        discrete_gripper: If True, treat the input action as discrete (0: open, 1: close, 2: stay).
    """

    speed_factor: float = 20.0
    clip_min: float = 0.0
    clip_max: float = 100.0
    discrete_gripper: bool = False

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION).copy()

        gripper_vel = action.pop("ee.gripper_vel")

        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.discrete_gripper:
            # Discrete gripper actions are in [0, 1, 2]
            # 0: CLOSE, 1: STAY, 2: OPEN (from GripperAction enum)
            # We need to shift them to [-1, 0, 1] and then scale them to clip_max
            gripper_vel = (gripper_vel - 1) * self.clip_max

        # Compute desired gripper position
        delta = gripper_vel * float(self.speed_factor)
        # TODO: This assumes gripper is the last specified joint in the robot
        gripper_pos = float(np.clip(q_raw[-1] + delta, self.clip_min, self.clip_max))
        action["ee.gripper_pos"] = gripper_pos

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features[PipelineFeatureType.ACTION].pop("ee.gripper_vel", None)
        features[PipelineFeatureType.ACTION]["ee.gripper_pos"] = PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )

        return features


def compute_forward_kinematics_joints_to_ee(
    joints: dict[str, Any], kinematics: RobotKinematics, motor_names: list[str]
) -> dict[str, Any]:
    motor_joint_values = [joints[f"{n}.pos"] for n in motor_names]

    q = np.array(motor_joint_values, dtype=float)
    t = kinematics.forward_kinematics(q)
    pos = t[:3, 3]
    tw = Rotation.from_matrix(t[:3, :3]).as_rotvec()
    gripper_pos = joints["gripper.pos"]
    for n in motor_names:
        joints.pop(f"{n}.pos")
    joints["ee.x"] = float(pos[0])
    joints["ee.y"] = float(pos[1])
    joints["ee.z"] = float(pos[2])
    joints["ee.wx"] = float(tw[0])
    joints["ee.wy"] = float(tw[1])
    joints["ee.wz"] = float(tw[2])
    joints["ee.gripper_pos"] = float(gripper_pos)

    # Update observation.state to include ee pose instead of joint positions
    if OBS_STATE in joints:
        ee_state = torch.tensor(
            [pos[0], pos[1], pos[2], tw[0], tw[1], tw[2], gripper_pos], dtype=torch.float32
        )
        # If observation.state had joint positions, replace with ee pose
        # If it had joint positions + velocities/currents, keep those extras
        current_state = joints[OBS_STATE]
        num_joints = len(motor_names) + 1  # +1 for gripper

        if current_state.shape[-1] > num_joints:
            # Has additional features (velocities, currents, etc.)
            # Keep everything after the joint positions
            extra_features = current_state[..., num_joints:]
            ee_state = torch.cat(
                [ee_state.unsqueeze(0) if ee_state.dim() == 1 else ee_state, extra_features], dim=-1
            )
        else:
            # Just joint positions, replace entirely
            if ee_state.dim() == 1:
                ee_state = ee_state.unsqueeze(0)

        joints[OBS_STATE] = ee_state

    return joints


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_observation")
@dataclass
class ForwardKinematicsJointsToEEObservation(ObservationProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return compute_forward_kinematics_joints_to_ee(observation, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.OBSERVATION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.OBSERVATION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )

        # Update observation.state shape to reflect ee pose (7 values) instead of joint positions
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION]:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            num_joints = len(self.motor_names) + 1  # +1 for gripper
            ee_dim = 7  # x, y, z, wx, wy, wz, gripper_pos

            # Calculate the difference and adjust shape
            if original_feature.shape[0] > num_joints:
                # Has extra features (velocities, currents, etc.)
                extra_dim = original_feature.shape[0] - num_joints
                new_shape = (ee_dim + extra_dim,) + original_feature.shape[1:]
            else:
                # Just joint positions
                new_shape = (ee_dim,) + original_feature.shape[1:]

            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )

        return features


@ProcessorStepRegistry.register("forward_kinematics_joints_to_ee_action")
@dataclass
class ForwardKinematicsJointsToEEAction(RobotActionProcessorStep):
    """
    Computes the end-effector pose from joint positions using forward kinematics (FK).

    This step is typically used to add the robot's Cartesian pose to the observation space,
    which can be useful for visualization or as an input to a policy.

    Attributes:
        kinematics: The robot's kinematic model.
    """

    kinematics: RobotKinematics
    motor_names: list[str]

    def action(self, action: RobotAction) -> RobotAction:
        return compute_forward_kinematics_joints_to_ee(action, self.kinematics, self.motor_names)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # We only use the ee pose in the dataset, so we don't need the joint positions
        for n in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{n}.pos", None)
        # We specify the dataset features of this step that we want to be stored in the dataset
        for k in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION][f"ee.{k}"] = PolicyFeature(
                type=FeatureType.STATE, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register(name="forward_kinematics_joints_to_ee")
@dataclass
class ForwardKinematicsJointsToEE(ProcessorStep):
    kinematics: RobotKinematics
    motor_names: list[str]

    def __post_init__(self):
        self.joints_to_ee_action_processor = ForwardKinematicsJointsToEEAction(
            kinematics=self.kinematics, motor_names=self.motor_names
        )
        self.joints_to_ee_observation_processor = ForwardKinematicsJointsToEEObservation(
            kinematics=self.kinematics, motor_names=self.motor_names
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if transition.get(TransitionKey.ACTION) is not None:
            transition = self.joints_to_ee_action_processor(transition)
        if transition.get(TransitionKey.OBSERVATION) is not None:
            transition = self.joints_to_ee_observation_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        if features[PipelineFeatureType.ACTION] is not None:
            features = self.joints_to_ee_action_processor.transform_features(features)
        if features[PipelineFeatureType.OBSERVATION] is not None:
            features = self.joints_to_ee_observation_processor.transform_features(features)
        return features


@ProcessorStepRegistry.register("inverse_kinematics_rl_step")
@dataclass
class InverseKinematicsRLStep(ProcessorStep):
    """
    Computes desired joint positions from a target end-effector pose using inverse kinematics (IK).

    This is modified from the InverseKinematicsEEToJoints step to be used in the RL pipeline.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    q_curr: np.ndarray | None = field(default=None, init=False, repr=False)
    initial_guess_current_joints: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = new_transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )
        if q_raw is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        if self.initial_guess_current_joints:  # Use current joints as initial guess
            self.q_curr = q_raw
        else:  # Use previous ik solution as initial guess
            if self.q_curr is None:
                self.q_curr = q_raw

        # Build desired 4x4 transform from pos + rotvec (twist)
        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        # Compute inverse kinematics
        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        # TODO: This is sentitive to order of motor_names = q_target mapping
        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        # Always add gripper position back, even if not in motor_names
        # This handles robots where gripper is not part of kinematics
        if "gripper.pos" not in action and gripper_pos is not None:
            action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for feat in ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]:
            features[PipelineFeatureType.ACTION].pop(f"ee.{feat}", None)

        for name in self.motor_names:
            features[PipelineFeatureType.ACTION][f"{name}.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        # Always add gripper feature, even if not in motor_names
        # This handles robots where gripper is not part of kinematics
        if "gripper.pos" not in features[PipelineFeatureType.ACTION]:
            features[PipelineFeatureType.ACTION]["gripper.pos"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features

    def reset(self):
        """Resets the initial guess for the IK solver."""
        self.q_curr = None


@ProcessorStepRegistry.register("leader_joint_positions_to_ee_deltas")
@dataclass
class LeaderJointPositionsToEEDeltasStep(ProcessorStep):
    """
    Converts leader arm joint positions to end-effector delta actions with rate limiting.

    This processor is used when teloperating with a leader arm (e.g., yam_leader, so101_leader).
    It computes deltas from leader to follower pose, clips them to end_effector_step_sizes
    (preventing divergence), and normalizes to [-1, 1] range for pipeline compatibility.

    How it works:
    - Gets leader joint positions from complementary_data["teleop_action"]
    - Converts leader joints to EE pose via forward kinematics
    - Gets current follower EE pose from observation
    - Computes raw deltas: target_ee - current_ee
    - Clips each axis to ±end_effector_step_sizes (prevents divergence and ensures smooth following)
    - Normalizes to [-1, 1] by dividing by end_effector_step_sizes
    - EEReferenceAndDelta multiplies back by step_sizes, resulting in the clipped delta

    This ensures the follower moves at most step_sizes per control cycle, preventing
    divergence while allowing smooth tracking of the leader arm.

    Attributes:
        kinematics: The robot's kinematic model for forward kinematics.
        motor_names: A list of motor names for which to compute joint positions.
        end_effector_step_sizes: Step sizes for clipping and normalization. MUST match EEReferenceAndDelta.
        deadband_m: Deadband threshold for position (meters). Output zero if error is smaller.
        deadband_rad: Deadband threshold for rotation (radians). Output zero if error is smaller.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    end_effector_step_sizes: dict[str, float]  # Must match EEReferenceAndDelta step sizes
    deadband_m: float = 0.0005  # 0.5mm - output zero delta if closer than this
    deadband_rad: float = 0.005  # ~0.3 degrees - output zero delta if closer than this

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert leader joint positions to EE deltas with rate limiting."""
        new_transition = dict(transition)

        # Get leader joint positions from teleop_action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get("teleop_action")

        # Only process if we have teleop action in joint position format
        if teleop_action is None or not isinstance(teleop_action, dict):
            return new_transition

        # Check if this is joint position format (has .pos keys) vs delta format (has delta_x keys)
        has_joint_pos = any(key.endswith(".pos") for key in teleop_action)
        has_delta_keys = "delta_x" in teleop_action

        if has_delta_keys or not has_joint_pos:
            # Already in delta format (gamepad) or unknown format, pass through
            return new_transition

        # Get current follower EE pose from observation
        observation = new_transition.get(TransitionKey.OBSERVATION, {})
        if not observation:
            logging.warning("[LeaderToEEDeltas] No observation found, cannot compute deltas")
            return new_transition

        # Get current follower joint positions to compute current EE pose
        # Use IK solution if available (must match what EEReferenceAndDelta uses)
        if "IK_solution" in complementary_data:
            current_joints = complementary_data["IK_solution"]
        else:
            current_joints = np.array(
                [float(observation[f"{n}.pos"]) for n in self.motor_names if f"{n}.pos" in observation],
                dtype=float,
            )

        # Get leader target joint positions
        leader_joints = np.array(
            [float(teleop_action[f"{n}.pos"]) for n in self.motor_names if f"{n}.pos" in teleop_action],
            dtype=float,
        )

        # Compute current and target EE poses via FK
        current_ee_transform = self.kinematics.forward_kinematics(current_joints)
        target_ee_transform = self.kinematics.forward_kinematics(leader_joints)

        # Extract positions and rotations
        current_pos = current_ee_transform[:3, 3]
        target_pos = target_ee_transform[:3, 3]

        current_rot = Rotation.from_matrix(current_ee_transform[:3, :3])
        target_rot = Rotation.from_matrix(target_ee_transform[:3, :3])

        # Compute raw deltas in meters and radians
        raw_delta_pos = target_pos - current_pos

        # For rotation, compute the relative rotation from current to target
        relative_rot = target_rot * current_rot.inv()
        raw_delta_rot = relative_rot.as_rotvec()

        # Get gripper target
        target_gripper_pos = teleop_action["gripper.pos"]

        # Apply deadband to prevent jitter when very close
        pos_norm = np.linalg.norm(raw_delta_pos)
        rot_norm = np.linalg.norm(raw_delta_rot)

        if pos_norm < self.deadband_m and rot_norm < self.deadband_rad:
            # Very close to target - output zero delta to stabilize
            delta_action = {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "delta_wx": 0.0,
                "delta_wy": 0.0,
                "delta_wz": 0.0,
                "gripper": float(target_gripper_pos),
            }
            complementary_data["teleop_action"] = delta_action
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
            return new_transition

        # Clip each axis to its corresponding step size (prevents divergence)
        # This ensures follower can't move faster than step_sizes per control cycle
        clipped_delta_pos = np.array(
            [
                np.clip(
                    raw_delta_pos[0], -self.end_effector_step_sizes["x"], self.end_effector_step_sizes["x"]
                ),
                np.clip(
                    raw_delta_pos[1], -self.end_effector_step_sizes["y"], self.end_effector_step_sizes["y"]
                ),
                np.clip(
                    raw_delta_pos[2], -self.end_effector_step_sizes["z"], self.end_effector_step_sizes["z"]
                ),
            ]
        )

        wx_max = self.end_effector_step_sizes.get("wx", 1.0)
        wy_max = self.end_effector_step_sizes.get("wy", 1.0)
        wz_max = self.end_effector_step_sizes.get("wz", 1.0)
        clipped_delta_rot = np.array(
            [
                np.clip(raw_delta_rot[0], -wx_max, wx_max),
                np.clip(raw_delta_rot[1], -wy_max, wy_max),
                np.clip(raw_delta_rot[2], -wz_max, wz_max),
            ]
        )

        # Normalize to [-1, 1] range by dividing by step sizes
        # After EEReferenceAndDelta multiplies back, we get the clipped delta
        delta_pos = np.array(
            [
                clipped_delta_pos[0] / self.end_effector_step_sizes["x"],
                clipped_delta_pos[1] / self.end_effector_step_sizes["y"],
                clipped_delta_pos[2] / self.end_effector_step_sizes["z"],
            ]
        )

        delta_rot = np.array(
            [
                clipped_delta_rot[0] / wx_max,
                clipped_delta_rot[1] / wy_max,
                clipped_delta_rot[2] / wz_max,
            ]
        )

        # Create delta action dict compatible with InterventionActionProcessorStep
        delta_action = {
            "delta_x": float(delta_pos[0]),
            "delta_y": float(delta_pos[1]),
            "delta_z": float(delta_pos[2]),
            "delta_wx": float(delta_rot[0]),
            "delta_wy": float(delta_rot[1]),
            "delta_wz": float(delta_rot[2]),
            "gripper": float(target_gripper_pos),
        }

        # Update teleop_action in complementary data with delta format
        complementary_data["teleop_action"] = delta_action
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # This step doesn't modify the feature structure, just converts action format
        return features
