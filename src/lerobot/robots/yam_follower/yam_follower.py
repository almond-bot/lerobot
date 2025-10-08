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
import time
from functools import cached_property
from typing import Any

import numpy as np

from i2rt.i2rt.robots.get_robot import get_yam_robot
from i2rt.i2rt.robots.motor_chain_robot import MotorChainRobot
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_yam_follower import YAMFollowerConfig

logger = logging.getLogger(__name__)

# YAM arm motor names (6 joints) + gripper
YAM_ARM_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_rotate",
    "gripper",
]


class YAMFollower(Robot):
    """
    YAM follower arm designed by I2RT.
    """

    config_class = YAMFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YAMFollowerConfig):
        super().__init__(config)
        self.config = config
        self.robot: MotorChainRobot | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

        # Joint info (initialized in connect)
        self.joint_limits: np.ndarray | None = None
        self.gripper_limits: tuple[float, float] | None = None
        self.gripper_index: int | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in YAM_ARM_MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        robot_is_connected = (
            self.robot is not None
            and hasattr(self.robot.motor_chain, "running")
            and self.robot.motor_chain.running
        )
        cameras_are_connected = all(cam.is_connected for cam in self.cameras.values())
        return robot_is_connected and cameras_are_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot = get_yam_robot(
            channel=self.config.port, gripper_type=self.config.gripper_type, zero_gravity_mode=False
        )

        # Cache joint limits and gripper info
        robot_info = self.robot.get_robot_info()
        self.joint_limits = robot_info["joint_limits"]  # Shape: (6, 2) for 6 arm joints in radians
        self.gripper_limits = robot_info.get("gripper_limits")  # [closed, open] in radians or None
        self.gripper_index = robot_info.get("gripper_index")  # 6 if gripper present

        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # YAM doesn't need calibration
        return True

    def calibrate(self) -> None:
        raise NotImplementedError("YAM doesn't need calibration")

    def configure(self) -> None:
        raise NotImplementedError("YAM doesn't need configuration")

    def setup_motors(self) -> None:
        raise NotImplementedError("YAM doesn't need setup motors")

    def _normalize_arm_position(self, val_rad: float, min_rad: float, max_rad: float) -> float:
        """Normalize arm joint position from radians to degrees or -100 to 100 range."""
        if self.config.use_degrees:
            return val_rad * 180.0 / np.pi
        else:
            bounded_val = min(max_rad, max(min_rad, val_rad))
            return (((bounded_val - min_rad) / (max_rad - min_rad)) * 200.0) - 100.0

    def _denormalize_arm_position(self, normalized_val: float, min_rad: float, max_rad: float) -> float:
        """Denormalize arm joint position from degrees or -100 to 100 range to radians."""
        if self.config.use_degrees:
            return normalized_val * np.pi / 180.0
        else:
            return ((normalized_val + 100.0) / 200.0) * (max_rad - min_rad) + min_rad

    def _normalize_gripper_position(
        self, val_rad: float, gripper_closed: float, gripper_open: float
    ) -> float:
        """Normalize gripper position from radians to 0-100 range.

        Args:
            val_rad: Current gripper position in radians
            gripper_closed: Closed position in radians (maps to 0)
            gripper_open: Open position in radians (maps to 100)

        Returns:
            Normalized position where 0=closed, 100=open
        """
        return ((val_rad - gripper_closed) / (gripper_open - gripper_closed)) * 100.0

    def _denormalize_gripper_position(
        self, normalized_val: float, gripper_closed: float, gripper_open: float
    ) -> float:
        """Denormalize gripper position from 0-100 range to radians.

        Args:
            normalized_val: Normalized position (0=closed, 100=open)
            gripper_closed: Closed position in radians
            gripper_open: Open position in radians

        Returns:
            Position in radians
        """
        return (normalized_val / 100.0) * (gripper_open - gripper_closed) + gripper_closed

    def _get_normalized_motor_positions(self) -> dict[str, float]:
        """Get current motor positions in normalized space (without cameras).

        Returns:
            Dict mapping motor names to normalized positions.
        """
        # Read arm position (returns positions in radians)
        joint_positions = self.robot.get_joint_pos()  # Returns np.ndarray in radians

        motor_positions = {}

        # Normalize arm joints (first 6 motors)
        for i, motor_name in enumerate(YAM_ARM_MOTOR_NAMES[:-1]):  # Exclude gripper
            val_rad = joint_positions[i]
            min_rad = self.joint_limits[i][0]
            max_rad = self.joint_limits[i][1]
            normalized_val = self._normalize_arm_position(val_rad, min_rad, max_rad)
            motor_positions[motor_name] = float(normalized_val)

        # Normalize gripper (always 0-100 range, where 0=closed, 100=open)
        if (
            self.gripper_index is not None
            and self.gripper_limits is not None
            and len(joint_positions) > self.gripper_index
        ):
            gripper_val_rad = joint_positions[self.gripper_index]
            gripper_closed = self.gripper_limits[0]
            gripper_open = self.gripper_limits[1]
            normalized_gripper = self._normalize_gripper_position(
                gripper_val_rad, gripper_closed, gripper_open
            )
            motor_positions[YAM_ARM_MOTOR_NAMES[-1]] = float(normalized_gripper)

        return motor_positions

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Get normalized motor positions
        start = time.perf_counter()
        obs_dict = {f"{motor}.pos": val for motor, val in self._get_normalized_motor_positions().items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract normalized goal positions from action
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Cap goal position when too far away from present position (in normalized space)
        if self.config.max_relative_target is not None:
            # Get current normalized motor positions (without cameras)
            present_pos = self._get_normalized_motor_positions()

            # Build dict for ensure_safe_goal_position
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}

            # Clip in normalized space
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Denormalize to radians and build joint positions array
        joint_positions_rad = [0.0] * (
            len(YAM_ARM_MOTOR_NAMES[:-1])
            + (1 if self.gripper_index is not None and self.gripper_limits is not None else 0)
        )

        # Denormalize arm joints (first 6 motors)
        for i, motor_name in enumerate(YAM_ARM_MOTOR_NAMES[:-1]):  # Exclude gripper
            if motor_name in goal_pos:
                normalized_val = goal_pos[motor_name]
                min_rad = self.joint_limits[i][0]
                max_rad = self.joint_limits[i][1]
                val_rad = self._denormalize_arm_position(normalized_val, min_rad, max_rad)
                joint_positions_rad[i] = val_rad

        # Denormalize gripper (0-100 range, where 0=closed, 100=open)
        if self.gripper_index is not None and self.gripper_limits is not None:
            gripper_name = YAM_ARM_MOTOR_NAMES[-1]
            if gripper_name in goal_pos:
                normalized_gripper = goal_pos[gripper_name]
                gripper_closed = self.gripper_limits[0]
                gripper_open = self.gripper_limits[1]
                gripper_val_rad = self._denormalize_gripper_position(
                    normalized_gripper, gripper_closed, gripper_open
                )
                joint_positions_rad[self.gripper_index] = gripper_val_rad

        # Send command to robot
        self.robot.command_joint_pos(np.array(joint_positions_rad))

        # Return the (potentially clipped) action that was sent
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.robot.close()
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
