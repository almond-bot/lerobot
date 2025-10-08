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

import numpy as np
from i2rt.lerobot.helpers import (
    YAM_ARM_MOTOR_NAMES,
    YAMLeaderRobot,
    normalize_arm_position,
    normalize_gripper_position,
)
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_yam_leader import YAMLeaderConfig

logger = logging.getLogger(__name__)


class YAMLeader(Teleoperator):
    """
    YAM leader arm designed by I2RT.
    """

    config_class = YAMLeaderConfig
    name = "yam_leader"

    def __init__(self, config: YAMLeaderConfig):
        super().__init__(config)
        self.config = config
        self.robot: YAMLeaderRobot | None = None

        # Joint info (initialized in connect)
        self.joint_limits: np.ndarray | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in YAM_ARM_MOTOR_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self.robot is not None
            and hasattr(self.robot.motor_chain, "running")
            and self.robot.motor_chain.running
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot = YAMLeaderRobot(
            get_yam_robot(
                channel=self.config.port, gripper_type=GripperType.YAM_TEACHING_HANDLE, zero_gravity_mode=True
            )
        )

        # Cache joint limits directly from the robot object (avoids buggy get_robot_info())
        self.joint_limits = self.robot._robot._joint_limits  # Shape: (6, 2) for 6 arm joints in radians

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # YAM doesn't need to calibrate
        return True

    def calibrate(self) -> None:
        raise NotImplementedError("YAM doesn't need to calibrate")

    def configure(self) -> None:
        raise NotImplementedError("YAM doesn't need to configure")

    def setup_motors(self) -> None:
        raise NotImplementedError("YAM doesn't need to setup motors")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Read arm position + gripper (returns positions in radians for arm, 0-1 for gripper)
        joint_positions, _ = self.robot.get_info()  # Returns (qpos_with_gripper, button_states)

        action = {}

        # Normalize arm joints (first 6 motors)
        for i, motor_name in enumerate(YAM_ARM_MOTOR_NAMES[:-1]):  # Exclude gripper
            val_rad = joint_positions[i]
            min_rad = self.joint_limits[i][0]
            max_rad = self.joint_limits[i][1]
            normalized_val = normalize_arm_position(val_rad, min_rad, max_rad, self.config.use_degrees)
            action[f"{motor_name}.pos"] = float(normalized_val)

        # Normalize gripper (gripper comes from get_info as 0-1, scale to 0-100)
        gripper_val_0_1 = joint_positions[-1]  # get_info returns 0-1 (0=closed, 1=open)
        gripper_normalized = normalize_gripper_position(gripper_val_0_1)  # Scale to 0-100
        action[f"{YAM_ARM_MOTOR_NAMES[-1]}.pos"] = float(gripper_normalized)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.robot._robot.close()

        logger.info(f"{self} disconnected.")
