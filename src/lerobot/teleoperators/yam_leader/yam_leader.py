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
    denormalize_arm_position,
    normalize_arm_position,
)
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..gamepad.gamepad_utils import GamepadController
from ..gamepad.teleop_gamepad import GripperAction
from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
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
        self.gamepad = None

        # Joint info (initialized in connect)
        self.joint_limits: np.ndarray | None = None

        # Bilateral control
        self.bilateral_kp: float = 0.1  # Force feedback strength (0.0 = no feedback, 1.0 = full feedback)
        self.leader_kp: np.ndarray | None = None  # Original kp values
        self.leader_kd: np.ndarray | None = None  # Original kd values
        self.is_intervening: bool = False

    @property
    def observation_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in YAM_ARM_MOTOR_NAMES}

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in YAM_ARM_MOTOR_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        # Expect follower joint positions to command leader for bilateral control
        return {f"{motor}.pos": float for motor in YAM_ARM_MOTOR_NAMES[:-1]}  # Exclude gripper

    @property
    def is_connected(self) -> bool:
        return (
            self.robot is not None
            and hasattr(self.robot._motor_chain, "running")
            and self.robot._motor_chain.running
        )

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.robot = YAMLeaderRobot(
            get_yam_robot(
                channel=self.config.port,
                gripper_type=GripperType.YAM_TEACHING_HANDLE,
                zero_gravity_mode=False,
            )
        )

        # Cache joint limits directly from the robot object (avoids buggy get_robot_info())
        self.joint_limits = self.robot._robot._joint_limits  # Shape: (6, 2) for 6 arm joints in radians

        # Store original kp/kd values for bilateral control
        self.leader_kp = self.robot._robot._kp.copy()
        self.leader_kd = self.robot._robot._kd.copy()

        self.gamepad = GamepadController()
        self.gamepad.start()

        time.sleep(1)
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

    def _get_normalized_motor_positions(self) -> dict[str, float]:
        """Get current motor positions in normalized space.

        Returns:
            Dict mapping motor names to normalized positions.
        """
        # Read arm position + gripper (returns positions in radians for arm, 0-1 for gripper)
        joint_positions, _ = self.robot.get_info()  # Returns (qpos_with_gripper, button_states)

        motor_positions = {}

        # Normalize arm joints (first 6 motors)
        for i, motor_name in enumerate(YAM_ARM_MOTOR_NAMES[:-1]):  # Exclude gripper
            val_rad = joint_positions[i]
            min_rad = self.joint_limits[i][0]
            max_rad = self.joint_limits[i][1]
            normalized_val = normalize_arm_position(val_rad, min_rad, max_rad, self.config.use_degrees)
            motor_positions[motor_name] = float(normalized_val)

        # Normalize gripper (gripper comes from get_info as 0-1, scale to 0-100)
        gripper_val_0_1 = joint_positions[-1]  # get_info returns 0-1 (0=closed, 1=open)
        normalized_gripper = GripperAction.OPEN.value if gripper_val_0_1 > 0.5 else GripperAction.CLOSE.value
        motor_positions[YAM_ARM_MOTOR_NAMES[-1]] = float(normalized_gripper)

        return motor_positions

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        # Get normalized motor positions
        action = {f"{motor}.pos": val for motor, val in self._get_normalized_motor_positions().items()}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")

        return action

    def get_teleop_events(self) -> dict[str, bool]:
        """
        Get teleop events from the YAM teaching handle button and gamepad controller.

        YAM Teaching Handle:
        - Button press: Intervention flag (must hold to maintain intervention)

        Gamepad button mapping (for episode control):
        - Y/Triangle button: Success (end episode successfully)
        - X/Square button: Failure (end episode as failure)
        - A/Cross button: Rerecord episode

        Returns:
            Dictionary containing:
                - TeleopEvents.IS_INTERVENTION: bool - Whether human is currently intervening
                - TeleopEvents.TERMINATE_EPISODE: bool - Whether to terminate the current episode
                - TeleopEvents.SUCCESS: bool - Whether the episode was successful
                - TeleopEvents.RERECORD_EPISODE: bool - Whether to rerecord the episode
        """
        # Get button states from teaching handle
        _, button_states = self.robot.get_info()  # Returns (qpos_with_gripper, button_states)

        # Update intervention state based on teaching handle button (must hold to maintain intervention)
        old_is_intervening = self.is_intervening
        self.is_intervening = bool(button_states[0]) if button_states and len(button_states) > 0 else False
        if self.is_intervening and not old_is_intervening:
            # Start intervention: enable bilateral control and zero torque mode
            logger.info("Enabling bilateral force feedback...")
            self.robot.update_kp_kd(kp=self.leader_kp * self.bilateral_kp, kd=np.ones(6) * 0.0)
            self.robot._robot.zero_torque_mode()
        elif not self.is_intervening and old_is_intervening:
            # Stop intervention: revert to original kp/kd and hold position
            logger.info("Disabling bilateral force feedback...")
            self.robot.update_kp_kd(kp=self.leader_kp, kd=self.leader_kd)
            self.robot._robot.hold_current_position()

        # Use gamepad for episode control events (success, failure, rerecord)
        terminate_episode = False
        success = False
        rerecord_episode = False

        if self.gamepad is not None:
            # Update gamepad state to get fresh inputs
            self.gamepad.update()

            # Get episode end status from gamepad
            episode_end_status = self.gamepad.get_episode_end_status()
            terminate_episode = episode_end_status in [
                TeleopEvents.RERECORD_EPISODE,
                TeleopEvents.FAILURE,
            ]
            success = episode_end_status == TeleopEvents.SUCCESS
            rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: self.is_intervening,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send bilateral force feedback to the leader arm.

        When intervening: Commands the leader to follow the follower's position,
        creating a "force feedback" effect that resists the user's movements.

        When NOT intervening: The leader is in zero-gravity mode, no feedback is applied.

        Args:
            feedback: Dictionary containing follower joint positions (e.g., {"shoulder_pan.pos": 0.5, ...})
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract follower joint positions (exclude gripper)
        follower_joints = np.array(
            [feedback.get(f"{motor}.pos", 0.0) for motor in YAM_ARM_MOTOR_NAMES[:-1]],
            dtype=float,
        )

        # Denormalize from [-100, 100] or [0, 360] back to radians using helper function
        follower_joints_rad = np.array(
            [
                denormalize_arm_position(
                    follower_joints[i],
                    self.joint_limits[i][0],
                    self.joint_limits[i][1],
                    self.config.use_degrees,
                )
                for i in range(6)
            ],
            dtype=float,
        )

        # Command leader to follow follower position (creates force feedback)
        self.robot.command_joint_pos(follower_joints_rad)

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Stop gamepad controller
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

        # Reset leader robot to zero gravity mode before closing
        logger.info("Disabling bilateral force feedback...")
        self.robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)

        # Close robot connection
        self.robot._robot.close()

        logger.info(f"{self} disconnected.")
