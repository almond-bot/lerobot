#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import traceback
import asyncio
from typing import Coroutine
import time
from dataclasses import replace
from threading import Thread, Event, Lock

import torch

from lerobot.common.robot_devices.motors.almond_gripper import AGGripper
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.fairino import RPC, RobotStatePkg
from lerobot.common.robot_devices.robots.configs import AlmondRobotConfig
from lerobot.common.robot_devices.motors.utils import make_motors_buses_from_configs

DYNAMIXEL_RESOLUTION = 4096
FR_ZERO_POSITION = [0, -135, 155, -130, -90, 0]
DMXL_ZERO_POSITION = [2038, 1653, 865, 3418, 174, -34]

DMXL_CLOSE_GRIPPER = 1958
DMXL_OPEN_GRIPPER = 2661

def run_async_in_thread(coro: Coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()

class AlmondRobot:
    ARM_IP = "192.168.57.2"
    ARM_STATUS_RATE = 30
    ARM_VELOCITY = 50
    ARM_ACCELERATION = 20
    SMOOTHING_FACTOR = 0.1
    POSITION_DIFF_THRESHOLD = 0.1

    def __init__(self, config: AlmondRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = AlmondRobotConfig(**kwargs)
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)

        self.robot_type = self.config.type
        self.leader_arm = make_motors_buses_from_configs(self.config.leader_arms)["main"]
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

        self.arm_state: RobotStatePkg | None = None
        self.last_arm_state: RobotStatePkg | None = None
        self.arm_state_stop_event = Event()
        self.cached_gripper_pos = 100  # Default to open position
        self.gripper_lock = Lock()

        self.last_teleop_time = None
        self.is_first_teleop_step = True
        
        self.smoothed_positions = [0.0] * 6

    async def _get_arm_status(self):
        reader, self.stream_writer = await asyncio.open_connection(
            AlmondRobot.ARM_IP, RPC.ROBOT_REALTIME_PORT
        )

        while not self.arm_state_stop_event.is_set():
            recvbuf = bytearray(RPC.BUFFER_SIZE)
            tmp_recvbuf = bytearray(RPC.BUFFER_SIZE)
            state_pkg = bytearray(RPC.BUFFER_SIZE)
            find_head_flag = False
            index = 0
            length = 0
            tmp_len = 0

            data = await reader.read(RPC.BUFFER_SIZE)
            recvbuf[: len(data)] = data
            recvbyte = len(data)

            if recvbyte <= 0:
                print("Failed to receive robot state bytes, trying to reconnect...")
                self.arm_state = None

                reader, self.stream_writer = await asyncio.open_connection(
                    AlmondRobot.ARM_IP, RPC.ROBOT_REALTIME_PORT
                )

            try:
                if tmp_len > 0:
                    if tmp_len + recvbyte <= RPC.BUFFER_SIZE:
                        recvbuf = tmp_recvbuf[:tmp_len] + recvbuf[:recvbyte]
                        recvbyte += tmp_len
                        tmp_len = 0
                    else:
                        tmp_len = 0

                for i in range(recvbyte):
                    if recvbuf[i] == 0x5A and not find_head_flag:
                        if i + 4 < recvbyte:
                            if recvbuf[i + 1] == 0x5A:
                                find_head_flag = True
                                state_pkg[0] = recvbuf[i]
                                index += 1
                                length = length | recvbuf[i + 4]
                                length = length << 8
                                length = length | recvbuf[i + 3]
                            else:
                                continue
                        else:
                            tmp_recvbuf[: recvbyte - i] = recvbuf[i:recvbyte]
                            tmp_len = recvbyte - i
                            break
                    elif find_head_flag and index < length + 5:
                        state_pkg[index] = recvbuf[i]
                        index += 1
                    elif find_head_flag and index >= length + 5:
                        if i + 1 < recvbyte:
                            checksum = sum(state_pkg[:index])
                            checkdata = 0
                            checkdata = checkdata | recvbuf[i + 1]
                            checkdata = checkdata << 8
                            checkdata = checkdata | recvbuf[i]

                            if checksum == checkdata:
                                self.arm_state = RobotStatePkg.from_buffer_copy(recvbuf)
                            else:
                                find_head_flag = False
                                index = 0
                                length = 0
                                i += 1
                        else:
                            tmp_recvbuf[: recvbyte - i] = recvbuf[i:recvbyte]
                            tmp_len = recvbyte - i
                            break

            except Exception:
                traceback.print_exc()

    def _get_gripper_position(self):
        while not self.arm_state_stop_event.is_set():
            try:
                with self.gripper_lock:
                    pos = self.gripper.get_current_position()
                if pos is not None:
                    self.cached_gripper_pos = pos
            except Exception:
                traceback.print_exc()

            time.sleep(1/AlmondRobot.ARM_STATUS_RATE)

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            views = ["left", "right"] if cam.use_depth else ["left", "right"]
            for view in views:
                cam_ft[f"observation.images.{cam_key}.{view}"] = {
                    "shape": (cam.height, cam.width, cam.channels),
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": cam.fps,
                        "video.height": cam.height,
                        "video.width": cam.width,
                        "video.channels": cam.channels,
                        "video.codec": cam.codec,
                        "video.is_depth_map": False,
                        "has_audio": False
                    },
                }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_keys = self.get_action_state(keys_only=True)
        state_keys = self.get_observation_state(keys_only=True)

        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_keys),),
                "names": action_keys,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_keys),),
                "names": state_keys,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self) -> None:
        if self.is_connected:
            return

        self.arm = RPC(AlmondRobot.ARM_IP)
        if not self.arm.is_conect:
            self.arm = None
            return

        # Create separate RPC connection for gripper
        self.gripper = AGGripper()
        self.gripper.initialize()
        self.gripper.set_force(100)

        self.arm.ResetAllError()
        self.arm.SetRobotRealtimeStateSamplePeriod(1000 / AlmondRobot.ARM_STATUS_RATE)

        get_arm_status_thread = Thread(target=run_async_in_thread, args=(self._get_arm_status(),))
        get_arm_status_thread.start()

        # Start gripper position reading thread
        get_gripper_pos_thread = Thread(target=self._get_gripper_position)
        get_gripper_pos_thread.start()

        self.is_connected = True

        self.leader_arm.connect()
        self.leader_arm.write("Torque_Enable", 0)

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.set_preset()

        self.run_calibration()
        self.arm.ServoMoveStart()

        while self.arm_state is None:
            time.sleep(0.25)

    def run_calibration(self) -> None:
        self.arm.ServoMoveEnd()
        self.arm.MoveJ(FR_ZERO_POSITION, 0, 0, vel=AlmondRobot.ARM_VELOCITY, acc=AlmondRobot.ARM_ACCELERATION)

    def set_preset(self) -> None:
        if (self.leader_arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
            raise ValueError("To run set robot preset, the torque must be disabled on all motors.")

        # Use 'extended position mode' for all motors except gripper, because in joint mode the servos can't
        # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
        # you could end up with a servo with a position 0 or 4095 at a crucial point See [
        # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
        all_motors_except_gripper = [name for name in self.leader_arm.motor_names if name != "gripper"]
        if len(all_motors_except_gripper) > 0:
            # 4 corresponds to Extended Position on Koch motors
            self.leader_arm.write("Operating_Mode", 4, all_motors_except_gripper)

        # Use 'position control current based' for gripper to be limited by the limit of the current.
        # For the follower gripper, it means it can grasp an object without forcing too much even tho,
        # it's goal position is a complete grasp (both gripper fingers are ordered to join and reach a touch).
        # For the leader gripper, it means we can use it as a physical trigger, since we can force with our finger
        # to make it move, and it will move back to its original target position when we release the force.
        # 5 corresponds to Current Controlled Position on Koch gripper motors "xl330-m077, xl330-m288"
        self.leader_arm.write("Operating_Mode", 5, "gripper")

        # Enable torque on the gripper of the leader arms, and move it to 45 degrees,
        # so that we can use it as a trigger to close the gripper of the follower arms.
        self.leader_arm.write("Torque_Enable", 1, "gripper")
        self.leader_arm.write("Goal_Position", DMXL_OPEN_GRIPPER, "gripper")

    def get_observation_state(self, keys_only: bool = False) -> dict:
        keys = ["j1.pos", "j2.pos", "j3.pos", "j4.pos", "j5.pos", "j6.pos", "j1.tor", "j2.tor", "j3.tor", "j4.tor", "j5.tor", "j6.tor", "gripper.pos"]
        if keys_only:
            return keys

        values = [float(self.arm_state.jt_cur_pos[i]) if self.arm_state is not None else float(0) for i in range(6)]
        values.extend([float(self.arm_state.jt_cur_tor[i]) if self.arm_state is not None else float(0) for i in range(6)])
        values.append(float(self.cached_gripper_pos))

        return {keys[i]: values[i] for i in range(len(keys))}

    def get_action_state(self, values: list[float] = [], keys_only: bool = False) -> dict:
        keys = ["j1.pos", "j2.pos", "j3.pos", "j4.pos", "j5.pos", "j6.pos", "gripper.pos"]
        if keys_only:
            return keys
        elif len(values) != len(keys):
            raise ValueError(f"Expected {len(keys)} values, got {len(values)}")

        return {keys[i]: values[i] for i in range(len(keys))}

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        current_time = time.perf_counter()
        if self.last_teleop_time is None:
            self.teleop_fps = 0
        else:
            self.teleop_fps = 1.0 / (current_time - self.last_teleop_time)
        self.last_teleop_time = current_time

        cur_pos = self.arm_state.jt_cur_pos
        goal_pos = self.leader_arm.read("Present_Position")

        arm_pos, gripper_pos = goal_pos[:6], goal_pos[6]

        arm_pos = [(float(x) - z) / DYNAMIXEL_RESOLUTION * 360 for x, z in zip(arm_pos, DMXL_ZERO_POSITION)]
        arm_pos[5] = (arm_pos[5] + 180) % 360 - 180
        arm_pos = [g + z for g, z in zip(arm_pos, FR_ZERO_POSITION)]

        gripper_percent = (gripper_pos - DMXL_CLOSE_GRIPPER) / (DMXL_OPEN_GRIPPER - DMXL_CLOSE_GRIPPER) * 100
        gripper_percent = max(0, min(100, gripper_percent))

        # Only run initialization sequence on first teleop step
        if self.is_first_teleop_step:
            self.arm.ServoMoveEnd()
            self.arm.MoveJ(arm_pos, 0, 0, vel=AlmondRobot.ARM_VELOCITY, acc=AlmondRobot.ARM_ACCELERATION)
            self.arm.ServoMoveStart()
            self.is_first_teleop_step = False
            self.smoothed_positions = arm_pos  # Initialize smoothed positions
        else:
            # Apply smoothing to the goal positions
            for i in range(6):
                self.smoothed_positions[i] = (AlmondRobot.SMOOTHING_FACTOR * arm_pos[i] + 
                                           (1 - AlmondRobot.SMOOTHING_FACTOR) * self.smoothed_positions[i])
            
            # Only send command if the smoothed position difference is above threshold
            if any(abs(g - c) > AlmondRobot.POSITION_DIFF_THRESHOLD for g, c in zip(self.smoothed_positions, cur_pos)):
                self.arm.ServoJ(self.smoothed_positions, axisPos=[0]*6, cmdT=1/(self.teleop_fps or 20))

        with self.gripper_lock:
            self.gripper.set_position(gripper_percent)

        if not record_data:
            return

        before_read_t = time.perf_counter()
        observation = self.get_observation_state()
        action = self.get_action_state(values=self.smoothed_positions + [gripper_percent])
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        self.last_arm_state = self.arm_state

        observation = torch.as_tensor(list(observation.values()))
        action = torch.as_tensor(list(action.values()))

        # Capture images from cameras
        for name in self.cameras:
            before_camread_t = time.perf_counter()

            self.cameras[name].save_frame()

            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = observation
        action_dict["action"] = action

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        before_read_t = time.perf_counter()
        observation = self.get_observation_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        observation = torch.as_tensor(list(observation.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()

            if hasattr(self.cameras[name], "use_depth") and self.cameras[name].use_depth:
                left, right, depth = images[name]
                images[f"{name}.left"] = torch.from_numpy(left)
                images[f"{name}.right"] = torch.from_numpy(right)
                images[f"{name}.depth"] = torch.from_numpy(depth)
            else:
                left, right = images[name]
                images[f"{name}.left"] = torch.from_numpy(left)
                images[f"{name}.right"] = torch.from_numpy(right)

            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict = {}
        obs_dict["observation.state"] = observation
        for name in self.cameras:
            obs_dict[f"observation.images.{name}.left"] = images[f"{name}.left"]
            obs_dict[f"observation.images.{name}.right"] = images[f"{name}.right"]
            if hasattr(self.cameras[name], "use_depth") and self.cameras[name].use_depth:
                obs_dict[f"observation.images.{name}.depth"] = images[f"{name}.depth"]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        action_dict = dict(zip(self.get_action_state(keys_only=True), action.tolist(), strict=True))

        # TODO(aliberts): return action_sent when motion is limited
        return action

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        self.arm_state_stop_event.set()

        self.arm.ServoMoveEnd()
        self.gripper.set_position(0)
        self.gripper.close()

        self.arm.CloseRPC()
        self.arm = None

        self.leader_arm.write("Torque_Enable", 0)
        self.leader_arm.disconnect()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()
