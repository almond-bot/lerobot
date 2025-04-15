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
from threading import Thread, Event
from queue import Queue

import torch

from lerobot.common.robot_devices.robots.fairino import RPC, RobotStatePkg
from lerobot.common.robot_devices.robots.configs import AlmondRobotConfig

def run_async_in_thread(coro: Coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


class AlmondRobot:

    ARM_IP = "192.168.57.2"
    ARM_STATUS_RATE = 65 # Hz

    def __init__(self, config: AlmondRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = AlmondRobotConfig(**kwargs)
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)

        self.robot_type = self.config.type
        self.cameras = self.config.cameras
        self.is_connected = False
        self.logs = {}

        self.arm_state: RobotStatePkg | None = None
        self.arm_state_stop_event = Event()

        self.target_gripper_position = 0
        self.target_gripper_force = 0

        self.action_keys: list[str] | None = None

        self.action_queue: Queue[dict[str, float]] = Queue()
    
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

    def _move_gripper(self, position: float, force: float):
        self.arm.MoveGripper(1, position, 0, force, 5000, 0, 0, 0, 0, 0)

    def _send_action(self):
        last_action = None
        joint_direction_multiplier = 0.01

        while not self.arm_state_stop_event.is_set():
            action = self.action_queue.get()

            if last_action is None:
                self.arm.ServoMoveStart()

            joint_pos = [action[f"j{i}.dir"] * joint_direction_multiplier for i in range(1, 7)]
            self.arm.ServoJ(joint_pos, cmdT=1 / AlmondRobot.ARM_STATUS_RATE)

            if last_action is not None and (last_action["gripper.pos"] != action["gripper.pos"] or last_action["gripper.for"] != action["gripper.for"]):
                move_gripper_thread = Thread(target=self._move_gripper, args=(action["gripper.pos"], action["gripper.for"]))
                move_gripper_thread.start()

            last_action = action

    def connect(self) -> None:
        if self.is_connected:
            return

        self.arm = RPC(AlmondRobot.ARM_IP)
        if not self.arm.is_conect:
            self.arm = None
            return

        self.arm.ResetAllError()
        self.arm.SetRobotRealtimeStateSamplePeriod(1000 / AlmondRobot.ARM_STATUS_RATE)

        self.arm.SetGripperConfig(4, 0)
        self.arm.ActGripper(1, 1)

        time.sleep(2)
        self.arm.ResetAllError()
        self.arm.MoveGripper(1, 0, 0, 0, 5000, 0, 0, 0, 0, 0)
        self.arm.ResetAllError()

        self.arm.Mode(1)

        self.arm.DragTeachSwitch(1)

        get_arm_status_thread = Thread(target=run_async_in_thread, args=(self._get_arm_status(),))
        get_arm_status_thread.start()

        send_action_thread = Thread(target=self._send_action)
        send_action_thread.start()

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.run_calibration()

    def run_calibration(self) -> None:
        # TODO(shawnpatel): set after experiment
        pass

    def get_observation_state(self) -> dict:
        return {
            "j1.pos": self.arm_state.jt_cur_pos[0],
            "j2.pos": self.arm_state.jt_cur_pos[1],
            "j3.pos": self.arm_state.jt_cur_pos[2],
            "j4.pos": self.arm_state.jt_cur_pos[3],
            "j5.pos": self.arm_state.jt_cur_pos[4],
            "j6.pos": self.arm_state.jt_cur_pos[5],
            "gripper.pos": self.arm_state.gripper_position,
            "gripper.cur": self.arm_state.gripper_current
        }

    def get_action_state(self) -> dict:
        return {
            "j1.dir": 1 if self.arm_state.actual_qd[0] > 0 else -1,
            "j2.dir": 1 if self.arm_state.actual_qd[1] > 0 else -1,
            "j3.dir": 1 if self.arm_state.actual_qd[2] > 0 else -1,
            "j4.dir": 1 if self.arm_state.actual_qd[3] > 0 else -1,
            "j5.dir": 1 if self.arm_state.actual_qd[4] > 0 else -1,
            "j6.dir": 1 if self.arm_state.actual_qd[5] > 0 else -1,
            "gripper.pos": self.target_gripper_position,
            "gripper.for": self.target_gripper_force
        }

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()
    
        if not record_data:
            return

        before_read_t = time.perf_counter()
        observation = self.get_observation_state()
        action = self.get_action_state()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        observation = torch.as_tensor(list(observation.values()))
        action = torch.as_tensor(list(action.values()))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = observation
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

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
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict = {}
        obs_dict["observation.state"] = observation
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        if self.action_keys is None:
            self.action_keys = list(self.get_action_state().keys())

        action_dict = dict(zip(self.action_keys, action.tolist(), strict=True))
        self.action_queue.put(action_dict)

        # TODO(aliberts): return action_sent when motion is limited
        return action

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def disconnect(self) -> None:
        self.arm_state_stop_event.set()

        self.arm.ServoMoveEnd()
        self.arm.DragTeachSwitch(0)
        self.arm.MoveGripper(1, 0, 0, 0, 5000, 0, 0, 0, 0, 0)

        self.arm.CloseRPC()
        self.arm = None

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()
