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
import re

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
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
    ARM_VELOCITY = 50
    ARM_ACCELERATION = 20

    def __init__(self, config: AlmondRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = AlmondRobotConfig(**kwargs)
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)

        self.robot_type = self.config.type
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

        self.arm_state: RobotStatePkg | None = None
        self.arm_state_stop_event = Event()

        self.target_gripper_position = 0
        self.target_gripper_force = 0

        self.action_queue: Queue[dict[str, float]] = Queue()
        
        # Web server setup
        self.app = FastAPI()
        self.active_websocket = None
        self.setup_webserver()

    def setup_webserver(self):
        # HTML template with a text box and WebSocket connection
        html = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Almond Robot Control</title>
            </head>
            <body>
                <h1>Almond Robot Control</h1>
                <input type="text" id="commandInput" placeholder="Enter command (e.g., g(position,force))">
                <button onclick="sendCommand()">Send</button>
                <div id="status"></div>
                <script>
                    var ws = new WebSocket("ws://" + window.location.host + "/ws");
                    ws.onmessage = function(event) {
                        document.getElementById("status").innerHTML = "Status: " + event.data;
                    };
                    function sendCommand() {
                        var command = document.getElementById("commandInput").value;
                        ws.send(command);
                    }
                </script>
            </body>
        </html>
        """

        @self.app.get("/")
        async def get():
            return HTMLResponse(html)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_websocket = websocket
            try:
                while True:
                    data = await websocket.receive_text()
                    self._handle_gripper_command(data)
            except WebSocketDisconnect:
                self.active_websocket = None

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

    def _handle_gripper_command(self, command: str) -> bool:
        """Handle gripper commands in the format g(position,force)"""

        match = re.match(r"g\(([0-9.]+),([0-9.]+)\)", command)
        if match:
            position = float(match.group(1))
            force = float(match.group(2))

            self.target_gripper_position = position
            self.target_gripper_force = force

            self._move_gripper(position, force)

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
            self.arm.ServoJ(joint_pos, cmdT=1 / AlmondRobot.ARM_STATUS_RATE, vel=AlmondRobot.ARM_VELOCITY, acc=AlmondRobot.ARM_ACCELERATION)

            if last_action is not None and (last_action["gripper.pos"] != action["gripper.pos"] or last_action["gripper.for"] != action["gripper.for"]):
                move_gripper_thread = Thread(target=self._move_gripper, args=(action["gripper.pos"], action["gripper.for"]))
                move_gripper_thread.start()

            last_action = action

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = ["j1.dir", "j2.dir", "j3.dir", "j4.dir", "j5.dir", "j6.dir"]
        state_names = ["j1.pos", "j2.pos", "j3.pos", "j4.pos", "j5.pos", "j6.pos"]
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
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

        # Start the web server in a separate thread
        webserver_thread = Thread(target=uvicorn.run, args=(self.app,), kwargs={"host": "0.0.0.0", "port": 8000})
        webserver_thread.daemon = True
        webserver_thread.start()

        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print("Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.run_calibration()

        self.is_connected = True

    def run_calibration(self) -> None:
        self.arm.MoveJ([0, -135, 135, -180, -90, 0], 0, 0, vel=AlmondRobot.ARM_VELOCITY, acc=AlmondRobot.ARM_ACCELERATION)

    def get_observation_state(self, keys_only: bool = False) -> dict:
        keys = ["j1.pos", "j2.pos", "j3.pos", "j4.pos", "j5.pos", "j6.pos", "gripper.pos", "gripper.cur"]
        if keys_only:
            return keys
        
        values = [self.arm_state.jt_cur_pos[i] for i in range(6)]
        values.append(self.arm_state.gripper_position)
        values.append(self.arm_state.gripper_current)

        return {keys[i]: values[i] for i in range(len(keys))}

    def get_action_state(self, keys_only: bool = False) -> dict:
        keys = ["j1.dir", "j2.dir", "j3.dir", "j4.dir", "j5.dir", "j6.dir", "gripper.pos", "gripper.for"]
        if keys_only:
            return keys
        
        values = [1 if self.arm_state.actual_qd[i] > 0 else -1 for i in range(6)]
        values.append(self.target_gripper_position)
        values.append(self.target_gripper_force)

        return {keys[i]: values[i] for i in range(len(keys))}

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

        action_dict = dict(zip(self.get_action_state(keys_only=True), action.tolist(), strict=True))
        self.action_queue.put(action_dict)

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
