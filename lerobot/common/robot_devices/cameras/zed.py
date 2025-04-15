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

"""
This file contains utilities for recording frames from Zed cameras.
"""

import argparse
import concurrent.futures
import logging
import shutil
import threading
import time
from pathlib import Path
from threading import Thread

from PIL import Image
import numpy as np

from lerobot.common.robot_devices.cameras.configs import ZedCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait,
)
from lerobot.common.utils.utils import capture_timestamp_utc


def find_cameras(raise_when_empty=True, mock=False) -> list[dict]:
    """
    Find the names and the ids of the Zed cameras
    connected to the computer.
    """
    if mock:
        raise NotImplementedError("Mocking is not implemented for Zed cameras.")

    import pyzed.sl as sl

    cameras = []
    for camera in sl.Camera.get_device_list():
        id = camera.id
        name = str(camera.camera_model)

        cameras.append(
            {
                "id": id,
                "name": name,
            }
        )

    if raise_when_empty and len(cameras) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `pyzed`, or updating the firmware."
        )

    return cameras


def save_image(img_array, id, frame_index, images_dir):
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{id}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        logging.info(f"Saved image: {path}")
    except Exception as e:
        logging.error(f"Failed to save image for camera {id} frame {frame_index}: {e}")


def save_images_from_cameras(
    images_dir: Path,
    ids: list[int] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given serial number.
    """
    if ids is None or len(ids) == 0:
        camera_infos = find_cameras(mock=mock)
        ids = [cam["id"] for cam in camera_infos]

    if mock:
        import tests.cameras.mock_cv2 as cv2
    else:
        import cv2

    print("Connecting cameras")
    cameras = []
    for id in ids:
        print(f"{id=}")
        config = ZedCameraConfig(
            id=id, fps=fps, width=width, height=height, mock=mock
        )
        camera = ZedCamera(config)
        camera.connect()
        print(
            f"ZedCamera({camera.id}, fps={camera.fps}, width={camera.capture_width}, height={camera.capture_height})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    # If we use async_read when fps is None, the loop will go full speed, and we will end up
                    # saving the same images from the cameras multiple times until the RAM/disk is full.
                    image = camera.read() if fps is None else camera.async_read()
                    if image is None:
                        print("No Frame")

                    bgr_converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    executor.submit(
                        save_image,
                        bgr_converted_image,
                        camera.id,
                        frame_index,
                        images_dir,
                    )

                if fps is not None:
                    dt_s = time.perf_counter() - now
                    busy_wait(1 / fps - dt_s)

                if time.perf_counter() - start_time > record_time_s:
                    break

                print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

                frame_index += 1
    finally:
        print(f"Images have been saved to {images_dir}")
        for camera in cameras:
            camera.disconnect()


class ZedCamera:
    """
    The ZedCamera class is similar to OpenCVCamera class but adds additional features for Zed cameras:
    - is instantiated with the id of the camera - won't randomly change as it can be the case of OpenCVCamera for Linux,
    - depth map can be returned.

    To find the camera indices of your cameras, you can run our utility script that will save a few frames for each camera:
    ```bash
    python lerobot/common/robot_devices/cameras/zed.py --images-dir outputs/images_from_zed_cameras
    ```

    When an ZedCamera is instantiated, if no specific config is provided, the default fps, width, and height
    of the given camera will be used.

    Example of instantiating with an id:
    ```python
    from lerobot.common.robot_devices.cameras.configs import ZedCameraConfig

    config = ZedCameraConfig(id=0)
    camera = ZedCamera(config)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```


    Example of changing default fps, width, and height:
    ```python
    config = ZedCameraConfig(id=0, fps=30, width=1280, height=720)
    config = ZedCameraConfig(id=0, fps=90, width=640, height=480)
    # Note: might error out upon `camera.connect()` if these settings are not compatible with the camera
    ```

    Example of returning depth:
    ```python
    config = ZedCameraConfig(id=0, use_depth=True)
    camera = ZedCamera(config)
    camera.connect()
    color_image, depth_map = camera.read()
    ```
    """

    def __init__(
        self,
        config: ZedCameraConfig,
    ):
        self.config = config
        self.id = config.id

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height

        self.width = config.width
        self.height = config.height

        self.fps = config.fps
        self.use_depth = config.use_depth
        self.mock = config.mock
        self.channels = config.channels

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.logs = {}

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"ZedCamera({self.id}) is already connected."
            )

        if self.mock:
            raise NotImplementedError("Mocking is not implemented for Zed cameras.")

        import pyzed.sl as sl

        self.camera = sl.Camera()

        init_params = sl.InitParameters()
        if self.height == 1200:
            init_params.camera_resolution = sl.RESOLUTION.HD1200
        elif self.height == 1080:
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        elif self.height == 600:
            init_params.camera_resolution = sl.RESOLUTION.SVGA
        else:
            raise ValueError(f"Expected height to be 1200, 1080, or 600, but {self.height} is provided.")

        init_params.camera_fps = self.fps

        if self.use_depth:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            init_params.depth_minimum_distance = 0
            init_params.coordinate_units = sl.UNIT.MILLIMETER
        else:
            init_params.depth_mode = sl.DEPTH_MODE.NONE

        init_params.set_from_camera_id(self.id)
        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            camera_infos = find_cameras()
            ids = [cam["id"] for cam in camera_infos]
            if self.id not in ids:
                raise ValueError(
                    f"`id` is expected to be one of these available cameras {ids}, but {self.id} is provided instead. "
                    "To find the id you should use, run `python lerobot/common/robot_devices/cameras/zed.py`."
                )

            raise OSError(f"Can't access ZedCamera({self.id}).")

        self.runtime_parameters = sl.RuntimeParameters()

        self.is_connected = True

    def read(self) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera returned in the format height x width x channels (e.g. 480 x 640 x 3)
        of type `np.uint8`, contrarily to the pytorch format which is float channel first.

        When `use_depth=True`, returns a tuple `(color_image, depth_map)` with a depth map in the format
        height x width (e.g. 480 x 640) of type np.uint16.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ZedCamera({self.id}) is not connected. Try running `camera.connect()` first."
            )

        if self.mock:
            raise NotImplementedError("Mocking is not implemented for Zed cameras.")

        import pyzed.sl as sl

        start_time = time.perf_counter()

        err = self.camera.grab(self.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            raise OSError(f"Can't capture frame from ZedCamera({self.id}).")
        
        image = sl.Mat()
        self.camera.retrieve_image(image, sl.VIEW.SIDE_BY_SIDE)

        h = image.get_height()
        w = image.get_width()
        if h != self.capture_height or w != self.capture_width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )
        
        image = image.get_data()

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = capture_timestamp_utc()

        if self.use_depth:
            depth_map = sl.Mat()
            self.camera.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
            depth_map = depth_map.get_data()

            h, w = depth_map.shape
            if h != self.capture_height or w != self.capture_width:
                raise OSError(
                    f"Can't capture depth map with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
                )

            return image, depth_map
        else:
            return image

    def read_loop(self):
        while not self.stop_event.is_set():
            if self.use_depth:
                self.color_image, self.depth_map = self.read()
            else:
                self.color_image = self.read()

    def async_read(self):
        """Access the latest color image"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ZedCamera({self.id}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            # TODO(rcadene, aliberts): zed has diverged compared to opencv over here
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )

        if self.use_depth:
            return self.color_image, self.depth_map
        else:
            return self.color_image

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"ZedCamera({self.id}) is not connected. Try running `camera.connect()` first."
            )

        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.close()
        self.camera = None

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `ZedCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="*",
        default=None,
        help="List of ids used to instantiate the `ZedCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_zed_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
