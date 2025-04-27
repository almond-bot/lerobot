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

import argparse
import logging
import shutil
from pathlib import Path

import cv2
import pyzed.sl as sl
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.utils.utils import init_logging


def extract_svo_frames(svo_path: Path, output_path: Path, fps: int, features: dict):
    """Convert a ZED SVO file to MP4 format, saving left, right, and depth (if available) as separate files."""
    # Get camera name from SVO file
    camera_name = svo_path.stem

    # Create output directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = output_path.parent.parent.parent / "images" / f"observation.images.{camera_name}" / output_path.parent.name
    images_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    if f"observation.images.{camera_name}.depth" in features:
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file: {svo_path}")

    # Get camera parameters
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    # Get video properties
    nb_frames = zed.get_svo_number_of_frames()
    resolution = zed.get_camera_information().camera_resolution
    width = resolution.width
    height = resolution.height

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    left_writer = None
    right_writer = None
    depth_writer = None

    if f"observation.images.{camera_name}.left" in features:
        left_writer = cv2.VideoWriter(str(output_path.with_name(f"{output_path.stem}.left.mp4")), fourcc, fps, (width, height))
    if f"observation.images.{camera_name}.right" in features:
        right_writer = cv2.VideoWriter(str(output_path.with_name(f"{output_path.stem}.right.mp4")), fourcc, fps, (width, height))
    if f"observation.images.{camera_name}.depth" in features:
        depth_writer = cv2.VideoWriter(str(output_path.with_name(f"{output_path.stem}.depth.mp4")), fourcc, fps, (width, height))

    # Process frames
    for i in tqdm(range(nb_frames), desc=f"Converting {svo_path.name}"):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left and right images
            if left_writer is not None:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                left = image.get_data()[..., :3]
                left_writer.write(left)
                cv2.imwrite(str(images_dir / "left" / f"frame_{i:06d}.png"), left)

            if right_writer is not None:
                zed.retrieve_image(image, sl.VIEW.RIGHT)
                right = image.get_data()[..., :3]
                right_writer.write(right)
                cv2.imwrite(str(images_dir / "right" / f"frame_{i:06d}.png"), right)

            # If depth is enabled, retrieve and write depth map
            if depth_writer is not None:
                zed.retrieve_image(depth, sl.VIEW.DEPTH)
                depth_image = depth.get_data()[..., :3]
                depth_writer.write(depth_image)
                cv2.imwrite(str(images_dir / "depth" / f"frame_{i:06d}.png"), depth_image)

    # Cleanup
    if left_writer is not None:
        left_writer.release()
    if right_writer is not None:
        right_writer.release()
    if depth_writer is not None:
        depth_writer.release()
    zed.close()


def process_episode_stats(dataset: LeRobotDataset, episode_index: int):
    """Calculate and save statistics for a specific episode."""
    episode_buffer = dataset.get_episode_buffer(episode_index)
    ep_stats = compute_episode_stats(episode_buffer, dataset.features)
    dataset.meta.save_episode_stats(episode_index, ep_stats)


def main():
    parser = argparse.ArgumentParser(description="Convert ZED SVO files to MP4 and calculate statistics.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to the LeRobotDataset directory")
    args = parser.parse_args()

    init_logging()

    # Load the dataset
    dataset = LeRobotDataset(args.dataset_dir)
    fps = dataset.meta.info["fps"]

    # Find all chunk directories
    videos_dir = dataset.root / "videos"
    chunks = sorted([int(d.name.split("-")[1]) for d in videos_dir.glob("chunk-*") if d.is_dir()])
    if not chunks:
        raise RuntimeError("No chunk directories found in videos folder. Please create at least one chunk directory (e.g., chunk-000)")

    # Process each chunk
    for chunk_num in chunks:
        chunk_dir = videos_dir / f"chunk-{chunk_num:03d}"
        logging.info(f"Processing chunk {chunk_num}")

        # Track processed episodes
        processed_episodes = set()

        # Find all episode directories in the chunk
        for camera_dir in chunk_dir.glob("observation.images.*"):
            camera_name = camera_dir.name.split(".")[-1]
            for episode_dir in camera_dir.glob("episode_*"):
                episode_index = int(episode_dir.name.split("_")[1])
                processed_episodes.add(episode_index)
                logging.info(f"Processing episode {episode_index} in {camera_name}")

                # Find SVO file for this episode
                svo_path = dataset.root / f"episode_{episode_index:06d}" / f"{camera_name}.svo2"
                if not svo_path.exists():
                    logging.warning(f"SVO file not found: {svo_path}")
                    continue

                # Convert SVO to MP4
                extract_svo_frames(svo_path, episode_dir, fps, dataset.meta.info["features"])

                # Calculate and save episode statistics
                process_episode_stats(dataset, episode_index)

        # After processing all episodes in the chunk, clean up SVO files and their parent folders
        for episode_index in processed_episodes:
            episode_dir = dataset.root / f"episode_{episode_index:06d}"
            if episode_dir.exists():
                logging.info(f"Removing episode directory: {episode_dir}")
                shutil.rmtree(episode_dir)

    logging.info("Conversion complete!")


if __name__ == "__main__":
    main() 