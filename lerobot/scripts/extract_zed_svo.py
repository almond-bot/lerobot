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
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import compute_episode_stats
from lerobot.common.datasets.utils import write_episode_stats
from lerobot.common.utils.utils import init_logging


def extract_svo_frames(svo_path: Path, dataset: LeRobotDataset):
    """Convert a ZED SVO file to MP4 format, saving left, right, and depth (if available) as separate files."""
    # Get camera name from SVO file
    episode_name = svo_path.stem
    camera_name = svo_path.parent.name
    chunk_name = svo_path.parent.parent.name

    fps = dataset.meta.info["fps"]
    features = dataset.meta.info["features"]

    # Initialize ZED camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    if f"observation.images.{camera_name}.depth" in features:
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    else:
        init_params.depth_mode = sl.DEPTH_MODE.NONE
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO file: {svo_path}")

    # Get camera parameters
    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()

    # Get video properties
    nb_frames = zed.get_svo_number_of_frames()
    
    # Get dimensions for each feature
    dimensions = {}
    for view in ["left", "right", "depth"]:
        feature_key = f"observation.images.{camera_name}.{view}"
        if feature_key in features:
            dimensions[view] = {
                "width": features[feature_key]["info"]["video.width"],
                "height": features[feature_key]["info"]["video.height"]
            }

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    left_writer = None
    right_writer = None
    depth_writer = None

    # Create output directories
    videos_dir = dataset.root / "videos" / chunk_name
    images_dir = dataset.root / "images" / chunk_name

    # Create video output paths
    if f"observation.images.{camera_name}.left" in features:
        left_video_dir = videos_dir / f"observation.images.{camera_name}.left"
        left_video_dir.mkdir(parents=True, exist_ok=True)
        left_writer = cv2.VideoWriter(str(left_video_dir / f"{episode_name}.mp4"), fourcc, fps, (dimensions["left"]["width"], dimensions["left"]["height"]))
    if f"observation.images.{camera_name}.right" in features:
        right_video_dir = videos_dir / f"observation.images.{camera_name}.right"
        right_video_dir.mkdir(parents=True, exist_ok=True)
        right_writer = cv2.VideoWriter(str(right_video_dir / f"{episode_name}.mp4"), fourcc, fps, (dimensions["right"]["width"], dimensions["right"]["height"]))
    if f"observation.images.{camera_name}.depth" in features:
        depth_video_dir = videos_dir / f"observation.images.{camera_name}.depth"
        depth_video_dir.mkdir(parents=True, exist_ok=True)
        depth_writer = cv2.VideoWriter(str(depth_video_dir / f"{episode_name}.mp4"), fourcc, fps, (dimensions["depth"]["width"], dimensions["depth"]["height"]))

    # Create image output paths
    if f"observation.images.{camera_name}.left" in features:
        left_image_dir = images_dir / f"observation.images.{camera_name}.left" / episode_name
        left_image_dir.mkdir(parents=True, exist_ok=True)
    if f"observation.images.{camera_name}.right" in features:
        right_image_dir = images_dir / f"observation.images.{camera_name}.right" / episode_name
        right_image_dir.mkdir(parents=True, exist_ok=True)
    if f"observation.images.{camera_name}.depth" in features:
        depth_image_dir = images_dir / f"observation.images.{camera_name}.depth" / episode_name
        depth_image_dir.mkdir(parents=True, exist_ok=True)

    # Process frames
    for i in tqdm(range(nb_frames), desc=f"Converting {svo_path.name}"):
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left and right images
            if left_writer is not None:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                left = image.get_data()[..., :3]
                left_writer.write(left)
                cv2.imwrite(str(left_image_dir / f"frame_{i:06d}.png"), left)

            if right_writer is not None:
                zed.retrieve_image(image, sl.VIEW.RIGHT)
                right = image.get_data()[..., :3]
                right_writer.write(right)
                cv2.imwrite(str(right_image_dir / f"frame_{i:06d}.png"), right)

            # If depth is enabled, retrieve and write depth map
            if depth_writer is not None:
                zed.retrieve_image(depth, sl.VIEW.DEPTH)
                depth_image = depth.get_data()[..., :3]
                depth_writer.write(depth_image)
                cv2.imwrite(str(depth_image_dir / f"frame_{i:06d}.png"), depth_image)

    # Cleanup
    if left_writer is not None:
        left_writer.release()
    if right_writer is not None:
        right_writer.release()
    if depth_writer is not None:
        depth_writer.release()
    zed.close()


def process_episode_stats(svo_path: Path, dataset: LeRobotDataset):
    """Calculate and save statistics for a specific episode."""
    # Get the episode data from the dataset
    episode_index = svo_path.stem.split("_")[1]
    ep_start_idx = dataset.episode_data_index["from"][episode_index]
    ep_end_idx = dataset.episode_data_index["to"][episode_index]
    ep_data = dataset.hf_dataset.select(range(ep_start_idx, ep_end_idx))

    chunk_name = svo_path.parent.parent.name
    
    # Create episode_data dictionary with the correct structure
    episode_data = {}
    for key, ft in dataset.features.items():
        if ft["dtype"] in ["image", "video"]:
            # For images and videos, we need to get the paths
            chunk_dir = dataset.root / "images" / chunk_name
            episode_dir = chunk_dir / key / f"episode_{episode_index:06d}"
            if not episode_dir.exists():
                raise RuntimeError(f"Episode directory not found: {episode_dir}")

            # Get all frame paths in order
            frame_paths = sorted(episode_dir.glob("frame_*.png"))
            episode_data[key] = [str(path) for path in frame_paths]
        else:
            # For other data types, we can use the numpy array directly
            episode_data[key] = np.array(ep_data[key])
    
    # Compute and save the episode statistics
    episode_stats = compute_episode_stats(episode_data, dataset.features)
    write_episode_stats(episode_index, episode_stats, dataset.root)


def main():
    parser = argparse.ArgumentParser(description="Convert ZED SVO files to MP4 and calculate statistics.")
    parser.add_argument("--dataset_repo_id", type=str, required=True, help="Path to the LeRobotDataset directory")
    args = parser.parse_args()

    init_logging()

    # Load the dataset
    dataset = LeRobotDataset(args.dataset_repo_id, verify=False)
    fps = dataset.meta.info["fps"]

    # Find all SVO files in the dataset directory
    svo_files = list(dataset.root.glob("videos/chunk-*/*/episode_*.svo2"))
    if not svo_files:
        raise RuntimeError("No SVO files found in dataset directory")

    # Group SVO files by their parent directory
    svo_files_by_dir = {}
    for svo_path in svo_files:
        parent_dir = svo_path.parent
        if parent_dir not in svo_files_by_dir:
            svo_files_by_dir[parent_dir] = []
        svo_files_by_dir[parent_dir].append(svo_path)

    # Process each directory's SVO files
    for parent_dir, svo_files in svo_files_by_dir.items():
        logging.info(f"Processing directory: {parent_dir}")
        for svo_path in tqdm(svo_files, desc=f"Processing SVO files in {parent_dir.name}"):
            # Process the SVO file directly
            extract_svo_frames(svo_path, dataset)
            process_episode_stats(svo_path, dataset)

            # Delete the processed SVO file
            logging.info(f"Removing SVO file: {svo_path}")
            svo_path.unlink()

            # Check if directory is empty and delete if it is
            if not any(parent_dir.iterdir()):
                logging.info(f"Removing empty directory: {parent_dir}")
                shutil.rmtree(parent_dir)

    logging.info("Conversion complete!")


if __name__ == "__main__":
    main() 