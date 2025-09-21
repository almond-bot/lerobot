import argparse
import threading
import time

import cv2
from env import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FOLLOWER_CAM_PORT,
    FOLLOWER_ID,
    FOLLOWER_PORT,
    FPS,
    HF_USER,
    LEADER_ID,
    LEADER_PORT,
    OVERHEAD_CAM_PORT,
)

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.record import record_loop
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader

EPISODE_COUNTDOWN_SECONDS = 3

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="The name of the task")
parser.add_argument("--description", type=str, required=True, help="The description of the task")
parser.add_argument(
    "--extend", action="store_true", required=False, help="Extend the dataset with more episodes"
)
parser.add_argument("--policy", type=str, required=False, default=None, help="The path to the policy")
args = parser.parse_args()

camera_config = {
    "overhead": OpenCVCameraConfig(
        index_or_path=OVERHEAD_CAM_PORT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS
    ),
    "follower": OpenCVCameraConfig(
        index_or_path=FOLLOWER_CAM_PORT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS
    ),
}
follower_config = SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras=camera_config)
leader_config = SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID) if args.policy is None else None

# Initialize the leader and follower
follower = SO101Follower(follower_config)
leader = SO101Leader(leader_config) if args.policy is None else None

# Configure the dataset features
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset or load existing dataset
if args.extend:
    dataset = LeRobotDataset(
        repo_id=f"{HF_USER}/{args.name}",
    )
    dataset.start_image_writer()
else:
    dataset = LeRobotDataset.create(
        repo_id=f"{HF_USER}/{args.name}",
        fps=FPS,
        features=dataset_features,
        robot_type=follower.name,
        use_videos=True,
        image_writer_threads=4,
    )

# Connect the leader and follower
follower.connect()
leader.connect() if args.policy is None else None

opencv_cams = [cam for cam in follower.cameras.values() if isinstance(cam, OpenCVCamera)]
for cam in opencv_cams:
    cam.videocapture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "U", "Y", "2"))

# Initialize the events and command loop
events = {
    "exit_early": False,
    "rerecord_episode": False,
    "stop_recording": False,
    "record_episode": False,
}


def cmd_loop():
    while True:
        cmd = input("CMD: ")
        if cmd == "s":
            events["record_episode"] = True
        if cmd == "d":
            events["exit_early"] = True
        elif cmd == "r":
            events["exit_early"] = True
            events["rerecord_episode"] = True
        elif cmd == "q":
            events["exit_early"] = True
            events["stop_recording"] = True
            break


cmd_thread = threading.Thread(target=cmd_loop)
cmd_thread.start()

# Initialize the policy
if args.policy is not None:
    policy_type = args.policy.split("/")[-1].split("_")[0]
    if policy_type == ACTPolicy.name:
        policy = ACTPolicy.from_pretrained(args.policy)
    elif policy_type == PI0Policy.name:
        policy = PI0Policy.from_pretrained(args.policy)
    elif policy_type == PI0FASTPolicy.name:
        policy = PI0FASTPolicy.from_pretrained(args.policy)
    elif policy_type == SmolVLAPolicy.name:
        policy = SmolVLAPolicy.from_pretrained(args.policy)
    else:
        raise ValueError(f"Invalid policy: {args.policy}")
else:
    policy = None

# Record the episodes
episode_idx = 0
while not events["stop_recording"]:
    # Wait for the record episode command
    while not events["record_episode"]:
        print("Waiting to record episode")
        time.sleep(1 / FPS)
    events["record_episode"] = False

    # Countdown to the start of the episode
    for i in range(EPISODE_COUNTDOWN_SECONDS):
        print("Starting in", EPISODE_COUNTDOWN_SECONDS - i)
        time.sleep(1)

    # Record the episode
    print(f"Recording episode {episode_idx + 1}")
    record_loop(
        robot=follower,
        policy=policy,
        events=events,
        fps=FPS,
        teleop=leader,
        dataset=dataset,
        single_task=args.description,
    )

    # Delete the episode if we need to re-record or save the episode
    if events["rerecord_episode"]:
        print("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
    else:
        dataset.save_episode()
        episode_idx += 1

# Clean up and push to HF
print("Stop recording")
cmd_thread.join()
follower.disconnect()
leader.disconnect() if args.policy is None else None
dataset.push_to_hub()
