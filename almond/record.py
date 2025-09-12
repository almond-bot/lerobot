import argparse
import threading
import time

from env import FOLLOWER_CAM_PORT, FOLLOWER_PORT, HF_USER, LEADER_PORT, OVERHEAD_CAM_PORT

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader

FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
EPISODE_COUNTDOWN_SECONDS = 3

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="The name of the task")
parser.add_argument("--description", type=str, required=True, help="The description of the task")
args = parser.parse_args()

camera_config = {
    "overhead": OpenCVCameraConfig(
        index_or_path=OVERHEAD_CAM_PORT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS
    ),
    "follower": OpenCVCameraConfig(
        index_or_path=FOLLOWER_CAM_PORT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS
    ),
}
follower_config = SO100FollowerConfig(port=FOLLOWER_PORT, cameras=camera_config)
leader_config = SO100LeaderConfig(port=LEADER_PORT)

# Initialize the leader and follower
follower = SO100Follower(follower_config)
leader = SO100Leader(leader_config)

# Configure the dataset features
action_features = hw_to_dataset_features(follower.action_features, "action")
obs_features = hw_to_dataset_features(follower.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset
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
leader.connect()

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

# Record the episodes
episode_idx = 0
while not events["stop_recording"]:
    # Wait for the record episode command
    while not events["record_episode"]:
        print("Waiting to record episode")
        time.sleep(1 / FPS)
    events["record_episode"] = False

    for i in range(EPISODE_COUNTDOWN_SECONDS):
        print("Starting in", EPISODE_COUNTDOWN_SECONDS - i)
        time.sleep(1)

    # Record the episode
    print(f"Recording episode {episode_idx + 1}")
    record_loop(
        robot=follower,
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
leader.disconnect()
dataset.push_to_hub()
