import argparse

import rerun
from env import FOLLOWER_CAM_PORT, FOLLOWER_PORT, LEADER_PORT, OVERHEAD_CAM_PORT

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Visualize joint positions and camera feeds")
    args = parser.parse_args()

    if args.visualize:
        _init_rerun(session_name="teleop")

    camera_config = {
        "overhead": OpenCVCameraConfig(index_or_path=OVERHEAD_CAM_PORT, width=1920, height=1080, fps=30),
        "follower": OpenCVCameraConfig(index_or_path=FOLLOWER_CAM_PORT, width=1920, height=1080, fps=30),
    }

    follower_config = SO101FollowerConfig(port=FOLLOWER_PORT, cameras=camera_config)
    leader_config = SO101LeaderConfig(
        port=LEADER_PORT,
    )

    follower = SO101Follower(follower_config)
    leader = SO101Leader(leader_config)

    follower.connect()
    leader.connect()

    try:
        while True:
            action = leader.get_action()
            follower.send_action(action)

            if args.visualize:
                observation = follower.get_observation()
                log_rerun_data(observation, action)
    except KeyboardInterrupt:
        pass
    finally:
        if args.visualize:
            rerun.rerun_shutdown()
        follower.disconnect()
        leader.disconnect()


if __name__ == "__main__":
    main()
