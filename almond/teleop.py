from env import FOLLOWER_CAM_PORT, FOLLOWER_PORT, LEADER_PORT, OVERHEAD_CAM_PORT

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


def main():
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

    while True:
        action = leader.get_action()
        follower.send_action(action)


if __name__ == "__main__":
    main()
