from env import FOLLOWER_PORT

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def main():
    config = SO101FollowerConfig(port=FOLLOWER_PORT)

    follower = SO101Follower(config)
    follower.connect(calibrate=False)
    follower.calibrate()
    follower.disconnect()


if __name__ == "__main__":
    main()
