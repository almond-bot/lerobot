import argparse

from env import FOLLOWER_PORT, LEADER_PORT

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


def configure_leader():
    config = SO101LeaderConfig(port=LEADER_PORT)
    leader = SO101Leader(config)

    leader.setup_motors()


def configure_follower():
    config = SO101FollowerConfig(port=FOLLOWER_PORT)
    follower = SO101Follower(config)

    follower.setup_motors()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", action="store_true", help="Configure the leader arm")
    parser.add_argument("--follower", action="store_true", help="Configure the follower arm")

    args = parser.parse_args()

    if args.leader:
        configure_leader()
    if args.follower:
        configure_follower()


if __name__ == "__main__":
    main()
