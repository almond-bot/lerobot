import argparse

from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from env import LEADER_PORT, FOLLOWER_PORT

def calibrate_leader():
    config = SO101LeaderConfig(port=LEADER_PORT)
    leader = SO101Leader(config)
    leader.setup_motors()

def calibrate_follower():
    config = SO101FollowerConfig(port=FOLLOWER_PORT)
    follower = SO101Follower(config)
    follower.setup_motors()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", action="store_true", help="Calibrate the leader arm")
    parser.add_argument("--follower", action="store_true", help="Calibrate the follower arm")

    args = parser.parse_args()

    if args.leader:
        calibrate_leader()
    if args.follower:
        calibrate_follower()

if __name__ == "__main__":
    main()