import argparse
import threading

from env import CAMERA_HEIGHT, CAMERA_WIDTH, FOLLOWER_CAM_PORT, FOLLOWER_PORT, FPS, OVERHEAD_CAM_PORT

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101FollowerConfig
from lerobot.scripts.server.configs import PolicyServerConfig, RobotClientConfig
from lerobot.scripts.server.helpers import visualize_action_queue_size
from lerobot.scripts.server.policy_server import serve
from lerobot.scripts.server.robot_client import RobotClient

HOST = "localhost"
PORT = 8000

CHUNK_SIZE_THRESHOLD = 0.5
ACTIONS_PER_CHUNK = 50

parser = argparse.ArgumentParser()
parser.add_argument("--server", action="store_true", required=False, help="Start the policy server")
parser.add_argument("--client", action="store_true", required=False, help="Start the client")
parser.add_argument("--policy", type=str, required=False, default=None, help="The path to the policy")
parser.add_argument("--task", type=str, required=False, default=None, help="The task to perform")
args = parser.parse_args()

if args.server:
    config = PolicyServerConfig(
        host=HOST,
        port=PORT,
    )

    serve(config)
elif args.client:
    if args.policy is None:
        raise ValueError("Policy is required")
    if args.task is None:
        raise ValueError("Task is required")

    # 1. Create the robot instance
    camera_config = {
        "overhead": OpenCVCameraConfig(
            index_or_path=OVERHEAD_CAM_PORT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS
        ),
        "follower": OpenCVCameraConfig(
            index_or_path=FOLLOWER_CAM_PORT, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS
        ),
    }
    follower_config = SO101FollowerConfig(port=FOLLOWER_PORT, cameras=camera_config)

    # 3. Create client configuration
    client_cfg = RobotClientConfig(
        robot=follower_config,
        server_address=f"{HOST}:{PORT}",
        policy_device="cuda",
        policy_type=args.policy.split("/")[-1].split("_")[0],
        pretrained_name_or_path=args.policy,
        chunk_size_threshold=CHUNK_SIZE_THRESHOLD,
        actions_per_chunk=ACTIONS_PER_CHUNK,
    )

    # 4. Create and start client
    client = RobotClient(client_cfg)

    if client.start():
        # Start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            # Run the control loop
            client.control_loop(args.task)
        except KeyboardInterrupt:
            client.stop()
            action_receiver_thread.join()

            try:
                visualize_action_queue_size(client.action_queue_size)
            except Exception as e:
                print(f"Error visualizing action queue size: {e}")
