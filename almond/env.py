import os

from dotenv import load_dotenv

DIR_NAME = os.path.dirname(__file__)

load_dotenv(f"{DIR_NAME}/.env")
load_dotenv(f"{DIR_NAME}/.env.local", override=True)

HF_USER = os.environ["HF_USER"]

LEADER_PORT = os.environ["LEADER_PORT"]
FOLLOWER_PORT = os.environ["FOLLOWER_PORT"]
OVERHEAD_CAM_PORT = os.environ["OVERHEAD_CAM_PORT"]
FOLLOWER_CAM_PORT = os.environ["FOLLOWER_CAM_PORT"]
