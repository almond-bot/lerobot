# Almond LeRobot
Modifying LeRobot to teleoperate, collect data, and train on Almond Bot.

## Modifications
- Added ZED and Fairino controller files:
    - `lerobot/common/robot_devices/cameras/zed.py`
    - `lerobot/common/robot_devices/robots/almond.py`
    - `lerobot/common/robot_devices/robots/fairino.py` (TODO: replace with our new API on PyPi)
- Scripts under `almond/` to teleop, record, and train.

## Installation
- Turn on the FR5 and make sure the Jetson is connectd to the control box.
- Clone this repo on the Jetson.
- Install with `uv sync --extras "almond" --extras "pi0"`
- Then install the ZED Python SDK and point the location to `.venv/bin/python3` (TODO: we have to do this everything we use `uv` to update the packages since it's not tracked by `uv`)
    - There is a copy of the SDK in `/home`.
- Follow the instructions in [README](README.md) to install `ffmpeg` on Linux (build from source).

## Using Scripts
- Turn on the FR5 and make sure the Jetson is connectd to the control box.
- To get a hang of using the scripts and controlling the FR5, start by running `./almond/teleop.sh`.
    - Ensure you open [localhost:8000](http://localhost:8000) to access the keyboard controller.
- Once you are comfortable controlling the robot, run `./almond/record.sh` to start collecting data.
    - The settings are already set for the `clean_workspace` task.
    - Once the FR5 and ZED initialize, there is a one-time "warmup" that you can wait through.
    - The following will loop until all the episodes are recorded:
        - The episode will start recording when you should teleop to complete the task. Ensure after you finish the task, you move the robot back to the same spot.
        - The recording will save and there will be a reset period to reset the environment to what it was at start.
    - The state of the record process is printed in the terminal. Since we are on the Jetson, the recording window and sound alerts do not work.