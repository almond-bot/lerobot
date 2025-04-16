rm -fr ~/.cache/huggingface/lerobot/shawnptl8/open_3d_printer_door

uv run lerobot/scripts/control_robot.py \
    --robot.type=almond \
    --control.type=teleoperate