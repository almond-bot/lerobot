MODEL=$1

if [ -z "$MODEL" ]; then
    echo "Error: Model not specified"
    exit 1
fi

uv run lerobot/scripts/control_robot.py \
    --robot.type=almond \
    --control.type=teleoperate \
    --control.fps=20 \
    --control.policy.path=outputs/train/${MODEL}/checkpoints/last/pretrained_model