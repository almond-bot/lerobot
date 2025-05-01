TASK=$1

if [ -z "$TASK" ]; then
    echo "Error: Task not specified"
    exit 1
fi

rm -fr ~/.cache/huggingface/lerobot/shawnptl8/eval_pi0_${TASK}

uv run lerobot/scripts/control_robot.py \
    --robot.type=almond \
    --control.type=record \
    --control.fps=20 \
    --control.repo_id=shawnptl8/eval_pi0_${TASK} \
    --control.warmup_time_s=15 \
    --control.episode_time_s=30 \
    --control.reset_time_s=15 \
    --control.num_episodes=1 \
    --control.push_to_hub=false \
    --control.num_image_writer_processes=0 \
    --control.num_image_writer_threads_per_camera=0 \
    --control.display_data=false \
    --control.play_sounds=false \
    --control.policy.path=outputs/train/pi0_${TASK}/checkpoints/last/pretrained_model