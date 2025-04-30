TASK=$1

rm -fr ~/.cache/huggingface/lerobot/shawnptl8/${TASK}

uv run lerobot/scripts/control_robot.py \
    --robot.type=almond \
    --control.type=record \
    --control.single_task="${TASK}" \
    --control.fps=20 \
    --control.repo_id=shawnptl8/${TASK} \
    --control.warmup_time_s=15 \
    --control.episode_time_s=30 \
    --control.reset_time_s=15 \
    --control.num_episodes=50 \
    --control.push_to_hub=false \
    --control.num_image_writer_processes=0 \
    --control.num_image_writer_threads_per_camera=0 \
    --control.display_data=false \
    --control.play_sounds=false
