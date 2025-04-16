rm -fr ~/.cache/huggingface/lerobot/shawnptl8/clean_workspace

uv run lerobot/scripts/control_robot.py \
    --robot.type=almond \
    --control.type=record \
    --control.single_task="Put tools away." \
    --control.fps=30 \
    --control.repo_id=shawnptl8/clean_workspace \
    --control.warmup_time_s=15 \
    --control.episode_time_s=60 \
    --control.reset_time_s=10 \
    --control.num_episodes=20 \
    --control.push_to_hub=false