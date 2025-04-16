uv sync --extra "pi0"

python lerobot/scripts/train.py \
  --dataset.repo_id=shawnptl8/clean_workspace \
  --policy.type=act \
  --output_dir=outputs/train/pi0_clean_workspace \
  --job_name=pi0_clean_workspace \
  --policy.device=cuda \
  --wandb.enable=false