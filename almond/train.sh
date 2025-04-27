TASK=$1

uv sync --extra "pi0"

python lerobot/scripts/train.py \
  --dataset.repo_id=shawnptl8/${TASK} \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_${TASK} \
  --job_name=pi0_${TASK} \
  --policy.device=cuda \
  --wandb.enable=false