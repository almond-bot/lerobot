TASK=$1

if [ -z "$TASK" ]; then
    echo "Error: Task not specified"
    exit 1
fi

uv sync --extra "pi0"

python lerobot/scripts/train.py \
  --dataset.repo_id=shawnptl8/${TASK} \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_${TASK} \
  --job_name=pi0_${TASK} \
  --policy.device=cuda \
  --wandb.enable=false