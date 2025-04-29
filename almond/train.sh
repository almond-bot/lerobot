TASK=$1

sudo apt update && sudo apt upgrade -y
wget -qO- https://astral.sh/uv/install.sh | sh
uv sync --extra "almond" --extra "pi0"
sudo apt install zstd -y

python lerobot/scripts/train.py \
  --dataset.repo_id=shawnptl8/${TASK} \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_${TASK} \
  --job_name=pi0_${TASK} \
  --policy.device=cuda \
  --wandb.enable=false