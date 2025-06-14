TASK=$1

git config --global user.email "lambda@almondbot.com"
git config --global user.name "Almond Lambda"

sudo apt update && sudo apt upgrade -y
wget -qO- https://astral.sh/uv/install.sh | sh

uv sync --extra "almond" --extra "pi0" --extra "test"

sudo apt install zstd -y
wget -O ~/almond/zed_v5.run https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu22
sudo chmod +x ~/almond/zed_v5.run

# sudo apt install nvidia-driver-570

export HF_LEROBOT_HOME="~/almond/data"

touch data/${TASK}/episodes_stats.jsonl

uv run lerobot/scripts/extract_zed_svo.py \
  --dataset_repo_id ${TASK}

python lerobot/scripts/train.py \
  --dataset.repo_id=${TASK} \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_${TASK} \
  --job_name=pi0_${TASK} \
  --policy.device=cuda \
  --batch_size=2 \
  --wandb.enable=false
